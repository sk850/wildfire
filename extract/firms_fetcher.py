"""
NASA FIRMS fire detection fetcher.

Pulls active fire detections for a bounding box, filters out likely
non-wildfire thermal anomalies (see landcover_filter.py), and either:
    - writes them to a local date-stamped CSV (default), or
    - pushes them onto a Kafka topic (set USE_KAFKA=true)

Setup:
    1. Register for a free MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/
    2. Set it as an env var: export FIRMS_MAP_KEY=your_key_here
    3. pip install requests pandas
    4. For Kafka mode: pip install kafka-python, and have a broker running
       (export KAFKA_BOOTSTRAP_SERVERS if not localhost:9092)

Usage:
    python firms_fetcher.py                    # writes local CSV
    USE_KAFKA=true python firms_fetcher.py      # streams to Kafka
"""

import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from landcover_filter import filter_fire_detections

FIRMS_MAP_KEY = os.environ.get("FIRMS_MAP_KEY")
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Sensor choice matters:
#   VIIRS_NOAA20_NRT / VIIRS_SNPP_NRT -> higher spatial resolution, good default
#   MODIS_NRT                          -> longer historical record, coarser
SENSOR = "VIIRS_NOAA20_NRT"

# west,south,east,north — this example box roughly covers California.
# Swap in your target region.
BOUNDING_BOX = "-124,32,-114,42"

# How many days back to pull (max useful range depends on transaction cost —
# see rate limit note below)
DAY_RANGE = 3

OUTPUT_DIR = Path("data/raw/firms")


def build_url(map_key: str, sensor: str, bbox: str, day_range: int, target_date: str | None = None) -> str:
    """Construct the FIRMS Area API URL.

    target_date, if given, should be 'YYYY-MM-DD' and fetches historical
    data ending on that date instead of the most recent day_range.
    """
    parts = [FIRMS_BASE_URL, map_key, sensor, bbox, str(day_range)]
    if target_date:
        parts.append(target_date)
    return "/".join(parts)


def fetch_fire_data(map_key: str, sensor: str = SENSOR, bbox: str = BOUNDING_BOX,
                     day_range: int = DAY_RANGE, target_date: str | None = None) -> pd.DataFrame:
    """Fetch fire detections and return as a DataFrame.

    Raises requests.HTTPError on a bad response — FIRMS returns HTTP 200
    with an error message in the body sometimes, so we also sanity-check
    that the response looks like CSV before parsing.
    """
    if not map_key:
        raise ValueError(
            "No FIRMS_MAP_KEY set. Register at "
            "https://firms.modaps.eosdis.nasa.gov/api/ and export it."
        )

    url = build_url(map_key, sensor, bbox, day_range, target_date)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    if resp.text.strip().lower().startswith(("invalid", "error")):
        raise RuntimeError(f"FIRMS API returned an error body: {resp.text[:200]}")

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))
    return df


def save_snapshot(df: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> Path:
    """Save the fetched data with a date-stamped filename for easy partitioning
    later when this becomes a Spark job reading a directory of daily files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"firms_{date.today().isoformat()}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def send_to_kafka(df: pd.DataFrame, topic: str = "firms-raw",
                   bootstrap_servers: str = None) -> None:
    """Push each row as a JSON message onto a Kafka topic.

    Column selection matters here: this must match FIRMS_SCHEMA in
    spark_transform.py exactly (latitude, longitude, confidence, acq_date,
    acq_time) — extra columns are harmless (Spark's from_json just ignores
    fields not in the schema), but a renamed/missing column silently
    produces nulls on the consuming side rather than an error, so keep
    the two in sync by hand.
    """
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
    import json

    bootstrap_servers = bootstrap_servers or os.environ.get(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
    )

    schema_columns = ["latitude", "longitude", "confidence", "acq_date", "acq_time"]
    missing = [c for c in schema_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing columns Spark expects: {missing}. "
            f"Check FIRMS_SCHEMA in spark_transform.py stays in sync with this."
        )

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        # Wait for the leader broker to ack — good enough for a local/dev
        # single-broker setup. Bump to acks="all" once you care about
        # durability against broker failure, at some latency cost.
        acks="all",
        retries=3,
    )

    sent, failed = 0, 0
    futures = []

    for _, row in df.iterrows():
        message = row[schema_columns].to_dict()
        future = producer.send(topic, message)
        futures.append(future)

    # Block until every send is acknowledged (or fails) rather than firing
    # and forgetting — for a batch job like this, you want to know before
    # the process exits whether anything actually got through.
    for future in futures:
        try:
            future.get(timeout=10)
            sent += 1
        except KafkaError as e:
            failed += 1
            print(f"  Failed to send a message: {e}")

    producer.flush()
    producer.close()

    print(f"Kafka: {sent} sent, {failed} failed, topic={topic}")
    if failed > 0:
        print("  Failed rows were NOT retried — consider re-running the "
              "fetch, or persisting a local CSV as a fallback (see "
              "save_snapshot) so nothing is silently lost.")


# Set to "true" once Kafka is actually running and you want to stream
# instead of writing local CSVs. Defaults to CSV so this still runs
# standalone during early development.
USE_KAFKA = os.environ.get("USE_KAFKA", "false").lower() == "true"

if __name__ == "__main__":
    print(f"Fetching FIRMS data for bbox={BOUNDING_BOX}, sensor={SENSOR}, "
          f"day_range={DAY_RANGE}...")

    start = time.time()
    df = fetch_fire_data(FIRMS_MAP_KEY)
    elapsed = time.time() - start

    print(f"Got {len(df)} detections in {elapsed:.1f}s")

    if df.empty:
        print("No fire detections in this window — that's a valid result, "
              "not necessarily a bug.")
    else:
        # Filter out likely agricultural/industrial thermal anomalies before
        # this data ever reaches Kafka or the fire_events table — see
        # landcover_filter.py for why FIRMS alone can't distinguish these.
        df = filter_fire_detections(df, apply_landcover = False)

        if df.empty:
            print("All detections filtered out as non-wildfire (agriculture/"
                  "low-confidence/low-intensity) — nothing to save.")
        elif USE_KAFKA:
            send_to_kafka(df)
        else:
            out_path = save_snapshot(df)
            print(f"Saved to {out_path}")
            print(df[["latitude", "longitude", "acq_date", "confidence"]].head())