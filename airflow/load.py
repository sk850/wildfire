"""
Batch loader: firms_fetcher.py's CSV output -> Postgres fire_events table.

Mirrors what engineer_fire_events() does in spark_transform.py (daily
count per grid cell), but as a plain batch job — needed because the
Airflow retraining DAG works with completed batch runs, not Spark's
always-running streaming job. Both paths write to the same fire_events
table using the same GRID_PRECISION, so training data is consistent
regardless of which path populated a given row.

Setup:
    pip install pandas sqlalchemy psycopg2-binary

Usage:
    python load_firms_to_postgres.py data/raw/firms/firms_2026-08-14.csv
"""

import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

POSTGRES_URL = os.environ.get(
    "POSTGRES_SQLALCHEMY_URL",
    "postgresql+psycopg2://postgres:@localhost:5432/disaster_risk",
)

TABLE_NAME = "fire_events"

# Must match load_training_data.py's GRID_PRECISION exactly — this is
# what lets rows written by this batch loader join correctly against
# historical_weather_features at training time. If you ever change one,
# change both.
GRID_PRECISION = 2


def aggregate_to_daily_grid(df: pd.DataFrame, grid_precision: int = GRID_PRECISION) -> pd.DataFrame:
    """Collapses raw per-detection FIRMS rows into one row per (day, grid
    cell) with a detection count — the same shape fire_events expects,
    whether the row came from this batch path or the Spark streaming
    path."""
    df = df.copy()
    df["grid_lat"] = df["latitude"].round(grid_precision)
    df["grid_lon"] = df["longitude"].round(grid_precision)
    df["day_start"] = pd.to_datetime(df["acq_date"])
    df["day_end"] = df["day_start"] + pd.Timedelta(days=1)

    grouped = (
        df.groupby(["day_start", "day_end", "grid_lat", "grid_lon"])
        .size()
        .reset_index(name="fire_detection_count")
        .rename(columns={"grid_lat": "latitude", "grid_lon": "longitude"})
    )
    return grouped


def upsert_to_postgres(df: pd.DataFrame, engine) -> int:
    """Same staging-table upsert pattern as load_historical_features.py —
    ON CONFLICT DO NOTHING on (day_start, latitude, longitude), so
    re-running this on overlapping FIRMS pulls doesn't duplicate rows."""
    with engine.begin() as conn:
        df.to_sql("fire_events_staging", conn, if_exists="replace", index=False)

        result = conn.execute(text(f"""
            INSERT INTO {TABLE_NAME} (
                day_start, day_end, latitude, longitude, fire_detection_count
            )
            SELECT day_start, day_end, latitude, longitude, fire_detection_count
            FROM fire_events_staging
            ON CONFLICT (day_start, latitude, longitude) DO NOTHING
        """))

        conn.execute(text("DROP TABLE fire_events_staging"))
        return result.rowcount


def load_firms_csv_to_fire_events(csv_path: Path, engine=None) -> int:
    """End-to-end: read a FIRMS CSV (already filtered by
    landcover_filter.py upstream in firms_fetcher.py), aggregate to
    daily-grid counts, upsert into fire_events. Returns rows inserted."""
    engine = engine or create_engine(POSTGRES_URL)

    df = pd.read_csv(csv_path)
    required = {"latitude", "longitude", "acq_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if df.empty:
        print(f"{csv_path} has no rows (nothing survived filtering) — nothing to load.")
        return 0

    grouped = aggregate_to_daily_grid(df)
    inserted = upsert_to_postgres(grouped, engine)
    return inserted


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python load_firms_to_postgres.py <path_to_csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    inserted = load_firms_csv_to_fire_events(csv_path)
    print(f"Inserted {inserted} new (day, grid cell) rows into {TABLE_NAME}")