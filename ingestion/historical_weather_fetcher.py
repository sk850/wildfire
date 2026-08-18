"""
Historical weather fetcher — Open-Meteo Archive API.

This is the TRAINING DATA source, distinct from nws_fetcher.py (which is
forward-looking forecasts, used only for live/production inference — see
spark_transform.py's module docstring for why the two must not be
conflated). This script pulls PAST observed weather conditions, which is
what lets you actually build (weather on day N) -> (fire on day N+1)
training pairs against your FIRMS fire history.

No API key needed for non-commercial use (up to 10,000 calls/day).

Setup:
    pip install requests pandas

Usage:
    python historical_weather_fetcher.py
"""

import time
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Example point — swap in your target region's coordinates (should match
# whatever region you're pulling FIRMS fire history for).
LATITUDE = 34.05
LONGITUDE = -118.25

# Daily aggregates match FIRMS' daily fire_events windowing in schema.sql.
DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]
# NOTE: humidity is deliberately left off the default list — the daily
# aggregate endpoint doesn't reliably expose a max-humidity field across
# all API versions. Verify current field names against
# open-meteo.com/en/docs/historical-weather-api before a real run; if you
# need humidity, pull hourly relative_humidity_2m instead and aggregate
# to daily yourself.

OUTPUT_DIR = Path("data/raw/historical_weather")

# Open-Meteo asks that you space out large batch jobs — this is a shared
# free service, not a dedicated endpoint for your project.
REQUEST_DELAY_SECONDS = 1.0


def fetch_historical_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    daily_vars: list[str] = DAILY_VARIABLES,
) -> pd.DataFrame:
    """Fetch daily historical weather for one point over a date range.
    start_date/end_date must be 'YYYY-MM-DD'."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(daily_vars),
        "timezone": "auto",  # required whenever daily variables are requested
    }

    resp = requests.get(ARCHIVE_BASE_URL, params=params, timeout=30)

    if resp.status_code == 400:
        # Open-Meteo returns a JSON body explaining exactly which parameter
        # was invalid — surface it rather than a generic HTTP error, since
        # a renamed/removed variable is the most common failure mode.
        raise ValueError(f"Open-Meteo rejected the request: {resp.json().get('reason')}")

    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    if not daily:
        return pd.DataFrame()

    df = pd.DataFrame(daily)
    df["latitude"] = data["latitude"]
    df["longitude"] = data["longitude"]
    return df


def fetch_multi_year_range(
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    daily_vars: list[str] = DAILY_VARIABLES,
) -> pd.DataFrame:
    """Pull multiple years in yearly chunks, concatenating the results —
    a transient failure only costs you one year's retry, not the whole
    historical pull."""
    frames = []

    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        print(f"Fetching {year}...")
        try:
            year_df = fetch_historical_weather(lat, lon, start_date, end_date, daily_vars)
            if not year_df.empty:
                frames.append(year_df)
        except (requests.RequestException, ValueError) as e:
            print(f"  Failed for {year}: {e} — skipping, retry this year manually later")

        time.sleep(REQUEST_DELAY_SECONDS)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def save_snapshot(df: pd.DataFrame, lat: float, lon: float, output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"historical_weather_{lat}_{lon}.csv"
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    # Start small (a couple of years) to validate the pipeline before
    # committing to a full decade-plus pull. This range should cover at
    # least as far back as your FIRMS fire_events history, or the
    # training join in schema.sql will have unmatched fire rows.
    START_YEAR = 2022
    END_YEAR = 2024

    print(f"Fetching historical weather for ({LATITUDE}, {LONGITUDE}), "
          f"{START_YEAR}-{END_YEAR}...")

    df = fetch_multi_year_range(LATITUDE, LONGITUDE, START_YEAR, END_YEAR)

    if df.empty:
        print("No data returned — check coordinates and date range.")
    else:
        out_path = save_snapshot(df, LATITUDE, LONGITUDE)
        print(f"Saved {len(df)} daily records to {out_path}")
        print(f"Next step: run load_historical_features.py to load this "
              f"into the historical_weather_features Postgres table.")
        print(df.head())