"""
Batch loader: historical_weather_fetcher.py's CSV output -> Postgres
historical_weather_features table.

This is a plain batch job (no Spark needed — the data volume here is
small: daily rows for one region over a few years, not a continuous
stream). Run it after historical_weather_fetcher.py, and re-run whenever
you pull a new year's data.

Setup:
    pip install pandas sqlalchemy psycopg2-binary

Usage:
    python load_historical_features.py data/raw/historical_weather/historical_weather_34.05_-118.25.csv
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

TABLE_NAME = "historical_weather_features"

# Maps the raw Open-Meteo column names to the schema.sql column names.
# Kept as an explicit dict rather than relying on matching names, so a
# renamed Open-Meteo field fails loudly (KeyError) instead of silently
# writing a column of nulls.
COLUMN_MAP = {
    "time": "observation_date",
    "latitude": "latitude",
    "longitude": "longitude",
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "precipitation_sum": "precipitation_mm",
    "wind_speed_10m_max": "wind_speed_max_kmh",
    "wind_gusts_10m_max": "wind_gusts_max_kmh",
}


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {missing}. Either the "
            f"Open-Meteo response shape changed, or DAILY_VARIABLES in "
            f"historical_weather_fetcher.py was edited without updating "
            f"COLUMN_MAP here to match."
        )

    df = df.rename(columns=COLUMN_MAP)
    return df[list(COLUMN_MAP.values())]


def upsert_to_postgres(df: pd.DataFrame, engine) -> int:
    """Insert rows, skipping duplicates on (observation_date, latitude,
    longitude) rather than erroring or blindly duplicating — this makes
    the loader safe to re-run on overlapping date ranges."""
    with engine.begin() as conn:
        # Write to a temp staging table, then upsert from there — simplest
        # reliable way to get ON CONFLICT DO NOTHING semantics through
        # pandas.to_sql, which doesn't support upserts natively.
        df.to_sql("historical_weather_features_staging", conn,
                   if_exists="replace", index=False)

        result = conn.execute(text(f"""
            INSERT INTO {TABLE_NAME} (
                observation_date, latitude, longitude,
                temp_max, temp_min, precipitation_mm,
                wind_speed_max_kmh, wind_gusts_max_kmh
            )
            SELECT
                observation_date, latitude, longitude,
                temp_max, temp_min, precipitation_mm,
                wind_speed_max_kmh, wind_gusts_max_kmh
            FROM historical_weather_features_staging
            ON CONFLICT (observation_date, latitude, longitude) DO NOTHING
        """))

        conn.execute(text("DROP TABLE historical_weather_features_staging"))

        return result.rowcount


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python load_historical_features.py <path_to_csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    print(f"Loading {csv_path}...")
    df = load_csv(csv_path)
    print(f"Parsed {len(df)} rows")

    engine = create_engine(POSTGRES_URL)
    inserted = upsert_to_postgres(df, engine)
    print(f"Inserted {inserted} new rows into {TABLE_NAME} "
          f"({len(df) - inserted} were already present and skipped)")