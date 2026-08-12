"""
Loads training data: historical_weather_features joined against
fire_events with an explicit one-day lag (weather on day N predicts fire
on day N+1). See schema.sql for why this table pair and this lag matter —
NOT weather_features_live, which has no historical depth to join against.

Setup:
    pip install pandas sqlalchemy psycopg2-binary
"""

import os

import pandas as pd
from sqlalchemy import create_engine, text

POSTGRES_URL = os.environ.get(
    "POSTGRES_SQLALCHEMY_URL",
    "postgresql+psycopg2://postgres:@localhost:5432/disaster_risk",
)

# Rounding precision for matching weather observations to fire detections
# by location. 2 decimal degrees is roughly ~1.1km at the equator — this
# defines your model's effective spatial grid cell size. Coarser (fewer
# decimals) means more training examples per cell but blurrier risk
# boundaries; finer means the opposite. Worth treating as a tuning
# decision, not a fixed constant, once you have a working baseline.
GRID_PRECISION = 2

TRAINING_QUERY = f"""
    SELECT
        hwf.observation_date,
        round(hwf.latitude::numeric, {GRID_PRECISION}) AS grid_lat,
        round(hwf.longitude::numeric, {GRID_PRECISION}) AS grid_lon,
        hwf.temp_max,
        hwf.temp_min,
        hwf.precipitation_mm,
        hwf.wind_speed_max_kmh,
        hwf.wind_gusts_max_kmh,
        COALESCE(fe.fire_detection_count, 0) > 0 AS fire_occurred
    FROM historical_weather_features hwf
    LEFT JOIN fire_events fe
        ON round(fe.latitude::numeric, {GRID_PRECISION}) = round(hwf.latitude::numeric, {GRID_PRECISION})
       AND round(fe.longitude::numeric, {GRID_PRECISION}) = round(hwf.longitude::numeric, {GRID_PRECISION})
       AND fe.day_start = hwf.observation_date + interval '1 day'
    ORDER BY hwf.observation_date
"""


def load_training_data(engine=None) -> pd.DataFrame:
    """Returns one row per (date, grid cell) with weather predictors and
    a fire_occurred boolean label. Sorted by date ascending — callers
    should NOT shuffle this before splitting; see train.py for why."""
    engine = engine or create_engine(POSTGRES_URL)

    with engine.connect() as conn:
        df = pd.read_sql(text(TRAINING_QUERY), conn)

    if df.empty:
        raise ValueError(
            "Training query returned zero rows. Likely causes: "
            "historical_weather_features or fire_events tables are empty "
            "(did you run historical_weather_fetcher.py + "
            "load_historical_features.py, and the FIRMS batch pull, "
            "against overlapping date ranges and the same region?)."
        )

    return df


def check_label_balance(df: pd.DataFrame) -> None:
    """Prints the positive rate — worth checking before training, since
    wildfire events are rare relative to total (location, day) pairs.
    Severe imbalance changes which metrics are meaningful (see train.py's
    use of PR-AUC over plain accuracy) and may need class weighting."""
    rate = df["fire_occurred"].mean()
    total = len(df)
    positives = int(df["fire_occurred"].sum())
    print(f"Label balance: {positives}/{total} positive ({rate:.2%})")
    if rate < 0.01:
        print("  NOTE: severe class imbalance (<1% positive). A model "
              "that always predicts 'no fire' would already score >99% "
              "accuracy while being useless — don't evaluate on accuracy "
              "alone. See train.py's evaluation metrics.")


if __name__ == "__main__":
    df = load_training_data()
    print(f"Loaded {len(df)} rows, "
          f"{df['observation_date'].min()} to {df['observation_date'].max()}")
    check_label_balance(df)
    print(df.head())