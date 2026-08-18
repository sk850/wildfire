"""
Weekly retraining DAG: fetch recent weather + fire data -> load into
Postgres -> train -> regression-gate -> (optionally) promote.

This DAG handles INCREMENTAL updates for ongoing retraining, not the
initial historical backfill. Before turning this DAG on, run
historical_weather_fetcher.py and a multi-year FIRMS backfill manually
once to seed historical_weather_features and fire_events with enough
history to train on — see the project's ML pipeline milestones.

This orchestrates existing scripts in ingestion/ and ml/ rather than
reimplementing their logic — each task is a thin wrapper calling an
already-tested function. If those modules aren't installed as a package,
DAGS_SCRIPT_ROOT below needs to point at wherever they actually live on
the Airflow worker.

Requires:
    pip install apache-airflow
    The ingestion/ and ml/ directories from this project, importable
    (see DAGS_SCRIPT_ROOT below)
    FIRMS_MAP_KEY and POSTGRES_SQLALCHEMY_URL set as Airflow Variables
    or environment variables on the worker
"""

import os
import sys
from datetime import datetime, timedelta

from airflow.sdk import dag, task
from airflow.exceptions import AirflowException

# Adjust these to wherever ingestion/ and ml/ actually live relative to
# this DAG file on your Airflow deployment. In a real deployment you'd
# more likely package these as an installable module and just `import`
# them — this sys.path approach is the fastest way to wire up a first
# working DAG against the existing project layout without restructuring
# it yet.
DAGS_SCRIPT_ROOT = os.environ.get("PROJECT_ROOT", "/opt/airflow/project")
sys.path.insert(0, os.path.join(DAGS_SCRIPT_ROOT, "ingestion"))
sys.path.insert(0, os.path.join(DAGS_SCRIPT_ROOT, "ml"))

# How many days back to pull on each run. A weekly schedule pulling ~10
# days gives a couple days of overlap/safety margin without re-pulling
# the entire history — both loaders are upsert-based (ON CONFLICT DO
# NOTHING), so overlap is harmless, just slightly wasteful.
LOOKBACK_DAYS = 10

DEFAULT_ARGS = {
    "owner": "wildfire-risk-project",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="wildfire_retrain_pipeline",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,  # don't backfill every missed week if the DAG was paused — this
                     # is a retraining loop, not a historical data pipeline; it only
                     # cares about "recent enough" data, not every week since start_date
    default_args=DEFAULT_ARGS,
    tags=["wildfire", "ml", "retraining"],
)
def wildfire_retrain_pipeline():

    @task
    def fetch_recent_weather() -> str:
        """Pulls the last LOOKBACK_DAYS of observed weather via the
        Open-Meteo Archive API and saves it as a CSV. Returns the CSV
        path via XCom for the load task to pick up."""
        from datetime import date

        from ingestion.historical_weather_fetcher import (
            LATITUDE, LONGITUDE, fetch_historical_weather, save_snapshot,
        )

        end = date.today()
        start = end - timedelta(days=LOOKBACK_DAYS)

        df = fetch_historical_weather(
            LATITUDE, LONGITUDE, start.isoformat(), end.isoformat()
        )
        if df.empty:
            raise AirflowException(
                "Open-Meteo returned no data for the requested window — "
                "check the API is reachable and the date range is valid "
                "(the Archive API often lags a few days behind real-time)."
            )

        path = save_snapshot(df, LATITUDE, LONGITUDE)
        return str(path)

    @task
    def load_weather_features(csv_path: str) -> int:
        """Loads the fetched CSV into historical_weather_features."""
        from pathlib import Path

        from transform_and_load.load_historical_features import load_csv, upsert_to_postgres
        from sqlalchemy import create_engine

        engine = create_engine(
            os.environ.get(
                "POSTGRES_SQLALCHEMY_URL",
                "postgresql+psycopg2://postgres:@localhost:5432/disaster_risk",
            )
        )
        df = load_csv(Path(csv_path))
        inserted = upsert_to_postgres(df, engine)
        return inserted

    @task
    def fetch_recent_firms() -> str:
        """Pulls recent FIRMS fire detections, filters out non-wildfire
        thermal anomalies, and saves the result as a CSV.

        NOTE: on_lookup_failure defaults to "raise" inside
        filter_fire_detections — if the NLCD land cover service is down,
        this task fails loudly rather than silently shipping unfiltered
        data. That's deliberate (see landcover_filter.py), but it does
        mean a flaky external service can fail this task; Airflow's
        retries (see DEFAULT_ARGS) give it a couple of automatic chances
        before you'd need to intervene.
        """
        from ingestion.firms_fetcher import (
            FIRMS_MAP_KEY, fetch_fire_data, save_snapshot,
        )
        from ingestion.landcover_filter import filter_fire_detections

        df = fetch_fire_data(FIRMS_MAP_KEY, day_range=LOOKBACK_DAYS)
        if df.empty:
            # A quiet week with zero fire detections is a legitimate
            # outcome, not a failure — downstream tasks need to handle
            # "no new fire data this run" gracefully.
            return ""

        df = filter_fire_detections(df)
        if df.empty:
            return ""

        path = save_snapshot(df)
        return str(path)

    @task
    def load_firms_features(csv_path: str) -> int:
        """Loads the fetched FIRMS CSV into fire_events. Skips cleanly
        if the fetch task found nothing this run."""
        from pathlib import Path

        if not csv_path:
            return 0

        from load import load_firms_csv_to_fire_events
        return load_firms_csv_to_fire_events(Path(csv_path))

    @task
    def train() -> str:
        """Runs the full train/validate/save cycle. Returns the path to
        the new model's metrics.json for the gate task to evaluate."""
        from ml.train import (
            evaluate_model, save_model_artifact, time_based_split,
            train_model,
        )
        from ml.load_train_data import check_label_balance, load_training_data

        df = load_training_data()
        check_label_balance(df)

        train_df, val_df = time_based_split(df)

        if train_df["fire_occurred"].sum() == 0:
            raise AirflowException(
                "Zero positive examples in the training split this run — "
                "not enough fire history yet, or a grid-precision mismatch "
                "between the weather and FIRMS loaders. See "
                "load_training_data.py's GRID_PRECISION vs "
                "load_firms_to_postgres.py's GRID_PRECISION — they must match."
            )

        model = train_model(train_df)
        metrics = evaluate_model(model, val_df)
        _, metrics_path = save_model_artifact(model, metrics)
        return str(metrics_path)

    @task
    def gate_and_promote(metrics_path: str) -> bool:
        """Runs the regression gate. Returns whether the new model was
        promoted — a downstream deploy task (not yet built) would branch
        on this via a ShortCircuitOperator so it only runs after a real
        promotion, not after every training run."""
        from pathlib import Path

        from ml.evaluate import evaluate_candidate, promote

        metrics_file = Path(metrics_path)
        registry_dir = metrics_file.parent

        should_promote = evaluate_candidate(metrics_file, registry_dir)
        if should_promote:
            promote(metrics_file, registry_dir)

        return should_promote

    # --- Task dependencies ---
    weather_csv = fetch_recent_weather()
    weather_loaded = load_weather_features(weather_csv)

    firms_csv = fetch_recent_firms()
    firms_loaded = load_firms_features(firms_csv)

    metrics_path = train()
    promoted = gate_and_promote(metrics_path)

    [weather_loaded, firms_loaded] >> metrics_path


wildfire_retrain_pipeline()