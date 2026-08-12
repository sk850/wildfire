"""
Trains a gradient-boosted classifier predicting wildfire risk from
weather features, using the day-lag join defined in load_training_data.py.

Two decisions that matter more than the model choice itself:

1. TIME-BASED SPLIT, NOT RANDOM. Weather is autocorrelated day to day —
   a random shuffle would leak information (near-identical conditions
   from adjacent days landing in both train and validation), making
   validation metrics look better than the model would actually perform
   on genuinely future, unseen data. Train on the earlier portion of the
   timeline, validate on the later portion, matching how the model will
   actually be used in production (predict forward from what's known).

2. CLASS IMBALANCE. Wildfire days are a small minority of all (location,
   day) rows. Plain accuracy is close to meaningless here — see
   check_label_balance in load_training_data.py. This script reports
   precision, recall, and PR-AUC (not just ROC-AUC, which is more
   forgiving on imbalanced data than it looks) and uses class weighting
   during training.

Setup:
    pip install xgboost scikit-learn pandas sqlalchemy psycopg2-binary joblib

Usage:
    python train.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score, classification_report,
    precision_recall_curve, roc_auc_score,
)
from xgboost import XGBClassifier

from load_train_data import check_label_balance, load_training_data

FEATURE_COLUMNS = [
    "temp_max", "temp_min", "precipitation_mm",
    "wind_speed_max_kmh", "wind_gusts_max_kmh",
]
LABEL_COLUMN = "fire_occurred"

# Fraction of the timeline (by date, not row count) held out for
# validation — the LAST 20% of dates, not a random 20% of rows.
VALIDATION_FRACTION = 0.2

MODEL_REGISTRY_DIR = Path("ml/model_registry")


def time_based_split(df: pd.DataFrame, validation_fraction: float = VALIDATION_FRACTION):
    """Splits by date, not by shuffling rows. Everything before the cutoff
    date goes to train; everything on/after it goes to validation. This
    mirrors how the model is actually used: trained on the past, asked to
    predict a future it hasn't seen."""
    df = df.sort_values("observation_date")
    unique_dates = df["observation_date"].unique()

    if len(unique_dates) < 10:
        raise ValueError(
            f"Only {len(unique_dates)} unique dates in the training data — "
            f"too few for a meaningful time-based split. Pull a longer "
            f"date range with historical_weather_fetcher.py first."
        )

    cutoff_idx = int(len(unique_dates) * (1 - validation_fraction))
    cutoff_date = unique_dates[cutoff_idx]

    train_df = df[df["observation_date"] < cutoff_date]
    val_df = df[df["observation_date"] >= cutoff_date]

    print(f"Split at {cutoff_date}: {len(train_df)} train rows, "
          f"{len(val_df)} validation rows")

    return train_df, val_df


def train_model(train_df: pd.DataFrame) -> XGBClassifier:
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[LABEL_COLUMN]

    # scale_pos_weight counteracts class imbalance by upweighting the
    # minority (fire) class during training — without this, a model
    # trained on heavily imbalanced data tends to just predict the
    # majority class most of the time, since that's what minimizes raw
    # training loss.
    positive_count = y_train.sum()
    negative_count = len(y_train) - positive_count
    scale_pos_weight = negative_count / max(positive_count, 1)

    print(f"scale_pos_weight = {scale_pos_weight:.1f} "
          f"({negative_count} negative / {positive_count} positive)")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,          # shallow trees — this is a small tabular
                               # dataset (a few years of daily data for one
                               # region); deep trees would overfit fast
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",  # PR-AUC as the internal eval metric too,
                               # not the default (accuracy-adjacent) metric
        random_state=42,
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model: XGBClassifier, val_df: pd.DataFrame) -> dict:
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[LABEL_COLUMN]

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    pr_auc = average_precision_score(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba) if y_val.nunique() > 1 else float("nan")

    print("\nClassification report (threshold=0.5):")
    print(classification_report(y_val, y_pred, target_names=["no_fire", "fire"]))

    print(f"PR-AUC:  {pr_auc:.4f}  (more informative than ROC-AUC here — "
          f"see module docstring)")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # A default 0.5 classification threshold is rarely right for an
    # imbalanced problem like this — surfacing a few candidate thresholds
    # so you can pick one based on your actual precision/recall priority
    # (e.g. for a risk dashboard, missing a real fire (false negative) is
    # probably costlier than a false alarm, which argues for a lower
    # threshold than 0.5).
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
    print("\nPrecision/recall at a few candidate thresholds:")
    for target_recall in [0.9, 0.75, 0.5]:
        idx = next((i for i, r in enumerate(recalls) if r <= target_recall), None)
        if idx is not None and idx < len(thresholds):
            print(f"  recall>={target_recall}: threshold={thresholds[idx]:.3f}, "
                  f"precision={precisions[idx]:.3f}")

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc) if roc_auc == roc_auc else None,  # NaN check
        "val_rows": len(val_df),
        "val_positive_rate": float(y_val.mean()),
    }


def save_model_artifact(model: XGBClassifier, metrics: dict, registry_dir: Path = MODEL_REGISTRY_DIR):
    """Saves the model plus a metrics sidecar file, timestamped. The
    metrics file is what the CI/CD regression gate (evaluate.py) reads to
    decide whether a newly retrained model is allowed to replace the
    currently deployed one."""
    registry_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = registry_dir / f"model_{timestamp}.joblib"
    metrics_path = registry_dir / f"model_{timestamp}.metrics.json"

    joblib.dump(model, model_path)

    metadata = {
        "trained_at": timestamp,
        "feature_columns": FEATURE_COLUMNS,
        **metrics,
    }
    metrics_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")

    return model_path, metrics_path


if __name__ == "__main__":
    print("Loading training data...")
    df = load_training_data()
    check_label_balance(df)

    train_df, val_df = time_based_split(df)

    if train_df[LABEL_COLUMN].sum() == 0:
        raise ValueError(
            "Zero positive examples in the training split — check that "
            "fire_events actually has data overlapping your weather date "
            "range, and that GRID_PRECISION in load_training_data.py isn't "
            "too fine to match FIRMS points to weather points."
        )

    print("\nTraining model...")
    model = train_model(train_df)

    print("\nEvaluating on held-out validation period...")
    metrics = evaluate_model(model, val_df)

    save_model_artifact(model, metrics)