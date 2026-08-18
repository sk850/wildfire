"""
Regression gate for the CI/CD retraining loop.

Compares a newly trained model's validation metrics against the metrics
of whatever model is currently marked "in production" in the registry.
Only promotes the new model (updates the current_model.json pointer) if
it doesn't regress beyond a small tolerance — otherwise exits non-zero,
which a GitHub Actions step can use to block deployment.

This is what makes ml/train.py -> retrain-model.yml -> deploy.yml a real
MLOps loop instead of "retrain weekly and hope nothing got worse."

Usage:
    # After train.py has produced a new model + metrics.json:
    python evaluate.py ml/model_registry/model_20260812T114821Z.metrics.json

Exit codes:
    0 = promoted (new model is now current)
    1 = rejected (regression beyond tolerance; current model unchanged)
    2 = usage/setup error (bad args, missing files)
"""

import json
import sys
from pathlib import Path

MODEL_REGISTRY_DIR = Path("ml/model_registry")  # default only, when run as a script

# The metric used to decide promotion. PR-AUC over ROC-AUC deliberately —
# see train.py's module docstring for why, given how imbalanced fire
# events are relative to total (location, day) rows.
GATE_METRIC = "pr_auc"

# How much the new model is allowed to be WORSE than the current one and
# still get promoted. A small positive tolerance absorbs run-to-run noise
# from things like random seeds or minor data drift; it should not be
# large enough to mask a real regression. Tune this once you've seen a
# few real retraining cycles and have a sense of normal metric variance.
REGRESSION_TOLERANCE = 0.02  # new model can be up to 2 percentage points
                              # of PR-AUC worse and still pass


def load_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        print(f"ERROR: metrics file not found: {metrics_path}", file=sys.stderr)
        sys.exit(2)
    return json.loads(metrics_path.read_text())


def load_current_pointer(registry_dir: Path = MODEL_REGISTRY_DIR) -> dict | None:
    pointer_path = registry_dir / "current_model.json"
    if not pointer_path.exists():
        return None
    return json.loads(pointer_path.read_text())


def evaluate_candidate(new_metrics_path: Path, registry_dir: Path = MODEL_REGISTRY_DIR) -> bool:
    """Returns True if the candidate should be promoted, False otherwise.
    Prints its reasoning either way — this runs in CI, where the log is
    the only record of why a promotion did or didn't happen."""
    new_metrics = load_metrics(new_metrics_path)
    new_score = new_metrics.get(GATE_METRIC)

    if new_score is None:
        print(f"ERROR: new model's metrics file has no '{GATE_METRIC}' field. "
              f"Was it produced by the current version of train.py?", file=sys.stderr)
        sys.exit(2)

    current = load_current_pointer(registry_dir)

    if current is None:
        print(f"No current production model on record — this is the first "
              f"model, or the pointer file is missing. Auto-promoting "
              f"(new {GATE_METRIC}={new_score:.4f}).")
        return True

    current_metrics_path = registry_dir / current["metrics_file"]
    current_metrics = load_metrics(current_metrics_path)
    current_score = current_metrics[GATE_METRIC]

    delta = new_score - current_score
    print(f"Current model {GATE_METRIC}: {current_score:.4f} "
          f"(trained {current_metrics.get('trained_at', 'unknown')})")
    print(f"Candidate model {GATE_METRIC}: {new_score:.4f}")
    print(f"Delta: {delta:+.4f} (tolerance: -{REGRESSION_TOLERANCE:.4f})")

    if delta >= -REGRESSION_TOLERANCE:
        print(f"PASS: within tolerance — promoting candidate.")
        return True
    else:
        print(f"FAIL: candidate regresses {GATE_METRIC} by {-delta:.4f}, "
              f"more than the {REGRESSION_TOLERANCE:.4f} tolerance. "
              f"Current model stays in production.")
        return False


def promote(new_metrics_path: Path, registry_dir: Path = MODEL_REGISTRY_DIR):
    """Updates current_model.json to point at the candidate. Assumes the
    model .joblib file sits alongside its .metrics.json file with the
    same stem, matching train.py's save_model_artifact naming."""
    model_file = new_metrics_path.name.replace(".metrics.json", ".joblib")
    model_path = registry_dir / model_file

    if not model_path.exists():
        print(f"ERROR: expected model file {model_path} does not exist "
              f"alongside its metrics file.", file=sys.stderr)
        sys.exit(2)

    pointer = {
        "model_file": model_file,
        "metrics_file": new_metrics_path.name,
    }
    pointer_path = registry_dir / "current_model.json"
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(json.dumps(pointer, indent=2))
    print(f"Promoted: {pointer_path} now points to {model_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <path_to_candidate_metrics.json>", file=sys.stderr)
        sys.exit(2)

    candidate_metrics_path = Path(sys.argv[1])
    registry_dir = candidate_metrics_path.parent

    should_promote = evaluate_candidate(candidate_metrics_path, registry_dir)

    if should_promote:
        promote(candidate_metrics_path, registry_dir)
        sys.exit(0)
    else:
        sys.exit(1)