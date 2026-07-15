"""Detect feature drift between a reference dataset and current (possibly shifted) data."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from predictops.data import TARGET_COL, load_dataset
from predictops.train import EXPERIMENT_NAME

RANDOM_STATE = 42
NOISE_STD_MULTIPLIER = 1.5
DRIFT_SHARE_THRESHOLD = 0.15
REPORT_PATH = Path(__file__).resolve().parents[2] / "reports" / "drift_report.html"


def simulate_drift(
    df: pd.DataFrame,
    noise_std_multiplier: float = NOISE_STD_MULTIPLIER,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Inject Gaussian noise into numeric feature columns to simulate distribution shift."""
    rng = np.random.default_rng(random_state)
    drifted = df.copy()
    numeric_cols = drifted.select_dtypes(include="number").columns.drop(TARGET_COL)
    for col in numeric_cols:
        drifted[col] = drifted[col] + rng.normal(
            0, noise_std_multiplier * drifted[col].std(), size=len(drifted)
        )
    return drifted


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Run Evidently's data-drift preset and return a summary dict, saving the HTML report."""
    ref_features = reference.drop(columns=[TARGET_COL])
    cur_features = current.drop(columns=[TARGET_COL])

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_features, current_data=cur_features)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))

    drift_result = report.as_dict()["metrics"][0]["result"]
    return {
        "number_of_columns": drift_result["number_of_columns"],
        "number_of_drifted_columns": drift_result["number_of_drifted_columns"],
        "share_of_drifted_columns": drift_result["share_of_drifted_columns"],
        "evidently_dataset_drift": drift_result["dataset_drift"],
    }


def main() -> dict:
    df = load_dataset()
    reference = df.sample(frac=0.5, random_state=RANDOM_STATE)
    current = simulate_drift(df.drop(reference.index))

    summary = run_drift_report(reference, current)
    triggered = summary["share_of_drifted_columns"] >= DRIFT_SHARE_THRESHOLD
    summary["drift_warning_triggered"] = triggered

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_param("drift_share_threshold", DRIFT_SHARE_THRESHOLD)
        mlflow.log_metric("share_of_drifted_columns", summary["share_of_drifted_columns"])
        mlflow.log_metric("number_of_drifted_columns", summary["number_of_drifted_columns"])
        mlflow.log_artifact(str(REPORT_PATH))

    print(f"Drift summary: {summary}")
    if triggered:
        print(
            f"[WARNING] share_of_drifted_columns={summary['share_of_drifted_columns']:.2f} "
            f">= threshold={DRIFT_SHARE_THRESHOLD}. Investigate upstream data."
        )
    else:
        print("No significant drift detected.")

    return summary


if __name__ == "__main__":
    main()
