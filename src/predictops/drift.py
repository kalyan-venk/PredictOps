"""Detect per-feature drift (PSI) between a reference dataset and current (possibly shifted) data."""

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
PSI_ALERT_THRESHOLD = 0.15
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
    """Run Evidently's data-drift preset with PSI as the per-column stattest, save the HTML
    report, and return a summary including per-feature PSI scores."""
    ref_features = reference.drop(columns=[TARGET_COL])
    cur_features = current.drop(columns=[TARGET_COL])

    report = Report(metrics=[DataDriftPreset(stattest="psi", stattest_threshold=PSI_ALERT_THRESHOLD)])
    report.run(reference_data=ref_features, current_data=cur_features)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))

    result = report.as_dict()
    dataset_result = result["metrics"][0]["result"]
    table_result = result["metrics"][1]["result"]

    per_feature_psi = {
        col: r["drift_score"] for col, r in table_result["drift_by_columns"].items()
    }

    return {
        "number_of_columns": dataset_result["number_of_columns"],
        "number_of_drifted_columns": dataset_result["number_of_drifted_columns"],
        "share_of_drifted_columns": dataset_result["share_of_drifted_columns"],
        "per_feature_psi": per_feature_psi,
        "max_psi": max(per_feature_psi.values()),
    }


def main() -> dict:
    df = load_dataset()
    reference = df.sample(frac=0.5, random_state=RANDOM_STATE)
    current = simulate_drift(df.drop(reference.index))

    summary = run_drift_report(reference, current)
    triggered = summary["max_psi"] > PSI_ALERT_THRESHOLD
    summary["drift_warning_triggered"] = triggered

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_param("psi_alert_threshold", PSI_ALERT_THRESHOLD)
        mlflow.log_metric("max_psi", summary["max_psi"])
        mlflow.log_metric("number_of_drifted_columns", summary["number_of_drifted_columns"])
        mlflow.log_metric("share_of_drifted_columns", summary["share_of_drifted_columns"])
        for feature, psi in summary["per_feature_psi"].items():
            mlflow.log_metric(f"psi_{feature}", psi)
        mlflow.log_artifact(str(REPORT_PATH))

    print(f"Drift summary: {summary}")
    if triggered:
        worst_feature = max(summary["per_feature_psi"], key=summary["per_feature_psi"].get)
        print(
            f"[WARNING] {worst_feature} PSI={summary['max_psi']:.4f} "
            f"> threshold={PSI_ALERT_THRESHOLD}. Investigate upstream data."
        )
    else:
        print("No significant drift detected.")

    return summary


if __name__ == "__main__":
    main()
