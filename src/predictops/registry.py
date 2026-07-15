"""Register the winning model in the MLflow Model Registry and promote it to production."""

import mlflow
from mlflow.tracking import MlflowClient

from predictops.config import EXPERIMENT_NAME

MODEL_NAME = "predictops-classifier"


def find_best_run_id(experiment_name: str = EXPERIMENT_NAME) -> str:
    """Find the child run with the highest test ROC-AUC across all training sessions."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found. Run training first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.test_roc_auc DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No finished runs found for experiment '{experiment_name}'.")
    return runs[0].info.run_id


def register_and_promote(run_id: str | None = None, model_name: str = MODEL_NAME) -> int:
    """Register the given run's model, stage it, then promote it to production."""
    client = MlflowClient()
    run_id = run_id or find_best_run_id()
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = result.version
    print(f"Registered {model_name} v{version} from run {run_id}")

    client.transition_model_version_stage(name=model_name, version=version, stage="Staging")
    print(f"{model_name} v{version} -> Staging")

    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production", archive_existing_versions=True
    )
    print(f"{model_name} v{version} -> Production")

    return version


def main() -> None:
    register_and_promote()


if __name__ == "__main__":
    main()
