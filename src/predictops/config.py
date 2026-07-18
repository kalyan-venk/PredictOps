"""Shared constants with no heavy imports, so the serving image can import them without
pulling in training-only dependencies (mlflow, evidently)."""

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "model.joblib"
ONNX_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "model.onnx"
MODEL_META_PATH = Path(__file__).resolve().parents[2] / "models" / "model_meta.json"
EXPERIMENT_NAME = "predictops"
