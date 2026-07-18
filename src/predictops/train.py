"""Train candidate models, compare them, log to MLflow, and persist the winner."""

import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from predictops.config import (
    EXPERIMENT_NAME,
    MODEL_META_PATH,
    MODEL_PATH,
    N_SPLITS,
    ONNX_MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from predictops.data import TARGET_COL, load_dataset


def build_preprocessor() -> ColumnTransformer:
    """Impute -> scale numeric columns; impute -> one-hot encode categorical columns."""
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    # No imputer on the categorical branch: every categorical feature is a
    # schema-constrained Literal enum (see schemas.PredictRequest), so a
    # missing/unknown category cannot reach inference, and the training data
    # has zero missing categoricals. Dropping it keeps predictions identical
    # (verified) while making the fitted pipeline exportable to ONNX — the
    # ONNX imputer op does not support float-sentinel missing values on
    # string columns.
    categorical_pipeline = Pipeline(
        steps=[
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, make_column_selector(dtype_include="number")),
            ("categorical", categorical_pipeline, make_column_selector(dtype_include="object")),
        ]
    )


def build_candidates() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.1,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def export_onnx(pipeline: Pipeline, X_train, model_name: str) -> None:
    """Export the fitted pipeline to ONNX so the serving image can run it with
    onnxruntime alone — no scikit-learn, xgboost, pandas, or scipy at runtime.
    This is what lets the final container drop the entire scientific stack.

    Each column becomes its own typed input: numeric columns as float tensors
    (integer-valued columns like tenure are fed as floats — numerically
    identical through impute/scale/linear ops), object columns as string
    tensors. `zipmap=False` makes the classifier emit a plain probability
    array instead of a list of dicts.
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType

    if model_name == "xgboost":
        # XGBClassifier has no built-in skl2onnx converter; register the one
        # onnxmltools ships before converting a pipeline that contains it.
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
        from skl2onnx import update_registered_converter
        from skl2onnx.common.shape_calculator import (
            calculate_linear_classifier_output_shapes,
        )

        update_registered_converter(
            XGBClassifier,
            "XGBoostXGBClassifier",
            calculate_linear_classifier_output_shapes,
            convert_xgboost,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )

    initial_types = [
        (
            col,
            StringTensorType([None, 1])
            if str(X_train.dtypes[col]) == "object"
            else FloatTensorType([None, 1]),
        )
        for col in X_train.columns
    ]
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        options={id(pipeline): {"zipmap": False}},
        target_opset=17,
    )
    ONNX_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ONNX_MODEL_PATH.write_bytes(onnx_model.SerializeToString())
    MODEL_META_PATH.write_text(
        json.dumps({"classifier_type": type(pipeline.named_steps["classifier"]).__name__})
    )


def cross_validate(pipeline: Pipeline, X_train, y_train) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    return float(np.mean(scores)), float(np.std(scores))


def evaluate_on_test(pipeline: Pipeline, X_test, y_test) -> dict[str, float]:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main() -> str:
    """Train every candidate under one parent run, persist the winner, return its child run_id."""
    df = load_dataset()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    best_name, best_pipeline, best_cv_auc, best_run_id = None, None, -1.0, None
    with mlflow.start_run(run_name="training_session"):
        for name, pipeline in build_candidates().items():
            cv_auc_mean, cv_auc_std = cross_validate(pipeline, X_train, y_train)
            pipeline.fit(X_train, y_train)
            test_metrics = evaluate_on_test(pipeline, X_test, y_test)

            with mlflow.start_run(run_name=name, nested=True) as child_run:
                mlflow.log_param("model_type", name)
                mlflow.log_metric("cv_roc_auc_mean", cv_auc_mean)
                mlflow.log_metric("cv_roc_auc_std", cv_auc_std)
                for metric_name, value in test_metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", value)
                mlflow.sklearn.log_model(
                    pipeline, artifact_path="model", input_example=X_train.iloc[:5]
                )

            print(f"[{name}] cv_roc_auc={cv_auc_mean:.4f}+/-{cv_auc_std:.4f} test={test_metrics}")

            if cv_auc_mean > best_cv_auc:
                best_name, best_pipeline, best_cv_auc = name, pipeline, cv_auc_mean
                best_run_id = child_run.info.run_id

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    export_onnx(best_pipeline, X_train, best_name)
    print(f"\nWinner: {best_name} (cv_roc_auc={best_cv_auc:.4f}) -> saved to {MODEL_PATH}")
    print(f"ONNX serving artifact -> {ONNX_MODEL_PATH}")
    return best_run_id


if __name__ == "__main__":
    main()
