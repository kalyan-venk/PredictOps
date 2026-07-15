"""Train candidate models, compare them, log to MLflow, and persist the winner."""

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

from predictops.config import EXPERIMENT_NAME, MODEL_PATH, N_SPLITS, RANDOM_STATE, TEST_SIZE
from predictops.data import TARGET_COL, load_dataset


def build_preprocessor() -> ColumnTransformer:
    """Impute -> scale numeric columns; impute -> one-hot encode categorical columns."""
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
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
    print(f"\nWinner: {best_name} (cv_roc_auc={best_cv_auc:.4f}) -> saved to {MODEL_PATH}")
    return best_run_id


if __name__ == "__main__":
    main()
