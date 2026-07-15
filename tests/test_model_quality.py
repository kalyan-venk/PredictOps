"""Eval gate: CI fails if the trained model's ROC-AUC drops below the quality bar."""

import joblib
from sklearn.model_selection import train_test_split

from predictops.config import MODEL_PATH, RANDOM_STATE, TEST_SIZE
from predictops.data import TARGET_COL, load_dataset
from predictops.train import evaluate_on_test, main

ROC_AUC_THRESHOLD = 0.80


def test_winning_model_clears_roc_auc_gate() -> None:
    main()

    df = load_dataset()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    model = joblib.load(MODEL_PATH)
    metrics = evaluate_on_test(model, X_test, y_test)

    assert metrics["roc_auc"] >= ROC_AUC_THRESHOLD, (
        f"roc_auc {metrics['roc_auc']:.4f} fell below gate {ROC_AUC_THRESHOLD}"
    )
