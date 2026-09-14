"""
evaluate.py
Evaluation metrics appropriate for imbalanced fraud detection.

IMPORTANT: accuracy is not reported as a headline metric here on purpose.
With ~0.17% fraud rate, a model that predicts "not fraud" every time would
score ~99.8% accuracy while catching zero fraud. Use precision, recall,
F1, and AUPRC instead.
"""

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test, y_test, model_name: str = "model") -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "auprc": average_precision_score(y_test, y_proba),  # key metric for this task
    }

    print(f"\n--- {model_name} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return metrics
