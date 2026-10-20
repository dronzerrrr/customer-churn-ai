"""Metric helpers. Every number is computed from real predictions."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    fbeta_score, precision_score, recall_score, roc_auc_score,
)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_threshold(y_true, proba, beta: float = 2.0) -> float:
    """Threshold maximising F-beta on the data given (use VALIDATION data)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    candidates = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = [
        fbeta_score(y_true, (proba >= t).astype(int), beta=beta, zero_division=0)
        for t in candidates
    ]
    return float(candidates[int(np.argmax(scores))])
