"""Save evaluation figures (confusion matrix, ROC, precision-recall)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
)


def save_confusion_matrix(y_true, proba, threshold: float, path: Path, title: str) -> None:
    pred = (proba >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, pred, display_labels=["Stayed", "Churned"], cmap="Blues", ax=ax)
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_curves(y_true, proba_by_model: dict, roc_path: Path, pr_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, proba in proba_by_model.items():
        RocCurveDisplay.from_predictions(y_true, proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC curve - {title}")
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, proba in proba_by_model.items():
        PrecisionRecallDisplay.from_predictions(y_true, proba, name=name, ax=ax)
    ax.axhline(float(y_true.mean()), ls="--", color="k", lw=1, label="No-skill (churn rate)")
    ax.legend(fontsize=7)
    ax.set_title(f"Precision-recall curve - {title}")
    fig.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
