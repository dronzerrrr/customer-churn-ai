"""Train, compare, tune, select and evaluate churn models.

Protocol (prevents leakage / test contamination):
  * Hyperparameters tuned with cross-validation on TRAIN only.
  * Models compared on VALIDATION. Decision threshold chosen on VALIDATION.
  * TEST set used exactly once, for the final selected model.
"""
import json
import logging
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    ARTIFACTS_DIR, CV_FOLDS, METADATA_PATH, METRICS_PATH, MODEL_PATH,
    PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, TARGET, THRESHOLD_BETA,
)
from src.evaluation.metrics import best_threshold, compute_metrics
from src.evaluation.plots import save_confusion_matrix, save_curves
from src.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.features.preprocess import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_splits() -> dict[str, pd.DataFrame]:
    paths = {n: PROCESSED_DIR / f"{n}.csv" for n in ("train", "val", "test")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing}. Run: python -m src.data.split")
    return {n: pd.read_csv(p) for n, p in paths.items()}


def split_xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET]), (df[TARGET] == "Yes").astype(int).to_numpy()


def candidate_models(pos_weight: float) -> dict:
    """name -> (estimator, param grid). Grids are deliberately small so the
    whole script finishes in a few minutes on an ordinary laptop."""
    rs = RANDOM_STATE
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=rs),
            {"C": [0.01, 0.1, 1, 10], "class_weight": [None, "balanced"]}),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=rs),
            {"max_depth": [3, 5, 8], "min_samples_leaf": [20],
             "class_weight": [None, "balanced"]}),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, random_state=rs, n_jobs=-1),
            {"max_depth": [8], "min_samples_leaf": [5, 15],
             "class_weight": [None, "balanced"]}),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=rs),
            {"n_estimators": [100, 200], "learning_rate": [0.1], "max_depth": [2, 3]}),
        "XGBoost": (
            XGBClassifier(random_state=rs, n_jobs=2, eval_metric="logloss",
                          n_estimators=200, learning_rate=0.05),
            {"max_depth": [2, 3], "scale_pos_weight": [1.0, float(round(pos_weight, 2))]}),
    }


def main() -> None:
    splits = load_splits()
    X_train, y_train = split_xy(splits["train"])
    X_val, y_val = split_xy(splits["val"])
    X_test, y_test = split_xy(splits["test"])

    churn_rate = y_train.mean()
    pos_weight = (1 - churn_rate) / churn_rate
    log.info("Class balance (train): %.1f%% churn -> imbalance ratio %.2f : 1",
             100 * churn_rate, pos_weight)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fitted: dict[str, Pipeline] = {}
    val_proba: dict[str, np.ndarray] = {}
    best_params: dict[str, dict] = {}

    # Baseline: always predicts the training churn rate
    baseline = Pipeline([("prep", build_preprocessor()),
                         ("model", DummyClassifier(strategy="prior"))]).fit(X_train, y_train)
    fitted["Dummy Baseline"] = baseline
    val_proba["Dummy Baseline"] = baseline.predict_proba(X_val)[:, 1]

    for name, (estimator, grid) in candidate_models(pos_weight).items():
        log.info("Tuning %s ...", name)
        pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
        search = GridSearchCV(
            pipe, {f"model__{k}": v for k, v in grid.items()},
            scoring="average_precision", cv=cv, n_jobs=1, refit=True)
        search.fit(X_train, y_train)  # CV happens inside TRAIN only
        fitted[name] = search.best_estimator_
        best_params[name] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        val_proba[name] = search.best_estimator_.predict_proba(X_val)[:, 1]
        log.info("  best CV PR-AUC=%.4f params=%s", search.best_score_, best_params[name])

    # ---- Compare on VALIDATION
    rows = []
    for name, proba in val_proba.items():
        m = compute_metrics(y_val, proba, threshold=0.5)
        rows.append({"model": name, **{k: m[k] for k in
                     ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")}})
    comparison = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    print("\nVALIDATION comparison (precision/recall/F1 at threshold 0.5):")
    print(comparison.round(4).to_string(index=False))

    # ---- Select: best validation PR-AUC among real models (threshold-independent,
    # suited to imbalanced data). Baseline is excluded from selection.
    real = comparison[comparison["model"] != "Dummy Baseline"]
    selected = real.iloc[0]["model"]
    threshold = best_threshold(y_val, val_proba[selected], beta=THRESHOLD_BETA)
    log.info("Selected model: %s | threshold %.2f (F%.0f on validation)",
             selected, threshold, THRESHOLD_BETA)

    # ---- FINAL evaluation on TEST (first and only time)
    final = fitted[selected]
    test_proba = final.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_proba, threshold)
    test_at_05 = compute_metrics(y_test, test_proba, 0.5)
    baseline_test = compute_metrics(
        y_test, fitted["Dummy Baseline"].predict_proba(X_test)[:, 1], 0.5)
    val_metrics = compute_metrics(y_val, val_proba[selected], threshold)

    print(f"\nFINAL TEST RESULTS - {selected} @ threshold {threshold:.2f}")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        print(f"  {k:10s}: {test_metrics[k]:.4f}")
    print(f"  confusion : {test_metrics['confusion_matrix']}")
    print(f"  (dummy baseline accuracy on test: {baseline_test['accuracy']:.4f}, "
          f"recall {baseline_test['recall']:.2f})")

    # ---- Save artifacts
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    fig_dir = REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    save_confusion_matrix(y_test, test_proba, threshold, fig_dir / "confusion_matrix_test.png",
                          f"{selected} (test, threshold {threshold:.2f})")
    save_curves(y_val, {n: p for n, p in val_proba.items()},
                fig_dir / "roc_validation.png", fig_dir / "pr_validation.png", "validation")
    save_curves(y_test, {selected: test_proba},
                fig_dir / "roc_test.png", fig_dir / "pr_test.png", "test")

    METRICS_PATH.write_text(json.dumps({
        "validation_comparison": comparison.to_dict(orient="records"),
        "selected_model": selected,
        "validation_metrics_at_threshold": val_metrics,
        "test_metrics": test_metrics,
        "test_metrics_at_0.5": test_at_05,
        "dummy_baseline_test": baseline_test,
        "split_sizes": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "train_churn_rate": float(churn_rate),
    }, indent=2))
    METADATA_PATH.write_text(json.dumps({
        "selected_model": selected,
        "threshold": threshold,
        "threshold_beta": THRESHOLD_BETA,
        "best_params": {k: {p: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                            for p, v in d.items()} for k, d in best_params.items()},
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "random_state": RANDOM_STATE,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }, indent=2))
    log.info("Saved model + metrics to %s", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
