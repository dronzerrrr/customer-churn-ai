"""Model explainability: global + per-customer SHAP, plus permutation importance.

IMPORTANT: these explain what the MODEL uses to predict. They describe
associations learned from data, NOT causes of churn.

SHAP values for tree/linear models here are in LOG-ODDS units: positive
pushes the predicted churn risk up, negative pushes it down. Contributions
add up (with the base value) to the model's log-odds output.
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from src.config import PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, TARGET
from src.features.build_features import CATEGORICAL_FEATURES, RAW_FEATURES, add_features
from src.models.predict import load_model

log = logging.getLogger(__name__)


def original_feature_name(col: str) -> str:
    """'cat__Contract_One year' -> 'Contract'; 'num__tenure' -> 'tenure'."""
    name = col.split("__", 1)[1]
    if col.startswith("num__"):
        return name.replace("missingindicator_", "")
    matches = [f for f in CATEGORICAL_FEATURES if name.startswith(f + "_")]
    return max(matches, key=len) if matches else name


class ChurnExplainer:
    def __init__(self, model, background_df: pd.DataFrame | None = None):
        self.model = model
        self.prep = model.named_steps["prep"]         # fitted preprocessing pipeline
        self.estimator = model.named_steps["model"]   # fitted classifier
        self.columns = list(self.prep.named_steps["columns"].get_feature_names_out())
        self.groups = [original_feature_name(c) for c in self.columns]

        if isinstance(self.estimator, LogisticRegression):
            if background_df is None:
                raise ValueError("A background sample is required for linear models.")
            bg = self.prep.transform(background_df)
            self.explainer = shap.LinearExplainer(self.estimator, bg)
        else:
            self.explainer = shap.TreeExplainer(self.estimator)

    def _values(self, X_t: np.ndarray) -> np.ndarray:
        vals = self.explainer.shap_values(X_t)
        if isinstance(vals, list):          # older SHAP: [class0, class1]
            vals = vals[1]
        vals = np.asarray(vals)
        if vals.ndim == 3:                  # newer SHAP for some models
            vals = vals[:, :, 1]
        return vals

    def _base(self) -> float:
        base = np.asarray(self.explainer.expected_value).ravel()
        return float(base[-1])

    def shap_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_t = self.prep.transform(df)
        return X_t, self._values(X_t)

    def grouped_shap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-row SHAP summed back to the ORIGINAL features (one-hot columns merged)."""
        _, vals = self.shap_matrix(df)
        frame = pd.DataFrame(vals, columns=self.groups)
        return frame.T.groupby(level=0).sum().T

    def explain_customer(self, row: pd.DataFrame, top_k: int = 6) -> dict:
        """Explain ONE customer (a 1-row DataFrame of raw input columns)."""
        if len(row) != 1:
            raise ValueError("explain_customer expects exactly one row.")
        grouped = self.grouped_shap(row).iloc[0]
        shown = add_features(row).iloc[0]          # values the model actually sees
        proba = float(self.model.predict_proba(row)[0, 1])
        top = grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(top_k)
        reasons = [{
            "feature": f,
            "value": shown.get(f, None),
            "shap_log_odds": float(v),
            "effect": "raises predicted risk" if v > 0 else "lowers predicted risk",
        } for f, v in top.items()]
        return {"probability": proba, "base_log_odds": self._base(), "reasons": reasons}


def global_importance(explainer: ChurnExplainer, df: pd.DataFrame) -> pd.Series:
    return explainer.grouped_shap(df).abs().mean().sort_values(ascending=False)


def permutation_importance_table(model, df: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Model-agnostic cross-check: drop in PR-AUC when a feature is shuffled."""
    res = permutation_importance(
        model, df[RAW_FEATURES], y, scoring="average_precision",
        n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    return pd.Series(res.importances_mean, index=RAW_FEATURES).sort_values(ascending=False)


def save_global_plot(importance: pd.Series, path: Path, top: int = 12) -> None:
    top_imp = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(top_imp.index, top_imp.values, color="#4C78A8")
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title("Which features the model relies on most (validation data)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_beeswarm(explainer: ChurnExplainer, df: pd.DataFrame, path: Path) -> None:
    X_t, vals = explainer.shap_matrix(df)
    plt.figure()
    shap.summary_plot(vals, X_t, feature_names=explainer.columns, max_display=12, show=False)
    plt.title("SHAP summary (red = high feature value)")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    model, meta = load_model()
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")   # explain on VALIDATION, keep test untouched
    y_val = (val[TARGET] == "Yes").astype(int).to_numpy()
    X_val = val.drop(columns=[TARGET])

    exp = ChurnExplainer(model, background_df=train.drop(columns=[TARGET]).sample(
        300, random_state=RANDOM_STATE))
    shap_imp = global_importance(exp, X_val)
    perm_imp = permutation_importance_table(model, X_val, y_val)

    table = pd.DataFrame({"mean_abs_shap": shap_imp, "permutation_pr_auc_drop": perm_imp})
    table = table.sort_values("mean_abs_shap", ascending=False)
    fig_dir = REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table.round(5).to_csv(REPORTS_DIR / "feature_importance.csv")
    save_global_plot(shap_imp, fig_dir / "shap_global.png")
    save_beeswarm(exp, X_val.sample(500, random_state=RANDOM_STATE), fig_dir / "shap_beeswarm.png")

    print(f"Model explained: {meta['selected_model']}")
    print("\nTop 10 features (SHAP) next to permutation importance:")
    print(table.head(10).round(4).to_string())

    proba = model.predict_proba(X_val)[:, 1]
    for label, idx in (("HIGHEST-risk", int(np.argmax(proba))), ("LOWEST-risk", int(np.argmin(proba)))):
        result = exp.explain_customer(X_val.iloc[[idx]])
        print(f"\n{label} validation customer - predicted churn probability {result['probability']:.1%}")
        for r in result["reasons"]:
            print(f"  {r['feature']:<22} value={str(r['value']):<18} "
                  f"{r['shap_log_odds']:+.2f}  ({r['effect']})")


if __name__ == "__main__":
    main()
