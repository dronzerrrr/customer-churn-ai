"""One prediction service shared by the API and the dashboard."""
import json
import logging
from typing import Any

import pandas as pd

from src.config import METRICS_PATH, PROCESSED_DIR, RANDOM_STATE, TARGET
from src.models.predict import load_model
from src.models.risk import band_rules, risk_bands

log = logging.getLogger(__name__)


class ChurnService:
    def __init__(self, model, metadata: dict, metrics: dict | None = None,
                 background: pd.DataFrame | None = None):
        self.model = model
        self.metadata = metadata
        self.metrics = metrics or {}
        self.background = background
        self._explainer = None

    @classmethod
    def load(cls) -> "ChurnService":
        model, metadata = load_model()           # raises FileNotFoundError if untrained
        metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
        train_path = PROCESSED_DIR / "train.csv"
        background = None
        if train_path.exists():
            background = (pd.read_csv(train_path).drop(columns=[TARGET])
                          .sample(300, random_state=RANDOM_STATE))
        return cls(model, metadata, metrics, background)

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """X must already be validated (RAW_FEATURES columns, numeric TotalCharges)."""
        proba = self.model.predict_proba(X)[:, 1].astype("float64")
        return pd.DataFrame({
            "churn_probability": proba.round(4),
            "predicted_churn": proba >= self.threshold,
            "risk_band": risk_bands(proba, self.threshold),
        }, index=X.index)

    def explain(self, X_one: pd.DataFrame, top_k: int = 6) -> dict[str, Any]:
        from src.evaluation.explain import ChurnExplainer   # imports shap lazily
        if self._explainer is None:
            self._explainer = ChurnExplainer(self.model, background_df=self.background)
        result = self._explainer.explain_customer(X_one, top_k=top_k)
        for r in result["reasons"]:
            v = r["value"]
            r["value"] = v.item() if hasattr(v, "item") else v
        return result

    def info(self) -> dict[str, Any]:
        test = self.metrics.get("test_metrics", {})
        return {
            "selected_model": self.metadata["selected_model"],
            "threshold": self.threshold,
            "threshold_beta": self.metadata.get("threshold_beta"),
            "risk_band_rules": band_rules(self.threshold),
            "trained_at_utc": self.metadata.get("trained_at_utc"),
            "versions": self.metadata.get("versions", {}),
            "numeric_features": self.metadata.get("numeric_features", []),
            "categorical_features": self.metadata.get("categorical_features", []),
            "held_out_test_metrics": {k: test[k] for k in
                                      ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
                                      if k in test},
            "note": "Metrics were measured on a held-out test set when the model was trained. "
                    "Probabilities are estimates, not certainties.",
        }
