"""Inference helpers. Uses the SAME saved pipeline as training."""
import json

import joblib
import pandas as pd

from src.config import METADATA_PATH, MODEL_PATH
from src.features.build_features import RAW_FEATURES


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run: python -m src.models.train")
    return joblib.load(MODEL_PATH), json.loads(METADATA_PATH.read_text())


def predict_df(df: pd.DataFrame, model=None, threshold: float | None = None) -> pd.DataFrame:
    if model is None or threshold is None:
        model, meta = load_model()
        threshold = meta["threshold"] if threshold is None else threshold
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    proba = model.predict_proba(df)[:, 1]
    out = df.copy()
    out["churn_probability"] = proba
    out["predicted_churn"] = (proba >= threshold).astype(int)
    return out
