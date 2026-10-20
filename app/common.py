"""Shared loaders and helpers for the Streamlit app. All numbers come from
the real dataset and the artifacts written by training - nothing is hardcoded."""
import json

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    METADATA_PATH, METRICS_PATH, MODEL_PATH, PROCESSED_DIR, RAW_CSV, TARGET,
)
from src.data.clean import clean_for_analysis
from src.data.load_data import load_raw_data
from src.models.predict import load_model
from src.models.risk import band_summary
from src.models.service import ChurnService

STAY_COLOR = "#4C78A8"
CHURN_COLOR = "#E45756"
CHURN_COLORS = {"No": STAY_COLOR, "Yes": CHURN_COLOR}
TENURE_ORDER = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]

REQUIRED_FILES = {
    "dataset": (RAW_CSV, "python -m src.data.download_data"),
    "train/val/test splits": (PROCESSED_DIR / "test.csv", "python -m src.data.split"),
    "trained model": (MODEL_PATH, "python -m src.models.train"),
    "metrics": (METRICS_PATH, "python -m src.models.train"),
    "model metadata": (METADATA_PATH, "python -m src.models.train"),
}


def require_artifacts() -> None:
    missing = [(n, cmd) for n, (p, cmd) in REQUIRED_FILES.items() if not p.exists()]
    if missing:
        st.error("Some required files are missing. Run these commands from the project root:")
        for cmd in dict.fromkeys(command for _, command in missing):   # each command once
            st.code(cmd, language="powershell")
        st.stop()


@st.cache_data(show_spinner=False)
def load_customers() -> pd.DataFrame:
    return clean_for_analysis(load_raw_data())


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text())


@st.cache_resource(show_spinner=False)
def load_trained_model():
    return load_model()[0]


@st.cache_data(show_spinner="Scoring customers...")
def split_predictions(split: str) -> tuple[np.ndarray, np.ndarray]:
    """True labels and predicted churn probabilities for 'val' or 'test'."""
    df = pd.read_csv(PROCESSED_DIR / f"{split}.csv")
    y = (df[TARGET] == "Yes").astype(int).to_numpy()
    proba = load_trained_model().predict_proba(df.drop(columns=[TARGET]))[:, 1]
    return y, proba


def pct(x: float) -> str:
    return f"{x:.1%}"

@st.cache_resource(show_spinner="Loading model...")
def load_service() -> ChurnService:
    return ChurnService.load()


@st.cache_data(show_spinner=False)
def validation_band_table() -> pd.DataFrame:
    """OBSERVED churn rate per risk band on the validation set."""
    y, proba = split_predictions("val")
    return band_summary(y, proba, load_service().threshold)
