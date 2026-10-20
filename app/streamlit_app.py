"""Churn intelligence dashboard. Run from the project root:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="Customer Churn Intelligence", page_icon="📉", layout="wide")

from app.common import require_artifacts
from app.views import batch, eda, insights, overview, performance, predict

PAGES = {
    "Executive Overview": overview.render,
    "Exploratory Data Analysis": eda.render,
    "Model Performance": performance.render,
    "Customer Risk Prediction": predict.render,
    "Batch Prediction": batch.render,
    "Business Insights": insights.render,
}

require_artifacts()
st.sidebar.title("Churn Intelligence")
choice = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
st.sidebar.caption("IBM Telco sample data. Predictions are estimates, not certainties.")
PAGES[choice]()
