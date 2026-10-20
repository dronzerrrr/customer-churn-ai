"""Shared test helpers: valid records and a small fitted toy service."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocess import build_preprocessor
from src.models.schema import ADDON_COLUMNS
from src.models.service import ChurnService


def valid_record(**overrides) -> dict:
    rec = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 477.5,
    }
    rec.update(overrides)
    return rec


def valid_frame(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Random but internally consistent customers with a learnable churn signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        internet = rng.choice(["DSL", "Fiber optic", "No"])
        phone = rng.choice(["Yes", "No"])
        addon = "No internet service" if internet == "No" else None
        rec = valid_record(
            tenure=int(rng.integers(0, 73)), MonthlyCharges=float(rng.uniform(20, 110)),
            InternetService=internet, PhoneService=phone,
            MultipleLines="No phone service" if phone == "No" else rng.choice(["Yes", "No"]),
            Contract=rng.choice(["Month-to-month", "One year", "Two year"]),
            PaymentMethod=rng.choice(["Electronic check", "Mailed check",
                                      "Bank transfer (automatic)", "Credit card (automatic)"]),
            **{c: addon or rng.choice(["Yes", "No"]) for c in ADDON_COLUMNS})
        rec["TotalCharges"] = round(rec["tenure"] * rec["MonthlyCharges"], 2)
        rows.append(rec)
    X = pd.DataFrame(rows)
    score = (X["Contract"].eq("Month-to-month") * 1.5 - X["tenure"] / 40
             + rng.normal(0, 0.5, n))
    return X, (score > 0.2).astype(int).to_numpy()


def toy_service(threshold: float = 0.4) -> ChurnService:
    X, y = valid_frame()
    model = Pipeline([("prep", build_preprocessor()),
                      ("model", LogisticRegression(max_iter=1000))]).fit(X, y)
    meta = {"selected_model": "Toy Logistic Regression", "threshold": threshold,
            "threshold_beta": 2.0, "trained_at_utc": "test", "versions": {}}
    return ChurnService(model, meta, {}, background=X.head(100))
