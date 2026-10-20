"""Row-wise feature engineering shared by training AND inference.

Every operation here looks at ONE row at a time (no dataset-wide statistics),
so it cannot leak information from the test set into training.
"""
import pandas as pd

ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
DROP_COLS = ["customerID", "Churn", "churn_flag", "tenure_group"]

NUMERIC_FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "num_addon_services",
]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_band",
]
RAW_FEATURES = [
    c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if c not in ("num_addon_services", "tenure_band")
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a model-ready copy: coerce types, add engineered columns,
    drop identifiers and the target (if present)."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].str.strip()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce")
    out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"], errors="coerce")

    # How many optional add-on services does the customer pay for?
    out["num_addon_services"] = sum(
        (out[c] == "Yes").astype(int) for c in ADDON_SERVICES
    )
    # Coarse tenure band lets linear models capture non-linear tenure effects
    out["tenure_band"] = pd.cut(
        out["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    ).astype(str)

    out = out.drop(columns=[c for c in DROP_COLS if c in out.columns])
    return out[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
