"""Light cleaning for analytics/EDA. Modeling-time imputation happens later,
inside the sklearn Pipeline, so it is fit on training data only."""
import pandas as pd

from src.config import TARGET

TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]


def clean_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Strip stray whitespace in text columns
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].str.strip()
    # TotalCharges arrives as text; blanks become NaN (NOT filled here)
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    # Numeric target for easy averaging: mean(churn_flag) = churn rate
    out["churn_flag"] = (out[TARGET] == "Yes").astype(int)
    # Business-friendly tenure groups for analysis
    out["tenure_group"] = pd.cut(
        out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
    ).astype(str)
    return out
