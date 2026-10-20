"""Risk bands. A presentation convention built on the model's decision threshold,
NOT calibrated probabilities."""
import numpy as np
import pandas as pd

BAND_ORDER = ["Low", "Medium", "High"]
MEDIUM_FRACTION = 0.5   # Medium starts at half the decision threshold


def band_edges(threshold: float) -> dict[str, tuple[float, float]]:
    mid = threshold * MEDIUM_FRACTION
    return {"Low": (0.0, mid), "Medium": (mid, threshold), "High": (threshold, 1.0)}


def band_rules(threshold: float) -> str:
    mid = threshold * MEDIUM_FRACTION
    return (f"High: probability >= {threshold:.0%} (the decision threshold, flagged for outreach). "
            f"Medium: {mid:.0%} to {threshold:.0%}. Low: below {mid:.0%}.")


def risk_band(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "High"
    return "Medium" if probability >= threshold * MEDIUM_FRACTION else "Low"


def risk_bands(probabilities, threshold: float) -> list[str]:
    return [risk_band(float(p), threshold) for p in probabilities]


def band_summary(y_true, proba, threshold: float) -> pd.DataFrame:
    """OBSERVED churn rate per band on labelled data (e.g. validation set)."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "band": risk_bands(proba, threshold)})
    out = (frame.groupby("band")["y"].agg(customers="size", churned="sum", churn_rate="mean")
                .reindex(BAND_ORDER).dropna().reset_index())
    out["customers"] = out["customers"].astype(int)
    out["churned"] = out["churned"].astype(int)
    return out
