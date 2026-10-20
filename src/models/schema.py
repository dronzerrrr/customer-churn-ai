"""Single source of truth for what a valid customer record looks like.
Used by the API, the batch validator and the dashboard."""
from typing import Mapping

from src.features.build_features import RAW_FEATURES

ADDON_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
_YES_NO = ["No", "Yes"]
_ADDON_VALUES = ["No", "No internet service", "Yes"]

ALLOWED_VALUES: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{c: _ADDON_VALUES for c in ADDON_COLUMNS},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)",
                      "Electronic check", "Mailed check"],
}
SENIOR_VALUES = [0, 1]
# Inclusive bounds. tenure is capped at 72 because the model was trained on 0-72 months.
TENURE_BOUNDS = (0, 72)
MONTHLY_BOUNDS = (0.01, 500.0)
TOTAL_CHARGES_MAX = 100_000.0
MAX_BATCH_ROWS = 10_000

# TotalCharges is optional: when absent it is estimated as tenure x MonthlyCharges.
REQUIRED_COLUMNS = [c for c in RAW_FEATURES if c != "TotalCharges"]


def estimate_total_charges(tenure: float, monthly_charges: float) -> float:
    """Approximation used when TotalCharges is not supplied (tenure x monthly)."""
    return round(float(tenure) * float(monthly_charges), 2)


def consistency_problems(row: Mapping) -> list[str]:
    """Combinations that cannot occur in real records (and the model never saw)."""
    problems = []
    internet, phone = row.get("InternetService"), row.get("PhoneService")
    if internet == "No":
        bad = [c for c in ADDON_COLUMNS if row.get(c) != "No internet service"]
        if bad:
            problems.append("InternetService is 'No', so these must be "
                            f"'No internet service': {', '.join(bad)}")
    elif internet in ("DSL", "Fiber optic"):
        bad = [c for c in ADDON_COLUMNS if row.get(c) == "No internet service"]
        if bad:
            problems.append("Customer has internet, so these cannot be "
                            f"'No internet service': {', '.join(bad)}")
    if phone == "No" and row.get("MultipleLines") != "No phone service":
        problems.append("PhoneService is 'No', so MultipleLines must be 'No phone service'")
    elif phone == "Yes" and row.get("MultipleLines") == "No phone service":
        problems.append("Customer has phone service, so MultipleLines cannot be 'No phone service'")
    return problems
