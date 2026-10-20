"""Pydantic models for the API. Allowed values come from src.models.schema."""
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.features.build_features import RAW_FEATURES
from src.models.schema import (
    MAX_BATCH_ROWS, MONTHLY_BOUNDS, TENURE_BOUNDS, TOTAL_CHARGES_MAX,
    consistency_problems, estimate_total_charges,
)


# Explicit Literal types: static type checkers (Pylance/Pyright) understand these, and
# Swagger shows them as dropdown enums. tests/test_api.py asserts they stay identical
# to ALLOWED_VALUES in src/models/schema.py, so there is still one source of truth.
YesNo = Literal["No", "Yes"]
AddOn = Literal["No", "No internet service", "Yes"]


EXAMPLE_CUSTOMER = {
    "customerID": "EXAMPLE-0001", "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
    "Dependents": "No", "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5,
}


class CustomerInput(BaseModel):
    """One customer. TotalCharges is optional (estimated as tenure x MonthlyCharges)."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    customerID: str | None = Field(None, max_length=64, description="Optional; echoed back.")
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=TENURE_BOUNDS[0], le=TENURE_BOUNDS[1], description="Months with the company.")
    PhoneService: YesNo
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Bank transfer (automatic)", "Credit card (automatic)",
        "Electronic check", "Mailed check"]
    MonthlyCharges: float = Field(ge=MONTHLY_BOUNDS[0], le=MONTHLY_BOUNDS[1])
    TotalCharges: float | None = Field(None, ge=0, le=TOTAL_CHARGES_MAX)

    @model_validator(mode="after")
    def _check_combinations(self):
        problems = consistency_problems(self.model_dump())
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def to_frame(self) -> tuple[pd.DataFrame, bool]:
        row = self.model_dump()
        estimated = row["TotalCharges"] is None
        if estimated:
            row["TotalCharges"] = estimate_total_charges(row["tenure"], row["MonthlyCharges"])
        return pd.DataFrame([row], columns=RAW_FEATURES), estimated


class Reason(BaseModel):
    feature: str
    value: Any = None
    shap_log_odds: float
    effect: str


class PredictionResponse(BaseModel):
    customerID: str | None = None
    churn_probability: float
    predicted_churn: bool
    risk_band: str
    threshold: float
    total_charges_estimated: bool
    reasons: list[Reason] | None = None
    disclaimer: str = ("Probabilities are model estimates, not certainties. "
                       "Reasons describe associations the model learned, not causes of churn.")


class BatchRequest(BaseModel):
    customers: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_BATCH_ROWS)


class BatchPrediction(BaseModel):
    row: int
    customerID: str | None = None
    churn_probability: float
    predicted_churn: bool
    risk_band: str


class BatchError(BaseModel):
    row: int
    field: str
    message: str


class BatchResponse(BaseModel):
    n_received: int
    n_scored: int
    n_rejected: int
    n_total_charges_estimated: int
    threshold: float
    predictions: list[BatchPrediction]
    errors: list[BatchError]
    errors_truncated: bool = False
