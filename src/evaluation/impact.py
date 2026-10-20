"""Scenario calculator for a retention campaign.

This is arithmetic on ASSUMPTIONS the user supplies. The data cannot tell us how much an
offer would reduce churn, so nothing here is a forecast or a measured result.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ImpactAssumptions:
    targeted: int                 # customers contacted
    baseline_churn: float         # observed churn rate of that group (0-1)
    relative_reduction: float     # ASSUMED share of would-be churners retained (0-1)
    cost_per_offer: float         # cost per contacted customer
    monthly_charge: float         # average monthly charge of the group
    margin: float                 # share of the charge that is profit (0-1)
    months_retained: float        # months a retained customer is assumed to stay

    def __post_init__(self):
        for name in ("baseline_churn", "relative_reduction", "margin"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        for name in ("targeted", "cost_per_offer", "monthly_charge", "months_retained"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative.")


def estimate_impact(a: ImpactAssumptions) -> dict:
    churners = a.targeted * a.baseline_churn
    retained = churners * a.relative_reduction
    value_per_retained = a.monthly_charge * a.margin * a.months_retained
    value = retained * value_per_retained
    cost = a.targeted * a.cost_per_offer
    denominator = churners * value_per_retained
    return {
        "expected_churners": churners,
        "customers_retained": retained,
        "value_per_retained_customer": value_per_retained,
        "value_retained": value,
        "campaign_cost": cost,
        "net_value": value - cost,
        "break_even_reduction": (cost / denominator) if denominator > 0 else None,
        "offers_per_churner_reached": (1 / a.baseline_churn) if a.baseline_churn > 0 else None,
    }


def sensitivity(a: ImpactAssumptions, reductions: list[float]) -> pd.DataFrame:
    rows = []
    for r in reductions:
        res = estimate_impact(ImpactAssumptions(**{**a.__dict__, "relative_reduction": r}))
        rows.append({"relative_reduction": r, "customers_retained": res["customers_retained"],
                     "net_value": res["net_value"]})
    return pd.DataFrame(rows)
