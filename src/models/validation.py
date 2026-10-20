"""Row-level validation for batch scoring (CSV upload and API batch)."""
from dataclasses import dataclass, field

import pandas as pd

from src.features.build_features import RAW_FEATURES
from src.models.schema import (
    ALLOWED_VALUES, MONTHLY_BOUNDS, REQUIRED_COLUMNS, SENIOR_VALUES, TENURE_BOUNDS,
    TOTAL_CHARGES_MAX, consistency_problems,
)


@dataclass
class ValidationResult:
    valid: pd.DataFrame                      # RAW_FEATURES columns; index = 0-based input position
    errors: list[dict] = field(default_factory=list)   # {"row": 1-based, "field", "message"}
    missing_columns: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_total_charges_estimated: int = 0

    @property
    def n_rejected(self) -> int:
        return self.n_rows - len(self.valid) if not self.missing_columns else self.n_rows


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def validate_batch(df: pd.DataFrame) -> ValidationResult:
    """Validate every row. Bad rows are reported, never silently fixed or dropped."""
    df = df.reset_index(drop=True).copy()
    df.columns = [str(c).strip() for c in df.columns]
    n = len(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return ValidationResult(pd.DataFrame(columns=RAW_FEATURES), [], missing, n)

    errors: list[dict] = []
    bad_rows: set[int] = set()

    def flag(mask: pd.Series, col: str, message) -> None:
        for i in df.index[mask.fillna(True)]:
            errors.append({"row": int(i) + 1, "field": col,
                           "message": message(df.at[i, col]) if callable(message) else message})
            bad_rows.add(int(i))

    # Categorical columns
    for col, allowed in ALLOWED_VALUES.items():
        missing_mask = df[col].isna()
        cleaned = df[col].astype(str).str.strip()
        bad = missing_mask | ~cleaned.isin(allowed)
        df[col] = cleaned.where(~missing_mask, None)
        flag(bad, col, lambda v, a=allowed: "is missing" if v is None or pd.isna(v)
             else f"'{v}' is not valid; allowed values: {', '.join(a)}")

    # Numeric columns
    senior = _num(df["SeniorCitizen"])
    flag(~senior.isin(SENIOR_VALUES), "SeniorCitizen", "must be 0 or 1")
    df["SeniorCitizen"] = senior

    tenure = _num(df["tenure"])
    lo, hi = TENURE_BOUNDS
    flag(tenure.isna() | (tenure < lo) | (tenure > hi) | (tenure % 1 != 0), "tenure",
         f"must be a whole number of months between {lo} and {hi}")
    df["tenure"] = tenure

    monthly = _num(df["MonthlyCharges"])
    lo, hi = MONTHLY_BOUNDS
    flag(monthly.isna() | (monthly < lo) | (monthly > hi), "MonthlyCharges",
         f"must be a number between {lo} and {hi}")
    df["MonthlyCharges"] = monthly

    # TotalCharges: optional; blanks (or absent column) are estimated
    if "TotalCharges" in df.columns:
        raw = df["TotalCharges"]
        blank = raw.isna() | (raw.astype(str).str.strip() == "")
        total = _num(raw)
        flag(~blank & (total.isna() | (total < 0) | (total > TOTAL_CHARGES_MAX)),
             "TotalCharges", f"must be a number between 0 and {TOTAL_CHARGES_MAX:,.0f}, or left blank")
    else:
        blank = pd.Series(True, index=df.index)
        total = pd.Series(float("nan"), index=df.index)
    estimate = (tenure * monthly).round(2)
    df["TotalCharges"] = total.where(~blank, estimate)

    # Impossible combinations (only for rows whose individual values are valid)
    ok = df.index[~df.index.isin(bad_rows)]
    for i in ok:
        for problem in consistency_problems(df.loc[i].to_dict()):
            errors.append({"row": int(i) + 1, "field": "(combination)", "message": problem})
            bad_rows.add(int(i))

    keep = [i for i in df.index if i not in bad_rows]
    valid = df.loc[keep, RAW_FEATURES].copy()
    valid["tenure"] = valid["tenure"].astype(int)
    valid["SeniorCitizen"] = valid["SeniorCitizen"].astype(int)
    n_est = int(blank.loc[keep].sum()) if keep else 0
    errors.sort(key=lambda e: e["row"])
    return ValidationResult(valid, errors, [], n, n_est)
