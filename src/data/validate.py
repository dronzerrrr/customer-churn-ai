"""Data quality checks. Reports problems; does NOT silently fix them."""
import logging

import pandas as pd

from src.config import EXPECTED_COLUMNS, ID_COL, TARGET
from src.data.load_data import load_raw_data

log = logging.getLogger(__name__)


def validate_raw(df: pd.DataFrame) -> dict:
    report = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "errors": [],
        "warnings": [],
    }

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        report["errors"].append(f"Missing required columns: {missing_cols}")
        return report  # nothing else is safe to check

    if df.empty:
        report["errors"].append("Dataset has no rows.")
        return report

    # Duplicates
    dup_ids = int(df[ID_COL].duplicated().sum())
    dup_rows = int(df.duplicated().sum())
    report["duplicate_ids"], report["duplicate_rows"] = dup_ids, dup_rows
    if dup_ids:
        report["warnings"].append(f"{dup_ids} duplicate customerID values.")

    # Target
    bad_target = set(df[TARGET].dropna().unique()) - {"Yes", "No"}
    if bad_target:
        report["errors"].append(f"Unexpected values in {TARGET}: {bad_target}")
    else:
        report["churn_rate"] = float((df[TARGET] == "Yes").mean())

    # Missing values (true NaN)
    nulls = df.isna().sum()
    report["null_counts"] = {c: int(n) for c, n in nulls.items() if n > 0}

    # TotalCharges is text in the raw file; blanks hide as strings, not NaN
    total_num = pd.to_numeric(df["TotalCharges"], errors="coerce")
    unparseable = int(total_num.isna().sum() - df["TotalCharges"].isna().sum())
    report["total_charges_unparseable"] = unparseable
    if unparseable:
        report["warnings"].append(
            f"{unparseable} TotalCharges values are blank/non-numeric (handled in preprocessing)."
        )

    # Sanity ranges
    if (df["tenure"] < 0).any():
        report["errors"].append("Negative tenure values found.")
    if (pd.to_numeric(df["MonthlyCharges"], errors="coerce") <= 0).any():
        report["warnings"].append("MonthlyCharges <= 0 found.")

    return report


def print_report(report: dict) -> None:
    print("=" * 50)
    print("DATA VALIDATION REPORT")
    print("=" * 50)
    print(f"Rows: {report['n_rows']}   Columns: {report['n_cols']}")
    if "churn_rate" in report:
        print(f"Churn rate: {report['churn_rate']:.2%}")
    for key in ("duplicate_ids", "duplicate_rows", "total_charges_unparseable"):
        if key in report:
            print(f"{key}: {report[key]}")
    print(f"Columns with nulls: {report.get('null_counts', {}) or 'none'}")
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    for e in report["errors"]:
        print(f"ERROR: {e}")
    print("RESULT:", "FAILED" if report["errors"] else "PASSED")


if __name__ == "__main__":
    rep = validate_raw(load_raw_data())
    print_report(rep)
    raise SystemExit(1 if rep["errors"] else 0)
