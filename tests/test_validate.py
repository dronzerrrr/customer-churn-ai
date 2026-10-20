import pandas as pd

from src.config import EXPECTED_COLUMNS
from src.data.validate import validate_raw


def make_df(**overrides) -> pd.DataFrame:
    row = {c: "x" for c in EXPECTED_COLUMNS}
    row.update(customerID="A-1", tenure=5, MonthlyCharges=50.0,
               TotalCharges="250.0", Churn="No", SeniorCitizen=0)
    row.update(overrides)
    return pd.DataFrame([row])


def test_valid_data_passes():
    assert validate_raw(make_df())["errors"] == []


def test_missing_column_is_error():
    df = make_df().drop(columns=["Contract"])
    assert any("Missing" in e for e in validate_raw(df)["errors"])


def test_bad_target_value_is_error():
    assert validate_raw(make_df(Churn="Maybe"))["errors"]


def test_blank_total_charges_is_warning_not_error():
    rep = validate_raw(make_df(TotalCharges=" "))
    assert rep["errors"] == [] and rep["total_charges_unparseable"] == 1


def test_negative_tenure_is_error():
    assert validate_raw(make_df(tenure=-1))["errors"]
