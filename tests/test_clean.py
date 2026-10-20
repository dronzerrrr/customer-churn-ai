import pandas as pd

from src.data.clean import clean_for_analysis


def sample() -> pd.DataFrame:
    return pd.DataFrame({
        "tenure": [0, 5, 30, 70],
        "TotalCharges": [" ", "100.5", "900", "5000"],
        "Churn": ["No", "Yes", "No", "Yes"],
        "Contract": [" Month-to-month", "One year", "Two year", "Two year"],
    })


def test_blank_total_charges_becomes_nan_not_zero():
    out = clean_for_analysis(sample())
    assert pd.isna(out.loc[0, "TotalCharges"])
    assert out["TotalCharges"].dtype.kind == "f"


def test_churn_flag_and_whitespace():
    out = clean_for_analysis(sample())
    assert out["churn_flag"].tolist() == [0, 1, 0, 1]
    assert out.loc[0, "Contract"] == "Month-to-month"


def test_tenure_groups():
    out = clean_for_analysis(sample())
    assert out["tenure_group"].tolist() == [
        "0-12 months", "0-12 months", "25-48 months", "49-72 months"]


def test_input_not_mutated():
    df = sample()
    clean_for_analysis(df)
    assert "churn_flag" not in df.columns
