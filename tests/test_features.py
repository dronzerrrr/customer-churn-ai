import pandas as pd

from src.data.split import make_splits
from src.features.build_features import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_features)
from src.features.preprocess import build_preprocessor


def make_df(n: int = 200) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "customerID": f"C-{i}", "gender": "Male" if i % 2 else "Female",
            "SeniorCitizen": i % 2, "Partner": "Yes", "Dependents": "No",
            "tenure": i % 73, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "Yes", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Mailed check",
            "MonthlyCharges": 50.0 + i % 10,
            "TotalCharges": " " if i % 73 == 0 else str(50.0 * (i % 73)),
            "Churn": "Yes" if i % 4 == 0 else "No",
        })
    return pd.DataFrame(rows)


def test_add_features_drops_id_and_target_and_counts_addons():
    out = add_features(make_df(10))
    assert "customerID" not in out.columns and "Churn" not in out.columns
    assert list(out.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert (out["num_addon_services"] == 2).all()  # OnlineSecurity + TechSupport


def test_pipeline_has_no_nans_and_handles_blank_total_charges():
    df = make_df()
    X = build_preprocessor().fit_transform(df)
    assert not pd.isna(X).any()
    assert X.shape[0] == len(df)


def test_unknown_category_does_not_crash_inference():
    df = make_df()
    pipe = build_preprocessor().fit(df)
    new = df.head(3).copy()
    new["Contract"] = "Brand-new-contract"
    assert pipe.transform(new).shape[0] == 3


def test_inference_input_without_target_column_works():
    df = make_df()
    pipe = build_preprocessor().fit(df)
    assert pipe.transform(df.drop(columns=["Churn", "customerID"]).head(2)).shape[0] == 2


def test_splits_are_disjoint_stratified_and_sized():
    df = make_df(1000)
    s = make_splits(df)
    ids = [set(p["customerID"]) for p in s.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(p) for p in s.values()) == len(df)
    rates = [(p["Churn"] == "Yes").mean() for p in s.values()]
    assert max(rates) - min(rates) < 0.01
    assert len(s["test"]) == 200 and len(s["val"]) == 200
