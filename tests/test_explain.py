import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.evaluation.explain import ChurnExplainer, original_feature_name
from src.features.preprocess import build_preprocessor
from tests.test_features import make_df


def fit_model(estimator):
    df = make_df(400)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])
    # give the toy data a learnable signal
    X["MonthlyCharges"] = X["MonthlyCharges"] + 30 * y.to_numpy()
    X["Contract"] = np.where(y == 1, "Month-to-month", "Two year")
    pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)]).fit(X, y)
    return pipe, X


@pytest.mark.parametrize("name", ["xgb", "logreg"])
def test_shap_contributions_add_up_to_model_log_odds(name):
    est = (XGBClassifier(n_estimators=15, max_depth=2, random_state=0)
           if name == "xgb" else LogisticRegression(max_iter=500))
    model, X = fit_model(est)
    exp = ChurnExplainer(model, background_df=X.head(100))
    row = X.iloc[[3]]
    result = exp.explain_customer(row, top_k=50)
    grouped = exp.grouped_shap(row).iloc[0]
    total = exp._base() + grouped.sum()
    p = model.predict_proba(row)[0, 1]
    assert total == pytest.approx(np.log(p / (1 - p)), abs=1e-3)
    assert 0 <= result["probability"] <= 1


def test_original_feature_name_groups_one_hot_columns():
    assert original_feature_name("cat__Contract_One year") == "Contract"
    assert original_feature_name("cat__tenure_band_0-12") == "tenure_band"
    assert original_feature_name("cat__InternetService_Fiber optic") == "InternetService"
    assert original_feature_name("num__tenure") == "tenure"
    assert original_feature_name("num__missingindicator_TotalCharges") == "TotalCharges"


def test_explain_customer_requires_single_row():
    model, X = fit_model(LogisticRegression(max_iter=500))
    exp = ChurnExplainer(model, background_df=X.head(50))
    with pytest.raises(ValueError):
        exp.explain_customer(X.head(2))
