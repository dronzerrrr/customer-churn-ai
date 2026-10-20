import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import best_threshold, compute_metrics
from src.features.preprocess import build_preprocessor
from src.models.predict import predict_df
from tests.test_features import make_df


def test_compute_metrics_known_values():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    m = compute_metrics(y, p, threshold=0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["accuracy"] == 0.5


def test_best_threshold_favours_recall_when_beta_is_large():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.25).astype(int)
    p = np.clip(0.3 * y + rng.normal(0.3, 0.15, 2000), 0, 1)
    assert best_threshold(y, p, beta=2.0) <= best_threshold(y, p, beta=0.5)


def fitted_pipeline():
    df = make_df(300)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("model", LogisticRegression(max_iter=500))]).fit(X, y)
    return pipe, X


def test_saved_pipeline_gives_identical_predictions(tmp_path):
    pipe, X = fitted_pipeline()
    path = tmp_path / "m.joblib"
    joblib.dump(pipe, path)
    reloaded = joblib.load(path)
    np.testing.assert_allclose(pipe.predict_proba(X), reloaded.predict_proba(X))


def test_predict_df_outputs_and_unknown_category():
    pipe, X = fitted_pipeline()
    new = X.head(3).copy()
    new["PaymentMethod"] = "Crypto"
    out = predict_df(new, model=pipe, threshold=0.5)
    assert {"churn_probability", "predicted_churn"} <= set(out.columns)
    assert out["churn_probability"].between(0, 1).all()


def test_predict_df_missing_columns_raises_clear_error():
    pipe, X = fitted_pipeline()
    with pytest.raises(ValueError, match="Missing required columns"):
        predict_df(X.drop(columns=["Contract"]), model=pipe, threshold=0.5)
