import pytest
from fastapi.testclient import TestClient

from typing import get_args
from api.schemas import CustomerInput
from src.models.schema import ALLOWED_VALUES

from api.main import app as real_app
from api.main import create_app
from api.schemas import EXAMPLE_CUSTOMER
from src.config import MODEL_PATH
from tests.helpers import toy_service


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(toy_service)) as c:
        yield c


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True, "detail": None}
    info = client.get("/model-info").json()
    assert info["selected_model"] == "Toy Logistic Regression" and info["threshold"] == 0.4
    assert "risk_band_rules" in info and "estimates" in info["note"]


def test_predict_returns_probability_class_band_and_disclaimer(client):
    r = client.post("/predict", json=EXAMPLE_CUSTOMER)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1 and isinstance(body["predicted_churn"], bool)
    assert body["risk_band"] in {"Low", "Medium", "High"} and body["total_charges_estimated"]
    assert body["customerID"] == "EXAMPLE-0001" and body["reasons"] is None
    assert "estimates" in body["disclaimer"]


def test_predict_with_explanation(client):
    body = client.post("/predict?explain=true", json=EXAMPLE_CUSTOMER).json()
    assert len(body["reasons"]) == 6
    assert {"feature", "shap_log_odds", "effect"} <= set(body["reasons"][0])


def test_supplied_total_charges_is_used(client):
    body = client.post("/predict", json={**EXAMPLE_CUSTOMER, "TotalCharges": 480.0}).json()
    assert body["total_charges_estimated"] is False


@pytest.mark.parametrize("override, expect", [
    ({"Contract": "Lifetime"}, "Month-to-month"),          # unknown category
    ({"tenure": 500}, "less than or equal to 72"),         # out of range
    ({"MonthlyCharges": -3}, "greater than or equal"),
    ({"Contrac": "x"}, "Extra inputs"),                    # typo'd field name
    ({"InternetService": "No"}, "No internet service"),    # impossible combination
])
def test_invalid_input_gets_clear_422(client, override, expect):
    r = client.post("/predict", json={**EXAMPLE_CUSTOMER, **override})
    assert r.status_code == 422 and expect in r.text


def test_missing_field_is_422_naming_the_field(client):
    bad = {k: v for k, v in EXAMPLE_CUSTOMER.items() if k != "Contract"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422 and "Contract" in r.text


def test_batch_scores_good_rows_and_reports_bad_ones(client):
    rows = [EXAMPLE_CUSTOMER, {**EXAMPLE_CUSTOMER, "Contract": "Lifetime"},
            {**EXAMPLE_CUSTOMER, "tenure": 40, "customerID": "B"}]
    body = client.post("/predict-batch", json={"customers": rows}).json()
    assert (body["n_received"], body["n_scored"], body["n_rejected"]) == (3, 2, 1)
    assert [p["row"] for p in body["predictions"]] == [1, 3]
    assert body["errors"][0]["row"] == 2 and body["errors"][0]["field"] == "Contract"


def test_batch_with_missing_required_field_everywhere_is_422(client):
    rows = [{k: v for k, v in EXAMPLE_CUSTOMER.items() if k != "Contract"}]
    r = client.post("/predict-batch", json={"customers": rows})
    assert r.status_code == 422 and "Contract" in r.text


def test_batch_rejects_empty_list_and_wrong_shape(client):
    assert client.post("/predict-batch", json={"customers": []}).status_code == 422
    assert client.post("/predict-batch", json=[EXAMPLE_CUSTOMER]).status_code == 422


def test_api_stays_up_and_returns_503_when_model_is_missing():
    def broken():
        raise FileNotFoundError("model.joblib not found")
    with TestClient(create_app(broken)) as c:
        health = c.get("/health").json()
        assert health["status"] == "degraded" and "not found" in health["detail"]
        for call in (lambda: c.get("/model-info"), lambda: c.post("/predict", json=EXAMPLE_CUSTOMER)):
            r = call()
            assert r.status_code == 503 and "python -m src.models.train" in r.text


def test_openapi_docs_are_generated(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/model-info", "/predict", "/predict-batch"} <= set(paths)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Train the model first.")
def test_real_trained_model_end_to_end():
    with TestClient(real_app) as c:
        assert c.get("/health").json()["model_loaded"] is True
        low = c.post("/predict", json={**EXAMPLE_CUSTOMER, "Contract": "Two year", "tenure": 70,
                                       "MonthlyCharges": 25.0, "InternetService": "No",
                                       **{k: "No internet service" for k in (
                                           "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                                           "TechSupport", "StreamingTV", "StreamingMovies")}}).json()
        high = c.post("/predict", json=EXAMPLE_CUSTOMER).json()
        assert high["churn_probability"] > low["churn_probability"]
        assert high["risk_band"] == "High" and low["risk_band"] == "Low"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Train the model first.")
def test_production_path_matches_training_evaluation_path():
    """Same preprocessing at training and inference: batch results must equal the
    probabilities computed directly from the saved pipeline on the held-out test file."""
    import numpy as np
    import pandas as pd
    from src.config import PROCESSED_DIR, TARGET
    from src.models.batch import run_batch
    from src.models.service import ChurnService

    service = ChurnService.load()
    test = pd.read_csv(PROCESSED_DIR / "test.csv", dtype=str).drop(columns=[TARGET])
    result = run_batch(test, service)
    assert result.n_rejected == 0 and result.n_scored == len(test)
    direct = service.model.predict_proba(pd.read_csv(PROCESSED_DIR / "test.csv").drop(columns=[TARGET]))[:, 1]
    np.testing.assert_allclose(result.scored["churn_probability"].to_numpy(), direct, atol=1e-4)

def test_api_literal_types_match_the_single_source_of_truth():
    """The explicit Literal types in api/schemas.py must never drift from schema.py."""
    for column, allowed in ALLOWED_VALUES.items():
        declared = get_args(CustomerInput.model_fields[column].annotation)
        assert list(declared) == allowed, column
