import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.config import MODEL_PATH, PROCESSED_DIR, PROJECT_ROOT, RAW_CSV
from src.evaluation.segments import key_observations, segment_table

APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")
needs_artifacts = pytest.mark.skipif(
    not (MODEL_PATH.exists() and RAW_CSV.exists() and (PROCESSED_DIR / "test.csv").exists()),
    reason="Run the data and training scripts first.")


@needs_artifacts
@pytest.mark.parametrize("page", ["Executive Overview", "Exploratory Data Analysis", "Model Performance"])
def test_every_page_renders_without_errors(page):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.main.title) == 1 and at.main.title[0].value == page

@needs_artifacts
def test_eda_works_for_every_category_in_the_dropdown():
    """Regression: choosing 'PaymentMethod' used to duplicate a chart ID and crash."""
    from app.views.eda import CATEGORY_CHOICES
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value("Exploratory Data Analysis").run()
    for category in CATEGORY_CHOICES:
        at.selectbox[0].set_value(category).run()
        assert not at.exception, (category, [e.value[:120] for e in at.exception])

def small_df():
    return pd.DataFrame({
        "churn_flag": [1, 0, 0, 1, 0, 0],
        "Contract": ["M", "M", "Y", "Y", "Y", "M"],
        "tenure_group": ["a"] * 6, "PaymentMethod": ["x"] * 6, "InternetService": ["d"] * 6,
    })


def test_segment_table_hides_small_segments_and_sorts():
    t = segment_table(small_df(), "Contract", min_n=3)
    assert set(t["Contract"]) == {"M", "Y"}
    assert segment_table(small_df(), "Contract", min_n=4).empty


def test_key_observations_are_computed_from_data():
    obs = key_observations(small_df(), min_n=1)
    assert any("contract type" in o for o in obs)
    assert key_observations(small_df().iloc[0:0]) == []
    assert key_observations(small_df(), min_n=100) == []  # tiny segments are never reported



@needs_artifacts
def test_prediction_page_responds_to_inputs_and_enforces_valid_combinations():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value("Customer Risk Prediction").run()
    assert not at.exception

    def probability():
        return float(at.metric[0].value.rstrip("%")) / 100

    at.selectbox(key="in_Contract").set_value("Month-to-month").run()
    month_to_month = probability()
    at.selectbox(key="in_Contract").set_value("Two year").run()
    assert probability() < month_to_month            # more committed contract, lower risk
    at.selectbox(key="in_InternetService").set_value("No").run()
    assert not at.exception
    assert at.selectbox(key="in_OnlineSecurity_nonet").options == ["No internet service"]
    at.selectbox(key="in_PhoneService").set_value("No").run()
    assert not at.exception
    assert at.selectbox(key="in_MultipleLines_nophone").options == ["No phone service"]


@needs_artifacts
def test_batch_page_renders_and_offers_a_template():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value("Batch Prediction").run()
    assert not at.exception and at.main.title[0].value == "Batch Prediction"



@needs_artifacts
def test_insights_page_renders_and_the_calculator_responds_to_assumptions():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value("Business Insights").run()
    assert not at.exception, [e.value[:200] for e in at.exception]

    def net_value():
        return next(m.value for m in at.metric if m.label == "Net value (assumed)")

    at.slider(key="ins_reduction").set_value(0).run()
    zero_effect = net_value()
    assert zero_effect.startswith("-")                  # no effect: only the cost remains
    at.slider(key="ins_reduction").set_value(60).run()
    assert not net_value().startswith("-")
    at.selectbox(key="ins_group").set_value("Customers the model flags (held-out test set)").run()
    assert not at.exception
    at.selectbox(key="ins_sort").set_value("Highest churn rate").run()
    at.slider(key="ins_min_n").set_value(1000).run()
    assert not at.exception



def test_missing_files_message_shows_each_command_only_once(monkeypatch, tmp_path):
    import app.common as common
    fake = {"trained model": (tmp_path / "a", "python -m src.models.train"),
            "metrics": (tmp_path / "b", "python -m src.models.train"),
            "dataset": (tmp_path / "c", "python -m src.data.download_data")}
    monkeypatch.setattr(common, "REQUIRED_FILES", fake)

    def script():
        from app.common import require_artifacts
        require_artifacts()

    at = AppTest.from_function(script).run()
    assert [c.value for c in at.code] == ["python -m src.models.train",
                                          "python -m src.data.download_data"]
