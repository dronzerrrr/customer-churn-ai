import pandas as pd
import pytest

from src.models.batch import parse_csv_bytes, run_batch, sanitize_cell
from src.models.risk import band_rules, band_summary, risk_band
from src.models.schema import MAX_BATCH_ROWS, consistency_problems
from src.models.validation import validate_batch
from tests.helpers import toy_service, valid_frame, valid_record


def frame(*records) -> pd.DataFrame:
    return pd.DataFrame(list(records))


def test_valid_rows_pass_unchanged():
    res = validate_batch(frame(valid_record(), valid_record(tenure=40)))
    assert res.errors == [] and len(res.valid) == 2 and res.n_rejected == 0


def test_missing_total_charges_is_estimated_not_rejected():
    rec = valid_record(tenure=10, MonthlyCharges=50.0)
    rec["TotalCharges"] = None
    res = validate_batch(frame(rec))
    assert res.valid.iloc[0]["TotalCharges"] == 500.0 and res.n_total_charges_estimated == 1
    no_col = frame(valid_record()).drop(columns=["TotalCharges"])
    assert len(validate_batch(no_col).valid) == 1


def test_missing_required_column_is_reported_clearly():
    res = validate_batch(frame(valid_record()).drop(columns=["Contract", "gender"]))
    assert set(res.missing_columns) == {"Contract", "gender"}
    assert res.valid.empty and res.n_rejected == 1


def test_unknown_category_rejects_only_that_row_and_names_the_allowed_values():
    res = validate_batch(frame(valid_record(), valid_record(Contract="Lifetime")))
    assert len(res.valid) == 1 and res.n_rejected == 1
    err = res.errors[0]
    assert err["row"] == 2 and err["field"] == "Contract" and "Month-to-month" in err["message"]


@pytest.mark.parametrize("override, field", [
    ({"tenure": 99}, "tenure"), ({"tenure": -1}, "tenure"), ({"tenure": 5.5}, "tenure"),
    ({"MonthlyCharges": "abc"}, "MonthlyCharges"), ({"MonthlyCharges": 0}, "MonthlyCharges"),
    ({"TotalCharges": -5}, "TotalCharges"), ({"SeniorCitizen": 2}, "SeniorCitizen"),
    ({"gender": None}, "gender"),
])
def test_bad_values_are_flagged(override, field):
    res = validate_batch(frame(valid_record(**override)))
    assert res.valid.empty and res.errors[0]["field"] == field


def test_impossible_combinations_are_rejected():
    no_net = valid_record(InternetService="No")  # add-ons still say Yes/No
    res = validate_batch(frame(no_net))
    assert res.errors[0]["field"] == "(combination)"
    no_phone = valid_record(PhoneService="No", MultipleLines="Yes")
    assert validate_batch(frame(no_phone)).valid.empty
    assert consistency_problems(valid_record()) == []


def test_whitespace_is_stripped():
    res = validate_batch(frame(valid_record(Contract=" One year ", gender="Female ")))
    assert len(res.valid) == 1 and res.valid.iloc[0]["Contract"] == "One year"


# ---- risk bands
def test_risk_band_edges():
    assert risk_band(0.34, 0.34) == "High" and risk_band(0.339, 0.34) == "Medium"
    assert risk_band(0.17, 0.34) == "Medium" and risk_band(0.169, 0.34) == "Low"
    assert "34%" in band_rules(0.34)


def test_band_summary_counts_add_up():
    import numpy as np
    y = np.array([1, 0, 0, 1, 0, 1])
    p = np.array([0.9, 0.05, 0.2, 0.5, 0.1, 0.3])
    s = band_summary(y, p, 0.34)
    assert s["customers"].sum() == len(y) and s["churned"].sum() == y.sum()


# ---- batch parsing and scoring
def test_parse_csv_rejects_bad_files_with_friendly_messages():
    for data, text in [(b"", "empty"), (b"   \n", "empty"), (b"a,b,c\n", "no data rows")]:
        with pytest.raises(ValueError, match=text):
            parse_csv_bytes(data)
    big = ("x\n" + "1\n" * (MAX_BATCH_ROWS + 1)).encode()
    with pytest.raises(ValueError, match="limit"):
        parse_csv_bytes(big)


def test_parse_csv_handles_bom_and_latin1():
    assert list(parse_csv_bytes(b"\xef\xbb\xbfa,b\n1,2\n").columns) == ["a", "b"]
    assert parse_csv_bytes("name\ncaf\xe9\n".encode("latin-1")).iloc[0, 0] == "caf\xe9"


def test_garbage_binary_gives_a_friendly_error_never_a_crash():
    garbage = bytes(range(1, 200)) * 3 + b"\n1,2\n"
    try:
        df = parse_csv_bytes(garbage)
    except ValueError as exc:                      # unreadable as CSV: friendly message
        assert "CSV" in str(exc) or "decoded" in str(exc)
    else:                                          # parsed, but the columns are meaningless
        assert run_batch(df, toy_service()).missing_columns


def test_run_batch_scores_valid_rows_reports_rest_and_sanitizes_ids():
    X, _ = valid_frame(5)
    X.insert(0, "customerID", ["A", "=SUM(1)", "B", "C", "D"])
    X.loc[2, "Contract"] = "Nonsense"
    result = run_batch(X, toy_service())
    assert result.n_received == 5 and result.n_scored == 4 and result.n_rejected == 1
    assert result.errors[0]["row"] == 3
    assert "'=SUM(1)" in result.scored["customerID"].tolist()
    assert result.scored["churn_probability"].between(0, 1).all()
    assert set(result.scored["risk_band"]) <= {"Low", "Medium", "High"}


def test_sanitize_cell():
    assert sanitize_cell("=1+1") == "'=1+1" and sanitize_cell("@x") == "'@x"
    assert sanitize_cell("ABC-1") == "ABC-1" and sanitize_cell(5) == 5
