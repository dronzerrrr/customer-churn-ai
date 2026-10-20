import numpy as np
import pandas as pd
import pytest

from src.config import RAW_CSV
from src.evaluation.impact import ImpactAssumptions, estimate_impact, sensitivity
from src.evaluation.insights import (
    build_recommendations, compare_within, model_targeting_summary, robustness_sentence)
from src.evaluation.segments import segment_scan, wilson_interval


def assumptions(**kw) -> ImpactAssumptions:
    base = dict(targeted=1000, baseline_churn=0.4, relative_reduction=0.1, cost_per_offer=10.0,
                monthly_charge=70.0, margin=0.3, months_retained=12)
    return ImpactAssumptions(**{**base, **kw})


# ---- impact calculator
def test_impact_arithmetic_matches_hand_calculation():
    r = estimate_impact(assumptions())
    assert r["expected_churners"] == pytest.approx(400)
    assert r["customers_retained"] == pytest.approx(40)
    assert r["value_retained"] == pytest.approx(40 * 70 * 0.3 * 12)      # 10,080
    assert r["campaign_cost"] == pytest.approx(10_000)
    assert r["net_value"] == pytest.approx(80)
    assert r["break_even_reduction"] == pytest.approx(10_000 / (400 * 70 * 0.3 * 12))
    assert r["offers_per_churner_reached"] == pytest.approx(2.5)


def test_break_even_is_where_net_value_crosses_zero():
    be = estimate_impact(assumptions())["break_even_reduction"]
    assert estimate_impact(assumptions(relative_reduction=be))["net_value"] == pytest.approx(0, abs=1e-6)


def test_zero_effect_costs_money_and_zero_churn_has_no_break_even():
    assert estimate_impact(assumptions(relative_reduction=0))["net_value"] == pytest.approx(-10_000)
    assert estimate_impact(assumptions(baseline_churn=0))["break_even_reduction"] is None


@pytest.mark.parametrize("kw", [{"baseline_churn": 1.5}, {"relative_reduction": -0.1},
                                {"margin": 2}, {"cost_per_offer": -1}, {"targeted": -5}])
def test_impact_rejects_impossible_inputs(kw):
    with pytest.raises(ValueError):
        assumptions(**kw)


def test_sensitivity_net_value_rises_with_effect():
    s = sensitivity(assumptions(), [0.0, 0.1, 0.2, 0.4])
    assert s["net_value"].is_monotonic_increasing and len(s) == 4


# ---- statistics
def test_wilson_interval_known_values():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert wilson_interval(0, 10)[0] == 0.0
    assert np.isnan(wilson_interval(0, 0)[0])
    narrow = wilson_interval(500, 1000)
    assert (narrow[1] - narrow[0]) < (hi - lo)          # more data, tighter interval


def segment_df(n=600, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.choice(["x", "y"], n), "b": rng.choice(["p", "q"], n)})
    df["churn_flag"] = (rng.random(n) < np.where(df["a"] == "x", 0.5, 0.1)).astype(int)
    return df


def test_segment_scan_respects_min_n_and_reports_consistent_numbers():
    df = segment_df()
    scan = segment_scan(df, columns=["a", "b"], min_n=100)
    assert (scan["customers"] >= 100).all()
    row = scan[scan["segment"] == "a = x"].iloc[0]
    sub = df[df["a"] == "x"]
    assert row["customers"] == len(sub) and row["churned"] == sub["churn_flag"].sum()
    assert row["ci_low"] <= row["churn_rate"] <= row["ci_high"]
    assert row["excess_churners"] == pytest.approx(row["churned"] - len(sub) * df["churn_flag"].mean())
    assert segment_scan(df, ["a", "b"], min_n=10_000).empty


def test_segment_scan_does_not_list_identical_customer_sets_twice():
    df = segment_df()
    df["c"] = df["a"]                      # c is an exact copy of a
    scan = segment_scan(df, columns=["a", "c"], min_n=50)
    # c = x, c = y and the pairs describe the same customers as a = x / a = y
    assert sorted(scan["segment"]) == ["a = x", "a = y"]


# ---- comparisons within subgroups
def test_compare_within_marks_small_cells_and_sentence_reflects_it():
    df = pd.DataFrame({"churn_flag": [1] * 150 + [0] * 150 + [1] * 5 + [0] * 5,
                       "g": ["big"] * 300 + ["tiny"] * 10})
    a = pd.Series([i % 2 == 0 for i in range(len(df))], index=df.index)
    table = compare_within(df, a, ~a, df["g"], min_cell=100)
    assert table.set_index("group").loc["big", "comparable"]
    assert not table.set_index("group").loc["tiny", "comparable"]
    assert "1 comparable" in robustness_sentence(table, "Group A", "group B", "size", 100)
    none = table.assign(comparable=False)
    assert "cannot be checked" in robustness_sentence(none, "A", "B", "size", 100)


def test_model_targeting_summary_known_numbers():
    y = np.array([1, 1, 0, 0, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.1, 0.1, 0.1])
    test = pd.DataFrame({"MonthlyCharges": [10, 20, 30, 40, 50, 60, 70, 80]})
    s = model_targeting_summary(test, y, p, threshold=0.5)
    assert s["customers"] == 3 and s["churners_caught"] == 2 and s["total_churners"] == 3
    assert s["churn_rate"] == pytest.approx(2 / 3) and s["base_rate"] == pytest.approx(3 / 8)
    assert s["offers_per_churner_flagged"] < s["offers_per_churner_all"]
    assert s["monthly_charge"] == pytest.approx(20.0)


@pytest.mark.skipif(not RAW_CSV.exists(), reason="Download the dataset first.")
def test_recommendation_numbers_match_direct_computation_on_real_data():
    from src.data.clean import clean_for_analysis
    from src.data.load_data import load_raw_data
    df = clean_for_analysis(load_raw_data())
    recs = {r.key: r for r in build_recommendations(df)}
    assert set(recs) == {"early_tenure", "contract", "payment", "fiber", "support"}
    m2m = df[df["Contract"] == "Month-to-month"]
    assert recs["contract"].target_customers == len(m2m)
    assert recs["contract"].target_churn_rate == pytest.approx(m2m["churn_flag"].mean())
    assert f"{m2m['churn_flag'].mean():.1%}" in recs["contract"].headline
    early = df[df["tenure"] <= 12]
    assert recs["early_tenure"].target_customers == len(early)
    for r in recs.values():                           # every idea names its limits and a test
        assert r.caveat and r.test_idea and not r.evidence.empty
