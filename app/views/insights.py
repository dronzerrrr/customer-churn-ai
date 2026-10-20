import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.common import (
    CHURN_COLOR, STAY_COLOR, load_customers, load_service, split_predictions,
)
from src.config import PROCESSED_DIR
from src.evaluation.impact import ImpactAssumptions, estimate_impact, sensitivity
from src.evaluation.insights import build_recommendations, model_targeting_summary
from src.evaluation.segments import segment_scan

SORTS = {"Most excess churners": "excess_churners", "Highest churn rate": "churn_rate",
         "Highest lift": "lift"}
MODEL_GROUP = "Customers the model flags (held-out test set)"


@st.cache_data(show_spinner="Scanning customer segments...")
def scan(min_n: int, pairs: bool) -> pd.DataFrame:
    return segment_scan(load_customers(), min_n=min_n, pairs=pairs)


@st.cache_data(show_spinner=False)
def recommendations():
    return build_recommendations(load_customers())


@st.cache_data(show_spinner=False)
def targeting() -> dict:
    y, proba = split_predictions("test")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    return model_targeting_summary(test_df, y, proba, load_service().threshold)


def segments_tab() -> None:
    overall = load_customers()["churn_flag"].mean()
    st.info(
        f"Overall churn is {overall:.1%}. **Lift** is a segment's churn rate divided by that. "
        "**Excess churners** is how many more customers churned than the overall rate would "
        "predict for a group of that size. Segments overlap, so do not add them together. "
        "Because many segments are scanned, a few will look extreme by chance; the 95% "
        "interval shows how uncertain each rate is.")
    c1, c2, c3, c4 = st.columns(4)
    min_n = c1.slider("Minimum customers", 50, 1000, 200, 50, key="ins_min_n")
    sort_label = c2.selectbox("Rank by", list(SORTS), key="ins_sort")
    pairs = c3.checkbox("Include two-factor segments", True, key="ins_pairs")
    only_above = c4.checkbox("Only above-average churn", True, key="ins_above")

    table = scan(min_n, pairs)
    if only_above:
        table = table[table["churn_rate"] > overall]
    table = table.sort_values(SORTS[sort_label], ascending=False).head(15)
    if table.empty:
        st.warning("No segments match. Lower the minimum number of customers.")
        return

    show = table.assign(**{"95% interval": [f"{lo:.1%} to {hi:.1%}" for lo, hi in
                                            zip(table["ci_low"], table["ci_high"], strict=True)]})
    show = show[["segment", "customers", "churned", "churn_rate", "95% interval", "lift",
                 "share_of_churners", "excess_churners"]].rename(columns={
        "segment": "Segment", "customers": "Customers", "churned": "Churned",
        "churn_rate": "Churn rate", "lift": "Lift", "share_of_churners": "Share of all churners",
        "excess_churners": "Excess churners"})
    st.dataframe(show.style.format({"Churn rate": "{:.1%}", "Lift": "{:.2f}x",
                                    "Share of all churners": "{:.1%}",
                                    "Excess churners": "{:.0f}"}).hide(axis="index"),
                 width="stretch")

    top = table.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["churn_rate"] * 100, y=top["segment"], orientation="h", marker_color=CHURN_COLOR,
        error_x=dict(type="data", symmetric=False,
                     array=(top["ci_high"] - top["churn_rate"]) * 100,
                     arrayminus=(top["churn_rate"] - top["ci_low"]) * 100)))
    fig.add_vline(x=overall * 100, line_dash="dash", line_color="grey",
                  annotation_text="Overall", annotation_position="top")
    fig.update_layout(height=420, margin=dict(t=30, b=10), xaxis_title="Churn rate (%) with 95% interval",
                      yaxis=dict(automargin=True))
    st.plotly_chart(fig, width="stretch", key="ins_segments_chart")


def playbook_tab() -> None:
    st.warning(
        "Everything here is **observational**. The data shows where churn is higher, not that "
        "any action would reduce it. Each idea ends with the experiment that would find out.")
    for i, r in enumerate(recommendations()):
        with st.expander(f"{i + 1}. {r.title}", expanded=(i == 0)):
            st.markdown(f"**Idea:** {r.action}")
            st.markdown(f"**What the data shows:** {r.headline}")
            st.markdown(f"**Does it hold within subgroups?** {r.robustness}")
            st.caption(r.evidence_title)
            fmt = {c: "{:.1%}" for c in r.evidence.columns if c.endswith(": churn")}
            st.dataframe(r.evidence.style.format(fmt, na_rep="-").hide(axis="index"),
                         width="stretch")
            st.markdown(f"**What this does not show:** {r.caveat}")
            st.markdown(f"**How to find out:** {r.test_idea}")

    st.subheader("Where the model helps: choosing whom to contact")
    t = targeting()
    c = st.columns(4)
    c[0].metric("Customers flagged", f"{t['customers']:,}")
    c[1].metric("Churn rate among flagged", f"{t['churn_rate']:.1%}")
    c[2].metric("Churn rate among not flagged", f"{t['unflagged_churn_rate']:.1%}")
    c[3].metric("Contacts per churner reached",
                f"{t['offers_per_churner_flagged']:.1f} vs {t['offers_per_churner_all']:.1f}",
                help="Flagged customers only, versus contacting everyone.")
    st.caption(
        f"Measured on the held-out test set: the flagged group contains {t['churners_caught']} of "
        f"{t['total_churners']} churners. Contacting only flagged customers reaches a churner "
        f"about every {t['offers_per_churner_flagged']:.1f} contacts, against every "
        f"{t['offers_per_churner_all']:.1f} when contacting everyone. Reaching a churner is not "
        "the same as keeping them.")


def impact_tab() -> None:
    st.error(
        "**This is a what-if calculator, not a forecast.** The data cannot tell us how much a "
        "retention offer would reduce churn. The values below are placeholders: replace them with "
        "your own figures, and treat the result as a way to see which assumptions matter.")
    recs = recommendations()
    groups = {r.target_label: (r.target_customers, r.target_churn_rate, r.target_monthly_charge)
              for r in recs}
    t = targeting()
    groups[MODEL_GROUP] = (t["customers"], t["churn_rate"], t["monthly_charge"])
    label = st.selectbox("Who would you contact?", list(groups), key="ins_group")
    n, rate, charge = groups[label]
    idx = list(groups).index(label)
    f = st.columns(3)
    f[0].metric("Customers in this group", f"{n:,}")
    f[1].metric("Observed churn rate", f"{rate:.1%}")
    f[2].metric("Average monthly charge", f"{charge:.2f}")

    a, b = st.columns(2)
    with a:
        targeted = st.number_input("Customers you would contact", 1, 1_000_000, int(n), 50,
                                   key=f"ins_targeted_{idx}")
        cost = st.number_input("Cost per contacted customer", 0.0, 10_000.0, 10.0, 1.0,
                               key="ins_cost")
        reduction = st.slider("ASSUMED reduction in churn among contacted customers",
                              0, 60, 10, 1, format="%d%%", key="ins_reduction",
                              help="Placeholder. Only a controlled test can measure this.")
    with b:
        margin = st.slider("Profit margin on monthly charges", 0, 100, 30, 5, format="%d%%",
                           key="ins_margin")
        months = st.number_input("Months a retained customer is assumed to stay", 1, 120, 12, 1,
                                 key="ins_months")

    assumptions = ImpactAssumptions(
        targeted=int(targeted), baseline_churn=float(rate), relative_reduction=reduction / 100,
        cost_per_offer=float(cost), monthly_charge=float(charge), margin=margin / 100,
        months_retained=float(months))
    res = estimate_impact(assumptions)

    m = st.columns(4)
    m[0].metric("Churners expected in this group", f"{res['expected_churners']:,.0f}")
    m[1].metric("Customers retained (assumed)", f"{res['customers_retained']:,.0f}")
    m[2].metric("Campaign cost", f"{res['campaign_cost']:,.0f}")
    m[3].metric("Net value (assumed)", f"{res['net_value']:,.0f}")

    be = res["break_even_reduction"]
    if be is None:
        st.caption("Break-even cannot be computed for these inputs.")
    elif be > 1:
        st.warning("Under these assumptions the campaign cannot break even, even if every "
                   "would-be churner were retained.")
    else:
        st.write(f"Under these assumptions the campaign breaks even if it cuts churn among "
                 f"contacted customers by about **{be:.0%}**. If the true effect is smaller, "
                 "it loses money.")

    sens = sensitivity(assumptions, [x / 100 for x in range(0, 61, 5)])
    fig = go.Figure(go.Scatter(x=sens["relative_reduction"] * 100, y=sens["net_value"],
                               mode="lines+markers", line=dict(color=STAY_COLOR)))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(height=320, margin=dict(t=10, b=10),
                      xaxis_title="Assumed reduction in churn (%)", yaxis_title="Net value")
    st.plotly_chart(fig, width="stretch", key="ins_sensitivity")
    st.caption("Net value = retained customers x monthly charge x margin x months retained, "
               "minus campaign cost. Currency units are whatever your charges are in.")


def render() -> None:
    st.title("Business Insights")
    st.caption("From observed patterns to testable retention ideas.")
    t1, t2, t3 = st.tabs(["Where churn concentrates", "Retention playbook", "Impact scenario"])
    with t1:
        segments_tab()
    with t2:
        playbook_tab()
    with t3:
        impact_tab()
