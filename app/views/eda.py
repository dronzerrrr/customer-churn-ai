import pandas as pd
import plotly.express as px
import streamlit as st

from app.common import CHURN_COLORS, CHURN_COLOR, TENURE_ORDER, load_customers, pct
from src.evaluation.segments import segment_table

CATEGORY_CHOICES = [
    "Contract", "PaymentMethod", "InternetService", "tenure_group", "PaperlessBilling",
    "OnlineSecurity", "TechSupport", "SeniorCitizen", "Partner", "Dependents", "gender",
]


def rate_bar(data: pd.DataFrame, col: str, overall: float):
    t = segment_table(data, col, min_n=1)
    t["churn_pct"] = t["churn_rate"] * 100
    t[col] = t[col].astype(str)
    fig = px.bar(t, x=col, y="churn_pct", text=t["churn_pct"].map("{:.1f}%".format),
                 hover_data={"customers": True, "churn_pct": ":.1f"},
                 color_discrete_sequence=[CHURN_COLOR])
    fig.add_hline(y=overall * 100, line_dash="dash", line_color="grey",
                  annotation_text="Filtered overall", annotation_position="top left")
    fig.update_layout(yaxis_title="Churn rate (%)", xaxis_title=None, margin=dict(t=30, b=10))
    if col == "tenure_group":
        fig.update_xaxes(categoryorder="array", categoryarray=TENURE_ORDER)
    return fig


def render() -> None:
    st.title("Exploratory Data Analysis")
    df = load_customers()

    with st.expander("Filters", expanded=True):
        f1, f2 = st.columns(2)
        contracts = f1.multiselect("Contract", sorted(df["Contract"].unique()),
                                   default=sorted(df["Contract"].unique()))
        internet = f2.multiselect("Internet service", sorted(df["InternetService"].unique()),
                                  default=sorted(df["InternetService"].unique()))
        f3, f4 = st.columns(2)
        t_lo, t_hi = f3.slider("Tenure (months)", int(df["tenure"].min()),
                               int(df["tenure"].max()),
                               (int(df["tenure"].min()), int(df["tenure"].max())))
        m_lo, m_hi = f4.slider("Monthly charges", float(df["MonthlyCharges"].min()),
                               float(df["MonthlyCharges"].max()),
                               (float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())))

    data = df[df["Contract"].isin(contracts) & df["InternetService"].isin(internet)
              & df["tenure"].between(t_lo, t_hi) & df["MonthlyCharges"].between(m_lo, m_hi)]
    if data.empty:
        st.warning("No customers match these filters. Widen them to see charts.")
        return

    k1, k2, k3 = st.columns(3)
    k1.metric("Customers in view", f"{len(data):,}")
    k2.metric("Churn rate in view", pct(data["churn_flag"].mean()))
    k3.metric("Average monthly charge", f"{data['MonthlyCharges'].mean():.2f}")
    if len(data) < 100:
        st.caption("Fewer than 100 customers match: treat percentages as very uncertain.")
    overall = data["churn_flag"].mean()

    st.subheader("Churn distribution")
    counts = data["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "customers"]
    fig = px.pie(counts, names="Churn", values="customers", hole=0.55,
                 color="Churn", color_discrete_map=CHURN_COLORS)
    fig.update_layout(margin=dict(t=10, b=10), height=300)
    st.plotly_chart(fig, width="stretch", key="eda_churn_donut")

    st.subheader("Churn rate by segment")
    left, right = st.columns(2)
    with left:
        col = st.selectbox("Choose a category", CATEGORY_CHOICES, index=0)
        st.plotly_chart(rate_bar(data, col, overall), width="stretch", key="eda_rate_selected")
    with right:
        st.markdown("**Payment method**")
        st.plotly_chart(rate_bar(data, "PaymentMethod", overall), width="stretch",
                        key="eda_rate_payment")
    st.caption("Hover over a bar to see how many customers it contains. "
               "Small segments give noisy percentages.")

    st.subheader("Tenure and monthly charges")
    a, b = st.columns(2)
    with a:
        fig = px.histogram(data, x="tenure", color="Churn", nbins=36, barmode="overlay",
                           opacity=0.7, color_discrete_map=CHURN_COLORS)
        fig.update_layout(xaxis_title="Tenure (months)", yaxis_title="Customers",
                          margin=dict(t=30, b=10))
        st.plotly_chart(fig, width="stretch", key="eda_tenure_hist")
    with b:
        fig = px.box(data, x="Churn", y="MonthlyCharges", color="Churn",
                     color_discrete_map=CHURN_COLORS)
        fig.update_layout(showlegend=False, yaxis_title="Monthly charges", margin=dict(t=30, b=10))
        st.plotly_chart(fig, width="stretch", key="eda_charges_box")

    st.subheader("Contract and tenure together")
    rates = data.pivot_table(index="Contract", columns="tenure_group",
                             values="churn_flag", aggfunc="mean")
    sizes = data.pivot_table(index="Contract", columns="tenure_group",
                             values="churn_flag", aggfunc="size")
    rates = rates.where(sizes >= 30) * 100
    rates = rates.reindex(columns=[c for c in TENURE_ORDER if c in rates.columns])
    fig = px.imshow(rates, text_auto=".1f", color_continuous_scale="Reds", aspect="auto",
                    labels=dict(color="Churn %"))
    fig.update_layout(margin=dict(t=10, b=10), xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, width="stretch", key="eda_contract_tenure_heatmap")
    st.caption("Blank cells have fewer than 30 customers and are hidden.")
