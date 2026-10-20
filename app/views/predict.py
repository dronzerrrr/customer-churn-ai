import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from api.schemas import CustomerInput
from app.common import CHURN_COLOR, STAY_COLOR, load_service, validation_band_table
from src.models.risk import band_rules
from src.models.schema import ADDON_COLUMNS, ALLOWED_VALUES, MONTHLY_BOUNDS, TENURE_BOUNDS


def pick(col: str, *, options=None, index: int = 0, suffix: str = "", label: str | None = None):
    return st.selectbox(label or col, options or ALLOWED_VALUES[col], index=index,
                        key=f"in_{col}{suffix}")


def render() -> None:
    st.title("Customer Risk Prediction")
    st.caption("Describe a customer to get an estimated churn probability and the main factors "
               "behind it. This is a decision-support estimate, not a certainty.")
    service = load_service()
    form, result = st.columns([3, 2], gap="large")

    with form:
        st.subheader("Customer profile")
        a, b = st.columns(2)
        with a:
            gender = pick("gender")
            partner = pick("Partner")
            tenure = st.number_input("Tenure (months)", TENURE_BOUNDS[0], TENURE_BOUNDS[1],
                                     12, 1, key="in_tenure")
        with b:
            senior = st.selectbox("SeniorCitizen", ["No", "Yes"], key="in_SeniorCitizen")
            dependents = pick("Dependents")
            monthly = st.number_input("Monthly charges", MONTHLY_BOUNDS[0], MONTHLY_BOUNDS[1],
                                      70.0, 0.5, key="in_MonthlyCharges")

        st.subheader("Services")
        c1, c2 = st.columns(2)
        with c1:
            phone = pick("PhoneService", index=1)
            lines_opts = ["No phone service"] if phone == "No" else ["No", "Yes"]
            lines = pick("MultipleLines", options=lines_opts,
                         suffix="_nophone" if phone == "No" else "_phone")
        with c2:
            internet = pick("InternetService")
        no_net = internet == "No"
        addon_opts = ["No internet service"] if no_net else ["No", "Yes"]
        addons = {}
        cols = st.columns(3)
        for i, col in enumerate(ADDON_COLUMNS):
            with cols[i % 3]:
                addons[col] = pick(col, options=addon_opts, suffix="_nonet" if no_net else "_net")
        if no_net or phone == "No":
            st.caption("Options are fixed automatically where a service does not apply.")

        st.subheader("Contract and billing")
        d1, d2 = st.columns(2)
        with d1:
            contract = pick("Contract")
            paperless = pick("PaperlessBilling", index=1)
        with d2:
            payment = pick("PaymentMethod", index=2)

    values = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
        "Dependents": dependents, "tenure": int(tenure), "PhoneService": phone,
        "MultipleLines": lines, "InternetService": internet, **addons, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": float(monthly),
    }

    with result:
        st.subheader("Prediction")
        try:
            customer = CustomerInput(**values)
        except ValidationError as exc:
            st.error("These inputs are not a valid combination: "
                     + "; ".join(e["msg"] for e in exc.errors()))
            return
        X, estimated = customer.to_frame()
        scored = service.score(X).iloc[0]
        proba, band = float(scored["churn_probability"]), scored["risk_band"]

        st.metric("Estimated churn probability", f"{proba:.1%}")
        message = (f"**{band} risk.** " + (
            f"Above the {service.threshold:.0%} decision threshold, so this customer would be "
            "flagged for retention outreach." if scored["predicted_churn"] else
            f"Below the {service.threshold:.0%} decision threshold, so this customer would not "
            "be flagged."))
        {"High": st.error, "Medium": st.warning, "Low": st.success}[band](message)
        if estimated:
            st.caption(f"Total charges are estimated as tenure x monthly charges "
                       f"({customer.to_frame()[0]['TotalCharges'].iloc[0]:,.2f}).")

        reasons = service.explain(X, top_k=8)["reasons"]
        st.markdown("**Main factors in this estimate**")
        frame = pd.DataFrame(reasons).iloc[::-1]
        labels = [f"{r.feature} = {r.value}" for r in frame.itertuples()]
        fig = go.Figure(go.Bar(
            x=frame["shap_log_odds"], y=labels, orientation="h",
            marker_color=[CHURN_COLOR if v > 0 else STAY_COLOR for v in frame["shap_log_odds"]]))
        fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10),
                          xaxis_title="Pushes predicted risk down  <  0  >  up (log-odds)")
        st.plotly_chart(fig, width="stretch", key="predict_reasons")
        st.caption("Red raises and blue lowers this customer's predicted risk. These are "
                   "associations the model learned, not proven causes of churn.")

    st.divider()
    st.subheader("How to read the risk bands")
    st.write(band_rules(service.threshold))
    bands = validation_band_table().rename(columns={
        "band": "Band", "customers": "Validation customers",
        "churned": "Actually churned", "churn_rate": "Observed churn rate"})
    st.dataframe(bands.style.format({"Observed churn rate": "{:.1%}"}).hide(axis="index"),
                 width="stretch")
    st.caption("Observed churn in each band on held-out validation customers. Bands are a "
               "presentation convention built on the decision threshold; the probabilities "
               "themselves have not been calibration-checked.")
