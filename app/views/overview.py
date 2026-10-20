import streamlit as st

from app.common import load_customers, load_metadata, load_metrics, pct
from src.evaluation.segments import key_observations


def render() -> None:
    st.title("Executive Overview")
    st.caption("Customer churn analysis on the IBM Telco sample dataset. "
               "Every figure below is computed from the data and the trained model.")

    df = load_customers()
    metrics = load_metrics()
    meta = load_metadata()
    test = metrics["test_metrics"]

    st.subheader("The customer base")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers analysed", f"{len(df):,}")
    c2.metric("Actual churn rate", pct(df["churn_flag"].mean()))
    c3.metric("Customers who churned", f"{int(df['churn_flag'].sum()):,}")
    c4.metric("Monthly charges of churned customers",
              f"{df.loc[df['churn_flag'] == 1, 'MonthlyCharges'].sum():,.0f}",
              help="Observed in this dataset: the sum of monthly charges of customers who left. "
                   "It is a description of past loss, not a forecast of savings.")

    st.subheader("The prediction model")
    st.write(f"Selected model: **{metrics['selected_model']}**, "
             f"decision threshold **{meta['threshold']:.2f}** "
             f"(chosen on validation data to favour catching churners; recall counts "
             f"{meta['threshold_beta']:.0f}x as much as precision).")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recall (test)", pct(test["recall"]),
              help="Share of customers who actually churned that the model flagged.")
    m2.metric("Precision (test)", pct(test["precision"]),
              help="Share of flagged customers who actually churned.")
    m3.metric("ROC-AUC (test)", f"{test['roc_auc']:.3f}",
              help="Chance the model ranks a random churner above a random non-churner.")
    m4.metric("PR-AUC (test)", f"{test['pr_auc']:.3f}",
              help="Area under the precision-recall curve; the no-skill level equals the churn rate.")

    cm = test["confusion_matrix"]
    base = metrics["dummy_baseline_test"]
    st.info(
        f"On {sum(cm.values()):,} held-out test customers the model caught {cm['tp']} of "
        f"{cm['tp'] + cm['fn']} churners and raised {cm['fp']} false alarms. "
        f"A do-nothing baseline scores {pct(base['accuracy'])} accuracy but catches none, "
        f"which is why accuracy alone is misleading here."
    )

    st.subheader("Key observations from the data")
    for line in key_observations(df):
        st.markdown(f"- {line}")

    st.warning(
        "**Read with care.** These are observed associations in a sample dataset, not proof of "
        "what causes customers to leave. Predicted probabilities are estimates, not certainties."
    )
