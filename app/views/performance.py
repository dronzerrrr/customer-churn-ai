import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from app.common import (
    CHURN_COLOR, STAY_COLOR, load_metadata, load_metrics, pct, split_predictions,
)
from src.evaluation.metrics import compute_metrics

METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def render() -> None:
    st.title("Model Performance")
    metrics = load_metrics()
    meta = load_metadata()
    test = metrics["test_metrics"]
    cm = test["confusion_matrix"]

    st.subheader("Model comparison (validation data)")
    comp = pd.DataFrame(metrics["validation_comparison"]).rename(columns={"model": "Model"})
    styled = comp.style.format({c: "{:.3f}" for c in METRIC_COLS}).hide(axis="index")
    st.dataframe(styled, width="stretch")
    st.caption("Precision, recall, F1 and accuracy are at a 0.5 threshold so models are compared "
               "fairly; ROC-AUC and PR-AUC do not depend on a threshold. The model was selected "
               f"by validation PR-AUC. Top models are within a hair of each other, so the "
               f"choice of **{metrics['selected_model']}** is not a decisive win.")

    st.subheader(f"Final test results: {metrics['selected_model']} at threshold {test['threshold']:.2f}")
    c = st.columns(6)
    for col, (label, key) in zip(c, [("Accuracy", "accuracy"), ("Precision", "precision"),
                                     ("Recall", "recall"), ("F1", "f1"),
                                     ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc")],
                                    strict=True):
        col.metric(label, f"{test[key]:.3f}")
    st.caption("The test set was used exactly once, after the model and threshold were chosen.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix (test)**")
        z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        fig = go.Figure(go.Heatmap(
            z=z, x=["Predicted: stayed", "Predicted: churned"],
            y=["Actual: stayed", "Actual: churned"], text=z, texttemplate="%{text}",
            colorscale="Blues", showscale=False))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(margin=dict(t=10, b=10), height=320)
        st.plotly_chart(fig, width="stretch", key="perf_confusion_matrix")
    with right:
        st.markdown("**What this means**")
        st.write(
            f"- Of {cm['tp'] + cm['fn']} customers who churned, the model flagged "
            f"**{cm['tp']}** and missed **{cm['fn']}**.\n"
            f"- It also flagged **{cm['fp']}** customers who stayed (false alarms).\n"
            f"- So about **{pct(test['precision'])}** of flagged customers actually churn."
        )
        st.write(
            f"Baseline accuracy is {pct(metrics['dummy_baseline_test']['accuracy'])} "
            f"from predicting 'stays' for everyone, yet that catches zero churners. "
            f"Our accuracy of {pct(test['accuracy'])} is lower on purpose: the threshold trades "
            f"false alarms for catching more churners."
        )

    y_test, p_test = split_predictions("test")
    fpr, tpr, _ = roc_curve(y_test, p_test)
    prec, rec, _ = precision_recall_curve(y_test, p_test)
    op_fpr = cm["fp"] / (cm["fp"] + cm["tn"])
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**ROC curve (test)**")
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=metrics["selected_model"],
                        line=dict(color=STAY_COLOR))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="No skill",
                        line=dict(color="grey", dash="dash"))
        fig.add_scatter(x=[op_fpr], y=[test["recall"]], mode="markers", name="Chosen threshold",
                        marker=dict(color=CHURN_COLOR, size=11))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                          margin=dict(t=10, b=10), height=340, legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch", key="perf_roc_curve")
    with r2:
        st.markdown("**Precision-recall curve (test)**")
        fig = go.Figure()
        fig.add_scatter(x=rec, y=prec, mode="lines", name=metrics["selected_model"],
                        line=dict(color=STAY_COLOR))
        fig.add_hline(y=float(y_test.mean()), line_dash="dash", line_color="grey",
                      annotation_text="No skill (churn rate)")
        fig.add_scatter(x=[test["recall"]], y=[test["precision"]], mode="markers",
                        name="Chosen threshold", marker=dict(color=CHURN_COLOR, size=11))
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision",
                          margin=dict(t=10, b=10), height=340, legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch", key="perf_pr_curve")

    st.subheader("Threshold explorer (validation data)")
    st.write("Move the threshold to see the trade-off between catching churners and "
             "raising false alarms. This uses the validation set, where the saved threshold "
             "was chosen; the test set stays untouched.")
    y_val, p_val = split_predictions("val")
    t = st.slider("Flag customers whose churn probability is at least",
                  0.05, 0.95, float(meta["threshold"]), 0.01)
    m = compute_metrics(y_val, p_val, t)
    vcm = m["confusion_matrix"]
    e = st.columns(5)
    e[0].metric("Customers flagged", f"{vcm['tp'] + vcm['fp']:,}")
    e[1].metric("Churners caught", f"{vcm['tp']:,}")
    e[2].metric("Churners missed", f"{vcm['fn']:,}")
    e[3].metric("Recall", pct(m["recall"]))
    e[4].metric("Precision", pct(m["precision"]))
    st.caption("Lower threshold: more churners caught, more false alarms. "
               "Higher threshold: fewer false alarms, more churners missed. "
               "The right balance depends on what a retention offer costs versus a lost customer is worth.")
