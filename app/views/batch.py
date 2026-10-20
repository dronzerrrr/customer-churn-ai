import pandas as pd
import plotly.express as px
import streamlit as st

from app.common import CHURN_COLOR, STAY_COLOR, load_service
from src.config import PROCESSED_DIR, TARGET
from src.models.batch import parse_csv_bytes, run_batch
from src.models.risk import BAND_ORDER
from src.models.schema import ALLOWED_VALUES, REQUIRED_COLUMNS

MAX_UPLOAD_MB = 5
BAND_COLORS = {"Low": STAY_COLOR, "Medium": "#F2B134", "High": CHURN_COLOR}


@st.cache_data(show_spinner=False)
def template_csv() -> bytes:
    sample = pd.read_csv(PROCESSED_DIR / "test.csv", nrows=5).drop(columns=[TARGET], errors="ignore")
    return sample.to_csv(index=False).encode()


def render() -> None:
    st.title("Batch Prediction")
    service = load_service()
    st.write("Upload a CSV with one customer per row to score many customers at once.")

    with st.expander("Required columns and allowed values"):
        st.write("Required columns: " + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS) + ". "
                 "`TotalCharges` is optional (estimated as tenure x monthly charges when blank). "
                 "`customerID` is optional and is carried through to the results.")
        st.write("`SeniorCitizen`: 0 or 1. `tenure`: whole months, 0 to 72. "
                 "`MonthlyCharges`: a positive number.")
        rows = [{"Column": c, "Allowed values": ", ".join(v)} for c, v in ALLOWED_VALUES.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.download_button("Download a sample file (5 rows from the public dataset)",
                       template_csv(), "sample_customers.csv", "text/csv", key="dl_template")

    upload = st.file_uploader("Upload a CSV file", type=["csv"], key="batch_upload")
    if upload is None:
        return
    if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"The file is larger than {MAX_UPLOAD_MB} MB. Please split it.")
        return
    try:
        df = parse_csv_bytes(upload.getvalue())
        result = run_batch(df, service)
    except ValueError as exc:
        st.error(str(exc))
        return

    if result.missing_columns:
        st.error("The file is missing required columns: " + ", ".join(result.missing_columns))
        st.caption(f"Columns found: {', '.join(map(str, df.columns))}")
        return

    c = st.columns(4)
    c[0].metric("Rows uploaded", f"{result.n_received:,}")
    c[1].metric("Rows scored", f"{result.n_scored:,}")
    c[2].metric("Rows rejected", f"{result.n_rejected:,}")
    c[3].metric("Total charges estimated", f"{result.n_estimated_total_charges:,}")

    if result.errors:
        st.warning(f"{result.n_rejected:,} row(s) were not scored. Row 1 is the first data row "
                   "after the header. Nothing was changed or guessed for these rows.")
        errors = pd.DataFrame(result.errors)
        st.dataframe(errors.head(200), hide_index=True, width="stretch")
        if len(errors) > 200:
            st.caption(f"Showing the first 200 of {len(errors):,} problems; download for all.")
        st.download_button("Download all problems (CSV)", errors.to_csv(index=False).encode(),
                           "rejected_rows.csv", "text/csv", key="dl_errors")
    if result.scored.empty:
        st.error("No rows could be scored. Fix the problems above and upload again.")
        return

    st.success(f"Scored {result.n_scored:,} customers. {int(result.scored['predicted_churn'].sum()):,} "
               f"are above the {service.threshold:.0%} decision threshold.")
    counts = (result.scored["risk_band"].value_counts().reindex(BAND_ORDER, fill_value=0)
              .rename_axis("Risk band").reset_index(name="Customers"))
    fig = px.bar(counts, x="Risk band", y="Customers", color="Risk band",
                 color_discrete_map=BAND_COLORS, text="Customers")
    fig.update_layout(showlegend=False, height=280, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch", key="batch_band_counts")

    ranked = result.scored.sort_values("churn_probability", ascending=False)
    st.dataframe(ranked.style.format({"churn_probability": "{:.1%}"}).hide(axis="index"),
                 width="stretch", height=360)
    st.download_button("Download predictions (CSV)", ranked.to_csv(index=False).encode(),
                       "churn_predictions.csv", "text/csv", key="dl_results")
    st.caption("Probabilities are estimates, not certainties. Use them to prioritise outreach, "
               "not as proof that a customer will leave.")
