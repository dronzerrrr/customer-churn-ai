"""Evidence-backed retention recommendations, computed from the data.

Every number in the headline/robustness text is calculated here from the dataset.
All evidence is OBSERVATIONAL: it shows association, not that an action would work.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_CELL = 100


@dataclass
class Recommendation:
    key: str
    title: str
    action: str
    target_label: str
    headline: str
    evidence_title: str
    evidence: pd.DataFrame
    robustness: str
    caveat: str
    test_idea: str
    target_customers: int
    target_churn_rate: float
    target_monthly_charge: float
    extra: dict = field(default_factory=dict)


def group_stats(df: pd.DataFrame, mask: pd.Series) -> tuple[int, float, float]:
    sub = df[mask]
    if sub.empty:
        return 0, float("nan"), float("nan")
    return len(sub), float(sub["churn_flag"].mean()), float(sub["MonthlyCharges"].mean())


def compare_within(df: pd.DataFrame, a_mask: pd.Series, b_mask: pd.Series,
                   strata: pd.Series, min_cell: int = MIN_CELL) -> pd.DataFrame:
    """Churn of group A vs group B inside each stratum; cells under min_cell are not compared."""
    rows = []
    for value in sorted(strata.dropna().unique(), key=str):
        s = strata == value
        a, b = df[a_mask & s], df[b_mask & s]
        rows.append({
            "group": str(value), "n_a": len(a), "rate_a": a["churn_flag"].mean() if len(a) else np.nan,
            "n_b": len(b), "rate_b": b["churn_flag"].mean() if len(b) else np.nan,
            "comparable": len(a) >= min_cell and len(b) >= min_cell})
    return pd.DataFrame(rows)


def robustness_sentence(table: pd.DataFrame, label_a: str, label_b: str,
                        strata_name: str, min_cell: int = MIN_CELL) -> str:
    comp = table[table["comparable"]]
    if comp.empty:
        return (f"No {strata_name} subgroup has {min_cell}+ customers on both sides, so this "
                "cannot be checked within subgroups.")
    higher = int((comp["rate_a"] > comp["rate_b"]).sum())
    details = "; ".join(f"{r.group}: {r.rate_a:.1%} vs {r.rate_b:.1%}" for r in comp.itertuples())
    verdict = "Consistent" if higher == len(comp) else "Mixed evidence"
    label_a = label_a[:1].lower() + label_a[1:]
    return (f"{verdict}. Within {len(comp)} comparable {strata_name} subgroup(s) (at least "
            f"{min_cell} customers on each side), {label_a} have a higher churn rate than "
            f"{label_b} in {higher} of {len(comp)} ({details}).")


def _fmt_table(t: pd.DataFrame, label_a: str, label_b: str, strata_name: str) -> pd.DataFrame:
    out = pd.DataFrame({
        strata_name: t["group"],
        f"{label_a}: customers": t["n_a"], f"{label_a}: churn": t["rate_a"],
        f"{label_b}: customers": t["n_b"], f"{label_b}: churn": t["rate_b"],
        "Compared?": np.where(t["comparable"], "yes", "too few customers")})
    return out


def build_recommendations(df: pd.DataFrame, min_cell: int = MIN_CELL) -> list[Recommendation]:
    total_churned = df["churn_flag"].sum()
    recs: list[Recommendation] = []

    def finish(key, title, action, label, a_mask, b_mask, label_b, headline, ev_title, strat,
               strata_name, caveat, test_idea, pool=None):
        pool = df if pool is None else pool
        table = compare_within(pool, a_mask.loc[pool.index], b_mask.loc[pool.index],
                               strat.loc[pool.index], min_cell)
        n, rate, charge = group_stats(df, a_mask)
        recs.append(Recommendation(
            key, title, action, label, headline, ev_title,
            _fmt_table(table, label, label_b, strata_name),
            robustness_sentence(table, label, label_b, strata_name, min_cell),
            caveat, test_idea, n, rate, charge))

    # 1. Early-tenure onboarding
    a = df["tenure"] <= 12
    b = ~a
    n, rate, _ = group_stats(df, a)
    finish("early_tenure", "Strengthen the first year",
           "Proactive onboarding and check-ins for customers in their first 12 months.",
           "First-year customers", a, b, "longer-tenured customers",
           f"Customers in their first 12 months churn at {rate:.1%}, versus "
           f"{df[b]['churn_flag'].mean():.1%} for longer-tenured customers. They are "
           f"{n / len(df):.0%} of customers but {df[a]['churn_flag'].sum() / total_churned:.0%} "
           f"of all churners.",
           "First-year vs later, within each contract type", df["Contract"], "contract",
           "Newer customers differ in many ways (plan, price, how they were acquired). The data "
           "shows where churn happens, not what would keep these customers.",
           "Randomly assign half of new customers an onboarding programme; compare churn at 12 months.")

    # 2. Contract
    a = df["Contract"] == "Month-to-month"
    b = ~a
    n, rate, _ = group_stats(df, a)
    finish("contract", "Encourage longer contracts",
           "Offer an incentive to move month-to-month customers onto a one- or two-year plan.",
           "Month-to-month customers", a, b, "annual-contract customers",
           f"Month-to-month customers churn at {rate:.1%}, versus {df[b]['churn_flag'].mean():.1%} "
           f"for customers on one- or two-year contracts.",
           "Month-to-month vs annual, within each tenure group", df["tenure_group"], "tenure group",
           "Customers who choose a long contract may already be more loyal, so moving someone "
           "onto one need not make them stay. The size of any real effect is unknown.",
           "Randomly offer half of month-to-month customers a discount for a 1-year contract; "
           "compare churn after 6-12 months.")

    # 3. Payment method
    a = df["PaymentMethod"] == "Electronic check"
    b = ~a
    n, rate, _ = group_stats(df, a)
    finish("payment", "Nudge towards automatic payment",
           "Encourage electronic-check customers to switch to automatic bank or card payment.",
           "Electronic-check customers", a, b, "other payment methods",
           f"Electronic-check customers churn at {rate:.1%}, versus {df[b]['churn_flag'].mean():.1%} "
           f"for customers on other payment methods.",
           "Electronic check vs other methods, within each contract type", df["Contract"], "contract",
           "Payment method may be a stand-in for something else (customer profile, billing "
           "experience). Changing the payment method is not shown to change churn.",
           "Randomly invite half of electronic-check customers to set up autopay; compare churn.")

    # 4. Fiber optic
    pool = df[df["InternetService"] != "No"]
    a = df["InternetService"] == "Fiber optic"
    b = df["InternetService"] == "DSL"
    n, rate, charge = group_stats(df, a)
    bands = pd.Series(pd.qcut(pool["MonthlyCharges"], 3, labels=["lower price", "middle price",
                                                                  "higher price"]).astype(str),
                      index=pool.index).reindex(df.index)
    finish("fiber", "Review fiber-optic service and value",
           "Investigate service quality, pricing and competitor offers for fiber customers.",
           "Fiber-optic customers", a, b, "DSL customers",
           f"Fiber-optic customers churn at {rate:.1%}, versus "
           f"{df[b]['churn_flag'].mean():.1%} for DSL customers, at an average monthly charge of "
           f"{charge:.0f} versus {df[b]['MonthlyCharges'].mean():.0f}.",
           "Fiber vs DSL, within monthly-charge bands (internet customers)", bands, "price band",
           "This dataset has no data on outages, speeds, support contacts or competitors, so it "
           "cannot say why fiber customers leave more. Treat it as a pointer for investigation.",
           "Survey and interview a sample of churned fiber customers; test a service-quality "
           "intervention on a random subset.", pool=pool)

    # 5. Support add-ons
    internet = df["InternetService"] != "No"
    a = internet & (df["TechSupport"] != "Yes") & (df["OnlineSecurity"] != "Yes")
    b = internet & ~a
    n, rate, _ = group_stats(df, a)
    finish("support", "Promote support and security add-ons",
           "Offer tech support or online security trials to internet customers who have neither.",
           "Internet customers with neither", a, b, "internet customers with at least one",
           f"Internet customers with neither TechSupport nor OnlineSecurity churn at {rate:.1%}, "
           f"versus {df[b]['churn_flag'].mean():.1%} for internet customers with at least one.",
           "Neither vs at least one, within each internet service type",
           df["InternetService"], "internet service",
           "Customers who buy add-ons are probably more engaged to begin with. A free trial may "
           "not reproduce that gap.",
           "Randomly offer a free 3-month add-on trial to half of these customers; compare churn.",
           pool=df[internet])
    return recs


def model_targeting_summary(test_df: pd.DataFrame, y, proba, threshold: float) -> dict:
    """How well the model's flags concentrate churners (on held-out test customers)."""
    y = np.asarray(y)
    flagged = np.asarray(proba) >= threshold
    base = float(y.mean())
    rate = float(y[flagged].mean()) if flagged.any() else float("nan")
    return {
        "customers": int(flagged.sum()),
        "churn_rate": rate,
        "unflagged_churn_rate": float(y[~flagged].mean()) if (~flagged).any() else float("nan"),
        "base_rate": base,
        "churners_caught": int(y[flagged].sum()),
        "total_churners": int(y.sum()),
        "offers_per_churner_flagged": (1 / rate) if rate and rate > 0 else None,
        "offers_per_churner_all": 1 / base if base > 0 else None,
        "monthly_charge": float(pd.to_numeric(test_df["MonthlyCharges"])[flagged].mean())
        if flagged.any() else float("nan"),
    }
