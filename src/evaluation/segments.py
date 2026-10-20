"""Data-driven segment summaries (no hardcoded numbers)."""
from itertools import combinations

import numpy as np
import pandas as pd


def segment_table(df: pd.DataFrame, col: str, min_n: int = 100) -> pd.DataFrame:
    """Observed churn rate per category, hiding segments smaller than min_n."""
    g = (df.groupby(col, observed=True)["churn_flag"]
           .agg(customers="size", churned="sum", churn_rate="mean").reset_index())
    g = g[g["customers"] >= min_n]
    return g.sort_values("churn_rate", ascending=False).reset_index(drop=True)


OBSERVATION_COLUMNS = [
    ("Contract", "contract type"),
    ("tenure_group", "tenure group"),
    ("PaymentMethod", "payment method"),
    ("InternetService", "internet service"),
]


def key_observations(df: pd.DataFrame, min_n: int = 100) -> list[str]:
    if df.empty:
        return []
    overall = df["churn_flag"].mean()
    out = []
    for col, label in OBSERVATION_COLUMNS:
        t = segment_table(df, col, min_n=min_n)
        if len(t) < 2:
            continue
        hi, lo = t.iloc[0], t.iloc[-1]
        out.append(
            f"By {label}, observed churn ranges from {lo['churn_rate']:.1%} "
            f"({lo[col]}) to {hi['churn_rate']:.1%} ({hi[col]}); overall is {overall:.1%}."
        )
    return out


def wilson_interval(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a proportion k/n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


SCAN_COLUMNS = ["Contract", "tenure_group", "InternetService", "PaymentMethod",
                "PaperlessBilling", "TechSupport", "OnlineSecurity", "SeniorCitizen",
                "Partner", "Dependents"]


def segment_scan(df: pd.DataFrame, columns: list[str] = SCAN_COLUMNS,
                 min_n: int = 100, pairs: bool = True) -> pd.DataFrame:
    """Observed churn for single and paired categories, with confidence intervals.

    Segments overlap, so their excess_churners must NOT be added together. Scanning many
    segments means some will look extreme by chance: use the intervals, not just the rates.
    """
    overall = df["churn_flag"].mean()
    total_churned = int(df["churn_flag"].sum())
    rows, seen = [], set()

    def add(label: str, sub: pd.DataFrame) -> None:
        n = len(sub)
        if n < min_n:
            return
        key = (n, hash(sub.index.to_numpy().tobytes()))
        if key in seen:                       # identical customer set already listed
            return
        seen.add(key)
        k = int(sub["churn_flag"].sum())
        lo, hi = wilson_interval(k, n)
        rows.append({"segment": label, "customers": n, "churned": k, "churn_rate": k / n,
                     "ci_low": lo, "ci_high": hi, "lift": (k / n) / overall,
                     "share_of_churners": k / total_churned,
                     "excess_churners": k - n * overall})

    for c in columns:
        for v, sub in df.groupby(c, observed=True):
            add(f"{c} = {v}", sub)
    if pairs:
        for c1, c2 in combinations(columns, 2):
            for (v1, v2), sub in df.groupby([c1, c2], observed=True):
                add(f"{c1} = {v1} & {c2} = {v2}", sub)
    return pd.DataFrame(rows)
