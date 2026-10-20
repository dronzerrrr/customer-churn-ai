"""SQL business analytics. Every number comes from the actual database."""
import sqlite3

import pandas as pd

from src.config import DB_PATH

QUERIES = {
    "Overall": """
        SELECT COUNT(*) AS customers,
               SUM(churn_flag) AS churned,
               ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct,
               ROUND(SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges END), 2)
                   AS monthly_revenue_of_churned
        FROM customers
    """,
    "Churn by contract": """
        SELECT Contract, COUNT(*) AS customers,
               ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
        FROM customers GROUP BY Contract ORDER BY churn_rate_pct DESC
    """,
    "Churn by tenure group": """
        SELECT tenure_group, COUNT(*) AS customers,
               ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
        FROM customers GROUP BY tenure_group ORDER BY MIN(tenure)
    """,
    "Churn by payment method": """
        SELECT PaymentMethod, COUNT(*) AS customers,
               ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
        FROM customers GROUP BY PaymentMethod ORDER BY churn_rate_pct DESC
    """,
    "Churn by internet service": """
        SELECT InternetService, COUNT(*) AS customers,
               ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct,
               ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge
        FROM customers GROUP BY InternetService ORDER BY churn_rate_pct DESC
    """,
    "Contract x tenure (segments with 100+ customers)": """
        SELECT Contract, tenure_group, COUNT(*) AS customers,
               ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
        FROM customers GROUP BY Contract, tenure_group
        HAVING COUNT(*) >= 100 ORDER BY churn_rate_pct DESC LIMIT 5
    """,
}


def run_all() -> dict[str, pd.DataFrame]:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} not found. Run: python -m src.data.database")
    with sqlite3.connect(DB_PATH) as conn:
        return {name: pd.read_sql_query(q, conn) for name, q in QUERIES.items()}


if __name__ == "__main__":
    for name, result in run_all().items():
        print(f"\n--- {name} ---")
        print(result.to_string(index=False))
