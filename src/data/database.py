"""Load the cleaned dataset into a local SQLite database."""
import logging
import sqlite3

from src.config import DB_PATH
from src.data.clean import clean_for_analysis
from src.data.load_data import load_raw_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def build_database() -> None:
    df = clean_for_analysis(load_raw_data())
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("customers", conn, if_exists="replace", index=False)
        n = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    log.info("Wrote %d rows to table 'customers' in %s", n, DB_PATH)


if __name__ == "__main__":
    build_database()
