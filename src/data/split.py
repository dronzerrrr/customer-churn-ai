"""Stratified train / validation / test split (60/20/20), saved to disk.

The TEST set is touched only once, at the very end (Phase 4 final evaluation).
"""
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, RANDOM_STATE, TARGET, TEST_SIZE, VAL_SIZE
from src.data.load_data import load_raw_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def make_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    y = (df[TARGET] == "Yes").astype(int)
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    y_tv = (train_val[TARGET] == "Yes").astype(int)
    val_fraction = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_fraction, stratify=y_tv, random_state=RANDOM_STATE)
    return {"train": train, "val": val, "test": test}


def save_splits() -> None:
    splits = make_splits(load_raw_data())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        rate = (part[TARGET] == "Yes").mean()
        log.info("%-5s rows=%d  churn_rate=%.2f%%", name, len(part), 100 * rate)


if __name__ == "__main__":
    save_splits()
