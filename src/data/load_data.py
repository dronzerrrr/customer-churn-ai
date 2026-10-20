"""Load the raw churn dataset from disk."""
import pandas as pd

from src.config import RAW_CSV


def load_raw_data() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_CSV}. Run: python -m src.data.download_data"
        )
    # Keep everything as read; cleaning happens later, in the pipeline.
    return pd.read_csv(RAW_CSV)
