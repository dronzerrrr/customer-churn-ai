"""Download the IBM Telco Customer Churn CSV into data/raw/."""
import logging
import urllib.request

from src.config import DATASET_URL, RAW_CSV

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def download_dataset(force: bool = False) -> None:
    if RAW_CSV.exists() and not force:
        log.info("Dataset already exists at %s (use force=True to re-download).", RAW_CSV)
        return
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading dataset from %s", DATASET_URL)
    try:
        urllib.request.urlretrieve(DATASET_URL, RAW_CSV)
    except Exception as exc:  # network, DNS, proxy, etc.
        log.error("Download failed: %s", exc)
        log.error(
            "Manual fallback: download 'Telco-Customer-Churn.csv' from the IBM "
            "GitHub repo or Kaggle ('Telco Customer Churn') and save it as:\n  %s",
            RAW_CSV,
        )
        raise SystemExit(1) from exc
    log.info("Saved to %s (%.1f KB)", RAW_CSV, RAW_CSV.stat().st_size / 1024)


if __name__ == "__main__":
    download_dataset()
