"""Central configuration: paths, dataset info, constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
THRESHOLD_BETA = 2.0   # F-beta used to pick the threshold; 2 = recall counts double
CV_FOLDS = 3

RAW_CSV = RAW_DIR / "Telco-Customer-Churn.csv"
DB_PATH = PROCESSED_DIR / "churn.db"
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

TARGET = "Churn"          # "Yes" = customer left, "No" = customer stayed
ID_COL = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]
