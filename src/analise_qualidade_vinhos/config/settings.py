from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

# data
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "winequality-red.csv"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# artifacts
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# ml
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "quality_label"
QUALITY_LABELS = [
    "Baixa qualidade",
    "Alta qualidade",
]

for path in [DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, LOG_DIR, MODEL_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)