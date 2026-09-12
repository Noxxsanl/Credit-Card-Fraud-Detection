"""Hằng số dùng chung cho cả notebook lẫn API.

Mọi giá trị ở đây xuất hiện trong tài liệu thiết kế; đổi ở đây thì phải đổi
cả tài liệu tương ứng trong docs/.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_PATH = DATA_DIR / "creditcard.csv"
TEST_SET_PATH = DATA_DIR / "test_set.parquet"
SAMPLE_POOL_PATH = DATA_DIR / "sample_pool.json"

MODEL_PATH = MODELS_DIR / "model.joblib"
EXPLAINER_PATH = MODELS_DIR / "explainer.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"

TARGET = "Class"

# ML-05: mọi bước ngẫu nhiên dùng chung hạt giống này
RANDOM_STATE = 42

TEST_SIZE = 0.2
CV_FOLDS = 5

# 04 §6.2 — hai giả định A-01, A-02, người dùng đổi được lúc chạy
DEFAULT_COST_FN = 122.21
DEFAULT_COST_FP = 5.00

# Bộ dữ liệu trải trong hai ngày (02 §1)
DATASET_DAYS = 2.0

# Kỳ vọng về dữ liệu gốc, dùng cho kiểm tra toàn vẹn (02 §6)
EXPECTED_ROWS = 284_807
EXPECTED_COLUMNS = 31
EXPECTED_FRAUD = 492
