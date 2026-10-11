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
OOF_SCORES_PATH = MODELS_DIR / "oof_scores.npz"

TARGET = "Class"

# ML-05: mọi bước ngẫu nhiên dùng chung hạt giống này
RANDOM_STATE = 42

TEST_SIZE = 0.2
CV_FOLDS = 5

# 04 §6.2 — hai giả định A-01, A-02, người dùng đổi được lúc chạy.
# A-01: trung bình Amount của gian lận **chỉ trên tập huấn luyện** (378 vụ, sau loại trùng lặp):
# 115,939… EUR. Trước review con số là 122,21 — trung bình trên dữ liệu thô, gồm cả tập kiểm thử
# (docs/11 m3). notebook 06 §6.1 kiểm lại con số này mỗi lần chạy.
DEFAULT_COST_FN = 115.94
DEFAULT_COST_FP = 5.00

# 04 §6.3 — ràng buộc vận hành. Ngân sách thẩm định là ràng buộc của ngưỡng MẶC ĐỊNH (docs/11 M1):
# τ mặc định = cực tiểu chi phí trong số các ngưỡng có cảnh báo/ngày ≤ ngân sách.
DEFAULT_ALERT_BUDGET = 200.0
DEFAULT_MIN_RECALL = 0.90
THRESHOLD_POLICY = "min_cost_within_budget"

# docs/11 M2 — ngưỡng chọn trên điểm forward-chaining theo Time, không trên CV ngẫu nhiên
TEMPORAL_SPLITS = 4

# Cấu hình mô hình cuối — chọn ở notebook 04 §5.8 bằng quy tắc tie-break đặt trước (docs/11 M4):
# mọi ứng viên XGBoost hoà với cấu hình có PR-AUC CV lặp cao nhất (KTC của hiệu chứa 0), nên chọn
# chiến lược đơn giản nhất — không xử lý mất cân bằng. Notebook 05–08 đọc từ đây, không tự ghi tên mô hình.
FINAL_MODEL = "xgboost"
FINAL_STRATEGY = "none"
FINAL_RATIO = None

# Bộ dữ liệu trải trong hai ngày (02 §1)
DATASET_DAYS = 2.0

# Kỳ vọng về dữ liệu gốc, dùng cho kiểm tra toàn vẹn (02 §6)
EXPECTED_ROWS = 284_807
EXPECTED_COLUMNS = 31
EXPECTED_FRAUD = 492
