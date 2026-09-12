"""Nạp và chia dữ liệu creditcard.csv."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

TARGET = "Class"
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    """Đọc dataset gốc, chưa xử lý."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Tải creditcard.csv từ Kaggle và đặt vào thư mục data/."
        )
    return pd.read_csv(path)


def split_data(df, test_size=0.2, val_size=0.0, random_state=RANDOM_STATE):
    """Chia stratified theo nhãn Class.

    Trả về (X_train, X_test, y_train, y_test), hoặc thêm bộ validation
    nếu val_size > 0: (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if val_size <= 0:
        return X_train, X_test, y_train, y_test

    # val_size tính theo tỉ lệ trên toàn bộ dữ liệu ban đầu
    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=rel_val, stratify=y_train, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def class_balance(y):
    """Tóm tắt mức độ mất cân bằng của nhãn."""
    counts = y.value_counts().sort_index()
    return pd.DataFrame({"count": counts, "ratio": counts / counts.sum()})
