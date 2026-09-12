"""Nạp, kiểm tra và chia dữ liệu ``creditcard.csv``.

Thứ tự bắt buộc: nạp → kiểm tra toàn vẹn → loại dòng trùng lặp → chia tập
(DS-20, ML-06). Loại trùng lặp sau khi chia sẽ để lọt cùng một dòng vào cả hai
tập, tạo rò rỉ nhẹ nhưng có thật.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    DATASET_DAYS,
    EXPECTED_COLUMNS,
    EXPECTED_FRAUD,
    EXPECTED_ROWS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET,
    TEST_SIZE,
)
from .features import (
    RAW_REQUIRED_COLUMNS,
    TIME_COLUMN,
    build_features,
    hour_of_day,
)

SECONDS_PER_DAY = 86_400

# Giữ lại để mã cũ trong notebook không gãy
DATA_PATH = RAW_DATA_PATH


@dataclass
class IntegrityReport:
    """Kết quả kiểm tra toàn vẹn dữ liệu gốc (02 §6)."""

    n_rows: int
    n_columns: int
    n_missing_cells: int
    n_duplicates: int
    n_fraud: int
    fraud_rate: float
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        lines = [
            f"Số dòng        : {self.n_rows:,}",
            f"Số cột         : {self.n_columns}",
            f"Ô thiếu giá trị: {self.n_missing_cells:,}",
            f"Dòng trùng lặp : {self.n_duplicates:,}",
            f"Gian lận       : {self.n_fraud:,} ({self.fraud_rate:.3%})",
        ]
        if self.problems:
            lines.append("Vấn đề phát hiện:")
            lines.extend(f"  - {p}" for p in self.problems)
        else:
            lines.append("Không phát hiện vấn đề.")
        return "\n".join(lines)


def load_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Đọc dataset gốc, chưa xử lý."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Tải creditcard.csv từ Kaggle và đặt vào thư mục data/ "
            "(xem docs/10-van-hanh-tai-lap.md §2.4)."
        )
    return pd.read_csv(path)


def check_integrity(df: pd.DataFrame, *, strict: bool = False) -> IntegrityReport:
    """Đối chiếu dữ liệu với kỳ vọng đã ghi trong đặc tả.

    Sai lệch về số dòng hay số mẫu gian lận nghĩa là tệp tải về không phải bản
    chuẩn — mọi kết quả về sau sẽ không so sánh được với báo cáo.
    """
    problems: list[str] = []

    missing_columns = [c for c in [*RAW_REQUIRED_COLUMNS, TARGET] if c not in df.columns]
    if missing_columns:
        problems.append("thiếu cột: " + ", ".join(missing_columns))

    n_rows, n_columns = df.shape
    if n_rows != EXPECTED_ROWS:
        problems.append(f"số dòng {n_rows:,}, kỳ vọng {EXPECTED_ROWS:,}")
    if n_columns != EXPECTED_COLUMNS:
        problems.append(f"số cột {n_columns}, kỳ vọng {EXPECTED_COLUMNS}")

    n_missing = int(df.isna().sum().sum())
    if n_missing:
        problems.append(f"{n_missing:,} ô thiếu giá trị, kỳ vọng 0")

    n_fraud = int(df[TARGET].sum()) if TARGET in df.columns else 0
    if TARGET in df.columns and n_fraud != EXPECTED_FRAUD:
        problems.append(f"số mẫu gian lận {n_fraud:,}, kỳ vọng {EXPECTED_FRAUD:,}")

    report = IntegrityReport(
        n_rows=n_rows,
        n_columns=n_columns,
        n_missing_cells=n_missing,
        n_duplicates=int(df.duplicated().sum()),
        n_fraud=n_fraud,
        fraud_rate=n_fraud / n_rows if n_rows else 0.0,
        problems=problems,
    )

    if strict and not report.ok:
        raise ValueError("Dữ liệu không đạt kiểm tra toàn vẹn:\n" + report.summary())
    return report


def drop_duplicates(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Loại dòng trùng lặp hoàn toàn — phải gọi TRƯỚC khi chia tập (DS-20)."""
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(cleaned)

    if verbose:
        fraud_removed = 0
        if TARGET in df.columns:
            fraud_removed = int(df[TARGET].sum()) - int(cleaned[TARGET].sum())
        print(
            f"Loại {removed:,} dòng trùng lặp ({removed / before:.3%}), "
            f"trong đó {fraud_removed} mẫu gian lận. Còn lại {len(cleaned):,} dòng."
        )
    return cleaned


def add_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột ``hour`` cho EDA. Không dùng cho huấn luyện — xem ``build_features``."""
    out = df.copy()
    out["hour"] = hour_of_day(out[TIME_COLUMN].to_numpy()).astype(int)
    return out


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = 0.0,
    random_state: int = RANDOM_STATE,
    *,
    as_features: bool = True,
):
    """Chia phân tầng theo nhãn ``Class`` (ML-02).

    ``as_features=True`` trả về ma trận đặc trưng đã qua ``build_features``;
    đặt ``False`` khi cần giữ nguyên cột thô (ví dụ để lưu ``test_set.parquet``).

    Trả về ``(X_train, X_test, y_train, y_test)``, hoặc thêm bộ validation khi
    ``val_size > 0``.
    """
    X = build_features(df) if as_features else df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if val_size <= 0:
        return X_train, X_test, y_train, y_test

    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=rel_val, stratify=y_train, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def temporal_split(df: pd.DataFrame, *, as_features: bool = True):
    """Chia theo thời gian: ngày 1 huấn luyện, ngày 2 kiểm thử (FR-10).

    Cách chia này mô phỏng triển khai thật hơn cách chia ngẫu nhiên, và hiệu năng
    thường THẤP hơn. Chênh lệch giữa hai cách chính là phần thảo luận có giá trị
    trong báo cáo, không phải lỗi cần giấu.
    """
    is_day_one = df[TIME_COLUMN] < SECONDS_PER_DAY
    train_df, test_df = df[is_day_one], df[~is_day_one]

    if as_features:
        return (
            build_features(train_df),
            build_features(test_df),
            train_df[TARGET].astype(int),
            test_df[TARGET].astype(int),
        )
    return (
        train_df.drop(columns=[TARGET]),
        test_df.drop(columns=[TARGET]),
        train_df[TARGET].astype(int),
        test_df[TARGET].astype(int),
    )


def class_balance(y) -> pd.DataFrame:
    """Tóm tắt mức độ mất cân bằng của nhãn."""
    counts = pd.Series(y).value_counts().sort_index()
    total = counts.sum()
    return pd.DataFrame(
        {
            "count": counts,
            "ratio": counts / total,
            "one_in": (total / counts).round(1),
        }
    )


def load_prepared(path=RAW_DATA_PATH, *, strict: bool = False, verbose: bool = True):
    """Nạp → kiểm tra → loại trùng lặp, trả về ``(df, report)``.

    Đây là đường vào tiêu chuẩn cho mọi notebook từ 03 trở đi.
    """
    df = load_data(path)
    report = check_integrity(df, strict=strict)
    if verbose:
        print(report.summary())
    return drop_duplicates(df, verbose=verbose), report


def observed_days(df: pd.DataFrame) -> float:
    """Số ngày mà dữ liệu trải qua — dùng để quy đổi cảnh báo mỗi ngày."""
    if TIME_COLUMN not in df.columns or df.empty:
        return DATASET_DAYS
    span = float(df[TIME_COLUMN].max() - df[TIME_COLUMN].min())
    return max(span / SECONDS_PER_DAY, 1e-9)
