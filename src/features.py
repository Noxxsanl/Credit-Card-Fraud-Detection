"""Sinh đặc trưng — ranh giới chống lệch train/serve.

Đây là đường vào DUY NHẤT để biến dữ liệu thô thành ma trận đặc trưng. Cả
notebook huấn luyện lẫn API đều gọi ``build_features``; không nơi nào khác được
tự tính ``hour`` hay tự sắp xếp cột (DS-04, 03 §4).

Hai quy tắc quan trọng:

* ``Time`` thô KHÔNG phải đặc trưng (ML-07). Nó là số giây tương đối trong một
  cửa sổ hai ngày cụ thể; chỉ giờ trong ngày mới lặp lại được.
* ``build_features`` KHÔNG chuẩn hoá ``Amount``. Việc đó thuộc về
  ``make_preprocessor()`` và phải nằm trong pipeline để chỉ fit trên tập huấn
  luyện (ML-04).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from .config import TARGET

V_COLUMNS = [f"V{i}" for i in range(1, 29)]

TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
CYCLIC_COLUMNS = ["hour_sin", "hour_cos"]

#: 30 cột bắt buộc của hợp đồng dữ liệu vào (02 §5)
RAW_REQUIRED_COLUMNS = [TIME_COLUMN, *V_COLUMNS, AMOUNT_COLUMN]

#: 31 đặc trưng đưa vào mô hình, thứ tự cố định (02 §4)
FEATURE_ORDER = [*V_COLUMNS, AMOUNT_COLUMN, *CYCLIC_COLUMNS]

#: Thứ tự cột SAU preprocessor. ``ColumnTransformer`` đưa cột đã biến đổi lên
#: đầu, nên ``Amount`` (lúc này là giá trị đã qua RobustScaler) đứng trước V1.
#: Dùng khi cần gán tên cho giá trị SHAP.
MODEL_FEATURE_NAMES = [AMOUNT_COLUMN, *V_COLUMNS, *CYCLIC_COLUMNS]

SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24


class MissingFeatureError(ValueError):
    """Thiếu cột bắt buộc trong dữ liệu vào (DS-10)."""

    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__("Thiếu cột bắt buộc: " + ", ".join(self.missing))


class InvalidFeatureValueError(ValueError):
    """Giá trị không hợp lệ: NaN, Inf, hoặc Amount âm (DS-11..DS-13)."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def hour_of_day(time_seconds) -> np.ndarray:
    """Giờ trong ngày (0..23) suy ra từ cột ``Time`` thô.

    ``Time`` là số giây kể từ giao dịch đầu tiên của bộ dữ liệu, nên phép chia
    lấy dư theo 24 giờ cho ra giờ trong ngày — thông tin duy nhất lặp lại được
    khi triển khai (DS-03).
    """
    seconds = np.asarray(time_seconds, dtype="float64")
    return np.floor(seconds / SECONDS_PER_HOUR) % HOURS_PER_DAY


def encode_hour(hour) -> tuple[np.ndarray, np.ndarray]:
    """Mã hoá giờ theo vòng tròn để 23h và 0h nằm kề nhau."""
    angle = 2.0 * np.pi * np.asarray(hour, dtype="float64") / HOURS_PER_DAY
    return np.sin(angle), np.cos(angle)


def validate_raw(df: pd.DataFrame) -> list[str]:
    """Kiểm tra hợp đồng dữ liệu vào, trả về danh sách vấn đề (rỗng nếu sạch).

    Dùng cho tầng API để trả lỗi 422 có nội dung cụ thể (DS-10..DS-13). Hàm này
    không ném ngoại lệ — bên gọi tự quyết định xử lý thế nào.
    """
    problems: list[str] = []

    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append("thiếu cột: " + ", ".join(missing))
        return problems

    numeric = df[RAW_REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")

    for column in RAW_REQUIRED_COLUMNS:
        values = numeric[column].to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            problems.append(f"{column}: có giá trị NaN hoặc vô cực")

    if (numeric[TIME_COLUMN] < 0).any():
        problems.append(f"{TIME_COLUMN}: có giá trị âm")
    if (numeric[AMOUNT_COLUMN] < 0).any():
        problems.append(f"{AMOUNT_COLUMN}: có giá trị âm")

    return problems


def build_features(df: pd.DataFrame, *, validate: bool = False) -> pd.DataFrame:
    """Biến dữ liệu thô thành ma trận đặc trưng theo đúng ``FEATURE_ORDER``.

    Tham số
    -------
    df
        DataFrame chứa tối thiểu 30 cột trong ``RAW_REQUIRED_COLUMNS``. Cột thừa
        (kể cả ``Class``) bị bỏ qua (DS-14).
    validate
        Bật kiểm tra giá trị NaN/Inf/âm và ném ``InvalidFeatureValueError``.
        Mặc định tắt vì đường huấn luyện đã kiểm tra toàn vẹn từ trước; tầng API
        thì bật.

    Trả về
    ------
    DataFrame 31 cột, cùng chỉ mục với ``df``, kiểu float64.
    """
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MissingFeatureError(missing)

    if validate:
        problems = validate_raw(df)
        if problems:
            raise InvalidFeatureValueError(problems)

    hour = hour_of_day(df[TIME_COLUMN].to_numpy())
    hour_sin, hour_cos = encode_hour(hour)

    features = df[[*V_COLUMNS, AMOUNT_COLUMN]].astype("float64").copy()
    features["hour_sin"] = hour_sin
    features["hour_cos"] = hour_cos

    return features[FEATURE_ORDER]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Tách ma trận đặc trưng và nhãn từ một DataFrame có cột ``Class``."""
    if TARGET not in df.columns:
        raise MissingFeatureError([TARGET])
    return build_features(df), df[TARGET].astype(int)


def make_preprocessor() -> ColumnTransformer:
    """Bước tiền xử lý đặt ở đầu pipeline.

    Chỉ ``Amount`` được chuẩn hoá, bằng ``RobustScaler`` vì phân bố lệch phải rất
    mạnh (DS-02). V1–V28 đã là thành phần chính sau PCA nên giữ nguyên (DS-01),
    ``hour_sin``/``hour_cos`` vốn đã nằm trong [-1, 1].

    Scaler nằm trong pipeline chứ không fit sẵn ở đây — đó là điều kiện để nó chỉ
    học thống kê từ tập huấn luyện trong mỗi fold (ML-04).
    """
    preprocessor = ColumnTransformer(
        transformers=[("amount", RobustScaler(), [AMOUNT_COLUMN])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")
