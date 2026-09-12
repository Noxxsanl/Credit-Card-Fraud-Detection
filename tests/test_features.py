"""TC-01…TC-06 — canh giữ ranh giới train/serve.

Nếu ai đó đổi cách sinh đặc trưng mà quên cập nhật một phía, các ca ở đây đỏ
ngay. Không có lưới an toàn này, lỗi chỉ lộ ra khi số liệu demo lệch báo cáo và
lúc đó rất khó truy nguyên (03 §4).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_ORDER,
    MODEL_FEATURE_NAMES,
    RAW_REQUIRED_COLUMNS,
    V_COLUMNS,
    InvalidFeatureValueError,
    MissingFeatureError,
    build_features,
    encode_hour,
    hour_of_day,
    make_preprocessor,
    validate_raw,
)

GOLDEN_DIR = Path(__file__).parent / "data"
GOLDEN_INPUT = GOLDEN_DIR / "golden_input.csv"
GOLDEN_FEATURES = GOLDEN_DIR / "golden_features.csv"


def make_raw(n_rows: int = 5, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"Time": np.arange(n_rows, dtype="float64") * 3600.0}
    for column in V_COLUMNS:
        data[column] = rng.normal(size=n_rows)
    data["Amount"] = rng.uniform(0, 500, size=n_rows)
    return pd.DataFrame(data)


# --------------------------------------------------------------------------
# TC-01 — kiểm thử vàng
# --------------------------------------------------------------------------

def test_tc01_golden_sample_matches_stored_matrix():
    """20 dòng mẫu cố định phải cho ra đúng ma trận đã lưu, sai số < 1e-9."""
    raw = pd.read_csv(GOLDEN_INPUT)
    expected = pd.read_csv(GOLDEN_FEATURES)

    actual = build_features(raw)

    assert list(actual.columns) == list(expected.columns)
    np.testing.assert_allclose(
        actual.to_numpy(dtype="float64"),
        expected.to_numpy(dtype="float64"),
        atol=1e-9,
        rtol=0,
    )


# --------------------------------------------------------------------------
# TC-02 — thứ tự và số lượng cột
# --------------------------------------------------------------------------

def test_tc02_column_order_is_fixed():
    features = build_features(make_raw())

    assert list(features.columns) == FEATURE_ORDER
    # 28 thành phần PCA + Amount + hai cột mã hoá giờ
    assert len(FEATURE_ORDER) == 31


def test_tc02_column_order_independent_of_input_order():
    raw = make_raw()
    shuffled = raw[list(reversed(raw.columns))]

    pd.testing.assert_frame_equal(build_features(raw), build_features(shuffled))


def test_tc02_preprocessor_output_names_match_constant():
    raw = make_raw(n_rows=20)
    transformed = make_preprocessor().fit_transform(build_features(raw))

    assert list(transformed.columns) == MODEL_FEATURE_NAMES


# --------------------------------------------------------------------------
# TC-03 — biên của giờ trong ngày
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("time_seconds", "expected_hour"),
    [
        (0.0, 0),
        (3599.0, 0),
        (3600.0, 1),
        (86_399.0, 23),
        (86_400.0, 0),  # sang ngày thứ hai, giờ quay lại 0
        (90_000.0, 1),
        (172_792.0, 23),  # giá trị Time lớn nhất trong bộ dữ liệu
    ],
)
def test_tc03_hour_of_day(time_seconds, expected_hour):
    assert hour_of_day([time_seconds])[0] == expected_hour


def test_tc03_cyclic_encoding_within_unit_circle():
    raw = make_raw(n_rows=48)
    features = build_features(raw)

    assert features["hour_sin"].between(-1, 1).all()
    assert features["hour_cos"].between(-1, 1).all()

    radius = features["hour_sin"] ** 2 + features["hour_cos"] ** 2
    np.testing.assert_allclose(radius.to_numpy(), 1.0, atol=1e-12)


# --------------------------------------------------------------------------
# TC-04 — tính tuần hoàn: 23h và 0h phải kề nhau
# --------------------------------------------------------------------------

def test_tc04_hour_23_is_closer_to_hour_0_than_to_hour_12():
    def point(hour: int) -> np.ndarray:
        sin, cos = encode_hour([hour])
        return np.array([sin[0], cos[0]])

    d_23_to_0 = np.linalg.norm(point(23) - point(0))
    d_23_to_12 = np.linalg.norm(point(23) - point(12))

    assert d_23_to_0 < d_23_to_12
    # Khoảng cách giữa hai giờ liền kề trên vòng tròn 24 điểm
    assert d_23_to_0 == pytest.approx(2 * np.sin(np.pi / 24), abs=1e-12)


# --------------------------------------------------------------------------
# TC-05 — Time thô không được lọt vào đặc trưng (ML-07)
# --------------------------------------------------------------------------

def test_tc05_raw_time_is_not_a_feature():
    features = build_features(make_raw())

    assert "Time" not in features.columns
    assert "hour" not in features.columns


def test_tc05_shifting_time_by_whole_days_does_not_change_features():
    """Cùng một giao dịch ở ngày khác phải cho cùng đặc trưng."""
    raw = make_raw(n_rows=10)
    shifted = raw.copy()
    shifted["Time"] = shifted["Time"] + 86_400.0 * 3

    pd.testing.assert_frame_equal(build_features(raw), build_features(shifted))


# --------------------------------------------------------------------------
# TC-06 — dữ liệu vào không hợp lệ
# --------------------------------------------------------------------------

def test_tc06_missing_columns_are_named():
    raw = make_raw().drop(columns=["V13", "V27"])

    with pytest.raises(MissingFeatureError) as excinfo:
        build_features(raw)

    assert set(excinfo.value.missing) == {"V13", "V27"}
    assert "V13" in str(excinfo.value)


def test_tc06_extra_columns_are_ignored():
    raw = make_raw()
    raw["Class"] = 0
    raw["khong_lien_quan"] = "x"

    features = build_features(raw)

    assert list(features.columns) == FEATURE_ORDER


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [("Amount", -1.0), ("Time", -5.0), ("V5", np.nan), ("V9", np.inf)],
)
def test_tc06_invalid_values_rejected_when_validating(column, bad_value):
    raw = make_raw()
    raw.loc[0, column] = bad_value

    assert validate_raw(raw), f"{column}={bad_value} lẽ ra phải bị bắt lỗi"
    with pytest.raises(InvalidFeatureValueError):
        build_features(raw, validate=True)


def test_tc06_clean_frame_has_no_problems():
    assert validate_raw(make_raw(n_rows=50)) == []


def test_raw_contract_has_thirty_columns():
    """Hợp đồng dữ liệu vào là 30 cột; đặc trưng đầu ra là 31 (02 §4, §5)."""
    assert len(RAW_REQUIRED_COLUMNS) == 30
