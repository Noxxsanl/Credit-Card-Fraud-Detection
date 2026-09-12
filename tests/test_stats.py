"""Kiểm thử tầng thống kê (FR-03, 04 §5.4).

Bảng xếp hạng ở `reports/feature_ranking.csv` là đầu vào cho phần đối chiếu với
SHAP ở notebook 07. Nếu hiệu chỉnh BH hay công thức effect size sai, cả hai
notebook cùng sai theo mà không có gì báo động — nên phần logic đó được chốt ở
đây bằng các ví dụ tính tay được.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import RAW_REQUIRED_COLUMNS, V_COLUMNS
from src.stats import (
    benjamini_hochberg,
    cliff_label,
    cliffs_delta_from_u,
    cohens_d,
    compare_groups,
    effect_size_label,
    feature_ranking,
    pearson_from_cohens_d,
    significance_summary,
)


def make_frame(n_legit: int = 400, n_fraud: int = 40, *, seed: int = 0) -> pd.DataFrame:
    """Khung dữ liệu tổng hợp đúng lược đồ thô, có tín hiệu cài sẵn ở V1 và V2."""
    rng = np.random.default_rng(seed)
    n = n_legit + n_fraud
    labels = np.zeros(n, dtype=int)
    labels[:n_fraud] = 1

    data = {"Time": rng.uniform(0, 172_792, size=n)}
    for column in V_COLUMNS:
        data[column] = rng.normal(size=n)
    data["Amount"] = rng.uniform(0, 500, size=n)

    frame = pd.DataFrame(data)
    # V1 tách rất mạnh, V2 tách vừa phải, phần còn lại là nhiễu
    frame.loc[labels == 1, "V1"] += 5.0
    frame.loc[labels == 1, "V2"] -= 1.0
    frame["Class"] = labels
    return frame


# --------------------------------------------------------------------------
# Benjamini–Hochberg
# --------------------------------------------------------------------------

def test_bh_khop_vi_du_tinh_tay():
    """Ví dụ chuẩn: m = 4, hiệu chỉnh là p·m/i rồi lấy cực tiểu tích luỹ."""
    p = np.array([0.01, 0.02, 0.03, 0.04])
    # thô: 0,04 · 4/1 = 0,16 | 0,03 · 4/2 = 0,06 | 0,02 · 4/3 = 0,0267 | 0,01 · 4/4 = 0,04
    # cực tiểu tích luỹ từ phải: 0,04 → 0,04 → 0,0267 → 0,04 giữ nguyên 0,04
    expected = np.array([0.04, 0.04, 0.04, 0.04])
    np.testing.assert_allclose(benjamini_hochberg(p), expected, rtol=1e-12)


def test_bh_khong_bao_gio_nho_hon_p_tho_va_don_dieu():
    rng = np.random.default_rng(42)
    p = rng.uniform(0, 1, size=50)
    adjusted = benjamini_hochberg(p)

    assert np.all(adjusted >= p - 1e-12), "p hiệu chỉnh phải ≥ p thô"
    assert np.all(adjusted <= 1.0)

    order = np.argsort(p)
    assert np.all(np.diff(adjusted[order]) >= -1e-12), "phải đơn điệu theo p thô"


def test_bh_giu_dung_thu_tu_dau_vao():
    p = np.array([0.9, 0.001, 0.5])
    adjusted = benjamini_hochberg(p)
    assert adjusted.argmin() == 1, "phần tử nhỏ nhất phải ở đúng vị trí ban đầu"


def test_bh_p_bang_nhau_giu_nguyen_gia_tri():
    """10 p-value bằng nhau → cực tiểu tích luỹ rơi vào hạng cuối (m/m = 1).

    Đây là chỗ BH khác hẳn Bonferroni: Bonferroni nhân tất cả với 10 thành 0,3
    và loại sạch, BH giữ nguyên 0,03 và giữ lại cả 10.
    """
    adjusted = benjamini_hochberg(np.full(10, 0.03))
    np.testing.assert_allclose(adjusted, np.full(10, 0.03), rtol=1e-12)


def test_bh_loai_bot_so_ket_qua_co_y_nghia():
    """Đây là lý do phải hiệu chỉnh: p thô sát 0,05 không sống sót."""
    p = np.array([0.001, 0.04, 0.045, 0.048, 0.30])
    adjusted = benjamini_hochberg(p)
    assert (p < 0.05).sum() == 4
    assert (adjusted < 0.05).sum() == 1


def test_bh_tu_choi_dau_vao_hong():
    with pytest.raises(ValueError):
        benjamini_hochberg(np.array([0.1, np.nan]))
    with pytest.raises(ValueError):
        benjamini_hochberg(np.array([[0.1, 0.2], [0.3, 0.4]]))
    assert benjamini_hochberg(np.array([])).size == 0


# --------------------------------------------------------------------------
# Effect size
# --------------------------------------------------------------------------

def test_cohens_d_tinh_tay():
    """Hai nhóm cùng phương sai 1, lệch trung bình đúng 2 → d = 2."""
    x = np.array([1.0, 2.0, 3.0]) + 2.0
    y = np.array([1.0, 2.0, 3.0])
    assert cohens_d(x, y) == pytest.approx(2.0)


def test_cohens_d_doi_dau_khi_doi_cho():
    x = np.array([5.0, 6.0, 7.0, 8.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert cohens_d(x, y) == pytest.approx(-cohens_d(y, x))


def test_cohens_d_bang_khong_khi_hai_nhom_trung_nhau():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert cohens_d(values, values) == pytest.approx(0.0)


def test_cohens_d_can_it_nhat_hai_quan_sat():
    with pytest.raises(ValueError):
        cohens_d(np.array([1.0]), np.array([1.0, 2.0]))


def test_cliffs_delta_tai_hai_bien_va_diem_giua():
    n_x, n_y = 10, 20
    assert cliffs_delta_from_u(n_x * n_y, n_x, n_y) == pytest.approx(1.0)
    assert cliffs_delta_from_u(0, n_x, n_y) == pytest.approx(-1.0)
    assert cliffs_delta_from_u(n_x * n_y / 2, n_x, n_y) == pytest.approx(0.0)


def test_cliffs_delta_khop_dinh_nghia_so_cap():
    """Đối chiếu công thức từ U với phép so từng cặp trên mẫu nhỏ."""
    rng = np.random.default_rng(7)
    x = rng.normal(1.0, size=30)
    y = rng.normal(0.0, size=40)

    pairs = np.sign(x[:, None] - y[None, :])
    expected = pairs.mean()

    from scipy.stats import mannwhitneyu

    u_statistic, _ = mannwhitneyu(x, y, alternative="two-sided")
    assert cliffs_delta_from_u(u_statistic, x.size, y.size) == pytest.approx(expected)


def test_nhan_effect_size_theo_moc_quy_uoc():
    assert effect_size_label(0.1) == "không đáng kể"
    assert effect_size_label(0.3) == "nhỏ"
    assert effect_size_label(-0.6) == "trung bình"
    assert effect_size_label(8.3) == "lớn"

    assert cliff_label(0.10) == "không đáng kể"
    assert cliff_label(-0.90) == "lớn"


def test_tuong_quan_diem_nhi_phan_bi_nen_khi_lop_duong_hiem():
    """Cùng một Cohen's d, lớp dương càng hiếm thì tương quan càng bé.

    Đây chính là cái bẫy mà notebook 02 phải nêu: |r| ≈ |d|·sqrt(p·q).
    """
    d = 8.0
    r_can_bang = pearson_from_cohens_d(d, 0.5)
    r_hiem = pearson_from_cohens_d(d, 0.0017)

    assert abs(r_can_bang) > 0.9
    assert abs(r_hiem) < 0.35

    # |d|·sqrt(p·q) là chặn trên của |r|, sát dần khi d² nhỏ so với 1/(p·q)
    tran = abs(d) * np.sqrt(0.0017 * 0.9983)
    assert abs(r_hiem) < tran
    assert abs(r_hiem) == pytest.approx(tran, rel=0.10)

    with pytest.raises(ValueError):
        pearson_from_cohens_d(d, 0.0)


def test_tuong_quan_suy_ra_khop_tuong_quan_thuc_te():
    """Công thức phải tái tạo đúng hệ số Pearson tính trực tiếp.

    Sai lệch còn lại chỉ đến từ bậc tự do: Pearson dùng ddof = 0 còn độ lệch
    chuẩn gộp của Cohen's d dùng n₁ + n₀ − 2. Ở n = 2.100 sai lệch cỡ 1e-4;
    trên dữ liệu thật (n ≈ 284.000) nó xuống dưới 1e-6.
    """
    frame = make_frame(n_legit=2000, n_fraud=100, seed=3)
    row = compare_groups(frame, "V1")
    positive_rate = frame["Class"].mean()

    predicted = float(pearson_from_cohens_d(row["cohens_d"], positive_rate))
    assert predicted == pytest.approx(row["pearson_r"], rel=1e-3)


# --------------------------------------------------------------------------
# Bảng xếp hạng
# --------------------------------------------------------------------------

def test_ranking_mac_dinh_chay_dung_30_kiem_dinh():
    ranking = feature_ranking(make_frame())
    assert len(ranking) == 30
    assert len(RAW_REQUIRED_COLUMNS) == 30
    assert set(ranking["feature"]) == set(RAW_REQUIRED_COLUMNS)


def test_ranking_co_du_cot_ma_notebook_07_can():
    """T-15: tệp xuất ra bắt buộc có `cohens_d` và `p_adjusted`."""
    ranking = feature_ranking(make_frame())
    for column in ("feature", "cohens_d", "p_adjusted", "cliffs_delta", "pearson_r"):
        assert column in ranking.columns


def test_ranking_sap_theo_do_lon_hieu_ung_giam_dan():
    ranking = feature_ranking(make_frame())
    values = ranking["abs_cohens_d"].to_numpy()
    assert np.all(np.diff(values) <= 1e-12)
    assert ranking["rank_cohens_d"].tolist() == list(range(1, 31))


def test_ranking_tim_ra_dac_trung_co_tin_hieu_cai_san():
    ranking = feature_ranking(make_frame()).set_index("feature")
    assert ranking.loc["V1", "rank_cohens_d"] == 1
    assert ranking.loc["V1", "cohens_d"] > 0
    assert ranking.loc["V2", "cohens_d"] < 0
    assert ranking.loc["V1", "significant"]
    assert ranking.loc["V2", "significant"]


def test_ranking_danh_dau_dac_trung_nao_thuc_su_vao_mo_hinh():
    """`Time` được kiểm định cho đủ 30, nhưng không phải đặc trưng mô hình (DS-03)."""
    ranking = feature_ranking(make_frame()).set_index("feature")
    assert not ranking.loc["Time", "in_model"]
    assert ranking.loc["V1", "in_model"]
    assert ranking.loc["Amount", "in_model"]


def test_ranking_p_hieu_chinh_luon_lon_hon_p_tho():
    ranking = feature_ranking(make_frame())
    assert np.all(ranking["p_adjusted"] >= ranking["p_value"] - 1e-12)
    assert int(ranking["significant"].sum()) == int((ranking["p_adjusted"] < 0.05).sum())


def test_ranking_bao_loi_khi_thieu_cot():
    frame = make_frame().drop(columns=["V5"])
    with pytest.raises(KeyError):
        feature_ranking(frame)

    with pytest.raises(KeyError):
        feature_ranking(make_frame().drop(columns=["Class"]))


def test_ranking_nhan_danh_sach_dac_trung_tuy_chon():
    ranking = feature_ranking(make_frame(), features=["V1", "V2", "Amount"])
    assert ranking["feature"].tolist() == ["V1", "V2", "Amount"]


def test_tom_tat_neu_dung_so_kiem_dinh_con_y_nghia():
    ranking = feature_ranking(make_frame())
    summary = significance_summary(ranking)
    assert "30 kiểm định" in summary
    assert str(int(ranking["significant"].sum())) in summary
