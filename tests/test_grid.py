"""Kiểm thử lưới 5 chiến lược × 4 mô hình (FR-05, 04 §3.3).

Lưới thật chạy 1–3 giờ và chỉ chạy đúng một lần, nên không có cơ hội phát hiện
lỗi bằng cách chạy lại. Những gì kiểm được trên dữ liệu tổng hợp nhỏ thì phải
kiểm ở đây trước.

Ca quan trọng nhất là ``test_chi_so_theo_fold_khop_cross_validate``:
``run_grid`` chỉ huấn luyện MỘT lượt rồi suy chỉ số từng fold ra từ điểm
out-of-fold, thay vì gọi ``cross_validate`` rồi ``cross_val_predict`` thành hai
lượt. Ca đó chốt rằng phép rút gọn ấy cho đúng con số cũ.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import cross_validate

from src.config import DEFAULT_COST_FN, DEFAULT_COST_FP
from src.features import V_COLUMNS, build_features
from src.modeling import (
    GRID_COLUMNS,
    build_pipeline,
    imbalance_ratio,
    make_cv,
    run_grid,
)
from src.threshold import metrics_at_threshold

MO_HINH = ("logistic_regression", "decision_tree")
CHIEN_LUOC = ("none", "class_weight")


@pytest.fixture
def dataset():
    """600 giao dịch, 30 gian lận — đủ để StratifiedKFold(5) luôn có mẫu dương."""
    rng = np.random.default_rng(11)
    n_rows, n_fraud = 600, 30

    raw = {"Time": rng.uniform(0, 172_000, n_rows)}
    for column in V_COLUMNS:
        raw[column] = rng.normal(size=n_rows)
    raw["Amount"] = rng.lognormal(3.0, 1.2, n_rows)

    y = np.zeros(n_rows, dtype=int)
    fraud_idx = rng.choice(n_rows, size=n_fraud, replace=False)
    y[fraud_idx] = 1

    frame = pd.DataFrame(raw)
    frame.loc[fraud_idx, "V14"] -= 3.0
    frame.loc[fraud_idx, "V17"] -= 2.0

    return build_features(frame), pd.Series(y)


@pytest.fixture
def ket_qua(dataset):
    X, y = dataset
    return run_grid(X, y, models=MO_HINH, strategies=CHIEN_LUOC,
                    return_scores=True, verbose=False)


# --------------------------------------------------------------------------
# Hình dạng bảng kết quả
# --------------------------------------------------------------------------

def test_moi_to_hop_dung_mot_dong(ket_qua):
    bang, _ = ket_qua
    assert len(bang) == len(MO_HINH) * len(CHIEN_LUOC)
    assert set(map(tuple, bang[["model", "strategy"]].to_numpy())) == {
        (m, s) for m in MO_HINH for s in CHIEN_LUOC
    }


def test_cot_dung_thu_tu_cua_04_muc_3_3(ket_qua):
    """Bảng phải có sẵn Recall@τ* và Precision@τ*, không để notebook tự thêm."""
    bang, _ = ket_qua
    assert list(bang.columns) == GRID_COLUMNS
    for cot in ("pr_auc_mean", "pr_auc_std", "roc_auc_mean",
                "threshold", "recall", "precision", "f1", "total_seconds"):
        assert cot in bang.columns


def test_sap_theo_pr_auc_giam_dan(ket_qua):
    bang, _ = ket_qua
    assert np.all(np.diff(bang["pr_auc_mean"].to_numpy()) <= 1e-12)


def test_chi_so_nam_trong_mien_hop_le(ket_qua):
    bang, _ = ket_qua
    for cot in ("pr_auc_mean", "roc_auc_mean", "recall", "precision", "f1", "threshold"):
        assert bang[cot].between(0, 1).all(), cot
    assert (bang["pr_auc_std"] >= 0).all()
    assert (bang["total_seconds"] > 0).all()
    assert (bang["n_distinct_scores"] >= 1).all()


# --------------------------------------------------------------------------
# Một lượt huấn luyện phải cho đúng con số của hai lượt
# --------------------------------------------------------------------------

def test_chi_so_theo_fold_khop_cross_validate(dataset):
    """Chốt phép rút gọn: suy từ điểm OOF ra phải trùng cross_validate.

    Nếu ca này đỏ thì bảng 20 dòng đang báo cáo một thứ khác với cách tính
    chuẩn của sklearn, và toàn bộ phần so sánh chiến lược mất giá trị.
    """
    X, y = dataset
    bang = run_grid(X, y, models=("logistic_regression",), strategies=("none",),
                    verbose=False)

    cv = make_cv()
    pipeline = build_pipeline("logistic_regression", "none", y=y)
    tham_chieu = cross_validate(
        pipeline, X, y, cv=cv,
        scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
    )

    assert bang.loc[0, "pr_auc_mean"] == pytest.approx(
        float(np.mean(tham_chieu["test_pr_auc"])), rel=1e-9)
    assert bang.loc[0, "pr_auc_std"] == pytest.approx(
        float(np.std(tham_chieu["test_pr_auc"])), rel=1e-9)
    assert bang.loc[0, "roc_auc_mean"] == pytest.approx(
        float(np.mean(tham_chieu["test_roc_auc"])), rel=1e-9)


# --------------------------------------------------------------------------
# Điểm out-of-fold
# --------------------------------------------------------------------------

def test_tra_ve_diem_oof_cho_tung_to_hop(ket_qua, dataset):
    """T-21 cần điểm của 5 chiến lược để vẽ chồng đường PR — không được bắt chạy lại."""
    X, y = dataset
    _, diem = ket_qua

    assert set(diem) == {f"{m}__{s}" for m in MO_HINH for s in CHIEN_LUOC}
    for mang in diem.values():
        assert mang.shape == (len(X),)
        assert np.all((mang >= 0) & (mang <= 1))
        assert np.isfinite(mang).all()


def test_khong_yeu_cau_diem_thi_chi_tra_bang(dataset):
    X, y = dataset
    ket_qua = run_grid(X, y, models=("decision_tree",), strategies=("none",),
                       verbose=False)
    assert isinstance(ket_qua, pd.DataFrame)


# --------------------------------------------------------------------------
# Ngưỡng chọn trên dữ liệu out-of-fold (ML-08)
# --------------------------------------------------------------------------

def test_recall_precision_khop_nguong_da_chon(ket_qua, dataset):
    """Ba cột recall/precision/f1 phải đúng là chỉ số tại τ* ghi cùng dòng."""
    _, y = dataset
    bang, diem = ket_qua

    for _, dong in bang.iterrows():
        oof = diem[f"{dong['model']}__{dong['strategy']}"]
        met = metrics_at_threshold(
            y, oof, dong["threshold"],
            cost_fn=DEFAULT_COST_FN, cost_fp=DEFAULT_COST_FP, sample_fraction=0.8,
        )
        assert met.recall == pytest.approx(dong["recall"])
        assert met.precision == pytest.approx(dong["precision"])
        assert met.f1 == pytest.approx(dong["f1"])
        assert met.tp == dong["tp"]
        assert met.fp == dong["fp"]


def test_nguong_toi_uu_chi_phi_khong_te_hon_nguong_mac_dinh(ket_qua, dataset):
    """τ* chọn theo chi phí thì chi phí tại đó phải ≤ chi phí tại 0,5."""
    _, y = dataset
    bang, diem = ket_qua

    for _, dong in bang.iterrows():
        oof = diem[f"{dong['model']}__{dong['strategy']}"]
        mac_dinh = metrics_at_threshold(y, oof, 0.5, sample_fraction=0.8)
        assert dong["expected_cost"] <= mac_dinh.expected_cost + 1e-9


def test_doi_ty_le_chi_phi_lam_nguong_dich_chuyen(dataset):
    """FN đắt hơn thì ngưỡng phải hạ xuống để bắt thêm — kiểm hướng, không kiểm trị."""
    X, y = dataset
    chung = dict(models=("logistic_regression",), strategies=("none",), verbose=False)

    re_ = run_grid(X, y, cost_fn=10.0, cost_fp=10.0, **chung)
    dat = run_grid(X, y, cost_fn=1000.0, cost_fp=1.0, **chung)

    assert dat.loc[0, "threshold"] <= re_.loc[0, "threshold"]
    assert dat.loc[0, "recall"] >= re_.loc[0, "recall"]


# --------------------------------------------------------------------------
# Điểm lưu và chạy nối tiếp
# --------------------------------------------------------------------------

def test_ghi_diem_luu_sau_moi_to_hop(dataset, tmp_path):
    X, y = dataset
    duong_dan = tmp_path / "grid_results.csv"

    bang = run_grid(X, y, models=MO_HINH, strategies=("none",),
                    checkpoint_path=duong_dan, verbose=False)

    assert duong_dan.exists()
    assert duong_dan.with_suffix(".npz").exists()
    da_luu = pd.read_csv(duong_dan)
    assert len(da_luu) == len(bang) == len(MO_HINH)


def test_chay_noi_tiep_bo_qua_to_hop_da_xong(dataset, tmp_path):
    """Lưới 1–3 giờ mà kernel chết giữa chừng thì phải chạy tiếp được, không làm lại."""
    X, y = dataset
    duong_dan = tmp_path / "grid_results.csv"

    run_grid(X, y, models=("logistic_regression",), strategies=("none",),
             checkpoint_path=duong_dan, verbose=False)
    lan_dau = pd.read_csv(duong_dan)

    day_du = run_grid(X, y, models=MO_HINH, strategies=("none",),
                      checkpoint_path=duong_dan, verbose=False)

    assert len(day_du) == len(MO_HINH)
    # dòng đã chạy phải giữ nguyên số cũ, không bị huấn luyện lại
    cu = day_du[day_du["model"] == "logistic_regression"].iloc[0]
    assert cu["pr_auc_mean"] == pytest.approx(lan_dau.loc[0, "pr_auc_mean"])
    assert cu["total_seconds"] == pytest.approx(lan_dau.loc[0, "total_seconds"])


def test_resume_tat_thi_chay_lai_tu_dau(dataset, tmp_path):
    X, y = dataset
    duong_dan = tmp_path / "grid_results.csv"

    run_grid(X, y, models=("logistic_regression",), strategies=("none",),
             checkpoint_path=duong_dan, verbose=False)
    lai = run_grid(X, y, models=("logistic_regression",), strategies=("none",),
                   checkpoint_path=duong_dan, resume=False, verbose=False)

    assert len(lai) == 1


# --------------------------------------------------------------------------
# Kích thước lưới chuẩn
# --------------------------------------------------------------------------

def test_luoi_mac_dinh_dung_20_to_hop():
    """T-20: reports/grid_results.csv phải có đúng 20 dòng."""
    from src.modeling import DEFAULT_MODELS, DEFAULT_STRATEGIES

    assert len(DEFAULT_MODELS) * len(DEFAULT_STRATEGIES) == 20


def test_scale_pos_weight_bang_ty_le_mat_can_bang(dataset):
    """T-22 so sánh scale_pos_weight với SMOTE, nên trọng số phải đúng định nghĩa."""
    X, y = dataset
    pipeline = build_pipeline("xgboost", "class_weight", y=y)
    assert pipeline.named_steps["clf"].scale_pos_weight == pytest.approx(imbalance_ratio(y))


def test_model_params_ghi_de_theo_tung_mo_hinh(dataset):
    """Phương án dự phòng của T-20: hạ n_estimators cho riêng mô hình ensemble."""
    X, y = dataset
    bang = run_grid(X, y, models=("random_forest", "logistic_regression"),
                    strategies=("none",),
                    model_params={"random_forest": {"n_estimators": 20}},
                    verbose=False)

    assert len(bang) == 2
    # LogisticRegression không nhận n_estimators; nếu tham số bị truyền nhầm sang
    # nó thì dòng này đã ném TypeError trước khi tới đây.
    assert set(bang["model"]) == {"random_forest", "logistic_regression"}


def test_model_params_thuc_su_doi_cau_hinh(dataset):
    X, y = dataset
    nhanh = run_grid(X, y, models=("random_forest",), strategies=("none",),
                     model_params={"random_forest": {"n_estimators": 10}}, verbose=False)
    cham = run_grid(X, y, models=("random_forest",), strategies=("none",), verbose=False)

    assert nhanh.loc[0, "total_seconds"] < cham.loc[0, "total_seconds"]


def test_noi_tiep_toan_bo_van_tra_du_diem_oof(dataset, tmp_path):
    """Đường đi thật của người dùng: chạy script trước, mở notebook sau.

    Khi mọi tổ hợp đã có trong điểm lưu, run_grid không huấn luyện gì nữa nhưng
    vẫn phải trả đủ điểm out-of-fold, nếu không thì hình PR của T-21 không vẽ được.
    """
    X, y = dataset
    duong_dan = tmp_path / "grid_results.csv"

    run_grid(X, y, models=MO_HINH, strategies=CHIEN_LUOC,
             checkpoint_path=duong_dan, verbose=False)

    bang, diem = run_grid(X, y, models=MO_HINH, strategies=CHIEN_LUOC,
                          checkpoint_path=duong_dan, return_scores=True, verbose=False)

    assert len(bang) == len(MO_HINH) * len(CHIEN_LUOC)
    assert set(diem) == {f"{m}__{s}" for m in MO_HINH for s in CHIEN_LUOC}
    for mang in diem.values():
        assert mang.shape == (len(X),)


def test_cot_thua_trong_diem_luu_khong_lam_gay(dataset, tmp_path):
    """Notebook ghi thêm cột `che_do` vào CSV; lần chạy sau phải bỏ qua được."""
    X, y = dataset
    duong_dan = tmp_path / "grid_results.csv"

    run_grid(X, y, models=("logistic_regression",), strategies=("none",),
             checkpoint_path=duong_dan, verbose=False)

    bang = pd.read_csv(duong_dan)
    bang["che_do"] = "đầy đủ"
    bang.to_csv(duong_dan, index=False, encoding="utf-8")

    lai = run_grid(X, y, models=("logistic_regression",), strategies=("none",),
                   checkpoint_path=duong_dan, verbose=False)
    assert list(lai.columns) == GRID_COLUMNS
