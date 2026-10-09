"""Demo Streamlit (phương án A, dự phòng): chấm điểm rủi ro gian lận.

Chạy: streamlit run app.py

Dùng đúng hiện vật mà API dùng — ``models/model.joblib`` (pipeline hoàn chỉnh), τ* trong
``models/threshold.json`` và thư viện mẫu ``data/sample_pool.json`` — nên không cần
PostgreSQL, Docker hay ``creditcard.csv``. Đặc trưng sinh bằng ``build_features`` như lúc
huấn luyện (ranh giới chống lệch train/serve, 03 §4).
"""

import json

import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH, SAMPLE_POOL_PATH, TARGET, THRESHOLD_PATH
from src.features import InvalidFeatureValueError, MissingFeatureError, build_features

CATEGORY_LABELS = {
    "fraud_easy": "Gian lận dễ",
    "fraud_hard": "Gian lận khó (ngưỡng 0,5 bỏ lọt)",
    "legit_easy": "Hợp lệ dễ",
    "legit_hard": "Hợp lệ khó (cảnh báo giả ở τ*)",
}

st.set_page_config(page_title="Fraud Detection Demo", page_icon="💳", layout="wide")


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_threshold() -> dict | None:
    if not THRESHOLD_PATH.exists():
        return None
    return json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))


@st.cache_data
def get_samples() -> pd.DataFrame | None:
    if not SAMPLE_POOL_PATH.exists():
        return None
    pool = json.loads(SAMPLE_POOL_PATH.read_text(encoding="utf-8"))
    rows = [{"id": i["id"], "category": i["category"], "label": i["label"], "description": i["description"],
             **i["features"]} for i in pool["items"]]
    return pd.DataFrame(rows)


def score(model, raw: pd.DataFrame) -> pd.Series:
    """Điểm rủi ro của dữ liệu thô 30 cột — cùng đường với API (build_features → pipeline)."""
    return pd.Series(model.predict_proba(build_features(raw, validate=True))[:, 1], index=raw.index)


st.title("💳 Phát hiện gian lận thẻ tín dụng")

model, threshold_info = get_model(), get_threshold()
if model is None or threshold_info is None:
    st.error(
        f"Chưa có hiện vật `{MODEL_PATH.name}` / `{THRESHOLD_PATH.name}` trong `models/`. "
        "Chạy notebooks/08_export_artifacts.ipynb hoặc giải nén gói hiện vật (README, bước 3)."
    )
    st.stop()

tau_star = float(threshold_info["default_threshold"])
threshold = st.sidebar.number_input(
    "Ngưỡng cảnh báo τ", min_value=0.0001, max_value=0.9999, value=tau_star, step=0.001, format="%.4f",
)
st.sidebar.caption(
    f"Mặc định τ* = {tau_star:.4f}: cực tiểu chi phí kỳ vọng, chọn trên out-of-fold "
    f"(bỏ lọt {threshold_info['cost_false_negative']:g}, thẩm định {threshold_info['cost_false_positive']:g} "
    "mỗi cảnh báo). Hạ ngưỡng: bắt thêm gian lận, thêm cảnh báo giả."
)
st.sidebar.caption(f"Mô hình `{threshold_info['model_version']}`, huấn luyện {threshold_info['trained_at']}.")

tab_single, tab_batch = st.tabs(["Giao dịch mẫu", "Tải file CSV"])

with tab_single:
    samples = get_samples()
    if samples is None:
        st.info(f"Không có `{SAMPLE_POOL_PATH.name}` — hãy dùng tab tải CSV.")
    else:
        category = st.selectbox("Nhóm mẫu", list(CATEGORY_LABELS), format_func=CATEGORY_LABELS.get)
        group = samples[samples["category"] == category].reset_index(drop=True)
        idx = st.number_input(f"Chọn giao dịch (0–{len(group) - 1})", min_value=0, max_value=len(group) - 1,
                              value=0, step=1)
        row = group.iloc[[int(idx)]]

        p = float(score(model, row).iloc[0])
        col1, col2, col3 = st.columns(3)
        col1.metric("Điểm rủi ro", f"{p:.4f}")
        col2.metric("Kết luận", "🚨 Cảnh báo" if p >= threshold else "✅ Cho qua")
        col3.metric("Nhãn thật", "Gian lận" if row["label"].iloc[0] == 1 else "Hợp lệ")
        st.caption(f"{row['id'].iloc[0]} — {row['description'].iloc[0]}")
        st.caption("V1–V28 là thành phần chính sau PCA: chúng không mang nghĩa nghiệp vụ trực tiếp.")
        with st.expander("Chi tiết đặc trưng"):
            st.dataframe(build_features(row).T, width="stretch")

with tab_batch:
    uploaded = st.file_uploader("CSV cùng định dạng creditcard.csv (30 cột Time, V1–V28, Amount)", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        try:
            df["risk_score"] = score(model, df)
        except (MissingFeatureError, InvalidFeatureValueError) as exc:
            st.error(f"Tệp không đúng hợp đồng dữ liệu: {exc}")
            st.stop()
        df["alert"] = (df["risk_score"] >= threshold).astype(int)

        n_flagged = int(df["alert"].sum())
        st.write(f"Cảnh báo **{n_flagged}** / {len(df)} giao dịch ({n_flagged / len(df):.2%}) ở τ = {threshold:.4f}.")
        if TARGET in df.columns:
            caught = int(((df["alert"] == 1) & (df[TARGET] == 1)).sum())
            st.write(f"Bắt được {caught} / {int(df[TARGET].sum())} vụ gian lận có nhãn trong tệp.")
        st.dataframe(df.sort_values("risk_score", ascending=False).head(100), width="stretch")
