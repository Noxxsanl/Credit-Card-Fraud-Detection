"""Demo Streamlit: chấm điểm rủi ro gian lận cho một giao dịch.

Chạy: streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data import TARGET, load_data

MODEL_PATH = Path(__file__).parent / "models" / "best_model.pkl"
DEFAULT_THRESHOLD = 0.5

st.set_page_config(page_title="Fraud Detection Demo", page_icon="💳", layout="wide")


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_sample(n=200):
    """Vài dòng dữ liệu thật để người dùng thử nhanh (nếu có sẵn csv)."""
    try:
        df = load_data()
    except FileNotFoundError:
        return None
    return df.sample(n=min(n, len(df)), random_state=0).reset_index(drop=True)


def score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


st.title("💳 Phát hiện gian lận thẻ tín dụng")

model = get_model()
if model is None:
    st.error(
        f"Chưa có model tại `{MODEL_PATH}`. "
        "Chạy notebook 05_advanced_models.ipynb để huấn luyện và lưu best_model.pkl."
    )
    st.stop()

threshold = st.sidebar.slider("Ngưỡng cảnh báo", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)
st.sidebar.caption("Hạ ngưỡng để tăng recall (bắt được nhiều gian lận hơn, nhiều báo động nhầm hơn).")

tab_single, tab_batch = st.tabs(["Một giao dịch", "Tải file CSV"])

with tab_single:
    sample = get_sample()
    if sample is None:
        st.info("Không tìm thấy data/creditcard.csv — hãy dùng tab tải CSV.")
    else:
        idx = st.number_input(
            "Chọn giao dịch mẫu", min_value=0, max_value=len(sample) - 1, value=0, step=1
        )
        row = sample.iloc[[int(idx)]]
        features = row.drop(columns=[TARGET], errors="ignore")

        p = float(score(model, features)[0])
        col1, col2 = st.columns(2)
        col1.metric("Xác suất gian lận", f"{p:.4f}")
        col2.metric("Kết luận", "🚨 GIAN LẬN" if p >= threshold else "✅ Hợp lệ")

        if TARGET in row.columns:
            st.caption(f"Nhãn thật: {int(row[TARGET].iloc[0])}")
        with st.expander("Chi tiết đặc trưng"):
            st.dataframe(features.T, use_container_width=True)

with tab_batch:
    uploaded = st.file_uploader("CSV cùng định dạng creditcard.csv", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        features = df.drop(columns=[TARGET], errors="ignore")
        df["fraud_proba"] = score(model, features)
        df["prediction"] = (df["fraud_proba"] >= threshold).astype(int)

        n_flagged = int(df["prediction"].sum())
        st.write(f"Đã gắn cờ **{n_flagged}** / {len(df)} giao dịch ({n_flagged / len(df):.2%}).")
        st.dataframe(
            df.sort_values("fraud_proba", ascending=False).head(100),
            use_container_width=True,
        )
