"""Autoencoder phát hiện bất thường — phương án tùy chọn FR-11 (04 §8).

Mô hình chỉ học "thế nào là bình thường": huấn luyện trên giao dịch hợp lệ của
tập train, rồi dùng sai số tái tạo làm điểm rủi ro. Nó cố ý KHÔNG dùng nhãn gian
lận nào — đó vừa là điểm mạnh (bắt được kiểu gian lận chưa từng thấy) vừa là lý
do nó thua mô hình có giám sát trên bộ dữ liệu này.

Môi trường dự án không có PyTorch, nên mạng 30 → 16 → 8 → 16 → 30 dựng bằng
``MLPRegressor`` của scikit-learn: ReLU ở các tầng ẩn, đầu ra tuyến tính, hàm
mất mát bình phương — đúng thiết kế ở 04 §8.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from .config import RANDOM_STATE
from .features import AMOUNT_COLUMN

#: Kiến trúc ở 04 §8 (tầng vào/ra là số đặc trưng)
HIDDEN_LAYERS = (16, 8, 16)


class ReconstructionScorer(BaseEstimator):
    """Autoencoder với giao diện giống bộ phân loại: ``fit`` rồi ``score_samples``.

    Tiền xử lý nằm trong ``fit`` để chỉ học từ dữ liệu được truyền vào (ML-04):
    ``Amount`` lấy ``log1p`` vì lệch phải rất mạnh, sau đó mọi cột chuẩn hoá theo
    trung bình và độ lệch chuẩn của chính tập huấn luyện đó.
    """

    def __init__(
        self,
        hidden_layer_sizes=HIDDEN_LAYERS,
        *,
        max_iter: int = 100,
        batch_size: int = 256,
        learning_rate_init: float = 1e-3,
        random_state: int = RANDOM_STATE,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state

    @staticmethod
    def _to_matrix(X) -> np.ndarray:
        frame = pd.DataFrame(X).copy()
        if AMOUNT_COLUMN in frame.columns:
            frame[AMOUNT_COLUMN] = np.log1p(frame[AMOUNT_COLUMN].clip(lower=0))
        return frame.to_numpy(dtype="float64")

    def _standardize(self, X) -> np.ndarray:
        return (self._to_matrix(X) - self.mean_) / self.std_

    def fit(self, X, y=None):
        """Học tái tạo ``X``. Truyền vào CHỈ giao dịch hợp lệ; ``y`` bị bỏ qua."""
        matrix = self._to_matrix(X)
        self.mean_ = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        self.std_ = np.where(std > 0, std, 1.0)
        standardized = (matrix - self.mean_) / self.std_

        self.network_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.network_.fit(standardized, standardized)
        return self

    def score_samples(self, X) -> np.ndarray:
        """Sai số tái tạo bình phương trung bình của từng dòng — càng cao càng bất thường."""
        standardized = self._standardize(X)
        reconstructed = self.network_.predict(standardized)
        return ((reconstructed - standardized) ** 2).mean(axis=1)

    def feature_errors(self, X) -> pd.DataFrame:
        """Sai số tái tạo theo từng cột — cột nào mô hình "không quen" nhất."""
        standardized = self._standardize(X)
        reconstructed = self.network_.predict(standardized)
        columns = list(pd.DataFrame(X).columns)
        return pd.DataFrame((reconstructed - standardized) ** 2, columns=columns)
