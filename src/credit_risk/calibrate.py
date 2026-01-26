from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression

@dataclass
class PlattCalibrator:
    eps: float = 1e-6

    def __post_init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def _to_logit(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, self.eps, 1 - self.eps)
        return np.log(p / (1 - p))

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        z = self._to_logit(p_raw).reshape(-1, 1)
        self.model.fit(z, y.astype(int))
        return self

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        z = self._to_logit(p_raw).reshape(-1, 1)
        return self.model.predict_proba(z)[:, 1]
