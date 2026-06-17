"""Statistical anomaly detectors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AnomalyDetector


class ZScoreDetector(AnomalyDetector):
    name = "Z-Score"

    def fit_score(self, X: pd.DataFrame) -> pd.Series:
        mu = X.mean()
        sigma = X.std(ddof=0).replace(0, 1e-9)
        z = ((X - mu) / sigma).abs()
        return z.max(axis=1).rename("score")


class IQRDetector(AnomalyDetector):
    name = "IQR"

    def fit_score(self, X: pd.DataFrame) -> pd.Series:
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = (q3 - q1).replace(0, 1e-9)
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        dist = pd.DataFrame(index=X.index)
        for col in X.columns:
            d = np.maximum(lower[col] - X[col], X[col] - upper[col])
            dist[col] = np.maximum(d, 0) / iqr[col]
        return dist.max(axis=1).rename("score")


class RollingZDetector(AnomalyDetector):
    name = "Rolling Z-Score"

    def __init__(self, contamination: float = 0.05, window: int = 30, **kwargs):
        super().__init__(contamination=contamination, **kwargs)
        self.window = window

    def fit_score(self, X: pd.DataFrame) -> pd.Series:
        w = max(5, min(self.window, len(X) // 2))
        roll = X.rolling(window=w, min_periods=max(2, w // 2))
        mu = roll.mean()
        sigma = roll.std(ddof=0).replace(0, 1e-9)
        z = ((X - mu) / sigma).abs().fillna(0)
        return z.max(axis=1).rename("score")
