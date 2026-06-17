"""Statistical anomaly detectors.

All detectors expose the common ``fit_score(X) -> pd.Series`` interface.
ARIMA and STL are univariate techniques, so they are fit per numeric column
and aggregated by taking the max standardized residual across columns
(consistent with the column-max convention used by Rolling Z-Score).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import AnomalyDetector


def _standardize_abs(resid: np.ndarray) -> np.ndarray:
    """Absolute residual scaled by its std so columns are comparable."""
    r = np.abs(np.asarray(resid, dtype=float))
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = r.std()
    if sigma < 1e-12:
        return r
    return r / sigma


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


class ARIMADetector(AnomalyDetector):
    """Per-column ARIMA; anomaly score = magnitude of the forecast residual."""

    name = "ARIMA"

    def __init__(
        self,
        contamination: float = 0.05,
        order: tuple[int, int, int] = (1, 1, 1),
        **kwargs,
    ):
        super().__init__(contamination=contamination, **kwargs)
        self.order = order

    def fit_score(self, X: pd.DataFrame) -> pd.Series:
        from statsmodels.tsa.arima.model import ARIMA

        n = len(X)
        scores = np.zeros(n)
        d = self.order[1]
        for col in X.columns:
            series = X[col].astype(float).to_numpy()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ARIMA(series, order=self.order).fit()
                resid = np.asarray(res.resid, dtype=float)
            except Exception:
                # fallback: first-difference residual
                resid = np.concatenate([[0.0], np.diff(series)])
            # residuals from the initial differencing are unreliable
            if d > 0 and len(resid) > d:
                resid[:d] = 0.0
            scores = np.maximum(scores, _standardize_abs(resid)[:n])
        return pd.Series(scores, index=X.index, name="score")


class STLDetector(AnomalyDetector):
    """Per-column STL decomposition; anomaly score = magnitude of the remainder."""

    name = "STL"

    def __init__(self, contamination: float = 0.05, period: int = 24, **kwargs):
        super().__init__(contamination=contamination, **kwargs)
        self.period = period

    def fit_score(self, X: pd.DataFrame) -> pd.Series:
        from statsmodels.tsa.seasonal import STL

        n = len(X)
        # STL needs at least two full cycles and an integer period >= 2
        period = max(2, min(int(self.period), n // 2))
        scores = np.zeros(n)
        for col in X.columns:
            series = X[col].astype(float).to_numpy()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = STL(series, period=period, robust=True).fit()
                resid = np.asarray(result.resid, dtype=float)
            except Exception:
                # fallback: deviation from a centered rolling mean
                s = pd.Series(series)
                resid = (
                    s - s.rolling(period, min_periods=1, center=True).mean()
                ).to_numpy()
            scores = np.maximum(scores, _standardize_abs(resid)[:n])
        return pd.Series(scores, index=X.index, name="score")
