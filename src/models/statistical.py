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


def estimate_period(X: pd.DataFrame, fallback: int = 24) -> int:
    """Estimate the dominant seasonal period from the data.

    Averages each column's autocorrelation function and returns the lag of the
    strongest non-trivial peak (a peak higher than both neighbours). This makes
    STL's seasonality data-driven instead of hard-coding "24 = hourly/daily".
    Returns ``fallback`` when no clear periodicity is found.
    """
    n = len(X)
    if n < 8:
        return max(2, min(fallback, n // 2 or 2))
    max_lag = min(n // 2, 500)
    acf_sum = np.zeros(max_lag + 1)
    for col in X.columns:
        s = X[col].astype(float).to_numpy()
        s = s - s.mean()
        denom = np.dot(s, s)
        if denom < 1e-12:
            continue
        full = np.correlate(s, s, mode="full")[len(s) - 1 :]
        acf_sum[: len(full[: max_lag + 1])] += (full / denom)[: max_lag + 1]

    best_lag, best_val = fallback, -np.inf
    for lag in range(2, max_lag):
        v = acf_sum[lag]
        if v > acf_sum[lag - 1] and v >= acf_sum[lag + 1] and v > best_val:
            best_lag, best_val = lag, v
    if best_val <= 0:
        return max(2, min(fallback, n // 2 or 2))
    return max(2, min(best_lag, n // 2))


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

    def fit_score(self, X: pd.DataFrame, n_train: int | None = None) -> pd.Series:
        # rolling z-score is local and non-parametric: each point is compared to
        # its own trailing window, so there is no global fit to leak. n_train is
        # accepted for interface uniformity but not needed here.
        w = max(5, min(self.window, len(X) // 2))
        roll = X.rolling(window=w, min_periods=max(2, w // 2))
        mu = roll.mean()
        sigma = roll.std(ddof=0).replace(0, 1e-9)
        z = ((X - mu) / sigma).abs().fillna(0)
        return z.max(axis=1).rename("score")


_AUTO_ORDER_GRID = [
    (1, 1, 1), (0, 1, 1), (1, 1, 0), (2, 1, 1),
    (1, 0, 0), (2, 0, 2), (0, 1, 2),
]


def auto_arima_order(
    series: np.ndarray, grid: list[tuple[int, int, int]] | None = None
) -> tuple[int, int, int]:
    """Pick the ARIMA (p,d,q) order with the lowest AIC over a small grid.

    Replaces the hard-coded (1,1,1). The search is capped to the last 800
    points for speed; it is run on the *training* series by the caller so the
    chosen order is not informed by the test period.
    """
    from statsmodels.tsa.arima.model import ARIMA

    grid = grid or _AUTO_ORDER_GRID
    s = series[-800:] if len(series) > 800 else series
    best, best_aic = (1, 1, 1), np.inf
    for order in grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aic = ARIMA(s, order=order).fit().aic
            if np.isfinite(aic) and aic < best_aic:
                best, best_aic = order, aic
        except Exception:
            continue
    return best


class ARIMADetector(AnomalyDetector):
    """Per-column ARIMA; anomaly score = magnitude of the forecast residual."""

    name = "ARIMA"

    def __init__(
        self,
        contamination: float = 0.05,
        order: tuple[int, int, int] | str = (1, 1, 1),
        **kwargs,
    ):
        super().__init__(contamination=contamination, **kwargs)
        self.order = order  # tuple, or "auto" for per-column AIC selection

    def fit_score(self, X: pd.DataFrame, n_train: int | None = None) -> pd.Series:
        from statsmodels.tsa.arima.model import ARIMA

        n = len(X)
        nt = n if n_train is None else min(n_train, n)
        scores = np.zeros(n)
        for col in X.columns:
            series = X[col].astype(float).to_numpy()
            train = series[:nt]
            order = auto_arima_order(train) if self.order == "auto" else self.order
            d = order[1]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ARIMA(train, order=order).fit()
                    if nt < n:
                        # extend with the held-out tail using the *fitted*
                        # parameters (no refit) → out-of-sample residuals
                        res = res.append(series[nt:], refit=False)
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

    def __init__(self, contamination: float = 0.05, period: int | str = 24, **kwargs):
        super().__init__(contamination=contamination, **kwargs)
        self.period = period

    def fit_score(self, X: pd.DataFrame, n_train: int | None = None) -> pd.Series:
        from statsmodels.tsa.seasonal import STL

        n = len(X)
        # STL is a whole-series decomposition with no fit/predict split, so it
        # is applied over the full series; n_train is accepted for interface
        # uniformity. Seasonality is estimated from the data when period="auto".
        if self.period in ("auto", None):
            base_period = estimate_period(X)
        else:
            base_period = int(self.period)
        # STL needs at least two full cycles and an integer period >= 2
        period = max(2, min(base_period, n // 2))
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
