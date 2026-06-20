"""Preprocessing utilities: missing-value handling, scaling, transforms."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "none": None,
}


def fill_missing(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Fill missing values. method in {ffill, interpolate, drop, zero}."""
    if method == "drop":
        return df.dropna()
    if method == "zero":
        return df.fillna(0)
    if method == "interpolate":
        return df.interpolate(method="linear", limit_direction="both")
    # ffill default
    return df.ffill().bfill()


def transform(df: pd.DataFrame, kind: str = "none") -> pd.DataFrame:
    """Apply value transform. kind in {none, diff, logreturn}."""
    if kind == "diff":
        return df.diff().dropna()
    if kind == "logreturn":
        safe = df.replace(0, np.nan).ffill().bfill()
        return np.log(safe / safe.shift(1)).dropna()
    return df


def scale(
    df: pd.DataFrame, kind: str = "standard", n_train: int | None = None
) -> tuple[pd.DataFrame, object | None]:
    """Scale features. Returns (scaled_df, fitted_scaler).

    When ``n_train`` is given the scaler is *fit on the first n_train rows only*
    (the training window) and then applied to the whole series, so test-period
    statistics never leak into the normalization.
    """
    cls = SCALERS.get(kind)
    if cls is None:
        return df.copy(), None
    scaler = cls()
    fit_part = df.values if n_train is None else df.values[:n_train]
    scaler.fit(fit_part)
    arr = scaler.transform(df.values)
    return pd.DataFrame(arr, index=df.index, columns=df.columns), scaler


def add_temporal_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3),
    roll_windows: tuple[int, ...] = (5, 15),
) -> pd.DataFrame:
    """Augment each feature with lag and rolling statistics.

    Plain iid detectors (Isolation Forest, LOF, OC-SVM) ignore time order.
    By concatenating lagged values and rolling mean/std we give them an
    explicit window of temporal context, so an anomaly is judged relative to
    its recent past rather than to the global distribution. Leading rows that
    have no history are back-filled so the output keeps the original index.
    """
    feats = [df]
    for lag in lags:
        feats.append(df.shift(lag).add_suffix(f"_lag{lag}"))
    for w in roll_windows:
        w = max(2, min(w, len(df) // 2 or 2))
        roll = df.rolling(window=w, min_periods=1)
        feats.append(roll.mean().add_suffix(f"_rmean{w}"))
        feats.append(roll.std(ddof=0).add_suffix(f"_rstd{w}"))
    out = pd.concat(feats, axis=1)
    return out.bfill().ffill().fillna(0.0)


def preprocess_pipeline(
    df: pd.DataFrame,
    fill: str = "ffill",
    value_transform: str = "none",
    scaling: str = "standard",
    train_ratio: float = 1.0,
) -> tuple[pd.DataFrame, int]:
    """Run full preprocessing pipeline.

    Returns ``(scaled_df, n_train)`` where ``n_train`` is the number of leading
    rows that make up the training window (== len(df) when ``train_ratio`` >= 1,
    i.e. no holdout). The scaler is fit on that window only.
    """
    out = fill_missing(df, method=fill)
    out = transform(out, kind=value_transform)
    n = len(out)
    if train_ratio >= 1.0:
        n_train = n
        out, _ = scale(out, kind=scaling)
    else:
        n_train = min(n, max(10, int(round(train_ratio * n))))
        out, _ = scale(out, kind=scaling, n_train=n_train)
    return out, n_train
