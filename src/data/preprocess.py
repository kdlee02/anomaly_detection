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


def scale(df: pd.DataFrame, kind: str = "standard") -> tuple[pd.DataFrame, object | None]:
    """Scale features. Returns (scaled_df, fitted_scaler)."""
    cls = SCALERS.get(kind)
    if cls is None:
        return df.copy(), None
    scaler = cls()
    arr = scaler.fit_transform(df.values)
    return pd.DataFrame(arr, index=df.index, columns=df.columns), scaler


def preprocess_pipeline(
    df: pd.DataFrame,
    fill: str = "ffill",
    value_transform: str = "none",
    scaling: str = "standard",
) -> pd.DataFrame:
    """Run full preprocessing pipeline. Returns scaled dataframe."""
    out = fill_missing(df, method=fill)
    out = transform(out, kind=value_transform)
    out, _ = scale(out, kind=scaling)
    return out
