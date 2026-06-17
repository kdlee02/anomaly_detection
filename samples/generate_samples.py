"""Generate synthetic multivariate time-series datasets with injected anomalies."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent


def _make_dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="h")


def synthetic_sensors(n: int = 2000, n_features: int = 5, anomaly_rate: float = 0.03, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    cols = {}
    for i in range(n_features):
        period = 50 + i * 17
        amp = 1 + i * 0.3
        signal = amp * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.2, size=n)
        cols[f"sensor_{i+1}"] = signal
    df = pd.DataFrame(cols, index=_make_dates(n))

    label = np.zeros(n, dtype=int)
    n_anom = int(n * anomaly_rate)
    anom_idx = rng.choice(n, size=n_anom, replace=False)
    for idx in anom_idx:
        feature_idx = rng.integers(0, n_features)
        col = f"sensor_{feature_idx+1}"
        spike = rng.choice([-1, 1]) * rng.uniform(4, 8)
        df.loc[df.index[idx], col] = df.iloc[idx][col] + spike
        label[idx] = 1
    df["label"] = label
    df.index.name = "timestamp"
    return df


def synthetic_market(n: int = 1500, seed: int = 1):
    """Synthetic OHLCV-like multi-series with regime shifts and shocks."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 100 + np.cumsum(rng.normal(0, 0.5, size=n))
    vol = 1 + 0.5 * np.sin(2 * np.pi * t / 200)
    close = base + rng.normal(0, vol)
    open_ = close + rng.normal(0, 0.3, size=n)
    high = np.maximum(close, open_) + np.abs(rng.normal(0, 0.4, size=n))
    low = np.minimum(close, open_) - np.abs(rng.normal(0, 0.4, size=n))
    volume = np.abs(rng.normal(1_000_000, 200_000, size=n))

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=_make_dates(n))

    # inject flash crashes / volume spikes
    n_anom = int(n * 0.02)
    anom_idx = rng.choice(n, size=n_anom, replace=False)
    label = np.zeros(n, dtype=int)
    for idx in anom_idx:
        kind = rng.choice(["crash", "spike", "volume"])
        if kind == "crash":
            df.iloc[idx, df.columns.get_loc("close")] *= 0.92
            df.iloc[idx, df.columns.get_loc("low")] *= 0.90
        elif kind == "spike":
            df.iloc[idx, df.columns.get_loc("close")] *= 1.08
            df.iloc[idx, df.columns.get_loc("high")] *= 1.10
        else:
            df.iloc[idx, df.columns.get_loc("volume")] *= 5
        label[idx] = 1
    df["label"] = label
    df.index.name = "timestamp"
    return df


def synthetic_unlabeled(n: int = 1000, seed: int = 2):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    df = pd.DataFrame({
        "temperature": 20 + 5 * np.sin(2 * np.pi * t / 100) + rng.normal(0, 0.5, size=n),
        "humidity": 50 + 10 * np.cos(2 * np.pi * t / 120) + rng.normal(0, 1, size=n),
        "pressure": 1013 + np.cumsum(rng.normal(0, 0.05, size=n)),
    }, index=_make_dates(n))
    # silent anomalies, no label
    idx = rng.choice(n, size=15, replace=False)
    for i in idx:
        df.iloc[i, 0] += rng.choice([-1, 1]) * 8
    df.index.name = "timestamp"
    return df


if __name__ == "__main__":
    synthetic_sensors().to_csv(OUT / "sensors_labeled.csv")
    synthetic_market().to_csv(OUT / "market_ohlcv_labeled.csv")
    synthetic_unlabeled().to_csv(OUT / "weather_unlabeled.csv")
    print("samples written to", OUT)
