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


def synthetic_showcase(n: int = 2000, seed: int = 7):
    """Detector-friendly dataset: correlated signals with clearly injected,
    event-style anomalies of the kinds the current models handle well.

    4 sensors are driven by a shared latent factor ``z`` (trend + seasonality),
    so they are strongly correlated -> PCA reconstruction error is ~0 normally
    and spikes on a *correlation break*. Anomalies are contiguous segments
    (events), so point-adjusted / event metrics are meaningful too.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    # shared latent driver + per-sensor seasonality => correlated sensors
    z = 0.0006 * t + np.sin(2 * np.pi * t / 120)
    s1 = 2.0 * z + 0.30 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.15, n)
    s2 = 1.5 * z + rng.normal(0, 0.15, n)
    s3 = -1.0 * z + 0.20 * np.cos(2 * np.pi * t / 48) + rng.normal(0, 0.15, n)
    s4 = 1.0 * z + rng.normal(0, 0.15, n)
    cols = {"sensor_a": s1, "sensor_b": s2, "sensor_c": s3, "sensor_d": s4}
    df = pd.DataFrame(cols, index=_make_dates(n))
    label = np.zeros(n, dtype=int)

    # well-separated anomaly events: (start, length, type)
    events = [
        (180, 1, "spike"), (430, 2, "spike"),
        (640, 10, "level"), (1480, 12, "level"),
        (880, 8, "variance"), (1700, 8, "variance"),
        (1080, 12, "corr_break"), (1320, 10, "corr_break"),
        (300, 1, "spike"), (1900, 6, "variance"),
    ]
    sensors = list(cols.keys())
    for start, length, kind in events:
        sl = slice(start, start + length)
        col = sensors[rng.integers(0, len(sensors))]
        if kind == "spike":
            df.loc[df.index[sl], col] += rng.choice([-1, 1]) * rng.uniform(5, 9)
        elif kind == "level":
            df.loc[df.index[sl], col] += rng.choice([-1, 1]) * rng.uniform(3, 5)
        elif kind == "variance":
            df.loc[df.index[sl], col] += rng.normal(0, 3.0, size=length)
        elif kind == "corr_break":
            # decouple sensor_c from the latent factor -> breaks correlation
            df.loc[df.index[sl], "sensor_c"] = rng.normal(0, 0.3, size=length)
        label[sl] = 1

    df["label"] = label
    df.index.name = "timestamp"
    return df


if __name__ == "__main__":
    synthetic_sensors().to_csv(OUT / "sensors_labeled.csv")
    synthetic_market().to_csv(OUT / "market_ohlcv_labeled.csv")
    synthetic_unlabeled().to_csv(OUT / "weather_unlabeled.csv")
    synthetic_showcase().to_csv(OUT / "showcase_labeled.csv")
    print("samples written to", OUT)
