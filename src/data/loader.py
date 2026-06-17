"""CSV loader with automatic schema detection."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import pandas as pd


@dataclass
class LoadedData:
    df: pd.DataFrame                  # original dataframe (indexed by time if detected)
    numeric_cols: list[str]           # numeric feature columns
    time_col: str | None              # detected datetime column name (or None)
    label_col: str | None             # detected label column (binary 0/1) or None
    n_rows: int
    n_missing: int
    file_hash: str

    @property
    def features(self) -> pd.DataFrame:
        return pd.DataFrame(self.df[self.numeric_cols])

    @property
    def labels(self) -> pd.Series | None:
        if self.label_col is None:
            return None
        return pd.Series(self.df[self.label_col]).astype(int)


LABEL_CANDIDATES = {"label", "labels", "anomaly", "is_anomaly", "target", "y"}


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _detect_time_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if df[col].dtype == "object" or "date" in col.lower() or "time" in col.lower():
            try:
                parsed = pd.to_datetime(df[col], errors="raise")
                if parsed.notna().sum() >= len(df) * 0.9:
                    return col
            except (ValueError, TypeError):
                continue
    # try first column as datetime
    if len(df.columns) > 0:
        try:
            parsed = pd.to_datetime(df.iloc[:, 0], errors="raise")
            if parsed.notna().sum() >= len(df) * 0.9:
                return str(df.columns[0])
        except (ValueError, TypeError):
            pass
    return None


def _detect_label_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col.lower() in LABEL_CANDIDATES:
            vals = df[col].dropna().unique()
            if set(vals).issubset({0, 1, True, False, 0.0, 1.0}):
                return col
    return None


def load_csv(file_bytes: bytes) -> LoadedData:
    """Load CSV bytes, auto-detect time/label/numeric columns."""
    file_hash = _hash_bytes(file_bytes)
    df = pd.read_csv(io.BytesIO(file_bytes))

    time_col = _detect_time_col(df)
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        df = df.set_index(time_col)

    label_col = _detect_label_col(df)

    numeric_cols = [
        c for c in df.columns
        if c != label_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    n_missing = int(df[numeric_cols].isna().sum().sum())

    return LoadedData(
        df=df,
        numeric_cols=numeric_cols,
        time_col=time_col,
        label_col=label_col,
        n_rows=len(df),
        n_missing=n_missing,
        file_hash=file_hash,
    )
