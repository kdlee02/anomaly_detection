"""Common interface for anomaly detectors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DetectionResult:
    scores: pd.Series             # anomaly score per timestamp (higher = more anomalous)
    predictions: pd.Series        # 0/1 anomaly flag at chosen threshold
    threshold: float
    model_name: str

    @property
    def n_anomalies(self) -> int:
        return int(self.predictions.sum())


class AnomalyDetector(ABC):
    name: str = "Detector"

    def __init__(self, contamination: float = 0.05, **kwargs):
        self.contamination = contamination
        self.kwargs = kwargs

    @abstractmethod
    def fit_score(self, X: pd.DataFrame) -> pd.Series:
        """Fit on X and return per-row anomaly scores (higher = more anomalous)."""
        ...

    def detect(self, X: pd.DataFrame) -> DetectionResult:
        scores = self.fit_score(X)
        s = np.asarray(scores.values, dtype=float)
        n = len(s)
        # Flag the top-k highest-scoring points (k = contamination fraction).
        # Selecting exactly k by rank avoids the degenerate case where many
        # points tie at the quantile value (e.g. IQR scores of 0), which would
        # otherwise flag far more than `contamination` of the data.
        k = max(1, int(round(self.contamination * n)))
        k = min(k, n)
        top_idx = np.argsort(-s, kind="stable")[:k]
        threshold = float(s[top_idx[-1]])
        pred_arr = np.zeros(n, dtype=int)
        pred_arr[top_idx] = 1
        predictions = pd.Series(pred_arr, index=scores.index, name="prediction")
        return DetectionResult(
            scores=scores,
            predictions=predictions,
            threshold=threshold,
            model_name=self.name,
        )
