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

    def __init__(
        self,
        contamination: float = 0.05,
        threshold_method: str = "topk",
        mad_k: float = 3.5,
        **kwargs,
    ):
        self.contamination = contamination
        # "topk"  -> flag a fixed contamination fraction (rank-based)
        # "mad"   -> data-adaptive: flag points far from the score median in
        #            units of robust scale, so the *count* of anomalies is
        #            decided by the data rather than fixed in advance.
        self.threshold_method = threshold_method
        self.mad_k = mad_k
        self.kwargs = kwargs

    @abstractmethod
    def fit_score(self, X: pd.DataFrame, n_train: int | None = None) -> pd.Series:
        """Fit on X and return per-row anomaly scores (higher = more anomalous).

        If ``n_train`` is given, the model is *fit* on the first ``n_train``
        rows (assumed mostly-normal) and *scores* the whole series, so test-set
        anomalies are judged out-of-sample instead of transductively.
        """
        ...

    def _threshold_value_from_ref(self, ref: np.ndarray) -> float:
        """Top-k score cutoff derived from a reference (training) slice."""
        n = len(ref)
        k = min(max(1, int(round(self.contamination * n))), n)
        return float(np.sort(ref)[::-1][k - 1])

    def _threshold_topk(self, s: np.ndarray) -> tuple[np.ndarray, float]:
        """Flag the top-k highest-scoring points (k = contamination fraction).

        Selecting exactly k by rank avoids the degenerate case where many
        points tie at the quantile value (e.g. IQR scores of 0), which would
        otherwise flag far more than ``contamination`` of the data.
        """
        n = len(s)
        k = min(max(1, int(round(self.contamination * n))), n)
        top_idx = np.argsort(-s, kind="stable")[:k]
        threshold = float(s[top_idx[-1]])
        pred = np.zeros(n, dtype=int)
        pred[top_idx] = 1
        return pred, threshold

    def _threshold_mad(self, s: np.ndarray) -> tuple[np.ndarray, float]:
        """Data-adaptive threshold via the median/MAD robust z-score.

        threshold = median + mad_k * 1.4826 * MAD. The number of flagged points
        is whatever exceeds it, so genuinely clean data can yield zero
        anomalies and a burst of outliers can yield many -- which is what makes
        the dashboard's "is this detection appropriate?" question meaningful.
        Falls back to top-k if the score scale is degenerate (MAD == 0).
        """
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med)))
        if mad < 1e-12:
            return self._threshold_topk(s)
        threshold = med + self.mad_k * 1.4826 * mad
        pred = (s > threshold).astype(int)
        return pred, float(threshold)

    def detect(self, X: pd.DataFrame, n_train: int | None = None) -> DetectionResult:
        scores = self.fit_score(X, n_train)
        s = np.asarray(scores.values, dtype=float)

        if n_train is None or n_train >= len(s):
            # transductive path (unchanged): threshold from the whole series
            if self.threshold_method == "mad":
                pred_arr, threshold = self._threshold_mad(s)
            else:
                pred_arr, threshold = self._threshold_topk(s)
        else:
            # holdout path: the threshold is *learned on the training scores*
            # only, then applied to every point so the test set is judged
            # against a rule that never saw it.
            ref = s[:n_train]
            if self.threshold_method == "mad":
                med = float(np.median(ref))
                mad = float(np.median(np.abs(ref - med)))
                threshold = (
                    med + self.mad_k * 1.4826 * mad
                    if mad >= 1e-12
                    else self._threshold_value_from_ref(ref)
                )
            else:
                threshold = self._threshold_value_from_ref(ref)
            pred_arr = (s > threshold).astype(int)

        predictions = pd.Series(pred_arr, index=scores.index, name="prediction")
        return DetectionResult(
            scores=scores,
            predictions=predictions,
            threshold=threshold,
            model_name=self.name,
        )
