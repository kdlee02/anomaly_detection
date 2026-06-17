"""ML-based anomaly detectors (sklearn)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from .base import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    name = "Isolation Forest"

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100, **kwargs):
        super().__init__(contamination=contamination, **kwargs)
        self.n_estimators = n_estimators

    def fit_score(self, X: pd.DataFrame) -> pd.Series:
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
        )
        model.fit(X.values)
        # score_samples: higher = more normal → invert
        raw = -model.score_samples(X.values)
        return pd.Series(raw, index=X.index, name="score")


class LOFDetector(AnomalyDetector):
    name = "LOF"

    def __init__(self, contamination: float = 0.05, n_neighbors: int = 20, **kwargs):
        super().__init__(contamination=contamination, **kwargs)
        self.n_neighbors = n_neighbors

    def fit_score(self, X: pd.DataFrame) -> pd.Series:
        n_neighbors = min(self.n_neighbors, max(2, len(X) - 1))
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=self.contamination)
        model.fit_predict(X.values)
        raw = -model.negative_outlier_factor_
        return pd.Series(raw, index=X.index, name="score")


class OCSVMDetector(AnomalyDetector):
    name = "One-Class SVM"

    def __init__(self, contamination: float = 0.05, nu: float | None = None, **kwargs):
        super().__init__(contamination=contamination, **kwargs)
        self.nu = nu if nu is not None else contamination

    def fit_score(self, X: pd.DataFrame) -> pd.Series:
        # subsample if too large for OC-SVM
        Xv = X.values
        if len(Xv) > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(Xv), size=5000, replace=False)
            fit_X = Xv[idx]
        else:
            fit_X = Xv
        model = OneClassSVM(nu=self.nu, kernel="rbf", gamma="scale")
        model.fit(fit_X)
        raw = -model.decision_function(Xv)
        return pd.Series(raw, index=X.index, name="score")
