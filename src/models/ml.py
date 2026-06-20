"""ML-based anomaly detectors (sklearn)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from ..data.preprocess import add_temporal_features
from .base import AnomalyDetector


def _featurize(X: pd.DataFrame, use_temporal: bool) -> pd.DataFrame:
    """Optionally enrich iid detectors with lag/rolling temporal context.

    Returns a frame with the *same index* as X so resulting scores stay aligned.
    """
    if not use_temporal:
        return X
    return add_temporal_features(X)


class IsolationForestDetector(AnomalyDetector):
    name = "Isolation Forest"

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        use_temporal: bool = True,
        **kwargs,
    ):
        super().__init__(contamination=contamination, **kwargs)
        self.n_estimators = n_estimators
        self.use_temporal = use_temporal

    def fit_score(self, X: pd.DataFrame, n_train: int | None = None) -> pd.Series:
        F = _featurize(X, self.use_temporal)
        fit_part = F.values if n_train is None else F.values[:n_train]
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
        )
        model.fit(fit_part)
        # score_samples: higher = more normal → invert
        raw = -model.score_samples(F.values)
        return pd.Series(raw, index=X.index, name="score")


class LOFDetector(AnomalyDetector):
    name = "LOF"

    def __init__(
        self,
        contamination: float = 0.05,
        n_neighbors: int = 20,
        use_temporal: bool = True,
        **kwargs,
    ):
        super().__init__(contamination=contamination, **kwargs)
        self.n_neighbors = n_neighbors
        self.use_temporal = use_temporal

    def fit_score(self, X: pd.DataFrame, n_train: int | None = None) -> pd.Series:
        F = _featurize(X, self.use_temporal)
        if n_train is not None and n_train < len(F):
            # novelty mode: fit on the training window, score all points
            nn = min(self.n_neighbors, max(2, n_train - 1))
            model = LocalOutlierFactor(n_neighbors=nn, novelty=True)
            model.fit(F.values[:n_train])
            raw = -model.score_samples(F.values)
        else:
            nn = min(self.n_neighbors, max(2, len(F) - 1))
            model = LocalOutlierFactor(n_neighbors=nn, contamination=self.contamination)
            model.fit_predict(F.values)
            raw = -model.negative_outlier_factor_
        return pd.Series(raw, index=X.index, name="score")


class OCSVMDetector(AnomalyDetector):
    name = "One-Class SVM"

    def __init__(
        self,
        contamination: float = 0.05,
        nu: float | None = None,
        use_temporal: bool = True,
        **kwargs,
    ):
        super().__init__(contamination=contamination, **kwargs)
        self.nu = nu if nu is not None else contamination
        self.use_temporal = use_temporal

    def fit_score(self, X: pd.DataFrame, n_train: int | None = None) -> pd.Series:
        F = _featurize(X, self.use_temporal)
        Xv = F.values
        fit_pool = Xv if n_train is None else Xv[:n_train]
        # subsample if too large for OC-SVM
        if len(fit_pool) > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(fit_pool), size=5000, replace=False)
            fit_X = fit_pool[idx]
        else:
            fit_X = fit_pool
        model = OneClassSVM(nu=self.nu, kernel="rbf", gamma="scale")
        model.fit(fit_X)
        raw = -model.decision_function(Xv)
        return pd.Series(raw, index=X.index, name="score")


class PCAReconstructionDetector(AnomalyDetector):
    """Multivariate detector based on PCA reconstruction error.

    PCA learns the dominant linear correlation structure across variables from
    the bulk of the data. A point that violates that structure -- e.g. two
    sensors that are individually in-range but break their usual co-movement --
    reconstructs poorly, so the squared reconstruction error is a natural
    *cross-variable* anomaly score that the per-column STL/ARIMA models miss.
    """

    name = "PCA Reconstruction"

    def __init__(
        self,
        contamination: float = 0.05,
        n_components: float = 0.9,
        **kwargs,
    ):
        super().__init__(contamination=contamination, **kwargs)
        self.n_components = n_components

    def fit_score(self, X: pd.DataFrame, n_train: int | None = None) -> pd.Series:
        n_features = X.shape[1]
        if n_features < 2:
            # reconstruction is trivial with a single variable; fall back to
            # absolute deviation from the (training) column mean
            col = X.iloc[:, 0].astype(float)
            ref = col.iloc[:n_train] if n_train else col
            raw = (col - ref.mean()).abs().to_numpy()
            return pd.Series(raw, index=X.index, name="score")

        fit_part = X.values if n_train is None else X.values[:n_train]

        # decide how many components to keep, always leaving >=1 residual
        # dimension so reconstruction error cannot be identically zero
        nc = self.n_components
        full = PCA(svd_solver="full").fit(fit_part)
        if isinstance(nc, float) and 0 < nc < 1:
            cum = np.cumsum(full.explained_variance_ratio_)
            k = int(np.searchsorted(cum, nc) + 1)
        else:
            k = int(nc)
        k = max(1, min(k, n_features - 1))

        model = PCA(n_components=k).fit(fit_part)
        Z = model.transform(X.values)
        recon = model.inverse_transform(Z)
        raw = np.sqrt(((X.values - recon) ** 2).sum(axis=1))
        return pd.Series(raw, index=X.index, name="score")
