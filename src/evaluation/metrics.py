"""Evaluation metrics. Supervised (label-based) and unsupervised."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _segments(y: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index ranges of contiguous positive (==1) runs."""
    segs = []
    start = None
    for i, v in enumerate(y):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(y)))
    return segs


def point_adjust(y_true: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Point-adjust predictions (Xu et al., 2018).

    If *any* point inside a true anomaly segment is flagged, the whole segment
    counts as detected. This is the standard correction for time-series
    anomaly detection, where anomalies are events (ranges) rather than isolated
    points and a single hit within an event should count as catching it.
    """
    adj = preds.copy()
    for start, end in _segments(y_true):
        if preds[start:end].any():
            adj[start:end] = 1
    return adj


def event_counts(y_true: np.ndarray, preds: np.ndarray) -> dict:
    """Count true events and how many were caught by at least one flagged point."""
    segs = _segments(y_true)
    detected = sum(1 for s, e in segs if preds[s:e].any())
    return {"n_events": len(segs), "events_detected": detected}


def supervised_metrics(y_true: pd.Series, scores: pd.Series, preds: pd.Series) -> dict:
    """Compute point-wise and point-adjusted precision/recall/F1 + ROC/PR-AUC."""
    y = y_true.astype(int).values
    s = scores.values
    p = preds.astype(int).values
    out: dict = {
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
    }
    # point-adjusted variants (event-aware)
    p_adj = point_adjust(y, p)
    out["pa_precision"] = float(precision_score(y, p_adj, zero_division=0))
    out["pa_recall"] = float(recall_score(y, p_adj, zero_division=0))
    out["pa_f1"] = float(f1_score(y, p_adj, zero_division=0))
    ev = event_counts(y, p)
    out["events"] = f"{ev['events_detected']}/{ev['n_events']}"
    try:
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    except ValueError:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def roc_points(y_true: pd.Series, scores: pd.Series):
    fpr, tpr, _ = roc_curve(y_true.astype(int).values, scores.values)
    return fpr, tpr


def pr_points(y_true: pd.Series, scores: pd.Series):
    precision, recall, _ = precision_recall_curve(y_true.astype(int).values, scores.values)
    return precision, recall


def score_summary(scores: pd.Series) -> dict:
    s = np.asarray(scores.values, dtype=float)
    return {
        "mean": float(np.mean(s)),
        "std": float(np.std(s)),
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "q50": float(np.quantile(s, 0.5)),
        "q95": float(np.quantile(s, 0.95)),
        "q99": float(np.quantile(s, 0.99)),
    }


def jaccard(a: pd.Series, b: pd.Series) -> float:
    """Jaccard similarity between two binary anomaly flag series."""
    a_set = set(a[a == 1].index)
    b_set = set(b[b == 1].index)
    union = a_set | b_set
    if not union:
        return 1.0
    return len(a_set & b_set) / len(union)


def agreement_matrix(predictions: dict[str, pd.Series]) -> pd.DataFrame:
    """Build pairwise Jaccard agreement matrix between models."""
    names = list(predictions.keys())
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for i in names:
        for j in names:
            mat.loc[i, j] = jaccard(predictions[i], predictions[j])
    return mat
