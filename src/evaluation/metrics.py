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


def supervised_metrics(y_true: pd.Series, scores: pd.Series, preds: pd.Series) -> dict:
    """Compute precision/recall/F1/ROC-AUC/PR-AUC."""
    y = y_true.astype(int).values
    s = scores.values
    p = preds.astype(int).values
    out: dict = {
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
    }
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
