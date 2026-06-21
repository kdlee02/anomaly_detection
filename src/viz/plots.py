"""Plotly visualizations for the anomaly detection dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def timeseries_with_anomalies(
    df: pd.DataFrame,
    predictions: pd.Series,
    title: str = "Time Series with Anomalies",
    split_x=None,
) -> go.Figure:
    """Multi-feature time series with anomaly markers overlaid."""
    cols = list(df.columns)
    fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, subplot_titles=cols)
    anomaly_idx = predictions[predictions == 1].index

    # all feature lines share one neutral color so the only highlighted thing
    # is the anomaly marker; otherwise plotly's default palette makes the 2nd
    # feature line red and it collides with the (formerly red) anomaly markers.
    line_color = "#1f77b4"      # steel blue, matches the app theme
    anomaly_color = "#FF8C00"   # dark orange — distinct, not red
    for i, col in enumerate(cols, start=1):
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[col], mode="lines", name=col,
                line=dict(color=line_color, width=1.2), showlegend=False,
            ),
            row=i, col=1,
        )
        if len(anomaly_idx):
            fig.add_trace(
                go.Scatter(
                    x=anomaly_idx,
                    y=df.loc[anomaly_idx, col],
                    mode="markers",
                    marker=dict(color=anomaly_color, size=9, symbol="x", line=dict(width=1)),
                    name="anomaly",
                    showlegend=(i == 1),
                ),
                row=i, col=1,
            )
    if split_x is not None:
        fig.add_vline(x=split_x, line_dash="dash", line_color="green")
    fig.update_layout(height=200 * len(cols), title=title, hovermode="x unified")
    return fig


def score_timeseries(
    scores_by_model: dict[str, pd.Series],
    thresholds: dict[str, float],
    split_x=None,
) -> go.Figure:
    """Anomaly score time series per model with threshold lines."""
    fig = go.Figure()
    for name, s in scores_by_model.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=name))
        thr = thresholds.get(name)
        if thr is not None:
            fig.add_hline(y=thr, line_dash="dot", annotation_text=f"{name} thr", line_color="gray")
    if split_x is not None:
        # NB: no annotation_text here -- plotly's vline annotation placement
        # averages the x-coords, which raises on a datetime axis (Timestamp +
        # int). The divider line alone marks the train/detect boundary.
        fig.add_vline(x=split_x, line_dash="dash", line_color="green")
    fig.update_layout(title="Anomaly Scores", hovermode="x unified", height=400)
    return fig


def score_distribution(scores_by_model: dict[str, pd.Series]) -> go.Figure:
    """Histogram of scores per model (overlapping)."""
    fig = go.Figure()
    for name, s in scores_by_model.items():
        fig.add_trace(go.Histogram(x=s.values, name=name, opacity=0.5, nbinsx=50))
    fig.update_layout(barmode="overlay", title="Score Distribution", height=400)
    return fig


def agreement_heatmap(mat: pd.DataFrame) -> go.Figure:
    """Heatmap of pairwise Jaccard agreement."""
    fig = px.imshow(
        mat.values.astype(float),
        x=mat.columns,
        y=mat.index,
        color_continuous_scale="Blues",
        text_auto=".2f",
        aspect="auto",
    )
    fig.update_layout(title="Model Agreement (Jaccard)", height=400)
    return fig


def consensus_timeline(preds_by_model: dict[str, pd.Series]) -> go.Figure:
    """How many models flag each timestamp (label-free confidence signal).

    With no ground truth, points that several independent detectors agree on
    are the most trustworthy anomalies; a lone-model flag is more likely noise.
    """
    pred_df = pd.DataFrame(preds_by_model)
    consensus = pred_df.sum(axis=1)
    n_models = pred_df.shape[1]
    fig = go.Figure(
        go.Bar(
            x=consensus.index,
            y=consensus.values,
            marker=dict(color=consensus.values, colorscale="Reds", cmin=0, cmax=n_models),
        )
    )
    fig.update_layout(
        title=f"모델 합의: 동시에 이상으로 판단한 모델 수 (최대 {n_models})",
        yaxis_title="플래그한 모델 수",
        yaxis=dict(dtick=1, range=[0, n_models]),
        height=300,
        hovermode="x unified",
    )
    return fig


def sorted_score_curve(
    scores_by_model: dict[str, pd.Series], thresholds: dict[str, float]
) -> go.Figure:
    """Scores sorted high→low (per-model min-max normalized) with the cutoff.

    A sharp elbow means normal and anomalous points separate cleanly, so the
    threshold sits in a natural gap; a smooth ramp means the cut is arbitrary.
    The ✕ marks where each model's threshold falls along its own curve.
    """
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for i, (name, s) in enumerate(scores_by_model.items()):
        color = palette[i % len(palette)]
        v = np.sort(np.asarray(s.values, dtype=float))[::-1]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
        rank = np.arange(1, len(v) + 1)
        fig.add_trace(go.Scatter(x=rank, y=norm, mode="lines", name=name, line=dict(color=color)))
        thr = thresholds.get(name)
        if thr is not None and hi > lo:
            n_above = int((np.asarray(s.values, dtype=float) > thr).sum())
            if 0 < n_above <= len(v):
                fig.add_trace(go.Scatter(
                    x=[n_above], y=[(thr - lo) / (hi - lo)],
                    mode="markers", marker=dict(symbol="x", size=11, color=color),
                    name=f"{name} cut", showlegend=False,
                ))
    fig.update_layout(
        title="정렬된 이상 점수 (정규화) — 꺾임(elbow)이 뚜렷할수록 분리 양호",
        xaxis_title="순위 (높은 점수 → 낮은 점수)",
        yaxis_title="정규화 점수",
        height=350,
        hovermode="x unified",
    )
    return fig


def score_hist_with_threshold(
    scores_by_model: dict[str, pd.Series], thresholds: dict[str, float]
) -> go.Figure:
    """Overlaid score histograms with each model's threshold as a dotted line.

    A bimodal distribution (a separate right-hand bump past the threshold)
    means anomalies stand apart from the bulk; a single smooth mode means the
    threshold is slicing through ordinary points.
    """
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for i, (name, s) in enumerate(scores_by_model.items()):
        color = palette[i % len(palette)]
        fig.add_trace(go.Histogram(x=s.values, name=name, opacity=0.5, nbinsx=50, marker_color=color))
        thr = thresholds.get(name)
        if thr is not None:
            fig.add_vline(x=float(thr), line_dash="dot", line_color=color)
    fig.update_layout(
        barmode="overlay",
        title="점수 분포 + 임계값 — 이봉(bimodal)일수록 이상이 뚜렷",
        xaxis_title="anomaly score",
        height=350,
    )
    return fig


def roc_curve_fig(curves: dict[str, tuple]) -> go.Figure:
    fig = go.Figure()
    for name, (fpr, tpr) in curves.items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="random"))
    fig.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR", height=400)
    return fig


def pr_curve_fig(curves: dict[str, tuple]) -> go.Figure:
    fig = go.Figure()
    for name, (prec, rec) in curves.items():
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=name))
    fig.update_layout(title="Precision-Recall Curves", xaxis_title="Recall", yaxis_title="Precision", height=400)
    return fig
