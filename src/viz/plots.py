"""Plotly visualizations for the anomaly detection dashboard."""
from __future__ import annotations

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

    for i, col in enumerate(cols, start=1):
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], mode="lines", name=col, showlegend=False),
            row=i, col=1,
        )
        if len(anomaly_idx):
            fig.add_trace(
                go.Scatter(
                    x=anomaly_idx,
                    y=df.loc[anomaly_idx, col],
                    mode="markers",
                    marker=dict(color="red", size=7, symbol="x"),
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
