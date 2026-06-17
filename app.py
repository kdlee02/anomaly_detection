"""Streamlit web app: multivariate time-series anomaly detection."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.loader import load_csv
from src.data.preprocess import preprocess_pipeline
from src.evaluation.metrics import (
    agreement_matrix,
    pr_points,
    roc_points,
    score_summary,
    supervised_metrics,
)
from src.models import ALL_DETECTORS
from src.viz.plots import (
    agreement_heatmap,
    pr_curve_fig,
    roc_curve_fig,
    score_distribution,
    score_timeseries,
    timeseries_with_anomalies,
)

st.set_page_config(
    page_title="Time-Series Anomaly Detection",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Multivariate Time-Series Anomaly Detection")
st.caption(
    "임의의 다변량 시계열 CSV를 업로드하면 자동으로 이상탐지를 수행하고 대시보드로 비교합니다."
)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. 데이터 업로드")
    uploaded = st.file_uploader("CSV 파일", type=["csv"])

    use_sample = st.checkbox("샘플 데이터 사용", value=not bool(uploaded))
    sample_choice = None
    if use_sample:
        samples_dir = Path(__file__).parent / "samples"
        sample_files = sorted(samples_dir.glob("*.csv")) if samples_dir.exists() else []
        if sample_files:
            sample_choice = st.selectbox(
                "샘플 선택", sample_files, format_func=lambda p: p.name
            )

    st.header("2. 전처리")
    fill_method = st.selectbox("결측치 처리", ["ffill", "interpolate", "zero", "drop"], index=0)
    value_transform = st.selectbox("값 변환", ["none", "diff", "logreturn"], index=0)
    scaling = st.selectbox("정규화", ["standard", "robust", "minmax", "none"], index=0)

    st.header("3. 모델 선택")
    _default_models = [
        m for m in ["STL", "Isolation Forest", "Rolling Z-Score"]
        if m in ALL_DETECTORS
    ]
    selected_models = st.multiselect(
        "이상탐지 모델",
        list(ALL_DETECTORS.keys()),
        default=_default_models,
    )
    contamination = st.slider("이상 비율 (threshold 분위수)", 0.01, 0.20, 0.05, 0.01)

    with st.expander("고급 설정"):
        rolling_window = st.slider("Rolling Z 윈도우", 5, 200, 30)
        seasonal_period = st.slider("STL 계절 주기", 2, 365, 24)

    run_button = st.button("🚀 이상탐지 실행", type="primary", use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────
def _get_bytes() -> bytes | None:
    if uploaded is not None:
        return uploaded.getvalue()
    if use_sample and sample_choice is not None:
        return Path(sample_choice).read_bytes()
    return None


file_bytes = _get_bytes()
if file_bytes is None:
    st.info("⬅️ 왼쪽 사이드바에서 CSV 파일을 업로드하거나 샘플 데이터를 선택하세요.")
    st.stop()

data = load_csv(file_bytes)

# Overview row
col1, col2, col3, col4 = st.columns(4)
col1.metric("행 수", data.n_rows)
col2.metric("수치형 변수", len(data.numeric_cols))
col3.metric("결측치", data.n_missing)
col4.metric("라벨 컬럼", data.label_col or "없음")

with st.expander("원본 데이터 미리보기"):
    st.dataframe(data.df.head(50), use_container_width=True)

if not data.numeric_cols:
    st.error("수치형 컬럼이 없습니다. CSV를 확인하세요.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Run detection (cached by file hash + settings)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _run(
    file_hash: str,
    numeric_cols: tuple,
    fill_method: str,
    value_transform: str,
    scaling: str,
    selected_models: tuple,
    contamination: float,
    rolling_window: int,
    seasonal_period: int,
    _df: pd.DataFrame,
):
    X_raw = _df[list(numeric_cols)]
    X = preprocess_pipeline(
        X_raw, fill=fill_method, value_transform=value_transform, scaling=scaling
    )

    scores_by_model: dict = {}
    preds_by_model: dict = {}
    thresholds: dict = {}

    for name in selected_models:
        cls = ALL_DETECTORS[name]
        kwargs: dict = {"contamination": contamination}
        if name == "Rolling Z-Score":
            kwargs["window"] = rolling_window
        if name == "STL":
            kwargs["period"] = seasonal_period
        det = cls(**kwargs)
        result = det.detect(X)
        scores_by_model[name] = result.scores
        preds_by_model[name] = result.predictions
        thresholds[name] = result.threshold

    return X, scores_by_model, preds_by_model, thresholds


if not selected_models:
    st.warning("최소 1개 이상의 모델을 선택하세요.")
    st.stop()

# _run is cached by (file_hash + settings + selected_models), so this is cheap
# when nothing changed and always returns results matching the current
# selection — avoiding stale session_state keyed on a previous model set.
with st.spinner("이상탐지 수행 중..."):
    X, scores_by_model, preds_by_model, thresholds = _run(
        data.file_hash,
        tuple(data.numeric_cols),
        fill_method,
        value_transform,
        scaling,
        tuple(selected_models),
        contamination,
        rolling_window,
        seasonal_period,
        data.df,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Overview", "Scores", "Comparison", "Metrics", "Details"])

# --- Overview tab ---
with tabs[0]:
    st.subheader("시계열 + 이상점")
    model_for_overview = st.selectbox("기준 모델", selected_models, key="overview_model")
    fig = timeseries_with_anomalies(
        X,
        preds_by_model[model_for_overview],
        title=f"{model_for_overview}: 탐지된 이상점 = {int(preds_by_model[model_for_overview].sum())}",
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Scores tab ---
with tabs[1]:
    st.subheader("모델별 Anomaly Score")
    st.plotly_chart(score_timeseries(scores_by_model, thresholds), use_container_width=True)
    st.plotly_chart(score_distribution(scores_by_model), use_container_width=True)

    st.subheader("점수 요약 통계")
    summary_rows = []
    for name, s in scores_by_model.items():
        row = {"model": name, **score_summary(s), "threshold": thresholds[name]}
        summary_rows.append(row)
    st.dataframe(pd.DataFrame(summary_rows).set_index("model"), use_container_width=True)

# --- Comparison tab ---
with tabs[2]:
    st.subheader("모델 간 일치도 (Jaccard)")
    if len(selected_models) >= 2:
        mat = agreement_matrix(preds_by_model)
        st.plotly_chart(agreement_heatmap(mat), use_container_width=True)

        st.subheader("탐지 시점 비교")
        anomaly_table = pd.DataFrame(preds_by_model)
        anomaly_table = anomaly_table[anomaly_table.sum(axis=1) > 0]
        st.dataframe(anomaly_table, use_container_width=True)
    else:
        st.info("2개 이상의 모델을 선택하면 비교가 표시됩니다.")

# --- Metrics tab ---
with tabs[3]:
    if data.labels is not None:
        st.subheader("지도학습 평가지표 (label 컬럼 감지됨)")
        rows = []
        roc_data, pr_data = {}, {}
        for name in selected_models:
            m = supervised_metrics(data.labels, scores_by_model[name], preds_by_model[name])
            rows.append({"model": name, **m})
            roc_data[name] = roc_points(data.labels, scores_by_model[name])
            pr_data[name] = pr_points(data.labels, scores_by_model[name])
        st.dataframe(pd.DataFrame(rows).set_index("model"), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(roc_curve_fig(roc_data), use_container_width=True)
        with c2:
            st.plotly_chart(pr_curve_fig(pr_data), use_container_width=True)
    else:
        st.info(
            "label 컬럼이 감지되지 않아 비지도 지표만 표시합니다. "
            "CSV에 0/1 라벨 컬럼(label/anomaly/y 등)이 있으면 자동으로 인식됩니다."
        )
        rows = [{"model": n, **score_summary(s)} for n, s in scores_by_model.items()]
        st.dataframe(pd.DataFrame(rows).set_index("model"), use_container_width=True)

# --- Details tab ---
with tabs[4]:
    st.subheader("이상 시점 상세")
    model_for_details = st.selectbox("모델 선택", selected_models, key="details_model")
    preds = preds_by_model[model_for_details]
    scores = scores_by_model[model_for_details]
    detail = pd.DataFrame({
        "score": scores,
        "is_anomaly": preds,
    }).join(X, how="left")
    anomalies = detail[detail["is_anomaly"] == 1].sort_values("score", ascending=False)
    st.write(f"총 {len(anomalies)}개 이상점")
    st.dataframe(anomalies, use_container_width=True)

    csv_buf = io.StringIO()
    anomalies.to_csv(csv_buf)
    st.download_button(
        "⬇️ 이상점 CSV 다운로드",
        csv_buf.getvalue(),
        file_name=f"anomalies_{model_for_details}.csv",
        mime="text/csv",
    )
