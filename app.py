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
    consensus_timeline,
    pr_curve_fig,
    roc_curve_fig,
    score_distribution,
    score_hist_with_threshold,
    score_timeseries,
    sorted_score_curve,
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

    threshold_method = st.radio(
        "임계값 방식",
        ["topk", "mad"],
        index=0,
        format_func=lambda m: {
            "topk": "고정 비율 (top-k)",
            "mad": "데이터 적응형 (MAD)",
        }[m],
        help=(
            "top-k: 항상 이상 비율만큼 플래그. "
            "MAD: 점수 분포에서 통계적으로 벗어난 점만 플래그 → 이상 개수가 데이터에 따라 달라짐."
        ),
    )
    contamination = st.slider(
        "이상 비율 (top-k일 때 플래그 비율 / MAD일 때 모델 내부 파라미터)",
        0.01, 0.20, 0.05, 0.01,
    )
    mad_k = st.slider("MAD 민감도 (작을수록 더 많이 탐지)", 2.0, 6.0, 3.5, 0.5)

    with st.expander("고급 설정"):
        split_eval = st.checkbox(
            "학습/탐지 구간 분리 (holdout)", value=False,
            help="앞부분을 '정상' 학습 구간으로 보고 모델·스케일러·임계값을 적합한 뒤, "
                 "뒷부분(탐지 구간)을 out-of-sample로 평가합니다. 데이터 누수를 줄입니다.",
        )
        train_ratio = st.slider(
            "학습 구간 비율", 0.3, 0.9, 0.5, 0.05, disabled=not split_eval
        )
        use_temporal = st.checkbox(
            "시간 특징 사용 (ML 모델에 lag/rolling 추가)", value=True,
            help="Isolation Forest/LOF/OC-SVM이 시점 순서를 활용하도록 시차·이동통계 특징을 추가합니다.",
        )
        rolling_window = st.slider("Rolling Z 윈도우", 5, 200, 30)
        arima_auto = st.checkbox(
            "ARIMA 차수 자동 선택 (AIC)", value=False,
            help="(1,1,1) 고정 대신 작은 후보 격자에서 컬럼별로 AIC 최소 차수를 선택합니다. 느려질 수 있습니다.",
        )
        stl_auto_period = st.checkbox("STL 주기 자동 추정 (ACF)", value=False)
        seasonal_period = st.slider(
            "STL 계절 주기", 2, 365, 24, disabled=stl_auto_period
        )

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
    threshold_method: str,
    mad_k: float,
    use_temporal: bool,
    rolling_window: int,
    seasonal_period: int,
    stl_auto_period: bool,
    split_eval: bool,
    train_ratio: float,
    arima_auto: bool,
    _df: pd.DataFrame,
):
    X_raw = _df[list(numeric_cols)]
    X, n_train = preprocess_pipeline(
        X_raw,
        fill=fill_method,
        value_transform=value_transform,
        scaling=scaling,
        train_ratio=train_ratio if split_eval else 1.0,
    )
    # n_train passed to detectors only when a real holdout is requested
    fit_n_train = n_train if (split_eval and n_train < len(X)) else None

    ml_models = {"Isolation Forest", "LOF", "One-Class SVM"}

    scores_by_model: dict = {}
    preds_by_model: dict = {}
    thresholds: dict = {}

    for name in selected_models:
        cls = ALL_DETECTORS[name]
        kwargs: dict = {
            "contamination": contamination,
            "threshold_method": threshold_method,
            "mad_k": mad_k,
        }
        if name == "Rolling Z-Score":
            kwargs["window"] = rolling_window
        if name == "STL":
            kwargs["period"] = "auto" if stl_auto_period else seasonal_period
        if name == "ARIMA" and arima_auto:
            kwargs["order"] = "auto"
        if name in ml_models:
            kwargs["use_temporal"] = use_temporal
        det = cls(**kwargs)
        result = det.detect(X, n_train=fit_n_train)
        scores_by_model[name] = result.scores
        preds_by_model[name] = result.predictions
        thresholds[name] = result.threshold

    return X, scores_by_model, preds_by_model, thresholds, fit_n_train


if not selected_models:
    st.warning("최소 1개 이상의 모델을 선택하세요.")
    st.stop()

# _run is cached by (file_hash + settings + selected_models), so this is cheap
# when nothing changed and always returns results matching the current
# selection — avoiding stale session_state keyed on a previous model set.
with st.spinner("이상탐지 수행 중..."):
    X, scores_by_model, preds_by_model, thresholds, n_train = _run(
        data.file_hash,
        tuple(data.numeric_cols),
        fill_method,
        value_transform,
        scaling,
        tuple(selected_models),
        contamination,
        threshold_method,
        mad_k,
        use_temporal,
        rolling_window,
        seasonal_period,
        stl_auto_period,
        split_eval,
        train_ratio,
        arima_auto,
        data.df,
    )

# timestamp marking the train/detection boundary (None when no holdout)
split_x = X.index[n_train] if (n_train is not None and n_train < len(X)) else None
if split_x is not None:
    st.caption(
        f"🔎 학습/탐지 분리 활성화: 앞 {n_train}개 시점으로 적합, "
        f"이후 {len(X) - n_train}개 시점을 out-of-sample 탐지 (경계: {split_x})."
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
        split_x=split_x,
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Scores tab ---
with tabs[1]:
    st.subheader("모델별 Anomaly Score")
    st.plotly_chart(score_timeseries(scores_by_model, thresholds, split_x=split_x), use_container_width=True)
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
    # value_transform (diff/logreturn) can drop leading rows, so X may be
    # shorter than the original labels. Align labels to the rows that survived
    # preprocessing before computing any metric.
    labels = None
    if data.labels is not None:
        labels = data.labels.reindex(X.index).dropna().astype(int)
        if labels.empty:
            labels = None

    # when a holdout is active, report metrics on the detection (test) slice
    # only — training-period metrics would be optimistic.
    eval_index = X.index if split_x is None else X.index[n_train:]
    if labels is not None:
        labels = labels.reindex(eval_index).dropna().astype(int)
        if labels.empty:
            labels = None

    if labels is not None:
        st.subheader("지도학습 평가지표 (label 컬럼 감지됨)")
        if split_x is not None:
            st.caption(f"⚠️ 지표는 탐지 구간(out-of-sample, {len(labels)}개 시점)에서만 계산됩니다.")
        st.caption(
            "pa_* 는 point-adjusted 지표입니다: 실제 이상 구간 안에서 한 시점이라도 "
            "탐지하면 그 구간 전체를 탐지한 것으로 간주합니다 (시계열 표준). "
            "events 는 탐지한 이상 구간 수 / 전체 이상 구간 수."
        )
        rows = []
        roc_data, pr_data = {}, {}
        for name in selected_models:
            sc = scores_by_model[name].reindex(labels.index)
            pr = preds_by_model[name].reindex(labels.index).fillna(0).astype(int)
            m = supervised_metrics(labels, sc, pr)
            rows.append({"model": name, **m})
            roc_data[name] = roc_points(labels, sc)
            pr_data[name] = pr_points(labels, sc)
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

        st.subheader("비지도 진단 시각화")
        st.caption("정답 라벨이 없을 때 탐지 결과의 신뢰도를 간접적으로 판단하는 그림입니다.")

        if len(selected_models) >= 2:
            st.markdown(
                "**① 모델 합의 타임라인** — 여러 모델이 공통으로 잡은 시점일수록 실제 이상일 가능성이 높습니다."
            )
            st.plotly_chart(consensus_timeline(preds_by_model), use_container_width=True)
        else:
            st.info("모델을 2개 이상 선택하면 '모델 합의 타임라인'이 표시됩니다.")

        st.markdown(
            "**② 정렬된 점수 곡선** — 곡선에 뚜렷한 꺾임(elbow)이 있으면 정상/이상이 잘 분리된 것입니다. ✕는 임계값 지점."
        )
        st.plotly_chart(sorted_score_curve(scores_by_model, thresholds), use_container_width=True)

        st.markdown(
            "**③ 점수 분포 + 임계값** — 임계값 오른쪽에 별도의 봉우리(이봉)가 보이면 이상점이 본체에서 잘 분리된 신호입니다."
        )
        st.plotly_chart(score_hist_with_threshold(scores_by_model, thresholds), use_container_width=True)
        # trailing spacer: gives a non-chart area at the bottom so the page can
        # be scrolled past the last Plotly chart (charts otherwise capture the
        # mouse wheel, making it feel like the page is stuck).
        st.markdown("<div style='height:120px'></div>", unsafe_allow_html=True)

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
