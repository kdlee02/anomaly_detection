# 📈 Multivariate Time-Series Anomaly Detection

임의의 다변량 시계열 CSV를 업로드하면 자동으로 이상탐지를 수행하고, 다양한 모델/지표를 대시보드로 비교할 수 있는 Streamlit 웹앱.

## ✨ 주요 기능

- **자동 스키마 감지**: 시간 컬럼, 수치형 컬럼, 라벨 컬럼(`label`, `anomaly`, `y` 등 0/1)을 자동 인식
- **다중 모델 앙상블 비교**
  - 통계 기반: Z-Score, IQR, Rolling Z-Score
  - ML 기반: Isolation Forest, LOF, One-Class SVM
- **전처리 옵션**: 결측치 처리 (ffill/interpolate/zero/drop), 변환 (diff/log-return), 정규화 (Standard/Robust/MinMax)
- **5개 탭 대시보드**
  1. Overview — 시계열 + 이상점 오버레이
  2. Scores — 모델별 anomaly score & 분포
  3. Comparison — 모델 간 Jaccard 일치도 히트맵
  4. Metrics — 지도(P/R/F1/ROC/PR-AUC) 또는 비지도 지표
  5. Details — 이상 시점 표 + CSV 다운로드
- **파일 변경 시 자동 재실행**: 업로드 파일의 해시로 캐시 무효화

## 🚀 로컬 실행

```bash
cd anomaly_detection
pip install -r requirements.txt
streamlit run app.py
```

브라우저가 자동으로 열리지 않으면 http://localhost:8501 접속.

## 📂 디렉토리

```
anomaly_detection/
├── app.py                       # Streamlit 메인
├── requirements.txt
├── .streamlit/config.toml
├── src/
│   ├── data/                    # CSV 로더, 전처리
│   ├── models/                  # 통계/ML 이상탐지기
│   ├── evaluation/              # 평가 지표
│   └── viz/                     # Plotly 시각화
├── samples/                     # 데모용 합성 데이터 (라벨 있/없음)
│   ├── sensors_labeled.csv
│   ├── market_ohlcv_labeled.csv
│   └── weather_unlabeled.csv
└── tests/
```

## 📝 입력 CSV 포맷

- 첫 행 헤더
- 시간 컬럼이 있으면 자동 인식 (컬럼명에 `date`/`time` 포함, 또는 첫 컬럼이 datetime 파싱 가능)
- 수치형 컬럼은 모두 특징(feature)으로 사용
- 라벨 컬럼 (선택): `label`, `anomaly`, `is_anomaly`, `target`, `y` 중 하나, 값은 0/1
  → 있으면 지도학습 지표 (P/R/F1, ROC-AUC, PR-AUC) 자동 계산

예시:
```csv
timestamp,sensor_1,sensor_2,sensor_3,label
2024-01-01 00:00:00,0.21,1.04,-0.5,0
2024-01-01 01:00:00,0.22,1.03,-0.4,0
...
```

## ☁️ 배포 (Streamlit Cloud)

1. 이 폴더를 GitHub repo로 푸시
2. https://share.streamlit.io 접속 → "New app"
3. repo / branch / `anomaly_detection/app.py` 지정 → Deploy

`requirements.txt`가 자동으로 인식되어 의존성이 설치됩니다.

## 🧪 샘플 데이터 재생성

```bash
python samples/generate_samples.py
```

- `sensors_labeled.csv` — 다변량 센서 + 스파이크 주입 (라벨 있음)
- `market_ohlcv_labeled.csv` — OHLCV 형태 + flash crash/spike (라벨 있음)
- `weather_unlabeled.csv` — 온도/습도/기압 (라벨 없음, 비지도 시연용)

## 🛠 모델 인터페이스

새로운 모델을 추가하려면 `src/models/base.py`의 `AnomalyDetector`를 상속:

```python
from src.models.base import AnomalyDetector
import pandas as pd

class MyDetector(AnomalyDetector):
    name = "My Detector"

    def fit_score(self, X: pd.DataFrame) -> pd.Series:
        # 행마다 anomaly score 반환 (높을수록 이상)
        ...
```

그리고 `src/models/__init__.py`의 `ALL_DETECTORS`에 등록.
