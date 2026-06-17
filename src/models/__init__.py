from .base import AnomalyDetector, DetectionResult
from .statistical import ARIMADetector, STLDetector, RollingZDetector
from .ml import IsolationForestDetector, LOFDetector, OCSVMDetector

ALL_DETECTORS = {
    "ARIMA": ARIMADetector,
    "STL": STLDetector,
    "Rolling Z-Score": RollingZDetector,
    "Isolation Forest": IsolationForestDetector,
    "LOF": LOFDetector,
    "One-Class SVM": OCSVMDetector,
}

__all__ = [
    "AnomalyDetector",
    "DetectionResult",
    "ALL_DETECTORS",
]
