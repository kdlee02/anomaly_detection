from .base import AnomalyDetector, DetectionResult
from .statistical import ZScoreDetector, IQRDetector, RollingZDetector
from .ml import IsolationForestDetector, LOFDetector, OCSVMDetector

ALL_DETECTORS = {
    "Z-Score": ZScoreDetector,
    "IQR": IQRDetector,
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
