from .base import AnomalyDetector, DetectionResult
from .statistical import ARIMADetector, STLDetector, RollingZDetector
from .ml import (
    IsolationForestDetector,
    LOFDetector,
    OCSVMDetector,
    PCAReconstructionDetector,
)

ALL_DETECTORS = {
    "ARIMA": ARIMADetector,
    "STL": STLDetector,
    "Rolling Z-Score": RollingZDetector,
    "Isolation Forest": IsolationForestDetector,
    "LOF": LOFDetector,
    "One-Class SVM": OCSVMDetector,
    "PCA Reconstruction": PCAReconstructionDetector,
}

__all__ = [
    "AnomalyDetector",
    "DetectionResult",
    "ALL_DETECTORS",
]
