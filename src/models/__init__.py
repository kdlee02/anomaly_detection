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

# LSTM-Autoencoder requires PyTorch, which may be unavailable in some deploy
# environments (e.g. limited-memory hosts). Register it only if torch imports.
try:
    from .dl import LSTMAutoencoderDetector

    ALL_DETECTORS["LSTM-Autoencoder"] = LSTMAutoencoderDetector
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch missing / failed to load
    TORCH_AVAILABLE = False

__all__ = [
    "AnomalyDetector",
    "DetectionResult",
    "ALL_DETECTORS",
]
