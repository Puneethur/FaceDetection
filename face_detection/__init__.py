"""Face detection package."""

from .detector import DetectionConfig, FaceDetector
from .recognizer import FaceRecognitionStore, RecognizedFace, TrainingSummary

__all__ = [
    "DetectionConfig",
    "FaceDetector",
    "FaceRecognitionStore",
    "RecognizedFace",
    "TrainingSummary",
]
