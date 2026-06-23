"""Affective-computing prototype: engagement detection from biosignals."""

from .features import FEATURE_NAMES, build_dataset, extract_features
from .model import LogisticRegression, evaluate, train_test_split
from .preprocessing import decompose_eda, detect_scrs, lowpass
from .signals import Recording, generate_recording, label_windows

__all__ = [
    "generate_recording",
    "Recording",
    "label_windows",
    "lowpass",
    "decompose_eda",
    "detect_scrs",
    "extract_features",
    "build_dataset",
    "FEATURE_NAMES",
    "LogisticRegression",
    "train_test_split",
    "evaluate",
]

__version__ = "0.1.0"
