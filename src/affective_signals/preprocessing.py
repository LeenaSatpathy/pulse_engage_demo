"""Signal preprocessing: filtering and skin-conductance-response detection."""

from __future__ import annotations

import numpy as np
from scipy import signal as sps


def lowpass(x: np.ndarray, fs: float, cutoff_hz: float = 1.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass (removes high-frequency sensor noise)."""
    nyq = fs / 2.0
    wn = min(cutoff_hz / nyq, 0.99)
    b, a = sps.butter(order, wn, btype="low")
    return sps.filtfilt(b, a, x)


def decompose_eda(eda: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Split EDA into (tonic, phasic) via a slow low-pass baseline.

    A genuine cvxEDA-style decomposition is heavier; this baseline-subtraction
    approach is the standard lightweight approximation for prototypes.
    """
    tonic = lowpass(eda, fs, cutoff_hz=0.05, order=2)
    phasic = eda - tonic
    return tonic, phasic


def detect_scrs(
    phasic: np.ndarray,
    fs: float,
    min_amplitude: float = 0.15,
    min_distance_s: float = 2.0,
) -> np.ndarray:
    """Return indices of skin-conductance-response peaks in the phasic signal."""
    distance = max(1, int(min_distance_s * fs))
    peaks, _ = sps.find_peaks(phasic, height=min_amplitude, distance=distance)
    return peaks
