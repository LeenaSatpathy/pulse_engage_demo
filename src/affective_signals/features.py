"""Extract affective features per analysis window.

Features (per window) combine EDA and HR descriptors that are standard in the
affective-computing literature:

* ``scr_count``      — number of skin-conductance responses (phasic peaks)
* ``scr_amp_mean``   — mean SCR amplitude
* ``eda_slope``      — tonic trend within the window
* ``eda_std``        — phasic variability
* ``hr_mean``        — mean heart rate
* ``hr_std``         — heart-rate variability proxy
* ``hr_slope``       — heart-rate trend
"""

from __future__ import annotations

import numpy as np

from .preprocessing import decompose_eda, detect_scrs
from .signals import Recording, label_windows

FEATURE_NAMES = [
    "scr_count", "scr_amp_mean", "eda_slope", "eda_std",
    "hr_mean", "hr_std", "hr_slope",
]


def _slope(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    idx = np.arange(len(x))
    return float(np.polyfit(idx, x, 1)[0])


def extract_features(rec: Recording, window_s: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y): a per-window feature matrix and engagement labels."""
    tonic, phasic = decompose_eda(rec.eda, rec.fs)
    starts, labels = label_windows(rec, window_s=window_s)
    win = int(window_s * rec.fs)

    rows: list[list[float]] = []
    for s in starts:
        sl = slice(s, s + win)
        ph = phasic[sl]
        peaks = detect_scrs(ph, rec.fs)
        scr_amp_mean = float(np.mean(ph[peaks])) if len(peaks) else 0.0
        rows.append([
            float(len(peaks)),
            scr_amp_mean,
            _slope(tonic[sl]),
            float(np.std(ph)),
            float(np.mean(rec.hr[sl])),
            float(np.std(rec.hr[sl])),
            _slope(rec.hr[sl]),
        ])
    return np.asarray(rows, dtype=float), labels


def build_dataset(
    n_recordings: int = 30,
    window_s: float = 10.0,
    base_seed: int = 100,
    **rec_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate features/labels across many synthetic recordings."""
    from .signals import generate_recording

    X_parts, y_parts = [], []
    for i in range(n_recordings):
        rec = generate_recording(seed=base_seed + i, **rec_kwargs)
        X, y = extract_features(rec, window_s=window_s)
        X_parts.append(X)
        y_parts.append(y)
    return np.vstack(X_parts), np.concatenate(y_parts)
