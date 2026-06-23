"""Generate synthetic biosignals for affective-computing prototyping.

Produces two channels commonly used in affective sensing from wearables:

* **EDA** (electrodermal activity): a slow *tonic* drift plus *phasic* skin
  conductance responses (SCRs) that spike shortly after arousing events.
* **HR** (heart rate): a baseline with event-driven accelerations and
  beat-to-beat variability.

Each recording is labeled per fixed-length window as *engaged* (1) or
*not engaged* (0), driven by an underlying event train. This gives the
feature-extraction and model code something real-shaped to learn from without
any wearable hardware or PII.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Recording:
    fs: float            # sampling rate (Hz)
    eda: np.ndarray      # electrodermal activity (microsiemens-ish)
    hr: np.ndarray       # heart rate (bpm)
    events: np.ndarray   # event onset times (seconds)

    @property
    def duration_s(self) -> float:
        return len(self.eda) / self.fs

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.eda)) / self.fs


def _scr_kernel(fs: float, tau_rise: float = 0.8, tau_decay: float = 3.5) -> np.ndarray:
    """Biexponential skin-conductance-response shape."""
    t = np.arange(0, 12 * fs) / fs
    k = np.exp(-t / tau_decay) - np.exp(-t / tau_rise)
    k[k < 0] = 0
    peak = k.max()
    return k / peak if peak > 0 else k


def generate_recording(
    duration_s: float = 180.0,
    fs: float = 8.0,
    event_rate_per_min: float = 4.0,
    seed: int = 0,
) -> Recording:
    """Simulate one EDA+HR recording with arousal events."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    # --- Event train (arousal onsets) ---
    n_events = rng.poisson(event_rate_per_min * duration_s / 60.0)
    event_times = np.sort(rng.uniform(0, duration_s, size=n_events))

    # --- EDA: tonic drift + phasic SCRs at events ---
    tonic = 2.0 + 0.5 * np.sin(2 * np.pi * t / 90.0) + np.cumsum(
        rng.normal(0, 0.002, size=n)
    )
    phasic = np.zeros(n)
    kernel = _scr_kernel(fs)
    for et in event_times:
        idx = int(et * fs)
        amp = rng.uniform(0.4, 1.6)
        end = min(idx + len(kernel), n)
        phasic[idx:end] += amp * kernel[: end - idx]
    eda = tonic + phasic + rng.normal(0, 0.02, size=n)

    # --- HR: baseline + event-driven accelerations + variability ---
    hr = np.full(n, 68.0) + 3.0 * np.sin(2 * np.pi * t / 60.0)
    for et in event_times:
        idx = int(et * fs)
        bump = 12.0 * np.exp(-(t - et) ** 2 / (2 * 4.0**2))
        hr += bump
    hr += rng.normal(0, 1.2, size=n)

    return Recording(fs=fs, eda=eda, hr=hr, events=event_times)


def label_windows(rec: Recording, window_s: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (window_start_indices, labels).

    A window is labeled *engaged* (1) if at least one arousal event falls inside
    it, else 0.
    """
    win = int(window_s * rec.fs)
    n_windows = len(rec.eda) // win
    starts = np.arange(n_windows) * win
    labels = np.zeros(n_windows, dtype=int)
    for i, s in enumerate(starts):
        w_start_s = s / rec.fs
        w_end_s = (s + win) / rec.fs
        if np.any((rec.events >= w_start_s) & (rec.events < w_end_s)):
            labels[i] = 1
    return starts, labels
