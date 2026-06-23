import unittest

import numpy as np

from affective_signals import (
    FEATURE_NAMES,
    detect_scrs,
    extract_features,
    generate_recording,
    label_windows,
)
from affective_signals.preprocessing import decompose_eda


class SignalTests(unittest.TestCase):
    def test_recording_shape_and_determinism(self) -> None:
        a = generate_recording(duration_s=60, fs=8, seed=3)
        b = generate_recording(duration_s=60, fs=8, seed=3)
        self.assertEqual(len(a.eda), 60 * 8)
        self.assertEqual(len(a.hr), len(a.eda))
        np.testing.assert_array_equal(a.eda, b.eda)

    def test_scr_detection_finds_event_responses(self) -> None:
        rec = generate_recording(duration_s=120, fs=8, event_rate_per_min=6, seed=5)
        _, phasic = decompose_eda(rec.eda, rec.fs)
        peaks = detect_scrs(phasic, rec.fs)
        # Should detect a meaningful fraction of the arousal events.
        self.assertGreater(len(peaks), 0.4 * len(rec.events))


class FeatureTests(unittest.TestCase):
    def test_feature_matrix_shape(self) -> None:
        rec = generate_recording(duration_s=120, fs=8, seed=2)
        X, y = extract_features(rec, window_s=10.0)
        self.assertEqual(X.shape[1], len(FEATURE_NAMES))
        self.assertEqual(len(X), len(y))
        self.assertEqual(X.shape[0], 12)  # 120s / 10s windows

    def test_labels_are_binary(self) -> None:
        rec = generate_recording(duration_s=120, fs=8, seed=2)
        _, labels = label_windows(rec, window_s=10.0)
        self.assertTrue(set(np.unique(labels)).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
