import unittest

import numpy as np

from affective_signals import (
    LogisticRegression,
    build_dataset,
    evaluate,
    train_test_split,
)


class ModelTests(unittest.TestCase):
    def test_learns_separable_data(self) -> None:
        rng = np.random.default_rng(0)
        X0 = rng.normal(-2, 1, size=(100, 3))
        X1 = rng.normal(+2, 1, size=(100, 3))
        X = np.vstack([X0, X1])
        y = np.array([0] * 100 + [1] * 100)
        Xtr, Xte, ytr, yte = train_test_split(X, y, seed=0)
        model = LogisticRegression().fit(Xtr, ytr)
        metrics = evaluate(yte, model.predict(Xte))
        self.assertGreater(metrics["accuracy"], 0.9)

    def test_evaluate_perfect(self) -> None:
        y = np.array([0, 1, 0, 1])
        m = evaluate(y, y)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["f1"], 1.0)

    def test_end_to_end_on_biosignals(self) -> None:
        X, y = build_dataset(n_recordings=24, window_s=10.0)
        Xtr, Xte, ytr, yte = train_test_split(X, y, seed=1)
        model = LogisticRegression(seed=1).fit(Xtr, ytr)
        metrics = evaluate(yte, model.predict(Xte))
        # The engagement signal is learnable above chance from the features.
        self.assertGreater(metrics["accuracy"], 0.6)


if __name__ == "__main__":
    unittest.main()
