"""A small, dependency-light logistic-regression engagement classifier.

Implemented in NumPy (no scikit-learn) to keep the prototype's dependency
surface minimal and the math transparent. Includes feature standardization,
L2-regularized batch gradient descent, and the usual classification metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class LogisticRegression:
    lr: float = 0.1
    epochs: int = 800
    l2: float = 1e-3
    seed: int = 0
    w: np.ndarray | None = field(default=None, init=False)
    b: float = field(default=0.0, init=False)
    mu: np.ndarray | None = field(default=None, init=False)
    sigma: np.ndarray | None = field(default=None, init=False)

    def _standardize(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self.mu = X.mean(axis=0)
            self.sigma = X.std(axis=0)
            self.sigma[self.sigma == 0] = 1.0
        return (X - self.mu) / self.sigma

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        rng = np.random.default_rng(self.seed)
        Xs = self._standardize(X, fit=True)
        n, d = Xs.shape
        self.w = rng.normal(0, 0.01, size=d)
        self.b = 0.0
        y = y.astype(float)

        for _ in range(self.epochs):
            p = _sigmoid(Xs @ self.w + self.b)
            err = p - y
            grad_w = Xs.T @ err / n + self.l2 * self.w
            grad_b = float(err.mean())
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self._standardize(X, fit=False)
        return _sigmoid(Xs @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_frac: float = 0.25, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / max(1, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
