from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


def _to_numpy_1d(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")

    if arr.size == 0:
        raise ValueError(f"{name} is empty")

    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN values")

    return arr


def _validate_binary_targets(y_true: np.ndarray) -> None:
    unique = set(np.unique(y_true).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError("y_true must contain only 0/1 values")


def _clip_probs(y_prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(y_prob, eps, 1.0 - eps)


@dataclass
class CalibrationResult:
    method: str
    a: Optional[float]
    b: Optional[float]
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "a": self.a,
            "b": self.b,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
        }


def binary_log_loss(y_true: Any, y_prob: Any) -> float:
    y_true_arr = _to_numpy_1d(y_true, "y_true")
    y_prob_arr = _to_numpy_1d(y_prob, "y_prob")

    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError("y_true and y_prob must have same length")

    _validate_binary_targets(y_true_arr)
    y_prob_arr = _clip_probs(y_prob_arr)

    loss = -np.mean(
        y_true_arr * np.log(y_prob_arr) + (1.0 - y_true_arr) * np.log(1.0 - y_prob_arr)
    )
    return float(loss)


def brier_score(y_true: Any, y_prob: Any) -> float:
    y_true_arr = _to_numpy_1d(y_true, "y_true")
    y_prob_arr = _to_numpy_1d(y_prob, "y_prob")

    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError("y_true and y_prob must have same length")

    _validate_binary_targets(y_true_arr)
    return float(np.mean((y_prob_arr - y_true_arr) ** 2))


def expected_calibration_error(
    y_true: Any,
    y_prob: Any,
    n_bins: int = 10,
) -> float:
    y_true_arr = _to_numpy_1d(y_true, "y_true")
    y_prob_arr = _to_numpy_1d(y_prob, "y_prob")

    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError("y_true and y_prob must have same length")

    if n_bins <= 1:
        raise ValueError("n_bins must be > 1")

    _validate_binary_targets(y_true_arr)
    y_prob_arr = _clip_probs(y_prob_arr)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true_arr)

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (y_prob_arr >= lo) & (y_prob_arr <= hi)
        else:
            mask = (y_prob_arr >= lo) & (y_prob_arr < hi)

        if not np.any(mask):
            continue

        bin_true = y_true_arr[mask]
        bin_prob = y_prob_arr[mask]

        avg_conf = float(np.mean(bin_prob))
        avg_acc = float(np.mean(bin_true))
        weight = float(mask.sum()) / float(n)

        ece += weight * abs(avg_acc - avg_conf)

    return float(ece)


def summarize_binary_calibration(y_true: Any, y_prob: Any) -> Dict[str, float]:
    return {
        "log_loss": binary_log_loss(y_true, y_prob),
        "brier_score": brier_score(y_true, y_prob),
        "ece_10": expected_calibration_error(y_true, y_prob, n_bins=10),
    }


def fit_platt_scaler(
    y_true: Any,
    y_prob: Any,
    lr: float = 0.05,
    n_iter: int = 2000,
) -> Dict[str, float]:
    """
    Fits a simple Platt-style calibrator on probabilities:
        calibrated = sigmoid(a * logit(p) + b)
    """

    y_true_arr = _to_numpy_1d(y_true, "y_true")
    y_prob_arr = _to_numpy_1d(y_prob, "y_prob")

    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError("y_true and y_prob must have same length")

    _validate_binary_targets(y_true_arr)
    y_prob_arr = _clip_probs(y_prob_arr)

    logits = np.log(y_prob_arr / (1.0 - y_prob_arr))

    a = 1.0
    b = 0.0
    n = float(len(y_true_arr))

    for _ in range(n_iter):
        z = a * logits + b
        pred = 1.0 / (1.0 + np.exp(-z))

        error = pred - y_true_arr
        grad_a = float(np.dot(error, logits) / n)
        grad_b = float(np.sum(error) / n)

        a -= lr * grad_a
        b -= lr * grad_b

    return {"a": float(a), "b": float(b)}


def apply_platt_scaler(
    y_prob: Any,
    a: float,
    b: float,
) -> np.ndarray:
    y_prob_arr = _to_numpy_1d(y_prob, "y_prob")
    y_prob_arr = _clip_probs(y_prob_arr)

    logits = np.log(y_prob_arr / (1.0 - y_prob_arr))
    z = a * logits + b
    calibrated = 1.0 / (1.0 + np.exp(-z))
    return calibrated


def fit_and_apply_platt_scaler(
    y_true: Any,
    y_prob: Any,
    lr: float = 0.05,
    n_iter: int = 2000,
) -> Dict[str, Any]:
    y_true_arr = _to_numpy_1d(y_true, "y_true")
    y_prob_arr = _to_numpy_1d(y_prob, "y_prob")

    params = fit_platt_scaler(y_true_arr, y_prob_arr, lr=lr, n_iter=n_iter)
    calibrated = apply_platt_scaler(y_prob_arr, a=params["a"], b=params["b"])

    result = CalibrationResult(
        method="platt",
        a=params["a"],
        b=params["b"],
        metrics_before=summarize_binary_calibration(y_true_arr, y_prob_arr),
        metrics_after=summarize_binary_calibration(y_true_arr, calibrated),
    )

    return {
        "params": params,
        "calibrated_probabilities": calibrated,
        "summary": result.to_dict(),
    }