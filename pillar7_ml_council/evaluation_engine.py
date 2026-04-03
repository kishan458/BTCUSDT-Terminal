from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from pillar7_ml_council.calibration_engine import summarize_binary_calibration


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


def _validate_binary_predictions(y_pred: np.ndarray) -> None:
    unique = set(np.unique(y_pred).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError("y_pred must contain only 0/1 values")


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def threshold_probabilities(
    y_prob: Any,
    threshold: float = 0.5,
) -> np.ndarray:
    y_prob_arr = _to_numpy_1d(y_prob, "y_prob")

    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    return (y_prob_arr >= threshold).astype(float)


def confusion_counts(
    y_true: Any,
    y_pred: Any,
) -> Dict[str, int]:
    y_true_arr = _to_numpy_1d(y_true, "y_true")
    y_pred_arr = _to_numpy_1d(y_pred, "y_pred")

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have same length")

    _validate_binary_targets(y_true_arr)
    _validate_binary_predictions(y_pred_arr)

    tp = int(np.sum((y_true_arr == 1.0) & (y_pred_arr == 1.0)))
    tn = int(np.sum((y_true_arr == 0.0) & (y_pred_arr == 0.0)))
    fp = int(np.sum((y_true_arr == 0.0) & (y_pred_arr == 1.0)))
    fn = int(np.sum((y_true_arr == 1.0) & (y_pred_arr == 0.0)))

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def binary_classification_metrics(
    y_true: Any,
    y_pred: Any,
) -> Dict[str, float]:
    counts = confusion_counts(y_true, y_pred)

    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    specificity = _safe_div(tn, tn + fp)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
    }


def evaluate_binary_predictions(
    y_true: Any,
    y_prob: Any,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_true_arr = _to_numpy_1d(y_true, "y_true")
    y_prob_arr = _to_numpy_1d(y_prob, "y_prob")

    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError("y_true and y_prob must have same length")

    _validate_binary_targets(y_true_arr)

    y_pred_arr = threshold_probabilities(y_prob_arr, threshold=threshold)

    counts = confusion_counts(y_true_arr, y_pred_arr)
    cls_metrics = binary_classification_metrics(y_true_arr, y_pred_arr)
    cal_metrics = summarize_binary_calibration(y_true_arr, y_prob_arr)

    positive_rate = float(np.mean(y_true_arr))
    predicted_positive_rate = float(np.mean(y_pred_arr))

    return {
        "threshold": threshold,
        "counts": counts,
        "classification_metrics": cls_metrics,
        "calibration_metrics": cal_metrics,
        "support": {
            "n": int(len(y_true_arr)),
            "positive_rate": positive_rate,
            "predicted_positive_rate": predicted_positive_rate,
        },
    }


def summarize_eval_result(eval_result: Dict[str, Any]) -> Dict[str, Any]:
    cls = eval_result.get("classification_metrics", {})
    cal = eval_result.get("calibration_metrics", {})
    counts = eval_result.get("counts", {})
    support = eval_result.get("support", {})

    return {
        "n": support.get("n"),
        "threshold": eval_result.get("threshold"),
        "accuracy": cls.get("accuracy"),
        "precision": cls.get("precision"),
        "recall": cls.get("recall"),
        "f1": cls.get("f1"),
        "log_loss": cal.get("log_loss"),
        "brier_score": cal.get("brier_score"),
        "ece_10": cal.get("ece_10"),
        "tp": counts.get("tp"),
        "tn": counts.get("tn"),
        "fp": counts.get("fp"),
        "fn": counts.get("fn"),
    }