from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd

from pillar7_ml_council.agents.retail_agent.infer import (
    infer_retail_from_feature_row,
)
from pillar7_ml_council.evaluation_engine import (
    evaluate_binary_predictions,
)
from pillar7_ml_council.calibration_engine import summarize_binary_calibration


def evaluate_retail_model(
    dataset: pd.DataFrame,
    artifact_path: str | Path,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    required_cols = ["retail_target_v1"]
    for col in required_cols:
        if col not in dataset.columns:
            raise ValueError(f"missing column: {col}")

    dataset = dataset.copy()

    # convert to binary: CHASE_LONG (1) vs others (0)
    dataset["binary_target"] = (dataset["retail_target_v1"] == 1).astype(float)

    probs = []
    for _, row in dataset.iterrows():
        result = infer_retail_from_feature_row(
            feature_row=row.to_dict(),
            artifact_path=artifact_path,
            threshold=threshold,
        )
        probs.append(result["raw_probability"])

    y_true = dataset["binary_target"].values.astype(float)
    y_prob = np.array(probs, dtype=float)

    raw_eval = evaluate_binary_predictions(y_true, y_prob, threshold=threshold)
    calib_eval = summarize_binary_calibration(y_true, y_prob)

    return {
        "model_name": "retail_policy_rf",
        "target_mode": "long_vs_no_action",
        "rows_evaluated": len(dataset),
        "threshold": threshold,
        "raw_eval": raw_eval,
        "calibration_eval": calib_eval,
    }


def summarize_retail_evaluation(result: Dict[str, Any]) -> Dict[str, Any]:
    raw = result.get("raw_eval", {})
    return {
        "rows": result.get("rows_evaluated"),
        "accuracy": raw.get("classification_metrics", {}).get("accuracy"),
        "precision": raw.get("classification_metrics", {}).get("precision"),
        "recall": raw.get("classification_metrics", {}).get("recall"),
        "f1": raw.get("classification_metrics", {}).get("f1"),
        "brier_score": raw.get("calibration_metrics", {}).get("brier_score"),
        "log_loss": raw.get("calibration_metrics", {}).get("log_loss"),
    }