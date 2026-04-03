from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from pillar7_ml_council.agents.retail_agent.feature_view import build_retail_feature_row
from pillar7_ml_council.shared_state_builder import build_shared_state
from pillar7_ml_council.council_feature_store import build_feature_row
from pillar7_ml_council.calibration_engine import apply_platt_scaler


CLASS_LABELS = {
    0: "NO_ACTION",
    1: "CHASE_LONG",
}


def _load_artifact(artifact_path: str | Path) -> Dict[str, Any]:
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {artifact_path}")

    with path.open("rb") as f:
        artifact = pickle.load(f)

    required_keys = [
        "model_name",
        "target_mode",
        "version",
        "raw_feature_columns",
        "encoded_feature_columns",
        "calibration_params",
        "model",
    ]
    missing = [k for k in required_keys if k not in artifact]
    if missing:
        raise ValueError(f"artifact missing keys: {missing}")

    return artifact


def _prepare_single_row_dataframe(retail_feature_row: Dict[str, Any]) -> pd.DataFrame:
    row_df = pd.DataFrame([retail_feature_row])

    for col in row_df.columns:
        if row_df[col].dtype == "object":
            row_df[col] = row_df[col].astype("string")

    row_df = pd.get_dummies(row_df, dummy_na=True)
    return row_df


def _align_to_training_columns(
    row_df: pd.DataFrame,
    encoded_feature_columns: List[str],
) -> pd.DataFrame:
    return row_df.reindex(columns=encoded_feature_columns, fill_value=0)


def infer_retail_from_feature_row(
    feature_row: Dict[str, Any],
    artifact_path: str | Path,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    artifact = _load_artifact(artifact_path)

    retail_view = build_retail_feature_row(feature_row)
    row_df = _prepare_single_row_dataframe(retail_view)
    row_df = _align_to_training_columns(row_df, artifact["encoded_feature_columns"])

    model = artifact["model"]
    raw_prob = float(model.predict_proba(row_df)[:, 1][0])

    cal = artifact["calibration_params"]
    calibrated_prob = float(
        apply_platt_scaler([raw_prob], a=cal["a"], b=cal["b"])[0]
    )

    predicted_class = 1 if calibrated_prob >= threshold else 0
    predicted_label = CLASS_LABELS[predicted_class]

    return {
        "model_name": artifact["model_name"],
        "target_mode": artifact["target_mode"],
        "version": artifact["version"],
        "threshold": threshold,
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "raw_probability": raw_prob,
        "calibrated_probability": calibrated_prob,
        "retail_feature_view": retail_view,
    }


def infer_retail_from_shared_state(
    shared_state: Dict[str, Any],
    artifact_path: str | Path,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    feature_row = build_feature_row(shared_state)
    return infer_retail_from_feature_row(
        feature_row=feature_row,
        artifact_path=artifact_path,
        threshold=threshold,
    )


def summarize_retail_inference(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_name": result.get("model_name"),
        "target_mode": result.get("target_mode"),
        "version": result.get("version"),
        "predicted_label": result.get("predicted_label"),
        "raw_probability": result.get("raw_probability"),
        "calibrated_probability": result.get("calibrated_probability"),
        "threshold": result.get("threshold"),
    }