from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from pillar7_ml_council.agents.professor_agent.feature_view import build_professor_feature_row
from pillar7_ml_council.agents.professor_agent.label_policy import PROFESSOR_TARGET_COLUMN, build_professor_targets
from pillar7_ml_council.calibration_engine import apply_platt_scaler
from pillar7_ml_council.evaluation_engine import evaluate_binary_predictions, summarize_eval_result


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
        "encoded_feature_columns",
        "calibration_params",
        "model",
    ]
    missing = [k for k in required_keys if k not in artifact]
    if missing:
        raise ValueError(f"artifact missing keys: {missing}")

    return artifact


def _prepare_single_row_df(professor_view: Dict[str, Any], encoded_feature_columns: list[str]) -> pd.DataFrame:
    row_df = pd.DataFrame([professor_view])

    for col in row_df.columns:
        if row_df[col].dtype == "object":
            row_df[col] = row_df[col].astype("string")

    row_df = pd.get_dummies(row_df, dummy_na=True)
    row_df = row_df.reindex(columns=encoded_feature_columns, fill_value=0)
    return row_df


def _prepare_binary_target(dataset: pd.DataFrame, target_mode: str) -> pd.DataFrame:
    df = dataset.copy()

    if PROFESSOR_TARGET_COLUMN not in df.columns:
        df = build_professor_targets(df)

    if target_mode == "long_vs_no_trade":
        df = df[df[PROFESSOR_TARGET_COLUMN].isin([0, 1])].copy()
        df["binary_target"] = (df[PROFESSOR_TARGET_COLUMN] == 1).astype(float)
    elif target_mode == "short_vs_no_trade":
        df = df[df[PROFESSOR_TARGET_COLUMN].isin([0, 2])].copy()
        df["binary_target"] = (df[PROFESSOR_TARGET_COLUMN] == 2).astype(float)
    else:
        raise ValueError("target_mode must be 'long_vs_no_trade' or 'short_vs_no_trade'")

    if len(df) == 0:
        raise ValueError("no rows available after target-mode filtering")

    return df


def evaluate_professor_artifact_on_dataset(
    dataset: pd.DataFrame,
    artifact_path: str | Path,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("dataset must be a pandas DataFrame")
    if len(dataset) == 0:
        raise ValueError("dataset is empty")

    artifact = _load_artifact(artifact_path)
    df = _prepare_binary_target(dataset, artifact["target_mode"])

    raw_probs = []
    calibrated_probs = []

    model = artifact["model"]
    encoded_feature_columns = artifact["encoded_feature_columns"]
    cal = artifact["calibration_params"]

    for _, row in df.iterrows():
        professor_view = build_professor_feature_row(row.to_dict())
        x_row = _prepare_single_row_df(professor_view, encoded_feature_columns)

        raw_prob = float(model.predict_proba(x_row)[:, 1][0])
        calibrated_prob = float(apply_platt_scaler([raw_prob], a=cal["a"], b=cal["b"])[0])

        raw_probs.append(raw_prob)
        calibrated_probs.append(calibrated_prob)

    y_true = df["binary_target"].astype(float).values

    raw_eval = evaluate_binary_predictions(y_true, raw_probs, threshold=threshold)
    calibrated_eval = evaluate_binary_predictions(y_true, calibrated_probs, threshold=threshold)

    return {
        "model_name": artifact["model_name"],
        "target_mode": artifact["target_mode"],
        "version": artifact["version"],
        "rows_evaluated": int(len(df)),
        "threshold": threshold,
        "raw_eval": summarize_eval_result(raw_eval),
        "calibrated_eval": summarize_eval_result(calibrated_eval),
    }


def summarize_professor_batch_evaluation(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_name": result.get("model_name"),
        "target_mode": result.get("target_mode"),
        "version": result.get("version"),
        "rows_evaluated": result.get("rows_evaluated"),
        "threshold": result.get("threshold"),
        "raw_eval": result.get("raw_eval"),
        "calibrated_eval": result.get("calibrated_eval"),
    }