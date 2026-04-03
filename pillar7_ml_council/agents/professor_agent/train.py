from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from pillar7_ml_council.agents.professor_agent.feature_view import get_professor_feature_columns
from pillar7_ml_council.agents.professor_agent.label_policy import (
    PROFESSOR_TARGET_COLUMN,
    build_professor_targets,
)
from pillar7_ml_council.calibration_engine import apply_platt_scaler, fit_platt_scaler
from pillar7_ml_council.evaluation_engine import evaluate_binary_predictions, summarize_eval_result
from pillar7_ml_council.model_registry import ModelRegistry
from pillar7_ml_council.train_split_engine import (
    split_dataset_by_time,
    summarize_splits,
    validate_split_integrity,
)

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for professor_agent/train.py. "
        "Install it with: pip install scikit-learn"
    ) from exc


ARTIFACT_DIR = Path("pillar7_ml_council/artifacts")


def _ensure_artifact_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _validate_dataset(dataset: pd.DataFrame) -> None:
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("dataset must be a pandas DataFrame")
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    if "timestamp_utc" not in dataset.columns:
        raise ValueError("dataset missing required column: timestamp_utc")


def _prepare_professor_dataset(
    dataset: pd.DataFrame,
    target_mode: str = "long_vs_no_trade",
) -> pd.DataFrame:
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
        raise ValueError("no rows available after professor target filtering")

    return df


def _prepare_for_dummies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype("string")
    return out


def _build_feature_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    feature_cols = [c for c in get_professor_feature_columns() if c in train_df.columns]

    if not feature_cols:
        raise ValueError("no professor feature columns found in dataset")

    x_train = _prepare_for_dummies(train_df[feature_cols].copy())
    x_val = _prepare_for_dummies(val_df[feature_cols].copy())
    x_test = _prepare_for_dummies(test_df[feature_cols].copy())

    x_train = pd.get_dummies(x_train, dummy_na=True)
    x_val = pd.get_dummies(x_val, dummy_na=True)
    x_test = pd.get_dummies(x_test, dummy_na=True)

    x_val = x_val.reindex(columns=x_train.columns, fill_value=0)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    return x_train, x_val, x_test, feature_cols


def _class_info(y: pd.Series | Any) -> Dict[str, Any]:
    s = pd.Series(y)
    counts = s.value_counts(dropna=False).to_dict()
    unique_classes = sorted([float(v) for v in s.dropna().unique().tolist()])
    return {
        "counts": counts,
        "unique_classes": unique_classes,
        "n_unique": len(unique_classes),
    }


def _assert_split_has_both_classes(y: pd.Series | Any, split_name: str) -> None:
    info = _class_info(y)
    if info["n_unique"] < 2:
        raise ValueError(
            f"{split_name} split has only one class: {info['unique_classes']} "
            f"counts={info['counts']}. "
            f"You need both NO_TRADE and target class in this split."
        )


def train_professor_baseline(
    dataset: pd.DataFrame,
    *,
    target_mode: str = "long_vs_no_trade",
    model_name: str = "professor_policy_rf",
    version: str = "v1",
) -> Dict[str, Any]:
    """
    V1 baseline:
    - builds professor targets if missing
    - converts to binary task:
        long_vs_no_trade OR short_vs_no_trade
    - time splits dataset
    - trains RandomForest baseline
    - fits Platt calibration on validation probabilities
    - evaluates raw + calibrated test probabilities
    - saves artifact and registers model
    """

    _validate_dataset(dataset)
    _ensure_artifact_dir()

    df = _prepare_professor_dataset(dataset, target_mode=target_mode)

    splits = split_dataset_by_time(df, timestamp_col="timestamp_utc")
    split_validation = validate_split_integrity(splits)
    if not split_validation["is_valid"]:
        raise ValueError(f"invalid split integrity: {split_validation}")

    train_df = splits["train"]
    val_df = splits["validation"]
    test_df = splits["test"]

    _assert_split_has_both_classes(train_df["binary_target"], "train")
    _assert_split_has_both_classes(val_df["binary_target"], "validation")
    _assert_split_has_both_classes(test_df["binary_target"], "test")

    x_train, x_val, x_test, raw_feature_columns = _build_feature_frames(train_df, val_df, test_df)

    y_train = train_df["binary_target"].astype(float).values
    y_val = val_df["binary_target"].astype(float).values
    y_test = test_df["binary_target"].astype(float).values

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    if len(model.classes_) < 2:
        raise ValueError(
            f"model trained with only one class: {model.classes_.tolist()}. "
            "Training data is not diverse enough."
        )

    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    calibration_params = fit_platt_scaler(y_val, val_prob, lr=0.05, n_iter=2000)
    test_prob_calibrated = apply_platt_scaler(
        test_prob,
        a=calibration_params["a"],
        b=calibration_params["b"],
    )

    val_eval_raw = evaluate_binary_predictions(y_val, val_prob, threshold=0.5)
    test_eval_raw = evaluate_binary_predictions(y_test, test_prob, threshold=0.5)
    test_eval_calibrated = evaluate_binary_predictions(y_test, test_prob_calibrated, threshold=0.5)

    artifact_path = ARTIFACT_DIR / f"{model_name}_{target_mode}_{version}.pkl"
    artifact_payload = {
        "model_name": model_name,
        "target_mode": target_mode,
        "version": version,
        "model_type": "RandomForestClassifier",
        "raw_feature_columns": raw_feature_columns,
        "encoded_feature_columns": x_train.columns.tolist(),
        "calibration_params": calibration_params,
        "model": model,
    }

    with artifact_path.open("wb") as f:
        pickle.dump(artifact_payload, f)

    registry = ModelRegistry()
    registry_record = registry.register_model(
        model_name=model_name,
        agent_name="professor_agent",
        version=version,
        artifact_path=str(artifact_path),
        metrics={
            "val_raw": summarize_eval_result(val_eval_raw),
            "test_raw": summarize_eval_result(test_eval_raw),
            "test_calibrated": summarize_eval_result(test_eval_calibrated),
        },
        calibration_summary={
            "method": "platt",
            "a": calibration_params["a"],
            "b": calibration_params["b"],
        },
        train_summary={
            "rows_after_filter": int(len(df)),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "target_mode": target_mode,
            "positive_rate_train": float(train_df["binary_target"].mean()),
            "positive_rate_validation": float(val_df["binary_target"].mean()),
            "positive_rate_test": float(test_df["binary_target"].mean()),
            "class_info_train": _class_info(train_df["binary_target"]),
            "class_info_validation": _class_info(val_df["binary_target"]),
            "class_info_test": _class_info(test_df["binary_target"]),
        },
        split_summary=summarize_splits(splits),
        feature_columns=x_train.columns.tolist(),
        target_column="binary_target",
        notes=f"Professor baseline {target_mode} model",
    )

    return {
        "model_name": model_name,
        "target_mode": target_mode,
        "artifact_path": str(artifact_path),
        "split_summary": summarize_splits(splits),
        "split_validation": split_validation,
        "class_info_train": _class_info(train_df["binary_target"]),
        "class_info_validation": _class_info(val_df["binary_target"]),
        "class_info_test": _class_info(test_df["binary_target"]),
        "val_raw": summarize_eval_result(val_eval_raw),
        "test_raw": summarize_eval_result(test_eval_raw),
        "test_calibrated": summarize_eval_result(test_eval_calibrated),
        "registry_record": registry_record,
    }