from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd


def _validate_dataset(df: pd.DataFrame, timestamp_col: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    if len(df) == 0:
        raise ValueError("df is empty")

    if timestamp_col not in df.columns:
        raise ValueError(f"missing timestamp column: {timestamp_col}")


def _prepare_sorted_dataset(
    df: pd.DataFrame,
    timestamp_col: str,
) -> pd.DataFrame:
    prepared = df.copy()

    prepared[timestamp_col] = pd.to_datetime(prepared[timestamp_col], errors="coerce", utc=True)

    if prepared[timestamp_col].isna().any():
        bad_count = int(prepared[timestamp_col].isna().sum())
        raise ValueError(f"{timestamp_col} contains {bad_count} invalid timestamps")

    prepared = prepared.sort_values(timestamp_col).reset_index(drop=True)
    return prepared


def _compute_split_indices(
    n_rows: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[int, int]:
    if n_rows < 3:
        raise ValueError("need at least 3 rows to create train/val/test splits")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("all split ratios must be > 0")

    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)

    if train_end <= 0:
        train_end = 1
    if val_end <= train_end:
        val_end = train_end + 1
    if val_end >= n_rows:
        val_end = n_rows - 1

    if train_end <= 0 or val_end <= train_end or val_end >= n_rows:
        raise ValueError("unable to create valid non-overlapping splits with current dataset/ratios")

    return train_end, val_end


def split_dataset_by_time(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp_utc",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, pd.DataFrame]:
    """
    Strict time-based split.
    Oldest rows -> train
    Middle rows -> validation
    Newest rows -> test
    """

    _validate_dataset(df, timestamp_col)
    prepared = _prepare_sorted_dataset(df, timestamp_col)

    train_end, val_end = _compute_split_indices(
        n_rows=len(prepared),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_df = prepared.iloc[:train_end].reset_index(drop=True)
    val_df = prepared.iloc[train_end:val_end].reset_index(drop=True)
    test_df = prepared.iloc[val_end:].reset_index(drop=True)

    return {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }


def summarize_splits(
    splits: Dict[str, pd.DataFrame],
    timestamp_col: str = "timestamp_utc",
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    for split_name, split_df in splits.items():
        row_count = len(split_df)

        if row_count == 0:
            summary[split_name] = {
                "rows": 0,
                "start": None,
                "end": None,
            }
            continue

        summary[split_name] = {
            "rows": row_count,
            "start": str(split_df[timestamp_col].min()),
            "end": str(split_df[timestamp_col].max()),
        }

    return summary


def validate_split_integrity(
    splits: Dict[str, pd.DataFrame],
    timestamp_col: str = "timestamp_utc",
) -> Dict[str, Any]:
    errors = []
    warnings = []

    required = ["train", "validation", "test"]
    for key in required:
        if key not in splits:
            errors.append(f"missing split: {key}")

    if errors:
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
        }

    train_df = splits["train"]
    val_df = splits["validation"]
    test_df = splits["test"]

    if len(train_df) == 0:
        errors.append("train split is empty")
    if len(val_df) == 0:
        errors.append("validation split is empty")
    if len(test_df) == 0:
        errors.append("test split is empty")

    if not errors:
        train_end = train_df[timestamp_col].max()
        val_start = val_df[timestamp_col].min()
        val_end = val_df[timestamp_col].max()
        test_start = test_df[timestamp_col].min()

        if train_end >= val_start:
            errors.append("train and validation overlap or touch incorrectly")
        if val_end >= test_start:
            errors.append("validation and test overlap or touch incorrectly")

    total_rows = len(train_df) + len(val_df) + len(test_df)
    if total_rows < 50:
        warnings.append("dataset is very small; split metrics may be unstable")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }