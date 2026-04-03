from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


PROFESSOR_TARGET_COLUMN = "professor_target_v1"

PROFESSOR_CLASS_MAP = {
    0: "NO_TRADE",
    1: "LONG",
    2: "SHORT",
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_high_risk_row(row: pd.Series) -> bool:
    event_uncertainty = _safe_float(row.get("event_base_uncertainty"))
    allow_trade = _safe_int(row.get("allow_trade"))
    trap_risk = str(row.get("trap_risk", "UNKNOWN"))
    liquidation_risk = str(row.get("liquidation_risk", "UNKNOWN"))
    risk_flag_count = _safe_int(row.get("risk_flag_count"))

    if allow_trade == 0:
        return True

    if event_uncertainty is not None and event_uncertainty >= 0.70:
        return True

    if trap_risk in {"HIGH", "ELEVATED"}:
        return True

    if liquidation_risk in {"HIGH", "ELEVATED"}:
        return True

    if risk_flag_count is not None and risk_flag_count >= 2:
        return True

    return False


def _has_minimum_support(row: pd.Series) -> bool:
    analog_quality = _safe_float(row.get("analog_quality"))
    sentiment_conf = _safe_float(row.get("sentiment_confidence"))
    regime_state = str(row.get("regime_state", "UNKNOWN"))
    structure_state = str(row.get("structure_state", "UNKNOWN"))

    if analog_quality is None or analog_quality < 0.55:
        return False

    if sentiment_conf is not None and sentiment_conf < 0.45:
        return False

    if regime_state == "UNKNOWN":
        return False

    if structure_state == "UNKNOWN":
        return False

    return True


def _is_long_setup(row: pd.Series) -> bool:
    regime_state = str(row.get("regime_state", "UNKNOWN"))
    structure_state = str(row.get("structure_state", "UNKNOWN"))
    candle_state = str(row.get("candle_state", "UNKNOWN"))
    memory_state = str(row.get("memory_state", "UNKNOWN"))
    event_uncertainty = _safe_float(row.get("event_base_uncertainty"))

    if regime_state not in {"TREND", "BULLISH", "UPTREND"}:
        return False

    if structure_state not in {"UPTREND", "HIGHER_HIGH_HIGHER_LOW", "BULLISH"}:
        return False

    if candle_state not in {"CONTINUATION", "BULLISH", "NEUTRAL"}:
        return False

    if memory_state not in {"BULLISH", "NEUTRAL"}:
        return False

    if event_uncertainty is not None and event_uncertainty >= 0.50:
        return False

    return True


def _is_short_setup(row: pd.Series) -> bool:
    regime_state = str(row.get("regime_state", "UNKNOWN"))
    structure_state = str(row.get("structure_state", "UNKNOWN"))
    candle_state = str(row.get("candle_state", "UNKNOWN"))
    memory_state = str(row.get("memory_state", "UNKNOWN"))
    event_uncertainty = _safe_float(row.get("event_base_uncertainty"))

    if regime_state not in {"DOWNTREND", "BEARISH"}:
        return False

    if structure_state not in {"DOWNTREND", "LOWER_LOW_LOWER_HIGH", "BEARISH"}:
        return False

    if candle_state not in {"CONTINUATION", "BEARISH", "NEUTRAL"}:
        return False

    if memory_state not in {"BEARISH", "NEUTRAL"}:
        return False

    if event_uncertainty is not None and event_uncertainty >= 0.50:
        return False

    return True


def assign_professor_target(row: pd.Series) -> int:
    """
    V1 disciplined professor policy.

    Priority:
    1. If environment is risky/conflicted -> NO_TRADE
    2. If support quality is weak -> NO_TRADE
    3. If clean long setup -> LONG
    4. If clean short setup -> SHORT
    5. Else -> NO_TRADE
    """
    if _is_high_risk_row(row):
        return 0

    if not _has_minimum_support(row):
        return 0

    if _is_long_setup(row):
        return 1

    if _is_short_setup(row):
        return 2

    return 0


def build_professor_targets(dataset: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("dataset must be a pandas DataFrame")

    if len(dataset) == 0:
        raise ValueError("dataset is empty")

    df = dataset.copy()
    df[PROFESSOR_TARGET_COLUMN] = df.apply(assign_professor_target, axis=1)
    return df


def explain_professor_target(row: pd.Series) -> Dict[str, Any]:
    target = assign_professor_target(row)

    reasons: List[str] = []

    if _is_high_risk_row(row):
        reasons.append("high_risk_environment")

    if not _has_minimum_support(row):
        reasons.append("insufficient_support_quality")

    if target == 1:
        reasons.append("clean_long_alignment")

    if target == 2:
        reasons.append("clean_short_alignment")

    if target == 0 and not reasons:
        reasons.append("no_clean_asymmetric_setup")

    return {
        "target_value": target,
        "target_label": PROFESSOR_CLASS_MAP[target],
        "reasons": reasons,
    }


def summarize_professor_targets(dataset: pd.DataFrame) -> Dict[str, Any]:
    if PROFESSOR_TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"missing column: {PROFESSOR_TARGET_COLUMN}")

    counts = dataset[PROFESSOR_TARGET_COLUMN].value_counts(dropna=False).to_dict()

    return {
        "target_column": PROFESSOR_TARGET_COLUMN,
        "class_map": PROFESSOR_CLASS_MAP,
        "counts": counts,
        "no_trade_ratio": float((dataset[PROFESSOR_TARGET_COLUMN] == 0).mean()),
        "long_ratio": float((dataset[PROFESSOR_TARGET_COLUMN] == 1).mean()),
        "short_ratio": float((dataset[PROFESSOR_TARGET_COLUMN] == 2).mean()),
    }