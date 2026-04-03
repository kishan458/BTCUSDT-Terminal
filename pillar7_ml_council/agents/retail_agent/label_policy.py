from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


RETAIL_TARGET_COLUMN = "retail_target_v1"
RETAIL_TRAP_COLUMN = "retail_trap_flag_v1"

RETAIL_CLASS_MAP = {
    0: "NO_ACTION",
    1: "CHASE_LONG",
    2: "CHASE_SHORT",
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


def _as_str(value: Any, default: str = "UNKNOWN") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value)


def _retail_long_impulse(row: pd.Series) -> bool:
    sentiment_state = _as_str(row.get("sentiment_state"))
    candle_state = _as_str(row.get("candle_state"))
    breakout_quality = _as_str(row.get("p4_breakout_quality"))
    pressure_bias = _as_str(row.get("p4_pressure_bias"))
    memory_state = _as_str(row.get("memory_state"))
    event_uncertainty = _safe_float(row.get("event_base_uncertainty"))

    positive_sentiment = sentiment_state in {"BULLISH", "POSITIVE"}
    bullish_candle = candle_state in {"CONTINUATION", "BULLISH"}
    breakout_visible = breakout_quality in {"STRONG", "VERY_STRONG"}
    buying_pressure = pressure_bias in {"BUY_PRESSURE", "BULLISH_PRESSURE", "UNKNOWN"}
    memory_support = memory_state in {"BULLISH", "NEUTRAL"}

    if event_uncertainty is not None and event_uncertainty >= 0.90:
        return False

    return positive_sentiment and bullish_candle and breakout_visible and buying_pressure and memory_support


def _retail_short_impulse(row: pd.Series) -> bool:
    sentiment_state = _as_str(row.get("sentiment_state"))
    candle_state = _as_str(row.get("candle_state"))
    breakout_quality = _as_str(row.get("p4_breakout_quality"))
    pressure_bias = _as_str(row.get("p4_pressure_bias"))
    memory_state = _as_str(row.get("memory_state"))
    event_uncertainty = _safe_float(row.get("event_base_uncertainty"))

    negative_sentiment = sentiment_state in {"BEARISH", "NEGATIVE"}
    bearish_candle = candle_state in {"CONTINUATION", "BEARISH"}
    breakdown_visible = breakout_quality in {"STRONG", "VERY_STRONG"}
    selling_pressure = pressure_bias in {"SELL_PRESSURE", "BEARISH_PRESSURE", "UNKNOWN"}
    memory_support = memory_state in {"BEARISH", "NEUTRAL"}

    if event_uncertainty is not None and event_uncertainty >= 0.90:
        return False

    return negative_sentiment and bearish_candle and breakdown_visible and selling_pressure and memory_support


def assign_retail_target(row: pd.Series) -> int:
    """
    Retail V1:
    - chase visible strong upside setups
    - chase visible strong downside setups
    - otherwise no action
    """
    if _retail_long_impulse(row):
        return 1

    if _retail_short_impulse(row):
        return 2

    return 0


def assign_retail_trap_flag(row: pd.Series) -> int:
    trap_risk = _as_str(row.get("trap_risk"))
    liquidation_risk = _as_str(row.get("liquidation_risk"))
    breakout_quality = _as_str(row.get("p4_breakout_quality"))
    event_uncertainty = _safe_float(row.get("event_base_uncertainty"))
    target = assign_retail_target(row)

    if target == 0:
        return 0

    if trap_risk in {"HIGH", "ELEVATED"}:
        return 1

    if liquidation_risk in {"HIGH", "ELEVATED"}:
        return 1

    if breakout_quality in {"STRONG", "VERY_STRONG"} and event_uncertainty is not None and event_uncertainty >= 0.70:
        return 1

    return 0


def build_retail_targets(dataset: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("dataset must be a pandas DataFrame")

    if len(dataset) == 0:
        raise ValueError("dataset is empty")

    df = dataset.copy()
    df[RETAIL_TARGET_COLUMN] = df.apply(assign_retail_target, axis=1)
    df[RETAIL_TRAP_COLUMN] = df.apply(assign_retail_trap_flag, axis=1)
    return df


def explain_retail_target(row: pd.Series) -> Dict[str, Any]:
    target = assign_retail_target(row)
    trap_flag = assign_retail_trap_flag(row)

    reasons: List[str] = []

    if target == 1:
        reasons.append("visible_upside_chase_setup")

    if target == 2:
        reasons.append("visible_downside_chase_setup")

    if target == 0:
        reasons.append("no_clean_retail_chase_setup")

    if trap_flag == 1:
        reasons.append("retail_trap_risk_present")

    return {
        "target_value": target,
        "target_label": RETAIL_CLASS_MAP[target],
        "trap_flag": trap_flag,
        "reasons": reasons,
    }


def summarize_retail_targets(dataset: pd.DataFrame) -> Dict[str, Any]:
    if RETAIL_TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"missing column: {RETAIL_TARGET_COLUMN}")

    if RETAIL_TRAP_COLUMN not in dataset.columns:
        raise ValueError(f"missing column: {RETAIL_TRAP_COLUMN}")

    counts = dataset[RETAIL_TARGET_COLUMN].value_counts(dropna=False).to_dict()
    trap_rate = float(dataset[RETAIL_TRAP_COLUMN].mean())

    return {
        "target_column": RETAIL_TARGET_COLUMN,
        "trap_column": RETAIL_TRAP_COLUMN,
        "class_map": RETAIL_CLASS_MAP,
        "counts": counts,
        "no_action_ratio": float((dataset[RETAIL_TARGET_COLUMN] == 0).mean()),
        "chase_long_ratio": float((dataset[RETAIL_TARGET_COLUMN] == 1).mean()),
        "chase_short_ratio": float((dataset[RETAIL_TARGET_COLUMN] == 2).mean()),
        "trap_flag_ratio": trap_rate,
    }