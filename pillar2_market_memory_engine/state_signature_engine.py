from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import sys

import numpy as np
import pandas as pd


# ============================================================
# IMPORT FIX FOR DIRECT SCRIPT RUN
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pillar2_market_memory_engine.memory_feature_engine import (  # noqa: E402
    MemoryFeatureConfig,
)
from pillar2_market_memory_engine.memory_outcome_engine import (  # noqa: E402
    MemoryOutcomeConfig,
    run_memory_outcome_engine,
)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class StateSignatureConfig:
    require_complete_core_state: bool = True


# ============================================================
# HELPERS
# ============================================================

def _safe_str(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    return str(value)


def _safe_bool(value: Any) -> Optional[bool]:
    if pd.isna(value):
        return None
    return bool(value)


def _classify_path_efficiency_state(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None

    value = float(value)

    if value >= 0.70:
        return "VERY_HIGH_EFFICIENCY"
    if value >= 0.55:
        return "HIGH_EFFICIENCY"
    if value >= 0.35:
        return "MODERATE_EFFICIENCY"
    return "LOW_EFFICIENCY"


def _classify_overlap_state(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None

    value = float(value)

    if value >= 0.70:
        return "HIGH_OVERLAP"
    if value >= 0.35:
        return "MODERATE_OVERLAP"
    return "LOW_OVERLAP"


def _classify_follow_through_quality(progress_vs_range: Any, directional_efficiency: Any) -> Optional[str]:
    if pd.isna(progress_vs_range) or pd.isna(directional_efficiency):
        return None

    progress_vs_range = float(progress_vs_range)
    directional_efficiency = float(directional_efficiency)

    composite = 0.5 * progress_vs_range + 0.5 * directional_efficiency

    if composite >= 0.75:
        return "STRONG"
    if composite >= 0.50:
        return "MODERATE"
    if composite >= 0.30:
        return "WEAK"
    return "FAILING"


def _classify_pressure_bias(body: Any, close_location_value: Any) -> Optional[str]:
    if pd.isna(body) or pd.isna(close_location_value):
        return None

    body = float(body)
    close_location_value = float(close_location_value)

    if body > 0 and close_location_value >= 0.6:
        return "BUY_PRESSURE"
    if body < 0 and close_location_value <= 0.4:
        return "SELL_PRESSURE"
    return "BALANCED"


def _classify_breakout_state(
    range_expansion_ratio: Any,
    close_location_value: Any,
    body_pct_of_range: Any,
) -> Optional[str]:
    if pd.isna(range_expansion_ratio) or pd.isna(close_location_value) or pd.isna(body_pct_of_range):
        return None

    range_expansion_ratio = float(range_expansion_ratio)
    close_location_value = float(close_location_value)
    body_pct_of_range = float(body_pct_of_range)

    if range_expansion_ratio >= 1.5 and body_pct_of_range >= 0.6:
        if close_location_value >= 0.8 or close_location_value <= 0.2:
            return "CONFIRMED"
        return "ATTEMPT"

    if range_expansion_ratio >= 1.2 and body_pct_of_range < 0.35:
        return "FAILED"

    return "NONE"


def _classify_event_context(row: pd.Series) -> Optional[str]:
    if "event_context" in row and not pd.isna(row["event_context"]):
        return str(row["event_context"])
    return "NONE"


def _classify_regime_context(row: pd.Series) -> Optional[str]:
    if "volatility_regime" in row and not pd.isna(row["volatility_regime"]):
        vol_regime = str(row["volatility_regime"])
    else:
        vol_regime = None

    if "trend_regime" in row and not pd.isna(row["trend_regime"]):
        trend_regime = str(row["trend_regime"])
    else:
        trend_regime = None

    if vol_regime and trend_regime:
        return f"{vol_regime}__{trend_regime}"
    if vol_regime:
        return vol_regime
    if trend_regime:
        return trend_regime

    return "UNSPECIFIED"


# ============================================================
# CORE VALIDITY
# ============================================================

CORE_STATE_COLUMNS = [
    "timestamp_utc",
    "session_label",
    "session_transition_label",
    "weekday_name",
    "is_weekend",
    "volatility_bucket",
    "expansion_state",
    "compression_state",
    "momentum_state",
    "directional_efficiency_6",
    "overlap_pct_prev_bar",
    "progress_vs_range_6",
    "body",
    "close_location_value",
    "range_position",
    "range_expansion_ratio",
    "body_pct_of_range",
]


def _row_has_complete_core_state(row: pd.Series) -> bool:
    for col in CORE_STATE_COLUMNS:
        if col not in row.index:
            return False
        if pd.isna(row[col]):
            return False
    return True


def get_latest_valid_state_row(
    df: pd.DataFrame,
    require_complete_core_state: bool = True,
) -> pd.Series:
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    for idx in range(len(df) - 1, -1, -1):
        row = df.iloc[idx]

        if require_complete_core_state:
            if _row_has_complete_core_state(row):
                return row
        else:
            return row

    raise ValueError("No valid latest row found with complete core state.")


# ============================================================
# SIGNATURE BUILD
# ============================================================

def build_state_signature_from_row(row: pd.Series) -> Dict[str, Any]:
    signature = {
        "timestamp_utc": str(row["timestamp_utc"]),
        "session": _safe_str(row["session_label"]),
        "session_transition": _safe_str(row["session_transition_label"]),
        "weekday": _safe_str(row["weekday_name"]),
        "weekend_flag": _safe_bool(row["is_weekend"]),
        "volatility_bucket": _safe_str(row["volatility_bucket"]),
        "expansion_state": _safe_str(row["expansion_state"]),
        "compression_state": _safe_str(row["compression_state"]),
        "momentum_state": _safe_str(row["momentum_state"]),
        "path_efficiency_state": _classify_path_efficiency_state(row["directional_efficiency_6"]),
        "overlap_state": _classify_overlap_state(row["overlap_pct_prev_bar"]),
        "follow_through_quality": _classify_follow_through_quality(
            row["progress_vs_range_6"],
            row["directional_efficiency_6"],
        ),
        "pressure_bias": _classify_pressure_bias(
            row["body"],
            row["close_location_value"],
        ),
        "breakout_state": _classify_breakout_state(
            row["range_expansion_ratio"],
            row["close_location_value"],
            row["body_pct_of_range"],
        ),
        "range_position": _safe_str(row["range_position"]),
        "candle_intent": _safe_str(row["candle_intent"]) if "candle_intent" in row.index else None,
        "event_context": _classify_event_context(row),
        "regime_context": _classify_regime_context(row),
    }

    return signature


def build_state_signature(
    df: pd.DataFrame,
    config: Optional[StateSignatureConfig] = None,
) -> Dict[str, Any]:
    if config is None:
        config = StateSignatureConfig()

    latest_row = get_latest_valid_state_row(
        df,
        require_complete_core_state=config.require_complete_core_state,
    )

    return build_state_signature_from_row(latest_row)


# ============================================================
# RUNTIME
# ============================================================

def run_state_signature_engine(
    feature_config: Optional[MemoryFeatureConfig] = None,
    outcome_config: Optional[MemoryOutcomeConfig] = None,
    signature_config: Optional[StateSignatureConfig] = None,
) -> Dict[str, Any]:
    if feature_config is None:
        feature_config = MemoryFeatureConfig()

    if outcome_config is None:
        outcome_config = MemoryOutcomeConfig()

    if signature_config is None:
        signature_config = StateSignatureConfig()

    df = run_memory_outcome_engine(
        feature_config=feature_config,
        outcome_config=outcome_config,
    )

    return build_state_signature(df, signature_config)


if __name__ == "__main__":
    feature_cfg = MemoryFeatureConfig(
        db_path="database/btc_terminal.db",
        table_name="btc_price_1h",
        timestamp_col="timestamp",
        output_parquet_path=None,
    )

    outcome_cfg = MemoryOutcomeConfig(
        horizons=(1, 3, 6, 12),
        output_parquet_path=None,
    )

    signature_cfg = StateSignatureConfig(
        require_complete_core_state=True,
    )

    signature = run_state_signature_engine(
        feature_config=feature_cfg,
        outcome_config=outcome_cfg,
        signature_config=signature_cfg,
    )

    print("\n=== STATE SIGNATURE ENGINE SUCCESS ===")
    print("\nCurrent state signature:\n")
    for k, v in signature.items():
        print(f"{k}: {v}")