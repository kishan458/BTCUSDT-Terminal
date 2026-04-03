from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from pillar7_ml_council.shared_state_builder import build_shared_state
from pillar7_ml_council.council_feature_store import build_feature_row
from pillar7_ml_council.label_engine import build_label_set


def _validate_inputs(price_df: pd.DataFrame) -> None:
    if not isinstance(price_df, pd.DataFrame):
        raise ValueError("price_df must be a pandas DataFrame")

    if "close" not in price_df.columns:
        raise ValueError("price_df must contain 'close' column")

    if len(price_df) == 0:
        raise ValueError("price_df is empty")


# ============================================================
# CORE BUILDER
# ============================================================

def build_ml_dataset(
    price_df: pd.DataFrame,
    shared_states: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Combines:
    - shared_state → feature rows
    - price_df → labels

    Assumption:
    shared_states[i] corresponds to price_df.iloc[i]
    """

    _validate_inputs(price_df)

    if len(shared_states) != len(price_df):
        raise ValueError(
            f"Length mismatch: price_df={len(price_df)} vs shared_states={len(shared_states)}"
        )

    # ---------------------------------------
    # STEP 1 — BUILD FEATURES
    # ---------------------------------------
    feature_rows: List[Dict[str, Any]] = []

    for state in shared_states:
        feature_row = build_feature_row(state)
        feature_rows.append(feature_row)

    features_df = pd.DataFrame(feature_rows)

    # ---------------------------------------
    # STEP 2 — BUILD LABELS
    # ---------------------------------------
    labels_df = build_label_set(price_df)

    # ---------------------------------------
    # STEP 3 — MERGE
    # ---------------------------------------
    dataset = pd.concat([features_df, labels_df], axis=1)

    return dataset


# ============================================================
# SIMPLE GENERATOR (FOR TESTING ONLY)
# ============================================================

def build_dummy_shared_states(
    price_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    TEMP helper so you can test dataset_builder without real pillars.

    This should NOT be used in production.
    """

    states: List[Dict[str, Any]] = []

    for _ in range(len(price_df)):
        state = build_shared_state(
            pillar1_output={"sentiment": "NEUTRAL", "confidence": 0.5},
            pillar2_output={"memory_bias": "NEUTRAL", "analog_quality": 0.5},
            pillar3_output={"structure_state": "RANGE", "trap_risk": "LOW"},
            pillar4_output={"candle_intent": "NEUTRAL", "breakout_quality": "WEAK"},
            pillar5_output={"market_regime": "RANGE", "cycle_phase": "MID"},
            pillar6_output={"state": "IDLE", "base_uncertainty": 0.2, "trade_restrictions": {"allow_trade": True}},
        )
        states.append(state)

    return states


# ============================================================
# VALIDATOR
# ============================================================

def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or len(df) == 0:
        return {
            "is_valid": False,
            "errors": ["dataset is empty"],
            "warnings": [],
        }

    required_cols = [
        "asset",
        "timestamp_utc",
        "sentiment_state",
        "structure_state",
        "regime_state",
        "event_state",
        "fwd_ret_1",
        "long_quality_6",
    ]

    for col in required_cols:
        if col not in df.columns:
            errors.append(f"missing column: {col}")

    if "allow_trade" in df.columns:
        restricted_ratio = (df["allow_trade"] == 0).mean()
        if restricted_ratio > 0.5:
            warnings.append("more than 50% of rows are trade-restricted")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "rows": len(df),
        "columns": len(df.columns),
    }