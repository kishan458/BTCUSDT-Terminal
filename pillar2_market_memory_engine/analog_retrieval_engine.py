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
from pillar2_market_memory_engine.state_signature_engine import (  # noqa: E402
    StateSignatureConfig,
    run_state_signature_engine,
)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class AnalogRetrievalConfig:
    min_similarity_score: float = 0.45
    max_weighted_matches: int = 300
    exclude_current_bar: bool = True

    # field weights for symbolic scoring
    weight_session: float = 0.08
    weight_session_transition: float = 0.05
    weight_weekday: float = 0.06
    weight_weekend_flag: float = 0.03
    weight_volatility_bucket: float = 0.16
    weight_expansion_state: float = 0.10
    weight_compression_state: float = 0.06
    weight_momentum_state: float = 0.16
    weight_path_efficiency_state: float = 0.08
    weight_overlap_state: float = 0.08
    weight_follow_through_quality: float = 0.08
    weight_pressure_bias: float = 0.08
    weight_breakout_state: float = 0.05
    weight_range_position: float = 0.03

    exact_match_core_fields: tuple[str, ...] = (
        "volatility_bucket",
        "expansion_state",
        "momentum_state",
        "overlap_state",
        "follow_through_quality",
        "pressure_bias",
    )

    partial_match_required_fields: tuple[str, ...] = (
        "volatility_bucket",
        "momentum_state",
        "overlap_state",
    )


# ============================================================
# RUNTIME DERIVED STATE FIELDS FOR HISTORICAL ROWS
# ============================================================

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


def _classify_breakout_state(range_expansion_ratio: Any, close_location_value: Any, body_pct_of_range: Any) -> Optional[str]:
    if pd.isna(range_expansion_ratio) or pd.isna(close_location_value) or pd.isna(body_pct_of_range):
        return None

    rer = float(range_expansion_ratio)
    clv = float(close_location_value)
    bpr = float(body_pct_of_range)

    if rer >= 1.5 and bpr >= 0.6:
        if clv >= 0.8 or clv <= 0.2:
            return "CONFIRMED"
        return "ATTEMPT"

    if rer >= 1.2 and bpr < 0.35:
        return "FAILED"

    return "NONE"


def add_runtime_signature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["path_efficiency_state"] = out["directional_efficiency_6"].apply(_classify_path_efficiency_state)
    out["overlap_state_runtime"] = out["overlap_pct_prev_bar"].apply(_classify_overlap_state)

    out["follow_through_quality_runtime"] = [
        _classify_follow_through_quality(pvr, eff)
        for pvr, eff in zip(out["progress_vs_range_6"], out["directional_efficiency_6"])
    ]

    out["pressure_bias_runtime"] = [
        _classify_pressure_bias(body, clv)
        for body, clv in zip(out["body"], out["close_location_value"])
    ]

    out["breakout_state_runtime"] = [
        _classify_breakout_state(rer, clv, bpr)
        for rer, clv, bpr in zip(
            out["range_expansion_ratio"],
            out["close_location_value"],
            out["body_pct_of_range"],
        )
    ]

    return out


# ============================================================
# MATCH LOGIC
# ============================================================

def _normalize_weights(config: AnalogRetrievalConfig) -> Dict[str, float]:
    weights = {
        "session": config.weight_session,
        "session_transition": config.weight_session_transition,
        "weekday": config.weight_weekday,
        "weekend_flag": config.weight_weekend_flag,
        "volatility_bucket": config.weight_volatility_bucket,
        "expansion_state": config.weight_expansion_state,
        "compression_state": config.weight_compression_state,
        "momentum_state": config.weight_momentum_state,
        "path_efficiency_state": config.weight_path_efficiency_state,
        "overlap_state": config.weight_overlap_state,
        "follow_through_quality": config.weight_follow_through_quality,
        "pressure_bias": config.weight_pressure_bias,
        "breakout_state": config.weight_breakout_state,
        "range_position": config.weight_range_position,
    }

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Total weight must be > 0.")

    return {k: v / total for k, v in weights.items()}


def _historical_row_field_value(row: pd.Series, field: str) -> Any:
    mapping = {
        "session": "session_label",
        "session_transition": "session_transition_label",
        "weekday": "weekday_name",
        "weekend_flag": "is_weekend",
        "volatility_bucket": "volatility_bucket",
        "expansion_state": "expansion_state",
        "compression_state": "compression_state",
        "momentum_state": "momentum_state",
        "path_efficiency_state": "path_efficiency_state",
        "overlap_state": "overlap_state_runtime",
        "follow_through_quality": "follow_through_quality_runtime",
        "pressure_bias": "pressure_bias_runtime",
        "breakout_state": "breakout_state_runtime",
        "range_position": "range_position",
    }

    col = mapping[field]
    return row[col] if col in row.index else None


def _signature_field_value(signature: Dict[str, Any], field: str) -> Any:
    return signature.get(field)


def _field_matches(a: Any, b: Any) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    return a == b


def compute_symbolic_similarity_scores(
    df: pd.DataFrame,
    signature: Dict[str, Any],
    config: AnalogRetrievalConfig,
) -> pd.DataFrame:
    out = df.copy()
    weights = _normalize_weights(config)

    scores = np.zeros(len(out), dtype=np.float64)
    matched_weight = np.zeros(len(out), dtype=np.float64)

    for field, weight in weights.items():
        hist_vals = out.apply(lambda row: _historical_row_field_value(row, field), axis=1)
        sig_val = _signature_field_value(signature, field)

        match_mask = hist_vals == sig_val
        valid_mask = hist_vals.notna() & pd.notna(sig_val)

        scores += np.where(match_mask & valid_mask, weight, 0.0)
        matched_weight += np.where(valid_mask, weight, 0.0)

    out["analog_similarity_score"] = scores
    out["analog_valid_weight_coverage"] = matched_weight

    return out


def add_exact_and_partial_match_flags(
    df: pd.DataFrame,
    signature: Dict[str, Any],
    config: AnalogRetrievalConfig,
) -> pd.DataFrame:
    out = df.copy()

    exact_flags = []
    partial_flags = []

    for _, row in out.iterrows():
        exact_ok = True
        for field in config.exact_match_core_fields:
            hist_val = _historical_row_field_value(row, field)
            sig_val = _signature_field_value(signature, field)
            if not _field_matches(hist_val, sig_val):
                exact_ok = False
                break

        partial_ok = True
        for field in config.partial_match_required_fields:
            hist_val = _historical_row_field_value(row, field)
            sig_val = _signature_field_value(signature, field)
            if not _field_matches(hist_val, sig_val):
                partial_ok = False
                break

        exact_flags.append(exact_ok)
        partial_flags.append(partial_ok)

    out["exact_match_flag"] = exact_flags
    out["partial_match_flag"] = partial_flags

    return out


def filter_usable_analogs(
    df: pd.DataFrame,
    signature: Dict[str, Any],
    config: AnalogRetrievalConfig,
) -> pd.DataFrame:
    out = df.copy()

    # Exclude current bar from analog search
    if config.exclude_current_bar and "timestamp_utc" in out.columns and "timestamp_utc" in signature:
        out = out[out["timestamp_utc"].astype(str) != str(signature["timestamp_utc"])].copy()

    # Need usable core runtime-derived state
    required_cols = [
        "volatility_bucket",
        "expansion_state",
        "momentum_state",
        "path_efficiency_state",
        "overlap_state_runtime",
        "follow_through_quality_runtime",
        "pressure_bias_runtime",
        "breakout_state_runtime",
        "range_position",
    ]
    out = out.dropna(subset=required_cols, how="any").copy()

    # Must have usable outcomes later
    out = out.dropna(subset=["fwd_return_3", "fwd_return_6"], how="any").copy()

    return out


def retrieve_analog_sets(
    df: pd.DataFrame,
    signature: Dict[str, Any],
    config: AnalogRetrievalConfig,
) -> Dict[str, pd.DataFrame]:
    base = add_runtime_signature_columns(df)
    base = filter_usable_analogs(base, signature, config)
    base = compute_symbolic_similarity_scores(base, signature, config)
    base = add_exact_and_partial_match_flags(base, signature, config)

    exact_matches = (
        base[base["exact_match_flag"]]
        .sort_values(["analog_similarity_score", "timestamp_utc"], ascending=[False, True])
        .reset_index(drop=True)
    )

    partial_matches = (
        base[base["partial_match_flag"]]
        .sort_values(["analog_similarity_score", "timestamp_utc"], ascending=[False, True])
        .reset_index(drop=True)
    )

    weighted_matches = (
        base[base["analog_similarity_score"] >= config.min_similarity_score]
        .sort_values(
            ["analog_similarity_score", "exact_match_flag", "partial_match_flag", "timestamp_utc"],
            ascending=[False, False, False, True],
        )
        .head(config.max_weighted_matches)
        .reset_index(drop=True)
    )

    return {
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "weighted_matches": weighted_matches,
    }


def build_analog_summary(analog_sets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    exact_df = analog_sets["exact_matches"]
    partial_df = analog_sets["partial_matches"]
    weighted_df = analog_sets["weighted_matches"]

    if weighted_df.empty:
        avg_score = np.nan
        avg_coverage = np.nan
    else:
        avg_score = float(weighted_df["analog_similarity_score"].mean())
        avg_coverage = float(weighted_df["analog_valid_weight_coverage"].mean())

    return {
        "match_count": int(len(weighted_df)),
        "exact_match_count": int(len(exact_df)),
        "partial_match_count": int(len(partial_df)),
        "weighted_match_count": int(len(weighted_df)),
        "analog_quality_score": avg_score,
        "average_valid_weight_coverage": avg_coverage,
    }


# ============================================================
# RUNTIME
# ============================================================

def run_analog_retrieval_engine(
    feature_config: Optional[MemoryFeatureConfig] = None,
    outcome_config: Optional[MemoryOutcomeConfig] = None,
    signature_config: Optional[StateSignatureConfig] = None,
    retrieval_config: Optional[AnalogRetrievalConfig] = None,
) -> Dict[str, Any]:
    if feature_config is None:
        feature_config = MemoryFeatureConfig()

    if outcome_config is None:
        outcome_config = MemoryOutcomeConfig()

    if signature_config is None:
        signature_config = StateSignatureConfig()

    if retrieval_config is None:
        retrieval_config = AnalogRetrievalConfig()

    df = run_memory_outcome_engine(
        feature_config=feature_config,
        outcome_config=outcome_config,
    )

    signature = run_state_signature_engine(
        feature_config=feature_config,
        outcome_config=outcome_config,
        signature_config=signature_config,
    )

    analog_sets = retrieve_analog_sets(df, signature, retrieval_config)
    summary = build_analog_summary(analog_sets)

    return {
        "current_state_signature": signature,
        "historical_analogs": summary,
        "analog_sets": analog_sets,
    }


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

    retrieval_cfg = AnalogRetrievalConfig(
        min_similarity_score=0.45,
        max_weighted_matches=300,
        exclude_current_bar=True,
    )

    result = run_analog_retrieval_engine(
        feature_config=feature_cfg,
        outcome_config=outcome_cfg,
        signature_config=signature_cfg,
        retrieval_config=retrieval_cfg,
    )

    print("\n=== ANALOG RETRIEVAL ENGINE SUCCESS ===")
    print("\nHistorical analog summary:\n")
    for k, v in result["historical_analogs"].items():
        print(f"{k}: {v}")

    weighted = result["analog_sets"]["weighted_matches"]
    preview_cols = [
        "timestamp_utc",
        "session_label",
        "volatility_bucket",
        "momentum_state",
        "analog_similarity_score",
        "exact_match_flag",
        "partial_match_flag",
        "fwd_return_3",
        "fwd_return_6",
    ]

    print("\nTop 10 weighted analogs:\n")
    if weighted.empty:
        print("No weighted analogs found.")
    else:
        print(weighted[preview_cols].head(10).to_string(index=False))