from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from pillar4_candle_engine.absorption_engine import (
    AbsorptionConfig,
    build_absorption_context,
)
from pillar4_candle_engine.ai_overview_engine import build_ai_overview
from pillar4_candle_engine.breakout_quality_engine import (
    BreakoutQualityConfig,
    build_breakout_quality_context,
)
from pillar4_candle_engine.candle_features_engine import (
    CandleFeatureConfig,
    OhlcColumns,
    add_candle_features,
)
from pillar4_candle_engine.candle_intent_engine import (
    CandleIntentConfig,
    classify_candle_intents,
)
from pillar4_candle_engine.multi_candle_context_engine import (
    MultiCandleContextConfig,
    build_multi_candle_context,
)
from pillar4_candle_engine.pressure_engine import (
    PressureConfig,
    build_pressure_context,
)


@dataclass(frozen=True)
class Pillar4Config:
    candle_features: CandleFeatureConfig
    candle_intent: CandleIntentConfig
    multi_candle_context: MultiCandleContextConfig
    absorption: AbsorptionConfig
    breakout_quality: BreakoutQualityConfig
    pressure: PressureConfig


def run_pillar4_candle_intelligence(
    df: pd.DataFrame,
    pillar4_config: Pillar4Config,
    columns: OhlcColumns = OhlcColumns(),
    asset: str = "BTCUSDT",
    timeframe: str = "UNKNOWN",
    lookback_bars_used: int = 30,
    atr_method: str = "wilder",
    pillar3_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Full Pillar 4 structured output builder.
    AI overview is generated at the end from structured payload only.
    """

    feature_df = add_candle_features(
        df=df,
        config=pillar4_config.candle_features,
        columns=columns,
        atr_method=atr_method,
        copy=True,
    )

    intent_df = classify_candle_intents(
        feature_df=feature_df,
        config=pillar4_config.candle_intent,
    )

    context_df = build_multi_candle_context(
        feature_df=intent_df,
        config=pillar4_config.multi_candle_context,
    )

    absorption_df = build_absorption_context(
        feature_df=context_df,
        config=pillar4_config.absorption,
    )

    breakout_df = build_breakout_quality_context(
        feature_df=absorption_df,
        config=pillar4_config.breakout_quality,
    )

    final_df = build_pressure_context(
        feature_df=breakout_df,
        config=pillar4_config.pressure,
    )

    latest = final_df.iloc[-1]

    output: Dict[str, Any] = {
        "asset": asset,
        "timestamp_utc": _extract_timestamp(latest, columns.timestamp),
        "timeframe": timeframe,
        "lookback_bars_used": int(lookback_bars_used),

        "candle_summary": {
            "dominant_intent": latest["dominant_intent"],
            "intent_confidence": _to_float(latest["intent_confidence"]),
            "momentum_state": latest["momentum_state"],
            "control_state": latest["control_state"],
            "expansion_state": latest["expansion_state"],
            "overlap_state": latest["overlap_state"],
            "follow_through_quality": latest["follow_through_quality"],
            "exhaustion_state": latest["exhaustion_state"],
        },

        "latest_candle_features": {
            "direction": _to_int(latest["direction"]),
            "body_size": _to_float(latest["body_size"]),
            "full_range": _to_float(latest["full_range"]),
            "upper_wick": _to_float(latest["upper_wick"]),
            "lower_wick": _to_float(latest["lower_wick"]),
            "body_to_range_ratio": _to_float(latest["body_to_range_ratio"]),
            "upper_wick_to_range_ratio": _to_float(latest["upper_wick_to_range_ratio"]),
            "lower_wick_to_range_ratio": _to_float(latest["lower_wick_to_range_ratio"]),
            "total_wick_to_range_ratio": _to_float(latest["total_wick_to_range_ratio"]),
            "wick_imbalance": _to_float(latest["wick_imbalance"]),
            "bar_efficiency": _to_float(latest["bar_efficiency"]),
            "close_location_value": _to_float(latest["close_location_value"]),
            "open_location_value": _to_float(latest["open_location_value"]),
            "midpoint_displacement": _to_float(latest["midpoint_displacement"]),
            "gap_from_prev_close": _to_float(latest["gap_from_prev_close"]),
            "close_to_close_return": _to_float(latest["close_to_close_return"]),
            "high_extension_vs_prev_high": _to_float(latest["high_extension_vs_prev_high"]),
            "low_extension_vs_prev_low": _to_float(latest["low_extension_vs_prev_low"]),
            "inside_bar_flag": _to_int(latest["inside_bar_flag"]),
            "outside_bar_flag": _to_int(latest["outside_bar_flag"]),
            "true_range": _to_float(latest["true_range"]),
            "atr_scaled_range": _to_float(latest["atr_scaled_range"]),
            "atr_scaled_body": _to_float(latest["atr_scaled_body"]),
            "atr_scaled_gap": _to_float(latest["atr_scaled_gap"]),
            "range_expansion_score": _to_float(latest["range_expansion_score"]),
            "body_expansion_score": _to_float(latest["body_expansion_score"]),
            "range_zscore": _to_float(latest["range_zscore"]),
            "body_zscore": _to_float(latest["body_zscore"]),
            "overlap_ratio_vs_prev_bar": _to_float(latest["overlap_ratio_vs_prev_bar"]),
        },

        "volatility_context": {
            "atr": _to_float(latest["atr"]),
            "realized_volatility": _to_float(latest["realized_volatility"]),
            "realized_volatility_percentile": _to_float(latest["realized_volatility_percentile"]),
            "parkinson_vol": _to_float(latest["parkinson_vol"]),
            "garman_klass_vol": _to_float(latest["garman_klass_vol"]),
            "rogers_satchell_vol": _to_float(latest["rogers_satchell_vol"]),
            "yang_zhang_vol": _to_float(latest["yang_zhang_vol"]),
            "range_shock_percentile": _to_float(latest["range_shock_percentile"]),
            "body_shock_percentile": _to_float(latest["body_shock_percentile"]),
        },

        "multi_candle_context": {
            "same_direction_body_count": _to_int(latest["same_direction_body_count"]),
            "same_direction_close_count": _to_int(latest["same_direction_close_count"]),
            "close_upper_half_count_short": _to_float(latest["close_upper_half_count_short"]),
            "close_upper_half_count_medium": _to_float(latest["close_upper_half_count_medium"]),
            "close_lower_half_count_short": _to_float(latest["close_lower_half_count_short"]),
            "close_lower_half_count_medium": _to_float(latest["close_lower_half_count_medium"]),
            "rolling_body_dominance_short": _to_float(latest["rolling_body_dominance_short"]),
            "rolling_body_dominance_medium": _to_float(latest["rolling_body_dominance_medium"]),
            "rolling_body_dominance_long": _to_float(latest["rolling_body_dominance_long"]),
            "rolling_wick_dominance": _to_float(latest["rolling_wick_dominance"]),
            "rolling_sign_consistency_short": _to_float(latest["rolling_sign_consistency_short"]),
            "rolling_sign_consistency_medium": _to_float(latest["rolling_sign_consistency_medium"]),
            "avg_overlap_ratio_short": _to_float(latest["avg_overlap_ratio_short"]),
            "avg_overlap_ratio_long": _to_float(latest["avg_overlap_ratio_long"]),
            "net_progress_short": _to_float(latest["net_progress_short"]),
            "net_progress_medium": _to_float(latest["net_progress_medium"]),
            "net_progress_long": _to_float(latest["net_progress_long"]),
            "progress_efficiency_short": _to_float(latest["progress_efficiency_short"]),
            "progress_efficiency_medium": _to_float(latest["progress_efficiency_medium"]),
            "progress_efficiency_long": _to_float(latest["progress_efficiency_long"]),
            "inside_bar_frequency": _to_float(latest["inside_bar_frequency"]),
            "outside_bar_frequency": _to_float(latest["outside_bar_frequency"]),
            "range_contraction_streak": _to_int(latest["range_contraction_streak"]),
            "expansion_after_compression_score": _to_float(latest["expansion_after_compression_score"]),
            "post_expansion_fade_score": _to_float(latest["post_expansion_fade_score"]),
        },

        "intent_scores": {
            "bullish_continuation_score": max(
                _to_float(latest["intent_score_strong_bullish_continuation"]),
                _to_float(latest["intent_score_weak_bullish_continuation"]),
            ),
            "bearish_continuation_score": max(
                _to_float(latest["intent_score_strong_bearish_continuation"]),
                _to_float(latest["intent_score_weak_bearish_continuation"]),
            ),
            "buy_rejection_score": _to_float(latest["intent_score_buy_rejection"]),
            "sell_rejection_score": _to_float(latest["intent_score_sell_rejection"]),
            "buy_absorption_candidate_score": _to_float(latest["intent_score_buy_absorption_candidate"]),
            "sell_absorption_candidate_score": _to_float(latest["intent_score_sell_absorption_candidate"]),
            "indecision_score": _to_float(latest["intent_score_indecision"]),
            "inside_compression_score": _to_float(latest["intent_score_inside_compression"]),
            "outside_expansion_score": _to_float(latest["intent_score_outside_expansion"]),
            "exhaustion_up_candidate_score": _to_float(latest["intent_score_exhaustion_up_candidate"]),
            "exhaustion_down_candidate_score": _to_float(latest["intent_score_exhaustion_down_candidate"]),
        },

        "absorption": {
            "buy_absorption_score": _to_float(latest["buy_absorption_score"]),
            "sell_absorption_score": _to_float(latest["sell_absorption_score"]),
            "buy_rejection_score": _to_float(latest["buy_rejection_score"]),
            "sell_rejection_score": _to_float(latest["sell_rejection_score"]),
            "dominant_absorption": latest["dominant_absorption"],
            "dominant_rejection": latest["dominant_rejection"],
            "absorption_confidence": _to_float(latest["absorption_confidence"]),
            "failed_upside_extension_count": _to_float(latest["failed_upside_extension_count"]),
            "failed_downside_extension_count": _to_float(latest["failed_downside_extension_count"]),
        },

        "breakout_analysis": {
            "reference_range_high": _to_float_or_none(latest["reference_range_high"]),
            "reference_range_low": _to_float_or_none(latest["reference_range_low"]),
            "breakout_direction": latest["breakout_direction"],
            "breach_magnitude": _to_float(latest["breach_magnitude"]),
            "close_outside_range_ratio": _to_float(latest["close_outside_range_ratio"]),
            "acceptance_score": _to_float(latest["acceptance_score"]),
            "failure_score": _to_float(latest["failure_score"]),
            "wick_penalty": _to_float(latest["wick_penalty"]),
            "fake_breakout_risk": _to_float(latest["fake_breakout_risk"]),
            "breakout_quality_score": _to_float(latest["breakout_quality_score"]),
            "breakout_validity": latest["breakout_validity"],
            "breakout_state": latest["breakout_state"],
        },

        "pressure": {
            "buying_pressure_score": _to_float(latest["buying_pressure_score"]),
            "selling_pressure_score": _to_float(latest["selling_pressure_score"]),
            "net_pressure_score": _to_float(latest["net_pressure_score"]),
            "pressure_bias": latest["pressure_bias"],
            "pressure_strength": latest["pressure_strength"],
        },

        "sequence_similarity": {
            "enabled": False,
            "matched_pattern_family": None,
            "similarity_score": None,
            "historical_outcome_bias": None,
        },

        "context_alignment": _build_context_alignment(latest, pillar3_context),

        "risk_flags": _build_risk_flags(latest),

        "diagnostics": {
            "data_quality_ok": bool(latest["data_quality_ok"]),
            "ohlc_constraints_ok": bool(latest["ohlc_constraints_ok"]),
            "is_zero_range_bar": bool(latest["is_zero_range_bar"]),
            "nan_feature_count": _to_int(latest["nan_feature_count"]),
            "inf_feature_count": _to_int(latest["inf_feature_count"]),
        },

        "ai_overview": None,
    }

    output["ai_overview"] = build_ai_overview(output_payload_for_ai(output))
    return output


def output_payload_for_ai(output: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "asset": output.get("asset"),
        "timestamp_utc": output.get("timestamp_utc"),
        "timeframe": output.get("timeframe"),
        "lookback_bars_used": output.get("lookback_bars_used"),
        "candle_summary": output.get("candle_summary"),
        "latest_candle_features": output.get("latest_candle_features"),
        "volatility_context": output.get("volatility_context"),
        "multi_candle_context": output.get("multi_candle_context"),
        "intent_scores": output.get("intent_scores"),
        "absorption": output.get("absorption"),
        "breakout_analysis": output.get("breakout_analysis"),
        "pressure": output.get("pressure"),
        "context_alignment": output.get("context_alignment"),
        "risk_flags": output.get("risk_flags"),
        "diagnostics": output.get("diagnostics"),
    }


def _build_context_alignment(
    latest: pd.Series,
    pillar3_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if pillar3_context is None:
        return {
            "pillar3_liquidity_alignment": "NOT_AVAILABLE",
            "nearest_liquidity_magnet": None,
            "distance_to_nearest_liquidity_magnet_atr": None,
            "candle_vs_liquidity_story": "NOT_AVAILABLE",
            "pillar5_regime_alignment": "NOT_AVAILABLE",
            "pillar6_event_context": "NOT_AVAILABLE",
            "pillar2_memory_alignment": "NOT_AVAILABLE",
        }

    return {
        "pillar3_liquidity_alignment": pillar3_context.get("pillar3_liquidity_alignment", "NOT_AVAILABLE"),
        "nearest_liquidity_magnet": pillar3_context.get("nearest_liquidity_magnet"),
        "distance_to_nearest_liquidity_magnet_atr": pillar3_context.get("distance_to_nearest_liquidity_magnet_atr"),
        "candle_vs_liquidity_story": pillar3_context.get("candle_vs_liquidity_story", "NOT_AVAILABLE"),
        "pillar5_regime_alignment": pillar3_context.get("pillar5_regime_alignment", "NOT_AVAILABLE"),
        "pillar6_event_context": pillar3_context.get("pillar6_event_context", "NOT_AVAILABLE"),
        "pillar2_memory_alignment": pillar3_context.get("pillar2_memory_alignment", "NOT_AVAILABLE"),
    }


def _build_risk_flags(latest: pd.Series) -> list[str]:
    flags: list[str] = []

    if _to_float(latest["failure_score"]) >= 0.55:
        flags.append("Breakout failure risk is elevated")

    if _to_float(latest["fake_breakout_risk"]) >= 0.60:
        flags.append("Price action is behaving like a possible liquidity grab")

    if _to_float(latest["absorption_confidence"]) >= 0.50:
        flags.append("Absorption signal is meaningful and should not be ignored")

    if latest["follow_through_quality"] == "FAILED":
        flags.append("Follow-through has failed after recent movement")
    elif latest["follow_through_quality"] == "WEAK":
        flags.append("Follow-through remains weak")

    if latest["momentum_state"] in {"STALLING", "DECAYING", "REVERSAL_RISK"}:
        flags.append(f"Momentum context is {latest['momentum_state'].lower().replace('_', ' ')}")

    if latest["breakout_validity"] == "UNCONFIRMED":
        flags.append("Breakout acceptance is not yet confirmed")
    elif latest["breakout_validity"] == "FAILING":
        flags.append("Breakout attempt is currently failing")

    if latest["pressure_strength"] == "WEAK":
        flags.append("Directional pressure remains weak")

    if latest["exhaustion_state"] != "NONE":
        flags.append(f"Exhaustion condition detected: {latest['exhaustion_state']}")

    return flags


def _extract_timestamp(latest: pd.Series, timestamp_col: Optional[str]) -> Optional[str]:
    if timestamp_col is None or timestamp_col not in latest.index:
        return None

    value = latest[timestamp_col]
    if pd.isna(value):
        return None

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.isoformat() + "Z"
    return ts.tz_convert("UTC").isoformat()


def _to_float(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def _to_float_or_none(value: Any) -> Optional[float]:
    if pd.isna(value):
        return None
    return float(value)


def _to_int(value: Any) -> int:
    if pd.isna(value):
        return 0
    return int(value)