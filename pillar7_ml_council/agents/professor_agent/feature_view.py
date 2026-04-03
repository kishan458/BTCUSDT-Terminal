from __future__ import annotations

from typing import Any, Dict, List


PROFESSOR_FEATURE_COLUMNS: List[str] = [
    "sentiment_state",
    "sentiment_confidence",
    "memory_state",
    "memory_confidence",
    "structure_state",
    "structure_confidence",
    "candle_state",
    "candle_confidence",
    "regime_state",
    "regime_confidence",
    "event_state",
    "event_confidence",
    "institutional_vs_hype_spread",
    "analog_quality",
    "trap_risk",
    "liquidation_risk",
    "candle_intent",
    "cycle_phase",
    "event_base_uncertainty",
    "allow_trade",
    "size_multiplier",
    "leverage_cap",
    "risk_flag_count",
    "p1_sentiment",
    "p1_institutional_sentiment",
    "p1_sentiment_confidence",
    "p1_narrative_state",
    "p1_narrative_maturity",
    "p1_sentiment_divergence",
    "p2_memory_bias",
    "p2_match_quality",
    "p2_analog_quality",
    "p2_stability_score",
    "p2_stability_confidence",
    "p2_forward_bias",
    "p3_market_structure",
    "p3_structure_state",
    "p3_range_state",
    "p3_compression_state",
    "p3_nearest_liquidity_magnet",
    "p3_buy_side_liquidity",
    "p3_sell_side_liquidity",
    "p3_trap_risk",
    "p3_liquidation_risk",
    "p3_stop_hunt_risk",
    "p4_candle_intent",
    "p4_pressure_bias",
    "p4_breakout_quality",
    "p4_follow_through_quality",
    "p4_absorption_state",
    "p4_overlap_state",
    "p4_rejection_state",
    "p5_market_regime",
    "p5_volatility_regime",
    "p5_trend_regime",
    "p5_cycle_phase",
    "p5_strategy_compatibility",
    "p6_event_name",
    "p6_event_state",
    "p6_base_uncertainty",
    "p6_confidence_score",
    "p6_trade_allowed",
    "p6_size_multiplier",
    "p6_leverage_cap",
]


def get_professor_feature_columns() -> List[str]:
    return PROFESSOR_FEATURE_COLUMNS.copy()


def build_professor_feature_row(feature_row: Dict[str, Any]) -> Dict[str, Any]:
    return {col: feature_row.get(col) for col in PROFESSOR_FEATURE_COLUMNS}


def summarize_professor_feature_view(feature_view: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "feature_count": len(feature_view),
        "sentiment_state": feature_view.get("sentiment_state"),
        "memory_state": feature_view.get("memory_state"),
        "structure_state": feature_view.get("structure_state"),
        "candle_state": feature_view.get("candle_state"),
        "regime_state": feature_view.get("regime_state"),
        "event_state": feature_view.get("event_state"),
        "analog_quality": feature_view.get("analog_quality"),
        "event_base_uncertainty": feature_view.get("event_base_uncertainty"),
        "allow_trade": feature_view.get("allow_trade"),
        "risk_flag_count": feature_view.get("risk_flag_count"),
    }


def validate_professor_feature_view(feature_view: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    required_core = [
        "sentiment_state",
        "memory_state",
        "structure_state",
        "candle_state",
        "regime_state",
        "event_state",
        "analog_quality",
        "event_base_uncertainty",
        "allow_trade",
    ]

    for col in required_core:
        if col not in feature_view:
            errors.append(f"missing feature: {col}")

    if feature_view.get("allow_trade") == 0:
        warnings.append("trade is restricted for professor agent view")

    uncertainty = feature_view.get("event_base_uncertainty")
    if isinstance(uncertainty, (int, float)) and uncertainty >= 0.70:
        warnings.append("high event uncertainty")

    if feature_view.get("analog_quality") is None:
        warnings.append("analog_quality missing")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }