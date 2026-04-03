from __future__ import annotations

from typing import Any, Dict, List


RETAIL_FEATURE_COLUMNS: List[str] = [
    # what retail most visibly reacts to
    "sentiment_state",
    "sentiment_confidence",
    "memory_state",
    "candle_state",
    "regime_state",
    "event_state",
    "institutional_vs_hype_spread",
    "trap_risk",
    "liquidation_risk",
    "candle_intent",
    "event_base_uncertainty",
    "allow_trade",
    "risk_flag_count",
    # pillar 1 retail-facing narrative / sentiment
    "p1_sentiment",
    "p1_final_sentiment",
    "p1_institutional_sentiment",
    "p1_sentiment_confidence",
    "p1_narrative_state",
    "p1_narrative_maturity",
    "p1_sentiment_divergence",
    "p1_sentiment_shock",
    # pillar 2 simplified memory
    "p2_memory_bias",
    "p2_forward_bias",
    "p2_match_quality",
    "p2_analog_quality",
    # pillar 3 visible structure / danger
    "p3_structure_state",
    "p3_market_structure",
    "p3_range_state",
    "p3_compression_state",
    "p3_nearest_liquidity_magnet",
    "p3_trap_risk",
    "p3_liquidation_risk",
    "p3_stop_hunt_risk",
    # pillar 4 chart-impression features
    "p4_candle_intent",
    "p4_pressure_bias",
    "p4_breakout_quality",
    "p4_follow_through_quality",
    "p4_absorption_state",
    "p4_overlap_state",
    "p4_rejection_state",
    # pillar 5 broad market mode
    "p5_market_regime",
    "p5_volatility_regime",
    "p5_trend_regime",
    "p5_cycle_phase",
    "p5_market_state",
    # pillar 6 headline/event awareness
    "p6_event_name",
    "p6_event_state",
    "p6_base_uncertainty",
    "p6_confidence_score",
    "p6_trade_allowed",
]


def get_retail_feature_columns() -> List[str]:
    return RETAIL_FEATURE_COLUMNS.copy()


def build_retail_feature_row(feature_row: Dict[str, Any]) -> Dict[str, Any]:
    return {col: feature_row.get(col) for col in RETAIL_FEATURE_COLUMNS}


def summarize_retail_feature_view(feature_view: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "feature_count": len(feature_view),
        "sentiment_state": feature_view.get("sentiment_state"),
        "memory_state": feature_view.get("memory_state"),
        "candle_state": feature_view.get("candle_state"),
        "regime_state": feature_view.get("regime_state"),
        "event_state": feature_view.get("event_state"),
        "institutional_vs_hype_spread": feature_view.get("institutional_vs_hype_spread"),
        "trap_risk": feature_view.get("trap_risk"),
        "liquidation_risk": feature_view.get("liquidation_risk"),
        "event_base_uncertainty": feature_view.get("event_base_uncertainty"),
        "allow_trade": feature_view.get("allow_trade"),
    }


def validate_retail_feature_view(feature_view: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    required_core = [
        "sentiment_state",
        "memory_state",
        "candle_state",
        "regime_state",
        "event_state",
        "event_base_uncertainty",
        "allow_trade",
    ]

    for col in required_core:
        if col not in feature_view:
            errors.append(f"missing feature: {col}")

    if feature_view.get("p4_breakout_quality") in {"STRONG", "VERY_STRONG"}:
        warnings.append("retail_breakout_chase_risk")

    if feature_view.get("trap_risk") in {"HIGH", "ELEVATED"}:
        warnings.append("retail_trap_risk_high")

    if feature_view.get("liquidation_risk") in {"HIGH", "ELEVATED"}:
        warnings.append("retail_liquidation_risk_high")

    uncertainty = feature_view.get("event_base_uncertainty")
    if isinstance(uncertainty, (int, float)) and uncertainty >= 0.70:
        warnings.append("high_event_uncertainty")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }