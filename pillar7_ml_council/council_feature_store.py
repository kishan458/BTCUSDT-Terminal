from __future__ import annotations

from typing import Any, Dict, List, Optional


STRING_NONE_DEFAULT = "UNKNOWN"


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any, default: str = STRING_NONE_DEFAULT) -> str:
    if value is None:
        return default
    return str(value)


def _bool_to_int(value: Any) -> Optional[int]:
    if value is True:
        return 1
    if value is False:
        return 0
    return None


def _get_nested(d: Dict[str, Any], *keys: str) -> Any:
    current: Any = d
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def flatten_shared_state(shared_state: Dict[str, Any]) -> Dict[str, Any]:
    sentiment = shared_state.get("sentiment", {})
    market_memory = shared_state.get("market_memory", {})
    structure_liquidity = shared_state.get("structure_liquidity", {})
    candle = shared_state.get("candle", {})
    regime_cycle = shared_state.get("regime_cycle", {})
    high_impact_event = shared_state.get("high_impact_event", {})
    trade_restrictions = shared_state.get("trade_restrictions", {})
    meta = shared_state.get("meta", {})

    p1_raw = sentiment.get("raw", {}) if isinstance(sentiment, dict) else {}
    p2_raw = market_memory.get("raw", {}) if isinstance(market_memory, dict) else {}
    p3_raw = structure_liquidity.get("raw", {}) if isinstance(structure_liquidity, dict) else {}
    p4_raw = candle.get("raw", {}) if isinstance(candle, dict) else {}
    p5_raw = regime_cycle.get("raw", {}) if isinstance(regime_cycle, dict) else {}
    p6_raw = high_impact_event.get("raw", {}) if isinstance(high_impact_event, dict) else {}

    return {
        "asset": _safe_str(shared_state.get("asset")),
        "timestamp_utc": shared_state.get("timestamp_utc"),

        "sentiment_state": _safe_str(sentiment.get("state")),
        "sentiment_confidence": _safe_float(sentiment.get("confidence")),

        "memory_state": _safe_str(market_memory.get("state")),
        "memory_confidence": _safe_float(market_memory.get("confidence")),

        "structure_state": _safe_str(structure_liquidity.get("state")),
        "structure_confidence": _safe_float(structure_liquidity.get("confidence")),

        "candle_state": _safe_str(candle.get("state")),
        "candle_confidence": _safe_float(candle.get("confidence")),

        "regime_state": _safe_str(regime_cycle.get("state")),
        "regime_confidence": _safe_float(regime_cycle.get("confidence")),

        "event_state": _safe_str(high_impact_event.get("state")),
        "event_confidence": _safe_float(high_impact_event.get("confidence")),

        "institutional_vs_hype_spread": _safe_float(meta.get("institutional_vs_hype_spread")),
        "analog_quality": _safe_float(meta.get("analog_quality")),
        "trap_risk": _safe_str(meta.get("trap_risk")),
        "liquidation_risk": _safe_str(meta.get("liquidation_risk")),
        "candle_intent": _safe_str(meta.get("candle_intent")),
        "cycle_phase": _safe_str(meta.get("cycle_phase")),
        "event_name": _safe_str(meta.get("event_name")),
        "event_base_uncertainty": _safe_float(meta.get("event_base_uncertainty")),

        "allow_trade": _bool_to_int(trade_restrictions.get("allow_trade")),
        "size_multiplier": _safe_float(trade_restrictions.get("size_multiplier")),
        "leverage_cap": _safe_float(trade_restrictions.get("leverage_cap")),
        "restriction_reason": _safe_str(trade_restrictions.get("restriction_reason")),
        "risk_flag_count": len(shared_state.get("risk_flags", [])),
        "risk_flags_joined": "|".join(shared_state.get("risk_flags", [])) if shared_state.get("risk_flags") else "",

        "p1_sentiment": _safe_str(p1_raw.get("sentiment")),
        "p1_final_sentiment": _safe_str(p1_raw.get("final_sentiment")),
        "p1_institutional_sentiment": _safe_str(p1_raw.get("institutional_sentiment")),
        "p1_sentiment_confidence": _safe_float(
            p1_raw.get("sentiment_confidence") if p1_raw.get("sentiment_confidence") is not None else p1_raw.get("confidence")
        ),
        "p1_narrative_state": _safe_str(p1_raw.get("narrative_state")),
        "p1_narrative_maturity": _safe_str(p1_raw.get("narrative_maturity")),
        "p1_sentiment_divergence": _safe_float(p1_raw.get("sentiment_divergence")),

        "p2_memory_bias": _safe_str(p2_raw.get("memory_bias")),
        "p2_memory_state": _safe_str(p2_raw.get("memory_state")),
        "p2_match_quality": _safe_float(
            p2_raw.get("historical_match_quality") if p2_raw.get("historical_match_quality") is not None else p2_raw.get("match_quality")
        ),
        "p2_analog_quality": _safe_float(p2_raw.get("analog_quality")),
        "p2_stability_score": _safe_float(p2_raw.get("stability_score")),
        "p2_stability_confidence": _safe_float(p2_raw.get("stability_confidence")),
        "p2_forward_bias": _safe_str(p2_raw.get("forward_bias")),

        "p3_market_structure": _safe_str(p3_raw.get("market_structure")),
        "p3_structure_state": _safe_str(p3_raw.get("structure_state")),
        "p3_range_state": _safe_str(p3_raw.get("range_state")),
        "p3_compression_state": _safe_str(p3_raw.get("compression_state")),
        "p3_nearest_liquidity_magnet": _safe_str(p3_raw.get("nearest_liquidity_magnet")),
        "p3_buy_side_liquidity": _safe_float(p3_raw.get("buy_side_liquidity")),
        "p3_sell_side_liquidity": _safe_float(p3_raw.get("sell_side_liquidity")),
        "p3_trap_risk": _safe_str(p3_raw.get("trap_risk")),
        "p3_liquidation_risk": _safe_str(p3_raw.get("liquidation_risk")),
        "p3_stop_hunt_risk": _safe_str(p3_raw.get("stop_hunt_risk")),
        "p3_risk_flag": _safe_str(p3_raw.get("risk_flag")),

        "p4_candle_intent": _safe_str(p4_raw.get("candle_intent")),
        "p4_pressure_bias": _safe_str(p4_raw.get("pressure_bias")),
        "p4_breakout_quality": _safe_str(p4_raw.get("breakout_quality")),
        "p4_follow_through_quality": _safe_str(p4_raw.get("follow_through_quality")),
        "p4_absorption_state": _safe_str(p4_raw.get("absorption_state")),
        "p4_overlap_state": _safe_str(p4_raw.get("overlap_state")),
        "p4_rejection_state": _safe_str(p4_raw.get("rejection_state")),

        "p5_market_regime": _safe_str(p5_raw.get("market_regime")),
        "p5_regime_state": _safe_str(p5_raw.get("regime_state")),
        "p5_volatility_regime": _safe_str(p5_raw.get("volatility_regime")),
        "p5_trend_regime": _safe_str(p5_raw.get("trend_regime")),
        "p5_cycle_phase": _safe_str(p5_raw.get("cycle_phase")),
        "p5_market_state": _safe_str(p5_raw.get("market_state")),
        "p5_strategy_compatibility": _safe_str(p5_raw.get("strategy_compatibility")),
        "p5_risk_flag": _safe_str(p5_raw.get("risk_flag")),

        "p6_event_name": _safe_str(
            p6_raw.get("event") if p6_raw.get("event") is not None else p6_raw.get("event_name")
        ),
        "p6_event_state": _safe_str(
            p6_raw.get("state") if p6_raw.get("state") is not None else p6_raw.get("event_state")
        ),
        "p6_base_uncertainty": _safe_float(p6_raw.get("base_uncertainty")),
        "p6_confidence_score": _safe_float(
            p6_raw.get("confidence_score") if p6_raw.get("confidence_score") is not None else p6_raw.get("confidence")
        ),
        "p6_trade_allowed": _bool_to_int(_get_nested(p6_raw, "trade_restrictions", "allow_trade")),
        "p6_size_multiplier": _safe_float(_get_nested(p6_raw, "trade_restrictions", "size_multiplier")),
        "p6_leverage_cap": _safe_float(_get_nested(p6_raw, "trade_restrictions", "leverage_cap")),
        "p6_restriction_reason": _safe_str(_get_nested(p6_raw, "trade_restrictions", "restriction_reason")),
    }


def build_feature_row(shared_state: Dict[str, Any]) -> Dict[str, Any]:
    return flatten_shared_state(shared_state)


def build_feature_rows(shared_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [build_feature_row(state) for state in shared_states]


def summarize_feature_row(feature_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "asset": feature_row.get("asset"),
        "timestamp_utc": feature_row.get("timestamp_utc"),
        "sentiment_state": feature_row.get("sentiment_state"),
        "memory_state": feature_row.get("memory_state"),
        "structure_state": feature_row.get("structure_state"),
        "candle_state": feature_row.get("candle_state"),
        "regime_state": feature_row.get("regime_state"),
        "event_state": feature_row.get("event_state"),
        "institutional_vs_hype_spread": feature_row.get("institutional_vs_hype_spread"),
        "analog_quality": feature_row.get("analog_quality"),
        "event_base_uncertainty": feature_row.get("event_base_uncertainty"),
        "allow_trade": feature_row.get("allow_trade"),
        "risk_flag_count": feature_row.get("risk_flag_count"),
    }


def validate_feature_row(feature_row: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    required_keys = [
        "asset",
        "timestamp_utc",
        "sentiment_state",
        "memory_state",
        "structure_state",
        "candle_state",
        "regime_state",
        "event_state",
    ]

    for key in required_keys:
        if key not in feature_row:
            errors.append(f"missing required key: {key}")

    if feature_row.get("allow_trade") == 0:
        warnings.append("trade restricted in feature row")

    event_uncertainty = _safe_float(feature_row.get("event_base_uncertainty"))
    if event_uncertainty is not None and event_uncertainty >= 0.70:
        warnings.append("high event uncertainty")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }