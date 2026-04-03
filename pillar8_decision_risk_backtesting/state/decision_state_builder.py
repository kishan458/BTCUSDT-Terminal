from __future__ import annotations

from typing import Any, Dict, List, Optional

from pillar8_decision_risk_backtesting.state.decision_schema import (
    CandleIntelligenceState,
    CouncilState,
    DecisionState,
    EventState,
    MarketMemoryState,
    RegimeCycleState,
    SentimentState,
    StructureLiquidityState,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _safe_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def _safe_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def build_sentiment_state(payload: Optional[Dict[str, Any]]) -> SentimentState:
    payload = _safe_dict(payload)
    return SentimentState(
        sentiment_state=_safe_str(payload.get("sentiment_state"), "NEUTRAL"),
        confidence=_safe_float(payload.get("confidence"), 0.0),
        drivers=_safe_list(payload.get("drivers")),
        institutional_summary=_safe_str(payload.get("institutional_summary"), ""),
    )


def build_memory_state(payload: Optional[Dict[str, Any]]) -> MarketMemoryState:
    payload = _safe_dict(payload)
    return MarketMemoryState(
        memory_state=_safe_str(payload.get("memory_state"), "NEUTRAL"),
        analog_quality=_safe_float(payload.get("analog_quality"), 0.0),
        forward_bias=_safe_str(payload.get("forward_bias"), "NEUTRAL"),
        stability_score=_safe_float(payload.get("stability_score"), 0.0),
        context_notes=_safe_list(payload.get("context_notes")),
    )


def build_structure_state(payload: Optional[Dict[str, Any]]) -> StructureLiquidityState:
    payload = _safe_dict(payload)
    return StructureLiquidityState(
        structure_state=_safe_str(payload.get("structure_state"), "NEUTRAL"),
        liquidity_levels=_safe_list(payload.get("liquidity_levels")),
        trap_risk=_safe_float(payload.get("trap_risk"), 0.0),
        liquidation_risk=_safe_float(payload.get("liquidation_risk"), 0.0),
        risk_flags=_safe_list(payload.get("risk_flags")),
    )


def build_candle_state(payload: Optional[Dict[str, Any]]) -> CandleIntelligenceState:
    payload = _safe_dict(payload)
    return CandleIntelligenceState(
        dominant_intent=_safe_str(payload.get("dominant_intent"), "NEUTRAL"),
        momentum_state=_safe_str(payload.get("momentum_state"), "NEUTRAL"),
        breakout_quality=_safe_float(payload.get("breakout_quality"), 0.0),
        pressure_bias=_safe_str(payload.get("pressure_bias"), "NEUTRAL"),
        absorption_signals=_safe_list(payload.get("absorption_signals")),
        failure_risk=_safe_float(payload.get("failure_risk"), 0.0),
    )


def build_regime_state(payload: Optional[Dict[str, Any]]) -> RegimeCycleState:
    payload = _safe_dict(payload)
    return RegimeCycleState(
        regime_state=_safe_str(payload.get("regime_state"), "UNKNOWN"),
        cycle_phase=_safe_str(payload.get("cycle_phase"), "UNKNOWN"),
        strategy_compatibility=_safe_float(payload.get("strategy_compatibility"), 0.0),
    )


def build_event_state(payload: Optional[Dict[str, Any]]) -> EventState:
    payload = _safe_dict(payload)
    return EventState(
        event_state=_safe_str(payload.get("event_state"), "IDLE"),
        base_uncertainty=_safe_float(payload.get("base_uncertainty"), 0.0),
        trade_restrictions=_safe_dict(payload.get("trade_restrictions")),
        scenarios=_safe_list(payload.get("scenarios")),
        ai_reasoning=_safe_str(payload.get("ai_reasoning"), ""),
    )


def build_council_state(payload: Optional[Dict[str, Any]]) -> CouncilState:
    payload = _safe_dict(payload)
    return CouncilState(
        final_bias=_safe_str(payload.get("final_bias"), "NEUTRAL"),
        final_decision=_safe_str(payload.get("final_decision"), "NO_TRADE"),
        confidence=_safe_float(payload.get("confidence"), 0.0),
        agreement_score=_safe_float(payload.get("agreement_score"), 0.0),
        conflict_score=_safe_float(payload.get("conflict_score"), 0.0),
        dominant_agent=_safe_str(payload.get("dominant_agent"), "NONE"),
        reasoning=_safe_str(payload.get("reasoning"), ""),
    )


def build_decision_state(
    *,
    timestamp_utc: str = "",
    sentiment_payload: Optional[Dict[str, Any]] = None,
    memory_payload: Optional[Dict[str, Any]] = None,
    structure_payload: Optional[Dict[str, Any]] = None,
    candle_payload: Optional[Dict[str, Any]] = None,
    regime_payload: Optional[Dict[str, Any]] = None,
    events_payload: Optional[Dict[str, Any]] = None,
    council_payload: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DecisionState:
    return DecisionState(
        timestamp_utc=_safe_str(timestamp_utc, ""),
        sentiment=build_sentiment_state(sentiment_payload),
        memory=build_memory_state(memory_payload),
        structure=build_structure_state(structure_payload),
        candle=build_candle_state(candle_payload),
        regime=build_regime_state(regime_payload),
        events=build_event_state(events_payload),
        council=build_council_state(council_payload),
        metadata=_safe_dict(metadata),
    )