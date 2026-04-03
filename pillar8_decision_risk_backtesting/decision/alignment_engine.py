from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from pillar8_decision_risk_backtesting.state.decision_schema import DecisionState


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalize_signed_score(score: float) -> Tuple[float, float]:
    """
    Converts a signed score in [-1, 1] into (long_component, short_component)
    where each is in [0, 1].
    """
    score = max(-1.0, min(1.0, score))
    if score >= 0:
        return score, 0.0
    return 0.0, abs(score)


def _text_to_bias_score(text: str) -> float:
    """
    Transparent rule mapping:
    bullish/uptrend/long-like states => positive
    bearish/downtrend/short-like states => negative
    neutral/unknown => 0
    """
    value = (text or "").strip().upper()

    positive_keywords = {
        "BULLISH",
        "LONG",
        "UP",
        "UPTREND",
        "TREND_FOLLOW",
        "ACCUMULATION",
        "EXPANSION_UP",
        "HIGHER_HIGH_HIGHER_LOW",
        "HIGHER_HIGH",
        "HIGHER_LOW",
        "BUY",
    }

    negative_keywords = {
        "BEARISH",
        "SHORT",
        "DOWN",
        "DOWNTREND",
        "MEAN_REVERSION",
        "DISTRIBUTION",
        "EXPANSION_DOWN",
        "LOWER_LOW_LOWER_HIGH",
        "LOWER_LOW",
        "LOWER_HIGH",
        "SELL",
    }

    neutral_keywords = {
        "NEUTRAL",
        "NONE",
        "UNKNOWN",
        "NO_TRADE",
        "IDLE",
        "",
    }

    if value in positive_keywords:
        return 1.0
    if value in negative_keywords:
        return -1.0
    if value in neutral_keywords:
        return 0.0

    if "BULL" in value or "LONG" in value or "UP" in value:
        return 1.0
    if "BEAR" in value or "SHORT" in value or "DOWN" in value:
        return -1.0

    return 0.0


@dataclass
class AlignmentComponent:
    pillar: str
    raw_bias_score: float
    confidence_weight: float
    weighted_score: float
    explanation: str


@dataclass
class AlignmentResult:
    long_score: float
    short_score: float
    net_directional_edge: float
    directional_conflict: float
    dominant_direction: str
    components: List[AlignmentComponent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "long_score": self.long_score,
            "short_score": self.short_score,
            "net_directional_edge": self.net_directional_edge,
            "directional_conflict": self.directional_conflict,
            "dominant_direction": self.dominant_direction,
            "components": [
                {
                    "pillar": c.pillar,
                    "raw_bias_score": c.raw_bias_score,
                    "confidence_weight": c.confidence_weight,
                    "weighted_score": c.weighted_score,
                    "explanation": c.explanation,
                }
                for c in self.components
            ],
        }


def _build_sentiment_component(state: DecisionState) -> AlignmentComponent:
    raw = _text_to_bias_score(state.sentiment.sentiment_state)
    weight = _clip(state.sentiment.confidence)
    return AlignmentComponent(
        pillar="sentiment",
        raw_bias_score=raw,
        confidence_weight=weight,
        weighted_score=raw * weight,
        explanation=f"sentiment_state={state.sentiment.sentiment_state}, confidence={state.sentiment.confidence}",
    )


def _build_memory_component(state: DecisionState) -> AlignmentComponent:
    raw = _text_to_bias_score(state.memory.forward_bias)
    quality = _clip(state.memory.analog_quality)
    stability = _clip(state.memory.stability_score)
    weight = (quality + stability) / 2.0
    return AlignmentComponent(
        pillar="memory",
        raw_bias_score=raw,
        confidence_weight=weight,
        weighted_score=raw * weight,
        explanation=(
            f"forward_bias={state.memory.forward_bias}, "
            f"analog_quality={state.memory.analog_quality}, "
            f"stability_score={state.memory.stability_score}"
        ),
    )


def _build_structure_component(state: DecisionState) -> AlignmentComponent:
    raw = _text_to_bias_score(state.structure.structure_state)

    trap_penalty = _clip(state.structure.trap_risk)
    liquidation_penalty = _clip(state.structure.liquidation_risk)

    weight = _clip(1.0 - ((trap_penalty + liquidation_penalty) / 2.0))
    return AlignmentComponent(
        pillar="structure",
        raw_bias_score=raw,
        confidence_weight=weight,
        weighted_score=raw * weight,
        explanation=(
            f"structure_state={state.structure.structure_state}, "
            f"trap_risk={state.structure.trap_risk}, "
            f"liquidation_risk={state.structure.liquidation_risk}"
        ),
    )


def _build_candle_component(state: DecisionState) -> AlignmentComponent:
    intent_score = _text_to_bias_score(state.candle.dominant_intent)
    pressure_score = _text_to_bias_score(state.candle.pressure_bias)

    raw = (intent_score + pressure_score) / 2.0

    breakout_quality = _clip(state.candle.breakout_quality)
    failure_penalty = _clip(state.candle.failure_risk)

    weight = _clip((0.6 * breakout_quality) + (0.4 * (1.0 - failure_penalty)))
    return AlignmentComponent(
        pillar="candle",
        raw_bias_score=raw,
        confidence_weight=weight,
        weighted_score=raw * weight,
        explanation=(
            f"dominant_intent={state.candle.dominant_intent}, "
            f"pressure_bias={state.candle.pressure_bias}, "
            f"breakout_quality={state.candle.breakout_quality}, "
            f"failure_risk={state.candle.failure_risk}"
        ),
    )


def _build_regime_component(state: DecisionState) -> AlignmentComponent:
    raw = _text_to_bias_score(state.regime.regime_state)
    weight = _clip(state.regime.strategy_compatibility)
    return AlignmentComponent(
        pillar="regime",
        raw_bias_score=raw,
        confidence_weight=weight,
        weighted_score=raw * weight,
        explanation=(
            f"regime_state={state.regime.regime_state}, "
            f"cycle_phase={state.regime.cycle_phase}, "
            f"strategy_compatibility={state.regime.strategy_compatibility}"
        ),
    )


def _build_events_component(state: DecisionState) -> AlignmentComponent:
    """
    Events do not directly create bias here.
    They reduce confidence in directional alignment when uncertainty is high.
    """
    raw = 0.0
    uncertainty = _clip(state.events.base_uncertainty)
    weight = _clip(1.0 - uncertainty)
    return AlignmentComponent(
        pillar="events",
        raw_bias_score=raw,
        confidence_weight=weight,
        weighted_score=0.0,
        explanation=(
            f"event_state={state.events.event_state}, "
            f"base_uncertainty={state.events.base_uncertainty}"
        ),
    )


def _build_council_component(state: DecisionState) -> AlignmentComponent:
    raw = _text_to_bias_score(state.council.final_bias)

    confidence = _clip(state.council.confidence)
    agreement = _clip(state.council.agreement_score)
    conflict = _clip(state.council.conflict_score)

    weight = _clip((0.5 * confidence) + (0.3 * agreement) + (0.2 * (1.0 - conflict)))
    return AlignmentComponent(
        pillar="council",
        raw_bias_score=raw,
        confidence_weight=weight,
        weighted_score=raw * weight,
        explanation=(
            f"final_bias={state.council.final_bias}, "
            f"confidence={state.council.confidence}, "
            f"agreement_score={state.council.agreement_score}, "
            f"conflict_score={state.council.conflict_score}"
        ),
    )


def compute_alignment(state: DecisionState) -> AlignmentResult:
    components = [
        _build_sentiment_component(state),
        _build_memory_component(state),
        _build_structure_component(state),
        _build_candle_component(state),
        _build_regime_component(state),
        _build_events_component(state),
        _build_council_component(state),
    ]

    long_total = 0.0
    short_total = 0.0
    active_weight_total = 0.0

    for component in components:
        long_part, short_part = _normalize_signed_score(component.weighted_score)
        long_total += long_part
        short_total += short_part
        active_weight_total += abs(component.confidence_weight)

    if active_weight_total == 0:
        long_score = 0.0
        short_score = 0.0
    else:
        long_score = _clip(long_total / active_weight_total)
        short_score = _clip(short_total / active_weight_total)

    net_directional_edge = max(-1.0, min(1.0, long_score - short_score))
    directional_conflict = _clip(min(long_score, short_score) * 2.0)

    if net_directional_edge > 0.10:
        dominant_direction = "LONG"
    elif net_directional_edge < -0.10:
        dominant_direction = "SHORT"
    else:
        dominant_direction = "NONE"

    return AlignmentResult(
        long_score=round(long_score, 6),
        short_score=round(short_score, 6),
        net_directional_edge=round(net_directional_edge, 6),
        directional_conflict=round(directional_conflict, 6),
        dominant_direction=dominant_direction,
        components=components,
    )