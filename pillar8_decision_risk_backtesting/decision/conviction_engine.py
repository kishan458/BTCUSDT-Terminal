from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING

from pillar8_decision_risk_backtesting.decision.alignment_engine import AlignmentResult
from pillar8_decision_risk_backtesting.decision.veto_engine import VetoResult
from pillar8_decision_risk_backtesting.state.decision_schema import DecisionState

if TYPE_CHECKING:
    from pillar8_decision_risk_backtesting.risk.risk_score_engine import RiskScoreResult


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class ConvictionResult:
    decision_confidence: float
    tradability_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "decision_confidence": self.decision_confidence,
            "tradability_score": self.tradability_score,
        }


def compute_decision_confidence(
    state: DecisionState,
    alignment: AlignmentResult,
    risk: "RiskScoreResult",
) -> float:
    edge_strength = _clip(abs(alignment.net_directional_edge))
    alignment_clarity = _clip(1.0 - alignment.directional_conflict)

    council_confidence = _clip(state.council.confidence)
    council_agreement = _clip(state.council.agreement_score)
    memory_quality = _clip(state.memory.analog_quality)
    memory_stability = _clip(state.memory.stability_score)
    breakout_quality = _clip(state.candle.breakout_quality)
    regime_fit = _clip(state.regime.strategy_compatibility)

    event_penalty = _clip(state.events.base_uncertainty)
    risk_penalty = _clip(risk.risk_score)

    positive_stack = (
        (0.22 * edge_strength)
        + (0.14 * alignment_clarity)
        + (0.16 * council_confidence)
        + (0.10 * council_agreement)
        + (0.10 * memory_quality)
        + (0.08 * memory_stability)
        + (0.10 * breakout_quality)
        + (0.10 * regime_fit)
    )

    penalty_stack = (0.45 * risk_penalty) + (0.20 * event_penalty)

    return _clip(positive_stack - penalty_stack)


def compute_tradability_score(
    alignment: AlignmentResult,
    vetoes: VetoResult,
    risk: "RiskScoreResult",
) -> float:
    edge_strength = _clip(abs(alignment.net_directional_edge))
    clarity = _clip(1.0 - alignment.directional_conflict)
    risk_penalty = _clip(risk.risk_score)

    veto_penalty = 1.0 if vetoes.is_trade_blocked else 0.0
    warning_penalty = _clip(len(vetoes.warnings) * 0.08)

    score = (
        (0.45 * edge_strength)
        + (0.25 * clarity)
        + (0.20 * (1.0 - risk_penalty))
        - (0.60 * veto_penalty)
        - warning_penalty
    )
    return _clip(score)


def compute_conviction(
    state: DecisionState,
    alignment: AlignmentResult,
    vetoes: VetoResult,
    risk: "RiskScoreResult",
) -> ConvictionResult:
    decision_confidence = compute_decision_confidence(state, alignment, risk)
    tradability_score = compute_tradability_score(alignment, vetoes, risk)

    return ConvictionResult(
        decision_confidence=round(decision_confidence, 6),
        tradability_score=round(tradability_score, 6),
    )