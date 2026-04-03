from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING

from pillar8_decision_risk_backtesting.decision.alignment_engine import AlignmentResult
from pillar8_decision_risk_backtesting.decision.conviction_engine import compute_conviction
from pillar8_decision_risk_backtesting.decision.veto_engine import VetoResult
from pillar8_decision_risk_backtesting.state.decision_schema import (
    DecisionState,
    Direction,
    FinalAction,
)

if TYPE_CHECKING:
    from pillar8_decision_risk_backtesting.risk.risk_score_engine import RiskScoreResult


@dataclass
class DecisionGateResult:
    final_action: FinalAction
    direction: Direction
    tradability_score: float
    decision_confidence: float
    decision_archetype: str
    rationale: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "final_action": self.final_action.value,
            "direction": self.direction.value,
            "tradability_score": self.tradability_score,
            "decision_confidence": self.decision_confidence,
            "decision_archetype": self.decision_archetype,
            "rationale": self.rationale,
        }


def _resolve_direction(alignment: AlignmentResult) -> Direction:
    if alignment.dominant_direction == "LONG":
        return Direction.LONG
    if alignment.dominant_direction == "SHORT":
        return Direction.SHORT
    return Direction.NONE


def _infer_decision_archetype(
    state: DecisionState,
    direction: Direction,
    tradability_score: float,
    risk: "RiskScoreResult",
) -> str:
    if direction == Direction.NONE:
        if state.events.base_uncertainty >= 0.65:
            return "EVENT_AVOIDANCE"
        return "NO_EDGE"

    if tradability_score < 0.25:
        return "LOW_QUALITY_SETUP"

    if risk.risk_score >= 0.65:
        return "RISK_SUPPRESSED"

    if state.candle.breakout_quality >= 0.70:
        if direction == Direction.LONG:
            return "BREAKOUT_CONTINUATION_LONG"
        return "BREAKOUT_CONTINUATION_SHORT"

    if state.structure.trap_risk >= 0.60:
        if direction == Direction.LONG:
            return "POST_SWEEP_REVERSAL_LONG"
        return "POST_SWEEP_REVERSAL_SHORT"

    if state.memory.memory_state.upper() == "MEAN_REVERSION":
        if direction == Direction.LONG:
            return "MEAN_REVERSION_LONG"
        return "MEAN_REVERSION_SHORT"

    if direction == Direction.LONG:
        return "TREND_LONG"
    return "TREND_SHORT"


def run_decision_gate(
    state: DecisionState,
    alignment: AlignmentResult,
    vetoes: VetoResult,
    risk: "RiskScoreResult",
) -> DecisionGateResult:
    direction = _resolve_direction(alignment)

    conviction = compute_conviction(
        state=state,
        alignment=alignment,
        vetoes=vetoes,
        risk=risk,
    )

    decision_confidence = conviction.decision_confidence
    tradability_score = conviction.tradability_score

    decision_archetype = _infer_decision_archetype(
        state=state,
        direction=direction,
        tradability_score=tradability_score,
        risk=risk,
    )

    if vetoes.is_trade_blocked:
        final_action = FinalAction.NO_TRADE
        rationale = "Trade blocked by hard veto conditions."
    elif direction == Direction.NONE:
        final_action = FinalAction.WATCHLIST
        rationale = "No strong directional edge yet."
    elif risk.risk_score >= 0.85:
        final_action = FinalAction.NO_TRADE
        rationale = "Risk is extreme."
    elif risk.risk_score >= 0.65:
        final_action = (
            FinalAction.PROBE_LONG if direction == Direction.LONG else FinalAction.PROBE_SHORT
        )
        rationale = "Directional edge exists but risk is high, so only probe exposure is allowed."
    elif tradability_score >= 0.60 and decision_confidence >= 0.55:
        final_action = FinalAction.LONG if direction == Direction.LONG else FinalAction.SHORT
        rationale = "Tradability and confidence are strong enough for a full trade."
    elif tradability_score >= 0.35:
        final_action = (
            FinalAction.PROBE_LONG if direction == Direction.LONG else FinalAction.PROBE_SHORT
        )
        rationale = "Setup is viable but not strong enough for full size."
    else:
        final_action = FinalAction.WATCHLIST
        rationale = "Setup is not clean enough yet."

    return DecisionGateResult(
        final_action=final_action,
        direction=direction,
        tradability_score=round(tradability_score, 6),
        decision_confidence=round(decision_confidence, 6),
        decision_archetype=decision_archetype,
        rationale=rationale,
    )