from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING

from pillar8_decision_risk_backtesting.decision.conviction_engine import ConvictionResult
from pillar8_decision_risk_backtesting.decision.decision_gate_engine import DecisionGateResult
from pillar8_decision_risk_backtesting.state.decision_schema import DecisionState, FinalAction

if TYPE_CHECKING:
    from pillar8_decision_risk_backtesting.risk.risk_score_engine import RiskScoreResult


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class ExecutionRiskResult:
    event_execution_risk: float
    breakout_execution_risk: float
    tradability_execution_risk: float
    action_execution_risk: float
    aggregate_execution_risk: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "event_execution_risk": self.event_execution_risk,
            "breakout_execution_risk": self.breakout_execution_risk,
            "tradability_execution_risk": self.tradability_execution_risk,
            "action_execution_risk": self.action_execution_risk,
            "aggregate_execution_risk": self.aggregate_execution_risk,
        }


def _compute_event_execution_risk(state: DecisionState) -> float:
    return _clip(state.events.base_uncertainty)


def _compute_breakout_execution_risk(state: DecisionState) -> float:
    breakout_quality = _clip(state.candle.breakout_quality)
    failure_risk = _clip(state.candle.failure_risk)
    return _clip((0.55 * (1.0 - breakout_quality)) + (0.45 * failure_risk))


def _compute_tradability_execution_risk(conviction: ConvictionResult) -> float:
    tradability = _clip(conviction.tradability_score)
    confidence = _clip(conviction.decision_confidence)
    return _clip((0.60 * (1.0 - tradability)) + (0.40 * (1.0 - confidence)))


def _compute_action_execution_risk(
    gate: DecisionGateResult,
    risk: "RiskScoreResult",
) -> float:
    if gate.final_action in (FinalAction.NO_TRADE, FinalAction.WATCHLIST, FinalAction.EXIT):
        return 1.0

    base = _clip(risk.risk_score)

    if gate.final_action in (FinalAction.PROBE_LONG, FinalAction.PROBE_SHORT):
        return _clip(0.70 + (0.30 * base))

    if gate.final_action == FinalAction.REDUCE:
        return _clip(0.80 + (0.20 * base))

    return _clip(0.40 * base)


def compute_execution_risk(
    *,
    state: DecisionState,
    conviction: ConvictionResult,
    gate: DecisionGateResult,
    risk: "RiskScoreResult",
) -> ExecutionRiskResult:
    event_execution_risk = _compute_event_execution_risk(state)
    breakout_execution_risk = _compute_breakout_execution_risk(state)
    tradability_execution_risk = _compute_tradability_execution_risk(conviction)
    action_execution_risk = _compute_action_execution_risk(gate, risk)

    aggregate_execution_risk = _clip(
        (0.25 * event_execution_risk)
        + (0.25 * breakout_execution_risk)
        + (0.25 * tradability_execution_risk)
        + (0.25 * action_execution_risk)
    )

    return ExecutionRiskResult(
        event_execution_risk=round(event_execution_risk, 6),
        breakout_execution_risk=round(breakout_execution_risk, 6),
        tradability_execution_risk=round(tradability_execution_risk, 6),
        action_execution_risk=round(action_execution_risk, 6),
        aggregate_execution_risk=round(aggregate_execution_risk, 6),
    )