from __future__ import annotations

from typing import List

from pillar8_decision_risk_backtesting.decision.alignment_engine import AlignmentResult
from pillar8_decision_risk_backtesting.decision.decision_gate_engine import DecisionGateResult
from pillar8_decision_risk_backtesting.risk.risk_score_engine import RiskScoreResult
from pillar8_decision_risk_backtesting.state.decision_schema import DecisionState, Direction


def build_invalidators(
    *,
    state: DecisionState,
    alignment: AlignmentResult,
    risk: RiskScoreResult,
    gate: DecisionGateResult,
) -> List[str]:
    invalidators: List[str] = []

    if gate.direction == Direction.NONE:
        invalidators.append("Directional edge remains absent across pillars.")
        return invalidators

    if gate.direction == Direction.LONG:
        invalidators.append("Long thesis invalid if cross-pillar directional edge flips away from LONG.")
        if state.candle.breakout_quality < 0.50:
            invalidators.append("Long thesis invalid if breakout acceptance fails after entry.")
        if state.candle.failure_risk >= 0.60:
            invalidators.append("Long thesis invalid if candle failure conditions continue to rise.")
        if state.structure.trap_risk >= 0.60:
            invalidators.append("Long thesis invalid if upside move is revealed as a trap or post-sweep failure.")
        if state.structure.liquidation_risk >= 0.60:
            invalidators.append("Long thesis invalid if long-side liquidation pressure starts increasing sharply.")
    else:
        invalidators.append("Short thesis invalid if cross-pillar directional edge flips away from SHORT.")
        if state.candle.breakout_quality < 0.50:
            invalidators.append("Short thesis invalid if downside acceptance fails after entry.")
        if state.candle.failure_risk >= 0.60:
            invalidators.append("Short thesis invalid if candle failure conditions continue to rise.")
        if state.structure.trap_risk >= 0.60:
            invalidators.append("Short thesis invalid if downside move is revealed as a trap or failed breakdown.")
        if state.structure.liquidation_risk >= 0.60:
            invalidators.append("Short thesis invalid if short-side squeeze risk starts increasing sharply.")

    if state.events.base_uncertainty >= 0.65:
        invalidators.append("Thesis invalid if event uncertainty remains elevated or expands further.")

    if state.regime.strategy_compatibility <= 0.40:
        invalidators.append("Thesis invalid if regime remains incompatible with the current setup type.")

    if alignment.directional_conflict >= 0.50:
        invalidators.append("Thesis invalid if internal pillar conflict remains elevated.")

    if risk.risk_score >= 0.65:
        invalidators.append("Thesis invalid if aggregate risk score remains in the high-risk zone.")

    return invalidators