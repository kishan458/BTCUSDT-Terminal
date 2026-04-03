from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pillar8_decision_risk_backtesting.decision.alignment_engine import AlignmentResult
from pillar8_decision_risk_backtesting.decision.conviction_engine import ConvictionResult
from pillar8_decision_risk_backtesting.decision.decision_gate_engine import DecisionGateResult
from pillar8_decision_risk_backtesting.decision.veto_engine import VetoResult
from pillar8_decision_risk_backtesting.risk.execution_risk_engine import compute_execution_risk
from pillar8_decision_risk_backtesting.risk.market_risk_engine import compute_market_risk
from pillar8_decision_risk_backtesting.risk.model_risk_engine import compute_model_risk
from pillar8_decision_risk_backtesting.state.decision_schema import DecisionState, RiskState


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class RiskComponentBreakdown:
    market_risk: float
    execution_risk: float
    model_risk: float
    veto_risk: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "market_risk": self.market_risk,
            "execution_risk": self.execution_risk,
            "model_risk": self.model_risk,
            "veto_risk": self.veto_risk,
        }


@dataclass
class RiskScoreResult:
    risk_score: float
    risk_state: RiskState
    components: RiskComponentBreakdown

    def to_dict(self) -> Dict[str, object]:
        return {
            "risk_score": self.risk_score,
            "risk_state": self.risk_state.value,
            "components": self.components.to_dict(),
        }


def _compute_veto_risk(vetoes: VetoResult, state: DecisionState) -> float:
    structure_flag_count = len(state.structure.risk_flags)
    veto_count = len(vetoes.vetoes)
    warning_count = len(vetoes.warnings)

    raw_veto_risk = (
        (0.35 * veto_count)
        + (0.12 * warning_count)
        + (0.10 * structure_flag_count)
    )
    return _clip(raw_veto_risk)


def _map_risk_state(risk_score: float) -> RiskState:
    if risk_score >= 0.85:
        return RiskState.EXTREME
    if risk_score >= 0.65:
        return RiskState.HIGH
    if risk_score >= 0.35:
        return RiskState.MODERATE
    return RiskState.LOW


def compute_risk_score(
    *,
    state: DecisionState,
    alignment: AlignmentResult,
    conviction: ConvictionResult,
    gate: DecisionGateResult,
    vetoes: VetoResult,
) -> RiskScoreResult:
    market_risk_result = compute_market_risk(state)
    execution_risk_result = compute_execution_risk(
        state=state,
        conviction=conviction,
        gate=gate,
        risk=_bootstrap_risk_for_execution_gate(gate),
    )
    model_risk_result = compute_model_risk(
        state=state,
        alignment=alignment,
    )
    veto_risk = _compute_veto_risk(vetoes, state)

    risk_score = _clip(
        (0.34 * market_risk_result.market_risk_score)
        + (0.24 * execution_risk_result.aggregate_execution_risk)
        + (0.24 * model_risk_result.aggregate_model_risk)
        + (0.18 * veto_risk)
    )

    components = RiskComponentBreakdown(
        market_risk=round(market_risk_result.market_risk_score, 6),
        execution_risk=round(execution_risk_result.aggregate_execution_risk, 6),
        model_risk=round(model_risk_result.aggregate_model_risk, 6),
        veto_risk=round(veto_risk, 6),
    )

    return RiskScoreResult(
        risk_score=round(risk_score, 6),
        risk_state=_map_risk_state(risk_score),
        components=components,
    )


def _bootstrap_risk_for_execution_gate(gate: DecisionGateResult) -> RiskScoreResult:
    """
    Execution risk needs a risk object for action-aware scaling.
    During master risk computation, we don't yet have the final aggregate risk score,
    so we bootstrap a lightweight proxy from action type alone.

    This is not fake market data:
    it is an internal structural proxy used only to let the execution-risk module
    score action-mode risk before the full aggregate risk is finalized.
    """
    action_proxy = 1.0
    if gate.final_action.name in {"LONG", "SHORT"}:
        action_proxy = 0.25
    elif gate.final_action.name in {"PROBE_LONG", "PROBE_SHORT"}:
        action_proxy = 0.60
    elif gate.final_action.name == "REDUCE":
        action_proxy = 0.75
    elif gate.final_action.name in {"WATCHLIST", "NO_TRADE", "EXIT"}:
        action_proxy = 1.0

    return RiskScoreResult(
        risk_score=action_proxy,
        risk_state=_map_risk_state(action_proxy),
        components=RiskComponentBreakdown(
            market_risk=0.0,
            execution_risk=0.0,
            model_risk=0.0,
            veto_risk=0.0,
        ),
    )