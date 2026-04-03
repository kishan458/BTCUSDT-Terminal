from __future__ import annotations

from typing import Any, Dict, Optional

from pillar8_decision_risk_backtesting.state.decision_state_builder import build_decision_state
from pillar8_decision_risk_backtesting.decision.alignment_engine import compute_alignment
from pillar8_decision_risk_backtesting.decision.veto_engine import evaluate_vetoes
from pillar8_decision_risk_backtesting.decision.conviction_engine import compute_conviction
from pillar8_decision_risk_backtesting.decision.decision_gate_engine import run_decision_gate

from pillar8_decision_risk_backtesting.risk.risk_score_engine import (
    compute_risk_score,
    RiskComponentBreakdown,
    RiskScoreResult,
)
from pillar8_decision_risk_backtesting.sizing.volatility_target_engine import compute_volatility_target_size
from pillar8_decision_risk_backtesting.sizing.fractional_kelly_engine import compute_fractional_kelly_size
from pillar8_decision_risk_backtesting.sizing.size_allocator import allocate_position_size
from pillar8_decision_risk_backtesting.sizing.leverage_cap_engine import compute_leverage_cap

from pillar8_decision_risk_backtesting.execution.invalidation_engine import build_invalidators
from pillar8_decision_risk_backtesting.execution.holding_horizon_engine import infer_holding_horizon
from pillar8_decision_risk_backtesting.execution.trade_constructor import construct_trade_plan

from pillar8_decision_risk_backtesting.output.pillar8_output import build_pillar8_output

from pillar8_decision_risk_backtesting.backtesting.validation_report import ValidationReportResult
from pillar8_decision_risk_backtesting.stress.drawdown_stress_engine import DrawdownStressResult
from pillar8_decision_risk_backtesting.stress.cost_shock_engine import CostShockResult
from pillar8_decision_risk_backtesting.stress.regime_stress_engine import RegimeStressResult
from pillar8_decision_risk_backtesting.stress.ruin_probability_engine import RuinProbabilityResult

from pillar8_decision_risk_backtesting.state.decision_schema import RiskState


def _build_bootstrap_risk() -> RiskScoreResult:
    return RiskScoreResult(
        risk_score=0.25,
        risk_state=RiskState.LOW,
        components=RiskComponentBreakdown(
            market_risk=0.0,
            execution_risk=0.0,
            model_risk=0.0,
            veto_risk=0.0,
        ),
    )


def _default_validation_report() -> ValidationReportResult:
    return ValidationReportResult(
        overall_quality="UNKNOWN",
        strengths=[],
        weaknesses=[],
        warnings=["Validation report not supplied."],
        regime_notes=[],
    )


def _default_drawdown_stress() -> DrawdownStressResult:
    return DrawdownStressResult(
        max_drawdown=0.0,
        mean_drawdown=0.0,
        drawdown_breach_threshold=0.0,
        breach_count=0,
        breach_rate=0.0,
    )


def _default_cost_shock() -> CostShockResult:
    return CostShockResult(
        original_total_pnl=0.0,
        shocked_total_pnl=0.0,
        original_average_pnl=0.0,
        shocked_average_pnl=0.0,
        pnl_degradation=0.0,
        degradation_ratio=0.0,
        edge_survives=False,
    )


def _default_regime_stress() -> RegimeStressResult:
    return RegimeStressResult(
        best_regime="UNKNOWN",
        worst_regime="UNKNOWN",
        best_regime_pnl=0.0,
        worst_regime_pnl=0.0,
        regime_spread=0.0,
        regime_fragility_score=0.0,
        regime_fragility_label="UNKNOWN",
    )


def _default_ruin_result() -> RuinProbabilityResult:
    return RuinProbabilityResult(
        simulations=0,
        ruin_count=0,
        ruin_probability=0.0,
        avg_terminal_equity=0.0,
        p05_terminal_equity=0.0,
    )


def run_pillar8_engine(
    *,
    timestamp_utc: str,
    sentiment_payload: Optional[Dict[str, Any]] = None,
    memory_payload: Optional[Dict[str, Any]] = None,
    structure_payload: Optional[Dict[str, Any]] = None,
    candle_payload: Optional[Dict[str, Any]] = None,
    regime_payload: Optional[Dict[str, Any]] = None,
    events_payload: Optional[Dict[str, Any]] = None,
    council_payload: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    realized_volatility: Optional[float] = None,
    target_volatility: float = 0.2,
    win_probability: float = 0.5,
    payoff_ratio: float = 1.0,
    kelly_fraction: float = 0.25,
    max_size_fraction: float = 1.0,
    max_leverage: float = 3.0,
    min_leverage: float = 1.0,
    thesis_summary: str = "",
    validation_report: Optional[ValidationReportResult] = None,
    drawdown_stress: Optional[DrawdownStressResult] = None,
    cost_shock: Optional[CostShockResult] = None,
    regime_stress: Optional[RegimeStressResult] = None,
    ruin: Optional[RuinProbabilityResult] = None,
):
    state = build_decision_state(
        timestamp_utc=timestamp_utc,
        sentiment_payload=sentiment_payload,
        memory_payload=memory_payload,
        structure_payload=structure_payload,
        candle_payload=candle_payload,
        regime_payload=regime_payload,
        events_payload=events_payload,
        council_payload=council_payload,
        metadata=metadata,
    )

    alignment = compute_alignment(state)
    vetoes = evaluate_vetoes(state, alignment)

    bootstrap_risk = _build_bootstrap_risk()

    conviction = compute_conviction(state, alignment, vetoes, bootstrap_risk)
    gate = run_decision_gate(state, alignment, vetoes, bootstrap_risk)
    risk = compute_risk_score(
        state=state,
        alignment=alignment,
        conviction=conviction,
        gate=gate,
        vetoes=vetoes,
    )

    volatility_target = compute_volatility_target_size(
        realized_volatility=realized_volatility,
        target_volatility=target_volatility,
        max_size_fraction=max_size_fraction,
    )

    fractional_kelly = compute_fractional_kelly_size(
        win_probability=win_probability,
        payoff_ratio=payoff_ratio,
        kelly_fraction=kelly_fraction,
        max_size_fraction=max_size_fraction,
    )

    size = allocate_position_size(
        volatility_target=volatility_target,
        fractional_kelly=fractional_kelly,
        conviction=conviction,
        risk=risk,
        gate=gate,
        max_size_fraction=max_size_fraction,
    )

    leverage = compute_leverage_cap(
        state=state,
        risk_score=risk.risk_score,
        max_leverage=max_leverage,
        min_leverage=min_leverage,
    )

    invalidators = build_invalidators(
        state=state,
        alignment=alignment,
        risk=risk,
        gate=gate,
    )

    horizon = infer_holding_horizon(
        state=state,
        risk=risk,
        gate=gate,
    )

    trade_plan = construct_trade_plan(
        gate=gate,
        risk=risk,
        size=size,
        leverage=leverage,
        invalidators=invalidators,
        horizon=horizon,
    )

    output = build_pillar8_output(
        timestamp_utc=timestamp_utc,
        gate=gate,
        risk=risk,
        alignment=alignment,
        trade_plan=trade_plan,
        size=size,
        leverage=leverage,
        vetoes=vetoes.vetoes,
        warnings=vetoes.warnings,
        thesis_summary=thesis_summary,
        validation_report=validation_report or _default_validation_report(),
        drawdown_stress=drawdown_stress or _default_drawdown_stress(),
        cost_shock=cost_shock or _default_cost_shock(),
        regime_stress=regime_stress or _default_regime_stress(),
        ruin=ruin or _default_ruin_result(),
    )

    return output