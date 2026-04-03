from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from pillar8_decision_risk_backtesting.decision.decision_gate_engine import DecisionGateResult
from pillar8_decision_risk_backtesting.execution.holding_horizon_engine import HoldingHorizonResult
from pillar8_decision_risk_backtesting.risk.risk_score_engine import RiskScoreResult
from pillar8_decision_risk_backtesting.sizing.leverage_cap_engine import LeverageCapResult
from pillar8_decision_risk_backtesting.sizing.size_allocator import SizeAllocationResult
from pillar8_decision_risk_backtesting.state.decision_schema import Direction, FinalAction


@dataclass
class TradeConstructionResult:
    entry_style: str
    stop_framework: str
    target_framework: str
    invalidators: List[str] = field(default_factory=list)
    time_stop_bars: int = 0
    holding_horizon: str = "NONE"
    size_fraction: float = 0.0
    max_leverage_allowed: float = 1.0
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "entry_style": self.entry_style,
            "stop_framework": self.stop_framework,
            "target_framework": self.target_framework,
            "invalidators": self.invalidators,
            "time_stop_bars": self.time_stop_bars,
            "holding_horizon": self.holding_horizon,
            "size_fraction": self.size_fraction,
            "max_leverage_allowed": self.max_leverage_allowed,
            "notes": self.notes,
        }


def _infer_entry_style(gate: DecisionGateResult, risk: RiskScoreResult) -> str:
    if gate.final_action in (FinalAction.NO_TRADE, FinalAction.WATCHLIST, FinalAction.EXIT):
        return "NONE"

    if risk.risk_score >= 0.65:
        return "CONFIRMATION_ONLY"

    if gate.final_action in (FinalAction.PROBE_LONG, FinalAction.PROBE_SHORT):
        return "PROBE_ENTRY"

    return "STANDARD_ENTRY"


def _infer_stop_framework(direction: Direction, risk: RiskScoreResult) -> str:
    if direction == Direction.NONE:
        return "NONE"

    if risk.risk_score >= 0.65:
        return "TIGHT_INVALIDATION_STOP"

    return "STRUCTURE_PLUS_VOLATILITY_STOP"


def _infer_target_framework(direction: Direction, gate: DecisionGateResult) -> str:
    if direction == Direction.NONE:
        return "NONE"

    if gate.final_action in (FinalAction.PROBE_LONG, FinalAction.PROBE_SHORT):
        return "NEAR_LIQUIDITY_TARGET"

    return "LADDERED_LIQUIDITY_TARGETS"


def construct_trade_plan(
    *,
    gate: DecisionGateResult,
    risk: RiskScoreResult,
    size: SizeAllocationResult,
    leverage: LeverageCapResult,
    invalidators: List[str],
    horizon: HoldingHorizonResult,
) -> TradeConstructionResult:
    entry_style = _infer_entry_style(gate, risk)
    stop_framework = _infer_stop_framework(gate.direction, risk)
    target_framework = _infer_target_framework(gate.direction, gate)

    if gate.final_action in (FinalAction.NO_TRADE, FinalAction.WATCHLIST, FinalAction.EXIT):
        notes = "no_execution_plan_active"
    elif gate.final_action in (FinalAction.PROBE_LONG, FinalAction.PROBE_SHORT):
        notes = "probe_execution_plan"
    else:
        notes = "full_execution_plan"

    return TradeConstructionResult(
        entry_style=entry_style,
        stop_framework=stop_framework,
        target_framework=target_framework,
        invalidators=invalidators,
        time_stop_bars=horizon.time_stop_bars,
        holding_horizon=horizon.holding_horizon,
        size_fraction=round(size.final_size_fraction, 6),
        max_leverage_allowed=round(leverage.final_leverage_cap, 6),
        notes=notes,
    )