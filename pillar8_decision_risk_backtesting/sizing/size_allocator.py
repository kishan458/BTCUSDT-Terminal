from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pillar8_decision_risk_backtesting.decision.conviction_engine import ConvictionResult
from pillar8_decision_risk_backtesting.decision.decision_gate_engine import DecisionGateResult
from pillar8_decision_risk_backtesting.risk.risk_score_engine import RiskScoreResult
from pillar8_decision_risk_backtesting.sizing.fractional_kelly_engine import FractionalKellyResult
from pillar8_decision_risk_backtesting.sizing.volatility_target_engine import VolatilityTargetResult
from pillar8_decision_risk_backtesting.state.decision_schema import FinalAction


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class SizeAllocationResult:
    base_size_from_volatility: float
    kelly_cap: float
    conviction_multiplier: float
    risk_multiplier: float
    action_multiplier: float
    final_size_fraction: float
    notes: str

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "base_size_from_volatility": self.base_size_from_volatility,
            "kelly_cap": self.kelly_cap,
            "conviction_multiplier": self.conviction_multiplier,
            "risk_multiplier": self.risk_multiplier,
            "action_multiplier": self.action_multiplier,
            "final_size_fraction": self.final_size_fraction,
            "notes": self.notes,
        }


def _resolve_action_multiplier(final_action: FinalAction) -> float:
    if final_action in (FinalAction.NO_TRADE, FinalAction.WATCHLIST, FinalAction.EXIT):
        return 0.0
    if final_action in (FinalAction.PROBE_LONG, FinalAction.PROBE_SHORT):
        return 0.5
    if final_action in (FinalAction.LONG, FinalAction.SHORT):
        return 1.0
    if final_action == FinalAction.REDUCE:
        return 0.25
    return 0.0


def allocate_position_size(
    *,
    volatility_target: VolatilityTargetResult,
    fractional_kelly: FractionalKellyResult,
    conviction: ConvictionResult,
    risk: RiskScoreResult,
    gate: DecisionGateResult,
    max_size_fraction: float = 1.0,
) -> SizeAllocationResult:
    """
    Final size logic:
    1. Start with volatility-based size
    2. Cap by Kelly-based edge limit
    3. Scale by conviction
    4. Scale down by risk
    5. Scale by action type (probe/full/no-trade)
    """

    if max_size_fraction <= 0:
        raise ValueError("max_size_fraction must be > 0")

    base_size = _clip(volatility_target.capped_size_fraction, 0.0, max_size_fraction)
    kelly_cap = _clip(fractional_kelly.capped_kelly_fraction, 0.0, max_size_fraction)

    conviction_multiplier = _clip(
        (0.6 * conviction.decision_confidence) + (0.4 * conviction.tradability_score),
        0.0,
        1.0,
    )

    risk_multiplier = _clip(1.0 - risk.risk_score, 0.0, 1.0)
    action_multiplier = _resolve_action_multiplier(gate.final_action)

    size_before_scaling = min(base_size, kelly_cap)
    final_size_fraction = (
        size_before_scaling
        * conviction_multiplier
        * risk_multiplier
        * action_multiplier
    )
    final_size_fraction = _clip(final_size_fraction, 0.0, max_size_fraction)

    if action_multiplier == 0.0:
        notes = "size_blocked_by_final_action"
    elif kelly_cap == 0.0:
        notes = "size_blocked_by_no_positive_edge"
    elif base_size == 0.0:
        notes = "size_blocked_by_volatility_input"
    elif risk.risk_score >= 0.85:
        notes = "size_heavily_suppressed_by_extreme_risk"
    elif gate.final_action in (FinalAction.PROBE_LONG, FinalAction.PROBE_SHORT):
        notes = "probe_size_only"
    else:
        notes = "ok"

    return SizeAllocationResult(
        base_size_from_volatility=round(base_size, 6),
        kelly_cap=round(kelly_cap, 6),
        conviction_multiplier=round(conviction_multiplier, 6),
        risk_multiplier=round(risk_multiplier, 6),
        action_multiplier=round(action_multiplier, 6),
        final_size_fraction=round(final_size_fraction, 6),
        notes=notes,
    )