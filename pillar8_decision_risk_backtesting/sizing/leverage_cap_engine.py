from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pillar8_decision_risk_backtesting.state.decision_schema import DecisionState


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class LeverageCapResult:
    base_leverage_cap: float
    event_penalty: float
    structure_penalty: float
    final_leverage_cap: float
    notes: str

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "base_leverage_cap": self.base_leverage_cap,
            "event_penalty": self.event_penalty,
            "structure_penalty": self.structure_penalty,
            "final_leverage_cap": self.final_leverage_cap,
            "notes": self.notes,
        }


def compute_leverage_cap(
    *,
    state: DecisionState,
    risk_score: float,
    max_leverage: float = 3.0,
    min_leverage: float = 1.0,
) -> LeverageCapResult:
    """
    Computes leverage cap using:
    - aggregate risk score
    - event uncertainty
    - structure stress (trap + liquidation risk)

    Logic:
    1. start from max_leverage
    2. scale down by aggregate risk
    3. apply event penalty
    4. apply structure penalty
    5. never go below min_leverage
    """

    if max_leverage < min_leverage:
        raise ValueError("max_leverage must be >= min_leverage")

    if min_leverage <= 0:
        raise ValueError("min_leverage must be > 0")

    risk_score = _clip(float(risk_score), 0.0, 1.0)

    event_uncertainty = _clip(float(state.events.base_uncertainty), 0.0, 1.0)
    trap_risk = _clip(float(state.structure.trap_risk), 0.0, 1.0)
    liquidation_risk = _clip(float(state.structure.liquidation_risk), 0.0, 1.0)

    base_leverage_cap = max_leverage - ((max_leverage - min_leverage) * risk_score)

    event_penalty = 1.0 - (0.50 * event_uncertainty)
    structure_stress = (0.5 * trap_risk) + (0.5 * liquidation_risk)
    structure_penalty = 1.0 - (0.50 * structure_stress)

    raw_final = base_leverage_cap * event_penalty * structure_penalty
    final_leverage_cap = _clip(raw_final, min_leverage, max_leverage)

    if risk_score >= 0.85:
        notes = "extreme_risk_leverage_suppressed"
    elif event_uncertainty >= 0.65:
        notes = "event_risk_leverage_reduced"
    elif structure_stress >= 0.65:
        notes = "structure_risk_leverage_reduced"
    else:
        notes = "ok"

    return LeverageCapResult(
        base_leverage_cap=round(base_leverage_cap, 6),
        event_penalty=round(event_penalty, 6),
        structure_penalty=round(structure_penalty, 6),
        final_leverage_cap=round(final_leverage_cap, 6),
        notes=notes,
    )