from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from pillar8_decision_risk_backtesting.decision.alignment_engine import AlignmentResult
from pillar8_decision_risk_backtesting.state.decision_schema import DecisionState


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class VetoResult:
    is_trade_blocked: bool
    vetoes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "is_trade_blocked": self.is_trade_blocked,
            "vetoes": self.vetoes,
            "warnings": self.warnings,
        }


def _event_restriction_active(state: DecisionState) -> bool:
    restrictions = state.events.trade_restrictions or {}
    allow_trade = restrictions.get("allow_trade")

    if isinstance(allow_trade, bool):
        return allow_trade is False

    return False


def evaluate_vetoes(state: DecisionState, alignment: AlignmentResult) -> VetoResult:
    vetoes: List[str] = []
    warnings: List[str] = []

    event_uncertainty = _clip(state.events.base_uncertainty)
    trap_risk = _clip(state.structure.trap_risk)
    liquidation_risk = _clip(state.structure.liquidation_risk)
    breakout_quality = _clip(state.candle.breakout_quality)
    failure_risk = _clip(state.candle.failure_risk)
    strategy_compatibility = _clip(state.regime.strategy_compatibility)
    council_conflict = _clip(state.council.conflict_score)
    council_confidence = _clip(state.council.confidence)

    if _event_restriction_active(state):
        vetoes.append("EVENT_TRADE_RESTRICTION_ACTIVE")

    if event_uncertainty >= 0.85:
        vetoes.append("EVENT_UNCERTAINTY_EXTREME")
    elif event_uncertainty >= 0.65:
        warnings.append("EVENT_UNCERTAINTY_ELEVATED")

    if trap_risk >= 0.85:
        vetoes.append("TRAP_RISK_EXTREME")
    elif trap_risk >= 0.65:
        warnings.append("TRAP_RISK_ELEVATED")

    if liquidation_risk >= 0.85:
        vetoes.append("LIQUIDATION_RISK_EXTREME")
    elif liquidation_risk >= 0.65:
        warnings.append("LIQUIDATION_RISK_ELEVATED")

    if breakout_quality <= 0.25 and failure_risk >= 0.75:
        vetoes.append("BREAKOUT_QUALITY_TOO_WEAK")
    elif breakout_quality <= 0.40 and failure_risk >= 0.60:
        warnings.append("BREAKOUT_SETUP_FRAGILE")

    if strategy_compatibility <= 0.20:
        vetoes.append("REGIME_STRATEGY_MISMATCH")
    elif strategy_compatibility <= 0.40:
        warnings.append("REGIME_COMPATIBILITY_WEAK")

    if council_conflict >= 0.80 and council_confidence <= 0.40:
        vetoes.append("COUNCIL_CONFLICT_TOO_HIGH")
    elif council_conflict >= 0.60:
        warnings.append("COUNCIL_CONFLICT_ELEVATED")

    if alignment.directional_conflict >= 0.75:
        vetoes.append("CROSS_PILLAR_DIRECTIONAL_CONFLICT")
    elif alignment.directional_conflict >= 0.50:
        warnings.append("CROSS_PILLAR_ALIGNMENT_MIXED")

    if alignment.dominant_direction == "NONE" and abs(alignment.net_directional_edge) < 0.10:
        warnings.append("NO_CLEAR_DIRECTIONAL_EDGE")

    return VetoResult(
        is_trade_blocked=len(vetoes) > 0,
        vetoes=vetoes,
        warnings=warnings,
    )