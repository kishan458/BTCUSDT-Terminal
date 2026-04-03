from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pillar8_decision_risk_backtesting.state.decision_schema import DecisionState


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class MarketRiskResult:
    event_risk: float
    structure_risk: float
    candle_risk: float
    regime_risk: float
    market_risk_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "event_risk": self.event_risk,
            "structure_risk": self.structure_risk,
            "candle_risk": self.candle_risk,
            "regime_risk": self.regime_risk,
            "market_risk_score": self.market_risk_score,
        }


def _compute_event_risk(state: DecisionState) -> float:
    uncertainty = _clip(state.events.base_uncertainty)

    restrictions = state.events.trade_restrictions or {}
    allow_trade = restrictions.get("allow_trade")

    restriction_penalty = 1.0 if isinstance(allow_trade, bool) and allow_trade is False else 0.0
    return _clip((0.75 * uncertainty) + (0.25 * restriction_penalty))


def _compute_structure_risk(state: DecisionState) -> float:
    trap_risk = _clip(state.structure.trap_risk)
    liquidation_risk = _clip(state.structure.liquidation_risk)
    return _clip((0.55 * trap_risk) + (0.45 * liquidation_risk))


def _compute_candle_risk(state: DecisionState) -> float:
    failure_risk = _clip(state.candle.failure_risk)
    breakout_quality = _clip(state.candle.breakout_quality)
    return _clip((0.65 * failure_risk) + (0.35 * (1.0 - breakout_quality)))


def _compute_regime_risk(state: DecisionState) -> float:
    compatibility = _clip(state.regime.strategy_compatibility)
    return _clip(1.0 - compatibility)


def compute_market_risk(state: DecisionState) -> MarketRiskResult:
    event_risk = _compute_event_risk(state)
    structure_risk = _compute_structure_risk(state)
    candle_risk = _compute_candle_risk(state)
    regime_risk = _compute_regime_risk(state)

    market_risk_score = _clip(
        (0.28 * event_risk)
        + (0.28 * structure_risk)
        + (0.22 * candle_risk)
        + (0.22 * regime_risk)
    )

    return MarketRiskResult(
        event_risk=round(event_risk, 6),
        structure_risk=round(structure_risk, 6),
        candle_risk=round(candle_risk, 6),
        regime_risk=round(regime_risk, 6),
        market_risk_score=round(market_risk_score, 6),
    )