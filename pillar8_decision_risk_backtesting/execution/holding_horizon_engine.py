from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pillar8_decision_risk_backtesting.decision.decision_gate_engine import DecisionGateResult
from pillar8_decision_risk_backtesting.risk.risk_score_engine import RiskScoreResult
from pillar8_decision_risk_backtesting.state.decision_schema import DecisionState, FinalAction


@dataclass
class HoldingHorizonResult:
    holding_horizon: str
    time_stop_bars: int
    notes: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "holding_horizon": self.holding_horizon,
            "time_stop_bars": self.time_stop_bars,
            "notes": self.notes,
        }


def infer_holding_horizon(
    *,
    state: DecisionState,
    risk: RiskScoreResult,
    gate: DecisionGateResult,
) -> HoldingHorizonResult:
    if gate.final_action in (FinalAction.NO_TRADE, FinalAction.WATCHLIST, FinalAction.EXIT):
        return HoldingHorizonResult(
            holding_horizon="NONE",
            time_stop_bars=0,
            notes="no_live_trade_horizon",
        )

    high_event_uncertainty = state.events.base_uncertainty >= 0.65
    high_risk = risk.risk_score >= 0.65
    weak_breakout = state.candle.breakout_quality < 0.50
    strong_breakout = state.candle.breakout_quality >= 0.70
    strong_regime_fit = state.regime.strategy_compatibility >= 0.70

    if high_event_uncertainty or high_risk:
        return HoldingHorizonResult(
            holding_horizon="SHORT_DURATION",
            time_stop_bars=4,
            notes="compressed_due_to_risk",
        )

    if gate.final_action in (FinalAction.PROBE_LONG, FinalAction.PROBE_SHORT):
        return HoldingHorizonResult(
            holding_horizon="SHORT_DURATION",
            time_stop_bars=6,
            notes="probe_trade_short_horizon",
        )

    if strong_breakout and strong_regime_fit and not weak_breakout:
        return HoldingHorizonResult(
            holding_horizon="SWING",
            time_stop_bars=24,
            notes="trend_continuation_horizon",
        )

    return HoldingHorizonResult(
        holding_horizon="INTRADAY",
        time_stop_bars=12,
        notes="default_active_trade_horizon",
    )