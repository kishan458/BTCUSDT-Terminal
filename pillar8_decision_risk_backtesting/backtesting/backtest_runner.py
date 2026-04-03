from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from pillar8_decision_risk_backtesting.backtesting.fill_model import (
    TradeFillResult,
    compute_trade_fill,
)
from pillar8_decision_risk_backtesting.backtesting.metrics_engine import (
    BacktestMetricsResult,
    compute_backtest_metrics,
)
from pillar8_decision_risk_backtesting.state.decision_schema import Direction


@dataclass
class BacktestTradeInput:
    direction: Direction
    entry_price: float
    exit_price: float
    position_size: float
    entry_fee_rate: float = 0.0004
    exit_fee_rate: float = 0.0004
    slippage_rate: float = 0.0005


@dataclass
class BacktestRunnerResult:
    trades: List[TradeFillResult] = field(default_factory=list)
    metrics: BacktestMetricsResult | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "trades": [trade.to_dict() for trade in self.trades],
            "metrics": self.metrics.to_dict() if self.metrics else {},
        }


def run_backtest(trade_inputs: List[BacktestTradeInput]) -> BacktestRunnerResult:
    if not isinstance(trade_inputs, list):
        raise ValueError("trade_inputs must be a list of BacktestTradeInput objects")

    trades: List[TradeFillResult] = []
    net_pnls: List[float] = []

    for trade_input in trade_inputs:
        if not isinstance(trade_input, BacktestTradeInput):
            raise ValueError("All items in trade_inputs must be BacktestTradeInput instances")

        fill_result = compute_trade_fill(
            direction=trade_input.direction,
            entry_price=trade_input.entry_price,
            exit_price=trade_input.exit_price,
            position_size=trade_input.position_size,
            entry_fee_rate=trade_input.entry_fee_rate,
            exit_fee_rate=trade_input.exit_fee_rate,
            slippage_rate=trade_input.slippage_rate,
        )

        trades.append(fill_result)
        net_pnls.append(fill_result.net_pnl)

    metrics = compute_backtest_metrics(net_pnls)

    return BacktestRunnerResult(
        trades=trades,
        metrics=metrics,
    )