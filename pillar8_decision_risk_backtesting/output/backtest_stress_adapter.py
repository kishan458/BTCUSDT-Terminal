from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from pillar8_decision_risk_backtesting.backtesting.backtest_runner import (
    BacktestRunnerResult,
    BacktestTradeInput,
    run_backtest,
)
from pillar8_decision_risk_backtesting.stress.monte_carlo_engine import (
    MonteCarloResult,
    run_monte_carlo_reshuffle,
)


@dataclass
class BacktestStressAdapterResult:
    backtest: BacktestRunnerResult
    monte_carlo: MonteCarloResult
    sample_size: int
    expectancy: float
    max_drawdown: float
    profit_factor: float
    ruin_probability: float
    mc_p05_equity: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "sample_size": self.sample_size,
            "expectancy": self.expectancy,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
            "ruin_probability": self.ruin_probability,
            "mc_p05_equity": self.mc_p05_equity,
            "backtest": self.backtest.to_dict(),
            "monte_carlo": self.monte_carlo.to_dict(),
        }


def _compute_ruin_probability(net_pnls: List[float], threshold: float = 0.0) -> float:
    """
    Simple current version:
    ruin probability = fraction of trades with pnl <= threshold.

    This is not final portfolio ruin modeling yet.
    It is a transparent interim proxy until we build deeper stress modules.
    """
    if not net_pnls:
        return 0.0

    ruined = sum(1 for x in net_pnls if x <= threshold)
    return ruined / len(net_pnls)


def build_backtest_stress_context(
    *,
    trade_inputs: List[BacktestTradeInput],
    mc_simulations: int = 1000,
    mc_seed: int = 42,
) -> BacktestStressAdapterResult:
    backtest_result = run_backtest(trade_inputs)
    net_pnls = [trade.net_pnl for trade in backtest_result.trades]

    monte_carlo_result = run_monte_carlo_reshuffle(
        net_pnls=net_pnls,
        simulations=mc_simulations,
        seed=mc_seed,
    )

    metrics = backtest_result.metrics
    sample_size = metrics.total_trades if metrics else 0
    expectancy = metrics.average_pnl if metrics else 0.0
    max_drawdown = metrics.max_drawdown if metrics else 0.0
    profit_factor = metrics.profit_factor if metrics else 0.0
    ruin_probability = _compute_ruin_probability(net_pnls)
    mc_p05_equity = monte_carlo_result.p05_terminal_pnl

    return BacktestStressAdapterResult(
        backtest=backtest_result,
        monte_carlo=monte_carlo_result,
        sample_size=sample_size,
        expectancy=round(expectancy, 6),
        max_drawdown=round(max_drawdown, 6),
        profit_factor=round(profit_factor, 6),
        ruin_probability=round(ruin_probability, 6),
        mc_p05_equity=round(mc_p05_equity, 6),
    )