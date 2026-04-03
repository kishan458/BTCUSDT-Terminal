from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BacktestMetricsResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_net_pnl: float
    average_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    max_drawdown: float

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_net_pnl": self.total_net_pnl,
            "average_pnl": self.average_pnl,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
        }


def _compute_max_drawdown(pnls: List[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for pnl in pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown


def compute_backtest_metrics(net_pnls: List[float]) -> BacktestMetricsResult:
    if not isinstance(net_pnls, list):
        raise ValueError("net_pnls must be a list of numeric values")

    if len(net_pnls) == 0:
        return BacktestMetricsResult(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_net_pnl=0.0,
            average_pnl=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
        )

    clean_pnls: List[float] = []
    for pnl in net_pnls:
        try:
            clean_pnls.append(float(pnl))
        except (TypeError, ValueError) as exc:
            raise ValueError("All pnl values must be numeric") from exc

    total_trades = len(clean_pnls)
    winning_trades = sum(1 for x in clean_pnls if x > 0)
    losing_trades = sum(1 for x in clean_pnls if x < 0)

    total_net_pnl = sum(clean_pnls)
    average_pnl = total_net_pnl / total_trades if total_trades > 0 else 0.0

    gross_profit = sum(x for x in clean_pnls if x > 0)
    gross_loss = abs(sum(x for x in clean_pnls if x < 0))

    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
    max_drawdown = _compute_max_drawdown(clean_pnls)

    return BacktestMetricsResult(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=round(win_rate, 6),
        total_net_pnl=round(total_net_pnl, 6),
        average_pnl=round(average_pnl, 6),
        gross_profit=round(gross_profit, 6),
        gross_loss=round(gross_loss, 6),
        profit_factor=round(profit_factor, 6),
        max_drawdown=round(max_drawdown, 6),
    )