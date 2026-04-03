from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List


@dataclass
class MonteCarloResult:
    simulations: int
    original_total_pnl: float
    mean_terminal_pnl: float
    p05_terminal_pnl: float
    p50_terminal_pnl: float
    p95_terminal_pnl: float
    worst_max_drawdown: float
    mean_max_drawdown: float

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "simulations": self.simulations,
            "original_total_pnl": self.original_total_pnl,
            "mean_terminal_pnl": self.mean_terminal_pnl,
            "p05_terminal_pnl": self.p05_terminal_pnl,
            "p50_terminal_pnl": self.p50_terminal_pnl,
            "p95_terminal_pnl": self.p95_terminal_pnl,
            "worst_max_drawdown": self.worst_max_drawdown,
            "mean_max_drawdown": self.mean_max_drawdown,
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


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0

    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]

    index = q * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower

    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def run_monte_carlo_reshuffle(
    net_pnls: List[float],
    simulations: int = 1000,
    seed: int = 42,
) -> MonteCarloResult:
    if not isinstance(net_pnls, list):
        raise ValueError("net_pnls must be a list")

    if simulations <= 0:
        raise ValueError("simulations must be > 0")

    clean_pnls: List[float] = []
    for pnl in net_pnls:
        try:
            clean_pnls.append(float(pnl))
        except (TypeError, ValueError) as exc:
            raise ValueError("All pnl values must be numeric") from exc

    if len(clean_pnls) == 0:
        return MonteCarloResult(
            simulations=simulations,
            original_total_pnl=0.0,
            mean_terminal_pnl=0.0,
            p05_terminal_pnl=0.0,
            p50_terminal_pnl=0.0,
            p95_terminal_pnl=0.0,
            worst_max_drawdown=0.0,
            mean_max_drawdown=0.0,
        )

    rng = random.Random(seed)

    terminal_pnls: List[float] = []
    max_drawdowns: List[float] = []

    original_total_pnl = sum(clean_pnls)

    for _ in range(simulations):
        shuffled = clean_pnls[:]
        rng.shuffle(shuffled)

        terminal_pnls.append(sum(shuffled))
        max_drawdowns.append(_compute_max_drawdown(shuffled))

    terminal_pnls_sorted = sorted(terminal_pnls)
    mean_terminal_pnl = sum(terminal_pnls) / len(terminal_pnls)
    p05_terminal_pnl = _percentile(terminal_pnls_sorted, 0.05)
    p50_terminal_pnl = _percentile(terminal_pnls_sorted, 0.50)
    p95_terminal_pnl = _percentile(terminal_pnls_sorted, 0.95)

    mean_max_drawdown = sum(max_drawdowns) / len(max_drawdowns)
    worst_max_drawdown = max(max_drawdowns) if max_drawdowns else 0.0

    return MonteCarloResult(
        simulations=simulations,
        original_total_pnl=round(original_total_pnl, 6),
        mean_terminal_pnl=round(mean_terminal_pnl, 6),
        p05_terminal_pnl=round(p05_terminal_pnl, 6),
        p50_terminal_pnl=round(p50_terminal_pnl, 6),
        p95_terminal_pnl=round(p95_terminal_pnl, 6),
        worst_max_drawdown=round(worst_max_drawdown, 6),
        mean_max_drawdown=round(mean_max_drawdown, 6),
    )