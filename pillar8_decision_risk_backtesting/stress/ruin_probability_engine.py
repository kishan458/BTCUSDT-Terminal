from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List


@dataclass
class RuinProbabilityResult:
    simulations: int
    ruin_count: int
    ruin_probability: float
    avg_terminal_equity: float
    p05_terminal_equity: float

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "simulations": self.simulations,
            "ruin_count": self.ruin_count,
            "ruin_probability": self.ruin_probability,
            "avg_terminal_equity": self.avg_terminal_equity,
            "p05_terminal_equity": self.p05_terminal_equity,
        }


def _validate_numeric_list(values: List[float]) -> List[float]:
    if not isinstance(values, list):
        raise ValueError("pnls must be a list")

    clean: List[float] = []
    for value in values:
        try:
            clean.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError("All pnl values must be numeric") from exc
    return clean


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0

    index = q * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower

    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def run_ruin_probability_simulation(
    pnls: List[float],
    *,
    initial_equity: float = 1000.0,
    ruin_threshold: float = 200.0,
    simulations: int = 1000,
    seed: int = 42,
) -> RuinProbabilityResult:
    clean_pnls = _validate_numeric_list(pnls)

    if simulations <= 0:
        raise ValueError("simulations must be > 0")

    if initial_equity <= 0:
        raise ValueError("initial_equity must be > 0")

    if ruin_threshold < 0:
        raise ValueError("ruin_threshold must be >= 0")

    if len(clean_pnls) == 0:
        return RuinProbabilityResult(
            simulations=simulations,
            ruin_count=0,
            ruin_probability=0.0,
            avg_terminal_equity=initial_equity,
            p05_terminal_equity=initial_equity,
        )

    rng = random.Random(seed)

    ruin_count = 0
    terminal_equities: List[float] = []

    for _ in range(simulations):
        equity = initial_equity

        shuffled = clean_pnls[:]
        rng.shuffle(shuffled)

        for pnl in shuffled:
            equity += pnl

            if equity <= ruin_threshold:
                ruin_count += 1
                break

        terminal_equities.append(equity)

    ruin_probability = ruin_count / simulations
    avg_terminal_equity = sum(terminal_equities) / len(terminal_equities)

    terminal_equities_sorted = sorted(terminal_equities)
    p05_terminal_equity = _percentile(terminal_equities_sorted, 0.05)

    return RuinProbabilityResult(
        simulations=simulations,
        ruin_count=ruin_count,
        ruin_probability=round(ruin_probability, 6),
        avg_terminal_equity=round(avg_terminal_equity, 6),
        p05_terminal_equity=round(p05_terminal_equity, 6),
    )