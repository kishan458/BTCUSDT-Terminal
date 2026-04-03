from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DrawdownStressResult:
    max_drawdown: float
    mean_drawdown: float
    drawdown_breach_threshold: float
    breach_count: int
    breach_rate: float

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "max_drawdown": self.max_drawdown,
            "mean_drawdown": self.mean_drawdown,
            "drawdown_breach_threshold": self.drawdown_breach_threshold,
            "breach_count": self.breach_count,
            "breach_rate": self.breach_rate,
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


def compute_drawdown_stress(
    pnls: List[float],
    *,
    breach_threshold: float,
) -> DrawdownStressResult:
    clean_pnls = _validate_numeric_list(pnls)

    if breach_threshold < 0:
        raise ValueError("breach_threshold must be >= 0")

    if len(clean_pnls) == 0:
        return DrawdownStressResult(
            max_drawdown=0.0,
            mean_drawdown=0.0,
            drawdown_breach_threshold=round(float(breach_threshold), 6),
            breach_count=0,
            breach_rate=0.0,
        )

    equity = 0.0
    peak = 0.0
    drawdowns: List[float] = []

    for pnl in clean_pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        drawdowns.append(drawdown)

    max_drawdown = max(drawdowns) if drawdowns else 0.0
    mean_drawdown = sum(drawdowns) / len(drawdowns) if drawdowns else 0.0
    breach_count = sum(1 for x in drawdowns if x >= breach_threshold)
    breach_rate = breach_count / len(drawdowns) if drawdowns else 0.0

    return DrawdownStressResult(
        max_drawdown=round(max_drawdown, 6),
        mean_drawdown=round(mean_drawdown, 6),
        drawdown_breach_threshold=round(float(breach_threshold), 6),
        breach_count=breach_count,
        breach_rate=round(breach_rate, 6),
    )