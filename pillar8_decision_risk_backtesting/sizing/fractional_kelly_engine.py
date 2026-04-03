from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class FractionalKellyResult:
    win_probability: float
    loss_probability: float
    payoff_ratio: float
    full_kelly_fraction: float
    fractional_kelly_fraction: float
    capped_kelly_fraction: float
    notes: str

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "win_probability": self.win_probability,
            "loss_probability": self.loss_probability,
            "payoff_ratio": self.payoff_ratio,
            "full_kelly_fraction": self.full_kelly_fraction,
            "fractional_kelly_fraction": self.fractional_kelly_fraction,
            "capped_kelly_fraction": self.capped_kelly_fraction,
            "notes": self.notes,
        }


def compute_fractional_kelly_size(
    *,
    win_probability: float,
    payoff_ratio: float,
    kelly_fraction: float = 0.25,
    max_size_fraction: float = 1.0,
) -> FractionalKellyResult:
    """
    Kelly formula:
        f* = p - (q / b)

    where:
        p = win probability
        q = loss probability = 1 - p
        b = payoff ratio (avg win / avg loss)

    We then apply a fractional Kelly multiplier:
        fractional_kelly = full_kelly * kelly_fraction

    Final result is capped into [0, max_size_fraction].
    Negative Kelly values are clipped to 0.
    """

    try:
        win_probability = float(win_probability)
        payoff_ratio = float(payoff_ratio)
        kelly_fraction = float(kelly_fraction)
        max_size_fraction = float(max_size_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("All inputs must be numeric.") from exc

    if not (0.0 <= win_probability <= 1.0):
        raise ValueError("win_probability must be between 0 and 1")

    if payoff_ratio <= 0:
        raise ValueError("payoff_ratio must be > 0")

    if not (0.0 <= kelly_fraction <= 1.0):
        raise ValueError("kelly_fraction must be between 0 and 1")

    if max_size_fraction <= 0:
        raise ValueError("max_size_fraction must be > 0")

    loss_probability = 1.0 - win_probability
    full_kelly_fraction = win_probability - (loss_probability / payoff_ratio)
    clipped_full_kelly = max(0.0, full_kelly_fraction)

    fractional_kelly_fraction = clipped_full_kelly * kelly_fraction
    capped_kelly_fraction = _clip(fractional_kelly_fraction, 0.0, max_size_fraction)

    if full_kelly_fraction <= 0:
        notes = "no_positive_edge"
    elif fractional_kelly_fraction > max_size_fraction:
        notes = "kelly_capped_by_max_fraction"
    else:
        notes = "ok"

    return FractionalKellyResult(
        win_probability=round(win_probability, 6),
        loss_probability=round(loss_probability, 6),
        payoff_ratio=round(payoff_ratio, 6),
        full_kelly_fraction=round(full_kelly_fraction, 6),
        fractional_kelly_fraction=round(fractional_kelly_fraction, 6),
        capped_kelly_fraction=round(capped_kelly_fraction, 6),
        notes=notes,
    )