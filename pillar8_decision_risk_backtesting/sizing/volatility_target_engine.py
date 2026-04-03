from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class VolatilityTargetResult:
    realized_volatility: float
    target_volatility: float
    raw_size_fraction: float
    capped_size_fraction: float
    annualization_factor: float
    notes: str

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "realized_volatility": self.realized_volatility,
            "target_volatility": self.target_volatility,
            "raw_size_fraction": self.raw_size_fraction,
            "capped_size_fraction": self.capped_size_fraction,
            "annualization_factor": self.annualization_factor,
            "notes": self.notes,
        }


def compute_volatility_target_size(
    *,
    realized_volatility: Optional[float],
    target_volatility: float,
    max_size_fraction: float = 1.0,
    min_volatility_floor: float = 1e-6,
    annualization_factor: float = 1.0,
) -> VolatilityTargetResult:
    """
    Computes base size using inverse volatility targeting.

    Formula:
        raw_size_fraction = target_volatility / max(realized_volatility, min_volatility_floor)

    Notes:
    - realized_volatility must be supplied by upstream real calculations
    - annualization_factor is passed through for audit visibility only here
    - final size is capped by max_size_fraction
    """

    if target_volatility <= 0:
        raise ValueError("target_volatility must be > 0")

    if max_size_fraction <= 0:
        raise ValueError("max_size_fraction must be > 0")

    if min_volatility_floor <= 0:
        raise ValueError("min_volatility_floor must be > 0")

    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be > 0")

    if realized_volatility is None:
        return VolatilityTargetResult(
            realized_volatility=0.0,
            target_volatility=round(target_volatility, 6),
            raw_size_fraction=0.0,
            capped_size_fraction=0.0,
            annualization_factor=round(annualization_factor, 6),
            notes="realized_volatility_missing",
        )

    try:
        realized_volatility = float(realized_volatility)
    except (TypeError, ValueError) as exc:
        raise ValueError("realized_volatility must be numeric or None") from exc

    if realized_volatility < 0:
        raise ValueError("realized_volatility must be >= 0")

    effective_volatility = max(realized_volatility, min_volatility_floor)
    raw_size_fraction = target_volatility / effective_volatility
    capped_size_fraction = min(raw_size_fraction, max_size_fraction)
    capped_size_fraction = _clip(capped_size_fraction, 0.0, max_size_fraction)

    if realized_volatility == 0:
        notes = "volatility_floor_applied"
    elif raw_size_fraction > max_size_fraction:
        notes = "size_capped_by_max_fraction"
    else:
        notes = "ok"

    return VolatilityTargetResult(
        realized_volatility=round(realized_volatility, 6),
        target_volatility=round(target_volatility, 6),
        raw_size_fraction=round(raw_size_fraction, 6),
        capped_size_fraction=round(capped_size_fraction, 6),
        annualization_factor=round(annualization_factor, 6),
        notes=notes,
    )