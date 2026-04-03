from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class SlippageResult:
    slippage_rate: float
    notional_traded: float
    slippage_cost: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "slippage_rate": self.slippage_rate,
            "notional_traded": self.notional_traded,
            "slippage_cost": self.slippage_cost,
        }


def compute_slippage_cost(
    *,
    notional_traded: float,
    slippage_rate: float,
) -> SlippageResult:
    """
    Computes slippage cost as:

        slippage_cost = notional_traded * slippage_rate
    """

    try:
        notional_traded = float(notional_traded)
        slippage_rate = float(slippage_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("All inputs must be numeric.") from exc

    if notional_traded < 0:
        raise ValueError("notional_traded must be >= 0")

    if slippage_rate < 0:
        raise ValueError("slippage_rate must be >= 0")

    slippage_cost = notional_traded * slippage_rate

    return SlippageResult(
        slippage_rate=round(slippage_rate, 8),
        notional_traded=round(notional_traded, 6),
        slippage_cost=round(slippage_cost, 6),
    )