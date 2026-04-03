from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class TradingCostResult:
    entry_fee_rate: float
    exit_fee_rate: float
    total_fee_rate: float
    notional_traded: float
    total_fee_cost: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "entry_fee_rate": self.entry_fee_rate,
            "exit_fee_rate": self.exit_fee_rate,
            "total_fee_rate": self.total_fee_rate,
            "notional_traded": self.notional_traded,
            "total_fee_cost": self.total_fee_cost,
        }


def compute_trading_cost(
    *,
    notional_traded: float,
    entry_fee_rate: float,
    exit_fee_rate: float,
) -> TradingCostResult:
    """
    Computes simple round-trip fee cost.

    Example:
    - notional_traded = 1000
    - entry_fee_rate = 0.0004
    - exit_fee_rate = 0.0004

    total_fee_rate = 0.0008
    total_fee_cost = 1000 * 0.0008 = 0.8
    """

    try:
        notional_traded = float(notional_traded)
        entry_fee_rate = float(entry_fee_rate)
        exit_fee_rate = float(exit_fee_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("All inputs must be numeric.") from exc

    if notional_traded < 0:
        raise ValueError("notional_traded must be >= 0")

    if entry_fee_rate < 0:
        raise ValueError("entry_fee_rate must be >= 0")

    if exit_fee_rate < 0:
        raise ValueError("exit_fee_rate must be >= 0")

    total_fee_rate = entry_fee_rate + exit_fee_rate
    total_fee_cost = notional_traded * total_fee_rate

    return TradingCostResult(
        entry_fee_rate=round(entry_fee_rate, 8),
        exit_fee_rate=round(exit_fee_rate, 8),
        total_fee_rate=round(total_fee_rate, 8),
        notional_traded=round(notional_traded, 6),
        total_fee_cost=round(total_fee_cost, 6),
    )