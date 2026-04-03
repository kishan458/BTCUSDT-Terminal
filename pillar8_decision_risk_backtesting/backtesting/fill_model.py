from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pillar8_decision_risk_backtesting.backtesting.cost_model import compute_trading_cost
from pillar8_decision_risk_backtesting.backtesting.slippage_model import compute_slippage_cost
from pillar8_decision_risk_backtesting.state.decision_schema import Direction


@dataclass
class TradeFillResult:
    direction: str
    entry_price: float
    exit_price: float
    position_size: float
    gross_pnl: float
    fee_cost: float
    slippage_cost: float
    net_pnl: float

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "position_size": self.position_size,
            "gross_pnl": self.gross_pnl,
            "fee_cost": self.fee_cost,
            "slippage_cost": self.slippage_cost,
            "net_pnl": self.net_pnl,
        }


def _compute_gross_pnl(
    direction: Direction,
    entry_price: float,
    exit_price: float,
    position_size: float,
) -> float:
    if direction == Direction.LONG:
        return (exit_price - entry_price) * position_size
    elif direction == Direction.SHORT:
        return (entry_price - exit_price) * position_size
    return 0.0


def compute_trade_fill(
    *,
    direction: Direction,
    entry_price: float,
    exit_price: float,
    position_size: float,
    entry_fee_rate: float = 0.0004,
    exit_fee_rate: float = 0.0004,
    slippage_rate: float = 0.0005,
) -> TradeFillResult:
    """
    Full trade PnL calculation including:
    - gross pnl
    - fees
    - slippage
    - net pnl
    """

    try:
        entry_price = float(entry_price)
        exit_price = float(exit_price)
        position_size = float(position_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("Price and size inputs must be numeric.") from exc

    if entry_price <= 0 or exit_price <= 0:
        raise ValueError("Prices must be > 0")

    if position_size < 0:
        raise ValueError("position_size must be >= 0")

    notional_traded = entry_price * position_size

    gross_pnl = _compute_gross_pnl(
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        position_size=position_size,
    )

    fee_result = compute_trading_cost(
        notional_traded=notional_traded,
        entry_fee_rate=entry_fee_rate,
        exit_fee_rate=exit_fee_rate,
    )

    slippage_result = compute_slippage_cost(
        notional_traded=notional_traded,
        slippage_rate=slippage_rate,
    )

    total_cost = fee_result.total_fee_cost + slippage_result.slippage_cost
    net_pnl = gross_pnl - total_cost

    return TradeFillResult(
        direction=direction.value,
        entry_price=round(entry_price, 6),
        exit_price=round(exit_price, 6),
        position_size=round(position_size, 6),
        gross_pnl=round(gross_pnl, 6),
        fee_cost=round(fee_result.total_fee_cost, 6),
        slippage_cost=round(slippage_result.slippage_cost, 6),
        net_pnl=round(net_pnl, 6),
    )