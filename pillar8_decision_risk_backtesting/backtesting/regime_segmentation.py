from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RegimeTaggedTrade:
    net_pnl: float
    regime_state: str = "UNKNOWN"
    volatility_bucket: str = "UNKNOWN"
    event_state: str = "UNKNOWN"
    strategy_compatibility_bucket: str = "UNKNOWN"
    tag: str = "UNSPECIFIED"

    def to_dict(self) -> Dict[str, object]:
        return {
            "net_pnl": self.net_pnl,
            "regime_state": self.regime_state,
            "volatility_bucket": self.volatility_bucket,
            "event_state": self.event_state,
            "strategy_compatibility_bucket": self.strategy_compatibility_bucket,
            "tag": self.tag,
        }


@dataclass
class RegimeSegmentSummary:
    segment_name: str
    trade_count: int
    total_net_pnl: float
    average_net_pnl: float
    win_rate: float
    gross_profit: float
    gross_loss: float
    profit_factor: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "segment_name": self.segment_name,
            "trade_count": self.trade_count,
            "total_net_pnl": self.total_net_pnl,
            "average_net_pnl": self.average_net_pnl,
            "win_rate": self.win_rate,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "profit_factor": self.profit_factor,
        }


@dataclass
class RegimeSegmentationResult:
    by_regime_state: Dict[str, RegimeSegmentSummary] = field(default_factory=dict)
    by_volatility_bucket: Dict[str, RegimeSegmentSummary] = field(default_factory=dict)
    by_event_state: Dict[str, RegimeSegmentSummary] = field(default_factory=dict)
    by_strategy_compatibility_bucket: Dict[str, RegimeSegmentSummary] = field(default_factory=dict)
    by_tag: Dict[str, RegimeSegmentSummary] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "by_regime_state": {k: v.to_dict() for k, v in self.by_regime_state.items()},
            "by_volatility_bucket": {k: v.to_dict() for k, v in self.by_volatility_bucket.items()},
            "by_event_state": {k: v.to_dict() for k, v in self.by_event_state.items()},
            "by_strategy_compatibility_bucket": {
                k: v.to_dict() for k, v in self.by_strategy_compatibility_bucket.items()
            },
            "by_tag": {k: v.to_dict() for k, v in self.by_tag.items()},
        }


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric pnl, got {value!r}") from exc


def _bucket_strategy_compatibility(value: Optional[float]) -> str:
    if value is None:
        return "UNKNOWN"

    value = float(value)
    if value >= 0.70:
        return "HIGH_COMPATIBILITY"
    if value >= 0.40:
        return "MEDIUM_COMPATIBILITY"
    return "LOW_COMPATIBILITY"


def _build_segment_summary(segment_name: str, pnls: List[float]) -> RegimeSegmentSummary:
    trade_count = len(pnls)
    total_net_pnl = sum(pnls)
    average_net_pnl = total_net_pnl / trade_count if trade_count > 0 else 0.0
    winning_trades = sum(1 for x in pnls if x > 0)
    win_rate = winning_trades / trade_count if trade_count > 0 else 0.0
    gross_profit = sum(x for x in pnls if x > 0)
    gross_loss = abs(sum(x for x in pnls if x < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    return RegimeSegmentSummary(
        segment_name=segment_name,
        trade_count=trade_count,
        total_net_pnl=round(total_net_pnl, 6),
        average_net_pnl=round(average_net_pnl, 6),
        win_rate=round(win_rate, 6),
        gross_profit=round(gross_profit, 6),
        gross_loss=round(gross_loss, 6),
        profit_factor=round(profit_factor, 6),
    )


def _group_and_summarize(
    trades: List[RegimeTaggedTrade],
    attr_name: str,
) -> Dict[str, RegimeSegmentSummary]:
    grouped: Dict[str, List[float]] = {}

    for trade in trades:
        key = getattr(trade, attr_name)
        grouped.setdefault(key, []).append(trade.net_pnl)

    return {
        key: _build_segment_summary(segment_name=key, pnls=pnls)
        for key, pnls in grouped.items()
    }


def build_regime_tagged_trade(
    *,
    net_pnl: float,
    regime_state: str = "UNKNOWN",
    volatility_bucket: str = "UNKNOWN",
    event_state: str = "UNKNOWN",
    strategy_compatibility: Optional[float] = None,
    tag: str = "UNSPECIFIED",
) -> RegimeTaggedTrade:
    return RegimeTaggedTrade(
        net_pnl=round(_safe_float(net_pnl), 6),
        regime_state=str(regime_state),
        volatility_bucket=str(volatility_bucket),
        event_state=str(event_state),
        strategy_compatibility_bucket=_bucket_strategy_compatibility(strategy_compatibility),
        tag=str(tag),
    )


def segment_backtest_results(trades: List[RegimeTaggedTrade]) -> RegimeSegmentationResult:
    if not isinstance(trades, list):
        raise ValueError("trades must be a list of RegimeTaggedTrade")

    for trade in trades:
        if not isinstance(trade, RegimeTaggedTrade):
            raise ValueError("All items must be RegimeTaggedTrade instances")

    return RegimeSegmentationResult(
        by_regime_state=_group_and_summarize(trades, "regime_state"),
        by_volatility_bucket=_group_and_summarize(trades, "volatility_bucket"),
        by_event_state=_group_and_summarize(trades, "event_state"),
        by_strategy_compatibility_bucket=_group_and_summarize(
            trades, "strategy_compatibility_bucket"
        ),
        by_tag=_group_and_summarize(trades, "tag"),
    )