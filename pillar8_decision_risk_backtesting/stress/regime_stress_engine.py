from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pillar8_decision_risk_backtesting.backtesting.regime_segmentation import (
    RegimeSegmentationResult,
)


@dataclass
class RegimeStressResult:
    best_regime: str
    worst_regime: str
    best_regime_pnl: float
    worst_regime_pnl: float
    regime_spread: float
    regime_fragility_score: float
    regime_fragility_label: str

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "best_regime": self.best_regime,
            "worst_regime": self.worst_regime,
            "best_regime_pnl": self.best_regime_pnl,
            "worst_regime_pnl": self.worst_regime_pnl,
            "regime_spread": self.regime_spread,
            "regime_fragility_score": self.regime_fragility_score,
            "regime_fragility_label": self.regime_fragility_label,
        }


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def run_regime_stress(
    segmentation: RegimeSegmentationResult,
) -> RegimeStressResult:
    regime_map = segmentation.by_regime_state

    if not regime_map:
        return RegimeStressResult(
            best_regime="UNKNOWN",
            worst_regime="UNKNOWN",
            best_regime_pnl=0.0,
            worst_regime_pnl=0.0,
            regime_spread=0.0,
            regime_fragility_score=0.0,
            regime_fragility_label="UNKNOWN",
        )

    summaries = list(regime_map.values())
    best = max(summaries, key=lambda x: x.total_net_pnl)
    worst = min(summaries, key=lambda x: x.total_net_pnl)

    regime_spread = best.total_net_pnl - worst.total_net_pnl

    denominator = abs(best.total_net_pnl) + abs(worst.total_net_pnl)
    if denominator == 0:
        fragility_score = 0.0
    else:
        fragility_score = regime_spread / denominator

    fragility_score = _clip(fragility_score, 0.0, 1.0)

    if fragility_score >= 0.75:
        label = "HIGH_FRAGILITY"
    elif fragility_score >= 0.40:
        label = "MODERATE_FRAGILITY"
    else:
        label = "LOW_FRAGILITY"

    return RegimeStressResult(
        best_regime=best.segment_name,
        worst_regime=worst.segment_name,
        best_regime_pnl=round(best.total_net_pnl, 6),
        worst_regime_pnl=round(worst.total_net_pnl, 6),
        regime_spread=round(regime_spread, 6),
        regime_fragility_score=round(fragility_score, 6),
        regime_fragility_label=label,
    )