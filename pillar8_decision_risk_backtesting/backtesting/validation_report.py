from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from pillar8_decision_risk_backtesting.backtesting.metrics_engine import BacktestMetricsResult
from pillar8_decision_risk_backtesting.backtesting.regime_segmentation import (
    RegimeSegmentationResult,
    RegimeSegmentSummary,
)


@dataclass
class ValidationReportResult:
    overall_quality: str
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    regime_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "overall_quality": self.overall_quality,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "warnings": self.warnings,
            "regime_notes": self.regime_notes,
        }


def _evaluate_sample_size(metrics: BacktestMetricsResult, strengths: List[str], warnings: List[str]) -> None:
    if metrics.total_trades >= 100:
        strengths.append("Sample size is strong.")
    elif metrics.total_trades >= 30:
        strengths.append("Sample size is usable but still moderate.")
    else:
        warnings.append("Sample size is too small for strong statistical confidence.")


def _evaluate_expectancy(metrics: BacktestMetricsResult, strengths: List[str], weaknesses: List[str]) -> None:
    if metrics.average_pnl > 0:
        strengths.append("Expectancy is positive.")
    elif metrics.average_pnl < 0:
        weaknesses.append("Expectancy is negative.")
    else:
        weaknesses.append("Expectancy is flat.")


def _evaluate_profit_factor(metrics: BacktestMetricsResult, strengths: List[str], weaknesses: List[str]) -> None:
    if metrics.profit_factor >= 1.5:
        strengths.append("Profit factor is healthy.")
    elif metrics.profit_factor >= 1.0:
        strengths.append("Profit factor is positive but not strong.")
    else:
        weaknesses.append("Profit factor is weak.")


def _evaluate_drawdown(metrics: BacktestMetricsResult, strengths: List[str], warnings: List[str]) -> None:
    if metrics.max_drawdown <= 0:
        strengths.append("No drawdown observed in sample.")
    elif metrics.max_drawdown <= abs(metrics.total_net_pnl) * 0.25 if metrics.total_net_pnl != 0 else False:
        strengths.append("Drawdown is contained relative to total pnl.")
    else:
        warnings.append("Drawdown is heavy relative to total pnl.")


def _extract_top_segments(segment_map: Dict[str, RegimeSegmentSummary]) -> List[RegimeSegmentSummary]:
    return sorted(
        segment_map.values(),
        key=lambda x: x.total_net_pnl,
        reverse=True,
    )


def _evaluate_regime_concentration(
    segmentation: RegimeSegmentationResult,
    regime_notes: List[str],
    warnings: List[str],
) -> None:
    top_regimes = _extract_top_segments(segmentation.by_regime_state)
    if len(top_regimes) == 0:
        warnings.append("No regime segmentation data available.")
        return

    best = top_regimes[0]
    regime_notes.append(
        f"Best regime segment: {best.segment_name} | trades={best.trade_count} | total_net_pnl={best.total_net_pnl}"
    )

    if len(top_regimes) > 1:
        worst = top_regimes[-1]
        regime_notes.append(
            f"Worst regime segment: {worst.segment_name} | trades={worst.trade_count} | total_net_pnl={worst.total_net_pnl}"
        )

        if best.total_net_pnl > 0 and worst.total_net_pnl < 0:
            warnings.append("Performance is regime-sensitive and may not generalize well across market states.")


def build_validation_report(
    *,
    metrics: BacktestMetricsResult,
    segmentation: RegimeSegmentationResult,
) -> ValidationReportResult:
    strengths: List[str] = []
    weaknesses: List[str] = []
    warnings: List[str] = []
    regime_notes: List[str] = []

    _evaluate_sample_size(metrics, strengths, warnings)
    _evaluate_expectancy(metrics, strengths, weaknesses)
    _evaluate_profit_factor(metrics, strengths, weaknesses)
    _evaluate_drawdown(metrics, strengths, warnings)
    _evaluate_regime_concentration(segmentation, regime_notes, warnings)

    if weaknesses:
        overall_quality = "WEAK"
    elif warnings:
        overall_quality = "MODERATE"
    else:
        overall_quality = "STRONG"

    return ValidationReportResult(
        overall_quality=overall_quality,
        strengths=strengths,
        weaknesses=weaknesses,
        warnings=warnings,
        regime_notes=regime_notes,
    )