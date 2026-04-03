from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from pillar8_decision_risk_backtesting.backtesting.metrics_engine import (
    BacktestMetricsResult,
    compute_backtest_metrics,
)


@dataclass
class WalkForwardFoldResult:
    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_metrics: BacktestMetricsResult
    test_metrics: BacktestMetricsResult

    def to_dict(self) -> Dict[str, object]:
        return {
            "fold_index": self.fold_index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_metrics": self.train_metrics.to_dict(),
            "test_metrics": self.test_metrics.to_dict(),
        }


@dataclass
class WalkForwardRunnerResult:
    folds: List[WalkForwardFoldResult] = field(default_factory=list)
    aggregate_test_metrics: BacktestMetricsResult | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "folds": [fold.to_dict() for fold in self.folds],
            "aggregate_test_metrics": (
                self.aggregate_test_metrics.to_dict()
                if self.aggregate_test_metrics is not None
                else {}
            ),
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


def run_walkforward_validation(
    pnls: List[float],
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> WalkForwardRunnerResult:
    clean_pnls = _validate_numeric_list(pnls)

    if train_size <= 0:
        raise ValueError("train_size must be > 0")
    if test_size <= 0:
        raise ValueError("test_size must be > 0")

    if step_size is None:
        step_size = test_size

    if step_size <= 0:
        raise ValueError("step_size must be > 0")

    total_length = len(clean_pnls)
    folds: List[WalkForwardFoldResult] = []
    aggregate_test_pnls: List[float] = []

    start = 0
    fold_index = 0

    while True:
        train_start = start
        train_end = train_start + train_size
        test_start = train_end
        test_end = test_start + test_size

        if test_end > total_length:
            break

        train_slice = clean_pnls[train_start:train_end]
        test_slice = clean_pnls[test_start:test_end]

        train_metrics = compute_backtest_metrics(train_slice)
        test_metrics = compute_backtest_metrics(test_slice)

        folds.append(
            WalkForwardFoldResult(
                fold_index=fold_index,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            )
        )

        aggregate_test_pnls.extend(test_slice)

        fold_index += 1
        start += step_size

    aggregate_test_metrics = compute_backtest_metrics(aggregate_test_pnls)

    return WalkForwardRunnerResult(
        folds=folds,
        aggregate_test_metrics=aggregate_test_metrics,
    )