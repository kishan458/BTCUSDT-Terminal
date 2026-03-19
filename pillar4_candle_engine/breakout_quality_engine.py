from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BreakoutQualityConfig:
    range_window: int = 5
    acceptance_close_threshold: float = 0.55
    strong_breakout_threshold: float = 0.65
    weak_breakout_threshold: float = 0.40
    fake_breakout_overlap_threshold: float = 0.65
    minimum_breach_threshold: float = 0.0

    def validate(self) -> None:
        values = {
            "range_window": self.range_window,
            "acceptance_close_threshold": self.acceptance_close_threshold,
            "strong_breakout_threshold": self.strong_breakout_threshold,
            "weak_breakout_threshold": self.weak_breakout_threshold,
            "fake_breakout_overlap_threshold": self.fake_breakout_overlap_threshold,
            "minimum_breach_threshold": self.minimum_breach_threshold,
        }
        for name, value in values.items():
            if name == "range_window":
                if not isinstance(value, int) or value <= 1:
                    raise ValueError(f"{name} must be an integer greater than 1, got {value!r}.")
            else:
                if not isinstance(value, (int, float)):
                    raise TypeError(f"{name} must be numeric, got {type(value).__name__}.")
                if value < 0:
                    raise ValueError(f"{name} must be non-negative, got {value}.")


REQUIRED_COLUMNS = [
    "high",
    "low",
    "close",
    "full_range",
    "close_location_value",
    "body_to_range_ratio",
    "range_expansion_score",
    "atr_scaled_range",
    "overlap_ratio_vs_prev_bar",
    "avg_overlap_ratio_short",
    "progress_efficiency_short",
    "progress_efficiency_medium",
    "inside_bar_flag",
    "outside_bar_flag",
    "high_extension_vs_prev_high",
    "low_extension_vs_prev_low",
]


def build_breakout_quality_context(
    feature_df: pd.DataFrame,
    config: BreakoutQualityConfig = BreakoutQualityConfig(),
) -> pd.DataFrame:
    config.validate()
    _validate_feature_columns(feature_df)

    out = feature_df.copy(deep=True)
    breakout_df = pd.DataFrame(index=out.index)

    high = out["high"].astype(float)
    low = out["low"].astype(float)
    close = out["close"].astype(float)
    full_range = out["full_range"].astype(float)
    close_location = out["close_location_value"].astype(float)
    body_ratio = out["body_to_range_ratio"].astype(float)
    range_expansion = out["range_expansion_score"].astype(float)
    atr_scaled_range = out["atr_scaled_range"].astype(float)
    overlap_ratio = out["overlap_ratio_vs_prev_bar"].astype(float)
    avg_overlap_short = out["avg_overlap_ratio_short"].astype(float)
    progress_short = out["progress_efficiency_short"].astype(float)
    progress_medium = out["progress_efficiency_medium"].astype(float)
    outside_bar_flag = out["outside_bar_flag"].astype(float)

    reference_range_high = high.shift(1).rolling(
        window=config.range_window,
        min_periods=1,
    ).max()
    reference_range_low = low.shift(1).rolling(
        window=config.range_window,
        min_periods=1,
    ).min()

    breakout_df["reference_range_high"] = reference_range_high
    breakout_df["reference_range_low"] = reference_range_low

    upside_breach = (high - reference_range_high).clip(lower=0.0)
    downside_breach = (reference_range_low - low).clip(lower=0.0)

    breakout_direction = np.full(len(out), "NONE", dtype=object)
    breakout_direction[upside_breach > np.maximum(downside_breach, config.minimum_breach_threshold)] = "UPSIDE"
    breakout_direction[downside_breach > np.maximum(upside_breach, config.minimum_breach_threshold)] = "DOWNSIDE"

    breakout_direction_series = pd.Series(breakout_direction, index=out.index)

    breakout_df["breakout_direction"] = breakout_direction
    breakout_df["upside_breach_magnitude"] = upside_breach
    breakout_df["downside_breach_magnitude"] = downside_breach
    breakout_df["breach_magnitude"] = np.maximum(upside_breach, downside_breach)

    has_breakout = breakout_df["breach_magnitude"] > config.minimum_breach_threshold
    has_breakout_float = has_breakout.astype(float)

    reference_width = (reference_range_high - reference_range_low).clip(lower=0.0)
    breakout_df["reference_range_width"] = reference_width

    close_outside_up = (close - reference_range_high).clip(lower=0.0)
    close_outside_down = (reference_range_low - close).clip(lower=0.0)
    close_outside_amount = np.where(
        breakout_direction == "UPSIDE",
        close_outside_up,
        np.where(breakout_direction == "DOWNSIDE", close_outside_down, 0.0),
    )
    close_outside_amount = pd.Series(close_outside_amount, index=out.index, dtype=float)

    breakout_df["close_outside_amount"] = close_outside_amount
    breakout_df["close_outside_range_ratio"] = _safe_divide(close_outside_amount, full_range)

    retrace_ratio = 1.0 - breakout_df["close_outside_range_ratio"]
    breakout_df["retrace_ratio"] = retrace_ratio.clip(lower=0.0, upper=1.0)

    wick_penalty = _mean_score(
        [
            _scaled_inverse(body_ratio, 0.0, 0.60),
            _scaled_minmax(overlap_ratio, 0.25, 1.0),
            _scaled_inverse(close_outside_amount, 0.0, np.maximum(breakout_df["breach_magnitude"], 1e-12)),
        ]
    )
    breakout_df["wick_penalty"] = wick_penalty

    upside_acceptance_score = _mean_score(
        [
            _binary_gate(breakout_direction_series == "UPSIDE"),
            _scaled_minmax(breakout_df["close_outside_range_ratio"], 0.10, 1.0),
            _scaled_minmax(close_location, config.acceptance_close_threshold, 1.0),
            _scaled_minmax(progress_short, 0.20, 1.0),
            _scaled_inverse(avg_overlap_short, 0.0, 0.70),
            _scaled_minmax(range_expansion, 1.0, 2.5),
        ]
    )

    downside_acceptance_score = _mean_score(
        [
            _binary_gate(breakout_direction_series == "DOWNSIDE"),
            _scaled_minmax(breakout_df["close_outside_range_ratio"], 0.10, 1.0),
            _scaled_minmax(1.0 - close_location, config.acceptance_close_threshold, 1.0),
            _scaled_minmax(progress_short, 0.20, 1.0),
            _scaled_inverse(avg_overlap_short, 0.0, 0.70),
            _scaled_minmax(range_expansion, 1.0, 2.5),
        ]
    )

    acceptance_score = pd.Series(
        np.where(
            breakout_direction == "UPSIDE",
            upside_acceptance_score,
            np.where(breakout_direction == "DOWNSIDE", downside_acceptance_score, 0.0),
        ),
        index=out.index,
        dtype=float,
    )
    breakout_df["acceptance_score"] = acceptance_score

    raw_failure_score = _mean_score(
        [
            has_breakout_float,
            _scaled_inverse(breakout_df["close_outside_range_ratio"], 0.0, 0.60),
            _scaled_minmax(avg_overlap_short, 0.30, 1.0),
            _scaled_inverse(progress_short, 0.0, 0.50),
            _scaled_minmax(wick_penalty, 0.20, 1.0),
        ]
    )
    breakout_df["failure_score"] = raw_failure_score * has_breakout_float

    raw_fake_breakout_risk = _mean_score(
        [
            breakout_df["failure_score"],
            _scaled_minmax(avg_overlap_short, config.fake_breakout_overlap_threshold, 1.0),
            _scaled_inverse(progress_medium, 0.0, 0.50),
            _scaled_inverse(breakout_df["close_outside_range_ratio"], 0.0, 0.50),
            _scaled_minmax(outside_bar_flag, 0.0, 1.0),
        ]
    )
    breakout_df["fake_breakout_risk"] = raw_fake_breakout_risk * has_breakout_float

    breakout_quality_score = _mean_score(
        [
            acceptance_score,
            _scaled_minmax(body_ratio, 0.25, 1.0),
            _scaled_minmax(atr_scaled_range, 0.80, 2.50),
            _scaled_minmax(range_expansion, 1.0, 2.5),
            _scaled_inverse(wick_penalty, 0.0, 1.0),
        ]
    ) * has_breakout_float
    breakout_df["breakout_quality_score"] = breakout_quality_score

    breakout_validity = np.full(len(out), "NO_BREAKOUT", dtype=object)
    breakout_state = np.full(len(out), "NONE", dtype=object)

    weak_mask = has_breakout & (breakout_quality_score >= config.weak_breakout_threshold)
    strong_mask = has_breakout & (acceptance_score >= config.strong_breakout_threshold)
    failing_mask = has_breakout & (breakout_df["failure_score"] >= 0.55)
    early_mask = has_breakout & ~strong_mask & ~failing_mask
    liquidity_grab_mask = has_breakout & (breakout_df["fake_breakout_risk"] >= 0.60)

    breakout_validity[weak_mask] = "UNCONFIRMED"
    breakout_validity[strong_mask] = "CONFIRMED"
    breakout_validity[failing_mask] = "FAILING"

    breakout_state[early_mask] = "ACCEPTED_BUT_EARLY"
    breakout_state[strong_mask] = "CLEAN"
    breakout_state[failing_mask] = "FAILING"
    breakout_state[liquidity_grab_mask] = "LIQUIDITY_GRAB_LIKE"

    breakout_df["breakout_validity"] = breakout_validity
    breakout_df["breakout_state"] = breakout_state

    return pd.concat([out, breakout_df], axis=1)


def latest_breakout_snapshot(breakout_df: pd.DataFrame) -> Dict[str, object]:
    if breakout_df.empty:
        raise ValueError("breakout_df is empty.")

    latest = breakout_df.iloc[-1]
    return {
        "reference_range_high": float(latest["reference_range_high"]) if pd.notna(latest["reference_range_high"]) else None,
        "reference_range_low": float(latest["reference_range_low"]) if pd.notna(latest["reference_range_low"]) else None,
        "breakout_direction": latest["breakout_direction"],
        "breach_magnitude": float(latest["breach_magnitude"]),
        "close_outside_range_ratio": float(latest["close_outside_range_ratio"]),
        "acceptance_score": float(latest["acceptance_score"]),
        "failure_score": float(latest["failure_score"]),
        "wick_penalty": float(latest["wick_penalty"]),
        "fake_breakout_risk": float(latest["fake_breakout_risk"]),
        "breakout_quality_score": float(latest["breakout_quality_score"]),
        "breakout_validity": latest["breakout_validity"],
        "breakout_state": latest["breakout_state"],
    }


def _validate_feature_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns for breakout quality engine: {missing}")
    if df.empty:
        raise ValueError("feature_df is empty.")


def _binary_gate(condition: pd.Series) -> pd.Series:
    return condition.astype(float)


def _scaled_minmax(series: pd.Series, low: float, high) -> pd.Series:
    if isinstance(high, pd.Series):
        denom = (high - low).replace(0.0, np.nan)
        values = (series - low) / denom
        return values.clip(lower=0.0, upper=1.0).fillna(0.0)

    if high <= low:
        raise ValueError("high must be greater than low in _scaled_minmax.")
    values = (series - low) / (high - low)
    return values.clip(lower=0.0, upper=1.0).fillna(0.0)


def _scaled_inverse(series: pd.Series, low: float, high) -> pd.Series:
    return 1.0 - _scaled_minmax(series, low, high)


def _mean_score(parts: list[pd.Series]) -> pd.Series:
    stacked = pd.concat(parts, axis=1).fillna(0.0)
    return stacked.mean(axis=1).clip(lower=0.0, upper=1.0)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator_safe = denominator.replace(0.0, np.nan)
    return (numerator / denominator_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)