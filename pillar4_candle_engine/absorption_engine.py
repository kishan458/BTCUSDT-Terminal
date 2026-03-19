from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AbsorptionConfig:
    failed_extension_window: int = 5
    overlap_support_threshold: float = 0.45
    weak_progress_threshold: float = 0.30
    strong_wick_threshold: float = 0.30
    confidence_margin_floor: float = 0.05

    def validate(self) -> None:
        values = {
            "failed_extension_window": self.failed_extension_window,
            "overlap_support_threshold": self.overlap_support_threshold,
            "weak_progress_threshold": self.weak_progress_threshold,
            "strong_wick_threshold": self.strong_wick_threshold,
            "confidence_margin_floor": self.confidence_margin_floor,
        }
        for name, value in values.items():
            if name == "failed_extension_window":
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer, got {value!r}.")
            else:
                if not isinstance(value, (int, float)):
                    raise TypeError(f"{name} must be numeric, got {type(value).__name__}.")
                if value < 0:
                    raise ValueError(f"{name} must be non-negative, got {value}.")


REQUIRED_COLUMNS = [
    "upper_wick_to_range_ratio",
    "lower_wick_to_range_ratio",
    "close_location_value",
    "body_to_range_ratio",
    "wick_imbalance",
    "overlap_ratio_vs_prev_bar",
    "avg_overlap_ratio_short",
    "progress_efficiency_short",
    "progress_efficiency_medium",
    "high_extension_vs_prev_high",
    "low_extension_vs_prev_low",
    "higher_high_flag",
    "lower_low_flag",
    "range_expansion_score",
    "rolling_wick_dominance",
    "direction",
]


def build_absorption_context(
    feature_df: pd.DataFrame,
    config: AbsorptionConfig = AbsorptionConfig(),
) -> pd.DataFrame:
    config.validate()
    _validate_feature_columns(feature_df)

    out = feature_df.copy(deep=True)
    absorb_df = pd.DataFrame(index=out.index)

    upper_wick_ratio = out["upper_wick_to_range_ratio"]
    lower_wick_ratio = out["lower_wick_to_range_ratio"]
    close_location = out["close_location_value"]
    body_ratio = out["body_to_range_ratio"]
    wick_imbalance = out["wick_imbalance"]
    overlap_ratio = out["overlap_ratio_vs_prev_bar"]
    avg_overlap_short = out["avg_overlap_ratio_short"]
    progress_short = out["progress_efficiency_short"]
    progress_medium = out["progress_efficiency_medium"]
    high_extension = out["high_extension_vs_prev_high"]
    low_extension = out["low_extension_vs_prev_low"]
    higher_high_flag = out["higher_high_flag"]
    lower_low_flag = out["lower_low_flag"]
    range_expansion = out["range_expansion_score"]
    rolling_wick_dom = out["rolling_wick_dominance"]
    direction = out["direction"]

    upside_probe_flag = ((high_extension > 0) | (higher_high_flag == 1)).astype(int)
    downside_probe_flag = ((low_extension > 0) | (lower_low_flag == 1)).astype(int)

    weak_upside_acceptance = (
        (close_location < 0.55)
        & (progress_short < config.weak_progress_threshold)
    ).astype(int)

    weak_downside_acceptance = (
        (close_location > 0.45)
        & (progress_short < config.weak_progress_threshold)
    ).astype(int)

    failed_upside_extension_flag = (upside_probe_flag * weak_upside_acceptance).astype(int)
    failed_downside_extension_flag = (downside_probe_flag * weak_downside_acceptance).astype(int)

    absorb_df["failed_upside_extension_flag"] = failed_upside_extension_flag
    absorb_df["failed_downside_extension_flag"] = failed_downside_extension_flag

    absorb_df["failed_upside_extension_count"] = failed_upside_extension_flag.rolling(
        window=config.failed_extension_window,
        min_periods=1,
    ).sum()

    absorb_df["failed_downside_extension_count"] = failed_downside_extension_flag.rolling(
        window=config.failed_extension_window,
        min_periods=1,
    ).sum()

    absorb_df["buy_rejection_score"] = _mean_score(
        [
            _scaled_minmax(lower_wick_ratio, config.strong_wick_threshold, 1.0),
            _scaled_minmax(close_location, 0.50, 1.0),
            _scaled_inverse(body_ratio, 0.0, 0.50),
            _scaled_minmax(wick_imbalance, 0.0, 1.0),
            _scaled_minmax(-direction, 0.0, 1.0),
        ]
    )

    absorb_df["sell_rejection_score"] = _mean_score(
        [
            _scaled_minmax(upper_wick_ratio, config.strong_wick_threshold, 1.0),
            _scaled_minmax(1.0 - close_location, 0.50, 1.0),
            _scaled_inverse(body_ratio, 0.0, 0.50),
            _scaled_minmax(-wick_imbalance, 0.0, 1.0),
            _scaled_minmax(direction, 0.0, 1.0),
        ]
    )

    absorb_df["buy_absorption_score"] = _mean_score(
        [
            _scaled_minmax(absorb_df["failed_downside_extension_count"], 1.0, float(config.failed_extension_window)),
            _scaled_minmax(lower_wick_ratio, 0.15, 1.0),
            _scaled_minmax(close_location, 0.45, 1.0),
            _scaled_minmax(avg_overlap_short, config.overlap_support_threshold, 1.0),
            _scaled_inverse(progress_short, 0.0, 0.50),
            _scaled_minmax(rolling_wick_dom, 0.0, 1.0),
        ]
    )

    absorb_df["sell_absorption_score"] = _mean_score(
        [
            _scaled_minmax(absorb_df["failed_upside_extension_count"], 1.0, float(config.failed_extension_window)),
            _scaled_minmax(upper_wick_ratio, 0.15, 1.0),
            _scaled_minmax(1.0 - close_location, 0.45, 1.0),
            _scaled_minmax(avg_overlap_short, config.overlap_support_threshold, 1.0),
            _scaled_inverse(progress_short, 0.0, 0.50),
            _scaled_minmax(-rolling_wick_dom, 0.0, 1.0),
        ]
    )

    dominant_absorption = np.where(
        absorb_df["buy_absorption_score"] > absorb_df["sell_absorption_score"],
        "BUY_ABSORPTION",
        "SELL_ABSORPTION",
    )

    dominant_rejection = np.where(
        absorb_df["buy_rejection_score"] > absorb_df["sell_rejection_score"],
        "BUY_REJECTION",
        "SELL_REJECTION",
    )

    absorb_df["dominant_absorption"] = dominant_absorption
    absorb_df["dominant_rejection"] = dominant_rejection

    top_absorption_score = np.maximum(
        absorb_df["buy_absorption_score"],
        absorb_df["sell_absorption_score"],
    )
    second_absorption_score = np.minimum(
        absorb_df["buy_absorption_score"],
        absorb_df["sell_absorption_score"],
    )

    absorb_df["absorption_confidence"] = _safe_confidence(
        top_absorption_score.to_numpy(dtype=float),
        second_absorption_score.to_numpy(dtype=float),
        config.confidence_margin_floor,
    )

    return pd.concat([out, absorb_df], axis=1)


def latest_absorption_snapshot(absorption_df: pd.DataFrame) -> Dict[str, object]:
    if absorption_df.empty:
        raise ValueError("absorption_df is empty.")

    latest = absorption_df.iloc[-1]
    return {
        "buy_absorption_score": float(latest["buy_absorption_score"]),
        "sell_absorption_score": float(latest["sell_absorption_score"]),
        "buy_rejection_score": float(latest["buy_rejection_score"]),
        "sell_rejection_score": float(latest["sell_rejection_score"]),
        "dominant_absorption": latest["dominant_absorption"],
        "dominant_rejection": latest["dominant_rejection"],
        "absorption_confidence": float(latest["absorption_confidence"]),
        "failed_upside_extension_count": float(latest["failed_upside_extension_count"]),
        "failed_downside_extension_count": float(latest["failed_downside_extension_count"]),
    }


def _validate_feature_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns for absorption engine: {missing}")
    if df.empty:
        raise ValueError("feature_df is empty.")


def _scaled_minmax(series: pd.Series, low: float, high: float) -> pd.Series:
    if high <= low:
        raise ValueError("high must be greater than low in _scaled_minmax.")
    values = (series - low) / (high - low)
    return values.clip(lower=0.0, upper=1.0).fillna(0.0)


def _scaled_inverse(series: pd.Series, low: float, high: float) -> pd.Series:
    return 1.0 - _scaled_minmax(series, low, high)


def _mean_score(parts: list[pd.Series]) -> pd.Series:
    stacked = pd.concat(parts, axis=1).fillna(0.0)
    return stacked.mean(axis=1).clip(lower=0.0, upper=1.0)


def _safe_confidence(top: np.ndarray, second: np.ndarray, floor: float) -> np.ndarray:
    numerator = np.maximum(top - second, 0.0)
    denominator = np.maximum(top, floor)
    confidence = numerator / denominator
    return np.clip(confidence, 0.0, 1.0)