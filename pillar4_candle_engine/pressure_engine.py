from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PressureConfig:
    strong_pressure_threshold: float = 0.65
    moderate_pressure_threshold: float = 0.52
    weak_pressure_threshold: float = 0.40
    bias_neutral_band: float = 0.08

    def validate(self) -> None:
        values = {
            "strong_pressure_threshold": self.strong_pressure_threshold,
            "moderate_pressure_threshold": self.moderate_pressure_threshold,
            "weak_pressure_threshold": self.weak_pressure_threshold,
            "bias_neutral_band": self.bias_neutral_band,
        }
        for name, value in values.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric, got {type(value).__name__}.")
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")


REQUIRED_COLUMNS = [
    "direction",
    "close_location_value",
    "body_to_range_ratio",
    "bar_efficiency",
    "wick_imbalance",
    "atr_scaled_body",
    "atr_scaled_range",
    "progress_efficiency_short",
    "progress_efficiency_medium",
    "rolling_body_dominance_short",
    "rolling_body_dominance_medium",
    "rolling_wick_dominance",
    "rolling_sign_consistency_short",
    "rolling_sign_consistency_medium",
    "close_upper_half_count_short",
    "close_upper_half_count_medium",
    "close_lower_half_count_short",
    "close_lower_half_count_medium",
    "avg_overlap_ratio_short",
    "buyer_control_score",
    "seller_control_score",
    "follow_through_score",
    "acceptance_score",
    "failure_score",
]


def build_pressure_context(
    feature_df: pd.DataFrame,
    config: PressureConfig = PressureConfig(),
) -> pd.DataFrame:
    config.validate()
    _validate_feature_columns(feature_df)

    out = feature_df.copy(deep=True)
    pressure_df = pd.DataFrame(index=out.index)

    direction = out["direction"].astype(float)
    close_location = out["close_location_value"].astype(float)
    body_ratio = out["body_to_range_ratio"].astype(float)
    bar_efficiency = out["bar_efficiency"].astype(float)
    wick_imbalance = out["wick_imbalance"].astype(float)
    atr_scaled_body = out["atr_scaled_body"].astype(float)
    atr_scaled_range = out["atr_scaled_range"].astype(float)
    progress_short = out["progress_efficiency_short"].astype(float)
    progress_medium = out["progress_efficiency_medium"].astype(float)
    rolling_body_dom_short = out["rolling_body_dominance_short"].astype(float)
    rolling_body_dom_medium = out["rolling_body_dominance_medium"].astype(float)
    rolling_wick_dom = out["rolling_wick_dominance"].astype(float)
    sign_consistency_short = out["rolling_sign_consistency_short"].astype(float)
    sign_consistency_medium = out["rolling_sign_consistency_medium"].astype(float)
    close_upper_short = out["close_upper_half_count_short"].astype(float)
    close_upper_medium = out["close_upper_half_count_medium"].astype(float)
    close_lower_short = out["close_lower_half_count_short"].astype(float)
    close_lower_medium = out["close_lower_half_count_medium"].astype(float)
    avg_overlap_short = out["avg_overlap_ratio_short"].astype(float)
    buyer_control_score = out["buyer_control_score"].astype(float)
    seller_control_score = out["seller_control_score"].astype(float)
    follow_through_score = out["follow_through_score"].astype(float)
    acceptance_score = out["acceptance_score"].astype(float)
    failure_score = out["failure_score"].astype(float)

    bullish_impulse_score = _mean_score(
        [
            _scaled_minmax(direction, 0.0, 1.0),
            _scaled_minmax(close_location, 0.50, 1.0),
            _scaled_minmax(body_ratio, 0.25, 1.0),
            _scaled_minmax(bar_efficiency, 0.25, 1.0),
            _scaled_minmax(atr_scaled_body, 0.25, 2.50),
            _scaled_minmax(progress_short, 0.10, 1.0),
        ]
    )

    bearish_impulse_score = _mean_score(
        [
            _scaled_minmax(-direction, 0.0, 1.0),
            _scaled_minmax(1.0 - close_location, 0.50, 1.0),
            _scaled_minmax(body_ratio, 0.25, 1.0),
            _scaled_minmax(bar_efficiency, 0.25, 1.0),
            _scaled_minmax(atr_scaled_body, 0.25, 2.50),
            _scaled_minmax(progress_short, 0.10, 1.0),
        ]
    )

    sequence_buy_support = _mean_score(
        [
            _scaled_minmax(rolling_body_dom_short, 0.25, 1.0),
            _scaled_minmax(rolling_body_dom_medium, 0.25, 1.0),
            _scaled_minmax(sign_consistency_short, 0.50, 1.0),
            _scaled_minmax(sign_consistency_medium, 0.50, 1.0),
            _scaled_minmax(close_upper_short / 3.0, 0.30, 1.0),
            _scaled_minmax(close_upper_medium / 5.0, 0.30, 1.0),
        ]
    )

    sequence_sell_support = _mean_score(
        [
            _scaled_minmax(rolling_body_dom_short, 0.25, 1.0),
            _scaled_minmax(rolling_body_dom_medium, 0.25, 1.0),
            _scaled_minmax(sign_consistency_short, 0.50, 1.0),
            _scaled_minmax(sign_consistency_medium, 0.50, 1.0),
            _scaled_minmax(close_lower_short / 3.0, 0.30, 1.0),
            _scaled_minmax(close_lower_medium / 5.0, 0.30, 1.0),
        ]
    )

    structural_buy_support = _mean_score(
        [
            _scaled_minmax(wick_imbalance, 0.0, 1.0),
            _scaled_minmax(rolling_wick_dom, 0.0, 1.0),
            _scaled_inverse(avg_overlap_short, 0.0, 0.80),
            _scaled_minmax(buyer_control_score, 0.20, 1.0),
            _scaled_minmax(follow_through_score, 0.20, 1.0),
            _scaled_minmax(acceptance_score, 0.10, 1.0),
            _scaled_inverse(failure_score, 0.0, 1.0),
        ]
    )

    structural_sell_support = _mean_score(
        [
            _scaled_minmax(-wick_imbalance, 0.0, 1.0),
            _scaled_minmax(-rolling_wick_dom, 0.0, 1.0),
            _scaled_inverse(avg_overlap_short, 0.0, 0.80),
            _scaled_minmax(seller_control_score, 0.20, 1.0),
            _scaled_minmax(follow_through_score, 0.20, 1.0),
            _scaled_inverse(acceptance_score, 0.0, 1.0),
            _scaled_minmax(failure_score, 0.10, 1.0),
        ]
    )

    pressure_df["bullish_impulse_score"] = bullish_impulse_score
    pressure_df["bearish_impulse_score"] = bearish_impulse_score
    pressure_df["sequence_buy_support"] = sequence_buy_support
    pressure_df["sequence_sell_support"] = sequence_sell_support
    pressure_df["structural_buy_support"] = structural_buy_support
    pressure_df["structural_sell_support"] = structural_sell_support

    buying_pressure_score = _mean_score(
        [
            bullish_impulse_score,
            sequence_buy_support,
            structural_buy_support,
            _scaled_minmax(progress_medium, 0.10, 1.0),
            _scaled_minmax(atr_scaled_range, 0.40, 2.50),
        ]
    )

    selling_pressure_score = _mean_score(
        [
            bearish_impulse_score,
            sequence_sell_support,
            structural_sell_support,
            _scaled_minmax(progress_medium, 0.10, 1.0),
            _scaled_minmax(atr_scaled_range, 0.40, 2.50),
        ]
    )

    pressure_df["buying_pressure_score"] = buying_pressure_score.clip(0.0, 1.0)
    pressure_df["selling_pressure_score"] = selling_pressure_score.clip(0.0, 1.0)
    pressure_df["net_pressure_score"] = (
        pressure_df["buying_pressure_score"] - pressure_df["selling_pressure_score"]
    ).clip(-1.0, 1.0)

    pressure_bias = np.full(len(out), "NEUTRAL_PRESSURE", dtype=object)
    pressure_bias[pressure_df["net_pressure_score"] > config.bias_neutral_band] = "BUY_PRESSURE"
    pressure_bias[pressure_df["net_pressure_score"] < -config.bias_neutral_band] = "SELL_PRESSURE"
    pressure_df["pressure_bias"] = pressure_bias

    top_pressure = np.maximum(
        pressure_df["buying_pressure_score"],
        pressure_df["selling_pressure_score"],
    )

    pressure_strength = np.full(len(out), "WEAK", dtype=object)
    pressure_strength[top_pressure >= config.strong_pressure_threshold] = "STRONG"
    moderate_mask = (
        (top_pressure >= config.moderate_pressure_threshold)
        & (top_pressure < config.strong_pressure_threshold)
    )
    weak_moderate_mask = (
        (top_pressure >= config.weak_pressure_threshold)
        & (top_pressure < config.moderate_pressure_threshold)
    )
    pressure_strength[moderate_mask] = "MODERATE"
    pressure_strength[weak_moderate_mask] = "LIGHT"
    pressure_df["pressure_strength"] = pressure_strength

    return pd.concat([out, pressure_df], axis=1)


def latest_pressure_snapshot(pressure_df: pd.DataFrame) -> Dict[str, object]:
    if pressure_df.empty:
        raise ValueError("pressure_df is empty.")

    latest = pressure_df.iloc[-1]
    return {
        "buying_pressure_score": float(latest["buying_pressure_score"]),
        "selling_pressure_score": float(latest["selling_pressure_score"]),
        "net_pressure_score": float(latest["net_pressure_score"]),
        "pressure_bias": latest["pressure_bias"],
        "pressure_strength": latest["pressure_strength"],
        "bullish_impulse_score": float(latest["bullish_impulse_score"]),
        "bearish_impulse_score": float(latest["bearish_impulse_score"]),
        "sequence_buy_support": float(latest["sequence_buy_support"]),
        "sequence_sell_support": float(latest["sequence_sell_support"]),
        "structural_buy_support": float(latest["structural_buy_support"]),
        "structural_sell_support": float(latest["structural_sell_support"]),
    }


def _validate_feature_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns for pressure engine: {missing}")
    if df.empty:
        raise ValueError("feature_df is empty.")


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