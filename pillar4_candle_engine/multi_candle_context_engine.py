from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MultiCandleContextConfig:
    momentum_accelerating_threshold: float = 0.55
    momentum_building_threshold: float = 0.40
    momentum_stalling_threshold: float = 0.22

    buyer_control_threshold: float = 0.58
    seller_control_threshold: float = 0.42

    expansion_expanding_threshold: float = 1.20
    expansion_compressed_threshold: float = 0.85
    post_expansion_fade_threshold: float = 0.45

    overlap_low_threshold: float = 0.35
    overlap_high_threshold: float = 0.70

    follow_through_strong_threshold: float = 0.60
    follow_through_moderate_threshold: float = 0.35
    follow_through_failed_threshold: float = 0.15

    exhaustion_risk_threshold: float = 0.55
    exhaustion_confirmed_threshold: float = 0.72

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric, got {type(value).__name__}.")
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")


REQUIRED_COLUMNS = [
    "direction",
    "close_location_value",
    "body_to_range_ratio",
    "wick_imbalance",
    "range_expansion_score",
    "body_expansion_score",
    "overlap_ratio_vs_prev_bar",
    "avg_overlap_ratio_short",
    "avg_overlap_ratio_long",
    "progress_efficiency_short",
    "progress_efficiency_medium",
    "progress_efficiency_long",
    "net_progress_short",
    "net_progress_medium",
    "rolling_body_dominance_short",
    "rolling_body_dominance_medium",
    "rolling_body_dominance_long",
    "rolling_wick_dominance",
    "rolling_sign_consistency_short",
    "rolling_sign_consistency_medium",
    "close_upper_half_count_short",
    "close_upper_half_count_medium",
    "close_lower_half_count_short",
    "close_lower_half_count_medium",
    "inside_bar_frequency",
    "outside_bar_frequency",
    "range_contraction_streak",
    "expansion_after_compression_score",
    "post_expansion_fade_score",
    "atr_scaled_range",
    "atr_scaled_body",
]


def build_multi_candle_context(
    feature_df: pd.DataFrame,
    config: MultiCandleContextConfig = MultiCandleContextConfig(),
) -> pd.DataFrame:
    config.validate()
    _validate_feature_columns(feature_df)

    out = feature_df.copy(deep=True)
    context_df = pd.DataFrame(index=out.index)

    direction = out["direction"]
    close_location = out["close_location_value"]
    body_ratio = out["body_to_range_ratio"]
    wick_imbalance = out["wick_imbalance"]
    range_expansion = out["range_expansion_score"]
    body_expansion = out["body_expansion_score"]
    avg_overlap_short = out["avg_overlap_ratio_short"]
    avg_overlap_long = out["avg_overlap_ratio_long"]
    progress_short = out["progress_efficiency_short"]
    progress_medium = out["progress_efficiency_medium"]
    progress_long = out["progress_efficiency_long"]
    net_progress_short = out["net_progress_short"]
    net_progress_medium = out["net_progress_medium"]
    rolling_body_dom_short = out["rolling_body_dominance_short"]
    rolling_body_dom_medium = out["rolling_body_dominance_medium"]
    rolling_body_dom_long = out["rolling_body_dominance_long"]
    rolling_wick_dom = out["rolling_wick_dominance"]
    sign_consistency_short = out["rolling_sign_consistency_short"]
    sign_consistency_medium = out["rolling_sign_consistency_medium"]
    close_upper_short = out["close_upper_half_count_short"]
    close_upper_medium = out["close_upper_half_count_medium"]
    close_lower_short = out["close_lower_half_count_short"]
    close_lower_medium = out["close_lower_half_count_medium"]
    inside_bar_frequency = out["inside_bar_frequency"]
    outside_bar_frequency = out["outside_bar_frequency"]
    range_contraction_streak = out["range_contraction_streak"]
    expansion_after_compression = out["expansion_after_compression_score"]
    post_expansion_fade = out["post_expansion_fade_score"]
    atr_scaled_range = out["atr_scaled_range"]
    atr_scaled_body = out["atr_scaled_body"]

    # ------------------------------------------------------------
    # Helper rolling summaries
    # ------------------------------------------------------------
    context_df["directional_bias_score"] = _directional_bias_score(
        close_location=close_location,
        wick_imbalance=wick_imbalance,
        direction=direction,
        rolling_wick_dom=rolling_wick_dom,
    )

    context_df["momentum_continuation_score"] = _mean_score(
        [
            _scaled_minmax(progress_short, 0.15, 1.0),
            _scaled_minmax(progress_medium, 0.15, 1.0),
            _scaled_minmax(sign_consistency_short, 0.50, 1.0),
            _scaled_minmax(sign_consistency_medium, 0.50, 1.0),
            _scaled_minmax(rolling_body_dom_short, 0.30, 1.0),
            _scaled_minmax(rolling_body_dom_medium, 0.30, 1.0),
        ]
    )

    context_df["momentum_acceleration_score"] = _mean_score(
        [
            _scaled_minmax(progress_short - progress_medium, 0.00, 0.40),
            _scaled_minmax(range_expansion, 1.00, 2.50),
            _scaled_minmax(body_expansion, 1.00, 2.50),
            _scaled_inverse(avg_overlap_short, 0.0, 0.60),
        ]
    )

    context_df["stalling_score"] = _mean_score(
        [
            _scaled_inverse(progress_short, 0.0, 0.35),
            _scaled_minmax(avg_overlap_short, 0.35, 1.0),
            _scaled_inverse(rolling_body_dom_short, 0.0, 0.45),
            _scaled_inverse(np.abs(context_df["directional_bias_score"]), 0.0, 0.35),
        ]
    )

    context_df["reversal_risk_score"] = _mean_score(
        [
            _scaled_minmax(avg_overlap_short, 0.45, 1.0),
            _scaled_inverse(progress_short, 0.0, 0.35),
            _scaled_minmax(np.abs(rolling_wick_dom), 0.10, 1.0),
            _scaled_inverse(sign_consistency_short, 0.0, 0.60),
        ]
    )

    # ------------------------------------------------------------
    # Momentum state
    # ------------------------------------------------------------
    momentum_state = np.full(len(out), "STABLE", dtype=object)

    accelerating_mask = (
        (context_df["momentum_acceleration_score"] >= config.momentum_accelerating_threshold)
        & (context_df["momentum_continuation_score"] >= config.momentum_building_threshold)
    )
    building_mask = (
        (context_df["momentum_continuation_score"] >= config.momentum_building_threshold)
        & ~accelerating_mask
    )
    stalling_mask = context_df["stalling_score"] >= config.momentum_stalling_threshold
    reversal_risk_mask = context_df["reversal_risk_score"] >= 0.60
    decaying_mask = (
        (progress_short < progress_medium)
        & (context_df["momentum_continuation_score"] < config.momentum_building_threshold)
        & (avg_overlap_short >= config.overlap_low_threshold)
    )

    momentum_state[accelerating_mask] = "ACCELERATING"
    momentum_state[building_mask] = "BUILDING"
    momentum_state[decaying_mask] = "DECAYING"
    momentum_state[stalling_mask] = "STALLING"
    momentum_state[reversal_risk_mask] = "REVERSAL_RISK"

    context_df["momentum_state"] = momentum_state

    # ------------------------------------------------------------
    # Control state
    # ------------------------------------------------------------
    buyer_control_score = _mean_score(
        [
            _scaled_minmax(close_location, 0.50, 1.0),
            _scaled_minmax(rolling_body_dom_medium, 0.30, 1.0),
            _scaled_minmax(sign_consistency_medium, 0.50, 1.0),
            _scaled_minmax(close_upper_medium / 5.0, 0.30, 1.0),
            _scaled_minmax(context_df["directional_bias_score"], 0.0, 1.0),
        ]
    )

    seller_control_score = _mean_score(
        [
            _scaled_minmax(1.0 - close_location, 0.50, 1.0),
            _scaled_minmax(rolling_body_dom_medium, 0.30, 1.0),
            _scaled_minmax(sign_consistency_medium, 0.50, 1.0),
            _scaled_minmax(close_lower_medium / 5.0, 0.30, 1.0),
            _scaled_minmax(-context_df["directional_bias_score"], 0.0, 1.0),
        ]
    )

    context_df["buyer_control_score"] = buyer_control_score
    context_df["seller_control_score"] = seller_control_score

    control_state = np.full(len(out), "BALANCED", dtype=object)
    control_state[buyer_control_score >= config.buyer_control_threshold] = "BUYERS_IN_CONTROL"
    control_state[seller_control_score >= config.seller_control_threshold] = "SELLERS_IN_CONTROL"

    two_way_mask = (
        (avg_overlap_short >= config.overlap_high_threshold)
        & (sign_consistency_short < 0.70)
        & (np.abs(context_df["directional_bias_score"]) < 0.25)
    )
    control_state[two_way_mask] = "TWO_WAY_AUCTION"

    both_high_mask = (
        (buyer_control_score >= config.buyer_control_threshold)
        & (seller_control_score >= config.seller_control_threshold)
    )
    control_state[both_high_mask] = "TWO_WAY_AUCTION"

    context_df["control_state"] = control_state

    # ------------------------------------------------------------
    # Expansion state
    # ------------------------------------------------------------
    expansion_state = np.full(len(out), "NORMAL", dtype=object)

    expanding_mask = (
        (range_expansion >= config.expansion_expanding_threshold)
        | (atr_scaled_range >= 1.20)
        | (expansion_after_compression >= 0.60)
    )

    compressed_mask = (
        (range_expansion <= config.expansion_compressed_threshold)
        & (atr_scaled_range <= 1.0)
        & (inside_bar_frequency >= 0.20)
    )

    fade_mask = (
        (post_expansion_fade >= config.post_expansion_fade_threshold)
        & ~compressed_mask
    )

    expansion_state[expanding_mask] = "EXPANDING"
    expansion_state[compressed_mask] = "COMPRESSED"
    expansion_state[fade_mask] = "POST_EXPANSION_FADE"

    context_df["expansion_state"] = expansion_state

    # ------------------------------------------------------------
    # Overlap state
    # ------------------------------------------------------------
    overlap_state = np.full(len(out), "MODERATE_OVERLAP", dtype=object)
    overlap_state[avg_overlap_short <= config.overlap_low_threshold] = "LOW_OVERLAP"
    overlap_state[avg_overlap_short >= config.overlap_high_threshold] = "HIGH_OVERLAP"
    context_df["overlap_state"] = overlap_state

    # ------------------------------------------------------------
    # Follow-through quality
    # ------------------------------------------------------------
    follow_through_score = _mean_score(
        [
            _scaled_minmax(progress_short, 0.10, 1.0),
            _scaled_minmax(sign_consistency_short, 0.50, 1.0),
            _scaled_inverse(avg_overlap_short, 0.0, 0.70),
            _scaled_minmax(rolling_body_dom_short, 0.25, 1.0),
            _scaled_minmax(np.abs(net_progress_short), 0.0, np.maximum(np.abs(net_progress_medium), 1e-12)),
        ]
    )

    context_df["follow_through_score"] = follow_through_score

    follow_through_quality = np.full(len(out), "WEAK", dtype=object)
    follow_through_quality[follow_through_score >= config.follow_through_strong_threshold] = "STRONG"
    moderate_mask = (
        (follow_through_score >= config.follow_through_moderate_threshold)
        & (follow_through_score < config.follow_through_strong_threshold)
    )
    follow_through_quality[moderate_mask] = "MODERATE"
    follow_through_quality[follow_through_score <= config.follow_through_failed_threshold] = "FAILED"

    context_df["follow_through_quality"] = follow_through_quality

    # ------------------------------------------------------------
    # Exhaustion state
    # ------------------------------------------------------------
    upside_exhaustion_score = _mean_score(
        [
            _binary_gate(direction > 0),
            _scaled_minmax(range_expansion, 1.0, 2.5),
            _scaled_inverse(progress_short, 0.0, 0.40),
            _scaled_minmax(avg_overlap_short, 0.35, 1.0),
            _scaled_minmax(-wick_imbalance, 0.0, 1.0),
        ]
    )

    downside_exhaustion_score = _mean_score(
        [
            _binary_gate(direction < 0),
            _scaled_minmax(range_expansion, 1.0, 2.5),
            _scaled_inverse(progress_short, 0.0, 0.40),
            _scaled_minmax(avg_overlap_short, 0.35, 1.0),
            _scaled_minmax(wick_imbalance, 0.0, 1.0),
        ]
    )

    context_df["upside_exhaustion_score"] = upside_exhaustion_score
    context_df["downside_exhaustion_score"] = downside_exhaustion_score

    exhaustion_state = np.full(len(out), "NONE", dtype=object)

    upside_risk_mask = upside_exhaustion_score >= config.exhaustion_risk_threshold
    downside_risk_mask = downside_exhaustion_score >= config.exhaustion_risk_threshold
    upside_confirmed_mask = upside_exhaustion_score >= config.exhaustion_confirmed_threshold
    downside_confirmed_mask = downside_exhaustion_score >= config.exhaustion_confirmed_threshold

    exhaustion_state[upside_risk_mask] = "UPSIDE_EXHAUSTION_RISK"
    exhaustion_state[downside_risk_mask] = "DOWNSIDE_EXHAUSTION_RISK"
    exhaustion_state[upside_confirmed_mask] = "CONFIRMED_UPSIDE_EXHAUSTION"
    exhaustion_state[downside_confirmed_mask] = "CONFIRMED_DOWNSIDE_EXHAUSTION"

    both_exhaustion_mask = upside_confirmed_mask & downside_confirmed_mask
    exhaustion_state[both_exhaustion_mask] = "NONE"

    context_df["exhaustion_state"] = exhaustion_state

    return pd.concat([out, context_df], axis=1)


def latest_multi_candle_context_snapshot(context_df: pd.DataFrame) -> Dict[str, object]:
    if context_df.empty:
        raise ValueError("context_df is empty.")

    latest = context_df.iloc[-1]
    return {
        "momentum_state": latest["momentum_state"],
        "control_state": latest["control_state"],
        "expansion_state": latest["expansion_state"],
        "overlap_state": latest["overlap_state"],
        "follow_through_quality": latest["follow_through_quality"],
        "exhaustion_state": latest["exhaustion_state"],
        "momentum_continuation_score": float(latest["momentum_continuation_score"]),
        "momentum_acceleration_score": float(latest["momentum_acceleration_score"]),
        "stalling_score": float(latest["stalling_score"]),
        "reversal_risk_score": float(latest["reversal_risk_score"]),
        "buyer_control_score": float(latest["buyer_control_score"]),
        "seller_control_score": float(latest["seller_control_score"]),
        "follow_through_score": float(latest["follow_through_score"]),
        "upside_exhaustion_score": float(latest["upside_exhaustion_score"]),
        "downside_exhaustion_score": float(latest["downside_exhaustion_score"]),
    }


def _validate_feature_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns for multi candle context engine: {missing}")
    if df.empty:
        raise ValueError("feature_df is empty.")


def _binary_gate(condition: pd.Series) -> pd.Series:
    return condition.astype(float)


def _scaled_minmax(series: pd.Series, low: float, high: float) -> pd.Series:
    if isinstance(high, pd.Series):
        denom = (high - low).replace(0.0, np.nan)
        values = (series - low) / denom
        return values.clip(lower=0.0, upper=1.0).fillna(0.0)

    if high <= low:
        raise ValueError("high must be greater than low in _scaled_minmax.")
    values = (series - low) / (high - low)
    return values.clip(lower=0.0, upper=1.0).fillna(0.0)


def _scaled_inverse(series: pd.Series, low: float, high: float) -> pd.Series:
    return 1.0 - _scaled_minmax(series, low, high)


def _mean_score(parts: list[pd.Series]) -> pd.Series:
    stacked = pd.concat(parts, axis=1).fillna(0.0)
    return stacked.mean(axis=1).clip(lower=0.0, upper=1.0)


def _directional_bias_score(
    close_location: pd.Series,
    wick_imbalance: pd.Series,
    direction: pd.Series,
    rolling_wick_dom: pd.Series,
) -> pd.Series:
    close_component = (close_location - 0.5) * 2.0
    direction_component = direction.astype(float) * 0.25
    wick_component = wick_imbalance * 0.75
    rolling_wick_component = rolling_wick_dom * 0.75

    score = (
        close_component * 0.40
        + direction_component * 0.15
        + wick_component * 0.20
        + rolling_wick_component * 0.25
    )
    return score.clip(lower=-1.0, upper=1.0)