from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandleIntentConfig:
    strong_body_threshold: float = 0.60
    medium_body_threshold: float = 0.40
    strong_close_high_threshold: float = 0.75
    strong_close_low_threshold: float = 0.25
    long_wick_threshold: float = 0.35
    small_body_threshold: float = 0.25
    expansion_threshold: float = 1.20
    compression_threshold: float = 0.80
    low_overlap_threshold: float = 0.35
    high_overlap_threshold: float = 0.70
    exhaustion_progress_threshold: float = 0.30

    def validate(self) -> None:
        values = {
            "strong_body_threshold": self.strong_body_threshold,
            "medium_body_threshold": self.medium_body_threshold,
            "strong_close_high_threshold": self.strong_close_high_threshold,
            "strong_close_low_threshold": self.strong_close_low_threshold,
            "long_wick_threshold": self.long_wick_threshold,
            "small_body_threshold": self.small_body_threshold,
            "expansion_threshold": self.expansion_threshold,
            "compression_threshold": self.compression_threshold,
            "low_overlap_threshold": self.low_overlap_threshold,
            "high_overlap_threshold": self.high_overlap_threshold,
            "exhaustion_progress_threshold": self.exhaustion_progress_threshold,
        }
        for name, value in values.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric, got {type(value).__name__}.")
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")


INTENT_LABELS: List[str] = [
    "STRONG_BULLISH_CONTINUATION",
    "WEAK_BULLISH_CONTINUATION",
    "STRONG_BEARISH_CONTINUATION",
    "WEAK_BEARISH_CONTINUATION",
    "BUY_REJECTION",
    "SELL_REJECTION",
    "BUY_ABSORPTION_CANDIDATE",
    "SELL_ABSORPTION_CANDIDATE",
    "INDECISION",
    "INSIDE_COMPRESSION",
    "OUTSIDE_EXPANSION",
    "EXHAUSTION_UP_CANDIDATE",
    "EXHAUSTION_DOWN_CANDIDATE",
]


LABEL_TO_SCORE_COLUMN: Dict[str, str] = {
    "STRONG_BULLISH_CONTINUATION": "intent_score_strong_bullish_continuation",
    "WEAK_BULLISH_CONTINUATION": "intent_score_weak_bullish_continuation",
    "STRONG_BEARISH_CONTINUATION": "intent_score_strong_bearish_continuation",
    "WEAK_BEARISH_CONTINUATION": "intent_score_weak_bearish_continuation",
    "BUY_REJECTION": "intent_score_buy_rejection",
    "SELL_REJECTION": "intent_score_sell_rejection",
    "BUY_ABSORPTION_CANDIDATE": "intent_score_buy_absorption_candidate",
    "SELL_ABSORPTION_CANDIDATE": "intent_score_sell_absorption_candidate",
    "INDECISION": "intent_score_indecision",
    "INSIDE_COMPRESSION": "intent_score_inside_compression",
    "OUTSIDE_EXPANSION": "intent_score_outside_expansion",
    "EXHAUSTION_UP_CANDIDATE": "intent_score_exhaustion_up_candidate",
    "EXHAUSTION_DOWN_CANDIDATE": "intent_score_exhaustion_down_candidate",
}


REQUIRED_COLUMNS = [
    "direction",
    "body_to_range_ratio",
    "upper_wick_to_range_ratio",
    "lower_wick_to_range_ratio",
    "total_wick_to_range_ratio",
    "wick_imbalance",
    "close_location_value",
    "open_location_value",
    "bar_efficiency",
    "atr_scaled_range",
    "atr_scaled_body",
    "range_expansion_score",
    "body_expansion_score",
    "inside_bar_flag",
    "outside_bar_flag",
    "overlap_ratio_vs_prev_bar",
    "progress_efficiency_short",
    "progress_efficiency_medium",
    "rolling_body_dominance_short",
    "rolling_wick_dominance",
    "rolling_sign_consistency_short",
    "close_upper_half_count_short",
    "close_lower_half_count_short",
]


def classify_candle_intents(
    feature_df: pd.DataFrame,
    config: CandleIntentConfig = CandleIntentConfig(),
) -> pd.DataFrame:
    config.validate()
    _validate_feature_columns(feature_df)

    out = feature_df.copy(deep=True)
    score_df = pd.DataFrame(index=out.index)

    direction = out["direction"]
    body_ratio = out["body_to_range_ratio"]
    upper_wick_ratio = out["upper_wick_to_range_ratio"]
    lower_wick_ratio = out["lower_wick_to_range_ratio"]
    total_wick_ratio = out["total_wick_to_range_ratio"]
    wick_imbalance = out["wick_imbalance"]
    close_location = out["close_location_value"]
    bar_efficiency = out["bar_efficiency"]
    atr_scaled_range = out["atr_scaled_range"]
    atr_scaled_body = out["atr_scaled_body"]
    range_expansion = out["range_expansion_score"]
    body_expansion = out["body_expansion_score"]
    inside_bar = out["inside_bar_flag"]
    outside_bar = out["outside_bar_flag"]
    overlap_ratio = out["overlap_ratio_vs_prev_bar"]
    progress_short = out["progress_efficiency_short"]
    progress_medium = out["progress_efficiency_medium"]
    rolling_body_dom = out["rolling_body_dominance_short"]
    rolling_wick_dom = out["rolling_wick_dominance"]
    sign_consistency = out["rolling_sign_consistency_short"]

    bullish_strength = _mean_score(
        [
            _binary_gate(direction > 0),
            _scaled_minmax(body_ratio, config.medium_body_threshold, 1.0),
            _scaled_minmax(close_location, 0.50, 1.0),
            _scaled_minmax(bar_efficiency, config.medium_body_threshold, 1.0),
            _scaled_minmax(atr_scaled_body, 0.50, 2.50),
            _scaled_minmax(range_expansion, 0.90, 2.50),
            _scaled_minmax(progress_short, 0.20, 1.0),
            _scaled_minmax(rolling_body_dom, 0.35, 1.0),
            _scaled_minmax(sign_consistency, 0.50, 1.0),
        ]
    )

    bearish_strength = _mean_score(
        [
            _binary_gate(direction < 0),
            _scaled_minmax(body_ratio, config.medium_body_threshold, 1.0),
            _scaled_minmax(1.0 - close_location, 0.50, 1.0),
            _scaled_minmax(bar_efficiency, config.medium_body_threshold, 1.0),
            _scaled_minmax(atr_scaled_body, 0.50, 2.50),
            _scaled_minmax(range_expansion, 0.90, 2.50),
            _scaled_minmax(progress_short, 0.20, 1.0),
            _scaled_minmax(rolling_body_dom, 0.35, 1.0),
            _scaled_minmax(sign_consistency, 0.50, 1.0),
        ]
    )

    score_df["intent_score_strong_bullish_continuation"] = _mean_score(
        [
            bullish_strength,
            _scaled_minmax(body_ratio, config.strong_body_threshold, 1.0),
            _scaled_minmax(close_location, config.strong_close_high_threshold, 1.0),
            _scaled_inverse(upper_wick_ratio, 0.0, 0.25),
            _scaled_minmax(atr_scaled_range, 0.80, 2.50),
            _scaled_minmax(body_expansion, 1.0, 2.5),
            _scaled_inverse(overlap_ratio, 0.0, config.low_overlap_threshold),
        ]
    )

    score_df["intent_score_weak_bullish_continuation"] = _mean_score(
        [
            bullish_strength,
            _scaled_minmax(body_ratio, 0.25, config.strong_body_threshold),
            _scaled_minmax(close_location, 0.55, 0.85),
            _scaled_minmax(atr_scaled_range, 0.50, 1.75),
            _scaled_minmax(progress_medium, 0.10, 0.60),
        ]
    )

    score_df["intent_score_strong_bearish_continuation"] = _mean_score(
        [
            bearish_strength,
            _scaled_minmax(body_ratio, config.strong_body_threshold, 1.0),
            _scaled_minmax(1.0 - close_location, 1.0 - config.strong_close_low_threshold, 1.0),
            _scaled_inverse(lower_wick_ratio, 0.0, 0.25),
            _scaled_minmax(atr_scaled_range, 0.80, 2.50),
            _scaled_minmax(body_expansion, 1.0, 2.5),
            _scaled_inverse(overlap_ratio, 0.0, config.low_overlap_threshold),
        ]
    )

    score_df["intent_score_weak_bearish_continuation"] = _mean_score(
        [
            bearish_strength,
            _scaled_minmax(body_ratio, 0.25, config.strong_body_threshold),
            _scaled_minmax(1.0 - close_location, 0.55, 0.85),
            _scaled_minmax(atr_scaled_range, 0.50, 1.75),
            _scaled_minmax(progress_medium, 0.10, 0.60),
        ]
    )

    score_df["intent_score_buy_rejection"] = _mean_score(
        [
            _scaled_minmax(lower_wick_ratio, config.long_wick_threshold, 1.0),
            _scaled_minmax(close_location, 0.45, 1.0),
            _scaled_inverse(body_ratio, 0.0, 0.45),
            _scaled_minmax(wick_imbalance, 0.10, 1.0),
            _scaled_minmax(rolling_wick_dom, 0.0, 1.0),
        ]
    )

    score_df["intent_score_sell_rejection"] = _mean_score(
        [
            _scaled_minmax(upper_wick_ratio, config.long_wick_threshold, 1.0),
            _scaled_minmax(1.0 - close_location, 0.45, 1.0),
            _scaled_inverse(body_ratio, 0.0, 0.45),
            _scaled_minmax(-wick_imbalance, 0.10, 1.0),
            _scaled_minmax(-rolling_wick_dom, 0.0, 1.0),
        ]
    )

    score_df["intent_score_buy_absorption_candidate"] = _mean_score(
        [
            _scaled_minmax(lower_wick_ratio, 0.20, 1.0),
            _scaled_minmax(close_location, 0.40, 1.0),
            _scaled_minmax(overlap_ratio, 0.30, 1.0),
            _scaled_inverse(progress_short, 0.0, 0.50),
            _scaled_inverse(body_ratio, 0.0, 0.50),
        ]
    )

    score_df["intent_score_sell_absorption_candidate"] = _mean_score(
        [
            _scaled_minmax(upper_wick_ratio, 0.20, 1.0),
            _scaled_minmax(1.0 - close_location, 0.40, 1.0),
            _scaled_minmax(overlap_ratio, 0.30, 1.0),
            _scaled_inverse(progress_short, 0.0, 0.50),
            _scaled_inverse(body_ratio, 0.0, 0.50),
        ]
    )

    score_df["intent_score_indecision"] = _mean_score(
        [
            _scaled_inverse(body_ratio, 0.0, config.small_body_threshold),
            _scaled_minmax(total_wick_ratio, 0.50, 1.0),
            _scaled_inverse(np.abs(close_location - 0.5), 0.0, 0.30),
            _scaled_minmax(overlap_ratio, 0.30, 1.0),
            _scaled_inverse(progress_short, 0.0, 0.30),
        ]
    )

    score_df["intent_score_inside_compression"] = _mean_score(
        [
            inside_bar.astype(float),
            _scaled_inverse(range_expansion, 0.0, config.compression_threshold),
            _scaled_inverse(atr_scaled_range, 0.0, 1.0),
            _scaled_minmax(overlap_ratio, 0.40, 1.0),
            _scaled_inverse(body_ratio, 0.0, 0.40),
        ]
    )

    score_df["intent_score_outside_expansion"] = _mean_score(
        [
            outside_bar.astype(float),
            _scaled_minmax(range_expansion, config.expansion_threshold, 3.0),
            _scaled_minmax(atr_scaled_range, 1.0, 3.0),
            _scaled_minmax(body_ratio, 0.25, 1.0),
            _scaled_inverse(overlap_ratio, 0.0, 0.80),
        ]
    )

    score_df["intent_score_exhaustion_up_candidate"] = _mean_score(
        [
            _binary_gate(direction > 0),
            _scaled_minmax(upper_wick_ratio, 0.25, 1.0),
            _scaled_minmax(range_expansion, config.expansion_threshold, 3.0),
            _scaled_inverse(progress_short, 0.0, config.exhaustion_progress_threshold),
            _scaled_inverse(close_location, 0.40, 1.0),
        ]
    )

    score_df["intent_score_exhaustion_down_candidate"] = _mean_score(
        [
            _binary_gate(direction < 0),
            _scaled_minmax(lower_wick_ratio, 0.25, 1.0),
            _scaled_minmax(range_expansion, config.expansion_threshold, 3.0),
            _scaled_inverse(progress_short, 0.0, config.exhaustion_progress_threshold),
            _scaled_minmax(close_location, 0.0, 0.60),
        ]
    )

    out = pd.concat([out, score_df], axis=1)

    score_columns = list(LABEL_TO_SCORE_COLUMN.values())
    reverse_label_map = {v: k for k, v in LABEL_TO_SCORE_COLUMN.items()}

    score_values = out[score_columns].to_numpy(dtype=float)
    max_idx = np.argmax(score_values, axis=1)
    top_scores = score_values[np.arange(len(out)), max_idx]

    if len(score_columns) > 1:
        second_scores = np.partition(score_values, -2, axis=1)[:, -2]
    else:
        second_scores = np.zeros(len(out), dtype=float)

    confidence = _safe_confidence(top_scores, second_scores)
    dominant_labels = [reverse_label_map[score_columns[i]] for i in max_idx]

    out["dominant_intent_raw"] = dominant_labels
    out["intent_confidence_raw"] = confidence

    out["dominant_intent"] = _apply_intent_arbitration(out)
    out["intent_confidence"] = confidence
    out["intent_score_top"] = top_scores
    out["intent_score_second"] = second_scores

    return out


def latest_intent_snapshot(intent_df: pd.DataFrame) -> Dict[str, object]:
    if intent_df.empty:
        raise ValueError("intent_df is empty.")

    latest = intent_df.iloc[-1]

    snapshot: Dict[str, object] = {
        "dominant_intent": latest["dominant_intent"],
        "intent_confidence": float(latest["intent_confidence"]),
        "intent_score_top": float(latest["intent_score_top"]),
        "intent_score_second": float(latest["intent_score_second"]),
    }

    for label, score_col in LABEL_TO_SCORE_COLUMN.items():
        snapshot[score_col] = float(latest[score_col])

    return snapshot


def _apply_intent_arbitration(df: pd.DataFrame) -> pd.Series:
    dominant = df["dominant_intent_raw"].copy()

    bullish_cont = df[[
        "intent_score_strong_bullish_continuation",
        "intent_score_weak_bullish_continuation",
    ]].max(axis=1)

    bearish_cont = df[[
        "intent_score_strong_bearish_continuation",
        "intent_score_weak_bearish_continuation",
    ]].max(axis=1)

    indecision = df["intent_score_indecision"]
    absorption_buy = df["intent_score_buy_absorption_candidate"]
    absorption_sell = df["intent_score_sell_absorption_candidate"]
    exhaustion_up = df["intent_score_exhaustion_up_candidate"]
    exhaustion_down = df["intent_score_exhaustion_down_candidate"]
    confidence = df["intent_confidence_raw"]

    buy_absorption_override = (
        (dominant == "BUY_ABSORPTION_CANDIDATE")
        & (confidence < 0.35)
        & (bullish_cont >= absorption_buy * 0.85)
    )
    dominant.loc[buy_absorption_override] = "WEAK_BULLISH_CONTINUATION"

    sell_absorption_override = (
        (dominant == "SELL_ABSORPTION_CANDIDATE")
        & (confidence < 0.35)
        & (bearish_cont >= absorption_sell * 0.85)
    )
    dominant.loc[sell_absorption_override] = "WEAK_BEARISH_CONTINUATION"

    exhaustion_up_override = (
        (dominant == "EXHAUSTION_UP_CANDIDATE")
        & (confidence < 0.35)
        & (bullish_cont >= exhaustion_up * 0.85)
    )
    dominant.loc[exhaustion_up_override] = "WEAK_BULLISH_CONTINUATION"

    exhaustion_down_override = (
        (dominant == "EXHAUSTION_DOWN_CANDIDATE")
        & (confidence < 0.35)
        & (bearish_cont >= exhaustion_down * 0.85)
    )
    dominant.loc[exhaustion_down_override] = "WEAK_BEARISH_CONTINUATION"

    indecision_override = (
        (dominant.isin(["BUY_ABSORPTION_CANDIDATE", "SELL_ABSORPTION_CANDIDATE"]))
        & (indecision >= 0.85 * pd.concat([absorption_buy, absorption_sell], axis=1).max(axis=1))
        & (confidence < 0.30)
    )
    dominant.loc[indecision_override] = "INDECISION"

    return dominant


def _validate_feature_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns for candle intent engine: {missing}")
    if df.empty:
        raise ValueError("feature_df is empty.")


def _binary_gate(condition: pd.Series) -> pd.Series:
    return condition.astype(float)


def _scaled_minmax(series: pd.Series, low: float, high: float) -> pd.Series:
    if high <= low:
        raise ValueError("high must be greater than low in _scaled_minmax.")
    values = (series - low) / (high - low)
    return values.clip(lower=0.0, upper=1.0).fillna(0.0)


def _scaled_inverse(series: pd.Series, low: float, high: float) -> pd.Series:
    return 1.0 - _scaled_minmax(series, low, high)


def _mean_score(parts: List[pd.Series]) -> pd.Series:
    stacked = pd.concat(parts, axis=1).fillna(0.0)
    return stacked.mean(axis=1).clip(lower=0.0, upper=1.0)


def _safe_confidence(top: np.ndarray, second: np.ndarray) -> np.ndarray:
    numerator = np.maximum(top - second, 0.0)
    denominator = np.maximum(top, 1e-12)
    confidence = numerator / denominator
    return np.clip(confidence, 0.0, 1.0)