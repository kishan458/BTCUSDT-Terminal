from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Tuple

import numpy as np
import pandas as pd


AtrMethod = Literal["wilder", "sma"]


@dataclass(frozen=True)
class CandleFeatureConfig:
    atr_window: int
    range_mean_window: int
    body_mean_window: int
    zscore_window: int
    overlap_window_short: int
    overlap_window_long: int
    progress_window_short: int
    progress_window_medium: int
    progress_window_long: int
    persistence_window_short: int
    persistence_window_medium: int
    persistence_window_long: int
    realized_vol_window: int
    volatility_percentile_window: int
    range_percentile_window: int
    body_percentile_window: int
    contraction_window: int
    inside_outside_window: int
    entropy_window: int
    rolling_wick_window: int

    def validate(self) -> None:
        values = {
            "atr_window": self.atr_window,
            "range_mean_window": self.range_mean_window,
            "body_mean_window": self.body_mean_window,
            "zscore_window": self.zscore_window,
            "overlap_window_short": self.overlap_window_short,
            "overlap_window_long": self.overlap_window_long,
            "progress_window_short": self.progress_window_short,
            "progress_window_medium": self.progress_window_medium,
            "progress_window_long": self.progress_window_long,
            "persistence_window_short": self.persistence_window_short,
            "persistence_window_medium": self.persistence_window_medium,
            "persistence_window_long": self.persistence_window_long,
            "realized_vol_window": self.realized_vol_window,
            "volatility_percentile_window": self.volatility_percentile_window,
            "range_percentile_window": self.range_percentile_window,
            "body_percentile_window": self.body_percentile_window,
            "contraction_window": self.contraction_window,
            "inside_outside_window": self.inside_outside_window,
            "entropy_window": self.entropy_window,
            "rolling_wick_window": self.rolling_wick_window,
        }
        for name, value in values.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")


@dataclass(frozen=True)
class OhlcColumns:
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: Optional[str] = "volume"
    timestamp: Optional[str] = "timestamp"


def add_candle_features(
    df: pd.DataFrame,
    config: CandleFeatureConfig,
    columns: OhlcColumns = OhlcColumns(),
    atr_method: AtrMethod = "wilder",
    copy: bool = True,
) -> pd.DataFrame:
    config.validate()
    _validate_input_dataframe(df, columns)

    out = df.copy(deep=True) if copy else df.copy(deep=True)

    o = out[columns.open].astype(float)
    h = out[columns.high].astype(float)
    l = out[columns.low].astype(float)
    c = out[columns.close].astype(float)

    eps = np.finfo(float).eps
    feature_map: Dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # Diagnostics / structure checks
    # ------------------------------------------------------------------
    feature_map["ohlc_high_ge_low"] = (h >= l).astype(int)
    feature_map["ohlc_open_within_range"] = ((o >= l) & (o <= h)).astype(int)
    feature_map["ohlc_close_within_range"] = ((c >= l) & (c <= h)).astype(int)
    feature_map["ohlc_constraints_ok"] = (
        (feature_map["ohlc_high_ge_low"] == 1)
        & (feature_map["ohlc_open_within_range"] == 1)
        & (feature_map["ohlc_close_within_range"] == 1)
    ).astype(int)

    feature_map["is_zero_range_bar"] = ((h - l).abs() <= eps).astype(int)

    # ------------------------------------------------------------------
    # Core raw geometry
    # ------------------------------------------------------------------
    feature_map["body_size"] = (c - o).abs()
    feature_map["full_range"] = (h - l).clip(lower=0.0)
    feature_map["upper_wick"] = (h - np.maximum(o, c)).clip(lower=0.0)
    feature_map["lower_wick"] = (np.minimum(o, c) - l).clip(lower=0.0)

    feature_map["direction"] = np.sign(c - o).astype(int)
    feature_map["signed_body"] = c - o
    feature_map["close_to_close_return"] = c.diff()
    feature_map["open_to_open_return"] = o.diff()

    full_range = feature_map["full_range"]
    body_size = feature_map["body_size"]
    upper_wick = feature_map["upper_wick"]
    lower_wick = feature_map["lower_wick"]

    # ------------------------------------------------------------------
    # Range-position and ratio features
    # ------------------------------------------------------------------
    feature_map["body_to_range_ratio"] = _safe_divide(body_size, full_range, eps)
    feature_map["upper_wick_to_range_ratio"] = _safe_divide(upper_wick, full_range, eps)
    feature_map["lower_wick_to_range_ratio"] = _safe_divide(lower_wick, full_range, eps)
    feature_map["total_wick_to_range_ratio"] = _safe_divide(
        upper_wick + lower_wick, full_range, eps
    )
    feature_map["wick_imbalance"] = _safe_divide(
        lower_wick - upper_wick, full_range, eps
    )
    feature_map["bar_efficiency"] = _safe_divide(body_size, full_range, eps)

    feature_map["close_location_value"] = _safe_divide(c - l, full_range, eps)
    feature_map["open_location_value"] = _safe_divide(o - l, full_range, eps)

    feature_map["midpoint"] = (h + l) / 2.0
    feature_map["midpoint_displacement"] = _safe_divide(
        c - feature_map["midpoint"], full_range, eps
    )

    feature_map["body_midpoint"] = (o + c) / 2.0
    feature_map["body_midpoint_displacement"] = _safe_divide(
        feature_map["body_midpoint"] - feature_map["midpoint"],
        full_range,
        eps,
    )

    # ------------------------------------------------------------------
    # Previous-bar interaction
    # ------------------------------------------------------------------
    prev_close = c.shift(1)
    prev_open = o.shift(1)
    prev_high = h.shift(1)
    prev_low = l.shift(1)
    prev_range = full_range.shift(1)
    prev_body = body_size.shift(1)

    feature_map["prev_open"] = prev_open
    feature_map["prev_high"] = prev_high
    feature_map["prev_low"] = prev_low
    feature_map["prev_close"] = prev_close
    feature_map["prev_full_range"] = prev_range
    feature_map["prev_body_size"] = prev_body

    feature_map["gap_from_prev_close"] = o - prev_close
    feature_map["gap_from_prev_open"] = o - prev_open
    feature_map["high_extension_vs_prev_high"] = h - prev_high
    feature_map["low_extension_vs_prev_low"] = prev_low - l
    feature_map["close_progress_vs_prev_close"] = c - prev_close
    feature_map["open_progress_vs_prev_open"] = o - prev_open

    feature_map["inside_bar_flag"] = ((h <= prev_high) & (l >= prev_low)).astype(int)
    feature_map["outside_bar_flag"] = ((h > prev_high) & (l < prev_low)).astype(int)
    feature_map["higher_high_flag"] = (h > prev_high).astype(int)
    feature_map["lower_low_flag"] = (l < prev_low).astype(int)
    feature_map["higher_close_flag"] = (c > prev_close).astype(int)
    feature_map["lower_close_flag"] = (c < prev_close).astype(int)

    # ------------------------------------------------------------------
    # True range / ATR / normalization
    # ------------------------------------------------------------------
    feature_map["true_range"] = _true_range(h, l, prev_close)
    feature_map["atr"] = _atr(feature_map["true_range"], config.atr_window, atr_method)

    atr = feature_map["atr"]
    feature_map["atr_scaled_range"] = _safe_divide(full_range, atr, eps)
    feature_map["atr_scaled_body"] = _safe_divide(body_size, atr, eps)
    feature_map["atr_scaled_gap"] = _safe_divide(feature_map["gap_from_prev_close"], atr, eps)
    feature_map["atr_scaled_upper_wick"] = _safe_divide(upper_wick, atr, eps)
    feature_map["atr_scaled_lower_wick"] = _safe_divide(lower_wick, atr, eps)
    feature_map["atr_scaled_close_to_close_return"] = _safe_divide(
        feature_map["close_to_close_return"],
        atr,
        eps,
    )

    # ------------------------------------------------------------------
    # Rolling averages, shocks, z-scores
    # ------------------------------------------------------------------
    feature_map["rolling_avg_range"] = full_range.rolling(
        window=config.range_mean_window,
        min_periods=1,
    ).mean()
    feature_map["rolling_avg_body"] = body_size.rolling(
        window=config.body_mean_window,
        min_periods=1,
    ).mean()

    feature_map["range_expansion_score"] = _safe_divide(
        full_range,
        feature_map["rolling_avg_range"],
        eps,
    )
    feature_map["body_expansion_score"] = _safe_divide(
        body_size,
        feature_map["rolling_avg_body"],
        eps,
    )

    feature_map["range_zscore"] = _rolling_zscore(full_range, config.zscore_window)
    feature_map["body_zscore"] = _rolling_zscore(body_size, config.zscore_window)
    feature_map["close_location_zscore"] = _rolling_zscore(
        feature_map["close_location_value"],
        config.zscore_window,
    )
    feature_map["wick_imbalance_zscore"] = _rolling_zscore(
        feature_map["wick_imbalance"],
        config.zscore_window,
    )

    feature_map["range_shock"] = feature_map["range_expansion_score"]
    feature_map["body_shock"] = feature_map["body_expansion_score"]
    feature_map["gap_shock"] = _safe_divide(
        feature_map["gap_from_prev_close"].abs(),
        atr,
        eps,
    )

    # ------------------------------------------------------------------
    # Overlap / acceptance / churn
    # ------------------------------------------------------------------
    overlap_numerator = np.maximum(
        0.0,
        np.minimum(h, prev_high) - np.maximum(l, prev_low),
    )
    overlap_denominator = np.minimum(full_range, prev_range)

    feature_map["overlap_length_vs_prev_bar"] = overlap_numerator
    feature_map["overlap_ratio_vs_prev_bar"] = _safe_divide(
        overlap_numerator,
        overlap_denominator,
        eps,
    )

    feature_map["avg_overlap_ratio_short"] = feature_map["overlap_ratio_vs_prev_bar"].rolling(
        window=config.overlap_window_short,
        min_periods=1,
    ).mean()
    feature_map["avg_overlap_ratio_long"] = feature_map["overlap_ratio_vs_prev_bar"].rolling(
        window=config.overlap_window_long,
        min_periods=1,
    ).mean()

    feature_map["overlap_compression_score"] = feature_map["avg_overlap_ratio_long"]
    feature_map["overlap_decay_after_breakout"] = (
        feature_map["avg_overlap_ratio_long"] - feature_map["avg_overlap_ratio_short"]
    )

    # ------------------------------------------------------------------
    # Progress efficiency / net progress
    # ------------------------------------------------------------------
    (
        feature_map["net_progress_short"],
        feature_map["gross_travel_short"],
        feature_map["progress_efficiency_short"],
    ) = _progress_features(c, o, full_range, config.progress_window_short, eps)

    (
        feature_map["net_progress_medium"],
        feature_map["gross_travel_medium"],
        feature_map["progress_efficiency_medium"],
    ) = _progress_features(c, o, full_range, config.progress_window_medium, eps)

    (
        feature_map["net_progress_long"],
        feature_map["gross_travel_long"],
        feature_map["progress_efficiency_long"],
    ) = _progress_features(c, o, full_range, config.progress_window_long, eps)

    # ------------------------------------------------------------------
    # Sequence persistence / directional consistency
    # ------------------------------------------------------------------
    feature_map["same_direction_body_count"] = _consecutive_run_length(feature_map["direction"])
    feature_map["same_direction_close_count"] = _consecutive_run_length(
        np.sign(feature_map["close_to_close_return"]).fillna(0)
    )

    clv = feature_map["close_location_value"]
    feature_map["close_upper_half_flag"] = (clv >= 0.5).astype(int)
    feature_map["close_lower_half_flag"] = (clv <= 0.5).astype(int)

    feature_map["close_upper_half_count_short"] = feature_map["close_upper_half_flag"].rolling(
        window=config.persistence_window_short,
        min_periods=1,
    ).sum()
    feature_map["close_upper_half_count_medium"] = feature_map["close_upper_half_flag"].rolling(
        window=config.persistence_window_medium,
        min_periods=1,
    ).sum()
    feature_map["close_upper_half_count_long"] = feature_map["close_upper_half_flag"].rolling(
        window=config.persistence_window_long,
        min_periods=1,
    ).sum()

    feature_map["close_lower_half_count_short"] = feature_map["close_lower_half_flag"].rolling(
        window=config.persistence_window_short,
        min_periods=1,
    ).sum()
    feature_map["close_lower_half_count_medium"] = feature_map["close_lower_half_flag"].rolling(
        window=config.persistence_window_medium,
        min_periods=1,
    ).sum()
    feature_map["close_lower_half_count_long"] = feature_map["close_lower_half_flag"].rolling(
        window=config.persistence_window_long,
        min_periods=1,
    ).sum()

    feature_map["rolling_body_dominance_short"] = _safe_divide(
        body_size.rolling(config.persistence_window_short, min_periods=1).sum(),
        full_range.rolling(config.persistence_window_short, min_periods=1).sum(),
        eps,
    )
    feature_map["rolling_body_dominance_medium"] = _safe_divide(
        body_size.rolling(config.persistence_window_medium, min_periods=1).sum(),
        full_range.rolling(config.persistence_window_medium, min_periods=1).sum(),
        eps,
    )
    feature_map["rolling_body_dominance_long"] = _safe_divide(
        body_size.rolling(config.persistence_window_long, min_periods=1).sum(),
        full_range.rolling(config.persistence_window_long, min_periods=1).sum(),
        eps,
    )

    rolling_lower_wick = lower_wick.rolling(
        window=config.rolling_wick_window,
        min_periods=1,
    ).sum()
    rolling_upper_wick = upper_wick.rolling(
        window=config.rolling_wick_window,
        min_periods=1,
    ).sum()
    rolling_range = full_range.rolling(
        window=config.rolling_wick_window,
        min_periods=1,
    ).sum()

    feature_map["rolling_wick_dominance"] = _safe_divide(
        rolling_lower_wick - rolling_upper_wick,
        rolling_range,
        eps,
    )

    feature_map["rolling_sign_consistency_short"] = _rolling_sign_consistency(
        feature_map["close_to_close_return"],
        config.persistence_window_short,
    )
    feature_map["rolling_sign_consistency_medium"] = _rolling_sign_consistency(
        feature_map["close_to_close_return"],
        config.persistence_window_medium,
    )
    feature_map["rolling_sign_consistency_long"] = _rolling_sign_consistency(
        feature_map["close_to_close_return"],
        config.persistence_window_long,
    )

    # ------------------------------------------------------------------
    # Compression / expansion regime helpers
    # ------------------------------------------------------------------
    feature_map["inside_bar_frequency"] = feature_map["inside_bar_flag"].rolling(
        window=config.inside_outside_window,
        min_periods=1,
    ).mean()
    feature_map["outside_bar_frequency"] = feature_map["outside_bar_flag"].rolling(
        window=config.inside_outside_window,
        min_periods=1,
    ).mean()

    feature_map["range_contraction_flag"] = (full_range < full_range.shift(1)).astype(int)
    feature_map["range_contraction_streak"] = _streak_from_binary(
        feature_map["range_contraction_flag"]
    )

    inside_bar_freq_safe = feature_map["inside_bar_frequency"].replace(0.0, np.nan)
    feature_map["expansion_after_compression_score"] = _safe_divide(
        feature_map["range_expansion_score"],
        inside_bar_freq_safe,
        eps,
    ).fillna(0.0)

    feature_map["post_expansion_fade_score"] = (
        feature_map["avg_overlap_ratio_short"]
        * (1.0 - feature_map["progress_efficiency_short"])
    ).clip(lower=0.0)

    # ------------------------------------------------------------------
    # Realized / OHLC-aware volatility block
    # ------------------------------------------------------------------
    feature_map["log_hl"] = _safe_log_ratio(h, l, eps)
    feature_map["log_co"] = _safe_log_ratio(c, o, eps)
    feature_map["log_oc_prev"] = _safe_log_ratio(o, prev_close, eps)
    feature_map["log_ho"] = _safe_log_ratio(h, o, eps)
    feature_map["log_lo"] = _safe_log_ratio(l, o, eps)
    feature_map["log_hc"] = _safe_log_ratio(h, c, eps)
    feature_map["log_lc"] = _safe_log_ratio(l, c, eps)
    feature_map["log_cc"] = _safe_log_ratio(c, prev_close, eps)

    feature_map["realized_volatility"] = feature_map["log_cc"].rolling(
        window=config.realized_vol_window,
        min_periods=1,
    ).std(ddof=0)

    feature_map["parkinson_vol"] = _parkinson_vol(
        feature_map["log_hl"],
        config.realized_vol_window,
    )
    feature_map["garman_klass_vol"] = _garman_klass_vol(
        feature_map["log_hl"],
        feature_map["log_co"],
        config.realized_vol_window,
    )
    feature_map["rogers_satchell_vol"] = _rogers_satchell_vol(
        feature_map["log_ho"],
        feature_map["log_lo"],
        feature_map["log_hc"],
        feature_map["log_lc"],
        config.realized_vol_window,
    )
    feature_map["yang_zhang_vol"] = _yang_zhang_vol(
        feature_map["log_oc_prev"],
        feature_map["log_co"],
        feature_map["log_ho"],
        feature_map["log_lo"],
        feature_map["log_hc"],
        feature_map["log_lc"],
        config.realized_vol_window,
    )

    feature_map["realized_volatility_percentile"] = _rolling_percentile_rank(
        feature_map["realized_volatility"],
        config.volatility_percentile_window,
    )
    feature_map["range_shock_percentile"] = _rolling_percentile_rank(
        full_range,
        config.range_percentile_window,
    )
    feature_map["body_shock_percentile"] = _rolling_percentile_rank(
        body_size,
        config.body_percentile_window,
    )

    # ------------------------------------------------------------------
    # Entropy / irregularity helpers
    # ------------------------------------------------------------------
    direction_for_entropy = feature_map["direction"].replace(0, np.nan).ffill().fillna(0)
    feature_map["direction_entropy"] = _rolling_binary_entropy(
        (direction_for_entropy > 0).astype(int),
        config.entropy_window,
    )
    feature_map["close_location_entropy"] = _rolling_bucket_entropy(
        clv,
        config.entropy_window,
    )

    # ------------------------------------------------------------------
    # Attach features in one shot to avoid fragmentation
    # ------------------------------------------------------------------
    feature_df = pd.DataFrame(feature_map, index=out.index)
    out = pd.concat([out, feature_df], axis=1)

    # ------------------------------------------------------------------
    # General quality diagnostics
    # ------------------------------------------------------------------
    feature_columns = list(feature_df.columns)
    numeric_feature_df = feature_df.select_dtypes(include=[np.number])

    out["nan_feature_count"] = feature_df.isna().sum(axis=1)
    out["inf_feature_count"] = pd.DataFrame(
        np.isinf(numeric_feature_df.to_numpy()),
        index=numeric_feature_df.index,
        columns=numeric_feature_df.columns,
    ).sum(axis=1)

    out["data_quality_ok"] = (
        (out["ohlc_constraints_ok"] == 1)
        & (out["inf_feature_count"] == 0)
    ).astype(int)

    return out.copy()


def latest_candle_feature_snapshot(
    feature_df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    if feature_df.empty:
        raise ValueError("feature_df is empty.")

    latest = feature_df.iloc[-1]
    if columns is None:
        return latest.to_dict()

    selected = {}
    for col in columns:
        if col not in feature_df.columns:
            raise KeyError(f"Column '{col}' not found in feature_df.")
        value = latest[col]
        selected[col] = None if pd.isna(value) else value
    return selected


def _validate_input_dataframe(df: pd.DataFrame, columns: OhlcColumns) -> None:
    required = [columns.open, columns.high, columns.low, columns.close]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required OHLC columns: {missing}")

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    for col in required:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric.")

    if df[required].isna().all(axis=None):
        raise ValueError("All OHLC values are NaN.")


def _safe_divide(numerator: pd.Series, denominator: pd.Series, eps: float) -> pd.Series:
    denominator_safe = denominator.replace(0.0, np.nan)
    result = numerator / denominator_safe
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_log_ratio(a: pd.Series, b: pd.Series, eps: float) -> pd.Series:
    a_safe = a.clip(lower=eps)
    b_safe = b.clip(lower=eps)
    return np.log(a_safe / b_safe)


def _true_range(
    high: pd.Series,
    low: pd.Series,
    prev_close: pd.Series,
) -> pd.Series:
    hl = high - low
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def _atr(true_range: pd.Series, window: int, method: AtrMethod) -> pd.Series:
    if method == "wilder":
        return true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=1).mean()
    if method == "sma":
        return true_range.rolling(window=window, min_periods=1).mean()
    raise ValueError(f"Unsupported atr_method: {method}")


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std(ddof=0)
    std_safe = std.replace(0.0, np.nan)
    return ((series - mean) / std_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _progress_features(
    close: pd.Series,
    open_: pd.Series,
    full_range: pd.Series,
    window: int,
    eps: float,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    open_anchor = open_.shift(window - 1)
    net_progress = close - open_anchor
    gross_travel = full_range.rolling(window=window, min_periods=1).sum()
    progress_efficiency = _safe_divide(net_progress.abs(), gross_travel, eps)
    return net_progress, gross_travel, progress_efficiency


def _consecutive_run_length(series: pd.Series) -> pd.Series:
    s = pd.Series(series).fillna(0)
    change = s.ne(s.shift()).cumsum()
    run = s.groupby(change).cumcount() + 1
    run = run.where(s != 0, 0)
    return run.astype(int)


def _streak_from_binary(binary_series: pd.Series) -> pd.Series:
    s = binary_series.fillna(0).astype(int)
    groups = (s != s.shift()).cumsum()
    streak = s.groupby(groups).cumcount() + 1
    streak = streak.where(s == 1, 0)
    return streak.astype(int)


def _rolling_sign_consistency(series: pd.Series, window: int) -> pd.Series:
    sign_series = np.sign(series.fillna(0))

    def _consistency(arr: np.ndarray) -> float:
        arr = arr[arr != 0]
        if arr.size == 0:
            return 0.0
        positive = np.sum(arr > 0)
        negative = np.sum(arr < 0)
        dominant = max(positive, negative)
        return dominant / arr.size

    return sign_series.rolling(window=window, min_periods=1).apply(_consistency, raw=True)


def _parkinson_vol(log_hl: pd.Series, window: int) -> pd.Series:
    coefficient = 1.0 / (4.0 * np.log(2.0))
    variance = coefficient * (log_hl ** 2).rolling(window=window, min_periods=1).mean()
    return np.sqrt(np.maximum(variance, 0.0))


def _garman_klass_vol(log_hl: pd.Series, log_co: pd.Series, window: int) -> pd.Series:
    coefficient = 2.0 * np.log(2.0) - 1.0
    variance = ((0.5 * (log_hl ** 2)) - (coefficient * (log_co ** 2))).rolling(
        window=window,
        min_periods=1,
    ).mean()
    variance = np.maximum(variance, 0.0)
    return np.sqrt(variance)


def _rogers_satchell_vol(
    log_ho: pd.Series,
    log_lo: pd.Series,
    log_hc: pd.Series,
    log_lc: pd.Series,
    window: int,
) -> pd.Series:
    rs_component = (log_ho * log_hc) + (log_lo * log_lc)
    variance = rs_component.rolling(window=window, min_periods=1).mean()
    variance = np.maximum(variance, 0.0)
    return np.sqrt(variance)


def _yang_zhang_vol(
    log_oc_prev: pd.Series,
    log_co: pd.Series,
    log_ho: pd.Series,
    log_lo: pd.Series,
    log_hc: pd.Series,
    log_lc: pd.Series,
    window: int,
) -> pd.Series:
    k = 0.34 / (1.34 + ((window + 1.0) / (window - 1.0))) if window > 1 else 0.0

    overnight_var = (log_oc_prev ** 2).rolling(window=window, min_periods=1).mean()
    open_close_var = (log_co ** 2).rolling(window=window, min_periods=1).mean()
    rs_var = ((log_ho * log_hc) + (log_lo * log_lc)).rolling(
        window=window,
        min_periods=1,
    ).mean()

    variance = overnight_var + (k * open_close_var) + ((1.0 - k) * rs_var)
    variance = np.maximum(variance, 0.0)
    return np.sqrt(variance)


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    def _rank(arr: np.ndarray) -> float:
        if arr.size == 0:
            return 0.0
        current = arr[-1]
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return 0.0
        return float(np.mean(valid <= current))

    return series.rolling(window=window, min_periods=1).apply(_rank, raw=True)


def _rolling_binary_entropy(binary_series: pd.Series, window: int) -> pd.Series:
    def _entropy(arr: np.ndarray) -> float:
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return 0.0
        p = float(np.mean(arr))
        q = 1.0 - p
        entropy = 0.0
        if p > 0.0:
            entropy -= p * np.log2(p)
        if q > 0.0:
            entropy -= q * np.log2(q)
        return entropy

    return binary_series.rolling(window=window, min_periods=1).apply(_entropy, raw=True)


def _rolling_bucket_entropy(series: pd.Series, window: int) -> pd.Series:
    def _entropy(arr: np.ndarray) -> float:
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return 0.0

        low_bucket = np.sum(arr < (1.0 / 3.0))
        mid_bucket = np.sum((arr >= (1.0 / 3.0)) & (arr <= (2.0 / 3.0)))
        high_bucket = np.sum(arr > (2.0 / 3.0))

        counts = np.array([low_bucket, mid_bucket, high_bucket], dtype=float)
        probs = counts / counts.sum() if counts.sum() > 0 else counts

        entropy = 0.0
        for p in probs:
            if p > 0.0:
                entropy -= p * np.log2(p)
        return entropy

    return series.rolling(window=window, min_periods=1).apply(_entropy, raw=True)