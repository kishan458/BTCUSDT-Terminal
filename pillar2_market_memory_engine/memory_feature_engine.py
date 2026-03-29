from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import sqlite3


# ============================================================
# CONFIG
# ============================================================

@dataclass
class MemoryFeatureConfig:
    db_path: str = "database/btc_terminal.db"
    table_name: str = "btc_price_1h"
    timestamp_col: str = "timestamp"

    atr_window: int = 14
    rv_short_window: int = 6
    rv_medium_window: int = 12
    percentile_window: int = 252

    efficiency_short_window: int = 6
    efficiency_medium_window: int = 12

    month_end_days: int = 3

    output_parquet_path: Optional[str] = None


# ============================================================
# NUMPY ROLLING HELPERS
# ============================================================

def _rolling_mean_np(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    values = series.astype("float64").to_numpy()
    n = len(values)

    if min_periods is None:
        min_periods = window

    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        valid = chunk[~np.isnan(chunk)]

        if len(valid) >= min_periods:
            out[i] = valid.mean()

    return pd.Series(out, index=series.index)


def _rolling_std_np(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    values = series.astype("float64").to_numpy()
    n = len(values)

    if min_periods is None:
        min_periods = window

    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        valid = chunk[~np.isnan(chunk)]

        if len(valid) >= min_periods:
            out[i] = valid.std(ddof=1) if len(valid) > 1 else 0.0

    return pd.Series(out, index=series.index)


def _rolling_sum_np(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    values = series.astype("float64").to_numpy()
    n = len(values)

    if min_periods is None:
        min_periods = window

    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        valid = chunk[~np.isnan(chunk)]

        if len(valid) >= min_periods:
            out[i] = valid.sum()

    return pd.Series(out, index=series.index)


def _rolling_median_np(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    values = series.astype("float64").to_numpy()
    n = len(values)

    if min_periods is None:
        min_periods = window

    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        valid = chunk[~np.isnan(chunk)]

        if len(valid) >= min_periods:
            out[i] = np.median(valid)

    return pd.Series(out, index=series.index)


def _rolling_percentile_rank_np(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    values = series.astype("float64").to_numpy()
    n = len(values)

    if min_periods is None:
        min_periods = window

    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        valid = chunk[~np.isnan(chunk)]

        if len(valid) >= min_periods and not np.isnan(values[i]):
            out[i] = (valid <= values[i]).sum() / len(valid)

    return pd.Series(out, index=series.index)


# ============================================================
# LOADING / VALIDATION
# ============================================================

def load_btc_ohlcv_from_sqlite(
    db_path: str,
    table_name: str,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Load BTC OHLCV data from SQLite and return a clean DataFrame.
    """
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    query = f"""
        SELECT
            {timestamp_col} AS timestamp_utc,
            open,
            high,
            low,
            close,
            volume
        FROM {table_name}
        ORDER BY {timestamp_col} ASC
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        raise ValueError(f"No rows found in {table_name} from {db_path}")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=False, errors="coerce")
    if df["timestamp_utc"].isna().any():
        raise ValueError("Failed to parse some timestamp values.")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    if df[numeric_cols].isna().any().any():
        bad_counts = df[numeric_cols].isna().sum().to_dict()
        raise ValueError(f"NaNs found in OHLCV columns: {bad_counts}")

    return df


def validate_ohlcv(df: pd.DataFrame) -> None:
    required_cols = ["timestamp_utc", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["timestamp_utc"].duplicated().any():
        dupes = int(df["timestamp_utc"].duplicated().sum())
        raise ValueError(f"Duplicate timestamps found: {dupes}")

    if not df["timestamp_utc"].is_monotonic_increasing:
        raise ValueError("timestamp_utc is not sorted ascending.")

    bad_rows = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1)) |
        (df["low"] > df[["open", "close", "high"]].min(axis=1)) |
        (df["high"] < df["low"])
    )
    if bad_rows.any():
        raise ValueError(f"Found {int(bad_rows.sum())} OHLC constraint violations.")


# ============================================================
# TIME / SESSION FEATURES
# ============================================================

def _get_session_label(hour_utc: int) -> str:
    if 0 <= hour_utc < 8:
        return "ASIA"
    if 8 <= hour_utc < 13:
        return "LONDON"
    if 13 <= hour_utc < 21:
        return "NY"
    return "LATE_US"


def _get_session_transition_label(hour_utc: int) -> str:
    if 7 <= hour_utc <= 9:
        return "ASIA_TO_LONDON"
    if 12 <= hour_utc <= 14:
        return "LONDON_TO_NY"
    if 20 <= hour_utc <= 22:
        return "NY_TO_ASIA"
    return "NONE"


def add_time_features(df: pd.DataFrame, month_end_days: int = 3) -> pd.DataFrame:
    out = df.copy()

    ts = out["timestamp_utc"]
    out["hour_utc"] = ts.dt.hour
    out["weekday"] = ts.dt.weekday
    out["weekday_name"] = ts.dt.day_name().str.upper()
    out["is_weekend"] = out["weekday"] >= 5
    out["month"] = ts.dt.month

    month_end = ts.dt.days_in_month - ts.dt.day
    out["is_month_end"] = month_end < month_end_days
    out["is_quarter_end"] = out["month"].isin([3, 6, 9, 12]) & out["is_month_end"]

    out["session_label"] = out["hour_utc"].apply(_get_session_label)
    out["session_transition_label"] = out["hour_utc"].apply(_get_session_transition_label)

    out["is_sunday_open_window"] = (out["weekday"] == 6) & (out["hour_utc"] <= 5)
    out["is_monday_open_window"] = (out["weekday"] == 0) & (out["hour_utc"] <= 5)

    return out


# ============================================================
# RETURN FEATURES
# ============================================================

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype("float64")

    for h in [1, 3, 6, 12]:
        out[f"ret_{h}"] = close / close.shift(h) - 1.0
        out[f"log_ret_{h}"] = np.log(close / close.shift(h))

    return out


# ============================================================
# VOLATILITY FEATURES
# ============================================================

def add_volatility_features(
    df: pd.DataFrame,
    atr_window: int = 14,
    rv_short_window: int = 6,
    rv_medium_window: int = 12,
    percentile_window: int = 252,
) -> pd.DataFrame:
    out = df.copy()

    close = out["close"].astype("float64")
    high = out["high"].astype("float64")
    low = out["low"].astype("float64")
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["true_range"] = true_range.astype("float64")

    out["atr_14"] = _rolling_mean_np(out["true_range"], atr_window, min_periods=atr_window)
    out["atr_pct_of_close"] = out["atr_14"] / close
    out["range_pct"] = (high - low) / close

    log_ret_1 = np.log(close / close.shift(1))
    out["realized_vol_6"] = _rolling_std_np(log_ret_1, rv_short_window, min_periods=rv_short_window)
    out["realized_vol_12"] = _rolling_std_np(log_ret_1, rv_medium_window, min_periods=rv_medium_window)

    median_tr = _rolling_median_np(out["true_range"], 20, min_periods=20)
    out["range_expansion_ratio"] = out["true_range"] / median_tr.replace(0, np.nan)

    out["atr_percentile_252"] = _rolling_percentile_rank_np(
        out["atr_pct_of_close"],
        percentile_window,
        min_periods=percentile_window,
    )

    out["realized_vol_percentile_252"] = _rolling_percentile_rank_np(
        out["realized_vol_12"],
        percentile_window,
        min_periods=percentile_window,
    )

    return out


# ============================================================
# CANDLE GEOMETRY
# ============================================================

def add_candle_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    bar_range = out["high"] - out["low"]
    safe_range = bar_range.replace(0, np.nan)

    out["body"] = out["close"] - out["open"]
    out["body_abs"] = out["body"].abs()
    out["body_pct_of_range"] = out["body_abs"] / safe_range

    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]

    out["upper_wick_pct"] = out["upper_wick"] / safe_range
    out["lower_wick_pct"] = out["lower_wick"] / safe_range

    out["close_location_value"] = (out["close"] - out["low"]) / safe_range

    out["direction"] = np.select(
        [out["close"] > out["open"], out["close"] < out["open"]],
        [1, -1],
        default=0,
    )

    return out


# ============================================================
# PATH / BEHAVIOR FEATURES
# ============================================================

def _directional_efficiency(close: pd.Series, window: int) -> pd.Series:
    close = close.astype("float64")
    net_move = (close - close.shift(window)).abs()
    step_sum = _rolling_sum_np(close.diff().abs(), window, min_periods=window)
    return net_move / step_sum.replace(0, np.nan)


def _rolling_sign_consistency(close: pd.Series, window: int) -> pd.Series:
    close = close.astype("float64")
    signs = np.sign(close.diff()).astype("float64").to_numpy()
    n = len(signs)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        start = max(0, i - window + 1)
        chunk = signs[start:i + 1]
        valid = chunk[~np.isnan(chunk)]

        if len(valid) >= window:
            out[i] = np.abs(valid.mean())

    return pd.Series(out, index=close.index)


def add_path_features(
    df: pd.DataFrame,
    short_window: int = 6,
    medium_window: int = 12,
) -> pd.DataFrame:
    out = df.copy()

    prev_high = out["high"].shift(1)
    prev_low = out["low"].shift(1)

    overlap_high = np.minimum(out["high"], prev_high)
    overlap_low = np.maximum(out["low"], prev_low)
    overlap_raw = (overlap_high - overlap_low).clip(lower=0)

    prev_range = (prev_high - prev_low).replace(0, np.nan)
    out["overlap_pct_prev_bar"] = overlap_raw / prev_range

    out["directional_efficiency_6"] = _directional_efficiency(out["close"], short_window)
    out["directional_efficiency_12"] = _directional_efficiency(out["close"], medium_window)

    out["rolling_sign_consistency_6"] = _rolling_sign_consistency(out["close"], short_window)
    out["rolling_sign_consistency_12"] = _rolling_sign_consistency(out["close"], medium_window)

    out["progress_vs_range_6"] = (
        (out["close"] - out["close"].shift(short_window)).abs()
        / _rolling_sum_np(out["true_range"], short_window, min_periods=short_window).replace(0, np.nan)
    )

    out["progress_vs_range_12"] = (
        (out["close"] - out["close"].shift(medium_window)).abs()
        / _rolling_sum_np(out["true_range"], medium_window, min_periods=medium_window).replace(0, np.nan)
    )

    out["chop_score_6"] = 1.0 - out["directional_efficiency_6"]
    out["chop_score_12"] = 1.0 - out["directional_efficiency_12"]

    return out


# ============================================================
# STATE BUCKETS
# ============================================================

def _bucket_volatility(p: float) -> Optional[str]:
    if pd.isna(p):
        return None
    if p < 0.2:
        return "VERY_LOW"
    if p < 0.4:
        return "LOW"
    if p < 0.6:
        return "MODERATE"
    if p < 0.8:
        return "HIGH"
    return "EXTREME"


def _momentum_state(ret_6: float, eff_6: float) -> Optional[str]:
    if pd.isna(ret_6) or pd.isna(eff_6):
        return None

    if ret_6 > 0.02 and eff_6 > 0.55:
        return "STRONG_UP"
    if ret_6 > 0:
        return "WEAK_UP"
    if ret_6 < -0.02 and eff_6 > 0.55:
        return "STRONG_DOWN"
    if ret_6 < 0:
        return "WEAK_DOWN"
    return "NEUTRAL"


def _range_position(clv: float) -> Optional[str]:
    if pd.isna(clv):
        return None
    if clv < (1 / 3):
        return "LOW_RANGE"
    if clv < (2 / 3):
        return "MID_RANGE"
    return "HIGH_RANGE"


def add_state_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["volatility_bucket"] = out["atr_percentile_252"].apply(_bucket_volatility)

    out["expansion_state"] = np.select(
        [
            out["range_expansion_ratio"] >= 1.5,
            out["range_expansion_ratio"] <= 0.75,
        ],
        [
            "EXPANSION",
            "COMPRESSION",
        ],
        default="NORMAL",
    )

    out["compression_state"] = np.select(
        [
            out["range_expansion_ratio"] <= 0.75,
            out["range_expansion_ratio"] >= 1.5,
        ],
        [
            "COMPRESSED",
            "EXPANDED",
        ],
        default="NORMAL",
    )

    out["range_position"] = out["close_location_value"].apply(_range_position)

    out["momentum_state"] = [
        _momentum_state(r, e)
        for r, e in zip(out["ret_6"], out["directional_efficiency_6"])
    ]

    return out


# ============================================================
# RESERVED FUTURE-INTEGRATION FIELDS
# ============================================================

def add_reserved_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    reserved_cols = {
        "candle_intent": None,
        "breakout_quality": None,
        "overlap_state": None,
        "follow_through_quality": None,
        "pressure_bias": None,
        "nearest_liquidity_side": None,
        "trap_risk_bucket": None,
        "volatility_regime": None,
        "trend_regime": None,
        "event_context": None,
    }

    for col, val in reserved_cols.items():
        out[col] = val

    return out


# ============================================================
# FINAL BUILD
# ============================================================

def build_memory_features(df: pd.DataFrame, config: MemoryFeatureConfig) -> pd.DataFrame:
    validate_ohlcv(df)

    out = df.copy()
    out = add_time_features(out, month_end_days=config.month_end_days)
    out = add_return_features(out)
    out = add_volatility_features(
        out,
        atr_window=config.atr_window,
        rv_short_window=config.rv_short_window,
        rv_medium_window=config.rv_medium_window,
        percentile_window=config.percentile_window,
    )
    out = add_candle_geometry_features(out)
    out = add_path_features(
        out,
        short_window=config.efficiency_short_window,
        medium_window=config.efficiency_medium_window,
    )
    out = add_state_buckets(out)
    out = add_reserved_columns(out)

    return out


def run_memory_feature_engine(config: Optional[MemoryFeatureConfig] = None) -> pd.DataFrame:
    if config is None:
        config = MemoryFeatureConfig()

    df = load_btc_ohlcv_from_sqlite(
        db_path=config.db_path,
        table_name=config.table_name,
        timestamp_col=config.timestamp_col,
    )

    features = build_memory_features(df, config)

    if config.output_parquet_path:
        out_path = Path(config.output_parquet_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(out_path, index=False)

    return features


if __name__ == "__main__":
    cfg = MemoryFeatureConfig(
        db_path="database/btc_terminal.db",
        table_name="btc_price_1h",
        timestamp_col="timestamp",
        output_parquet_path=None,
    )

    df_features = run_memory_feature_engine(cfg)

    print("\n=== MEMORY FEATURE ENGINE SUCCESS ===")
    print(f"Rows: {len(df_features)}")
    print(f"Columns: {len(df_features.columns)}")
    print("\nLast 3 rows preview:")
    preview_cols = [
        "timestamp_utc",
        "close",
        "session_label",
        "volatility_bucket",
        "expansion_state",
        "momentum_state",
        "directional_efficiency_6",
        "atr_14",
        "range_expansion_ratio",
    ]
    print(df_features[preview_cols].tail(3).to_string(index=False))