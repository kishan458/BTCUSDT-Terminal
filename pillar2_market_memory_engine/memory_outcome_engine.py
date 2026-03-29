from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pandas as pd


# ============================================================
# IMPORT FIX FOR DIRECT SCRIPT RUN
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pillar2_market_memory_engine.memory_feature_engine import (  # noqa: E402
    MemoryFeatureConfig,
    run_memory_feature_engine,
)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class MemoryOutcomeConfig:
    horizons: tuple[int, ...] = (1, 3, 6, 12)
    output_parquet_path: Optional[str] = None

    continuation_atr_threshold_3: float = 0.50
    reversal_atr_threshold_3: float = 0.50
    mean_reversion_retrace_threshold: float = 0.50

    volatility_expansion_multiple: float = 1.20
    volatility_contraction_multiple: float = 0.80

    favorable_excursion_atr_threshold_6: float = 1.00
    adverse_excursion_atr_threshold_6: float = 1.00


# ============================================================
# FORWARD OUTCOME HELPERS
# ============================================================

def _future_close_return(close: pd.Series, horizon: int) -> pd.Series:
    close = close.astype("float64")
    return close.shift(-horizon) / close - 1.0


def _future_log_return(close: pd.Series, horizon: int) -> pd.Series:
    close = close.astype("float64")
    return np.log(close.shift(-horizon) / close)


def _future_atr_norm_return(close: pd.Series, atr: pd.Series, horizon: int) -> pd.Series:
    close = close.astype("float64")
    safe_atr = atr.astype("float64").replace(0, np.nan)
    return (close.shift(-horizon) - close) / safe_atr


def _future_realized_vol(log_ret_1: pd.Series, horizon: int) -> pd.Series:
    values = log_ret_1.astype("float64").to_numpy()
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        end = i + horizon
        if end >= n:
            continue

        chunk = values[i + 1 : end + 1]
        if np.isnan(chunk).any():
            continue

        out[i] = float(np.sum(chunk ** 2))

    return pd.Series(out, index=log_ret_1.index)


def _future_window_max(high: pd.Series, horizon: int) -> pd.Series:
    values = high.astype("float64").to_numpy()
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        end = i + horizon
        if end >= n:
            continue

        chunk = values[i + 1 : end + 1]
        if np.isnan(chunk).any():
            continue

        out[i] = float(np.max(chunk))

    return pd.Series(out, index=high.index)


def _future_window_min(low: pd.Series, horizon: int) -> pd.Series:
    values = low.astype("float64").to_numpy()
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        end = i + horizon
        if end >= n:
            continue

        chunk = values[i + 1 : end + 1]
        if np.isnan(chunk).any():
            continue

        out[i] = float(np.min(chunk))

    return pd.Series(out, index=low.index)


def _time_to_extreme(values: np.ndarray, mode: str) -> float:
    if len(values) == 0 or np.isnan(values).any():
        return np.nan

    if mode == "max":
        return float(np.argmax(values) + 1)
    if mode == "min":
        return float(np.argmin(values) + 1)

    raise ValueError("mode must be 'max' or 'min'")


def _future_time_to_mfe(high: pd.Series, horizon: int) -> pd.Series:
    values = high.astype("float64").to_numpy()
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        end = i + horizon
        if end >= n:
            continue

        chunk = values[i + 1 : end + 1]
        out[i] = _time_to_extreme(chunk, mode="max")

    return pd.Series(out, index=high.index)


def _future_time_to_mae(low: pd.Series, horizon: int) -> pd.Series:
    values = low.astype("float64").to_numpy()
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        end = i + horizon
        if end >= n:
            continue

        chunk = values[i + 1 : end + 1]
        out[i] = _time_to_extreme(chunk, mode="min")

    return pd.Series(out, index=low.index)


# ============================================================
# OUTCOME ENGINE
# ============================================================

def add_forward_return_outcomes(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype("float64")
    atr = out["atr_14"].astype("float64")

    for h in horizons:
        out[f"fwd_return_{h}"] = _future_close_return(close, h)
        out[f"fwd_log_return_{h}"] = _future_log_return(close, h)
        out[f"fwd_atr_norm_return_{h}"] = _future_atr_norm_return(close, atr, h)

    return out


def add_forward_volatility_outcomes(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype("float64")
    log_ret_1 = np.log(close / close.shift(1))

    for h in horizons:
        out[f"fwd_realized_vol_{h}"] = _future_realized_vol(log_ret_1, h)

    return out


def add_forward_excursion_outcomes(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()

    close = out["close"].astype("float64")
    high = out["high"].astype("float64")
    low = out["low"].astype("float64")
    atr = out["atr_14"].astype("float64").replace(0, np.nan)

    for h in horizons:
        future_max = _future_window_max(high, h)
        future_min = _future_window_min(low, h)

        out[f"fwd_max_high_{h}"] = future_max
        out[f"fwd_min_low_{h}"] = future_min

        out[f"fwd_mfe_{h}"] = (future_max - close) / atr
        out[f"fwd_mae_{h}"] = (future_min - close) / atr

        out[f"time_to_mfe_{h}"] = _future_time_to_mfe(high, h)
        out[f"time_to_mae_{h}"] = _future_time_to_mae(low, h)

        out[f"fwd_mfe_mae_ratio_{h}"] = (
            out[f"fwd_mfe_{h}"] / out[f"fwd_mae_{h}"].abs().replace(0, np.nan)
        )

    return out


def add_directional_labels(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()

    for h in horizons:
        out[f"next_{h}_bar_up"] = np.where(
            out[f"fwd_return_{h}"].isna(),
            np.nan,
            (out[f"fwd_return_{h}"] > 0).astype("float64"),
        )

    return out


def add_path_labels(
    df: pd.DataFrame,
    continuation_atr_threshold_3: float,
    reversal_atr_threshold_3: float,
    mean_reversion_retrace_threshold: float,
    volatility_expansion_multiple: float,
    volatility_contraction_multiple: float,
) -> pd.DataFrame:
    out = df.copy()

    body_direction = out["direction"].astype("float64")
    bar_range = (out["high"] - out["low"]).astype("float64")
    safe_range = bar_range.replace(0, np.nan)

    fwd_atr_norm_3 = out["fwd_atr_norm_return_3"].astype("float64")
    fwd_realized_vol_6 = out["fwd_realized_vol_6"].astype("float64")
    current_realized_vol_6 = out["realized_vol_6"].astype("float64")

    bullish = body_direction > 0
    bearish = body_direction < 0
    neutral = body_direction == 0

    continuation = np.full(len(out), np.nan, dtype=np.float64)
    reversal = np.full(len(out), np.nan, dtype=np.float64)

    valid_fwd_3 = ~fwd_atr_norm_3.isna()

    bullish_valid = bullish & valid_fwd_3
    bearish_valid = bearish & valid_fwd_3

    continuation[bullish_valid] = (
        fwd_atr_norm_3[bullish_valid] > continuation_atr_threshold_3
    ).astype("float64")
    continuation[bearish_valid] = (
        fwd_atr_norm_3[bearish_valid] < -continuation_atr_threshold_3
    ).astype("float64")
    continuation[neutral] = np.nan

    reversal[bullish_valid] = (
        fwd_atr_norm_3[bullish_valid] < -reversal_atr_threshold_3
    ).astype("float64")
    reversal[bearish_valid] = (
        fwd_atr_norm_3[bearish_valid] > reversal_atr_threshold_3
    ).astype("float64")
    reversal[neutral] = np.nan

    out["continuation_label_3"] = continuation
    out["reversal_label_3"] = reversal

    future_max_3 = out["fwd_max_high_3"].astype("float64")
    future_min_3 = out["fwd_min_low_3"].astype("float64")
    close_now = out["close"].astype("float64")

    retrace_down = (close_now - future_min_3) / safe_range
    retrace_up = (future_max_3 - close_now) / safe_range

    mean_reversion = np.full(len(out), np.nan, dtype=np.float64)

    valid_mean_rev = (~future_max_3.isna()) & (~future_min_3.isna()) & (~safe_range.isna())

    bullish_mean_rev = bullish & valid_mean_rev
    bearish_mean_rev = bearish & valid_mean_rev

    mean_reversion[bullish_mean_rev] = (
        retrace_down[bullish_mean_rev] >= mean_reversion_retrace_threshold
    ).astype("float64")
    mean_reversion[bearish_mean_rev] = (
        retrace_up[bearish_mean_rev] >= mean_reversion_retrace_threshold
    ).astype("float64")
    mean_reversion[neutral] = np.nan

    out["mean_reversion_label_3"] = mean_reversion

    valid_vol_compare = (~fwd_realized_vol_6.isna()) & (~current_realized_vol_6.isna())

    out["volatility_expansion_label_6"] = np.where(
        valid_vol_compare,
        (fwd_realized_vol_6 > current_realized_vol_6 * volatility_expansion_multiple).astype("float64"),
        np.nan,
    )

    out["volatility_contraction_label_6"] = np.where(
        valid_vol_compare,
        (fwd_realized_vol_6 < current_realized_vol_6 * volatility_contraction_multiple).astype("float64"),
        np.nan,
    )

    return out


def add_target_before_stop_labels(
    df: pd.DataFrame,
    favorable_excursion_atr_threshold_6: float,
    adverse_excursion_atr_threshold_6: float,
) -> pd.DataFrame:
    out = df.copy()

    fwd_mfe_6 = out["fwd_mfe_6"].astype("float64")
    fwd_mae_6 = out["fwd_mae_6"].astype("float64")
    time_to_mfe_6 = out["time_to_mfe_6"].astype("float64")
    time_to_mae_6 = out["time_to_mae_6"].astype("float64")

    favorable_hit = fwd_mfe_6 >= favorable_excursion_atr_threshold_6
    adverse_hit = fwd_mae_6 <= -adverse_excursion_atr_threshold_6

    target_before_stop = np.full(len(out), np.nan, dtype=np.float64)
    stop_before_target = np.full(len(out), np.nan, dtype=np.float64)

    for i in range(len(out)):
        if np.isnan(fwd_mfe_6.iloc[i]) or np.isnan(fwd_mae_6.iloc[i]):
            continue

        fav = favorable_hit.iloc[i]
        adv = adverse_hit.iloc[i]

        if fav and not adv:
            target_before_stop[i] = 1.0
            stop_before_target[i] = 0.0
        elif adv and not fav:
            target_before_stop[i] = 0.0
            stop_before_target[i] = 1.0
        elif fav and adv:
            if time_to_mfe_6.iloc[i] < time_to_mae_6.iloc[i]:
                target_before_stop[i] = 1.0
                stop_before_target[i] = 0.0
            elif time_to_mae_6.iloc[i] < time_to_mfe_6.iloc[i]:
                target_before_stop[i] = 0.0
                stop_before_target[i] = 1.0
            else:
                target_before_stop[i] = np.nan
                stop_before_target[i] = np.nan
        else:
            target_before_stop[i] = 0.0
            stop_before_target[i] = 0.0

    out["target_before_stop_label_6"] = target_before_stop
    out["stop_before_target_label_6"] = stop_before_target

    return out


# ============================================================
# FINAL BUILD
# ============================================================

def build_memory_outcomes(df_features: pd.DataFrame, config: MemoryOutcomeConfig) -> pd.DataFrame:
    out = df_features.copy()

    out = add_forward_return_outcomes(out, config.horizons)
    out = add_forward_volatility_outcomes(out, config.horizons)
    out = add_forward_excursion_outcomes(out, config.horizons)
    out = add_directional_labels(out, config.horizons)

    out = add_path_labels(
        out,
        continuation_atr_threshold_3=config.continuation_atr_threshold_3,
        reversal_atr_threshold_3=config.reversal_atr_threshold_3,
        mean_reversion_retrace_threshold=config.mean_reversion_retrace_threshold,
        volatility_expansion_multiple=config.volatility_expansion_multiple,
        volatility_contraction_multiple=config.volatility_contraction_multiple,
    )

    out = add_target_before_stop_labels(
        out,
        favorable_excursion_atr_threshold_6=config.favorable_excursion_atr_threshold_6,
        adverse_excursion_atr_threshold_6=config.adverse_excursion_atr_threshold_6,
    )

    return out


def run_memory_outcome_engine(
    feature_config: Optional[MemoryFeatureConfig] = None,
    outcome_config: Optional[MemoryOutcomeConfig] = None,
) -> pd.DataFrame:
    if feature_config is None:
        feature_config = MemoryFeatureConfig()

    if outcome_config is None:
        outcome_config = MemoryOutcomeConfig()

    df_features = run_memory_feature_engine(feature_config)
    df_outcomes = build_memory_outcomes(df_features, outcome_config)

    if outcome_config.output_parquet_path:
        out_path = Path(outcome_config.output_parquet_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_outcomes.to_parquet(out_path, index=False)

    return df_outcomes


if __name__ == "__main__":
    feature_cfg = MemoryFeatureConfig(
        db_path="database/btc_terminal.db",
        table_name="btc_price_1h",
        timestamp_col="timestamp",
        output_parquet_path=None,
    )

    outcome_cfg = MemoryOutcomeConfig(
        horizons=(1, 3, 6, 12),
        output_parquet_path=None,
    )

    df_outcomes = run_memory_outcome_engine(
        feature_config=feature_cfg,
        outcome_config=outcome_cfg,
    )

    print("\n=== MEMORY OUTCOME ENGINE SUCCESS ===")
    print(f"Rows: {len(df_outcomes)}")
    print(f"Columns: {len(df_outcomes.columns)}")

    preview_cols = [
        "timestamp_utc",
        "close",
        "fwd_return_3",
        "fwd_atr_norm_return_3",
        "fwd_mfe_6",
        "fwd_mae_6",
        "continuation_label_3",
        "reversal_label_3",
        "mean_reversion_label_3",
        "target_before_stop_label_6",
    ]

    print("\nLast 5 rows preview:")
    print(df_outcomes[preview_cols].tail(5).to_string(index=False))