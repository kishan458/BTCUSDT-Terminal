import pandas as pd
import numpy as np

from pillar3_structure_liquidity_engine.structure_engine import run_structure_engine
from pillar3_structure_liquidity_engine.liquidity_pool_engine import run_liquidity_pool_engine
from pillar3_structure_liquidity_engine.stop_hunt_engine import run_stop_hunt_engine


def _to_native(value):
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def run_trap_detection_engine(df: pd.DataFrame):
    if len(df) < 30:
        raise ValueError("Need at least 30 rows of data")

    df = df.reset_index(drop=True).copy()

    structure = run_structure_engine(df)
    liquidity = run_liquidity_pool_engine(df)
    stop_hunt = run_stop_hunt_engine(df)

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    current_price = float(close.iloc[-1])

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    last_ema20 = float(ema20.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])

    buy_side = liquidity["buy_side_liquidity"]
    sell_side = liquidity["sell_side_liquidity"]

    compression_ratio = structure["compression_ratio"]
    compression_ratio = float(compression_ratio) if compression_ratio is not None else 1.0

    recent = df.tail(5).copy()
    recent_range = (recent["high"] - recent["low"]).replace(0, np.nan)
    recent_body = (recent["close"] - recent["open"]).abs()
    close_quality = float((recent_body / recent_range).fillna(0.0).mean())

    failure_risk_score = _clamp(1.0 - close_quality)

    upside_extension = (
        0.6 * max(current_price - last_ema20, 0.0) / max(last_ema20, 1e-9)
        + 0.4 * max(current_price - last_ema50, 0.0) / max(last_ema50, 1e-9)
    )
    downside_extension = (
        0.6 * max(last_ema20 - current_price, 0.0) / max(last_ema20, 1e-9)
        + 0.4 * max(last_ema50 - current_price, 0.0) / max(last_ema50, 1e-9)
    )

    breakout_extension_score = _clamp(upside_extension * 10.0)
    breakdown_extension_score = _clamp(downside_extension * 10.0)

    compression_score = 0.0
    if compression_ratio > 0:
        compression_score = _clamp((1.0 - compression_ratio) / 0.5)

    stop_hunt_probability = float(stop_hunt["stop_hunt_probability"] or 0.0)
    likely_sweep_direction = stop_hunt["likely_sweep_direction"]

    recent_up_closes = int((recent["close"] > recent["open"]).sum())
    recent_down_closes = int((recent["close"] < recent["open"]).sum())

    bullish_pressure_score = _clamp(recent_up_closes / 5.0)
    bearish_pressure_score = _clamp(recent_down_closes / 5.0)

    breakout_liquidity_proximity = 0.0
    breakdown_liquidity_proximity = 0.0

    if buy_side is not None and buy_side > current_price:
        distance_to_buy_pct = ((buy_side - current_price) / current_price) * 100.0
        breakout_liquidity_proximity = _clamp(1.0 - (distance_to_buy_pct / 2.0))

    if sell_side is not None and sell_side < current_price:
        distance_to_sell_pct = ((current_price - sell_side) / current_price) * 100.0
        breakdown_liquidity_proximity = _clamp(1.0 - (distance_to_sell_pct / 2.0))

    breakout_sweep_bias = 1.0 if likely_sweep_direction == "BUY_SIDE_SWEEP" else 0.0
    breakdown_sweep_bias = 1.0 if likely_sweep_direction == "SELL_SIDE_SWEEP" else 0.0

    breakout_trap_probability = _clamp(
        0.25 * stop_hunt_probability +
        0.20 * compression_score +
        0.20 * breakout_extension_score +
        0.15 * failure_risk_score +
        0.10 * breakout_liquidity_proximity +
        0.10 * bullish_pressure_score
    ) * (0.75 + 0.25 * breakout_sweep_bias)

    breakdown_trap_probability = _clamp(
        0.25 * stop_hunt_probability +
        0.20 * compression_score +
        0.20 * breakdown_extension_score +
        0.15 * failure_risk_score +
        0.10 * breakdown_liquidity_proximity +
        0.10 * bearish_pressure_score
    ) * (0.75 + 0.25 * breakdown_sweep_bias)

    breakout_trap_probability = _clamp(breakout_trap_probability)
    breakdown_trap_probability = _clamp(breakdown_trap_probability)

    likely_trap_side = "NO_CLEAR_TRAP"
    if breakout_trap_probability > breakdown_trap_probability and breakout_trap_probability >= 0.5:
        likely_trap_side = "LONG_TRAP"
    elif breakdown_trap_probability > breakout_trap_probability and breakdown_trap_probability >= 0.5:
        likely_trap_side = "SHORT_TRAP"

    return {
        "breakout_trap_probability": _to_native(breakout_trap_probability),
        "breakdown_trap_probability": _to_native(breakdown_trap_probability),
        "likely_trap_side": likely_trap_side,
        "failure_risk_score": _to_native(failure_risk_score),
        "breakout_extension_score": _to_native(breakout_extension_score),
        "breakdown_extension_score": _to_native(breakdown_extension_score),
        "compression_score": _to_native(compression_score),
        "stop_hunt_probability": _to_native(stop_hunt_probability)
    }