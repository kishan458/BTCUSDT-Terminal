import pandas as pd
import numpy as np

from pillar3_structure_liquidity_engine.structure_engine import run_structure_engine
from pillar3_structure_liquidity_engine.liquidity_pool_engine import run_liquidity_pool_engine


def _to_native(value):
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def run_stop_hunt_engine(df: pd.DataFrame):
    if len(df) < 30:
        raise ValueError("Need at least 30 rows of data")

    df = df.reset_index(drop=True).copy()

    structure = run_structure_engine(df)
    liquidity = run_liquidity_pool_engine(df)

    current_price = float(df["close"].iloc[-1])

    buy_side = liquidity["buy_side_liquidity"]
    sell_side = liquidity["sell_side_liquidity"]

    compression_ratio = structure["compression_ratio"]
    compression_ratio = float(compression_ratio) if compression_ratio is not None else 1.0

    recent = df.tail(5).copy()
    recent_range = (recent["high"] - recent["low"]).mean()
    recent_body = (recent["close"] - recent["open"]).abs().mean()

    if recent_range == 0 or np.isnan(recent_range):
        body_efficiency = 0.0
    else:
        body_efficiency = float(recent_body / recent_range)

    momentum_exhaustion_score = _clamp(1.0 - body_efficiency)

    if compression_ratio <= 0:
        compression_score = 0.0
    else:
        compression_score = _clamp((1.0 - compression_ratio) / 0.5)

    buy_side_score = 0.0
    sell_side_score = 0.0

    distance_to_buy_pct = None
    distance_to_sell_pct = None

    if buy_side is not None and buy_side > current_price:
        distance_to_buy_pct = ((buy_side - current_price) / current_price) * 100.0
        buy_proximity_score = _clamp(1.0 - (distance_to_buy_pct / 2.0))
        buy_side_score = 0.65 * buy_proximity_score + 0.35 * (
            0.6 * compression_score + 0.4 * momentum_exhaustion_score
        )

    if sell_side is not None and sell_side < current_price:
        distance_to_sell_pct = ((current_price - sell_side) / current_price) * 100.0
        sell_proximity_score = _clamp(1.0 - (distance_to_sell_pct / 2.0))
        sell_side_score = 0.65 * sell_proximity_score + 0.35 * (
            0.6 * compression_score + 0.4 * momentum_exhaustion_score
        )

    stop_hunt_probability = max(buy_side_score, sell_side_score)

    likely_sweep_direction = "NONE"
    sweep_target = None

    if buy_side_score > sell_side_score and buy_side_score > 0:
        likely_sweep_direction = "BUY_SIDE_SWEEP"
        sweep_target = buy_side
    elif sell_side_score > buy_side_score and sell_side_score > 0:
        likely_sweep_direction = "SELL_SIDE_SWEEP"
        sweep_target = sell_side

    return {
        "stop_hunt_probability": _to_native(stop_hunt_probability),
        "likely_sweep_direction": likely_sweep_direction,
        "sweep_target": _to_native(sweep_target),
        "distance_to_buy_side_pct": _to_native(distance_to_buy_pct),
        "distance_to_sell_side_pct": _to_native(distance_to_sell_pct),
        "compression_score": _to_native(compression_score),
        "momentum_exhaustion_score": _to_native(momentum_exhaustion_score),
        "buy_side_stop_hunt_score": _to_native(buy_side_score),
        "sell_side_stop_hunt_score": _to_native(sell_side_score)
    }