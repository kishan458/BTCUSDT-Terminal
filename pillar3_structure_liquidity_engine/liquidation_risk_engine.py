import pandas as pd
import numpy as np


def _to_native(value):
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _label_risk(score):
    if score >= 0.7:
        return "HIGH"
    if score >= 0.4:
        return "MODERATE"
    return "LOW"


def run_liquidation_risk_engine(df: pd.DataFrame):
    if len(df) < 30:
        raise ValueError("Need at least 30 rows of data")

    df = df.reset_index(drop=True).copy()

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    current_price = float(close.iloc[-1])
    last_ema20 = float(ema20.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])

    returns = close.pct_change().fillna(0.0)
    recent_returns = returns.tail(10)

    realized_vol = float(recent_returns.std())
    vol_shock_score = _clamp(realized_vol * 100)

    stretch_up_score = _clamp(
        (0.6 * max(current_price - last_ema20, 0) / max(last_ema20, 1e-9) +
         0.4 * max(current_price - last_ema50, 0) / max(last_ema50, 1e-9)) * 10
    )

    stretch_down_score = _clamp(
        (0.6 * max(last_ema20 - current_price, 0) / max(last_ema20, 1e-9) +
         0.4 * max(last_ema50 - current_price, 0) / max(last_ema50, 1e-9)) * 10
    )

    recent_close_diff = close.diff().tail(5).fillna(0.0)
    upside_acceleration = float(recent_close_diff[recent_close_diff > 0].sum())
    downside_acceleration = float(abs(recent_close_diff[recent_close_diff < 0].sum()))

    avg_price = max(float(close.tail(10).mean()), 1e-9)
    upside_acceleration_score = _clamp((upside_acceleration / avg_price) * 20)
    downside_acceleration_score = _clamp((downside_acceleration / avg_price) * 20)

    recent_candles = df.tail(8).copy()
    up_closes = int((recent_candles["close"] > recent_candles["open"]).sum())
    down_closes = int((recent_candles["close"] < recent_candles["open"]).sum())

    one_sided_up_score = _clamp(up_closes / 8)
    one_sided_down_score = _clamp(down_closes / 8)

    recent_range = (high.tail(8) - low.tail(8)).replace(0, np.nan)
    recent_body = (close.tail(8) - df["open"].tail(8)).abs()
    body_efficiency = float((recent_body / recent_range).fillna(0.0).mean())

    thin_pullback_score = _clamp(body_efficiency)

    short_liquidation_score = _clamp(
        0.30 * stretch_up_score +
        0.25 * upside_acceleration_score +
        0.20 * vol_shock_score +
        0.15 * one_sided_up_score +
        0.10 * thin_pullback_score
    )

    long_liquidation_score = _clamp(
        0.30 * stretch_down_score +
        0.25 * downside_acceleration_score +
        0.20 * vol_shock_score +
        0.15 * one_sided_down_score +
        0.10 * thin_pullback_score
    )

    cascade_probability = _clamp(max(short_liquidation_score, long_liquidation_score))

    return {
        "long_liquidation_risk": _label_risk(long_liquidation_score),
        "short_liquidation_risk": _label_risk(short_liquidation_score),
        "long_liquidation_score": _to_native(long_liquidation_score),
        "short_liquidation_score": _to_native(short_liquidation_score),
        "cascade_probability": _to_native(cascade_probability),
        "volatility_shock_score": _to_native(vol_shock_score),
        "stretch_up_score": _to_native(stretch_up_score),
        "stretch_down_score": _to_native(stretch_down_score),
        "upside_acceleration_score": _to_native(upside_acceleration_score),
        "downside_acceleration_score": _to_native(downside_acceleration_score)
    }