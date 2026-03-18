import pandas as pd
import numpy as np


def validate_ohlcv(df: pd.DataFrame):
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError("Missing required OHLCV columns")
    if len(df) < 30:
        raise ValueError("Need at least 30 rows of data")


def find_swing_highs(highs, left=3, right=3):
    idx = []
    for i in range(left, len(highs) - right):
        if highs[i] > max(highs[i - left:i]) and highs[i] >= max(highs[i + 1:i + right + 1]):
            idx.append(i)
    return idx


def find_swing_lows(lows, left=3, right=3):
    idx = []
    for i in range(left, len(lows) - right):
        if lows[i] < min(lows[i - left:i]) and lows[i] <= min(lows[i + 1:i + right + 1]):
            idx.append(i)
    return idx


def classify_structure(latest_high, prev_high, latest_low, prev_low):
    if None in [latest_high, prev_high, latest_low, prev_low]:
        return "MIXED_STRUCTURE"

    if latest_high > prev_high and latest_low > prev_low:
        return "HIGHER_HIGH_HIGHER_LOW"

    if latest_high < prev_high and latest_low < prev_low:
        return "LOWER_HIGH_LOWER_LOW"

    return "MIXED_STRUCTURE"


def compute_atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.rolling(period).mean()


def _to_native(value):
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def run_structure_engine(df: pd.DataFrame):
    validate_ohlcv(df)
    df = df.reset_index(drop=True).copy()

    sh_idx = find_swing_highs(df["high"].values)
    sl_idx = find_swing_lows(df["low"].values)

    swing_highs = [df.loc[i, "high"] for i in sh_idx]
    swing_lows = [df.loc[i, "low"] for i in sl_idx]

    latest_high = swing_highs[-1] if len(swing_highs) >= 1 else None
    prev_high = swing_highs[-2] if len(swing_highs) >= 2 else None

    latest_low = swing_lows[-1] if len(swing_lows) >= 1 else None
    prev_low = swing_lows[-2] if len(swing_lows) >= 2 else None

    structure = classify_structure(latest_high, prev_high, latest_low, prev_low)

    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()

    slope20 = ema20.diff().tail(5).mean()
    slope50 = ema50.diff().tail(5).mean()

    last_close = df["close"].iloc[-1]
    last_ema20 = ema20.iloc[-1]
    last_ema50 = ema50.iloc[-1]

    if ((last_close > last_ema20 > last_ema50) or (last_close < last_ema20 < last_ema50)) and abs(slope20) > 0 and abs(slope50) > 0:
        range_state = "TRENDING"
    else:
        range_state = "RANGING"

    atr5 = compute_atr(df, 5)
    atr20 = compute_atr(df, 20)

    compression_ratio = None
    if atr20.iloc[-1] == 0 or np.isnan(atr5.iloc[-1]) or np.isnan(atr20.iloc[-1]):
        compression_state = "NEUTRAL"
    else:
        compression_ratio = atr5.iloc[-1] / atr20.iloc[-1]

        if compression_ratio < 0.75:
            compression_state = "COMPRESSING"
        elif compression_ratio > 1.25:
            compression_state = "EXPANDING"
        else:
            compression_state = "NEUTRAL"

    range_high = df["high"].tail(20).max()
    range_low = df["low"].tail(20).min()

    return {
        "market_structure": structure,
        "range_state": range_state,
        "compression_state": compression_state,
        "latest_swing_high": _to_native(latest_high),
        "latest_swing_low": _to_native(latest_low),
        "previous_swing_high": _to_native(prev_high),
        "previous_swing_low": _to_native(prev_low),
        "range_high": _to_native(range_high),
        "range_low": _to_native(range_low),
        "compression_ratio": _to_native(compression_ratio)
    }