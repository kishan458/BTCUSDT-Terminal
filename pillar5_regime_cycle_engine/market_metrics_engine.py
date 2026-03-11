import sqlite3
import math
import pandas as pd

DB_PATH = "database/btc_terminal.db"


def _load_price_data(limit: int = 400) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql(
        f"""
        SELECT timestamp, open, high, low, close, volume
        FROM btc_price_1h
        ORDER BY timestamp DESC
        LIMIT {limit}
        """,
        conn,
    )

    conn.close()

    if df.empty:
        raise ValueError("No BTC price data found in btc_price_1h")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _realized_vol(df: pd.DataFrame, window: int = 24) -> pd.Series:
    log_ret = (df["close"] / df["close"].shift(1)).apply(
        lambda x: math.log(x) if pd.notna(x) and x > 0 else None
    )
    return log_ret.rolling(window).std()


def _rolling_return(series: pd.Series, periods: int) -> float | None:
    if len(series) <= periods:
        return None
    prev = series.iloc[-(periods + 1)]
    curr = series.iloc[-1]
    if prev == 0 or pd.isna(prev) or pd.isna(curr):
        return None
    return float((curr / prev) - 1.0)


def _swing_points(df: pd.DataFrame, lookback: int = 20) -> tuple[float | None, float | None]:
    if len(df) < lookback:
        return None, None

    recent = df.tail(lookback)
    swing_high = float(recent["high"].max())
    swing_low = float(recent["low"].min())

    return swing_high, swing_low


def _ma_order(close: float, ema20: float, ema50: float, ema200: float) -> str:
    if close > ema20 > ema50 > ema200:
        return "BULLISH_STACKED"
    if close < ema20 < ema50 < ema200:
        return "BEARISH_STACKED"
    return "MIXED"


def _price_vs_ma(close: float, ma: float) -> str:
    if close > ma:
        return "ABOVE"
    if close < ma:
        return "BELOW"
    return "AT"


def _distance_pct(price: float, ref: float) -> float | None:
    if ref == 0 or pd.isna(ref):
        return None
    return float((price - ref) / ref)


def build_market_metrics() -> dict:
    df = _load_price_data()

    df["ema_20"] = _ema(df["close"], 20)
    df["ema_50"] = _ema(df["close"], 50)
    df["ema_200"] = _ema(df["close"], 200)
    df["atr_14"] = _atr(df, 14)
    df["realized_vol_24"] = _realized_vol(df, 24)

    latest = df.iloc[-1]

    close_series = df["close"]
    high_series = df["high"]
    low_series = df["low"]

    latest_close = float(latest["close"])
    latest_atr = float(latest["atr_14"]) if pd.notna(latest["atr_14"]) else None
    latest_realized_vol = (
        float(latest["realized_vol_24"]) if pd.notna(latest["realized_vol_24"]) else None
    )

    atr_pct = (latest_atr / latest_close) if latest_atr is not None and latest_close != 0 else None

    atr_series = df["atr_14"].dropna()
    vol_percentile = None
    if len(atr_series) > 10 and latest_atr is not None:
        vol_percentile = float((atr_series <= latest_atr).mean())

    ema20 = float(latest["ema_20"])
    ema50 = float(latest["ema_50"])
    ema200 = float(latest["ema_200"])

    ema20_slope = None
    ema50_slope = None
    if len(df) >= 6:
        prev_ema20 = float(df.iloc[-6]["ema_20"])
        prev_ema50 = float(df.iloc[-6]["ema_50"])
        ema20_slope = float((ema20 - prev_ema20) / 5.0)
        ema50_slope = float((ema50 - prev_ema50) / 5.0)

    roc = _rolling_return(close_series, 12)

    momentum_score = None
    if roc is not None and ema20_slope is not None and latest_close != 0:
        slope_component = ema20_slope / latest_close
        momentum_score = float(max(-1.0, min(1.0, (roc * 5.0) + (slope_component * 100.0))))

    range_24 = float(high_series.tail(24).max() - low_series.tail(24).min()) if len(df) >= 24 else None
    range_72 = float(high_series.tail(72).max() - low_series.tail(72).min()) if len(df) >= 72 else None

    range_compression_score = None
    expansion_score = None
    breakout_pressure_score = None

    if range_24 is not None and range_72 is not None and range_72 != 0:
        range_compression_score = float(max(0.0, min(1.0, 1.0 - (range_24 / range_72))))
        expansion_score = float(max(0.0, min(1.0, range_24 / range_72)))
        breakout_pressure_score = float((range_compression_score + expansion_score) / 2.0)

    swing_high, swing_low = _swing_points(df, 20)

    structure_state = "UNKNOWN"
    if len(df) >= 40 and swing_high is not None and swing_low is not None:
        prev_block = df.iloc[-40:-20]
        prev_high = float(prev_block["high"].max())
        prev_low = float(prev_block["low"].min())

        if swing_high > prev_high and swing_low > prev_low:
            structure_state = "HIGHER_HIGH_HIGHER_LOW"
        elif swing_high < prev_high and swing_low < prev_low:
            structure_state = "LOWER_HIGH_LOWER_LOW"
        elif swing_high > prev_high and swing_low <= prev_low:
            structure_state = "HIGHER_HIGH_LOWER_LOW"
        elif swing_high <= prev_high and swing_low > prev_low:
            structure_state = "LOWER_HIGH_HIGHER_LOW"
        else:
            structure_state = "MIXED_STRUCTURE"

    return {
        "timestamp_utc": latest["timestamp"],
        "market_metrics": {
            "ohlcv": {
                "open": float(latest["open"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
                "close": latest_close,
                "volume": float(latest["volume"]),
            },
            "returns": {
                "return_1bar": _rolling_return(close_series, 1),
                "return_4bar": _rolling_return(close_series, 4),
                "return_24bar": _rolling_return(close_series, 24),
                "return_7d": _rolling_return(close_series, 24 * 7),
            },
            "volatility": {
                "atr": latest_atr,
                "atr_pct": atr_pct,
                "realized_vol": latest_realized_vol,
                "volatility_percentile": vol_percentile,
            },
            "moving_average_structure": {
                "ema_20": ema20,
                "ema_50": ema50,
                "ema_200": ema200,
                "ma_order": _ma_order(latest_close, ema20, ema50, ema200),
                "price_vs_ema20": _price_vs_ma(latest_close, ema20),
                "price_vs_ema50": _price_vs_ma(latest_close, ema50),
                "price_vs_ema200": _price_vs_ma(latest_close, ema200),
            },
            "momentum": {
                "ema20_slope": ema20_slope,
                "ema50_slope": ema50_slope,
                "roc": roc,
                "momentum_score": momentum_score,
            },
            "compression_expansion": {
                "range_compression_score": range_compression_score,
                "expansion_score": expansion_score,
                "breakout_pressure_score": breakout_pressure_score,
            },
            "swing_structure": {
                "latest_swing_high": swing_high,
                "latest_swing_low": swing_low,
                "structure_state": structure_state,
            },
            "distance_from_key_mas": {
                "distance_to_ema20_pct": _distance_pct(latest_close, ema20),
                "distance_to_ema50_pct": _distance_pct(latest_close, ema50),
                "distance_to_ema200_pct": _distance_pct(latest_close, ema200),
            },
        },
    }