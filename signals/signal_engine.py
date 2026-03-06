import sqlite3
import pandas as pd

DB_PATH = "data/btc_terminal.db"

def get_latest_features():
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT * FROM btc_features_1h
        ORDER BY timestamp DESC
        LIMIT 1
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df.iloc[0]

def generate_signal():
    row = get_latest_features()

    sma_50 = row["sma_50"]
    sma_200 = row["sma_200"]
    rsi = row["rsi_14"]
    ema = row["ema_21"]
    close = row["close"]

    # --- Regime Detection ---
    if sma_50 > sma_200:
        regime = "BULL"
    else:
        regime = "BEAR"

    # --- Momentum Confirmation ---
    if regime == "BULL" and rsi > 55 and close > ema:
        signal = "LONG"
    elif regime == "BEAR" and rsi < 45 and close < ema:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    return {
        "regime": regime,
        "signal": signal,
        "close": close,
        "rsi": rsi
    }