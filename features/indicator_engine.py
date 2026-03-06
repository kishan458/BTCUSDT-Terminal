import sqlite3
import pandas as pd

DB_PATH = "database/btc_terminal.db"


def load_price_data():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        "SELECT timestamp, close FROM btc_price_1h ORDER BY timestamp ASC",
        conn
    )

    conn.close()

    # Convert types
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # DROP ANY hidden duplicates
    df = df.drop_duplicates(subset="timestamp")

    # CRITICAL: force clean integer index
    df = df.reset_index(drop=True)

    # Ensure strictly sorted again
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def compute_indicators(df):

    df = df.copy()

    # ===== SMA =====
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()

    # ===== EMA =====
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()

    # ===== RSI =====
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    return df