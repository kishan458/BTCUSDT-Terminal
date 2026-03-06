import sqlite3

DB_PATH = "data/btc_terminal.db"

def upsert_features(df):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO btc_features_1h 
            (timestamp, close, sma_50, sma_200, ema_21, rsi_14)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            float(row["close"]),
            float(row["sma_50"]) if row["sma_50"] is not None else None,
            float(row["sma_200"]) if row["sma_200"] is not None else None,
            float(row["ema_21"]) if row["ema_21"] is not None else None,
            float(row["rsi_14"]) if row["rsi_14"] is not None else None
        ))

    conn.commit()
    conn.close()