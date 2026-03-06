import sqlite3

DB_PATH = "database/btc_terminal.db"


def initialize_feature_table():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS btc_features_1h (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME UNIQUE,
            sma_50 REAL,
            sma_200 REAL,
            ema_21 REAL,
            rsi_14 REAL
        )
    """)

    conn.commit()
    conn.close()

    print("Feature table initialized.")