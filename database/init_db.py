import sqlite3
import os

DB_PATH = "database/btc_terminal.db"


def initialize_database():

    # Ensure folder exists
    os.makedirs("database", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS btc_price_1h (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME UNIQUE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    """)

    conn.commit()
    conn.close()

    print("Database initialized successfully.")