import requests
import sqlite3
from datetime import datetime

BINANCE_URL = "https://api.binance.com/api/v3/klines"
DB_PATH = "database/btc_terminal.db"


def fetch_1h_btc_batch(end_time=None, limit=1000):

    params = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "limit": limit
    }

    if end_time:
        params["endTime"] = end_time

    response = requests.get(BINANCE_URL, params=params)

    if response.status_code != 200:
        raise Exception(f"Binance API error: {response.text}")

    data = response.json()

    if not data:
        return []

    print(f"Fetched {len(data)} candles.")

    return data


def store_candles(candles):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    inserted = 0

    for c in candles:
        open_time = datetime.utcfromtimestamp(c[0] / 1000)
        open_price = float(c[1])
        high = float(c[2])
        low = float(c[3])
        close = float(c[4])
        volume = float(c[5])

        try:
            cursor.execute("""
                INSERT INTO btc_price_1h 
                (timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (open_time, open_price, high, low, close, volume))

            inserted += 1

        except sqlite3.IntegrityError:
            # Duplicate timestamp — skip
            pass

    conn.commit()
    conn.close()

    print(f"Inserted {inserted} new candles.")