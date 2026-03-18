import sqlite3

paths = ["data/btc_terminal.db", "database/btc_terminal.db"]

for path in paths:
    print("\nCHECKING:", path)

    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        print("TABLES:", cur.fetchall())

        try:
            cur.execute("SELECT COUNT(*) FROM btc_price_1h;")
            print("btc_price_1h rows:", cur.fetchone()[0])
        except Exception as e:
            print("btc_price_1h error:", e)

        conn.close()

    except Exception as e:
        print("DB ERROR:", e)