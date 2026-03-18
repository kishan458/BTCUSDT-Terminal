import sqlite3

conn = sqlite3.connect("database/btc_terminal.db")
cur = conn.cursor()

cur.execute("PRAGMA table_info(btc_price_1h);")
print("SCHEMA:")
for row in cur.fetchall():
    print(row)

cur.execute("SELECT * FROM btc_price_1h LIMIT 3;")
print("\nSAMPLE ROWS:")
for row in cur.fetchall():
    print(row)

conn.close()