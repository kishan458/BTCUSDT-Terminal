import os
import sqlite3
from typing import Optional, Tuple, List

CANDIDATES = [
    "data/btc_terminal.db",
    "database/btc_terminal.db",
    "btc_terminal.db",
]

def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None

def _row_count(conn: sqlite3.Connection, table: str) -> int:
    cur = conn.cursor()
    return int(cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

def resolve_db_path(require_price_rows: bool = True) -> str:
    """
    Picks the DB that contains btc_price_1h with rows (preferred).
    Falls back to first existing candidate.
    """
    existing: List[str] = [p for p in CANDIDATES if os.path.exists(p)]
    if not existing:
        raise RuntimeError(f"No database file found. Looked for: {CANDIDATES}")

    # Prefer DB that has btc_price_1h with rows
    if require_price_rows:
        for path in existing:
            try:
                conn = sqlite3.connect(path)
                if _table_exists(conn, "btc_price_1h") and _row_count(conn, "btc_price_1h") > 0:
                    conn.close()
                    return path
                conn.close()
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass

    # otherwise return first existing
    return existing[0]

def ensure_macro_events_schema(db_path: Optional[str] = None) -> str:
    """
    Ensures macro_events table exists in the chosen DB.
    Returns the DB path used.
    """
    path = db_path or resolve_db_path(require_price_rows=True)
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS macro_events (
      event_uid TEXT PRIMARY KEY,
      provider TEXT NOT NULL,
      provider_event_id TEXT,
      event_name TEXT NOT NULL,
      event_type TEXT,
      country TEXT,
      scheduled_time_utc TEXT NOT NULL,
      importance TEXT,
      actual REAL,
      forecast REAL,
      previous REAL,
      raw_json TEXT NOT NULL,
      state TEXT NOT NULL DEFAULT 'IDLE',
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()
    return path