import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

from core.db import resolve_db_path
DB_PATH = "database/btc_terminal.db"

UTC = ZoneInfo("UTC")


def _get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cur = conn.cursor()
    rows = cur.execute(f"PRAGMA table_info({table});").fetchall()
    return [r[1] for r in rows]


def _detect_column(cols: list[str], candidates: list[str]) -> str | None:
    colset = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in colset:
            return colset[cand.lower()]
    return None


def _parse_ts_series(s: pd.Series) -> pd.Series:
    """
    Robust parse for TEXT timestamps:
    - Handles numeric-in-text unix seconds/ms
    - Handles ISO8601 with Z / T / milliseconds
    - Handles "YYYY-MM-DD HH:MM:SS"
    """
    # strip and normalize
    s_str = s.astype(str).str.strip()

    # if it's numeric-in-text, parse as unix (ms or s)
    s_num = pd.to_numeric(s_str, errors="coerce")
    if s_num.notna().mean() > 0.8:
        med = float(s_num.dropna().median())
        if med > 1e11:
            return pd.to_datetime(s_num, unit="ms", utc=True, errors="coerce").dt.tz_convert(None)
        elif med > 1e8:
            return pd.to_datetime(s_num, unit="s", utc=True, errors="coerce").dt.tz_convert(None)

    # Normalize common ISO variants:
    # "2026-03-02T20:00:00Z" -> "2026-03-02 20:00:00"
    # "2026-03-02T20:00:00.000Z" -> "2026-03-02 20:00:00.000"
    s_norm = s_str.str.replace("T", " ", regex=False)
    s_norm = s_norm.str.replace("Z", "", regex=False)

    # First attempt: pandas general parser
    ts = pd.to_datetime(s_norm, errors="coerce", utc=False)

    # If still many NaT, try forcing UTC parse (helps some ISO strings)
    if ts.notna().mean() < 0.5:
        ts = pd.to_datetime(s_str, errors="coerce", utc=True).dt.tz_convert(None)

    return ts


def _load_price_series(hours_back: int = 24 * 365 * 2) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    cols = _get_table_columns(conn, "btc_price_1h")

    ts_col = _detect_column(
        cols,
        ["timestamp", "timestamp_utc", "open_time", "openTime", "time", "datetime", "date", "open_time_utc"]
    )
    close_col = _detect_column(
        cols,
        ["close", "Close", "close_price", "closePrice", "closing_price"]
    )

    if ts_col is None or close_col is None:
        conn.close()
        raise RuntimeError(
            f"btc_price_1h schema mismatch.\nFound columns: {cols}"
        )

    df = pd.read_sql(
        f"""
        SELECT {ts_col} AS ts, {close_col} AS close
        FROM btc_price_1h
        ORDER BY {ts_col} ASC
        """,
        conn,
    )
    conn.close()

    df["ts"] = _parse_ts_series(df["ts"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["ts", "close"]).reset_index(drop=True)

    if df.empty:
        # Show a sample to debug format immediately
        raise RuntimeError(
            "Loaded btc_price_1h but dataframe is empty after parsing.\n"
            f"Detected columns: ts_col={ts_col}, close_col={close_col}.\n"
            "Sample raw timestamps (first 5): "
            + str(pd.read_sql("SELECT timestamp_utc FROM btc_price_1h LIMIT 5", sqlite3.connect(DB_PATH))["timestamp_utc"].tolist())
        )

    if len(df) > hours_back:
        df = df.iloc[-hours_back:].reset_index(drop=True)

    return df


def _percentile_rank(series: pd.Series, value: float) -> float:
    s = series.dropna()
    if len(s) < 200:
        return 0.5
    return float((s <= value).mean())


def compute_base_uncertainty(event_time_utc_str: str) -> dict:
    df = _load_price_series()

    df["ret"] = df["close"].pct_change()

    df["vol_24h"] = df["ret"].rolling(24, min_periods=24).std()
    df["vol_7d"] = df["ret"].rolling(24 * 7, min_periods=24 * 7).std()
    df["vol_ratio"] = df["vol_24h"] / df["vol_7d"]

    latest = df.iloc[-1]

    if pd.isna(latest["vol_24h"]) or pd.isna(latest["vol_7d"]) or pd.isna(latest["vol_ratio"]):
        return {
            "base_uncertainty": 0.5,
            "components": {
                "note": "Insufficient history for volatility windows; neutral uncertainty returned."
            },
        }

    vol_24h = float(latest["vol_24h"])
    vol_7d = float(latest["vol_7d"])
    vol_ratio = float(latest["vol_ratio"])

    now = datetime.now(tz=UTC)
    event_time = datetime.strptime(event_time_utc_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    minutes_to_event = (event_time - now).total_seconds() / 60.0

    if minutes_to_event <= 0:
        time_score = 1.0
    else:
        time_score = float(1.0 / (1.0 + (minutes_to_event / 1440.0)))

    p_vol24 = _percentile_rank(df["vol_24h"], vol_24h)
    p_ratio = _percentile_rank(df["vol_ratio"], vol_ratio)

    base_uncertainty = 0.45 * p_vol24 + 0.35 * p_ratio + 0.20 * time_score
    base_uncertainty = max(0.0, min(1.0, float(base_uncertainty)))

    return {
        "base_uncertainty": base_uncertainty,
        "components": {
            "vol_24h": vol_24h,
            "vol_7d": vol_7d,
            "vol_ratio": vol_ratio,
            "pctl_vol_24h": p_vol24,
            "pctl_vol_ratio": p_ratio,
            "minutes_to_event": minutes_to_event,
            "time_score": time_score,
        },
    }