from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from pillar4_candle_engine.absorption_engine import AbsorptionConfig
from pillar4_candle_engine.breakout_quality_engine import BreakoutQualityConfig
from pillar4_candle_engine.candle_features_engine import CandleFeatureConfig, OhlcColumns
from pillar4_candle_engine.candle_intent_engine import CandleIntentConfig
from pillar4_candle_engine.multi_candle_context_engine import MultiCandleContextConfig
from pillar4_candle_engine.pillar4_output import (
    Pillar4Config,
    run_pillar4_candle_intelligence,
)
from pillar4_candle_engine.pressure_engine import PressureConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_candidate_databases() -> List[Path]:
    candidates = [
        PROJECT_ROOT / "data" / "btc_terminal.db",
        PROJECT_ROOT / "database" / "btc_terminal.db",
    ]
    return [p for p in candidates if p.exists()]


def list_tables(conn: sqlite3.Connection) -> List[str]:
    query = """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        ORDER BY name
    """
    rows = conn.execute(query).fetchall()
    return [row[0] for row in rows]


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [row[1] for row in rows]


def score_ohlc_table(columns: List[str]) -> int:
    cols = {c.lower() for c in columns}
    score = 0

    required_groups = [
        {"open"},
        {"high"},
        {"low"},
        {"close"},
    ]
    for group in required_groups:
        if cols & group:
            score += 3

    optional_hits = {
        "timestamp",
        "timestamp_utc",
        "time",
        "datetime",
        "date",
        "volume",
        "symbol",
        "asset",
        "pair",
        "interval",
        "timeframe",
    }
    score += sum(1 for c in cols if c in optional_hits)

    return score


def detect_best_ohlc_table(conn: sqlite3.Connection) -> Tuple[str, List[str]]:
    tables = list_tables(conn)
    if not tables:
        raise RuntimeError("No tables found in the SQLite database.")

    scored = []
    for table in tables:
        cols = get_table_columns(conn, table)
        scored.append((score_ohlc_table(cols), table, cols))

    scored.sort(reverse=True, key=lambda x: x[0])

    best_score, best_table, best_cols = scored[0]
    if best_score < 12:
        raise RuntimeError(
            f"Could not confidently detect an OHLC table. Best guess was '{best_table}' with columns {best_cols}"
        )

    return best_table, best_cols


def detect_column_mapping(columns: List[str]) -> OhlcColumns:
    lower_to_actual = {c.lower(): c for c in columns}

    def pick(*names: str, required: bool = True) -> Optional[str]:
        for name in names:
            if name in lower_to_actual:
                return lower_to_actual[name]
        if required:
            raise KeyError(f"Could not find any of these required columns: {names}")
        return None

    timestamp_col = None
    for candidate in ("timestamp", "timestamp_utc", "time", "datetime", "date"):
        if candidate in lower_to_actual:
            timestamp_col = lower_to_actual[candidate]
            break

    if timestamp_col is None:
        for actual_col in columns:
            if "time" in actual_col.lower():
                timestamp_col = actual_col
                break

    return OhlcColumns(
        open=pick("open"),
        high=pick("high"),
        low=pick("low"),
        close=pick("close"),
        volume=pick("volume", required=False),
        timestamp=timestamp_col,
    )


def maybe_detect_symbol_column(columns: List[str]) -> Optional[str]:
    lower_to_actual = {c.lower(): c for c in columns}
    for candidate in ("symbol", "asset", "pair", "ticker"):
        if candidate in lower_to_actual:
            return lower_to_actual[candidate]
    return None


def maybe_detect_timeframe_column(columns: List[str]) -> Optional[str]:
    lower_to_actual = {c.lower(): c for c in columns}
    for candidate in ("timeframe", "interval"):
        if candidate in lower_to_actual:
            return lower_to_actual[candidate]
    return None


def choose_database_with_price_data(db_paths: List[Path]) -> Path:
    for path in db_paths:
        with sqlite3.connect(path) as conn:
            tables = list_tables(conn)
            if "btc_price_1h" not in tables:
                continue
            count = conn.execute("SELECT COUNT(*) FROM btc_price_1h").fetchone()[0]
            if count and count > 0:
                return path
    raise RuntimeError("No database with valid btc_price_1h data found.")


def load_latest_ohlc_data(
    conn: sqlite3.Connection,
    table_name: str,
    columns: OhlcColumns,
    limit: int = 300,
    symbol_filter: Optional[str] = None,
    timeframe_filter: Optional[str] = None,
    all_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    if all_columns is None:
        all_columns = get_table_columns(conn, table_name)

    symbol_col = maybe_detect_symbol_column(all_columns)
    timeframe_col = maybe_detect_timeframe_column(all_columns)

    selected_cols = [columns.open, columns.high, columns.low, columns.close]
    if columns.volume:
        selected_cols.append(columns.volume)
    if columns.timestamp:
        selected_cols.append(columns.timestamp)
    if symbol_col and symbol_col not in selected_cols:
        selected_cols.append(symbol_col)
    if timeframe_col and timeframe_col not in selected_cols:
        selected_cols.append(timeframe_col)

    where_parts = []
    params: List[object] = []

    if symbol_filter and symbol_col:
        where_parts.append(f"{symbol_col} = ?")
        params.append(symbol_filter)

    if timeframe_filter and timeframe_col:
        where_parts.append(f"{timeframe_col} = ?")
        params.append(timeframe_filter)

    where_sql = ""
    if where_parts:
        where_sql = "WHERE " + " AND ".join(where_parts)

    order_col = columns.timestamp if columns.timestamp else "rowid"

    query = f"""
        SELECT {", ".join(selected_cols)}
        FROM {table_name}
        {where_sql}
        ORDER BY {order_col} DESC
        LIMIT ?
    """
    params.append(limit)

    df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        raise RuntimeError("Query returned no rows from the detected OHLC table.")

    if columns.timestamp and columns.timestamp in df.columns:
        df[columns.timestamp] = pd.to_datetime(df[columns.timestamp], errors="coerce")
        df = df.sort_values(columns.timestamp).reset_index(drop=True)
    else:
        df = df.iloc[::-1].reset_index(drop=True)

    return df


def build_default_pillar4_config() -> Pillar4Config:
    return Pillar4Config(
        candle_features=CandleFeatureConfig(
            atr_window=14,
            range_mean_window=20,
            body_mean_window=20,
            zscore_window=20,
            overlap_window_short=3,
            overlap_window_long=5,
            progress_window_short=3,
            progress_window_medium=5,
            progress_window_long=8,
            persistence_window_short=3,
            persistence_window_medium=5,
            persistence_window_long=8,
            realized_vol_window=20,
            volatility_percentile_window=50,
            range_percentile_window=50,
            body_percentile_window=50,
            contraction_window=5,
            inside_outside_window=5,
            entropy_window=5,
            rolling_wick_window=5,
        ),
        candle_intent=CandleIntentConfig(),
        multi_candle_context=MultiCandleContextConfig(),
        absorption=AbsorptionConfig(),
        breakout_quality=BreakoutQualityConfig(),
        pressure=PressureConfig(),
    )


def main() -> None:
    db_paths = find_candidate_databases()
    if not db_paths:
        raise FileNotFoundError(
            "Could not find btc_terminal.db in ./data or ./database"
        )

    db_path = choose_database_with_price_data(db_paths)
    print(f"\nUsing database: {db_path}\n")

    with sqlite3.connect(db_path) as conn:
        tables = list_tables(conn)
        print("Tables found:")
        for t in tables:
            print(f" - {t}")

        table_name, table_columns = detect_best_ohlc_table(conn)
        print(f"\nDetected OHLC table: {table_name}")
        print(f"Columns: {table_columns}\n")

        columns = detect_column_mapping(table_columns)
        print("Detected column mapping:")
        print(f" open      -> {columns.open}")
        print(f" high      -> {columns.high}")
        print(f" low       -> {columns.low}")
        print(f" close     -> {columns.close}")
        print(f" volume    -> {columns.volume}")
        print(f" timestamp -> {columns.timestamp}\n")

        df = load_latest_ohlc_data(
            conn=conn,
            table_name=table_name,
            columns=columns,
            limit=300,
            symbol_filter="BTCUSDT",
            timeframe_filter=None,
            all_columns=table_columns,
        )

    print("Loaded rows:", len(df))
    print("\nLast 5 raw rows:\n")
    print(df.tail(5).to_string(index=False))

    pillar4_config = build_default_pillar4_config()

    output = run_pillar4_candle_intelligence(
        df=df,
        pillar4_config=pillar4_config,
        columns=columns,
        asset="BTCUSDT",
        timeframe="AUTO",
        lookback_bars_used=min(30, len(df)),
        atr_method="wilder",
        pillar3_context=None,
    )

    print("\nPILLAR 4 OUTPUT ENGINE RAN SUCCESSFULLY\n")
    print("TOP-LEVEL KEYS:\n")
    print(list(output.keys()))

    print("\nCANDLE SUMMARY:\n")
    print(json.dumps(output["candle_summary"], indent=2))

    print("\nABSORPTION:\n")
    print(json.dumps(output["absorption"], indent=2))

    print("\nBREAKOUT ANALYSIS:\n")
    print(json.dumps(output["breakout_analysis"], indent=2))

    print("\nPRESSURE:\n")
    print(json.dumps(output["pressure"], indent=2))

    print("\nRISK FLAGS:\n")
    print(output["risk_flags"])

    print("\nFULL FINAL OUTPUT:\n")
    print(json.dumps(output, indent=2, default=str))

    required_top_keys = [
        "asset",
        "timestamp_utc",
        "timeframe",
        "lookback_bars_used",
        "candle_summary",
        "latest_candle_features",
        "volatility_context",
        "multi_candle_context",
        "intent_scores",
        "absorption",
        "breakout_analysis",
        "pressure",
        "sequence_similarity",
        "context_alignment",
        "risk_flags",
        "diagnostics",
        "ai_overview",
    ]

    missing = [key for key in required_top_keys if key not in output]
    print("\nCHECKING REQUIRED TOP-LEVEL KEYS:\n")
    if missing:
        print("Missing keys:", missing)
    else:
        print("All required top-level keys are present.")


if __name__ == "__main__":
    main()