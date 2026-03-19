from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from pillar4_candle_engine.absorption_engine import AbsorptionConfig
from pillar4_candle_engine.breakout_quality_engine import BreakoutQualityConfig
from pillar4_candle_engine.candle_features_engine import CandleFeatureConfig, OhlcColumns
from pillar4_candle_engine.candle_intent_engine import CandleIntentConfig
from pillar4_candle_engine.multi_candle_context_engine import MultiCandleContextConfig
from pillar4_candle_engine.pillar4_output import Pillar4Config, run_pillar4_candle_intelligence
from pillar4_candle_engine.pressure_engine import PressureConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "btc_terminal.db"
TABLE_NAME = "btc_price_1h"


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
        breakout_quality=BreakoutQualityConfig(
            range_window=5,
            acceptance_close_threshold=0.55,
            strong_breakout_threshold=0.65,
            weak_breakout_threshold=0.40,
            fake_breakout_overlap_threshold=0.65,
            minimum_breach_threshold=0.0,
        ),
        pressure=PressureConfig(),
    )


def load_real_btc_data(limit: int = 300) -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    query = f"""
        SELECT
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM {TABLE_NAME}
        ORDER BY timestamp DESC
        LIMIT ?
    """

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=(limit,))

    if df.empty:
        raise RuntimeError("No rows were loaded from the real BTC table.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def main() -> None:
    print(f"\nUsing database: {DB_PATH}")
    print(f"Using table   : {TABLE_NAME}\n")

    df = load_real_btc_data(limit=300)

    print("Loaded rows:", len(df))
    print("\nLast 5 raw rows:\n")
    print(df.tail(5).to_string(index=False))

    pillar4_config = build_default_pillar4_config()

    output = run_pillar4_candle_intelligence(
        df=df,
        pillar4_config=pillar4_config,
        columns=OhlcColumns(
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            timestamp="timestamp",
        ),
        asset="BTCUSDT",
        timeframe="1h",
        lookback_bars_used=min(30, len(df)),
        atr_method="wilder",
        pillar3_context=None,
    )

    print("\nPILLAR 4 REAL-DATA OUTPUT\n")
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()