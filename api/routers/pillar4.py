"""
Pillar 4 — Candle Intelligence router
GET /api/pillar4
GET /api/pillar4/refresh
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter
from api.cache import cache

router = APIRouter()

DB_PATH = str(Path(__file__).resolve().parent.parent.parent / "database" / "btc_terminal.db")


def _load_ohlcv() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            "SELECT timestamp, open, high, low, close, volume FROM btc_price_1h ORDER BY timestamp DESC LIMIT 300",
            conn,
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)


def _run_fresh() -> Dict[str, Any]:
    from pillar4_candle_engine.absorption_engine import AbsorptionConfig
    from pillar4_candle_engine.breakout_quality_engine import BreakoutQualityConfig
    from pillar4_candle_engine.candle_features_engine import CandleFeatureConfig, OhlcColumns
    from pillar4_candle_engine.candle_intent_engine import CandleIntentConfig
    from pillar4_candle_engine.multi_candle_context_engine import MultiCandleContextConfig
    from pillar4_candle_engine.pillar4_output import Pillar4Config, run_pillar4_candle_intelligence
    from pillar4_candle_engine.pressure_engine import PressureConfig

    df = _load_ohlcv()
    config = Pillar4Config(
        candle_features=CandleFeatureConfig(
            atr_window=14, range_mean_window=20, body_mean_window=20, zscore_window=20,
            overlap_window_short=3, overlap_window_long=5, progress_window_short=3,
            progress_window_medium=5, progress_window_long=8, persistence_window_short=3,
            persistence_window_medium=5, persistence_window_long=8, realized_vol_window=20,
            volatility_percentile_window=50, range_percentile_window=50, body_percentile_window=50,
            contraction_window=5, inside_outside_window=5, entropy_window=5, rolling_wick_window=5,
        ),
        candle_intent=CandleIntentConfig(),
        multi_candle_context=MultiCandleContextConfig(),
        absorption=AbsorptionConfig(),
        breakout_quality=BreakoutQualityConfig(
            range_window=5, acceptance_close_threshold=0.55, strong_breakout_threshold=0.65,
            weak_breakout_threshold=0.40, fake_breakout_overlap_threshold=0.65, minimum_breach_threshold=0.0,
        ),
        pressure=PressureConfig(),
    )
    return run_pillar4_candle_intelligence(
        df=df, pillar4_config=config,
        columns=OhlcColumns(open="open", high="high", low="low", close="close", volume="volume", timestamp="timestamp"),
        asset="BTCUSDT", timeframe="1h", lookback_bars_used=min(30, len(df)), atr_method="wilder",
    )


@router.get("/pillar4")
async def get_pillar4():
    data = cache.get("pillar4")
    if data is None:
        return {"status": "not_ready", "message": "Pillar 4 not yet computed.", "last_updated": None}
    return {"status": "ok", "last_updated": cache.last_updated("pillar4"), "age_seconds": round(cache.age_seconds("pillar4"), 1), "data": data}


@router.get("/pillar4/refresh")
async def refresh_pillar4():
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _run_fresh)
        cache.set("pillar4", result)
        try:
            from api.main import broadcast
            await broadcast("pillar4", result)
        except Exception:
            pass
        return {"status": "ok", "message": "Pillar 4 refreshed.", "last_updated": cache.last_updated("pillar4"), "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}