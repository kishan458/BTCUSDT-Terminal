"""
Pillar 3 — Structure & Liquidity router
GET /api/pillar3
GET /api/pillar3/refresh
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
    from pillar3_structure_liquidity_engine.pillar3_output import run_pillar3_output
    df = _load_ohlcv()
    return run_pillar3_output(df)


@router.get("/pillar3")
async def get_pillar3():
    data = cache.get("pillar3")
    if data is None:
        return {"status": "not_ready", "message": "Pillar 3 not yet computed.", "last_updated": None}
    return {"status": "ok", "last_updated": cache.last_updated("pillar3"), "age_seconds": round(cache.age_seconds("pillar3"), 1), "data": data}


@router.get("/pillar3/refresh")
async def refresh_pillar3():
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _run_fresh)
        cache.set("pillar3", result)
        try:
            from api.main import broadcast
            await broadcast("pillar3", result)
        except Exception:
            pass
        return {"status": "ok", "message": "Pillar 3 refreshed.", "last_updated": cache.last_updated("pillar3"), "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}