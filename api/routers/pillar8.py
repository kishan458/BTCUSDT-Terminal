"""
Pillar 8 — Decision, Risk & Sizing router
GET /api/pillar8
GET /api/pillar8/refresh
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter
from api.cache import cache

router = APIRouter()


def _run_fresh() -> Dict[str, Any]:
    from pillar8_decision_risk_backtesting.pillar8_engine import run_pillar8_engine
    from pillar8_decision_risk_backtesting.run_pillar8 import (
        _adapt_p1, _adapt_p2, _adapt_p3, _adapt_p4, _adapt_p5, _adapt_p6, _adapt_p7,
    )

    p1 = cache.get("pillar1") or {}
    p2 = cache.get("pillar2") or {}
    p3 = cache.get("pillar3") or {}
    p4 = cache.get("pillar4") or {}
    p5 = cache.get("pillar5") or {}
    p6 = cache.get("pillar6") or {}
    p7 = cache.get("pillar7") or {}

    result = run_pillar8_engine(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        sentiment_payload=_adapt_p1(p1),
        memory_payload=_adapt_p2(p2),
        structure_payload=_adapt_p3(p3),
        candle_payload=_adapt_p4(p4),
        regime_payload=_adapt_p5(p5),
        events_payload=_adapt_p6(p6),
        council_payload=_adapt_p7(p7),
        realized_volatility=None,
        target_volatility=0.20,
        win_probability=0.52,
        payoff_ratio=1.5,
        kelly_fraction=0.25,
        max_size_fraction=1.0,
        max_leverage=3.0,
        min_leverage=1.0,
        thesis_summary="Multi-pillar BTC/USDT terminal decision output.",
    )
    return result.to_dict()


@router.get("/pillar8")
async def get_pillar8():
    data = cache.get("pillar8")
    if data is None:
        return {"status": "not_ready", "message": "Pillar 8 not yet computed.", "last_updated": None}
    return {"status": "ok", "last_updated": cache.last_updated("pillar8"), "age_seconds": round(cache.age_seconds("pillar8"), 1), "data": data}


@router.get("/pillar8/refresh")
async def refresh_pillar8():
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _run_fresh)
        cache.set("pillar8", result)
        try:
            from api.main import broadcast
            await broadcast("pillar8", result)
        except Exception:
            pass
        return {"status": "ok", "message": "Pillar 8 refreshed.", "last_updated": cache.last_updated("pillar8"), "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}