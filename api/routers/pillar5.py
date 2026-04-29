"""
Pillar 5 — Regime & Cycle router
GET /api/pillar5
GET /api/pillar5/refresh
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from fastapi import APIRouter
from api.cache import cache

router = APIRouter()


def _run_fresh() -> Dict[str, Any]:
    from pillar5_regime_cycle_engine.pillar5_output import build_pillar5_output
    return build_pillar5_output()


@router.get("/pillar5")
async def get_pillar5():
    data = cache.get("pillar5")
    if data is None:
        return {"status": "not_ready", "message": "Pillar 5 not yet computed.", "last_updated": None}
    return {"status": "ok", "last_updated": cache.last_updated("pillar5"), "age_seconds": round(cache.age_seconds("pillar5"), 1), "data": data}


@router.get("/pillar5/refresh")
async def refresh_pillar5():
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _run_fresh)
        cache.set("pillar5", result)
        try:
            from api.main import broadcast
            await broadcast("pillar5", result)
        except Exception:
            pass
        return {"status": "ok", "message": "Pillar 5 refreshed.", "last_updated": cache.last_updated("pillar5"), "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}