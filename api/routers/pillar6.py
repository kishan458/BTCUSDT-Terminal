"""
Pillar 6 — High Impact Events router
GET /api/pillar6
GET /api/pillar6/refresh
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from fastapi import APIRouter
from api.cache import cache

router = APIRouter()


def _run_fresh() -> Dict[str, Any]:
    from pillar6_high_impact_events.pillar6_output import build_pillar6_output
    return build_pillar6_output()


@router.get("/pillar6")
async def get_pillar6():
    data = cache.get("pillar6")
    if data is None:
        return {"status": "not_ready", "message": "Pillar 6 not yet computed.", "last_updated": None}
    return {"status": "ok", "last_updated": cache.last_updated("pillar6"), "age_seconds": round(cache.age_seconds("pillar6"), 1), "data": data}


@router.get("/pillar6/refresh")
async def refresh_pillar6():
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _run_fresh)
        cache.set("pillar6", result)
        try:
            from api.main import broadcast
            await broadcast("pillar6", result)
        except Exception:
            pass
        return {"status": "ok", "message": "Pillar 6 refreshed.", "last_updated": cache.last_updated("pillar6"), "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}