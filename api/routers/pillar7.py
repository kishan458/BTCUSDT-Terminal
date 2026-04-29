"""
Pillar 7 — ML Council router
GET /api/pillar7
GET /api/pillar7/refresh
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter
from api.cache import cache

router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROFESSOR_ARTIFACT = str(REPO_ROOT / "pillar7_ml_council" / "artifacts" / "professor_policy_rf_long_vs_no_trade_v1_test.pkl")
RETAIL_ARTIFACT    = str(REPO_ROOT / "pillar7_ml_council" / "artifacts" / "retail_policy_rf_long_vs_no_action_v1_test.pkl")


def _run_fresh() -> Dict[str, Any]:
    from pillar7_ml_council.shared_state_builder import build_shared_state
    from pillar7_ml_council.pillar7_output import build_pillar7_output

    p1 = cache.get("pillar1") or {}
    p2 = cache.get("pillar2") or {}
    p3 = cache.get("pillar3") or {}
    p4 = cache.get("pillar4") or {}
    p5 = cache.get("pillar5") or {}
    p6 = cache.get("pillar6") or {}

    shared_state = build_shared_state(
        asset="BTCUSDT",
        pillar1_output=p1, pillar2_output=p2, pillar3_output=p3,
        pillar4_output=p4, pillar5_output=p5, pillar6_output=p6,
    )
    return build_pillar7_output(
        shared_state=shared_state,
        professor_artifact_path=PROFESSOR_ARTIFACT,
        retail_artifact_path=RETAIL_ARTIFACT,
        threshold=0.5,
    )


@router.get("/pillar7")
async def get_pillar7():
    data = cache.get("pillar7")
    if data is None:
        return {"status": "not_ready", "message": "Pillar 7 not yet computed.", "last_updated": None}
    return {"status": "ok", "last_updated": cache.last_updated("pillar7"), "age_seconds": round(cache.age_seconds("pillar7"), 1), "data": data}


@router.get("/pillar7/refresh")
async def refresh_pillar7():
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _run_fresh)
        cache.set("pillar7", result)
        try:
            from api.main import broadcast
            await broadcast("pillar7", result)
        except Exception:
            pass
        return {"status": "ok", "message": "Pillar 7 refreshed.", "last_updated": cache.last_updated("pillar7"), "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}