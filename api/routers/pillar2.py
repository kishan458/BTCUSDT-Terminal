"""
Pillar 2 — Market Memory router
GET /api/pillar2
GET /api/pillar2/refresh
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter
from api.cache import cache

router = APIRouter()

DB_PATH = str(Path(__file__).resolve().parent.parent.parent / "database" / "btc_terminal.db")


def _run_fresh() -> Dict[str, Any]:
    from pillar2_market_memory_engine.memory_feature_engine import MemoryFeatureConfig
    from pillar2_market_memory_engine.memory_outcome_engine import MemoryOutcomeConfig
    from pillar2_market_memory_engine.state_signature_engine import StateSignatureConfig
    from pillar2_market_memory_engine.analog_retrieval_engine import AnalogRetrievalConfig
    from pillar2_market_memory_engine.conditional_outcome_engine import ConditionalOutcomeConfig
    from pillar2_market_memory_engine.stability_engine import StabilityConfig
    from pillar2_market_memory_engine.session_memory_engine import SessionMemoryConfig
    from pillar2_market_memory_engine.calendar_memory_engine import CalendarMemoryConfig
    from pillar2_market_memory_engine.pillar2_output import Pillar2OutputConfig, run_pillar2_output
    return run_pillar2_output(
        feature_config=MemoryFeatureConfig(db_path=DB_PATH, table_name="btc_price_1h", timestamp_col="timestamp"),
        outcome_config=MemoryOutcomeConfig(horizons=(1, 3, 6, 12)),
        signature_config=StateSignatureConfig(require_complete_core_state=True),
        retrieval_config=AnalogRetrievalConfig(min_similarity_score=0.45, max_weighted_matches=300),
        conditional_config=ConditionalOutcomeConfig(use_weighted_matches=True),
        stability_config=StabilityConfig(minimum_window_samples=10),
        session_config=SessionMemoryConfig(min_samples_per_group=25),
        calendar_config=CalendarMemoryConfig(min_samples_per_group=25),
        output_config=Pillar2OutputConfig(asset="BTCUSDT", round_decimals=4),
    )


@router.get("/pillar2")
async def get_pillar2():
    data = cache.get("pillar2")
    if data is None:
        return {"status": "not_ready", "message": "Pillar 2 not yet computed (runs every 15 min).", "last_updated": None}
    return {"status": "ok", "last_updated": cache.last_updated("pillar2"), "age_seconds": round(cache.age_seconds("pillar2"), 1), "data": data}


@router.get("/pillar2/refresh")
async def refresh_pillar2():
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _run_fresh)
        cache.set("pillar2", result)
        try:
            from api.main import broadcast
            await broadcast("pillar2", result)
        except Exception:
            pass
        return {"status": "ok", "message": "Pillar 2 refreshed.", "last_updated": cache.last_updated("pillar2"), "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}