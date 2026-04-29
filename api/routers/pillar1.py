"""
Pillar 1 — Sentiment router
GET /api/pillar1         → returns cached result
GET /api/pillar1/refresh → triggers a fresh run immediately
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter

from api.cache import cache

router = APIRouter()


def _run_fresh() -> Dict[str, Any]:
    from pillar1_sentiment.run_pillar1 import run_pillar1
    return run_pillar1()


@router.get("/pillar1")
async def get_pillar1():
    """Return cached Pillar 1 sentiment data."""
    data = cache.get("pillar1")
    if data is None:
        return {
            "status": "not_ready",
            "message": "Pillar 1 has not completed its first run yet. Check back in ~30 seconds.",
            "last_updated": None,
        }
    return {
        "status": "ok",
        "last_updated": cache.last_updated("pillar1"),
        "age_seconds": round(cache.age_seconds("pillar1"), 1),
        "data": data,
    }


@router.get("/pillar1/refresh")
async def refresh_pillar1():
    """Force an immediate fresh run of Pillar 1."""
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _run_fresh)
        cache.set("pillar1", result)

        # Also push to websocket clients
        try:
            from api.main import broadcast
            await broadcast("pillar1", result)
        except Exception:
            pass

        return {
            "status": "ok",
            "message": "Pillar 1 refreshed successfully.",
            "last_updated": cache.last_updated("pillar1"),
            "data": result,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@router.get("/pillar1/sentiment-history")
async def get_sentiment_history():
    """
    Return last 7 days of sentiment snapshots from SQLite.
    Used for the sentiment history chart in the UI.
    """
    import sqlite3
    from pathlib import Path

    db_path = str(Path(__file__).resolve().parent.parent.parent / "database" / "btc_terminal.db")

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("""
                SELECT timestamp, label, score, confidence, article_count
                FROM sentiment_history
                WHERE timestamp >= datetime('now', '-7 days')
                ORDER BY timestamp ASC
            """).fetchall()

        return {
            "status": "ok",
            "count": len(rows),
            "history": [
                {
                    "timestamp": row[0],
                    "label": row[1],
                    "score": row[2],
                    "confidence": row[3],
                    "article_count": row[4],
                }
                for row in rows
            ],
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "history": [],
        }