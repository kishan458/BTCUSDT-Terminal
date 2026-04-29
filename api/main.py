"""
BTC/USDT Terminal — FastAPI Backend
Run with: uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.cache import cache
from api.routers import pillar1, pillar2, pillar3, pillar4, pillar5, pillar6, pillar7, pillar8
from api.scheduler import start_scheduler

app = FastAPI(title="BTC/USDT Terminal API", version="1.0.0")

# ── CORS (allow React frontend on any port) ───────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ROUTERS ───────────────────────────────────────────────────────────────────
app.include_router(pillar1.router, prefix="/api")
app.include_router(pillar2.router, prefix="/api")
app.include_router(pillar3.router, prefix="/api")
app.include_router(pillar4.router, prefix="/api")
app.include_router(pillar5.router, prefix="/api")
app.include_router(pillar6.router, prefix="/api")
app.include_router(pillar7.router, prefix="/api")
app.include_router(pillar8.router, prefix="/api")


# ── STARTUP ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Start background refresh scheduler on app startup."""
    asyncio.create_task(start_scheduler())
    print("✓ BTC/USDT Terminal API started")
    print("✓ Background scheduler running")


# ── HEALTH CHECK ──────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pillars": {
            pillar: {
                "cached": cache.has(pillar),
                "last_updated": cache.last_updated(pillar),
            }
            for pillar in ["pillar1", "pillar2", "pillar3", "pillar4", "pillar5", "pillar6", "pillar7", "pillar8"]
        },
    }


# ── FULL SNAPSHOT (all pillars at once) ───────────────────────────────────────
@app.get("/api/snapshot")
async def snapshot():
    """Return cached data from all 8 pillars in one call."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pillar1":  cache.get("pillar1"),
        "pillar2":  cache.get("pillar2"),
        "pillar3":  cache.get("pillar3"),
        "pillar4":  cache.get("pillar4"),
        "pillar5":  cache.get("pillar5"),
        "pillar6":  cache.get("pillar6"),
        "pillar7":  cache.get("pillar7"),
        "pillar8":  cache.get("pillar8"),
    }


# ── WEBSOCKET — streams live updates to frontend ──────────────────────────────
connected_clients: list[WebSocket] = []


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    print(f"WebSocket client connected. Total: {len(connected_clients)}")

    try:
        # Send current snapshot immediately on connect
        snapshot_data = {
            "type": "snapshot",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pillar1":  cache.get("pillar1"),
            "pillar2":  cache.get("pillar2"),
            "pillar3":  cache.get("pillar3"),
            "pillar4":  cache.get("pillar4"),
            "pillar5":  cache.get("pillar5"),
            "pillar6":  cache.get("pillar6"),
            "pillar7":  cache.get("pillar7"),
            "pillar8":  cache.get("pillar8"),
        }
        await websocket.send_text(json.dumps(snapshot_data, default=str))

        # Keep connection alive, ping every 30s
        while True:
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"type": "ping", "timestamp": datetime.now(timezone.utc).isoformat()}))

    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        print(f"WebSocket client disconnected. Total: {len(connected_clients)}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)


async def broadcast(pillar: str, data: Dict[str, Any]):
    """Called by scheduler when a pillar finishes updating — pushes to all clients."""
    if not connected_clients:
        return

    message = json.dumps({
        "type": "pillar_update",
        "pillar": pillar,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }, default=str)

    dead = []
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            dead.append(client)

    for d in dead:
        connected_clients.remove(d)