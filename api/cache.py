"""
In-memory cache for all 8 pillar outputs.
Each pillar stores its latest result + timestamp.
Thread-safe for async usage.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class PillarCache:
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def set(self, pillar: str, data: Any) -> None:
        """Store pillar result with current timestamp."""
        self._store[pillar] = {
            "data": data,
            "timestamp": time.time(),
            "timestamp_iso": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        }

    def get(self, pillar: str) -> Optional[Dict[str, Any]]:
        """Return cached pillar result, or None if not yet computed."""
        entry = self._store.get(pillar)
        if entry is None:
            return None
        return entry["data"]

    def has(self, pillar: str) -> bool:
        """Return True if pillar has been computed at least once."""
        return pillar in self._store

    def last_updated(self, pillar: str) -> Optional[str]:
        """Return ISO timestamp of last update, or None."""
        entry = self._store.get(pillar)
        if entry is None:
            return None
        return entry["timestamp_iso"]

    def age_seconds(self, pillar: str) -> Optional[float]:
        """Return seconds since last update, or None."""
        entry = self._store.get(pillar)
        if entry is None:
            return None
        return time.time() - entry["timestamp"]

    def get_all(self) -> Dict[str, Any]:
        """Return all cached pillar data."""
        return {
            pillar: entry["data"]
            for pillar, entry in self._store.items()
        }


# Singleton — import this everywhere
cache = PillarCache()