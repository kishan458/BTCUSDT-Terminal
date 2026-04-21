from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PillarRegistryEntry:
    key: str
    title: str
    order: int
    critical: bool
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PILLAR_REGISTRY: tuple[PillarRegistryEntry, ...] = (
    PillarRegistryEntry(
        key="pillar_1_sentiment",
        title="Pillar 1 — Sentiment & Narrative",
        order=1,
        critical=False,
    ),
    PillarRegistryEntry(
        key="pillar_6_events",
        title="Pillar 6 — High-Impact Event Intelligence",
        order=2,
        critical=True,
    ),
    PillarRegistryEntry(
        key="pillar_5_regime",
        title="Pillar 5 — Regime & Cycle",
        order=3,
        critical=True,
    ),
    PillarRegistryEntry(
        key="pillar_3_structure",
        title="Pillar 3 — Structure & Liquidity",
        order=4,
        critical=True,
    ),
    PillarRegistryEntry(
        key="pillar_4_candles",
        title="Pillar 4 — Candle Intelligence",
        order=5,
        critical=False,
    ),
    PillarRegistryEntry(
        key="pillar_2_memory",
        title="Pillar 2 — BTC Market Memory",
        order=6,
        critical=False,
    ),
    PillarRegistryEntry(
        key="pillar_7_ml_council",
        title="Pillar 7 — Multi-Agent ML Council",
        order=7,
        critical=False,
    ),
    PillarRegistryEntry(
        key="pillar_8_decision",
        title="Pillar 8 — Decision, Risk & Backtesting",
        order=8,
        critical=True,
    ),
)


def get_all_pillars() -> list[PillarRegistryEntry]:
    """Return all pillars in declared order."""
    return sorted(PILLAR_REGISTRY, key=lambda entry: entry.order)


def get_enabled_pillars() -> list[PillarRegistryEntry]:
    """Return only enabled pillars in declared order."""
    return [entry for entry in get_all_pillars() if entry.enabled]


def get_critical_pillars() -> list[PillarRegistryEntry]:
    """Return only critical pillars in declared order."""
    return [entry for entry in get_enabled_pillars() if entry.critical]


def get_pillar_by_key(pillar_key: str) -> PillarRegistryEntry:
    """Return a pillar entry by key or raise KeyError if missing."""
    for entry in PILLAR_REGISTRY:
        if entry.key == pillar_key:
            return entry
    raise KeyError(f"Unknown pillar key: {pillar_key}")


def get_pillar_title(pillar_key: str) -> str:
    """Return the human-readable title for a given pillar key."""
    return get_pillar_by_key(pillar_key).title


def get_enabled_pillar_keys() -> list[str]:
    """Return enabled pillar keys in execution order."""
    return [entry.key for entry in get_enabled_pillars()]


def get_critical_pillar_keys() -> list[str]:
    """Return critical pillar keys in execution order."""
    return [entry.key for entry in get_critical_pillars()]