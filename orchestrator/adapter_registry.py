from __future__ import annotations

from typing import Callable, Any

from orchestrator.pillar_adapters import AdapterCallable


def build_adapter_registry() -> dict[str, AdapterCallable]:
    """
    Return the currently connected real adapter registry.

    Start small and register real pillars one by one.
    """
    registry: dict[str, AdapterCallable] = {}

    return registry


def get_registered_adapter_keys() -> list[str]:
    """Return registered adapter keys in insertion order."""
    return list(build_adapter_registry().keys())