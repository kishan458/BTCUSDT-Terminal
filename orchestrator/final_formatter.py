from __future__ import annotations

from typing import Any

from orchestrator.orchestrator_config import OrchestratorConfig
from orchestrator.schema import (
    deep_copy_terminal_payload_skeleton,
    utc_now_iso,
)


def _count_active_pillars(raw_pillar_outputs: dict[str, dict[str, Any]]) -> int:
    """Count pillar outputs with status OK."""
    count = 0
    for wrapper in raw_pillar_outputs.values():
        if isinstance(wrapper, dict) and wrapper.get("status") == "OK":
            count += 1
    return count


def build_metadata(
    config: OrchestratorConfig,
    *,
    run_id: str,
    total_latency_ms: int | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Build metadata block."""
    return {
        "product": config.product_name,
        "version": config.version,
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "symbol": config.symbol,
        "timeframe": config.default_timeframe,
        "environment": config.environment,
        "orchestrator_status": "DEGRADED",
        "total_latency_ms": total_latency_ms,
        "active_pillars": 0,
        "notes": notes or [],
    }


def assemble_terminal_payload(
    config: OrchestratorConfig,
    *,
    run_id: str,
    system_health: dict[str, Any],
    market_snapshot: dict[str, Any] | None = None,
    control_panel: dict[str, Any] | None = None,
    pillar_tabs: dict[str, Any] | None = None,
    ui_blocks: dict[str, Any] | None = None,
    raw_pillar_outputs: dict[str, dict[str, Any]] | None = None,
    total_latency_ms: int | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the final top-level orchestrator payload."""
    payload = deep_copy_terminal_payload_skeleton()

    raw_pillar_outputs = raw_pillar_outputs or {}
    pillar_tabs = pillar_tabs or {}
    ui_blocks = ui_blocks or {}

    payload["metadata"] = build_metadata(
        config,
        run_id=run_id,
        total_latency_ms=total_latency_ms,
        notes=notes,
    )
    payload["metadata"]["orchestrator_status"] = system_health.get("status", "DEGRADED")
    payload["metadata"]["active_pillars"] = _count_active_pillars(raw_pillar_outputs)

    payload["system_health"] = system_health

    if market_snapshot is not None:
        payload["market_snapshot"] = market_snapshot

    if control_panel is not None:
        payload["control_panel"] = control_panel

    payload["pillar_tabs"] = pillar_tabs
    payload["ui_blocks"] = ui_blocks
    payload["raw_pillar_outputs"] = raw_pillar_outputs

    return payload