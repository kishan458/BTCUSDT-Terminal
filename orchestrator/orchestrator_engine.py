from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from orchestrator.final_formatter import assemble_terminal_payload
from orchestrator.fusion_engine import build_control_panel, build_market_snapshot
from orchestrator.health_monitor import build_system_health
from orchestrator.orchestrator_config import OrchestratorConfig, build_orchestrator_config
from orchestrator.pillar_registry import get_enabled_pillar_keys
from orchestrator.schema import build_pillar_wrapper
from orchestrator.ui_payload_builder import build_pillar_tabs, build_ui_blocks
from orchestrator.validators import validate_pillar_wrapper


def generate_run_id() -> str:
    """Generate a UTC run ID for orchestrator payloads."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def _build_failed_wrapper(
    pillar_key: str,
    errors: list[str],
) -> dict[str, Any]:
    """Build a safe failed wrapper when input is missing or invalid."""
    return build_pillar_wrapper(
        pillar_key,
        status="FAILED",
        confidence=0.0,
        summary="Pillar output failed validation.",
        bias="NEUTRAL",
        strength=0.0,
        state="INVALID_OUTPUT",
        is_fresh=False,
        is_complete=False,
        freshness_status="UNKNOWN",
        warnings=errors,
        payload={
            "validation_errors": errors,
        },
    )


def normalize_pillar_outputs(
    pillar_outputs: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """
    Normalize raw pillar outputs into a complete, validated map.

    Returns:
        normalized_outputs, notes
    """
    normalized: dict[str, dict[str, Any]] = {}
    notes: list[str] = []

    for pillar_key in get_enabled_pillar_keys():
        raw_output = pillar_outputs.get(pillar_key)

        if raw_output is None:
            notes.append(f"{pillar_key}: missing output; replaced with FAILED wrapper")
            normalized[pillar_key] = _build_failed_wrapper(
                pillar_key,
                ["missing pillar output"],
            )
            continue

        errors = validate_pillar_wrapper(raw_output)
        if errors:
            notes.append(f"{pillar_key}: invalid output; replaced with FAILED wrapper")
            normalized[pillar_key] = _build_failed_wrapper(pillar_key, errors)
            continue

        normalized[pillar_key] = raw_output

    return normalized, notes


class TerminalOrchestrator:
    """Main orchestrator engine for building the terminal payload."""

    def __init__(self, config: OrchestratorConfig | None = None) -> None:
        self.config = config or build_orchestrator_config()

    def run(
        self,
        pillar_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run the orchestrator over already-produced pillar outputs.

        Note:
            This version expects pillar_outputs to be passed in.
            Real live pillar execution will be connected after this layer is stable.
        """
        start = perf_counter()

        normalized_outputs, notes = normalize_pillar_outputs(pillar_outputs)

        system_health = build_system_health(
            normalized_outputs,
            self.config,
        )

        market_snapshot = build_market_snapshot(
            normalized_outputs,
            symbol=self.config.symbol,
        )

        control_panel = build_control_panel(
            normalized_outputs,
            market_snapshot,
        )

        pillar_tabs = build_pillar_tabs(normalized_outputs)
        ui_blocks = build_ui_blocks(normalized_outputs)

        total_latency_ms = int((perf_counter() - start) * 1000)

        final_payload = assemble_terminal_payload(
            self.config,
            run_id=generate_run_id(),
            system_health=system_health,
            market_snapshot=market_snapshot,
            control_panel=control_panel,
            pillar_tabs=pillar_tabs,
            ui_blocks=ui_blocks,
            raw_pillar_outputs=normalized_outputs,
            total_latency_ms=total_latency_ms,
            notes=notes,
        )

        return final_payload