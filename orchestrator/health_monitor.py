from __future__ import annotations

from typing import Any

from orchestrator.orchestrator_config import OrchestratorConfig
from orchestrator.pillar_registry import get_enabled_pillar_keys
from orchestrator.schema import SeverityEnum, StatusEnum


def _safe_wrapper_status(wrapper: dict[str, Any] | None) -> str:
    """Return wrapper status if present, otherwise FAILED."""
    if not isinstance(wrapper, dict):
        return StatusEnum.FAILED.value
    status = wrapper.get("status")
    if status in {item.value for item in StatusEnum}:
        return status
    return StatusEnum.FAILED.value


def build_pillar_status_map(
    pillar_outputs: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Return pillar -> status mapping for all enabled pillars."""
    status_map: dict[str, str] = {}

    for pillar_key in get_enabled_pillar_keys():
        wrapper = pillar_outputs.get(pillar_key)
        status_map[pillar_key] = _safe_wrapper_status(wrapper)

    return status_map


def count_statuses(
    pillar_statuses: dict[str, str],
) -> dict[str, int]:
    """Count OK / DEGRADED / FAILED statuses."""
    counts = {
        StatusEnum.OK.value: 0,
        StatusEnum.DEGRADED.value: 0,
        StatusEnum.FAILED.value: 0,
    }

    for status in pillar_statuses.values():
        if status in counts:
            counts[status] += 1

    return counts


def get_critical_failures(
    pillar_statuses: dict[str, str],
    config: OrchestratorConfig,
) -> list[str]:
    """Return failed critical pillar keys."""
    critical_failures: list[str] = []

    for pillar_key in config.critical_pillar_keys:
        if pillar_statuses.get(pillar_key) == StatusEnum.FAILED.value:
            critical_failures.append(pillar_key)

    return critical_failures


def determine_system_health_status(
    pillar_statuses: dict[str, str],
    config: OrchestratorConfig,
) -> tuple[str, str, str]:
    """
    Determine overall system health.

    Returns:
        (status, severity, summary)
    """
    counts = count_statuses(pillar_statuses)
    critical_failures = get_critical_failures(pillar_statuses, config)

    if len(critical_failures) >= config.health.unsafe_if_critical_pillars_failed_count_gte:
        return (
            StatusEnum.FAILED.value,
            SeverityEnum.CRITICAL.value,
            "Multiple critical pillars failed. Terminal output is unsafe.",
        )

    if counts[StatusEnum.FAILED.value] > 0 or counts[StatusEnum.DEGRADED.value] > 0:
        return (
            StatusEnum.DEGRADED.value,
            SeverityEnum.WARNING.value,
            "Some pillars are degraded or failed. Terminal output should be used with caution.",
        )

    return (
        StatusEnum.OK.value,
        SeverityEnum.INFO.value,
        "All enabled pillars are active. Terminal output is safe to use.",
    )


def build_system_health(
    pillar_outputs: dict[str, dict[str, Any]],
    config: OrchestratorConfig,
    *,
    data_freshness: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build the top-level system_health block."""
    pillar_statuses = build_pillar_status_map(pillar_outputs)
    counts = count_statuses(pillar_statuses)
    critical_failures = get_critical_failures(pillar_statuses, config)
    status, severity, summary = determine_system_health_status(pillar_statuses, config)

    return {
        "status": status,
        "severity": severity,
        "summary": summary,
        "active_pillars": counts[StatusEnum.OK.value],
        "degraded_pillars": counts[StatusEnum.DEGRADED.value],
        "failed_pillars": counts[StatusEnum.FAILED.value],
        "critical_failures": critical_failures,
        "warnings": warnings or [],
        "data_freshness": data_freshness or {},
        "pillar_statuses": pillar_statuses,
    }