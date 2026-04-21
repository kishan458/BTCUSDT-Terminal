from __future__ import annotations

from typing import Any

from orchestrator.pillar_registry import get_enabled_pillars
from orchestrator.schema import build_pillar_tab


def _extract_core_state(wrapper: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized core state from a pillar wrapper."""
    signal = wrapper.get("signal", {}) if isinstance(wrapper, dict) else {}

    return {
        "primary_state": signal.get("state", "UNKNOWN"),
        "bias": signal.get("bias", "NEUTRAL"),
        "strength": signal.get("strength", 0.0),
        "market_implication": wrapper.get("summary", ""),
    }


def _extract_ai_summary(wrapper: dict[str, Any]) -> dict[str, str]:
    """Extract AI summary block if present, else return safe empty structure."""
    payload = wrapper.get("payload", {}) if isinstance(wrapper, dict) else {}
    ai_summary = payload.get("ai_summary", {})

    if not isinstance(ai_summary, dict):
        ai_summary = {}

    return {
        "headline": ai_summary.get("headline", ""),
        "summary": ai_summary.get("summary", wrapper.get("summary", "")),
        "interpretation": ai_summary.get("interpretation", ""),
        "trade_relevance": ai_summary.get("trade_relevance", ""),
    }


def _extract_display_blocks(wrapper: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract frontend display blocks if present."""
    payload = wrapper.get("payload", {}) if isinstance(wrapper, dict) else {}
    display_blocks = payload.get("display_blocks", [])
    return display_blocks if isinstance(display_blocks, list) else []


def _extract_charts(wrapper: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract chart blocks if present."""
    payload = wrapper.get("payload", {}) if isinstance(wrapper, dict) else {}
    charts = payload.get("charts", [])
    return charts if isinstance(charts, list) else []


def _extract_metrics(wrapper: dict[str, Any]) -> dict[str, Any]:
    """Extract metric block if present."""
    payload = wrapper.get("payload", {}) if isinstance(wrapper, dict) else {}
    metrics = payload.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _extract_warnings(wrapper: dict[str, Any]) -> list[str]:
    """Extract warning list from payload and data_quality."""
    warnings: list[str] = []

    data_quality = wrapper.get("data_quality", {}) if isinstance(wrapper, dict) else {}
    dq_warnings = data_quality.get("warnings", [])
    if isinstance(dq_warnings, list):
        warnings.extend([str(item) for item in dq_warnings])

    payload = wrapper.get("payload", {}) if isinstance(wrapper, dict) else {}
    payload_warnings = payload.get("warnings", [])
    if isinstance(payload_warnings, list):
        warnings.extend([str(item) for item in payload_warnings])

    # Preserve order while removing duplicates
    deduped: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        if warning not in seen:
            deduped.append(warning)
            seen.add(warning)

    return deduped


def _extract_restrictions(wrapper: dict[str, Any]) -> list[str]:
    """Extract restriction list from payload."""
    payload = wrapper.get("payload", {}) if isinstance(wrapper, dict) else {}
    restrictions = payload.get("restrictions", [])
    return [str(item) for item in restrictions] if isinstance(restrictions, list) else []


def build_pillar_tabs(
    pillar_outputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build frontend-ready pillar tab payloads from raw pillar wrappers."""
    pillar_tabs: dict[str, dict[str, Any]] = {}

    for entry in get_enabled_pillars():
        wrapper = pillar_outputs.get(entry.key)
        if not isinstance(wrapper, dict):
            continue

        core_state = _extract_core_state(wrapper)

        pillar_tabs[entry.key] = build_pillar_tab(
            entry.key,
            entry.title,
            status=wrapper.get("status", "DEGRADED"),
            confidence=wrapper.get("confidence", 0.0),
            summary=wrapper.get("summary", ""),
            primary_state=core_state["primary_state"],
            bias=core_state["bias"],
            strength=core_state["strength"],
            market_implication=core_state["market_implication"],
            metrics=_extract_metrics(wrapper),
            display_blocks=_extract_display_blocks(wrapper),
            charts=_extract_charts(wrapper),
            ai_summary=_extract_ai_summary(wrapper),
            warnings=_extract_warnings(wrapper),
            restrictions=_extract_restrictions(wrapper),
            raw_payload=wrapper.get("payload", {}),
        )

    return pillar_tabs


def build_ui_blocks(
    pillar_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Build reusable UI blocks from available pillar wrappers.

    This first version keeps blocks minimal and only uses real available data.
    """
    top_cards: list[dict[str, Any]] = []
    warning_banners: list[dict[str, Any]] = []

    for entry in get_enabled_pillars():
        wrapper = pillar_outputs.get(entry.key)
        if not isinstance(wrapper, dict):
            continue

        signal = wrapper.get("signal", {})
        top_cards.append(
            {
                "id": f"{entry.key}_signal_card",
                "type": "stat_card",
                "title": entry.title,
                "value": signal.get("state", "UNKNOWN"),
                "confidence": wrapper.get("confidence", 0.0),
                "severity": (
                    "positive" if signal.get("bias") == "BULLISH"
                    else "negative" if signal.get("bias") == "BEARISH"
                    else "neutral"
                ),
            }
        )

        for warning in _extract_warnings(wrapper):
            warning_banners.append(
                {
                    "id": f"{entry.key}_warning_{len(warning_banners) + 1}",
                    "type": "banner",
                    "severity": "warning",
                    "title": entry.title,
                    "message": warning,
                }
            )

    return {
        "top_cards": top_cards,
        "warning_banners": warning_banners,
    }