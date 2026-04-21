from __future__ import annotations

from typing import Any, Callable

from orchestrator.schema import build_pillar_wrapper
from orchestrator.validators import validate_pillar_wrapper


AdapterCallable = Callable[[], Any]


def _failed_adapter_wrapper(
    pillar_key: str,
    summary: str,
    warnings: list[str] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a standardized FAILED wrapper for adapter failures."""
    return build_pillar_wrapper(
        pillar_key,
        status="FAILED",
        confidence=0.0,
        summary=summary,
        bias="NEUTRAL",
        strength=0.0,
        state="ADAPTER_FAILED",
        is_fresh=False,
        is_complete=False,
        freshness_status="UNKNOWN",
        warnings=warnings or [],
        payload=payload or {},
    )


def normalize_adapter_output(
    pillar_key: str,
    raw_output: Any,
) -> dict[str, Any]:
    """
    Normalize adapter output into a valid pillar wrapper.

    Rules:
    - If the adapter already returns a valid orchestrator wrapper, keep it.
    - Otherwise wrap the raw output as payload and mark status DEGRADED.
    - If raw output is unusable, return FAILED.
    """
    if raw_output is None:
        return _failed_adapter_wrapper(
            pillar_key,
            "Adapter returned no output.",
            warnings=["adapter returned None"],
        )

    if isinstance(raw_output, dict):
        validation_errors = validate_pillar_wrapper(raw_output)
        if not validation_errors:
            return raw_output

        return build_pillar_wrapper(
            pillar_key,
            status="DEGRADED",
            confidence=0.0,
            summary="Raw pillar output was not in orchestrator wrapper format.",
            bias="NEUTRAL",
            strength=0.0,
            state="UNSTANDARDIZED_OUTPUT",
            is_fresh=False,
            is_complete=False,
            freshness_status="UNKNOWN",
            warnings=validation_errors,
            payload={
                "raw_output": raw_output,
                "normalization_mode": "wrapped_raw_dict",
            },
        )

    return build_pillar_wrapper(
        pillar_key,
        status="DEGRADED",
        confidence=0.0,
        summary="Raw pillar output was not in orchestrator wrapper format.",
        bias="NEUTRAL",
        strength=0.0,
        state="UNSTANDARDIZED_OUTPUT",
        is_fresh=False,
        is_complete=False,
        freshness_status="UNKNOWN",
        warnings=[f"unsupported raw output type: {type(raw_output).__name__}"],
        payload={
            "raw_output_repr": repr(raw_output),
            "normalization_mode": "wrapped_non_dict_output",
        },
    )


def run_pillar_adapter(
    pillar_key: str,
    adapter: AdapterCallable,
) -> dict[str, Any]:
    """
    Execute one pillar adapter safely and return an orchestrator wrapper.
    """
    try:
        raw_output = adapter()
    except Exception as exc:
        return _failed_adapter_wrapper(
            pillar_key,
            "Adapter execution raised an exception.",
            warnings=[f"{type(exc).__name__}: {exc}"],
            payload={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )

    return normalize_adapter_output(pillar_key, raw_output)


def run_registered_adapters(
    adapter_registry: dict[str, AdapterCallable],
) -> dict[str, dict[str, Any]]:
    """
    Execute all registered adapters and return pillar_key -> wrapper output.
    """
    outputs: dict[str, dict[str, Any]] = {}

    for pillar_key, adapter in adapter_registry.items():
        outputs[pillar_key] = run_pillar_adapter(pillar_key, adapter)

    return outputs