from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from pillar7_ml_council.shared_state_builder import build_shared_state
from pillar7_ml_council.pillar7_output import build_pillar7_output


def _load_callable(module_path: str, func_name: str) -> Callable[..., Dict[str, Any]]:
    module = __import__(module_path, fromlist=[func_name])
    func = getattr(module, func_name, None)
    if func is None:
        raise ImportError(f"Could not find function '{func_name}' in module '{module_path}'")
    return func


def _safe_call(func: Callable[..., Dict[str, Any]], kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    kwargs = kwargs or {}
    result = func(**kwargs)

    if result is None:
        return {}

    if not isinstance(result, dict):
        raise ValueError(f"Expected dict output from {func.__name__}, got {type(result).__name__}")

    return result


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_pillar7(
    *,
    professor_artifact_path: str,
    retail_artifact_path: str,
    output_path: str = "pillar7_ml_council/artifacts/pillar7_output.json",
    threshold: float = 0.5,
    asset: str = "BTCUSDT",
    timestamp_utc: Optional[str] = None,

    # Pillar 1
    pillar1_module: Optional[str] = None,
    pillar1_func: Optional[str] = None,
    pillar1_kwargs: Optional[Dict[str, Any]] = None,

    # Pillar 2
    pillar2_module: Optional[str] = None,
    pillar2_func: Optional[str] = None,
    pillar2_kwargs: Optional[Dict[str, Any]] = None,

    # Pillar 3
    pillar3_module: Optional[str] = None,
    pillar3_func: Optional[str] = None,
    pillar3_kwargs: Optional[Dict[str, Any]] = None,

    # Pillar 4
    pillar4_module: Optional[str] = None,
    pillar4_func: Optional[str] = None,
    pillar4_kwargs: Optional[Dict[str, Any]] = None,

    # Pillar 5
    pillar5_module: Optional[str] = None,
    pillar5_func: Optional[str] = None,
    pillar5_kwargs: Optional[Dict[str, Any]] = None,

    # Pillar 6
    pillar6_module: Optional[str] = None,
    pillar6_func: Optional[str] = None,
    pillar6_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Real Pillar 7 runner.

    You pass the module + function names for each upstream pillar output builder.
    This keeps Pillar 7 decoupled from your exact repo naming while making it easy to wire in.

    Example:
        pillar1_module="pillar1_sentiment_narrative.pillar1_output"
        pillar1_func="build_pillar1_output"
    """

    pillar_outputs: Dict[str, Dict[str, Any]] = {
        "pillar1": {},
        "pillar2": {},
        "pillar3": {},
        "pillar4": {},
        "pillar5": {},
        "pillar6": {},
    }

    config = [
        ("pillar1", pillar1_module, pillar1_func, pillar1_kwargs),
        ("pillar2", pillar2_module, pillar2_func, pillar2_kwargs),
        ("pillar3", pillar3_module, pillar3_func, pillar3_kwargs),
        ("pillar4", pillar4_module, pillar4_func, pillar4_kwargs),
        ("pillar5", pillar5_module, pillar5_func, pillar5_kwargs),
        ("pillar6", pillar6_module, pillar6_func, pillar6_kwargs),
    ]

    for pillar_name, module_path, func_name, kwargs in config:
        if module_path and func_name:
            func = _load_callable(module_path, func_name)
            pillar_outputs[pillar_name] = _safe_call(func, kwargs)

    shared_state = build_shared_state(
        asset=asset,
        timestamp_utc=timestamp_utc,
        pillar1_output=pillar_outputs["pillar1"],
        pillar2_output=pillar_outputs["pillar2"],
        pillar3_output=pillar_outputs["pillar3"],
        pillar4_output=pillar_outputs["pillar4"],
        pillar5_output=pillar_outputs["pillar5"],
        pillar6_output=pillar_outputs["pillar6"],
    )

    pillar7_output = build_pillar7_output(
        shared_state=shared_state,
        professor_artifact_path=professor_artifact_path,
        retail_artifact_path=retail_artifact_path,
        threshold=threshold,
    )

    _write_json(output_path, pillar7_output)

    return pillar7_output