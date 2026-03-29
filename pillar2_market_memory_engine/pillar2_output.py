from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import json

import numpy as np


# ============================================================
# IMPORT FIX FOR DIRECT SCRIPT RUN
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pillar2_market_memory_engine.memory_feature_engine import (  # noqa: E402
    MemoryFeatureConfig,
)
from pillar2_market_memory_engine.memory_outcome_engine import (  # noqa: E402
    MemoryOutcomeConfig,
)
from pillar2_market_memory_engine.state_signature_engine import (  # noqa: E402
    StateSignatureConfig,
)
from pillar2_market_memory_engine.analog_retrieval_engine import (  # noqa: E402
    AnalogRetrievalConfig,
)
from pillar2_market_memory_engine.conditional_outcome_engine import (  # noqa: E402
    ConditionalOutcomeConfig,
    run_conditional_outcome_engine,
)
from pillar2_market_memory_engine.stability_engine import (  # noqa: E402
    StabilityConfig,
    build_stability_diagnostics,
)
from pillar2_market_memory_engine.session_memory_engine import (  # noqa: E402
    SessionMemoryConfig,
    run_session_memory_engine,
)
from pillar2_market_memory_engine.calendar_memory_engine import (  # noqa: E402
    CalendarMemoryConfig,
    run_calendar_memory_engine,
)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Pillar2OutputConfig:
    asset: str = "BTCUSDT"
    include_base_analogs_in_result: bool = False
    round_decimals: int = 6
    output_json_path: Optional[str] = None


# ============================================================
# HELPERS
# ============================================================

def _clean_scalar(value: Any, round_decimals: int) -> Any:
    if value is None:
        return None

    if isinstance(value, (bool, str, int)):
        return value

    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return round(value, round_decimals)

    if isinstance(value, np.floating):
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return round(value, round_decimals)

    if isinstance(value, np.integer):
        return int(value)

    return value


def _clean_dict(d: Dict[str, Any], round_decimals: int) -> Dict[str, Any]:
    return {k: _clean_scalar(v, round_decimals) for k, v in d.items()}


def _merge_risk_flags(*flag_lists: list[str]) -> list[str]:
    seen = set()
    merged: list[str] = []

    for flags in flag_lists:
        for flag in flags:
            if flag and flag not in seen:
                seen.add(flag)
                merged.append(flag)

    return merged


def _build_context_memory(
    session_context_memory: Dict[str, Any],
    calendar_context_memory: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "session_tendency": session_context_memory.get("session_tendency"),
        "session_weekpart_tendency": session_context_memory.get("session_weekpart_tendency"),
        "session_weekday_tendency": session_context_memory.get("session_weekday_tendency"),
        "session_transition_tendency": session_context_memory.get("session_transition_tendency"),
        "calendar_tendency": calendar_context_memory.get("weekday_tendency"),
        "hour_tendency": calendar_context_memory.get("hour_tendency"),
        "weekpart_tendency": calendar_context_memory.get("weekpart_tendency"),
        "weekday_hour_tendency": calendar_context_memory.get("weekday_hour_tendency"),
        "open_window_tendency": calendar_context_memory.get("open_window_tendency"),
        "volatility_tendency": None,
        "event_tendency": None,
        "regime_tendency": None,
    }


def _build_ml_readiness(base_analogs_used_len: int) -> Dict[str, Any]:
    feature_completeness_score = 1.0

    if base_analogs_used_len <= 0:
        feature_completeness_score = 0.0
    elif base_analogs_used_len < 25:
        feature_completeness_score = 0.5
    elif base_analogs_used_len < 75:
        feature_completeness_score = 0.75

    return {
        "state_vector_available": True,
        "point_in_time_valid": True,
        "feature_completeness_score": feature_completeness_score,
        "embedding_ready": False,
        "leakage_risk_flag": False,
    }


# ============================================================
# FINAL BUILD
# ============================================================

def build_pillar2_output(
    conditional_payload: Dict[str, Any],
    stability_diagnostics: Dict[str, Any],
    session_payload: Dict[str, Any],
    calendar_payload: Dict[str, Any],
    cfg: Optional[Pillar2OutputConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = Pillar2OutputConfig()

    current_state_signature = conditional_payload["current_state_signature"]
    memory_summary = conditional_payload["memory_summary"]
    historical_analogs = conditional_payload["historical_analogs"]
    forward_outcomes = conditional_payload["forward_outcomes"]
    distribution_diagnostics = conditional_payload["distribution_diagnostics"]
    base_analogs_used = conditional_payload["base_analogs_used"]

    context_memory = _build_context_memory(
        session_context_memory=session_payload["session_context_memory"],
        calendar_context_memory=calendar_payload["calendar_context_memory"],
    )

    ml_readiness = _build_ml_readiness(len(base_analogs_used))

    risk_flags = _merge_risk_flags(
        conditional_payload.get("risk_flags", []),
    )

    output = {
        "asset": cfg.asset,
        "timestamp_utc": current_state_signature.get("timestamp_utc"),

        "memory_summary": _clean_dict(memory_summary, cfg.round_decimals),
        "current_state_signature": _clean_dict(current_state_signature, cfg.round_decimals),
        "historical_analogs": _clean_dict(historical_analogs, cfg.round_decimals),
        "forward_outcomes": _clean_dict(forward_outcomes, cfg.round_decimals),
        "distribution_diagnostics": _clean_dict(distribution_diagnostics, cfg.round_decimals),
        "stability_diagnostics": _clean_dict(stability_diagnostics, cfg.round_decimals),
        "context_memory": _clean_dict(context_memory, cfg.round_decimals),
        "ml_readiness": _clean_dict(ml_readiness, cfg.round_decimals),
        "risk_flags": risk_flags,
        "ai_overview": None,
    }

    if cfg.include_base_analogs_in_result:
        output["base_analogs_preview_count"] = len(base_analogs_used)

    return output


def run_pillar2_output(
    feature_config: Optional[MemoryFeatureConfig] = None,
    outcome_config: Optional[MemoryOutcomeConfig] = None,
    signature_config: Optional[StateSignatureConfig] = None,
    retrieval_config: Optional[AnalogRetrievalConfig] = None,
    conditional_config: Optional[ConditionalOutcomeConfig] = None,
    stability_config: Optional[StabilityConfig] = None,
    session_config: Optional[SessionMemoryConfig] = None,
    calendar_config: Optional[CalendarMemoryConfig] = None,
    output_config: Optional[Pillar2OutputConfig] = None,
) -> Dict[str, Any]:
    if feature_config is None:
        feature_config = MemoryFeatureConfig()

    if outcome_config is None:
        outcome_config = MemoryOutcomeConfig()

    if signature_config is None:
        signature_config = StateSignatureConfig()

    if retrieval_config is None:
        retrieval_config = AnalogRetrievalConfig()

    if conditional_config is None:
        conditional_config = ConditionalOutcomeConfig()

    if stability_config is None:
        stability_config = StabilityConfig()

    if session_config is None:
        session_config = SessionMemoryConfig()

    if calendar_config is None:
        calendar_config = CalendarMemoryConfig()

    if output_config is None:
        output_config = Pillar2OutputConfig()

    conditional_payload = run_conditional_outcome_engine(
        feature_config=feature_config,
        outcome_config=outcome_config,
        signature_config=signature_config,
        retrieval_config=retrieval_config,
        conditional_config=conditional_config,
    )

    stability_diagnostics = build_stability_diagnostics(
        conditional_payload["base_analogs_used"],
        cfg=stability_config,
        score_column=conditional_config.weighted_score_column,
    )

    session_payload = run_session_memory_engine(
        feature_config=feature_config,
        outcome_config=outcome_config,
        signature_config=signature_config,
        session_config=session_config,
    )

    calendar_payload = run_calendar_memory_engine(
        feature_config=feature_config,
        outcome_config=outcome_config,
        signature_config=signature_config,
        calendar_config=calendar_config,
    )

    output = build_pillar2_output(
        conditional_payload=conditional_payload,
        stability_diagnostics=stability_diagnostics,
        session_payload=session_payload,
        calendar_payload=calendar_payload,
        cfg=output_config,
    )

    if output_config.output_json_path:
        out_path = Path(output_config.output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    return output


if __name__ == "__main__":
    feature_cfg = MemoryFeatureConfig(
        db_path="database/btc_terminal.db",
        table_name="btc_price_1h",
        timestamp_col="timestamp",
        output_parquet_path=None,
    )

    outcome_cfg = MemoryOutcomeConfig(
        horizons=(1, 3, 6, 12),
        output_parquet_path=None,
    )

    signature_cfg = StateSignatureConfig(
        require_complete_core_state=True,
    )

    retrieval_cfg = AnalogRetrievalConfig(
        min_similarity_score=0.45,
        max_weighted_matches=300,
        exclude_current_bar=True,
    )

    conditional_cfg = ConditionalOutcomeConfig(
        use_weighted_matches=True,
    )

    stability_cfg = StabilityConfig(
        minimum_window_samples=10,
    )

    session_cfg = SessionMemoryConfig(
        min_samples_per_group=25,
    )

    calendar_cfg = CalendarMemoryConfig(
        min_samples_per_group=25,
    )

    output_cfg = Pillar2OutputConfig(
        asset="BTCUSDT",
        include_base_analogs_in_result=False,
        round_decimals=6,
        output_json_path=None,
    )

    result = run_pillar2_output(
        feature_config=feature_cfg,
        outcome_config=outcome_cfg,
        signature_config=signature_cfg,
        retrieval_config=retrieval_cfg,
        conditional_config=conditional_cfg,
        stability_config=stability_cfg,
        session_config=session_cfg,
        calendar_config=calendar_cfg,
        output_config=output_cfg,
    )

    print("\n=== PILLAR 2 OUTPUT SUCCESS ===\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))