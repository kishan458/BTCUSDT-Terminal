from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# IMPORT FIX FOR DIRECT SCRIPT RUN
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT    = CURRENT_FILE.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pillar2_market_memory_engine.memory_feature_engine import MemoryFeatureConfig  # noqa: E402
from pillar2_market_memory_engine.memory_outcome_engine import MemoryOutcomeConfig  # noqa: E402
from pillar2_market_memory_engine.state_signature_engine import StateSignatureConfig  # noqa: E402
from pillar2_market_memory_engine.analog_retrieval_engine import AnalogRetrievalConfig  # noqa: E402
from pillar2_market_memory_engine.conditional_outcome_engine import ConditionalOutcomeConfig  # noqa: E402
from pillar2_market_memory_engine.stability_engine import StabilityConfig  # noqa: E402
from pillar2_market_memory_engine.session_memory_engine import SessionMemoryConfig  # noqa: E402
from pillar2_market_memory_engine.calendar_memory_engine import CalendarMemoryConfig  # noqa: E402
from pillar2_market_memory_engine.pillar2_output import Pillar2OutputConfig, run_pillar2_output  # noqa: E402


# ============================================================
# CONFIG
# ============================================================

@dataclass
class AIOverviewConfig:
    max_prompt_chars: int = 12000
    output_json_path: Optional[str] = None


# ============================================================
# HELPERS
# ============================================================

def _safe(value: Any, default: str = "UNKNOWN") -> Any:
    if value is None:
        return default
    return value


def _compact_json(payload: dict[str, Any], max_chars: int) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _fallback_overview(payload: dict[str, Any]) -> str:
    memory_summary = payload.get("memory_summary", {})
    signature      = payload.get("current_state_signature", {})
    forward        = payload.get("forward_outcomes", {})
    dist           = payload.get("distribution_diagnostics", {})
    stability      = payload.get("stability_diagnostics", {})
    context        = payload.get("context_memory", {})
    risk_flags     = payload.get("risk_flags", [])

    memory_bias   = _safe(memory_summary.get("memory_bias"))
    match_quality = _safe(memory_summary.get("historical_match_quality"))
    sample_size   = _safe(memory_summary.get("sample_size"))
    confidence    = _safe(memory_summary.get("headline_confidence"))

    session   = _safe(signature.get("session"))
    weekday   = _safe(signature.get("weekday"))
    vol_bucket = _safe(signature.get("volatility_bucket"))
    momentum  = _safe(signature.get("momentum_state"))
    overlap   = _safe(signature.get("overlap_state"))
    follow    = _safe(signature.get("follow_through_quality"))
    pressure  = _safe(signature.get("pressure_bias"))

    next_3_up    = _safe(forward.get("next_3_bar_up_probability"))
    next_6_up    = _safe(forward.get("next_6_bar_up_probability"))
    mean_ret_6   = _safe(forward.get("mean_forward_return_6"))
    mean_mfe_6   = _safe(forward.get("mean_mfe_6"))
    mean_mae_6   = _safe(forward.get("mean_mae_6"))
    continuation = _safe(forward.get("continuation_probability"))
    reversal     = _safe(forward.get("reversal_probability"))
    mean_rev     = _safe(forward.get("mean_reversion_probability"))

    std_6      = _safe(dist.get("return_std_6"))
    left_tail  = _safe(dist.get("left_tail_10pct_6"))
    right_tail = _safe(dist.get("right_tail_90pct_6"))

    older_bias        = _safe(stability.get("older_window_bias"))
    middle_bias       = _safe(stability.get("middle_window_bias"))
    recent_bias       = _safe(stability.get("recent_window_bias"))
    temporal_stab     = _safe(stability.get("temporal_stability_score"))
    regime_dep        = _safe(stability.get("regime_dependency_score"))

    session_tendency  = _safe(context.get("session_tendency"))
    calendar_tendency = _safe(context.get("calendar_tendency"))
    wdh_tendency      = _safe(context.get("weekday_hour_tendency"))

    risk_text = ", ".join(risk_flags) if risk_flags else "No major memory warning detected"

    return (
        f"BTC memory state: {match_quality} quality analog set (n={sample_size}, confidence={confidence}). "
        f"State profile: {vol_bucket} volatility, {momentum} momentum, {overlap} overlap, "
        f"{follow} follow-through, {pressure} pressure — {session} session on {weekday}. "
        f"Memory bias: {memory_bias}. "
        f"Next-3 up={next_3_up}, next-6 up={next_6_up}, mean 6-bar return={mean_ret_6}. "
        f"MFE={mean_mfe_6}, MAE={mean_mae_6}.\n\n"
        f"Path quality: continuation={continuation}, reversal={reversal}, mean-reversion={mean_rev}. "
        f"6-bar dispersion std={std_6}, left tail={left_tail}, right tail={right_tail}. "
        f"Session tendency={session_tendency}, calendar={calendar_tendency}, weekday-hour={wdh_tendency}.\n\n"
        f"Stability: older={older_bias}, middle={middle_bias}, recent={recent_bias}. "
        f"Temporal stability={temporal_stab}, regime dependency={regime_dep}. "
        f"Treat as probabilistic prior, not deterministic forecast. "
        f"Flags: {risk_text}."
    )


def _build_prompt(payload: dict[str, Any], max_chars: int) -> str:
    # Send focused subset to stay within token limits
    focused = {
        "memory_summary":          payload.get("memory_summary"),
        "current_state_signature": payload.get("current_state_signature"),
        "forward_outcomes":        payload.get("forward_outcomes"),
        "distribution_diagnostics": payload.get("distribution_diagnostics"),
        "stability_diagnostics":   payload.get("stability_diagnostics"),
        "context_memory":          payload.get("context_memory"),
        "risk_flags":              payload.get("risk_flags"),
        "historical_analogs":      payload.get("historical_analogs"),
    }
    compact = _compact_json(focused, max_chars)

    return f"""You are the senior BTC quantitative strategist on an elite prop desk writing for an institutional BTC/USDT intelligence terminal.

Write exactly 3 paragraphs:

1) Diagnose the current BTC memory state. Explain what class of state this is, how good the analog set is, and what the memory bias means in practical terms.

2) Explain the trade implication. Discuss continuation vs reversal vs mean reversion, path quality, excursion asymmetry, dispersion, and whether the edge is robust or fragile.

3) Explain the failure mode and handoff into the next pillar (ML council / decision layer). How should a serious model stack use this memory payload as a probabilistic prior, confidence modifier, and regime sensitivity input.

Rules:
- No bullet points. No emojis. No hype.
- Use exact payload values where relevant.
- Tone: high-signal, compressed, institutional, BTC-native.
- 190 to 260 words total.
- Do not summarize fields mechanically. Synthesize.

Structured Pillar 2 payload:
{compact}"""


# ============================================================
# PUBLIC API
# ============================================================

def build_ai_overview(
    payload: dict[str, Any],
    cfg: Optional[AIOverviewConfig] = None,
) -> tuple[str, str, Optional[str]]:
    if cfg is None:
        cfg = AIOverviewConfig()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _fallback_overview(payload), "fallback", "GROQ_API_KEY missing"

    try:
        from groq import Groq
    except ImportError:
        return _fallback_overview(payload), "fallback", "groq package not installed"

    try:
        client = Groq(api_key=api_key)
        prompt = _build_prompt(payload, cfg.max_prompt_chars)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior BTC quantitative strategist writing institutional-grade "
                        "market memory notes for a Bloomberg-style trading terminal. "
                        "Be direct, precise, and probabilistic. Never overstate conviction."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.3,
            max_tokens=500,
        )

        text = response.choices[0].message.content
        if text and text.strip():
            return text.strip(), "groq", None

        return _fallback_overview(payload), "fallback", "Groq returned empty text"

    except Exception as e:
        return _fallback_overview(payload), "fallback", f"Groq error: {type(e).__name__}: {e}"


def run_ai_overview_engine(
    feature_config: Optional[MemoryFeatureConfig] = None,
    outcome_config: Optional[MemoryOutcomeConfig] = None,
    signature_config: Optional[StateSignatureConfig] = None,
    retrieval_config: Optional[AnalogRetrievalConfig] = None,
    conditional_config: Optional[ConditionalOutcomeConfig] = None,
    stability_config: Optional[StabilityConfig] = None,
    session_config: Optional[SessionMemoryConfig] = None,
    calendar_config: Optional[CalendarMemoryConfig] = None,
    output_config: Optional[Pillar2OutputConfig] = None,
    ai_config: Optional[AIOverviewConfig] = None,
) -> dict[str, Any]:
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
    if ai_config is None:
        ai_config = AIOverviewConfig()

    payload = run_pillar2_output(
        feature_config=feature_config,
        outcome_config=outcome_config,
        signature_config=signature_config,
        retrieval_config=retrieval_config,
        conditional_config=conditional_config,
        stability_config=stability_config,
        session_config=session_config,
        calendar_config=calendar_config,
        output_config=output_config,
    )

    overview_text, overview_source, overview_error = build_ai_overview(payload, ai_config)

    payload["ai_overview"]     = overview_text
    payload["overview_source"] = overview_source
    payload["overview_error"]  = overview_error

    if ai_config.output_json_path:
        out_path = Path(ai_config.output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload