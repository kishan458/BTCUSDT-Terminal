from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import json
import os
import sys


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
)
from pillar2_market_memory_engine.stability_engine import (  # noqa: E402
    StabilityConfig,
)
from pillar2_market_memory_engine.session_memory_engine import (  # noqa: E402
    SessionMemoryConfig,
)
from pillar2_market_memory_engine.calendar_memory_engine import (  # noqa: E402
    CalendarMemoryConfig,
)
from pillar2_market_memory_engine.pillar2_output import (  # noqa: E402
    Pillar2OutputConfig,
    run_pillar2_output,
)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class AIOverviewConfig:
    model_name: str = "gemini-2.5-flash"
    enable_google_search_grounding: bool = True
    max_prompt_chars: int = 18000
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
    signature = payload.get("current_state_signature", {})
    analogs = payload.get("historical_analogs", {})
    forward = payload.get("forward_outcomes", {})
    dist = payload.get("distribution_diagnostics", {})
    stability = payload.get("stability_diagnostics", {})
    context = payload.get("context_memory", {})
    risk_flags = payload.get("risk_flags", [])

    memory_bias = _safe(memory_summary.get("memory_bias"))
    match_quality = _safe(memory_summary.get("historical_match_quality"))
    sample_size = _safe(memory_summary.get("sample_size"))
    confidence = _safe(memory_summary.get("headline_confidence"))

    session = _safe(signature.get("session"))
    weekday = _safe(signature.get("weekday"))
    vol_bucket = _safe(signature.get("volatility_bucket"))
    momentum = _safe(signature.get("momentum_state"))
    overlap = _safe(signature.get("overlap_state"))
    follow = _safe(signature.get("follow_through_quality"))
    pressure = _safe(signature.get("pressure_bias"))

    next_3_up = _safe(forward.get("next_3_bar_up_probability"))
    next_6_up = _safe(forward.get("next_6_bar_up_probability"))
    mean_ret_6 = _safe(forward.get("mean_forward_return_6"))
    mean_mfe_6 = _safe(forward.get("mean_mfe_6"))
    mean_mae_6 = _safe(forward.get("mean_mae_6"))
    continuation = _safe(forward.get("continuation_probability"))
    reversal = _safe(forward.get("reversal_probability"))
    mean_reversion = _safe(forward.get("mean_reversion_probability"))

    std_6 = _safe(dist.get("return_std_6"))
    left_tail = _safe(dist.get("left_tail_10pct_6"))
    right_tail = _safe(dist.get("right_tail_90pct_6"))

    older_bias = _safe(stability.get("older_window_bias"))
    middle_bias = _safe(stability.get("middle_window_bias"))
    recent_bias = _safe(stability.get("recent_window_bias"))
    temporal_stability = _safe(stability.get("temporal_stability_score"))
    regime_dependency = _safe(stability.get("regime_dependency_score"))

    session_tendency = _safe(context.get("session_tendency"))
    calendar_tendency = _safe(context.get("calendar_tendency"))
    weekday_hour_tendency = _safe(context.get("weekday_hour_tendency"))

    risk_text = ", ".join(risk_flags) if risk_flags else "No major memory warning detected"

    return (
        f"BTC is currently being classified through a {match_quality} quality historical memory set with sample size {sample_size} "
        f"and headline confidence {confidence}. The active state is a {vol_bucket} volatility, {momentum} momentum, {overlap} overlap profile "
        f"with {follow} follow-through and {pressure} pressure during the {session} session on {weekday}. The memory engine is leaning "
        f"toward {memory_bias}, but the edge is conditional rather than absolute: next-3-bar up probability is {next_3_up}, next-6-bar up probability is {next_6_up}, "
        f"and mean forward 6-bar return is {mean_ret_6}. Path quality matters here because mean favorable excursion is {mean_mfe_6} while mean adverse excursion is {mean_mae_6}.\n\n"
        f"The market implication is that this state does not justify blind directional conviction. Continuation probability is {continuation}, reversal probability is {reversal}, "
        f"and mean-reversion probability is {mean_reversion}, which keeps the setup more consistent with reaction trading than with unconditional trend-chasing. "
        f"Distributionally, 6-bar return dispersion is {std_6}, with the left 10th percentile at {left_tail} and the right 90th percentile at {right_tail}. "
        f"Session and calendar overlays also lean in the same direction: session tendency is {session_tendency}, calendar tendency is {calendar_tendency}, "
        f"and the current weekday-hour tendency is {weekday_hour_tendency}.\n\n"
        f"The important control variable is stability. Older, middle, and recent analog windows show {older_bias}, {middle_bias}, and {recent_bias} respectively, "
        f"with temporal stability score {temporal_stability} and regime dependency score {regime_dependency}. That means this memory edge should be treated as a probabilistic state prior, "
        f"not as a deterministic forecast. For the next pillar, the best handoff is to feed this payload into the ML council as a structured prior: use memory bias, analog quality, "
        f"stability, dispersion, and context tendencies as features for model weighting, disagreement detection, and confidence scaling. Main flags: {risk_text}."
    )


def _build_prompt(payload: dict[str, Any]) -> str:
    compact_payload = _compact_json(payload, 18000)

    return f"""
You are the senior BTC quantitative strategist on an elite prop desk writing for an institutional BTC/USDT intelligence terminal.

You are reviewing a fully computed Pillar 2 Market Memory payload.
The structured payload is the primary truth source.
If Google Search grounding is enabled, you may use fresh web context only to sharpen broad market framing and the handoff into the next pillar.
Never contradict the payload.
Never invent statistics.
Never claim certainty.
Never make deterministic predictions.

Write exactly 3 paragraphs:

Paragraph 1:
Diagnose the current BTC memory state.
Explain what class of state this is, how good the analog set is, and what the memory bias actually means in practical terms.

Paragraph 2:
Explain the trade implication.
Discuss continuation vs reversal vs mean reversion, path quality, excursion asymmetry, dispersion, and whether the edge is robust or fragile.
Write like a sharp quant desk note, not a generic analyst summary.

Paragraph 3:
Explain the failure mode and the handoff into the next pillar.
Treat the next pillar as the ML / model-council / decision-layer handoff.
Explain how a serious model stack should use this memory payload as:
- a probabilistic prior
- a confidence modifier
- a disagreement detector
- a regime sensitivity input
- a feature source for model weighting

Rules:
- No bullet points.
- No emojis.
- No hype.
- No “this guarantees”.
- No vague TA fluff.
- Use exact payload values where relevant.
- Prefer terms like probabilistic, conditional, asymmetric, unstable, dispersion, path, prior, handoff, regime.
- Tone: high-signal, compressed, institutional, BTC-native.
- Roughly 190 to 260 words total.
- Do not sound like a chatbot.
- Do not summarize fields mechanically.

Structured Pillar 2 payload:
{compact_payload}
""".strip()


def _try_gemini_overview(
    payload: dict[str, Any],
    cfg: AIOverviewConfig,
) -> tuple[Optional[str], Optional[str], str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None, "GEMINI_API_KEY missing", "fallback"

    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        return None, f"google-genai import failed: {type(e).__name__}: {e}", "fallback"

    prompt = _build_prompt(payload)

    try:
        client = genai.Client(api_key=api_key)

        if cfg.enable_google_search_grounding:
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            gen_config = types.GenerateContentConfig(
                tools=[grounding_tool]
            )

            response = client.models.generate_content(
                model=cfg.model_name,
                contents=prompt,
                config=gen_config,
            )
        else:
            response = client.models.generate_content(
                model=cfg.model_name,
                contents=prompt,
            )

        text = getattr(response, "text", None)
        if text and text.strip():
            return text.strip(), None, "gemini"

        return None, "Gemini returned empty text", "fallback"

    except Exception as e:
        # Retry once without grounding
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=cfg.model_name,
                contents=prompt,
            )
            text = getattr(response, "text", None)
            if text and text.strip():
                return text.strip(), f"Grounded call failed, non-grounded retry succeeded: {type(e).__name__}: {e}", "gemini"
            return None, f"Gemini retry returned empty text after error: {type(e).__name__}: {e}", "fallback"
        except Exception as e2:
            return None, f"Gemini failed: {type(e).__name__}: {e} | Retry failed: {type(e2).__name__}: {e2}", "fallback"


# ============================================================
# PUBLIC API
# ============================================================

def build_ai_overview(
    payload: dict[str, Any],
    cfg: Optional[AIOverviewConfig] = None,
) -> tuple[str, str, Optional[str]]:
    if cfg is None:
        cfg = AIOverviewConfig()

    text, error, source = _try_gemini_overview(payload, cfg)
    if text:
        return text, source, error

    return _fallback_overview(payload), "fallback", error


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

    payload["ai_overview"] = overview_text
    payload["overview_source"] = overview_source
    payload["overview_error"] = overview_error

    if ai_config.output_json_path:
        out_path = Path(ai_config.output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload


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

    ai_cfg = AIOverviewConfig(
        model_name="gemini-2.5-flash",
        enable_google_search_grounding=True,
        output_json_path=None,
    )

    result = run_ai_overview_engine(
        feature_config=feature_cfg,
        outcome_config=outcome_cfg,
        signature_config=signature_cfg,
        retrieval_config=retrieval_cfg,
        conditional_config=conditional_cfg,
        stability_config=stability_cfg,
        session_config=session_cfg,
        calendar_config=calendar_cfg,
        output_config=output_cfg,
        ai_config=ai_cfg,
    )

    print("\n=== AI OVERVIEW ENGINE SUCCESS ===\n")
    print("overview_source:", result.get("overview_source"))
    print("overview_error:", result.get("overview_error"))
    print("\nAI OVERVIEW:\n")
    print(result["ai_overview"])