import os
from typing import Any


def _fallback_overview(payload: dict[str, Any]) -> str:
    regime = payload.get("regime_summary", {})
    strategy = payload.get("strategy_compatibility", {})
    metrics = payload.get("market_metrics", {})
    risk_flags = payload.get("risk_flags", [])

    directional = regime.get("directional_regime", "UNKNOWN")
    volatility = regime.get("volatility_regime", "UNKNOWN")
    market_state = regime.get("market_state", "UNKNOWN")
    cycle_phase = regime.get("cycle_phase", "UNKNOWN")

    trend_following = strategy.get("trend_following", "NEUTRAL")
    breakout_trading = strategy.get("breakout_trading", "NEUTRAL")
    mean_reversion = strategy.get("mean_reversion", "NEUTRAL")

    ma = metrics.get("moving_average_structure", {})
    momentum = metrics.get("momentum", {})
    returns = metrics.get("returns", {})
    swing = metrics.get("swing_structure", {})
    vol = metrics.get("volatility", {})

    risk_text = ", ".join(risk_flags) if risk_flags else "No major regime risks identified"

    return (
        f"BTC is trading in a {directional} regime with a {market_state} state and a broader {cycle_phase} profile. "
        f"The structure remains constructive: moving averages are aligned at {ma.get('ma_order', 'UNKNOWN')}, "
        f"price is holding above the key trend references, and swing structure is {swing.get('structure_state', 'UNKNOWN')}. "
        f"Short-horizon return expansion and a positive momentum profile suggest the path of least resistance is still upward, "
        f"but this is happening in a {volatility} volatility regime rather than a clean, orderly trend.\n\n"
        f"From a trading perspective, this keeps trend participation as the primary playbook while reducing the quality of breakout chasing. "
        f"Strategy fit currently favors trend_following={trend_following}, breakout_trading={breakout_trading}, and "
        f"mean_reversion={mean_reversion}. The combination of elevated ATR, stretched positioning versus short-term averages, "
        f"and unstable volatility means continuation can still work, but entries need to be selective and risk must be wider and cleaner than usual.\n\n"
        f"The main issue is not directional weakness but regime instability. "
        f"24-bar return is {returns.get('return_24bar')}, 7-day return is {returns.get('return_7d')}, "
        f"momentum score is {momentum.get('momentum_score')}, and volatility percentile is {vol.get('volatility_percentile')}. "
        f"That combination argues for respecting trend strength without confusing it for a low-risk environment. "
        f"Primary regime risks: {risk_text}."
    )


def build_ai_overview(payload: dict[str, Any]) -> str:
    """
    Gemini-powered Pillar 5 regime commentary.

    Returns fallback overview if:
    - GEMINI_API_KEY is missing
    - google-genai is not installed
    - Gemini request fails
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _fallback_overview(payload)

    try:
        from google import genai
    except Exception:
        return _fallback_overview(payload)

    try:
        client = genai.Client(api_key=api_key)

        regime = payload.get("regime_summary", {})
        strategy = payload.get("strategy_compatibility", {})
        session = payload.get("session_context", {})
        metrics = payload.get("market_metrics", {})
        risk_flags = payload.get("risk_flags", [])
        confidence_score = payload.get("confidence_score")

        prompt = f"""
You are a professional BTC quantitative strategist writing a regime note for an advanced trading terminal.

Write exactly 3 paragraphs:
1) Regime diagnosis
2) Trading implication
3) Risk / failure mode

Rules:
- Sound like a serious market professional who studies BTC structure, volatility, and trend behavior full-time.
- Be detailed, but still efficient and high-signal.
- Do not summarize fields mechanically.
- Infer what matters from the structure.
- Mention chart structure, moving average alignment, momentum, volatility behavior, and strategy quality.
- Avoid generic chatbot language.
- No bullet points.
- No hype, no emojis.
- Aim for roughly 170 to 230 words total.

Structured input:
directional_regime: {regime.get("directional_regime")}
volatility_regime: {regime.get("volatility_regime")}
market_state: {regime.get("market_state")}
cycle_phase: {regime.get("cycle_phase")}
confidence_score: {confidence_score}

trend_following: {strategy.get("trend_following")}
breakout_trading: {strategy.get("breakout_trading")}
mean_reversion: {strategy.get("mean_reversion")}
stand_down: {strategy.get("stand_down")}

current_session: {session.get("current_session")}
session_high: {session.get("session_high")}
session_low: {session.get("session_low")}

ohlcv: {metrics.get("ohlcv")}
returns: {metrics.get("returns")}
volatility: {metrics.get("volatility")}
moving_average_structure: {metrics.get("moving_average_structure")}
momentum: {metrics.get("momentum")}
compression_expansion: {metrics.get("compression_expansion")}
swing_structure: {metrics.get("swing_structure")}
distance_from_key_mas: {metrics.get("distance_from_key_mas")}

risk_flags: {risk_flags}
""".strip()

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )

        text = getattr(response, "text", None)
        if text and text.strip():
            return text.strip()

        return _fallback_overview(payload)

    except Exception:
        return _fallback_overview(payload)