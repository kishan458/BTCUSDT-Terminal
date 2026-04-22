import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()


def _fallback_overview(payload: dict[str, Any]) -> str:
    regime   = payload.get("regime_summary", {})
    strategy = payload.get("strategy_compatibility", {})
    metrics  = payload.get("market_metrics", {})
    risk_flags = payload.get("risk_flags", [])

    directional  = regime.get("directional_regime", "UNKNOWN")
    volatility   = regime.get("volatility_regime", "UNKNOWN")
    market_state = regime.get("market_state", "UNKNOWN")
    cycle_phase  = regime.get("cycle_phase", "UNKNOWN")

    trend_following  = strategy.get("trend_following", "NEUTRAL")
    breakout_trading = strategy.get("breakout_trading", "NEUTRAL")
    mean_reversion   = strategy.get("mean_reversion", "NEUTRAL")

    ma       = metrics.get("moving_average_structure", {})
    momentum = metrics.get("momentum", {})
    returns  = metrics.get("returns", {})
    swing    = metrics.get("swing_structure", {})
    vol      = metrics.get("volatility", {})

    risk_text = ", ".join(risk_flags) if risk_flags else "No major regime risks identified"

    return (
        f"BTC is trading in a {directional} regime with a {market_state} state and a broader {cycle_phase} profile. "
        f"Moving averages are aligned at {ma.get('ma_order', 'UNKNOWN')}, "
        f"swing structure is {swing.get('structure_state', 'UNKNOWN')}, "
        f"volatility regime is {volatility}.\n\n"
        f"Strategy fit: trend_following={trend_following}, breakout_trading={breakout_trading}, "
        f"mean_reversion={mean_reversion}. "
        f"24-bar return={returns.get('return_24bar')}, 7d return={returns.get('return_7d')}, "
        f"momentum score={momentum.get('momentum_score')}, "
        f"volatility percentile={vol.get('volatility_percentile')}.\n\n"
        f"Primary regime risks: {risk_text}."
    )


def build_ai_overview(payload: dict[str, Any]) -> str:
    """
    Groq-powered Pillar 5 regime commentary.
    Falls back to rule-based text if Groq is unavailable.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _fallback_overview(payload)

    try:
        from groq import Groq
    except ImportError:
        return _fallback_overview(payload)

    try:
        client = Groq(api_key=api_key)

        regime     = payload.get("regime_summary", {})
        strategy   = payload.get("strategy_compatibility", {})
        session    = payload.get("session_context", {})
        metrics    = payload.get("market_metrics", {})
        risk_flags = payload.get("risk_flags", [])
        confidence = payload.get("confidence_score")

        prompt = f"""You are a professional BTC quantitative strategist writing a regime note for an advanced trading terminal.

Write exactly 3 paragraphs:
1) Regime diagnosis — what the current market structure looks like
2) Trading implication — what a serious trader should do right now
3) Risk / failure mode — what would invalidate the current thesis

Rules:
- Sound like a serious market professional who studies BTC structure, volatility, and trend behavior full-time.
- Be detailed but efficient — every sentence must carry signal.
- Do not summarize fields mechanically. Infer what matters from the structure.
- Mention chart structure, moving average alignment, momentum, volatility behavior, and strategy quality.
- No bullet points. No hype. No emojis.
- 170 to 230 words total.

Structured input:
directional_regime: {regime.get("directional_regime")}
volatility_regime: {regime.get("volatility_regime")}
market_state: {regime.get("market_state")}
cycle_phase: {regime.get("cycle_phase")}
confidence_score: {confidence}

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

risk_flags: {risk_flags}"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior BTC quantitative strategist writing professional, "
                        "high-signal regime notes for a Bloomberg-style trading terminal. "
                        "Be direct, structured, and avoid generic language."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.3,
            max_tokens=400,
        )

        text = response.choices[0].message.content
        if text and text.strip():
            return text.strip()

        return _fallback_overview(payload)

    except Exception as e:
        print(f"[AI Overview ERROR] {e}")
        return _fallback_overview(payload)