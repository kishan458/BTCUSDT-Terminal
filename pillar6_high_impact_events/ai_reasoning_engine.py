import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()


def _fallback_reasoning(payload: dict[str, Any]) -> str:
    event_name          = payload.get("event_name", "Unknown event")
    state               = payload.get("state", "UNKNOWN")
    base_uncertainty    = payload.get("base_uncertainty", 0.0)
    confidence_score    = payload.get("confidence_score", 0.0)
    dominant_risk_skew  = payload.get("dominant_risk_skew", "UNKNOWN")
    trade_restrictions  = payload.get("trade_restrictions", {})

    uncertainty_label = (
        "high"     if base_uncertainty  >= 0.75 else
        "moderate" if base_uncertainty  >= 0.55 else
        "lower"
    )
    confidence_label = (
        "strong"   if confidence_score  >= 0.75 else
        "moderate" if confidence_score  >= 0.55 else
        "limited"
    )

    return (
        f"{event_name} is currently in {state} state. "
        f"Macro uncertainty is {uncertainty_label} and model confidence is {confidence_label}. "
        f"Current risk skew is {dominant_risk_skew}. "
        f"Trade settings — allow_trade={trade_restrictions.get('allow_trade')}, "
        f"size_multiplier={trade_restrictions.get('size_multiplier')}, "
        f"leverage_cap={trade_restrictions.get('leverage_cap')}. "
        f"(Fallback reasoning — AI layer unavailable.)"
    )


def build_ai_reasoning(payload: dict[str, Any]) -> str:
    """
    Groq-powered macro commentary for:
    - macro summary
    - trader commentary
    - scenario explanation

    Falls back to rule-based text if Groq is unavailable.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _fallback_reasoning(payload)

    try:
        from groq import Groq
    except ImportError:
        return _fallback_reasoning(payload)

    try:
        client = Groq(api_key=api_key)

        event_name          = payload.get("event_name")
        state               = payload.get("state")
        base_uncertainty    = payload.get("base_uncertainty")
        confidence_score    = payload.get("confidence_score")
        dominant_risk_skew  = payload.get("dominant_risk_skew")
        trade_restrictions  = payload.get("trade_restrictions", {})
        scenarios           = payload.get("scenarios", [])

        scenario_lines = []
        for i, s in enumerate(scenarios, start=1):
            scenario_lines.append(
                f"{i}. case={s.get('case')}; "
                f"probability={s.get('probability')}; "
                f"risk_bias={s.get('risk_bias')}; "
                f"macro_interpretation={s.get('macro_interpretation')}; "
                f"btc_initial_move={s.get('expected_btc_reaction', {}).get('initial_move')}; "
                f"btc_volatility={s.get('expected_btc_reaction', {}).get('volatility')}"
            )

        prompt = f"""You are a macro trading analyst writing BTC-specific commentary for a professional trading terminal.

Write a concise response with exactly 3 short paragraphs:
1) Macro summary — what this event means for markets
2) Trader commentary — what a BTC trader should be thinking right now
3) Scenario explanation — briefly explain the dominant scenario and its implications

Rules:
- Be concrete, not fluffy.
- Do not invent data not in the input.
- Mention whether risk conditions favor caution or selective participation.
- Keep it under 170 words total.
- No bullet points. Prose only.

Structured input:
event_name: {event_name}
state: {state}
base_uncertainty: {base_uncertainty}
confidence_score: {confidence_score}
dominant_risk_skew: {dominant_risk_skew}
allow_trade: {trade_restrictions.get('allow_trade')}
size_multiplier: {trade_restrictions.get('size_multiplier')}
leverage_cap: {trade_restrictions.get('leverage_cap')}

Scenarios:
{chr(10).join(scenario_lines)}"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior macro trading analyst writing professional, "
                        "concise BTC-focused commentary for a Bloomberg-style terminal. "
                        "Be direct, data-driven, and avoid hype."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.3,
            max_tokens=300,
        )

        text = response.choices[0].message.content
        if text and text.strip():
            return text.strip()

        return _fallback_reasoning(payload)

    except Exception as e:
        print(f"[AI Reasoning ERROR] {e}")
        return _fallback_reasoning(payload)