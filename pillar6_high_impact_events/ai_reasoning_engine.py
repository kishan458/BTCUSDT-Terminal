import os
from typing import Any


def _fallback_reasoning(payload: dict[str, Any]) -> str:
    event_name = payload.get("event_name", "Unknown event")
    state = payload.get("state", "UNKNOWN")
    base_uncertainty = payload.get("base_uncertainty", 0.0)
    confidence_score = payload.get("confidence_score", 0.0)
    dominant_risk_skew = payload.get("dominant_risk_skew", "UNKNOWN")
    trade_restrictions = payload.get("trade_restrictions", {})

    uncertainty_label = (
        "high" if base_uncertainty >= 0.75 else
        "moderate" if base_uncertainty >= 0.55 else
        "lower"
    )
    confidence_label = (
        "strong" if confidence_score >= 0.75 else
        "moderate" if confidence_score >= 0.55 else
        "limited"
    )

    return (
        f"{event_name} is currently in {state} state. "
        f"Macro uncertainty is {uncertainty_label} and model confidence is {confidence_label}. "
        f"Current risk skew is {dominant_risk_skew}. "
        f"Trade settings currently allow_trade={trade_restrictions.get('allow_trade')}, "
        f"size_multiplier={trade_restrictions.get('size_multiplier')}, "
        f"leverage_cap={trade_restrictions.get('leverage_cap')}. "
        f"This is fallback reasoning because Gemini output was unavailable."
    )


def build_ai_reasoning(payload: dict[str, Any]) -> str:
    """
    Gemini-powered commentary layer for:
    - macro summary
    - trader commentary
    - explanation of scenario outputs

    Safe fallback is returned if:
    - google-genai is not installed
    - GEMINI_API_KEY is missing
    - Gemini request fails
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _fallback_reasoning(payload)

    try:
        from google import genai
    except Exception:
        return _fallback_reasoning(payload)

    try:
        client = genai.Client(api_key=api_key)

        event_name = payload.get("event_name")
        state = payload.get("state")
        base_uncertainty = payload.get("base_uncertainty")
        confidence_score = payload.get("confidence_score")
        dominant_risk_skew = payload.get("dominant_risk_skew")
        trade_restrictions = payload.get("trade_restrictions", {})
        scenarios = payload.get("scenarios", [])

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

        prompt = f"""
You are a macro trading analyst writing BTC-specific commentary for a trading terminal.

Write a concise response with exactly 3 short paragraphs:
1) Macro summary
2) Trader commentary
3) Scenario explanation

Rules:
- Be concrete, not fluffy.
- Do not invent data.
- Use only the structured input below.
- Mention whether risk conditions favor caution or selective participation.
- Keep it under 170 words total.
- No bullet points.

Structured input:
event_name: {event_name}
state: {state}
base_uncertainty: {base_uncertainty}
confidence_score: {confidence_score}
dominant_risk_skew: {dominant_risk_skew}
trade_restrictions: {trade_restrictions}

Scenarios:
{chr(10).join(scenario_lines)}
""".strip()

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )

        text = getattr(response, "text", None)
        if text and text.strip():
            return text.strip()

        return _fallback_reasoning(payload)

    except Exception:
        return _fallback_reasoning(payload)