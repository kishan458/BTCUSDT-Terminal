from __future__ import annotations

import json

from orchestrator.orchestrator_engine import TerminalOrchestrator
from orchestrator.schema import build_pillar_wrapper


def build_smoke_test_inputs() -> dict:
    """
    Minimal smoke-test inputs for orchestrator execution.

    This is only for local runner validation.
    It is not production/live integration data.
    """
    return {
        "pillar_1_sentiment": build_pillar_wrapper(
            "pillar_1_sentiment",
            status="OK",
            confidence=0.74,
            summary="Sentiment is constructive.",
            bias="BULLISH",
            strength=0.61,
            state="CAUTIOUS_BULLISH",
            is_fresh=True,
            is_complete=True,
            freshness_status="FRESH",
            payload={
                "metrics": {"institutional_sentiment_score": 0.64},
                "key_drivers": ["Institutional tone remains supportive"],
            },
        ),
        "pillar_5_regime": build_pillar_wrapper(
            "pillar_5_regime",
            status="OK",
            confidence=0.76,
            summary="Trend regime remains favorable.",
            bias="BULLISH",
            strength=0.76,
            state="TREND_FAVORABLE",
            is_fresh=True,
            is_complete=True,
            freshness_status="FRESH",
            payload={
                "metrics": {
                    "regime_state": "TRENDING",
                    "cycle_phase": "MID_BULL",
                    "volatility_regime": "ELEVATED",
                }
            },
        ),
        "pillar_6_events": build_pillar_wrapper(
            "pillar_6_events",
            status="OK",
            confidence=0.58,
            summary="Event risk is elevated.",
            bias="BEARISH",
            strength=0.58,
            state="EVENT_RISK_ELEVATED",
            is_fresh=True,
            is_complete=True,
            freshness_status="FRESH",
            payload={
                "conflicts": ["Macro event risk conflicts with trend continuation"],
                "restrictions": ["Reduce size near major releases"],
            },
        ),
        "pillar_8_decision": build_pillar_wrapper(
            "pillar_8_decision",
            status="OK",
            confidence=0.69,
            summary="Trade allowed with disciplined sizing.",
            bias="BULLISH",
            strength=0.69,
            state="TRADE_ALLOWED",
            is_fresh=True,
            is_complete=True,
            freshness_status="FRESH",
            payload={
                "metrics": {
                    "size_fraction": 0.05,
                    "max_leverage": 2.0,
                    "tradability": "MODERATE",
                    "risk_level": "MEDIUM",
                },
                "decision": {
                    "action": "LONG",
                    "execution_style": "STANDARD_ENTRY",
                    "thesis": "Bullish regime remains intact but event risk caps conviction.",
                    "time_horizon": "SHORT_SWING",
                },
                "ai_summary": {
                    "short_summary": "Constructive setup with macro risk overhead.",
                },
            },
        ),
    }


def main() -> None:
    orchestrator = TerminalOrchestrator()
    payload = orchestrator.run(build_smoke_test_inputs())

    summary = {
        "run_id": payload["metadata"]["run_id"],
        "system_status": payload["system_health"]["status"],
        "directional_bias": payload["market_snapshot"]["directional_bias"],
        "decision_action": payload["control_panel"]["decision"]["action"],
        "notes": payload["metadata"]["notes"],
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()