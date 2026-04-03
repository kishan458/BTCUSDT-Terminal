from pillar8_decision_risk_backtesting.state.decision_state_builder import build_decision_state
from pillar8_decision_risk_backtesting.decision.alignment_engine import compute_alignment
from pillar8_decision_risk_backtesting.decision.veto_engine import evaluate_vetoes
from pillar8_decision_risk_backtesting.decision.conviction_engine import compute_conviction
from pillar8_decision_risk_backtesting.decision.decision_gate_engine import run_decision_gate
from pillar8_decision_risk_backtesting.risk.risk_score_engine import (
    RiskComponentBreakdown,
    RiskScoreResult,
    compute_risk_score,
)
from pillar8_decision_risk_backtesting.state.decision_schema import RiskState


def main() -> None:
    state = build_decision_state(
        council_payload={
            "final_bias": "LONG",
            "confidence": 0.8,
            "agreement_score": 0.8,
            "conflict_score": 0.1,
        },
        regime_payload={
            "regime_state": "UPTREND",
            "strategy_compatibility": 0.8,
        },
        candle_payload={
            "dominant_intent": "BULLISH",
            "pressure_bias": "BULLISH",
            "breakout_quality": 0.7,
            "failure_risk": 0.2,
        },
        structure_payload={
            "structure_state": "HIGHER_HIGH_HIGHER_LOW",
            "trap_risk": 0.2,
            "liquidation_risk": 0.2,
        },
        sentiment_payload={
            "sentiment_state": "BULLISH",
            "confidence": 0.7,
        },
        memory_payload={
            "forward_bias": "LONG",
            "analog_quality": 0.7,
            "stability_score": 0.7,
        },
    )

    alignment = compute_alignment(state)
    vetoes = evaluate_vetoes(state, alignment)

    bootstrap_risk = RiskScoreResult(
        risk_score=0.25,
        risk_state=RiskState.LOW,
        components=RiskComponentBreakdown(
            market_risk=0.0,
            execution_risk=0.0,
            model_risk=0.0,
            veto_risk=0.0,
        ),
    )

    conviction = compute_conviction(state, alignment, vetoes, bootstrap_risk)
    gate = run_decision_gate(state, alignment, vetoes, bootstrap_risk)
    final_risk = compute_risk_score(
        state=state,
        alignment=alignment,
        conviction=conviction,
        gate=gate,
        vetoes=vetoes,
    )

    print("alignment:", alignment.to_dict())
    print("vetoes:", vetoes.to_dict())
    print("conviction:", conviction.to_dict())
    print("gate:", gate.to_dict())
    print("final_risk:", final_risk.to_dict())


if __name__ == "__main__":
    main()