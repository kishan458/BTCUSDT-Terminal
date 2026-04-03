from pillar8_decision_risk_backtesting.output.pillar8_output import build_pillar8_output
from pillar8_decision_risk_backtesting.output.terminal_printer import print_pillar8_output

from pillar8_decision_risk_backtesting.state.decision_state_builder import build_decision_state
from pillar8_decision_risk_backtesting.decision.alignment_engine import compute_alignment
from pillar8_decision_risk_backtesting.decision.veto_engine import evaluate_vetoes
from pillar8_decision_risk_backtesting.decision.conviction_engine import compute_conviction
from pillar8_decision_risk_backtesting.decision.decision_gate_engine import run_decision_gate
from pillar8_decision_risk_backtesting.risk.risk_score_engine import (
    compute_risk_score,
    RiskScoreResult,
    RiskComponentBreakdown,
)

from pillar8_decision_risk_backtesting.sizing.volatility_target_engine import compute_volatility_target_size
from pillar8_decision_risk_backtesting.sizing.fractional_kelly_engine import compute_fractional_kelly_size
from pillar8_decision_risk_backtesting.sizing.size_allocator import allocate_position_size
from pillar8_decision_risk_backtesting.sizing.leverage_cap_engine import compute_leverage_cap

from pillar8_decision_risk_backtesting.execution.invalidation_engine import build_invalidators
from pillar8_decision_risk_backtesting.execution.holding_horizon_engine import infer_holding_horizon
from pillar8_decision_risk_backtesting.execution.trade_constructor import construct_trade_plan

# BACKTEST + VALIDATION
from pillar8_decision_risk_backtesting.backtesting.backtest_runner import BacktestTradeInput, run_backtest
from pillar8_decision_risk_backtesting.backtesting.regime_segmentation import build_regime_tagged_trade, segment_backtest_results
from pillar8_decision_risk_backtesting.backtesting.validation_report import build_validation_report

# STRESS
from pillar8_decision_risk_backtesting.stress.drawdown_stress_engine import compute_drawdown_stress
from pillar8_decision_risk_backtesting.stress.cost_shock_engine import run_cost_shock_stress
from pillar8_decision_risk_backtesting.stress.regime_stress_engine import run_regime_stress
from pillar8_decision_risk_backtesting.stress.ruin_probability_engine import run_ruin_probability_simulation

from pillar8_decision_risk_backtesting.state.decision_schema import Direction, RiskState


def main():
    # ---------------- STATE ----------------
    s = build_decision_state(
        council_payload={"final_bias": "LONG", "confidence": 0.8, "agreement_score": 0.8, "conflict_score": 0.1},
        regime_payload={"regime_state": "UPTREND", "strategy_compatibility": 0.8},
        candle_payload={"dominant_intent": "BULLISH", "pressure_bias": "BULLISH", "breakout_quality": 0.7, "failure_risk": 0.2},
        structure_payload={"structure_state": "HIGHER_HIGH_HIGHER_LOW", "trap_risk": 0.2, "liquidation_risk": 0.2},
        sentiment_payload={"sentiment_state": "BULLISH", "confidence": 0.7},
        memory_payload={"forward_bias": "LONG", "analog_quality": 0.7, "stability_score": 0.7},
    )

    a = compute_alignment(s)
    v = evaluate_vetoes(s, a)

    # ---------------- FIXED: BOOTSTRAP RISK ----------------
    bootstrap_risk = RiskScoreResult(
        risk_score=0.25,
        risk_state=RiskState.LOW,
        components=RiskComponentBreakdown(
            market_risk=0,
            execution_risk=0,
            model_risk=0,
            veto_risk=0,
        ),
    )

    c = compute_conviction(s, a, v, bootstrap_risk)
    g = run_decision_gate(s, a, v, bootstrap_risk)
    r = compute_risk_score(state=s, alignment=a, conviction=c, gate=g, vetoes=v)

    # ---------------- SIZING ----------------
    vol = compute_volatility_target_size(realized_volatility=0.4, target_volatility=0.2)
    kelly = compute_fractional_kelly_size(win_probability=0.55, payoff_ratio=2.0)
    size = allocate_position_size(volatility_target=vol, fractional_kelly=kelly, conviction=c, risk=r, gate=g)
    lev = compute_leverage_cap(state=s, risk_score=r.risk_score)

    # ---------------- EXECUTION ----------------
    invalidators = build_invalidators(state=s, alignment=a, risk=r, gate=g)
    horizon = infer_holding_horizon(state=s, risk=r, gate=g)

    trade_plan = construct_trade_plan(
        gate=g,
        risk=r,
        size=size,
        leverage=lev,
        invalidators=invalidators,
        horizon=horizon,
    )

    # ---------------- BACKTEST ----------------
    trades = [
        BacktestTradeInput(direction=Direction.LONG, entry_price=100, exit_price=105, position_size=10),
        BacktestTradeInput(direction=Direction.SHORT, entry_price=120, exit_price=110, position_size=5),
        BacktestTradeInput(direction=Direction.LONG, entry_price=200, exit_price=190, position_size=2),
    ]

    backtest = run_backtest(trades)
    net_pnls = [t.net_pnl for t in backtest.trades]
    gross_pnls = [t.gross_pnl for t in backtest.trades]
    costs = [t.fee_cost + t.slippage_cost for t in backtest.trades]

    # ---------------- REGIME SEGMENTATION ----------------
    tagged = [
        build_regime_tagged_trade(net_pnl=x, regime_state="TREND")
        for x in net_pnls
    ]
    seg = segment_backtest_results(tagged)

    validation = build_validation_report(metrics=backtest.metrics, segmentation=seg)

    # ---------------- STRESS ----------------
    drawdown = compute_drawdown_stress(net_pnls, breach_threshold=20)
    cost_shock = run_cost_shock_stress(gross_pnls=gross_pnls, original_total_costs=costs, cost_multiplier=2.0)
    regime_stress = run_regime_stress(seg)
    ruin = run_ruin_probability_simulation(net_pnls, initial_equity=1000, ruin_threshold=900, simulations=500)

    # ---------------- OUTPUT ----------------
    out = build_pillar8_output(
        timestamp_utc="2026-04-03T00:00:00Z",
        gate=g,
        risk=r,
        alignment=a,
        trade_plan=trade_plan,
        size=size,
        leverage=lev,
        vetoes=v.vetoes,
        warnings=v.warnings,
        thesis_summary="Bullish alignment with controlled risk.",
        validation_report=validation,
        drawdown_stress=drawdown,
        cost_shock=cost_shock,
        regime_stress=regime_stress,
        ruin=ruin,
    )

    print_pillar8_output(out)


if __name__ == "__main__":
    main()