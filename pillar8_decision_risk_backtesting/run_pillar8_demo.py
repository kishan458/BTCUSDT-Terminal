from __future__ import annotations

from pillar8_decision_risk_backtesting.pillar8_engine import run_pillar8_engine
from pillar8_decision_risk_backtesting.output.terminal_printer import print_pillar8_output

from pillar8_decision_risk_backtesting.backtesting.backtest_runner import (
    BacktestTradeInput,
    run_backtest,
)
from pillar8_decision_risk_backtesting.backtesting.regime_segmentation import (
    build_regime_tagged_trade,
    segment_backtest_results,
)
from pillar8_decision_risk_backtesting.backtesting.validation_report import (
    build_validation_report,
)

from pillar8_decision_risk_backtesting.stress.drawdown_stress_engine import (
    compute_drawdown_stress,
)
from pillar8_decision_risk_backtesting.stress.cost_shock_engine import (
    run_cost_shock_stress,
)
from pillar8_decision_risk_backtesting.stress.regime_stress_engine import (
    run_regime_stress,
)
from pillar8_decision_risk_backtesting.stress.ruin_probability_engine import (
    run_ruin_probability_simulation,
)

from pillar8_decision_risk_backtesting.state.decision_schema import Direction


def main() -> None:
    # ---------------- DEMO BACKTEST TRADES ----------------
    backtest_trades = [
        BacktestTradeInput(direction=Direction.LONG, entry_price=100, exit_price=105, position_size=10),
        BacktestTradeInput(direction=Direction.SHORT, entry_price=120, exit_price=110, position_size=5),
        BacktestTradeInput(direction=Direction.LONG, entry_price=200, exit_price=190, position_size=2),
        BacktestTradeInput(direction=Direction.LONG, entry_price=300, exit_price=318, position_size=1.5),
        BacktestTradeInput(direction=Direction.SHORT, entry_price=150, exit_price=145, position_size=4),
    ]

    backtest = run_backtest(backtest_trades)
    net_pnls = [t.net_pnl for t in backtest.trades]
    gross_pnls = [t.gross_pnl for t in backtest.trades]
    total_costs = [t.fee_cost + t.slippage_cost for t in backtest.trades]

    # ---------------- REGIME TAGGING ----------------
    tagged_trades = [
        build_regime_tagged_trade(
            net_pnl=backtest.trades[0].net_pnl,
            regime_state="TREND",
            volatility_bucket="HIGH_VOL",
            event_state="NON_EVENT",
            strategy_compatibility=0.85,
            tag="BREAKOUT",
        ),
        build_regime_tagged_trade(
            net_pnl=backtest.trades[1].net_pnl,
            regime_state="TREND",
            volatility_bucket="HIGH_VOL",
            event_state="NON_EVENT",
            strategy_compatibility=0.80,
            tag="BREAKOUT",
        ),
        build_regime_tagged_trade(
            net_pnl=backtest.trades[2].net_pnl,
            regime_state="RANGE",
            volatility_bucket="LOW_VOL",
            event_state="EVENT",
            strategy_compatibility=0.30,
            tag="MEAN_REVERSION",
        ),
        build_regime_tagged_trade(
            net_pnl=backtest.trades[3].net_pnl,
            regime_state="TREND",
            volatility_bucket="HIGH_VOL",
            event_state="NON_EVENT",
            strategy_compatibility=0.90,
            tag="BREAKOUT",
        ),
        build_regime_tagged_trade(
            net_pnl=backtest.trades[4].net_pnl,
            regime_state="RANGE",
            volatility_bucket="LOW_VOL",
            event_state="EVENT",
            strategy_compatibility=0.35,
            tag="MEAN_REVERSION",
        ),
    ]

    segmentation = segment_backtest_results(tagged_trades)
    validation = build_validation_report(
        metrics=backtest.metrics,
        segmentation=segmentation,
    )

    # ---------------- STRESS ----------------
    drawdown_stress = compute_drawdown_stress(
        net_pnls,
        breach_threshold=20.0,
    )

    cost_shock = run_cost_shock_stress(
        gross_pnls=gross_pnls,
        original_total_costs=total_costs,
        cost_multiplier=2.0,
    )

    regime_stress = run_regime_stress(segmentation)

    ruin = run_ruin_probability_simulation(
        net_pnls,
        initial_equity=1000.0,
        ruin_threshold=900.0,
        simulations=500,
        seed=42,
    )

    # ---------------- ENGINE ----------------
    output = run_pillar8_engine(
        timestamp_utc="2026-04-03T00:00:00Z",
        sentiment_payload={
            "sentiment_state": "BULLISH",
            "confidence": 0.7,
        },
        memory_payload={
            "forward_bias": "LONG",
            "analog_quality": 0.7,
            "stability_score": 0.7,
        },
        structure_payload={
            "structure_state": "HIGHER_HIGH_HIGHER_LOW",
            "trap_risk": 0.2,
            "liquidation_risk": 0.2,
        },
        candle_payload={
            "dominant_intent": "BULLISH",
            "pressure_bias": "BULLISH",
            "breakout_quality": 0.7,
            "failure_risk": 0.2,
        },
        regime_payload={
            "regime_state": "UPTREND",
            "strategy_compatibility": 0.8,
        },
        council_payload={
            "final_bias": "LONG",
            "confidence": 0.8,
            "agreement_score": 0.8,
            "conflict_score": 0.1,
        },
        realized_volatility=0.4,
        target_volatility=0.2,
        win_probability=0.55,
        payoff_ratio=2.0,
        thesis_summary="Integrated Pillar 8 full production demo.",
        validation_report=validation,
        drawdown_stress=drawdown_stress,
        cost_shock=cost_shock,
        regime_stress=regime_stress,
        ruin=ruin,
    )

    print_pillar8_output(output)


if __name__ == "__main__":
    main()