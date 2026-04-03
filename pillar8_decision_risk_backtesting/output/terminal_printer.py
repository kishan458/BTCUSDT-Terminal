from __future__ import annotations

from typing import List

from pillar8_decision_risk_backtesting.output.pillar8_output import Pillar8Output


def _fmt_float(value: float) -> str:
    return f"{value:.6f}"


def _fmt_list(items: List[str], empty: str = "-") -> str:
    if not items:
        return empty
    return "; ".join(items)


def print_pillar8_output(output: Pillar8Output) -> None:
    data = output.to_dict()

    print("\n" + "=" * 80)
    print("PILLAR 8 — DECISION, RISK & BACKTESTING ENGINE")
    print("=" * 80)

    print(f"Timestamp           : {data['timestamp_utc']}")
    print(f"Final Action        : {data['final_action']}")
    print(f"Direction           : {data['direction']}")
    print(f"Decision Archetype  : {data['decision_archetype']}")
    print(f"Decision Confidence : {_fmt_float(data['decision_confidence'])}")
    print(f"Tradability Score   : {_fmt_float(data['tradability_score'])}")
    print(f"Risk Score          : {_fmt_float(data['risk_score'])} ({data['risk_state']})")
    print(f"Size Fraction       : {_fmt_float(data['size_fraction'])}")
    print(f"Max Leverage        : {_fmt_float(data['max_leverage_allowed'])}")

    print("\nTHESIS")
    print("-" * 80)
    print(data["thesis_summary"] or "-")

    print("\nALIGNMENT")
    print("-" * 80)
    alignment = data["alignment"]
    print(f"Long Score          : {_fmt_float(alignment['long_score'])}")
    print(f"Short Score         : {_fmt_float(alignment['short_score'])}")
    print(f"Net Edge            : {_fmt_float(alignment['net_directional_edge'])}")
    print(f"Directional Conflict: {_fmt_float(alignment['directional_conflict'])}")
    print(f"Dominant Direction  : {alignment['dominant_direction']}")

    print("\nEXECUTION PLAN")
    print("-" * 80)
    execution = data["execution_plan"]
    print(f"Entry Style         : {execution['entry_style']}")
    print(f"Stop Framework      : {execution['stop_framework']}")
    print(f"Target Framework    : {execution['target_framework']}")
    print(f"Time Stop Bars      : {execution['time_stop_bars']}")
    print(f"Invalidators        : {_fmt_list(execution['invalidators'])}")

    print("\nRISK CONTEXT")
    print("-" * 80)
    print(f"Vetoes              : {_fmt_list(data['vetoes'])}")
    print(f"Warnings            : {_fmt_list(data['warnings'])}")

    print("\nVALIDATION")
    print("-" * 80)
    validation = data["validation"]
    print(f"Overall Quality     : {validation['overall_quality']}")
    print(f"Strengths           : {_fmt_list(validation['strengths'])}")
    print(f"Weaknesses          : {_fmt_list(validation['weaknesses'])}")
    print(f"Warnings            : {_fmt_list(validation['warnings'])}")
    print(f"Regime Notes        : {_fmt_list(validation['regime_notes'])}")

    print("\nSTRESS")
    print("-" * 80)
    stress = data["stress"]

    drawdown = stress["drawdown"]
    print("Drawdown Stress")
    print(f"  Max Drawdown      : {_fmt_float(drawdown['max_drawdown'])}")
    print(f"  Mean Drawdown     : {_fmt_float(drawdown['mean_drawdown'])}")
    print(f"  Breach Threshold  : {_fmt_float(drawdown['drawdown_breach_threshold'])}")
    print(f"  Breach Count      : {drawdown['breach_count']}")
    print(f"  Breach Rate       : {_fmt_float(drawdown['breach_rate'])}")

    cost_shock = stress["cost_shock"]
    print("\nCost Shock Stress")
    print(f"  Original TotalPnL : {_fmt_float(cost_shock['original_total_pnl'])}")
    print(f"  Shocked TotalPnL  : {_fmt_float(cost_shock['shocked_total_pnl'])}")
    print(f"  Degradation       : {_fmt_float(cost_shock['pnl_degradation'])}")
    print(f"  Degradation Ratio : {_fmt_float(cost_shock['degradation_ratio'])}")
    print(f"  Edge Survives     : {cost_shock['edge_survives']}")

    regime = stress["regime"]
    print("\nRegime Stress")
    print(f"  Best Regime       : {regime['best_regime']}")
    print(f"  Worst Regime      : {regime['worst_regime']}")
    print(f"  Regime Spread     : {_fmt_float(regime['regime_spread'])}")
    print(f"  Fragility Score   : {_fmt_float(regime['regime_fragility_score'])}")
    print(f"  Fragility Label   : {regime['regime_fragility_label']}")

    ruin = stress["ruin"]
    print("\nRuin Simulation")
    print(f"  Simulations       : {ruin['simulations']}")
    print(f"  Ruin Count        : {ruin['ruin_count']}")
    print(f"  Ruin Probability  : {_fmt_float(ruin['ruin_probability'])}")
    print(f"  Avg Terminal Eq   : {_fmt_float(ruin['avg_terminal_equity'])}")
    print(f"  P05 Terminal Eq   : {_fmt_float(ruin['p05_terminal_equity'])}")

    print("\nAUDIT TRACE")
    print("-" * 80)
    audit = data["audit_trace"]
    print(f"Policy Version      : {audit['policy_version']}")
    print(f"Backtest Version    : {audit['backtest_version']}")

    print("=" * 80 + "\n")