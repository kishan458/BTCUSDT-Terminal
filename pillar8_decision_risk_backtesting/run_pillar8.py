from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.db import resolve_db_path

# ── Pillar imports ────────────────────────────────────────────────────────────
from pillar1_sentiment.run_pillar1 import run_pillar1

from pillar2_market_memory_engine.memory_feature_engine import MemoryFeatureConfig
from pillar2_market_memory_engine.memory_outcome_engine import MemoryOutcomeConfig
from pillar2_market_memory_engine.state_signature_engine import StateSignatureConfig
from pillar2_market_memory_engine.analog_retrieval_engine import AnalogRetrievalConfig
from pillar2_market_memory_engine.conditional_outcome_engine import ConditionalOutcomeConfig
from pillar2_market_memory_engine.stability_engine import StabilityConfig
from pillar2_market_memory_engine.session_memory_engine import SessionMemoryConfig
from pillar2_market_memory_engine.calendar_memory_engine import CalendarMemoryConfig
from pillar2_market_memory_engine.pillar2_output import Pillar2OutputConfig, run_pillar2_output

from pillar3_structure_liquidity_engine.pillar3_output import run_pillar3_output

from pillar4_candle_engine.absorption_engine import AbsorptionConfig
from pillar4_candle_engine.breakout_quality_engine import BreakoutQualityConfig
from pillar4_candle_engine.candle_features_engine import CandleFeatureConfig, OhlcColumns
from pillar4_candle_engine.candle_intent_engine import CandleIntentConfig
from pillar4_candle_engine.multi_candle_context_engine import MultiCandleContextConfig
from pillar4_candle_engine.pillar4_output import Pillar4Config, run_pillar4_candle_intelligence
from pillar4_candle_engine.pressure_engine import PressureConfig

from pillar5_regime_cycle_engine.pillar5_output import build_pillar5_output
from pillar6_high_impact_events.pillar6_output import build_pillar6_output

from pillar7_ml_council.shared_state_builder import build_shared_state
from pillar7_ml_council.pillar7_output import build_pillar7_output

from pillar8_decision_risk_backtesting.pillar8_engine import run_pillar8_engine

DB_PATH = str(resolve_db_path())

PROFESSOR_ARTIFACT = str(
    REPO_ROOT / "pillar7_ml_council" / "artifacts" /
    "professor_policy_rf_long_vs_no_trade_v1_test.pkl"
)
RETAIL_ARTIFACT = str(
    REPO_ROOT / "pillar7_ml_council" / "artifacts" /
    "retail_policy_rf_long_vs_no_action_v1_test.pkl"
)


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_ohlcv(limit: int = 300) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            f"SELECT timestamp, open, high, low, close, volume FROM btc_price_1h ORDER BY timestamp DESC LIMIT {limit}",
            conn,
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)


# ── Pillar runners ────────────────────────────────────────────────────────────

def _run_pillar1():
    return run_pillar1()

def _run_pillar2():
    return run_pillar2_output(
        feature_config=MemoryFeatureConfig(db_path=DB_PATH, table_name="btc_price_1h", timestamp_col="timestamp"),
        outcome_config=MemoryOutcomeConfig(horizons=(1, 3, 6, 12)),
        signature_config=StateSignatureConfig(require_complete_core_state=True),
        retrieval_config=AnalogRetrievalConfig(min_similarity_score=0.45, max_weighted_matches=300),
        conditional_config=ConditionalOutcomeConfig(use_weighted_matches=True),
        stability_config=StabilityConfig(minimum_window_samples=10),
        session_config=SessionMemoryConfig(min_samples_per_group=25),
        calendar_config=CalendarMemoryConfig(min_samples_per_group=25),
        output_config=Pillar2OutputConfig(asset="BTCUSDT", round_decimals=4),
    )

def _run_pillar3(df):
    return run_pillar3_output(df)

def _run_pillar4(df):
    config = Pillar4Config(
        candle_features=CandleFeatureConfig(
            atr_window=14, range_mean_window=20, body_mean_window=20, zscore_window=20,
            overlap_window_short=3, overlap_window_long=5, progress_window_short=3,
            progress_window_medium=5, progress_window_long=8, persistence_window_short=3,
            persistence_window_medium=5, persistence_window_long=8, realized_vol_window=20,
            volatility_percentile_window=50, range_percentile_window=50, body_percentile_window=50,
            contraction_window=5, inside_outside_window=5, entropy_window=5, rolling_wick_window=5,
        ),
        candle_intent=CandleIntentConfig(), multi_candle_context=MultiCandleContextConfig(),
        absorption=AbsorptionConfig(),
        breakout_quality=BreakoutQualityConfig(
            range_window=5, acceptance_close_threshold=0.55, strong_breakout_threshold=0.65,
            weak_breakout_threshold=0.40, fake_breakout_overlap_threshold=0.65, minimum_breach_threshold=0.0,
        ),
        pressure=PressureConfig(),
    )
    return run_pillar4_candle_intelligence(
        df=df, pillar4_config=config,
        columns=OhlcColumns(open="open", high="high", low="low", close="close", volume="volume", timestamp="timestamp"),
        asset="BTCUSDT", timeframe="1h", lookback_bars_used=min(30, len(df)), atr_method="wilder",
    )

def _run_pillar5():
    return build_pillar5_output()

def _run_pillar6():
    return build_pillar6_output()


# ── Payload adapters for Pillar 8 ─────────────────────────────────────────────

def _adapt_p1(p1: Dict[str, Any]) -> Dict[str, Any]:
    agg = p1.get("aggregate_sentiment", {})
    return {
        "sentiment_state": agg.get("label", "NEUTRAL"),
        "confidence":      agg.get("confidence", 0.0),
        "drivers":         p1.get("drivers", []),
        "institutional_summary": " ".join(p1.get("institutional_summary", [])),
    }


def _adapt_p2(p2: Dict[str, Any]) -> Dict[str, Any]:
    ms  = p2.get("memory_summary", {})
    fwd = p2.get("forward_outcomes", {})
    stab = p2.get("stability_diagnostics", {})
    ana = p2.get("historical_analogs", {})
    return {
        "memory_state":    ms.get("memory_bias", "NEUTRAL"),
        "analog_quality":  ana.get("analog_quality_score", 0.0),
        "forward_bias":    ms.get("memory_bias", "NEUTRAL"),
        "stability_score": stab.get("temporal_stability_score", 0.0),
        "context_notes":   [],
    }


def _adapt_p3(p3: Dict[str, Any]) -> Dict[str, Any]:
    struct  = p3.get("structure_state", {})
    liq     = p3.get("liquidation_risk", {})
    trap_risk_label = p3.get("structure_liquidity_summary", {}).get("trap_risk", "LOW")
    trap_score = {"LOW": 0.1, "MODERATE": 0.5, "HIGH": 0.8}.get(trap_risk_label, 0.1)
    liq_score  = {"LOW": 0.1, "MODERATE": 0.5, "HIGH": 0.8}.get(
        liq.get("long_liquidation_risk", "LOW"), 0.1
    )
    return {
        "structure_state":   struct.get("market_structure", "UNKNOWN"),
        "liquidity_levels":  p3.get("liquidity_targets", []),
        "trap_risk":         trap_score,
        "liquidation_risk":  liq_score,
        "risk_flags":        p3.get("risk_flags", []),
    }


def _adapt_p4(p4: Dict[str, Any]) -> Dict[str, Any]:
    cs  = p4.get("candle_summary", {})
    brk = p4.get("breakout_analysis", {})
    prs = p4.get("pressure", {})
    return {
        "dominant_intent":   cs.get("dominant_intent", "NEUTRAL"),
        "momentum_state":    cs.get("momentum_state", "NEUTRAL"),
        "breakout_quality":  brk.get("breakout_quality_score", 0.0),
        "pressure_bias":     prs.get("pressure_bias", "NEUTRAL"),
        "absorption_signals": [],
        "failure_risk":      brk.get("failure_score", 0.0),
    }


def _adapt_p5(p5: Dict[str, Any]) -> Dict[str, Any]:
    rs   = p5.get("regime_summary", {})
    strat = p5.get("strategy_compatibility", {})
    tf_compat = strat.get("trend_following", "NEUTRAL")
    compat_score = {"FAVORED": 0.9, "MODERATELY_FAVORED": 0.6,
                    "NEUTRAL": 0.5, "NOT_FAVORED": 0.2}.get(tf_compat, 0.5)
    return {
        "regime_state":           rs.get("directional_regime", "UNKNOWN"),
        "cycle_phase":            rs.get("cycle_phase", "UNKNOWN"),
        "strategy_compatibility": compat_score,
    }


def _adapt_p6(p6: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event_state":      p6.get("state", "IDLE"),
        "base_uncertainty": p6.get("base_uncertainty", 0.0),
        "trade_restrictions": p6.get("trade_restrictions", {}),
        "scenarios":        p6.get("scenarios", []),
        "ai_reasoning":     p6.get("ai_reasoning", ""),
    }


def _adapt_p7(p7: Dict[str, Any]) -> Dict[str, Any]:
    council     = p7.get("council", {})
    disagreement = p7.get("disagreement", {})
    ai_overview = p7.get("ai_overview", {})
    return {
        "final_bias":     council.get("council_bias", "NEUTRAL"),
        "final_decision": council.get("council_bias", "NO_TRADE"),
        "confidence":     council.get("tradeability_score", 0.0),
        "agreement_score": disagreement.get("agreement_score", 0.0),
        "conflict_score":  disagreement.get("conflict_score", 0.0),
        "dominant_agent":  council.get("dominant_agent", "NONE"),
        "reasoning":       ai_overview.get("overview", "") if isinstance(ai_overview, dict) else "",
    }


# ── Color helpers ─────────────────────────────────────────────────────────────

def _action_color(action: str) -> str:
    if action in ("LONG", "PROBE_LONG"):
        return "green"
    if action in ("SHORT", "PROBE_SHORT"):
        return "red"
    if action == "WATCHLIST":
        return "yellow"
    return "red"

def _risk_color(state: str) -> str:
    return {"LOW": "green", "MODERATE": "yellow", "HIGH": "red", "EXTREME": "bright_red"}.get(state, "white")

def _score_color(score: float) -> str:
    if score >= 0.65: return "green"
    if score >= 0.40: return "yellow"
    return "red"


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pillar8_terminal():
    console = Console()

    console.print()
    console.print(Panel.fit(
        "[bold white]BTC/USDT TERMINAL[/bold white] | [cyan]LIVE PILLAR 8 — DECISION, RISK & SIZING ENGINE[/cyan]",
        border_style="bright_blue"
    ))

    df = _load_ohlcv()

    with console.status("[bold cyan]Running Pillar 1...", spinner="dots"): p1 = _run_pillar1()
    with console.status("[bold cyan]Running Pillar 2...", spinner="dots"): p2 = _run_pillar2()
    with console.status("[bold cyan]Running Pillar 3...", spinner="dots"): p3 = _run_pillar3(df)
    with console.status("[bold cyan]Running Pillar 4...", spinner="dots"): p4 = _run_pillar4(df)
    with console.status("[bold cyan]Running Pillar 5...", spinner="dots"): p5 = _run_pillar5()
    with console.status("[bold cyan]Running Pillar 6...", spinner="dots"): p6 = _run_pillar6()

    with console.status("[bold cyan]Running Pillar 7 — ML Council...", spinner="dots"):
        shared_state = build_shared_state(
            asset="BTCUSDT",
            pillar1_output=p1, pillar2_output=p2, pillar3_output=p3,
            pillar4_output=p4, pillar5_output=p5, pillar6_output=p6,
        )
        p7 = build_pillar7_output(
            shared_state=shared_state,
            professor_artifact_path=PROFESSOR_ARTIFACT,
            retail_artifact_path=RETAIL_ARTIFACT,
            threshold=0.5,
        )

    with console.status("[bold cyan]Running Pillar 8 — Decision Engine...", spinner="dots"):
        ts = datetime.now(timezone.utc).isoformat()

        # Grab realized vol from Pillar 4
        realized_vol = None
        try:
            realized_vol = p4.get("volatility_context", {}).get("realized_volatility")
        except Exception:
            pass

        result = run_pillar8_engine(
            timestamp_utc=ts,
            sentiment_payload=_adapt_p1(p1),
            memory_payload=_adapt_p2(p2),
            structure_payload=_adapt_p3(p3),
            candle_payload=_adapt_p4(p4),
            regime_payload=_adapt_p5(p5),
            events_payload=_adapt_p6(p6),
            council_payload=_adapt_p7(p7),
            realized_volatility=realized_vol,
            target_volatility=0.20,
            win_probability=0.52,
            payoff_ratio=1.5,
            kelly_fraction=0.25,
            max_size_fraction=1.0,
            max_leverage=3.0,
            min_leverage=1.0,
            thesis_summary="Multi-pillar BTC/USDT terminal decision output.",
        )

    data = result.to_dict()

    final_action = data["final_action"]
    direction    = data["direction"]
    risk_state   = data["risk_state"]
    archetype    = data["decision_archetype"]
    ac = _action_color(final_action)
    rc = _risk_color(risk_state)

    # ── Decision Summary ──────────────────────────────────────────────────────
    dec_table = Table(show_header=True, header_style="bold magenta",
                      expand=True, box=box.SIMPLE_HEAVY)
    dec_table.add_column("Metric", style="dim", width=28)
    dec_table.add_column("Value")

    dec_table.add_row("Final Action",
                       f"[{ac} bold]{final_action}[/{ac} bold]")
    dec_table.add_row("Direction",
                       f"[{ac} bold]{direction}[/{ac} bold]")
    dec_table.add_row("Decision Archetype",   archetype)
    dec_table.add_row("Decision Confidence",
                       f"[{_score_color(data['decision_confidence'])} bold]{data['decision_confidence']:.3f}[/{_score_color(data['decision_confidence'])} bold]")
    dec_table.add_row("Tradability Score",
                       f"[{_score_color(data['tradability_score'])} bold]{data['tradability_score']:.3f}[/{_score_color(data['tradability_score'])} bold]")
    dec_table.add_row("Risk Score",
                       f"[{rc} bold]{data['risk_score']:.3f} ({risk_state})[/{rc} bold]")
    dec_table.add_row("Size Fraction",        f"{data['size_fraction']:.3f}")
    dec_table.add_row("Max Leverage",         f"{data['max_leverage_allowed']:.2f}x")
    dec_table.add_row("Timestamp (UTC)",      f"[dim]{data['timestamp_utc']}[/dim]")

    console.print(dec_table)

    # ── Alignment ─────────────────────────────────────────────────────────────
    alignment = data["alignment"]
    ali_table = Table(title="Cross-Pillar Alignment", show_header=True,
                      header_style="bold cyan", box=box.SIMPLE_HEAVY, expand=True)
    ali_table.add_column("Metric")
    ali_table.add_column("Value", justify="right")

    ali_table.add_row("Long Score",           f"[green]{alignment['long_score']:.3f}[/green]")
    ali_table.add_row("Short Score",          f"[red]{alignment['short_score']:.3f}[/red]")
    ali_table.add_row("Net Directional Edge", f"{alignment['net_directional_edge']:.3f}")
    ali_table.add_row("Directional Conflict", f"{alignment['directional_conflict']:.3f}")
    ali_table.add_row("Dominant Direction",   alignment["dominant_direction"])

    console.print(ali_table)

    # ── Per-Pillar Alignment Components ──────────────────────────────────────
    console.print("\n[bold underline]Alignment Components (per pillar):[/bold underline]")
    for comp in alignment.get("components", []):
        score = comp["weighted_score"]
        cc = "green" if score > 0.05 else "red" if score < -0.05 else "yellow"
        console.print(
            f"  [{cc}]{comp['pillar']:<12}[/{cc}] "
            f"bias={comp['raw_bias_score']:+.2f}  "
            f"weight={comp['confidence_weight']:.2f}  "
            f"weighted={score:+.3f}"
        )

    # ── Execution Plan ────────────────────────────────────────────────────────
    exec_plan = data["execution_plan"]
    exec_table = Table(title="Execution Plan", show_header=True,
                       header_style="bold cyan", box=box.SIMPLE_HEAVY)
    exec_table.add_column("Field")
    exec_table.add_column("Value", justify="right")

    exec_table.add_row("Entry Style",     exec_plan.get("entry_style", "N/A"))
    exec_table.add_row("Stop Framework",  exec_plan.get("stop_framework", "N/A"))
    exec_table.add_row("Target Framework", exec_plan.get("target_framework", "N/A"))
    exec_table.add_row("Time Stop Bars",  str(exec_plan.get("time_stop_bars", "N/A")))

    invalidators = exec_plan.get("invalidators", [])
    if invalidators:
        for inv in invalidators:
            exec_table.add_row("Invalidator", f"[dim]{inv}[/dim]")

    console.print(exec_table)

    # ── Vetoes & Warnings ─────────────────────────────────────────────────────
    vetoes   = data.get("vetoes", [])
    warnings = data.get("warnings", [])

    if vetoes:
        console.print("\n[bold underline red]Hard Vetoes:[/bold underline red]")
        for v in vetoes:
            console.print(f"  [red bold]🚫[/red bold]  {v}")

    if warnings:
        console.print("\n[bold underline yellow]Warnings:[/bold underline yellow]")
        for w in warnings:
            console.print(f"  [yellow bold]⚠[/yellow bold]  {w}")

    if not vetoes and not warnings:
        console.print("\n[green]✓ No vetoes or warnings[/green]")

    # ── Stress Summary ────────────────────────────────────────────────────────
    stress = data.get("stress", {})
    ruin   = stress.get("ruin", {})
    regime = stress.get("regime", {})

    stress_table = Table(title="Stress Summary", show_header=True,
                         header_style="bold cyan", box=box.SIMPLE_HEAVY)
    stress_table.add_column("Metric")
    stress_table.add_column("Value", justify="right")

    stress_table.add_row("Ruin Probability",    f"{ruin.get('ruin_probability', 0):.3f}")
    stress_table.add_row("P05 Terminal Equity", f"{ruin.get('p05_terminal_equity', 0):.3f}")
    stress_table.add_row("Regime Fragility",    regime.get("regime_fragility_label", "N/A"))
    stress_table.add_row("Best Regime",         regime.get("best_regime", "N/A"))
    stress_table.add_row("Worst Regime",        regime.get("worst_regime", "N/A"))

    console.print(stress_table)

    console.print("\n" + "—" * 60 + "\n")


if __name__ == "__main__":
    run_pillar8_terminal()