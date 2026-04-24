from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

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
            f"""
            SELECT timestamp, open, high, low, close, volume
            FROM btc_price_1h
            ORDER BY timestamp DESC
            LIMIT {limit}
            """,
            conn,
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)


# ── Pillar runners ────────────────────────────────────────────────────────────

def _run_pillar1():
    return run_pillar1()


def _run_pillar2():
    db = DB_PATH
    return run_pillar2_output(
        feature_config=MemoryFeatureConfig(db_path=db, table_name="btc_price_1h", timestamp_col="timestamp"),
        outcome_config=MemoryOutcomeConfig(horizons=(1, 3, 6, 12)),
        signature_config=StateSignatureConfig(require_complete_core_state=True),
        retrieval_config=AnalogRetrievalConfig(min_similarity_score=0.45, max_weighted_matches=300),
        conditional_config=ConditionalOutcomeConfig(use_weighted_matches=True),
        stability_config=StabilityConfig(minimum_window_samples=10),
        session_config=SessionMemoryConfig(min_samples_per_group=25),
        calendar_config=CalendarMemoryConfig(min_samples_per_group=25),
        output_config=Pillar2OutputConfig(asset="BTCUSDT", round_decimals=4),
    )


def _run_pillar3(df: pd.DataFrame):
    return run_pillar3_output(df)


def _run_pillar4(df: pd.DataFrame):
    config = Pillar4Config(
        candle_features=CandleFeatureConfig(
            atr_window=14, range_mean_window=20, body_mean_window=20,
            zscore_window=20, overlap_window_short=3, overlap_window_long=5,
            progress_window_short=3, progress_window_medium=5, progress_window_long=8,
            persistence_window_short=3, persistence_window_medium=5, persistence_window_long=8,
            realized_vol_window=20, volatility_percentile_window=50,
            range_percentile_window=50, body_percentile_window=50,
            contraction_window=5, inside_outside_window=5, entropy_window=5, rolling_wick_window=5,
        ),
        candle_intent=CandleIntentConfig(),
        multi_candle_context=MultiCandleContextConfig(),
        absorption=AbsorptionConfig(),
        breakout_quality=BreakoutQualityConfig(
            range_window=5, acceptance_close_threshold=0.55,
            strong_breakout_threshold=0.65, weak_breakout_threshold=0.40,
            fake_breakout_overlap_threshold=0.65, minimum_breach_threshold=0.0,
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


# ── Color helpers ─────────────────────────────────────────────────────────────

def _bias_color(bias: str) -> str:
    if bias == "LONG":
        return "green"
    if bias == "NO_TRADE":
        return "red"
    if bias == "WATCH":
        return "yellow"
    return "white"


def _alignment_color(alignment: str) -> str:
    if "BULLISH" in alignment:
        return "green"
    if "BEARISH" in alignment:
        return "red"
    if "CONFLICT" in alignment:
        return "red"
    if "INACTIVE" in alignment:
        return "yellow"
    return "white"


def _score_color(score: float | None) -> str:
    if score is None:
        return "white"
    if score >= 0.65:
        return "green"
    if score >= 0.45:
        return "yellow"
    return "red"


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pillar7_terminal():
    console = Console()

    console.print()
    console.print(Panel.fit(
        "[bold white]BTC/USDT TERMINAL[/bold white] | [cyan]LIVE PILLAR 7 — ML COUNCIL ENGINE[/cyan]",
        border_style="bright_blue"
    ))

    df = _load_ohlcv()

    with console.status("[bold cyan]Running Pillar 1 — Sentiment...", spinner="dots"):
        p1 = _run_pillar1()

    with console.status("[bold cyan]Running Pillar 2 — Market Memory...", spinner="dots"):
        p2 = _run_pillar2()

    with console.status("[bold cyan]Running Pillar 3 — Structure & Liquidity...", spinner="dots"):
        p3 = _run_pillar3(df)

    with console.status("[bold cyan]Running Pillar 4 — Candle Intelligence...", spinner="dots"):
        p4 = _run_pillar4(df)

    with console.status("[bold cyan]Running Pillar 5 — Regime & Cycle...", spinner="dots"):
        p5 = _run_pillar5()

    with console.status("[bold cyan]Running Pillar 6 — High Impact Events...", spinner="dots"):
        p6 = _run_pillar6()

    with console.status("[bold cyan]Building shared state & running ML council...", spinner="dots"):
        shared_state = build_shared_state(
            asset="BTCUSDT",
            pillar1_output=p1,
            pillar2_output=p2,
            pillar3_output=p3,
            pillar4_output=p4,
            pillar5_output=p5,
            pillar6_output=p6,
        )

        result = build_pillar7_output(
            shared_state=shared_state,
            professor_artifact_path=PROFESSOR_ARTIFACT,
            retail_artifact_path=RETAIL_ARTIFACT,
            threshold=0.5,
        )

    council      = result.get("council", {})
    agents       = result.get("agent_outputs", {})
    disagreement = result.get("disagreement", {})
    trade_ctx    = result.get("trade_context", {})
    shared_sum   = result.get("shared_state_summary", {})
    explanation  = result.get("explanation", {})
    ai_overview  = result.get("ai_overview", {})
    reason_stack = result.get("reason_stack", [])
    flags        = trade_ctx.get("risk_flags", [])

    professor = agents.get("professor_agent", {})
    retail    = agents.get("retail_agent", {})

    council_bias       = council.get("council_bias", "UNKNOWN")
    tradeability_score = council.get("tradeability_score", 0.0)
    dominant_agent     = council.get("dominant_agent", "UNKNOWN")
    alignment_class    = disagreement.get("alignment_class", "UNKNOWN")

    # ── Council Summary ───────────────────────────────────────────────────────
    bc = _bias_color(council_bias)
    ac = _alignment_color(alignment_class)

    council_table = Table(show_header=True, header_style="bold magenta",
                          expand=True, box=box.SIMPLE_HEAVY)
    council_table.add_column("Metric", style="dim", width=28)
    council_table.add_column("Value")

    council_table.add_row("Council Bias",
                           f"[{bc} bold]{council_bias}[/{bc} bold]")
    council_table.add_row("Tradeability Score",
                           f"[{_score_color(tradeability_score)} bold]{tradeability_score:.3f}[/{_score_color(tradeability_score)} bold]")
    council_table.add_row("Alignment",
                           f"[{ac} bold]{alignment_class}[/{ac} bold]")
    council_table.add_row("Dominant Agent",     dominant_agent)
    council_table.add_row("Agreement Score",    f"{disagreement.get('agreement_score', 0):.2f}")
    council_table.add_row("Conflict Score",     f"{disagreement.get('conflict_score', 0):.2f}")
    council_table.add_row("Dominance Gap",      f"{disagreement.get('dominance_gap', 0):.3f}")
    council_table.add_row("Timestamp (UTC)",    f"[dim]{result.get('timestamp_utc', 'N/A')}[/dim]")

    console.print(council_table)

    # ── Shared State Summary ──────────────────────────────────────────────────
    state_table = Table(title="Shared State Summary", show_header=True,
                        header_style="bold cyan", box=box.SIMPLE_HEAVY, expand=True)
    state_table.add_column("Pillar")
    state_table.add_column("State", justify="right")

    for label, key in [
        ("Sentiment (P1)",     "sentiment_state"),
        ("Memory (P2)",        "memory_state"),
        ("Structure (P3)",     "structure_state"),
        ("Candle (P4)",        "candle_state"),
        ("Regime (P5)",        "regime_state"),
        ("Event (P6)",         "event_state"),
    ]:
        state_table.add_row(label, str(shared_sum.get(key, "N/A")))

    console.print(state_table)

    # ── Agent Outputs ─────────────────────────────────────────────────────────
    agent_table = Table(title="Agent Outputs", show_header=True,
                        header_style="bold cyan", box=box.SIMPLE_HEAVY)
    agent_table.add_column("Agent")
    agent_table.add_column("Label", justify="right")
    agent_table.add_column("Raw Prob", justify="right")
    agent_table.add_column("Cal Prob", justify="right")

    def _agent_color(label: str) -> str:
        if label in ("LONG", "CHASE_LONG"):
            return "green"
        if label in ("SHORT", "CHASE_SHORT"):
            return "red"
        return "yellow"

    prof_label = professor.get("predicted_label", "N/A")
    ret_label  = retail.get("predicted_label", "N/A")

    agent_table.add_row(
        "Professor Agent",
        f"[{_agent_color(prof_label)} bold]{prof_label}[/{_agent_color(prof_label)} bold]",
        f"{professor.get('raw_probability', 0):.3f}",
        f"{professor.get('calibrated_probability', 0):.3f}",
    )
    agent_table.add_row(
        "Retail Agent",
        f"[{_agent_color(ret_label)} bold]{ret_label}[/{_agent_color(ret_label)} bold]",
        f"{retail.get('raw_probability', 0):.3f}",
        f"{retail.get('calibrated_probability', 0):.3f}",
    )

    console.print(agent_table)

    # ── Trade Context ─────────────────────────────────────────────────────────
    trade_table = Table(title="Trade Context", show_header=True,
                        header_style="bold cyan", box=box.SIMPLE_HEAVY)
    trade_table.add_column("Metric")
    trade_table.add_column("Value", justify="right")

    allow = trade_ctx.get("allow_trade")
    trade_table.add_row(
        "Allow Trade",
        f"[{'green' if allow else 'red'} bold]{'YES' if allow else 'NO'}[/{'green' if allow else 'red'} bold]"
    )
    trade_table.add_row("Event Uncertainty",  f"{trade_ctx.get('event_base_uncertainty', 0):.1%}")
    trade_table.add_row("Risk Flag Count",    str(trade_ctx.get("risk_flag_count", 0)))

    console.print(trade_table)

    # ── Reason Stack ──────────────────────────────────────────────────────────
    if reason_stack:
        console.print("\n[bold underline]Reason Stack:[/bold underline]")
        for r in reason_stack:
            console.print(f"  [dim]→[/dim] {r}")

    # ── Risk Flags ────────────────────────────────────────────────────────────
    if flags:
        console.print("\n[bold underline]Risk Flags:[/bold underline]")
        for flag in flags:
            console.print(f"  [red bold]⚠[/red bold]  {flag}")
    else:
        console.print("\n[green]✓ No active risk flags[/green]")

    # ── AI Council Overview ───────────────────────────────────────────────────
    ai_source = ai_overview.get("source", "fallback")
    console.print(f"\n[bold underline]AI Council Overview[/bold underline] [dim]({ai_source})[/dim]")
    for line in ai_overview.get("overview", "").split("\n"):
        if line.strip():
            console.print(f"  [white]{line.strip()}[/white]")

    console.print("\n" + "—" * 60 + "\n")


if __name__ == "__main__":
    run_pillar7_terminal()