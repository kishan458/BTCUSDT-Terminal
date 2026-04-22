from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from pillar4_candle_engine.absorption_engine import AbsorptionConfig
from pillar4_candle_engine.breakout_quality_engine import BreakoutQualityConfig
from pillar4_candle_engine.candle_features_engine import CandleFeatureConfig, OhlcColumns
from pillar4_candle_engine.candle_intent_engine import CandleIntentConfig
from pillar4_candle_engine.multi_candle_context_engine import MultiCandleContextConfig
from pillar4_candle_engine.pillar4_output import Pillar4Config, run_pillar4_candle_intelligence
from pillar4_candle_engine.pressure_engine import PressureConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "database" / "btc_terminal.db"


# ── Config ────────────────────────────────────────────────────────────────────

def _build_config() -> Pillar4Config:
    return Pillar4Config(
        candle_features=CandleFeatureConfig(
            atr_window=14,
            range_mean_window=20,
            body_mean_window=20,
            zscore_window=20,
            overlap_window_short=3,
            overlap_window_long=5,
            progress_window_short=3,
            progress_window_medium=5,
            progress_window_long=8,
            persistence_window_short=3,
            persistence_window_medium=5,
            persistence_window_long=8,
            realized_vol_window=20,
            volatility_percentile_window=50,
            range_percentile_window=50,
            body_percentile_window=50,
            contraction_window=5,
            inside_outside_window=5,
            entropy_window=5,
            rolling_wick_window=5,
        ),
        candle_intent=CandleIntentConfig(),
        multi_candle_context=MultiCandleContextConfig(),
        absorption=AbsorptionConfig(),
        breakout_quality=BreakoutQualityConfig(
            range_window=5,
            acceptance_close_threshold=0.55,
            strong_breakout_threshold=0.65,
            weak_breakout_threshold=0.40,
            fake_breakout_overlap_threshold=0.65,
            minimum_breach_threshold=0.0,
        ),
        pressure=PressureConfig(),
    )


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_data(limit: int = 300) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT timestamp, open, high, low, close, volume
            FROM btc_price_1h
            ORDER BY timestamp DESC
            LIMIT {limit}
            """,
            conn,
        )

    if df.empty:
        raise RuntimeError("No BTC price data found in btc_price_1h")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ── Color helpers ─────────────────────────────────────────────────────────────

def _score_color(score: float | None) -> str:
    if score is None:
        return "white"
    if score >= 0.65:
        return "green"
    if score >= 0.40:
        return "yellow"
    return "red"


def _intent_color(intent: str) -> str:
    if "BULLISH" in intent or "BUY" in intent:
        return "green"
    if "BEARISH" in intent or "SELL" in intent:
        return "red"
    if "EXHAUSTION" in intent:
        return "red"
    return "yellow"


def _momentum_color(state: str) -> str:
    return {
        "ACCELERATING": "green",
        "BUILDING":     "bright_green",
        "STABLE":       "white",
        "STALLING":     "yellow",
        "DECAYING":     "yellow",
        "REVERSAL_RISK": "red",
    }.get(state, "white")


def _pressure_color(bias: str) -> str:
    if bias == "BUY_PRESSURE":
        return "green"
    if bias == "SELL_PRESSURE":
        return "red"
    return "yellow"


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pillar4():
    console = Console()

    with console.status("[bold green]Analyzing BTC candle intelligence...", spinner="dots"):
        df     = _load_data()
        config = _build_config()
        result = run_pillar4_candle_intelligence(
            df=df,
            pillar4_config=config,
            columns=OhlcColumns(
                open="open", high="high", low="low",
                close="close", volume="volume", timestamp="timestamp",
            ),
            asset="BTCUSDT",
            timeframe="1h",
            lookback_bars_used=min(30, len(df)),
            atr_method="wilder",
        )

    summary   = result["candle_summary"]
    features  = result["latest_candle_features"]
    vol       = result["volatility_context"]
    pressure  = result["pressure"]
    absorption = result["absorption"]
    breakout  = result["breakout_analysis"]
    intents   = result["intent_scores"]
    flags     = result["risk_flags"]

    di   = summary["dominant_intent"]
    ms   = summary["momentum_state"]
    cs   = summary["control_state"]
    es   = summary["expansion_state"]
    os_  = summary["overlap_state"]
    ft   = summary["follow_through_quality"]
    exh  = summary["exhaustion_state"]
    conf = summary["intent_confidence"] or 0.0

    pb = pressure["pressure_bias"]
    ps = pressure["pressure_strength"]

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold white]BTC/USDT TERMINAL[/bold white] | [cyan]LIVE PILLAR 4 — CANDLE INTELLIGENCE ENGINE[/cyan]",
        border_style="bright_blue"
    ))

    # ── Candle Summary ────────────────────────────────────────────────────────
    summary_table = Table(show_header=True, header_style="bold magenta",
                          expand=True, box=box.SIMPLE_HEAVY)
    summary_table.add_column("Metric", style="dim", width=28)
    summary_table.add_column("Value")

    ic = _intent_color(di)
    summary_table.add_row("Dominant Intent",
                           f"[{ic} bold]{di}[/{ic} bold]")
    summary_table.add_row("Intent Confidence",
                           f"[{_score_color(conf)} bold]{conf:.1%}[/{_score_color(conf)} bold]")
    summary_table.add_row("Momentum State",
                           f"[{_momentum_color(ms)} bold]{ms}[/{_momentum_color(ms)} bold]")
    summary_table.add_row("Control State",
                           f"[bold]{cs}[/bold]")
    summary_table.add_row("Expansion State",
                           f"[bold]{es}[/bold]")
    summary_table.add_row("Overlap State",
                           f"[dim]{os_}[/dim]")
    summary_table.add_row("Follow-Through",
                           f"[{'green' if ft == 'STRONG' else 'yellow' if ft == 'MODERATE' else 'red'} bold]{ft}[/{'green' if ft == 'STRONG' else 'yellow' if ft == 'MODERATE' else 'red'} bold]")
    summary_table.add_row("Exhaustion State",
                           f"[{'red' if exh != 'NONE' else 'green'} bold]{exh}[/{'red' if exh != 'NONE' else 'green'} bold]")
    summary_table.add_row("Timestamp (UTC)",
                           f"[dim]{result['timestamp_utc']}[/dim]")

    console.print(summary_table)

    # ── Candle Features ───────────────────────────────────────────────────────
    feat_table = Table(title="Latest Candle Features", show_header=True,
                       header_style="bold cyan", box=box.SIMPLE_HEAVY, expand=True)
    feat_table.add_column("Feature")
    feat_table.add_column("Value", justify="right")

    feat_table.add_row("Body / Range Ratio",      f"{features['body_to_range_ratio']:.3f}")
    feat_table.add_row("Close Location",           f"{features['close_location_value']:.3f}")
    feat_table.add_row("Bar Efficiency",           f"{features['bar_efficiency']:.3f}")
    feat_table.add_row("Upper Wick / Range",       f"{features['upper_wick_to_range_ratio']:.3f}")
    feat_table.add_row("Lower Wick / Range",       f"{features['lower_wick_to_range_ratio']:.3f}")
    feat_table.add_row("ATR-Scaled Range",         f"{features['atr_scaled_range']:.3f}")
    feat_table.add_row("Range Expansion Score",    f"{features['range_expansion_score']:.3f}")
    feat_table.add_row("Overlap vs Prev Bar",      f"{features['overlap_ratio_vs_prev_bar']:.3f}")
    feat_table.add_row("ATR",                      f"${vol['atr']:,.2f}" if vol['atr'] else "N/A")
    feat_table.add_row("Realized Vol",             f"{vol['realized_volatility']:.4f}" if vol['realized_volatility'] else "N/A")
    feat_table.add_row("Vol Percentile",           f"{vol['realized_volatility_percentile']:.1%}" if vol['realized_volatility_percentile'] else "N/A")

    console.print(feat_table)

    # ── Pressure ──────────────────────────────────────────────────────────────
    pres_table = Table(title="Pressure Analysis", show_header=True,
                       header_style="bold cyan", box=box.SIMPLE_HEAVY)
    pres_table.add_column("Metric")
    pres_table.add_column("Value", justify="right")

    pc = _pressure_color(pb)
    pres_table.add_row("Pressure Bias",
                        f"[{pc} bold]{pb}[/{pc} bold]")
    pres_table.add_row("Pressure Strength",   f"[bold]{ps}[/bold]")
    pres_table.add_row("Buying Pressure",     f"[green]{pressure['buying_pressure_score']:.3f}[/green]")
    pres_table.add_row("Selling Pressure",    f"[red]{pressure['selling_pressure_score']:.3f}[/red]")
    pres_table.add_row("Net Pressure",        f"{pressure['net_pressure_score']:.3f}")

    console.print(pres_table)

    # ── Absorption ────────────────────────────────────────────────────────────
    abs_table = Table(title="Absorption & Rejection", show_header=True,
                      header_style="bold cyan", box=box.SIMPLE_HEAVY)
    abs_table.add_column("Metric")
    abs_table.add_column("Value", justify="right")

    abs_table.add_row("Dominant Absorption",  absorption["dominant_absorption"])
    abs_table.add_row("Dominant Rejection",   absorption["dominant_rejection"])
    abs_table.add_row("Buy Absorption Score", f"{absorption['buy_absorption_score']:.3f}")
    abs_table.add_row("Sell Absorption Score", f"{absorption['sell_absorption_score']:.3f}")
    abs_table.add_row("Confidence",           f"{absorption['absorption_confidence']:.3f}")

    console.print(abs_table)

    # ── Breakout ──────────────────────────────────────────────────────────────
    brk_table = Table(title="Breakout Analysis", show_header=True,
                      header_style="bold cyan", box=box.SIMPLE_HEAVY)
    brk_table.add_column("Metric")
    brk_table.add_column("Value", justify="right")

    bv = breakout["breakout_validity"]
    bvc = "green" if bv == "CONFIRMED" else "red" if bv == "FAILING" else "yellow"

    brk_table.add_row("Direction",       breakout["breakout_direction"])
    brk_table.add_row("Validity",        f"[{bvc} bold]{bv}[/{bvc} bold]")
    brk_table.add_row("State",           breakout["breakout_state"])
    brk_table.add_row("Quality Score",   f"{breakout['breakout_quality_score']:.3f}")
    brk_table.add_row("Acceptance",      f"{breakout['acceptance_score']:.3f}")
    brk_table.add_row("Failure Score",   f"{breakout['failure_score']:.3f}")
    brk_table.add_row("Fake BO Risk",    f"{breakout['fake_breakout_risk']:.3f}")

    console.print(brk_table)

    # ── Top Intent Scores ─────────────────────────────────────────────────────
    console.print("\n[bold underline]Top Intent Scores:[/bold underline]")
    sorted_intents = sorted(intents.items(), key=lambda x: x[1] or 0, reverse=True)[:5]
    for intent_name, score in sorted_intents:
        if score is None:
            continue
        color = _intent_color(intent_name.replace("_score", "").upper())
        bar = "█" * int(score * 20)
        console.print(
            f"  [{color}]{intent_name.replace('_score','').replace('_',' ').title():<35}[/{color}] "
            f"[dim]{bar:<20}[/dim] {score:.3f}"
        )

    # ── Risk Flags ────────────────────────────────────────────────────────────
    if flags:
        console.print("\n[bold underline]Risk Flags:[/bold underline]")
        for flag in flags:
            console.print(f"  [red bold]⚠[/red bold]  {flag}")
    else:
        console.print("\n[green]✓ No active candle risk flags[/green]")

    # ── AI Overview ───────────────────────────────────────────────────────────
    console.print("\n[bold underline]AI Candle Intelligence Overview:[/bold underline]")
    for line in result["ai_overview"].split("\n"):
        if line.strip():
            console.print(f"  [white]{line.strip()}[/white]")

    console.print("\n" + "—" * 60 + "\n")


if __name__ == "__main__":
    run_pillar4()