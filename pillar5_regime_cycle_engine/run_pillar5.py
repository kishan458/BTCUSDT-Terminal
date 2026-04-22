from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from pillar5_regime_cycle_engine.pillar5_output import build_pillar5_output


def _regime_color(regime: str) -> str:
    if regime in ["STRONG_UPTREND", "EXPANSION", "RECOVERY"]:
        return "green"
    if regime in ["STRONG_DOWNTREND", "MARKDOWN", "EXHAUSTION"]:
        return "red"
    if regime in ["WEAK_UPTREND", "BREAKOUT_TRANSITION"]:
        return "bright_green"
    if regime in ["WEAK_DOWNTREND", "BREAKDOWN_TRANSITION", "DISTRIBUTION"]:
        return "bright_red"
    if regime in ["DISLOCATED"]:
        return "red"
    if regime in ["EXPANDING"]:
        return "yellow"
    if regime in ["COMPRESSED"]:
        return "cyan"
    return "yellow"


def _compat_color(val: str) -> str:
    if val == "FAVORED":
        return "green"
    if val == "MODERATELY_FAVORED":
        return "yellow"
    if val == "NOT_FAVORED":
        return "red"
    return "white"


def run_pillar5():
    console = Console()

    with console.status("[bold green]Analyzing BTC market regime...", spinner="dots"):
        result = build_pillar5_output()

    regime   = result["regime_summary"]
    metrics  = result["market_metrics"]
    strategy = result["strategy_compatibility"]
    session  = result["session_context"]
    flags    = result["risk_flags"]
    conf     = result["confidence_score"]
    expl     = result["regime_explanation"]

    dr  = regime["directional_regime"]
    vr  = regime["volatility_regime"]
    ms  = regime["market_state"]
    cp  = regime["cycle_phase"]
    sd  = strategy["stand_down"]

    conf_color = "green" if conf >= 0.75 else "yellow" if conf >= 0.55 else "red"

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold white]BTC/USDT TERMINAL[/bold white] | [cyan]LIVE PILLAR 5 — REGIME & CYCLE ENGINE[/cyan]",
        border_style="bright_blue"
    ))

    # ── Regime Summary ────────────────────────────────────────────────────────
    regime_table = Table(show_header=True, header_style="bold magenta",
                         expand=True, box=box.SIMPLE_HEAVY)
    regime_table.add_column("Metric", style="dim", width=28)
    regime_table.add_column("Value")

    regime_table.add_row(
        "Directional Regime",
        f"[{_regime_color(dr)} bold]{dr}[/{_regime_color(dr)} bold]"
    )
    regime_table.add_row(
        "Volatility Regime",
        f"[{_regime_color(vr)} bold]{vr}[/{_regime_color(vr)} bold]"
    )
    regime_table.add_row(
        "Market State",
        f"[{_regime_color(ms)} bold]{ms}[/{_regime_color(ms)} bold]"
    )
    regime_table.add_row(
        "Cycle Phase",
        f"[{_regime_color(cp)} bold]{cp}[/{_regime_color(cp)} bold]"
    )
    regime_table.add_row(
        "Confidence Score",
        f"[{conf_color} bold]{conf:.1%}[/{conf_color} bold]"
    )
    regime_table.add_row(
        "Stand Down",
        f"[red bold]YES — AVOID TRADING[/red bold]" if sd else "[green]NO — CONDITIONS ACCEPTABLE[/green]"
    )
    regime_table.add_row(
        "Current Session",
        f"[cyan]{session['current_session']}[/cyan]"
    )
    regime_table.add_row(
        "Last Candle (UTC)",
        f"[dim]{result['timestamp_utc']}[/dim]"
    )

    console.print(regime_table)

    # ── Price & Metrics ───────────────────────────────────────────────────────
    ohlcv    = metrics["ohlcv"]
    returns  = metrics["returns"]
    vol      = metrics["volatility"]
    ma       = metrics["moving_average_structure"]
    momentum = metrics["momentum"]
    dist     = metrics["distance_from_key_mas"]

    price_table = Table(title="Price & Key Metrics", show_header=True,
                        header_style="bold cyan", box=box.SIMPLE_HEAVY, expand=True)
    price_table.add_column("Metric")
    price_table.add_column("Value", justify="right")

    price_table.add_row("Close",           f"${ohlcv['close']:,.2f}")
    price_table.add_row("High / Low",      f"${ohlcv['high']:,.2f} / ${ohlcv['low']:,.2f}")
    price_table.add_row("Return 1h",       f"{returns['return_1bar']:.2%}" if returns['return_1bar'] else "N/A")
    price_table.add_row("Return 24h",      f"{returns['return_24bar']:.2%}" if returns['return_24bar'] else "N/A")
    price_table.add_row("Return 7d",       f"{returns['return_7d']:.2%}" if returns['return_7d'] else "N/A")
    price_table.add_row("ATR %",           f"{vol['atr_pct']:.3%}" if vol['atr_pct'] else "N/A")
    price_table.add_row("Vol Percentile",  f"{vol['volatility_percentile']:.1%}" if vol['volatility_percentile'] else "N/A")
    price_table.add_row("Momentum Score",  f"{momentum['momentum_score']:.4f}" if momentum['momentum_score'] else "N/A")
    price_table.add_row("MA Order",        ma["ma_order"])
    price_table.add_row("Swing Structure", metrics["swing_structure"]["structure_state"])
    price_table.add_row("Dist EMA20",      f"{dist['distance_to_ema20_pct']:.2%}" if dist['distance_to_ema20_pct'] else "N/A")
    price_table.add_row("Dist EMA50",      f"{dist['distance_to_ema50_pct']:.2%}" if dist['distance_to_ema50_pct'] else "N/A")
    price_table.add_row("Session High",    f"${session['session_high']:,.2f}")
    price_table.add_row("Session Low",     f"${session['session_low']:,.2f}")

    console.print(price_table)

    # ── Strategy Compatibility ────────────────────────────────────────────────
    strat_table = Table(title="Strategy Compatibility", show_header=True,
                        header_style="bold cyan", box=box.SIMPLE_HEAVY)
    strat_table.add_column("Strategy")
    strat_table.add_column("Signal", justify="right")

    for strat, val in [
        ("Trend Following",  strategy["trend_following"]),
        ("Breakout Trading", strategy["breakout_trading"]),
        ("Mean Reversion",   strategy["mean_reversion"]),
    ]:
        color = _compat_color(val)
        strat_table.add_row(strat, f"[{color} bold]{val}[/{color} bold]")

    console.print(strat_table)

    # ── Regime Explanation ────────────────────────────────────────────────────
    console.print("\n[bold underline]Regime Explanation:[/bold underline]")
    console.print(f"  [cyan]Trend:[/cyan]    {expl['trend_context']}")
    console.print(f"  [yellow]Volatility:[/yellow] {expl['volatility_context']}")
    console.print(f"  [magenta]Cycle:[/magenta]    {expl['cycle_context']}")

    # ── Risk Flags ────────────────────────────────────────────────────────────
    if flags:
        console.print("\n[bold underline]Risk Flags:[/bold underline]")
        for flag in flags:
            console.print(f"  [red bold]⚠[/red bold]  {flag}")
    else:
        console.print("\n[green]✓ No active risk flags[/green]")

    # ── AI Overview ───────────────────────────────────────────────────────────
    console.print("\n[bold underline]AI Regime Overview:[/bold underline]")
    for line in result["ai_overview"].split("\n"):
        if line.strip():
            console.print(f"  [white]{line.strip()}[/white]")

    console.print("\n" + "—" * 60 + "\n")


if __name__ == "__main__":
    run_pillar5()