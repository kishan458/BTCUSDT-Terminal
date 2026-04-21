from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich import box

from pillar6_high_impact_events.pillar6_output import build_pillar6_output


def run_pillar6():
    console = Console()

    with console.status("[bold green]Fetching macro events & computing uncertainty...", spinner="dots"):
        result = build_pillar6_output()

    # ── Colour logic ──────────────────────────────────────────────────────────
    skew = result.get("dominant_risk_skew", "UNKNOWN")
    state = result.get("state", "UNKNOWN")

    if skew == "RISK_OFF":
        skew_color = "red"
        skew_icon  = "🔴"
    elif skew == "RISK_ON":
        skew_color = "green"
        skew_icon  = "🟢"
    else:
        skew_color = "yellow"
        skew_icon  = "🟡"

    state_color = {
        "PRE_EVENT":  "yellow",
        "LIVE":       "red",
        "POST_EVENT": "cyan",
        "COOL_OFF":   "dim",
        "IDLE":       "white",
    }.get(state, "white")

    allow_trade = result.get("trade_restrictions", {}).get("allow_trade", False)
    trade_color = "green" if allow_trade else "red"
    trade_icon  = "✅ ALLOWED" if allow_trade else "🚫 RESTRICTED"

    unc   = result.get("base_uncertainty", 0.0)
    conf  = result.get("confidence_score", 0.0)

    unc_color  = "red" if unc >= 0.75 else "yellow" if unc >= 0.55 else "green"
    conf_color = "green" if conf >= 0.75 else "yellow" if conf >= 0.55 else "red"

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold white]BTC/USDT TERMINAL[/bold white] | [cyan]LIVE PILLAR 6 — HIGH IMPACT EVENT ENGINE[/cyan]",
        border_style="bright_blue"
    ))

    # ── Core Metrics Table ────────────────────────────────────────────────────
    metrics = Table(show_header=True, header_style="bold magenta", expand=True, box=box.SIMPLE_HEAVY)
    metrics.add_column("Metric", style="dim", width=28)
    metrics.add_column("Value")

    metrics.add_row("Next Event",
                    f"[bold white]{result.get('event', 'N/A')}[/bold white]")
    metrics.add_row("Event State",
                    f"[{state_color} bold]{state}[/{state_color} bold]")
    metrics.add_row("Scheduled (UTC)",
                    f"[dim]{result['debug'].get('scheduled_time_utc', 'N/A')}[/dim]")
    metrics.add_row("Importance",
                    f"[bold red]{result['debug'].get('importance', 'N/A')}[/bold red]")
    metrics.add_row("Base Uncertainty",
                    f"[{unc_color} bold]{unc:.1%}[/{unc_color} bold]")
    metrics.add_row("Model Confidence",
                    f"[{conf_color} bold]{conf:.1%}[/{conf_color} bold]")
    metrics.add_row("Dominant Risk Skew",
                    f"[{skew_color} bold]{skew_icon}  {skew}[/{skew_color} bold]")
    metrics.add_row("Trade Status",
                    f"[{trade_color} bold]{trade_icon}[/{trade_color} bold]")

    console.print(metrics)

    # ── Trade Restrictions ────────────────────────────────────────────────────
    tr = result.get("trade_restrictions", {})
    tr_table = Table(
        title="Trade Restrictions",
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        expand=True,
    )
    tr_table.add_column("Parameter")
    tr_table.add_column("Value", justify="right")

    tr_table.add_row("Size Multiplier", f"{tr.get('size_multiplier', 0.0):.1f}x")
    tr_table.add_row("Leverage Cap",    f"{tr.get('leverage_cap', 0.0):.1f}x")
    tr_table.add_row("Reason",          f"[dim]{tr.get('restriction_reason', 'N/A')}[/dim]")

    console.print(tr_table)

    # ── Scenarios ─────────────────────────────────────────────────────────────
    console.print("\n[bold underline]Event Scenarios:[/bold underline]")

    for s in result.get("scenarios", []):
        bias = s.get("risk_bias", "NEUTRAL")
        bias_color = "red" if bias == "RISK_OFF" else "green" if bias == "RISK_ON" else "yellow"

        prob_str = f"{s['probability']:.0%}" if s.get("probability") is not None else "Qualitative"

        reaction = s.get("expected_btc_reaction", {})
        move     = reaction.get("initial_move", "?")
        vol      = reaction.get("volatility", "?")

        console.print(
            f"  [{bias_color} bold]●[/{bias_color} bold] "
            f"[bold]{s.get('case')}[/bold]  "
            f"[dim]prob={prob_str}  move={move}  vol={vol}[/dim]"
        )
        console.print(
            f"    [dim]{s.get('macro_interpretation', '')}[/dim]"
        )

    # ── Terminal Guidance ─────────────────────────────────────────────────────
    console.print(f"\n[bold underline]Terminal Guidance:[/bold underline]")
    console.print(f"  [bold yellow]⚡[/bold yellow] {result.get('terminal_guidance', 'N/A')}")

    # ── AI Reasoning ──────────────────────────────────────────────────────────
    console.print(f"\n[bold underline]AI Macro Reasoning:[/bold underline]")
    for line in result.get("ai_reasoning", "").split("\n"):
        if line.strip():
            console.print(f"  [white]{line.strip()}[/white]")

    # ── Volatility Debug ──────────────────────────────────────────────────────
    dbg  = result.get("debug", {})
    comp = dbg.get("uncertainty_components", {})

    if comp and "vol_24h" in comp:
        vol_table = Table(
            title="Volatility Components",
            show_header=True,
            header_style="bold cyan",
            box=box.SIMPLE_HEAVY,
        )
        vol_table.add_column("Component")
        vol_table.add_column("Value", justify="right")

        vol_table.add_row("24h Volatility",       f"{comp.get('vol_24h', 0):.6f}")
        vol_table.add_row("7d Volatility",         f"{comp.get('vol_7d', 0):.6f}")
        vol_table.add_row("Vol Ratio (24h/7d)",    f"{comp.get('vol_ratio', 0):.4f}")
        vol_table.add_row("Vol Percentile",        f"{comp.get('pctl_vol_24h', 0):.1%}")
        vol_table.add_row("Minutes to Event",      f"{comp.get('minutes_to_event', 0):,.0f}")

        console.print()
        console.print(vol_table)

    console.print("\n" + "—" * 60 + "\n")


if __name__ == "__main__":
    run_pillar6()