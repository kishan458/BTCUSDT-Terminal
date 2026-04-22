import sqlite3
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from core.db import resolve_db_path
from pillar3_structure_liquidity_engine.pillar3_output import run_pillar3_output

DB_PATH = str(resolve_db_path())


def _load_btc_data(limit: int = 200) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"""
        SELECT timestamp, open, high, low, close, volume
        FROM btc_price_1h
        ORDER BY timestamp DESC
        LIMIT {limit}
        """,
        conn,
    )
    conn.close()

    if df.empty:
        raise ValueError("No BTC price data found in btc_price_1h")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _risk_color(label: str) -> str:
    if label == "HIGH":
        return "red"
    if label == "MODERATE":
        return "yellow"
    if label == "LOW":
        return "green"
    return "white"


def _trap_color(prob: float | None) -> str:
    if prob is None:
        return "white"
    if prob >= 0.70:
        return "red"
    if prob >= 0.40:
        return "yellow"
    return "green"


def _side_color(side: str) -> str:
    if side in ["BUY_SIDE", "STOP_CLUSTER_ABOVE"]:
        return "green"
    if side in ["SELL_SIDE", "STOP_CLUSTER_BELOW"]:
        return "red"
    return "yellow"


def run_pillar3():
    console = Console()

    with console.status("[bold green]Fetching BTC structure & liquidity data...", spinner="dots"):
        df     = _load_btc_data()
        result = run_pillar3_output(df)

    summary     = result["structure_liquidity_summary"]
    liquidity   = result["liquidity_levels"]
    structure   = result["structure_state"]
    trap        = result["trap_detection"]
    liquidation = result["liquidation_risk"]
    targets     = result["liquidity_targets"]
    flags       = result["risk_flags"]

    current_price = float(df["close"].iloc[-1])
    dom_side      = summary["dominant_liquidity_side"]
    trap_risk     = summary["trap_risk"]
    liq_env       = summary["liquidity_environment"]

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold white]BTC/USDT TERMINAL[/bold white] | [cyan]LIVE PILLAR 3 — STRUCTURE & LIQUIDITY ENGINE[/cyan]",
        border_style="bright_blue"
    ))

    # ── Summary Table ─────────────────────────────────────────────────────────
    summary_table = Table(show_header=True, header_style="bold magenta",
                          expand=True, box=box.SIMPLE_HEAVY)
    summary_table.add_column("Metric", style="dim", width=30)
    summary_table.add_column("Value")

    summary_table.add_row("Current Price",
                           f"[bold white]${current_price:,.2f}[/bold white]")
    summary_table.add_row("Market Structure",
                           f"[bold cyan]{structure['market_structure']}[/bold cyan]")
    summary_table.add_row("Range State",
                           f"[bold]{structure['range_state']}[/bold]")
    summary_table.add_row("Compression State",
                           f"[bold]{structure['compression_state']}[/bold]")
    summary_table.add_row("Dominant Liquidity Side",
                           f"[{_side_color(dom_side)} bold]{dom_side}[/{_side_color(dom_side)} bold]")
    summary_table.add_row("Liquidity Environment",
                           f"[{_side_color(liq_env)} bold]{liq_env}[/{_side_color(liq_env)} bold]")
    trap_risk_color = _risk_color(trap_risk)
    summary_table.add_row("Trap Risk",
                           f"[{trap_risk_color} bold]{trap_risk}[/{trap_risk_color} bold]")
    summary_table.add_row("Timestamp (UTC)",
                           f"[dim]{result['timestamp_utc']}[/dim]")

    console.print(summary_table)

    # ── Liquidity Levels ──────────────────────────────────────────────────────
    liq_table = Table(title="Liquidity Levels", show_header=True,
                      header_style="bold cyan", box=box.SIMPLE_HEAVY, expand=True)
    liq_table.add_column("Level")
    liq_table.add_column("Price", justify="right")
    liq_table.add_column("Distance", justify="right")

    def _dist(price, ref):
        if ref is None:
            return "N/A"
        return f"{((ref - price) / price * 100):+.2f}%"

    buy_side  = liquidity["buy_side_liquidity"]
    sell_side = liquidity["sell_side_liquidity"]
    magnet    = liquidity["nearest_liquidity_magnet"]

    if buy_side:
        liq_table.add_row(
            "[green]Buy-Side Liquidity[/green]",
            f"[green]${buy_side:,.2f}[/green]",
            f"[green]{_dist(current_price, buy_side)}[/green]"
        )
    if sell_side:
        liq_table.add_row(
            "[red]Sell-Side Liquidity[/red]",
            f"[red]${sell_side:,.2f}[/red]",
            f"[red]{_dist(current_price, sell_side)}[/red]"
        )
    if magnet:
        liq_table.add_row(
            "[yellow bold]Nearest Magnet[/yellow bold]",
            f"[yellow bold]${magnet:,.2f}[/yellow bold]",
            f"[yellow bold]{_dist(current_price, magnet)}[/yellow bold]"
        )

    console.print(liq_table)

    # ── Trap Detection ────────────────────────────────────────────────────────
    trap_table = Table(title="Trap Detection", show_header=True,
                       header_style="bold cyan", box=box.SIMPLE_HEAVY)
    trap_table.add_column("Metric")
    trap_table.add_column("Value", justify="right")

    bt = trap["breakout_trap_probability"]
    bd = trap["breakdown_trap_probability"]
    lt = trap["likely_trap_side"]

    trap_table.add_row(
        "Breakout Trap Probability",
        f"[{_trap_color(bt)} bold]{bt:.1%}[/{_trap_color(bt)} bold]" if bt else "N/A"
    )
    trap_table.add_row(
        "Breakdown Trap Probability",
        f"[{_trap_color(bd)} bold]{bd:.1%}[/{_trap_color(bd)} bold]" if bd else "N/A"
    )
    trap_table.add_row(
        "Likely Trap Side",
        f"[yellow bold]{lt}[/yellow bold]"
    )

    console.print(trap_table)

    # ── Liquidation Risk ──────────────────────────────────────────────────────
    liq_risk_table = Table(title="Liquidation Risk", show_header=True,
                           header_style="bold cyan", box=box.SIMPLE_HEAVY)
    liq_risk_table.add_column("Metric")
    liq_risk_table.add_column("Value", justify="right")

    ll = liquidation["long_liquidation_risk"]
    sl = liquidation["short_liquidation_risk"]
    cp = liquidation["cascade_probability"]

    liq_risk_table.add_row(
        "Long Liquidation Risk",
        f"[{_risk_color(ll)} bold]{ll}[/{_risk_color(ll)} bold]"
    )
    liq_risk_table.add_row(
        "Short Liquidation Risk",
        f"[{_risk_color(sl)} bold]{sl}[/{_risk_color(sl)} bold]"
    )
    liq_risk_table.add_row(
        "Cascade Probability",
        f"[{_trap_color(cp)} bold]{cp:.1%}[/{_trap_color(cp)} bold]" if cp else "N/A"
    )

    console.print(liq_risk_table)

    # ── Liquidity Targets ─────────────────────────────────────────────────────
    if targets:
        console.print("\n[bold underline]Liquidity Targets (nearest first):[/bold underline]")
        for i, t in enumerate(targets[:6], 1):
            dist = _dist(current_price, t)
            arrow = "▲" if t > current_price else "▼"
            color = "green" if t > current_price else "red"
            console.print(
                f"  [{color}]{arrow}[/{color}] "
                f"[bold]${t:,.2f}[/bold]  [dim]{dist}[/dim]"
            )

    # ── Risk Flags ────────────────────────────────────────────────────────────
    if flags:
        console.print("\n[bold underline]Risk Flags:[/bold underline]")
        for flag in flags:
            console.print(f"  [red bold]⚠[/red bold]  {flag}")
    else:
        console.print("\n[green]✓ No active liquidity risk flags[/green]")

    # ── AI Overview ───────────────────────────────────────────────────────────
    console.print("\n[bold underline]AI Structure & Liquidity Overview:[/bold underline]")
    for line in result["ai_overview"].split("\n"):
        if line.strip():
            console.print(f"  [white]{line.strip()}[/white]")

    console.print("\n" + "—" * 60 + "\n")


if __name__ == "__main__":
    run_pillar3()