from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# ── Path fix ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.db import resolve_db_path  # noqa: E402

from pillar2_market_memory_engine.memory_feature_engine import MemoryFeatureConfig  # noqa: E402
from pillar2_market_memory_engine.memory_outcome_engine import MemoryOutcomeConfig  # noqa: E402
from pillar2_market_memory_engine.state_signature_engine import StateSignatureConfig  # noqa: E402
from pillar2_market_memory_engine.analog_retrieval_engine import AnalogRetrievalConfig  # noqa: E402
from pillar2_market_memory_engine.conditional_outcome_engine import ConditionalOutcomeConfig  # noqa: E402
from pillar2_market_memory_engine.stability_engine import StabilityConfig  # noqa: E402
from pillar2_market_memory_engine.session_memory_engine import SessionMemoryConfig  # noqa: E402
from pillar2_market_memory_engine.calendar_memory_engine import CalendarMemoryConfig  # noqa: E402
from pillar2_market_memory_engine.pillar2_output import Pillar2OutputConfig, run_pillar2_output  # noqa: E402
from pillar2_market_memory_engine.ai_overview_engine import AIOverviewConfig, build_ai_overview  # noqa: E402

DB_PATH = str(resolve_db_path())


# ── Color helpers ─────────────────────────────────────────────────────────────

def _bias_color(bias: str) -> str:
    if "UP" in bias or "CONTINUATION" in bias:
        return "green"
    if "DOWN" in bias or "REVERSAL" in bias:
        return "red"
    if "MEAN_REVERSION" in bias:
        return "cyan"
    return "yellow"


def _prob_color(prob: float | None) -> str:
    if prob is None:
        return "white"
    if prob >= 0.60:
        return "green"
    if prob >= 0.52:
        return "bright_green"
    if prob <= 0.40:
        return "red"
    return "yellow"


def _quality_color(label: str) -> str:
    if label == "HIGH":
        return "green"
    if label == "MODERATE":
        return "yellow"
    if label in ("LOW", "INSUFFICIENT"):
        return "red"
    return "white"


def _tendency_color(t: str) -> str:
    if "UPSIDE" in t or "CONTINUATION" in t:
        return "green"
    if "DOWNSIDE" in t or "REVERSAL" in t:
        return "red"
    if "MEAN_REVERSION" in t:
        return "cyan"
    return "dim"


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pillar2():
    console = Console()

    feature_cfg = MemoryFeatureConfig(
        db_path=DB_PATH,
        table_name="btc_price_1h",
        timestamp_col="timestamp",
    )
    outcome_cfg     = MemoryOutcomeConfig(horizons=(1, 3, 6, 12))
    signature_cfg   = StateSignatureConfig(require_complete_core_state=True)
    retrieval_cfg   = AnalogRetrievalConfig(min_similarity_score=0.45, max_weighted_matches=300)
    conditional_cfg = ConditionalOutcomeConfig(use_weighted_matches=True)
    stability_cfg   = StabilityConfig(minimum_window_samples=10)
    session_cfg     = SessionMemoryConfig(min_samples_per_group=25)
    calendar_cfg    = CalendarMemoryConfig(min_samples_per_group=25)
    output_cfg      = Pillar2OutputConfig(asset="BTCUSDT", round_decimals=4)
    ai_cfg          = AIOverviewConfig()

    with console.status("[bold green]Running BTC market memory engine...", spinner="dots"):
        result = run_pillar2_output(
            feature_config=feature_cfg,
            outcome_config=outcome_cfg,
            signature_config=signature_cfg,
            retrieval_config=retrieval_cfg,
            conditional_config=conditional_cfg,
            stability_config=stability_cfg,
            session_config=session_cfg,
            calendar_config=calendar_cfg,
            output_config=output_cfg,
        )

    with console.status("[bold green]Generating AI memory overview...", spinner="dots"):
        ai_text, ai_source, ai_error = build_ai_overview(result, ai_cfg)
        result["ai_overview"] = ai_text

    ms   = result["memory_summary"]
    sig  = result["current_state_signature"]
    ana  = result["historical_analogs"]
    fwd  = result["forward_outcomes"]
    dist = result["distribution_diagnostics"]
    stab = result["stability_diagnostics"]
    ctx  = result["context_memory"]
    flags = result["risk_flags"]

    memory_bias   = ms.get("memory_bias", "UNKNOWN")
    match_quality = ms.get("historical_match_quality", "UNKNOWN")

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold white]BTC/USDT TERMINAL[/bold white] | [cyan]LIVE PILLAR 2 — MARKET MEMORY ENGINE[/cyan]",
        border_style="bright_blue"
    ))

    # ── Memory Summary ────────────────────────────────────────────────────────
    mem_table = Table(show_header=True, header_style="bold magenta",
                      expand=True, box=box.SIMPLE_HEAVY)
    mem_table.add_column("Metric", style="dim", width=30)
    mem_table.add_column("Value")

    bc = _bias_color(memory_bias)
    qc = _quality_color(match_quality)

    mem_table.add_row("Memory Bias",
                       f"[{bc} bold]{memory_bias}[/{bc} bold]")
    mem_table.add_row("Match Quality",
                       f"[{qc} bold]{match_quality}[/{qc} bold]")
    mem_table.add_row("Sample Size",
                       f"[bold]{ms.get('sample_size', 'N/A')}[/bold]")
    mem_table.add_row("Headline Confidence",
                       f"{ms.get('headline_confidence', 0):.3f}" if ms.get('headline_confidence') else "N/A")
    mem_table.add_row("Timestamp (UTC)",
                       f"[dim]{result.get('timestamp_utc', 'N/A')}[/dim]")

    console.print(mem_table)

    # ── Current State Signature ───────────────────────────────────────────────
    sig_table = Table(title="Current State Signature", show_header=True,
                      header_style="bold cyan", box=box.SIMPLE_HEAVY, expand=True)
    sig_table.add_column("Field")
    sig_table.add_column("Value", justify="right")

    sig_table.add_row("Session",          f"[cyan]{sig.get('session', 'N/A')}[/cyan]")
    sig_table.add_row("Weekday",          sig.get("weekday", "N/A"))
    sig_table.add_row("Volatility",       sig.get("volatility_bucket", "N/A"))
    sig_table.add_row("Momentum State",   sig.get("momentum_state", "N/A"))
    sig_table.add_row("Overlap State",    sig.get("overlap_state", "N/A"))
    sig_table.add_row("Follow-Through",   sig.get("follow_through_quality", "N/A"))
    sig_table.add_row("Pressure Bias",    sig.get("pressure_bias", "N/A"))
    sig_table.add_row("Breakout State",   sig.get("breakout_state", "N/A"))
    sig_table.add_row("Range Position",   sig.get("range_position", "N/A"))

    console.print(sig_table)

    # ── Analog Match Summary ──────────────────────────────────────────────────
    ana_table = Table(title="Analog Match Summary", show_header=True,
                      header_style="bold cyan", box=box.SIMPLE_HEAVY)
    ana_table.add_column("Metric")
    ana_table.add_column("Value", justify="right")

    ana_table.add_row("Total Matches",       str(ana.get("match_count", 0)))
    ana_table.add_row("Exact Matches",       str(ana.get("exact_match_count", 0)))
    ana_table.add_row("Weighted Matches",    str(ana.get("weighted_match_count", 0)))
    ana_table.add_row("Avg Similarity Score",
                       f"{ana.get('analog_quality_score', 0):.3f}" if ana.get('analog_quality_score') else "N/A")

    console.print(ana_table)

    # ── Forward Outcomes ──────────────────────────────────────────────────────
    fwd_table = Table(title="Forward Outcome Probabilities", show_header=True,
                      header_style="bold cyan", box=box.SIMPLE_HEAVY, expand=True)
    fwd_table.add_column("Metric")
    fwd_table.add_column("Value", justify="right")

    def _pct(v):
        return f"{v:.1%}" if v is not None else "N/A"

    def _ret(v):
        return f"{v:.4f}" if v is not None else "N/A"

    n3 = fwd.get("next_3_bar_up_probability")
    n6 = fwd.get("next_6_bar_up_probability")
    c  = fwd.get("continuation_probability")
    r  = fwd.get("reversal_probability")
    mr = fwd.get("mean_reversion_probability")

    fwd_table.add_row("Next 3-Bar Up",        f"[{_prob_color(n3)}]{_pct(n3)}[/{_prob_color(n3)}]")
    fwd_table.add_row("Next 6-Bar Up",        f"[{_prob_color(n6)}]{_pct(n6)}[/{_prob_color(n6)}]")
    fwd_table.add_row("Continuation Prob",    f"[{_prob_color(c)}]{_pct(c)}[/{_prob_color(c)}]")
    fwd_table.add_row("Reversal Prob",        f"[{_prob_color(r)}]{_pct(r)}[/{_prob_color(r)}]")
    fwd_table.add_row("Mean Reversion Prob",  f"[{_prob_color(mr)}]{_pct(mr)}[/{_prob_color(mr)}]")
    fwd_table.add_row("Mean Return 6-Bar",    _ret(fwd.get("mean_forward_return_6")))
    fwd_table.add_row("Mean MFE 6-Bar",      _ret(fwd.get("mean_mfe_6")))
    fwd_table.add_row("Mean MAE 6-Bar",      _ret(fwd.get("mean_mae_6")))

    console.print(fwd_table)

    # ── Distribution ──────────────────────────────────────────────────────────
    dist_table = Table(title="Distribution Diagnostics", show_header=True,
                       header_style="bold cyan", box=box.SIMPLE_HEAVY)
    dist_table.add_column("Metric")
    dist_table.add_column("Value", justify="right")

    dist_table.add_row("Return Std 6-Bar",    _ret(dist.get("return_std_6")))
    dist_table.add_row("Left Tail (10%)",     _ret(dist.get("left_tail_10pct_6")))
    dist_table.add_row("Right Tail (90%)",    _ret(dist.get("right_tail_90pct_6")))
    dist_table.add_row("Skew Proxy",          _ret(dist.get("skew_proxy_6")))

    console.print(dist_table)

    # ── Stability ─────────────────────────────────────────────────────────────
    stab_table = Table(title="Temporal Stability", show_header=True,
                       header_style="bold cyan", box=box.SIMPLE_HEAVY)
    stab_table.add_column("Window")
    stab_table.add_column("Bias", justify="right")
    stab_table.add_column("Up Prob", justify="right")
    stab_table.add_column("N", justify="right")

    for window in ["older", "middle", "recent"]:
        bias = stab.get(f"{window}_window_bias", "N/A")
        prob = stab.get(f"{window}_window_up_probability")
        n    = stab.get(f"{window}_window_sample_size", 0)
        bc2  = _bias_color(bias)
        stab_table.add_row(
            window.capitalize(),
            f"[{bc2}]{bias}[/{bc2}]",
            _pct(prob),
            str(n),
        )

    console.print(stab_table)
    console.print(f"  [dim]Temporal Stability: {stab.get('temporal_stability_score', 'N/A'):.3f}   "
                  f"Regime Dependency: {stab.get('regime_dependency_score', 'N/A'):.3f}[/dim]")

    # ── Context Memory ────────────────────────────────────────────────────────
    console.print("\n[bold underline]Context Memory Tendencies:[/bold underline]")
    for label, key in [
        ("Session",       "session_tendency"),
        ("Calendar",      "calendar_tendency"),
        ("Weekday-Hour",  "weekday_hour_tendency"),
        ("Week Part",     "weekpart_tendency"),
        ("Open Window",   "open_window_tendency"),
    ]:
        val = ctx.get(key, "N/A")
        tc  = _tendency_color(val)
        console.print(f"  [dim]{label:<15}[/dim] [{tc}]{val}[/{tc}]")

    # ── Risk Flags ────────────────────────────────────────────────────────────
    if flags:
        console.print("\n[bold underline]Risk Flags:[/bold underline]")
        for flag in flags:
            console.print(f"  [red bold]⚠[/red bold]  {flag}")
    else:
        console.print("\n[green]✓ No active memory risk flags[/green]")

    # ── AI Overview ───────────────────────────────────────────────────────────
    console.print(f"\n[bold underline]AI Memory Overview[/bold underline] "
                  f"[dim]({ai_source})[/dim]")
    for line in result["ai_overview"].split("\n"):
        if line.strip():
            console.print(f"  [white]{line.strip()}[/white]")

    console.print("\n" + "—" * 60 + "\n")


if __name__ == "__main__":
    run_pillar2()