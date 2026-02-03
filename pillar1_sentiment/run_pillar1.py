import json
from datetime import datetime, timezone

# Rich components for the "Pro" look
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

# Your existing imports
from pillar1_sentiment.collectors.institutional_news import collect_institutional_news
from pillar1_sentiment.processors.finbert_processor import analyze_articles
from pillar1_sentiment.processors.institutional_aggregator import aggregate_institutional_sentiment
from pillar1_sentiment.processors.institutional_summary_engine import (
    generate_institutional_summary
)

def run_pillar1():
    """
    All data here comes from your imported logic. 
    None of the values below are hardcoded.
    """
    # 1. Collect raw institutional news
    articles = collect_institutional_news()

    # 2. Run FinBERT sentiment + confidence
    analyzed_articles = analyze_articles(articles)

    # 3. Aggregate institutional signal
    aggregate = aggregate_institutional_sentiment(analyzed_articles)

    # 4. Generate institutional macro summary
    summary_text = generate_institutional_summary(
        analyzed_articles,
        aggregate
    )

    # Returning a dynamic dictionary based on real-time analysis
    return {
        "pillar": "institutional_sentiment",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregate_sentiment": {
            "label": aggregate.get("sentiment", "N/A"),
            "confidence": aggregate.get("confidence", 0.0),
        },
        "drivers": aggregate.get("drivers", []),
        "institutional_summary": [line.strip() for line in summary_text.split("\n") if line.strip()],
    }

if __name__ == "__main__":
    console = Console()
    
    # 1. Real-time Status Spinner
    with console.status("[bold green]Fetching & Analyzing Live Data...", spinner="dots"):
        # This calls your actual logic functions
        result = run_pillar1()

    # 2. Dynamic Logic for UI Colors
    # This changes the UI based on the actual DATA returned
    label = result["aggregate_sentiment"]["label"].lower()
    if "positive" in label:
        status_color = "green"
        icon = "📈"
    elif "negative" in label:
        status_color = "red"
        icon = "📉"
    else:
        status_color = "yellow"
        icon = "⚖️"

    # 3. Header Panel
    console.print("\n")
    console.print(Panel.fit(
        f"[bold white]BTC/USDT TERMINAL[/bold white] | [cyan]LIVE PILLAR 1 DATA[/cyan]",
        border_style="bright_blue"
    ))

    # 4. Dynamic Data Table
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Metric", style="dim", width=20)
    table.add_column("Live Analysis Value")

    table.add_row("Sentiment State", f"[{status_color} bold]{label.upper()} {icon}[/{status_color} bold]")
    table.add_row("Model Confidence", f"{result['aggregate_sentiment']['confidence']:.2%}")
    table.add_row("Last Updated", f"[dim]{result['timestamp']}[/dim]")
    
    console.print(table)

    # 5. Dynamic Drivers List
    if result["drivers"]:
        console.print(f"\n[bold underline {status_color}]Top Market Drivers:[/bold underline {status_color}]")
        for driver in result["drivers"]:
            console.print(f" [white]»[/white] {driver}")

    # 6. Dynamic Summary Breakdown
    console.print(f"\n[bold underline]Institutional Macro Summary:[/bold underline]")
    for line in result["institutional_summary"]:
        # Clean up bullet points from the AI text for a cleaner UI
        clean_line = line.lstrip("* ").lstrip("- ")
        console.print(f" [bold {status_color}]●[/bold {status_color}] {clean_line}")

    console.print("\n" + "—" * 60 + "\n")