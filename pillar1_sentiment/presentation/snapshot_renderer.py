from datetime import datetime

def render_institutional_snapshot(snapshot: dict):
    print("\n" + "─" * 48)
    print("INSTITUTIONAL BTC SNAPSHOT")
    print("Generated :", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    print("Articles  :", len(snapshot["articles"]))
    print("─" * 48)

    if not snapshot["articles"]:
        print("No institutional articles found.\n")
        return

    for i, a in enumerate(snapshot["articles"], start=1):
        analysis = a.get("analysis", {})

        print(f"\n[{i}] {a.get('headline', 'N/A')}")
        print(f"    Source    : {a.get('source', 'N/A')}")
        print(f"    Published : {a.get('published_at', 'N/A')}")
        print(f"    Sentiment : {analysis.get('sentiment', 'N/A')} "
              f"({analysis.get('confidence', 0.0)})")
        print(f"    Tone      : {analysis.get('tone', 'N/A')}")

    print("\n" + "─" * 48 + "\n")
