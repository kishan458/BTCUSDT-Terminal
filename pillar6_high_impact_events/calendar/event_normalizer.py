from datetime import datetime, date, timedelta

EVENT_TYPE_MAP = {
    "CPI": "INFLATION",
    "CORE_CPI": "INFLATION",
    "PCE": "INFLATION",
    "GDP": "GROWTH",
    "UNEMPLOYMENT": "LABOR",
    "FED_FUNDS": "MONETARY_POLICY",
}

def normalize_events(raw_events, lookback_days=180):
    today = date.today()
    cutoff = today - timedelta(days=lookback_days)

    normalized = []

    for e in raw_events:
        event_date = datetime.strptime(e["date"], "%Y-%m-%d").date()

        if event_date < cutoff:
            continue  # skip old data

        normalized.append({
            "event": e["event"],
            "date": e["date"],
            "value": e["value"],
            "impact": e.get("impact", "HIGH"),
            "source": e.get("source", "UNKNOWN"),
            "type": EVENT_TYPE_MAP.get(e["event"], "OTHER"),
            "timing": "UPCOMING" if event_date >= today else "RELEASED",
        })

    return normalized

