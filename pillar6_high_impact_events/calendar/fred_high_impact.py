from datetime import datetime
from pillar6_high_impact_events.calendar.fred_client import fetch_series

HIGH_IMPACT_SERIES = {
    "CPI": "CPIAUCSL",
    "CORE_CPI": "CPILFESL",
    "PCE": "PCEPI",
    "GDP": "GDP",
    "UNEMPLOYMENT": "UNRATE",
    "FED_FUNDS": "FEDFUNDS",
}

def fetch_high_impact_events(limit=6):
    events = []

    for name, series_id in HIGH_IMPACT_SERIES.items():
        data = fetch_series(series_id, limit=limit)

        for row in data:
            if row["value"] == ".":
                continue

            events.append({
                "event": name,
                "date": row["date"],
                "value": float(row["value"]),
                "source": "FRED",
                "impact": "HIGH"
            })

    return events
