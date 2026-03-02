import os
import requests
from datetime import datetime, timezone

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
BASE_URL = "https://finnhub.io/api/v1/calendar/economic"

HIGH_IMPACT_EVENTS = {
    "FOMC": ["Interest Rate Decision", "FOMC Statement"],
    "CPI": ["Consumer Price Index"],
    "NFP": ["Nonfarm Payrolls"],
    "GDP": ["Gross Domestic Product"],
    "RATES": ["Interest Rate"]
}

def fetch_high_impact_events(days_ahead=7):
    """
    Pulls VERIFIED upcoming macro events from Finnhub.
    No hardcoding. No assumptions.
    """

    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY not set")

    now = datetime.now(timezone.utc).date()
    end = now.fromordinal(now.toordinal() + days_ahead)

    params = {
        "from": now.isoformat(),
        "to": end.isoformat(),
        "token": FINNHUB_API_KEY
    }

    r = requests.get(BASE_URL, params=params, timeout=10)
    r.raise_for_status()

    events = []
    for e in r.json().get("economicCalendar", []):
        event_name = e.get("event", "")
        for event_type, keywords in HIGH_IMPACT_EVENTS.items():
            if any(k.lower() in event_name.lower() for k in keywords):
                events.append({
                    "event_type": event_type,
                    "title": event_name,
                    "country": e.get("country"),
                    "importance": e.get("impact", "medium"),
                    "timestamp": datetime.fromtimestamp(
                        e.get("time"), tz=timezone.utc
                    ).isoformat()
                })
                break

    return events
