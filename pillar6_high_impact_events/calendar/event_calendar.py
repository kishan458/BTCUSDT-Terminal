import os
import requests
from datetime import datetime, timezone


FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
BASE_URL = "https://finnhub.io/api/v1/calendar/economic"


HIGH_IMPACT_EVENTS = {
    "FOMC": ["FOMC", "Fed", "Federal Reserve"],
    "CPI": ["CPI", "Consumer Price Index"],
    "PPI": ["PPI", "Producer Price Index"],
    "NFP": ["Nonfarm", "Employment", "Jobs"],
    "GDP": ["GDP", "Gross Domestic Product"],
}


def fetch_economic_calendar(from_date: str, to_date: str):
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY not set")

    params = {
        "from": from_date,
        "to": to_date,
        "token": FINNHUB_API_KEY,
    }

    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()

    return response.json().get("economicCalendar", [])


def classify_event(event_name: str):
    for event_type, keywords in HIGH_IMPACT_EVENTS.items():
        for kw in keywords:
            if kw.lower() in event_name.lower():
                return event_type
    return None


def get_high_impact_events(from_date: str, to_date: str):
    raw_events = fetch_economic_calendar(from_date, to_date)
    now = datetime.now(timezone.utc)

    events = []

    for e in raw_events:
        event_type = classify_event(e.get("event", ""))
        if not event_type:
            continue

        event_time = datetime.fromisoformat(e["time"].replace("Z", "+00:00"))

        if event_time > now:
            status = "upcoming"
        elif abs((now - event_time).total_seconds()) < 3600:
            status = "ongoing"
        else:
            status = "finished"

        events.append({
            "event_type": event_type,
            "event_name": e["event"],
            "country": e.get("country"),
            "importance": e.get("impact", "high"),
            "time_utc": event_time.isoformat(),
            "status": status,
            "minutes_to_event": (
                int((event_time - now).total_seconds() // 60)
                if event_time > now else 0
            )
        })

    return sorted(events, key=lambda x: x["time_utc"])
