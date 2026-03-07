MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}
from pillar6_high_impact_events.providers.fomc_provider import FOMCProvider
from datetime import datetime

def fetch_fomc_meetings():
    provider = FOMCProvider()
    return provider.fetch_events()


def _meeting_to_timestamp(meeting_str):
    meeting_str = meeting_str.replace("*", "")
    month_name, days = meeting_str.split(" ")

    day2 = int(days.split("-")[1])
    month = MONTH_MAP[month_name]

    # FOMC statement release ≈ 18:00 UTC (2pm NY)
    dt = datetime(2026, month, day2, 18, 0, 0)

    return dt.strftime("%Y-%m-%d %H:%M:%S")


def build_fomc_events():
    meetings = fetch_fomc_meetings()

    events = []
    for meeting in meetings:
        scheduled_time_utc = _meeting_to_timestamp(meeting)

        events.append({
            "event_name": "FOMC Rate Decision",
            "event_type": "CENTRAL_BANK",
            "country": "US",
            "scheduled_time_utc": scheduled_time_utc,
            "importance": "HIGH",
            "raw_json": {"meeting": meeting}
        })

    return events
