import re
import json
import hashlib
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup

from .base_provider import BaseEventProvider

FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

NY_TZ  = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# FOMC decisions are always announced at 2:00 PM ET
FOMC_ANNOUNCEMENT_HOUR   = 14
FOMC_ANNOUNCEMENT_MINUTE = 0


def _event_uid(provider: str, event_name: str, country: str, scheduled_time_utc: str) -> str:
    s = f"{provider}|{event_name}|{country}|{scheduled_time_utc}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _parse_fomc_date_range(month_str: str, date_range: str, year: int) -> datetime | None:
    """
    Parse FOMC date ranges like:
      "January 28-29"  → decision day = Jan 29
      "March 18-19"    → decision day = Mar 19
      "April 29-30"    → decision day = Apr 30
      "June 17-18"     → decision day = Jun 18

    The SECOND day is always the decision day (when the rate decision is announced).
    Returns a naive datetime at 14:00 ET on the decision day.
    """
    month_str = month_str.strip().rstrip("*")
    date_range = date_range.strip().rstrip("*")

    # Extract end day from range like "28-29"
    match = re.search(r"(\d{1,2})-(\d{1,2})", date_range)
    if not match:
        # Single day meeting
        match2 = re.search(r"(\d{1,2})", date_range)
        if not match2:
            return None
        day = int(match2.group(1))
    else:
        day = int(match.group(2))  # ← second day = decision day

    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }

    month_num = month_map.get(month_str.lower())
    if not month_num:
        return None

    try:
        return datetime(year, month_num, day,
                        FOMC_ANNOUNCEMENT_HOUR, FOMC_ANNOUNCEMENT_MINUTE,
                        tzinfo=NY_TZ)
    except ValueError:
        return None


def _to_utc_str(dt_et: datetime) -> str:
    return dt_et.astimezone(UTC_TZ).strftime("%Y-%m-%d %H:%M:%S")


class FOMCProvider(BaseEventProvider):
    provider_name = "fed_fomc"

    def _fetch_html(self) -> str:
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            response = requests.get(FOMC_CALENDAR_URL, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"[FOMCProvider ERROR] Failed to fetch FOMC calendar: {e}")
            return ""

    def _parse_meetings(self, html: str) -> list[dict]:
        """
        Parses the Fed calendar page and returns structured meeting dicts.
        Looks for the current year's FOMC meetings panel.
        """
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        current_year = datetime.now().year
        results = []

        # Try current year and next year (page often shows both)
        for year in [current_year, current_year + 1]:
            panels = soup.find_all("div", class_="panel panel-default")

            for panel in panels:
                text = panel.get_text(" ", strip=True)
                if f"{year} FOMC Meetings" not in text:
                    continue

                # Match patterns like "January 28-29" or "March 18-19*"
                pattern = (
                    r"(January|February|March|April|May|June|July|August|"
                    r"September|October|November|December)\s+(\d{1,2}-\d{1,2}\*?)"
                )
                matches = re.findall(pattern, text)

                for month_str, date_range in matches:
                    dt_et = _parse_fomc_date_range(month_str, date_range, year)
                    if dt_et is None:
                        continue

                    # Skip meetings in the past
                    now_utc = datetime.now(tz=UTC_TZ)
                    dt_utc = dt_et.astimezone(UTC_TZ)
                    if dt_utc < now_utc:
                        continue

                    scheduled_time_utc = _to_utc_str(dt_et)

                    raw = {
                        "source_url": FOMC_CALENDAR_URL,
                        "year": year,
                        "month": month_str,
                        "date_range": date_range,
                        "announcement_time_et": f"{FOMC_ANNOUNCEMENT_HOUR:02d}:{FOMC_ANNOUNCEMENT_MINUTE:02d} ET",
                    }

                    results.append({
                        "event_uid": _event_uid(
                            self.provider_name,
                            "FOMC Rate Decision",
                            "US",
                            scheduled_time_utc,
                        ),
                        "provider": self.provider_name,
                        "provider_event_id": None,
                        "event_name": "FOMC Rate Decision",
                        "event_type": "MONETARY_POLICY",
                        "country": "US",
                        "scheduled_time_utc": scheduled_time_utc,
                        "importance": "HIGH",
                        "actual": None,
                        "forecast": None,
                        "previous": None,
                        "raw_json": json.dumps(raw),
                    })

        print(f"[FOMCProvider] Found {len(results)} upcoming FOMC meetings.")
        return results

    def fetch_events(self) -> list[dict]:
        html = self._fetch_html()
        return self._parse_meetings(html)