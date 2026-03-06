import json
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
from io import StringIO

import pandas as pd
import requests

from .base_provider import BaseEventProvider

BLS_CPI_URL = "https://www.bls.gov/schedule/news_release/cpi.htm"
NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")


def _event_uid(provider: str, event_name: str, country: str, scheduled_time_utc: str) -> str:
    s = f"{provider}|{event_name}|{country}|{scheduled_time_utc}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _clean_time_str(t: str) -> str:
    """
    Normalize time strings like:
    "08:30 AM", "8:30 a.m.", "8:30 AM ET"
    to "08:30 AM"
    """
    t = str(t).strip()

    # remove timezone words
    for junk in ["ET", "E.T.", "EST", "EDT"]:
        t = t.replace(junk, "").strip()

    t = t.replace("a.m.", "AM").replace("p.m.", "PM")
    t = t.replace("A.M.", "AM").replace("P.M.", "PM")
    t = t.replace(".", "")
    t = " ".join(t.split())  # collapse whitespace
    t = t.upper()

    # If it looks like "8:30AM" -> "8:30 AM"
    if ("AM" in t or "PM" in t) and " " not in t:
        t = t.replace("AM", " AM").replace("PM", " PM")

    return t


def _parse_date(date_str: str) -> datetime:
    """
    Accepts multiple BLS date formats:
    - "Mar. 11, 2026"
    - "Mar 11, 2026"
    - "May 12, 2026"
    - "March 11, 2026"
    """
    ds = str(date_str).strip()
    ds = " ".join(ds.split())

    # Try formats from most common to broadest
    fmts = [
        "%b. %d, %Y",   # Mar. 11, 2026
        "%b %d, %Y",    # Mar 11, 2026
        "%B %d, %Y",    # March 11, 2026
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(ds, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date format: {ds!r}")


def _to_utc_str(date_str: str, time_str: str) -> str:
    """
    Convert BLS (ET) date+time -> UTC string "YYYY-MM-DD HH:MM:SS"
    Skips rows that are not parseable (TBA / Tentative).
    """
    ds = str(date_str).strip()
    ts = _clean_time_str(time_str)

    # Some rows can be "TBA", "Tentative", empty etc.
    if not ds or not ts or ts in {"TBA", "TBD", "TENTATIVE"}:
        raise ValueError(f"Unusable date/time: date={ds!r}, time={ts!r}")

    date_part = _parse_date(ds)

    # parse time part
    try:
        time_part = datetime.strptime(ts, "%I:%M %p")
    except ValueError:
        # sometimes it can be "8:30 AM" (still matches) or weird — fail safely
        raise ValueError(f"Unrecognized time format: {ts!r}")

    dt_local = datetime(
        year=date_part.year,
        month=date_part.month,
        day=date_part.day,
        hour=time_part.hour,
        minute=time_part.minute,
        second=0,
        tzinfo=NY_TZ,
    )
    dt_utc = dt_local.astimezone(UTC_TZ)
    return dt_utc.strftime("%Y-%m-%d %H:%M:%S")


def _fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.bls.gov/",
        "Connection": "keep-alive",
    }
    with requests.Session() as s:
        r = s.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.text


class BLSCPIProvider(BaseEventProvider):
    provider_name = "bls_cpi"

    def fetch_events(self):
        html = _fetch_html(BLS_CPI_URL)
        tables = pd.read_html(StringIO(html))

        schedule = None
        for t in tables:
            cols = [str(c).strip() for c in t.columns]
            if "Reference Month" in cols and "Release Date" in cols and "Release Time" in cols:
                schedule = t
                break

        if schedule is None:
            raise RuntimeError("Could not find CPI schedule table on BLS page.")

        out = []
        skipped = 0

        for _, row in schedule.iterrows():
            ref_month = str(row.get("Reference Month", "")).strip()
            release_date = str(row.get("Release Date", "")).strip()
            release_time = str(row.get("Release Time", "")).strip()

            try:
                scheduled_time_utc = _to_utc_str(release_date, release_time)
            except Exception:
                skipped += 1
                continue

            event_name = "US CPI"
            country = "US"
            importance = "HIGH"

            raw = {
                "source_url": BLS_CPI_URL,
                "reference_month": ref_month,
                "release_date_et": release_date,
                "release_time_et": release_time,
            }

            out.append({
                "event_uid": _event_uid(self.provider_name, event_name, country, scheduled_time_utc),
                "provider": self.provider_name,
                "provider_event_id": None,
                "event_name": event_name,
                "event_type": "INFLATION",
                "country": country,
                "scheduled_time_utc": scheduled_time_utc,
                "importance": importance,
                "actual": None,
                "forecast": None,
                "previous": None,
                "raw_json": json.dumps(raw),
            })

        # optional debug
        # print(f"[BLSCPIProvider] parsed={len(out)} skipped={skipped}")

        return out