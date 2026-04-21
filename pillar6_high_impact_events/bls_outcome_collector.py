import sqlite3

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DB_PATH = "database/btc_terminal.db"
BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


def _build_bls_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["POST"]),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=5, pool_maxsize=5)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    )

    return session


def fetch_bls_series(series_id: str, startyear: str, endyear: str):
    payload = {
        "seriesid": [series_id],
        "startyear": str(startyear),
        "endyear": str(endyear),
    }

    with _build_bls_session() as session:
        response = session.post(BLS_URL, json=payload, timeout=(10, 45))
        response.raise_for_status()

        try:
            data = response.json()
        except ValueError as e:
            raise RuntimeError(f"BLS returned non-JSON response for {series_id}") from e

    if data.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS request failed for {series_id}: {data}")

    results = data.get("Results", {})
    series_list = results.get("series", [])

    if not series_list:
        raise RuntimeError(f"BLS returned no series data for {series_id}")

    series = series_list[0].get("data", [])
    if not isinstance(series, list):
        raise RuntimeError(f"BLS returned malformed series data for {series_id}")

    return series


def normalize_bls_points(series):
    rows = []

    for item in series:
        period = str(item.get("period", "")).strip()
        value = item.get("value", None)

        if not period.startswith("M"):
            continue
        if value in {"-", "", None}:
            continue

        try:
            year = int(item["year"])
            month = int(period[1:])
            value_num = float(value)
        except (KeyError, TypeError, ValueError):
            continue

        if month < 1 or month > 12:
            continue

        rows.append(
            {
                "year": year,
                "month": month,
                "period_name": item.get("periodName"),
                "value": value_num,
            }
        )

    rows.sort(key=lambda x: (x["year"], x["month"]))
    return rows


def build_cpi_outcomes(startyear="2024", endyear="2026"):
    raw = fetch_bls_series("CUUR0000SA0", startyear, endyear)
    points = normalize_bls_points(raw)

    outcomes = []
    for i in range(1, len(points)):
        prev = points[i - 1]["value"]
        actual = points[i]["value"]

        outcomes.append(
            {
                "event_name": "US CPI",
                "release_year": points[i]["year"],
                "release_month": points[i]["month"],
                "actual": actual,
                "previous": prev,
                "change": actual - prev,
            }
        )

    return outcomes


def save_cpi_outcomes_to_db(outcomes):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    updates = 0

    for row in outcomes:
        year = row["release_year"]
        month = row["release_month"]

        cur.execute(
            """
            UPDATE macro_events
            SET actual = ?,
                previous = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE event_name = 'US CPI'
              AND strftime('%Y', scheduled_time_utc) = ?
              AND strftime('%m', scheduled_time_utc) = ?
            """,
            (
                row["actual"],
                row["previous"],
                str(year),
                f"{month:02d}",
            ),
        )

        updates += cur.rowcount

    conn.commit()
    conn.close()

    return updates