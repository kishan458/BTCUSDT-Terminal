import requests
import sqlite3
from datetime import datetime

DB_PATH = "database/btc_terminal.db"
BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


def fetch_bls_series(series_id: str, startyear: str, endyear: str):
    payload = {
        "seriesid": [series_id],
        "startyear": startyear,
        "endyear": endyear
    }

    response = requests.post(BLS_URL, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS request failed: {data}")

    series = data["Results"]["series"][0]["data"]
    return series


def normalize_bls_points(series):
    rows = []

    for item in series:
        period = item.get("period", "")
        value = item.get("value", "")

        if not period.startswith("M"):
            continue
        if value == "-" or value is None:
            continue

        year = int(item["year"])
        month = int(period[1:])

        rows.append({
            "year": year,
            "month": month,
            "period_name": item.get("periodName"),
            "value": float(value)
        })

    rows.sort(key=lambda x: (x["year"], x["month"]))
    return rows


def build_cpi_outcomes(startyear="2024", endyear="2026"):
    raw = fetch_bls_series("CUUR0000SA0", startyear, endyear)
    points = normalize_bls_points(raw)

    outcomes = []
    for i in range(1, len(points)):
        prev = points[i - 1]["value"]
        actual = points[i]["value"]

        outcomes.append({
            "event_name": "US CPI",
            "release_year": points[i]["year"],
            "release_month": points[i]["month"],
            "actual": actual,
            "previous": prev,
            "change": actual - prev
        })

    return outcomes


def save_cpi_outcomes_to_db(outcomes):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    updates = 0

    for row in outcomes:
        year = row["release_year"]
        month = row["release_month"]

        # match monthly CPI event rows already stored in macro_events
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
                f"{month:02d}"
            )
        )

        updates += cur.rowcount

    conn.commit()
    conn.close()

    return updates