import requests

NFP_SERIES_ID = "CES0000000001"
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


def fetch_nfp_series(startyear: str, endyear: str):
    payload = {
        "seriesid": [NFP_SERIES_ID],
        "startyear": startyear,
        "endyear": endyear
    }

    response = requests.post(BLS_API_URL, json=payload, timeout=30)
    response.raise_for_status()

    data = response.json()

    if data.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS request failed: {data}")

    return data["Results"]["series"][0]["data"]


def build_nfp_outcomes(startyear: str, endyear: str):
    raw = fetch_nfp_series(startyear, endyear)

    points = []
    for item in raw:
        period = item.get("period", "")
        value = item.get("value", "")

        if not period.startswith("M"):
            continue
        if value in ("-", None, ""):
            continue

        year = int(item["year"])
        month = int(period[1:])

        points.append({
            "year": year,
            "month": month,
            "value": float(value)
        })

    points.sort(key=lambda x: (x["year"], x["month"]))

    outcomes = []
    for i in range(1, len(points)):
        previous = points[i - 1]["value"]
        actual = points[i]["value"]

        outcomes.append({
            "event_name": "US Employment Situation (NFP)",
            "release_year": points[i]["year"],
            "release_month": points[i]["month"],
            "actual": actual,
            "previous": previous,
            "change": actual - previous
        })

    return outcomes