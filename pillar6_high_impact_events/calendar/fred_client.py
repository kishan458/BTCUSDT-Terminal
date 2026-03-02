import os
import requests

class FredError(Exception):
    pass

FRED_API_KEY = os.getenv("FRED_API_KEY")
BASE_URL = "https://api.stlouisfed.org/fred"

if not FRED_API_KEY:
    raise FredError("FRED_API_KEY not set")

def fetch_series(series_id, limit=12):
    url = f"{BASE_URL}/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit
    }

    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        raise FredError(f"HTTP {r.status_code}")

    data = r.json()
    return data["observations"]

