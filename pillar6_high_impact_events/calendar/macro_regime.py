from collections import defaultdict
from datetime import datetime

def parse_date(d):
    return datetime.strptime(d, "%Y-%m-%d")

def build_latest_series(events):
    """
    Groups events by type and sorts by date (latest first)
    """
    grouped = defaultdict(list)

    for e in events:
        grouped[e["event"]].append(e)

    for k in grouped:
        grouped[k].sort(key=lambda x: parse_date(x["date"]), reverse=True)

    return grouped


def macro_regime_score(events):
    series = build_latest_series(events)

    score = 0
    reasons = []

    # ---- INFLATION (CPI, CORE_CPI, PCE) ----
    for key in ["CPI", "CORE_CPI", "PCE"]:
        if key in series and len(series[key]) >= 2:
            latest = series[key][0]["value"]
            prev = series[key][1]["value"]

            if latest > prev:
                score -= 1
                reasons.append(f"{key} rising (inflation pressure)")
            else:
                score += 1
                reasons.append(f"{key} falling (disinflation)")

    # ---- INTEREST RATES ----
    if "FED_FUNDS" in series and len(series["FED_FUNDS"]) >= 2:
        latest = series["FED_FUNDS"][0]["value"]
        prev = series["FED_FUNDS"][1]["value"]

        if latest > prev:
            score -= 2
            reasons.append("Rates rising (liquidity tightening)")
        else:
            score += 2
            reasons.append("Rates falling (liquidity easing)")

    # ---- LABOR ----
    if "UNEMPLOYMENT" in series and len(series["UNEMPLOYMENT"]) >= 2:
        latest = series["UNEMPLOYMENT"][0]["value"]
        prev = series["UNEMPLOYMENT"][1]["value"]

        if latest > prev:
            score -= 1
            reasons.append("Unemployment rising (growth risk)")
        else:
            score += 1
            reasons.append("Labor market stable/strong")

    # ---- FINAL REGIME ----
    if score >= 3:
        regime = "RISK_ON"
    elif score <= -3:
        regime = "RISK_OFF"
    else:
        regime = "NEUTRAL"

    return {
        "regime": regime,
        "score": score,
        "reasons": reasons
    }
