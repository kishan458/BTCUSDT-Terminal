from pillar6_high_impact_events.bls_outcome_collector import build_cpi_outcomes


def build_cpi_probabilities(startyear="2024", endyear="2026"):
    outcomes = build_cpi_outcomes(startyear, endyear)

    if not outcomes or len(outcomes) < 5:
        return {
            "available": False,
            "historical_samples": len(outcomes) if outcomes else 0,
            "probabilities": None,
        }

    changes = []
    for row in outcomes:
        change = row.get("change")
        if isinstance(change, (int, float)):
            changes.append(float(change))

    if len(changes) < 5:
        return {
            "available": False,
            "historical_samples": len(changes),
            "probabilities": None,
        }

    mean_abs_change = sum(abs(x) for x in changes) / len(changes)

    if mean_abs_change <= 0:
        return {
            "available": False,
            "historical_samples": len(changes),
            "probabilities": None,
        }

    threshold = mean_abs_change * 0.25

    upside = 0
    inline = 0
    downside = 0

    for change in changes:
        if change > threshold:
            upside += 1
        elif change < -threshold:
            downside += 1
        else:
            inline += 1

    total = len(changes)

    return {
        "available": True,
        "historical_samples": total,
        "probabilities": {
            "UP": upside / total,
            "INLINE": inline / total,
            "DOWN": downside / total,
        },
    }