from macro.bls_outcome_collector import build_cpi_outcomes


def build_cpi_probabilities(startyear="2024", endyear="2026"):
    outcomes = build_cpi_outcomes(startyear, endyear)

    if len(outcomes) < 5:
        return {
            "available": False,
            "historical_samples": len(outcomes),
            "probabilities": None
        }

    changes = [row["change"] for row in outcomes]

    mean_abs_change = sum(abs(x) for x in changes) / len(changes)

    upside = 0
    inline = 0
    downside = 0

    for change in changes:
        if change > mean_abs_change * 0.25:
            upside += 1
        elif change < -mean_abs_change * 0.25:
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
            "DOWN": downside / total
        }
    }