import json
from datetime import datetime
from pathlib import Path

def save_snapshot_to_json(snapshot: dict, folder="data/snapshots"):
    Path(folder).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{folder}/institutional_snapshot_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(snapshot, f, indent=2)

    return filename
