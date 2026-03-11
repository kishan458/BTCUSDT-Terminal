from pillar5_regime_cycle_engine.pillar5_output import build_pillar5_output
import json

out = build_pillar5_output()

print(json.dumps(out, indent=2))
