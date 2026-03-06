import json

from macro.event_ingestor import ingest
from macro.providers.bls_cpi_provider import BLSCPIProvider
from macro.providers.bls_empsit_provider import BLSEmploymentProvider
from macro.event_engine import update_event_states
from macro.pillar6_output import build_pillar6_output


if __name__ == "__main__":
    n1 = ingest(BLSCPIProvider())
    n2 = ingest(BLSEmploymentProvider())

    update_event_states()

    out = build_pillar6_output()

    print("Events ingested:", n1 + n2)
    print(json.dumps(out, indent=2))