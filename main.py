import json

from pillar6_high_impact_events.event_ingestor import ingest
from pillar6_high_impact_events.fomc_ingestor import ingest_fomc_events
from pillar6_high_impact_events.providers.bls_cpi_provider import BLSCPIProvider
from pillar6_high_impact_events.providers.bls_empsit_provider import BLSEmploymentProvider
from pillar6_high_impact_events.event_engine import update_event_states
from pillar6_high_impact_events.pillar6_output import build_pillar6_output


if __name__ == "__main__":
    n1 = ingest(BLSCPIProvider())
    n2 = ingest(BLSEmploymentProvider())
    n3 = ingest_fomc_events()

    update_event_states()

    out = build_pillar6_output()

    print("Events ingested:", n1 + n2 + n3)
    print(json.dumps(out, indent=2))