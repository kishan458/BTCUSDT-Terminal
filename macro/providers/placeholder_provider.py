from .base_provider import BaseEventProvider

class PlaceholderProvider(BaseEventProvider):

    def fetch_events(self, start: str, end: str) -> list[dict]:
        # stub — returns empty list
        return []

    def normalize_event(self, raw_event: dict) -> dict:
        # This will never be called for placeholder
        return {}