from abc import ABC, abstractmethod
from typing import List, Dict

class BaseEventProvider(ABC):
    provider_name: str

    @abstractmethod
    def fetch_events(self) -> List[Dict]:
        """Return normalized events (already standardized)."""
        raise NotImplementedError