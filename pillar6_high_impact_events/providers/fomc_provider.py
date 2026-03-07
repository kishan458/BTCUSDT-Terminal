import re
import requests
from bs4 import BeautifulSoup
from .base_provider import BaseEventProvider

FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"


class FOMCProvider(BaseEventProvider):
    provider_name = "fed_fomc"

    def _fetch_html(self):
        response = requests.get(FOMC_CALENDAR_URL, timeout=30)
        response.raise_for_status()
        return response.text

    def _parse_meetings(self, html):
        soup = BeautifulSoup(html, "html.parser")

        block_text = None
        panels = soup.find_all("div", class_="panel panel-default")

        for panel in panels:
            text = panel.get_text(" ", strip=True)
            if "2026 FOMC Meetings" in text:
                block_text = text
                break

        if not block_text:
            return []

        # Extract meeting date ranges only
        pattern = r"(January \d{1,2}-\d{1,2}\*?|March \d{1,2}-\d{1,2}\*?|April \d{1,2}-\d{1,2}\*?|June \d{1,2}-\d{1,2}\*?|July \d{1,2}-\d{1,2}\*?|September \d{1,2}-\d{1,2}\*?|October \d{1,2}-\d{1,2}\*?|December \d{1,2}-\d{1,2}\*?)"

        meetings = re.findall(pattern, block_text)

        return meetings

    def fetch_events(self):
        html = self._fetch_html()
        meetings = self._parse_meetings(html)
        return meetings