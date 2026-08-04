"""Free, no-key economic calendar via RSS/iCal feeds (EC-003).

Federal Reserve and BLS both publish public RSS/iCal feeds for FOMC meeting
dates and economic releases. This client parses those feeds into
EconomicEvent rows. Feed URLs are configurable since government feed URLs
occasionally move.
"""
from __future__ import annotations

import datetime as dt

import feedparser

from swing_trader.logging_setup import get_logger

logger = get_logger("data.economic_calendar")

DEFAULT_FEEDS = {
    "FOMC": "https://www.federalreserve.gov/feeds/press_all.xml",
    "BLS": "https://www.bls.gov/feed/news_release.rss",
}

# High-impact keywords we care about for swing trading (CPI, PPI, NFP, GDP, FOMC)
KEYWORDS = ["FOMC", "CPI", "Consumer Price Index", "Employment Situation",
            "Nonfarm", "GDP", "Gross Domestic Product", "PPI", "Producer Price Index"]


class EconomicCalendarClient:
    def __init__(self, feeds: dict[str, str] | None = None):
        self.feeds = feeds or DEFAULT_FEEDS

    def fetch_upcoming_events(self, lookback_days: int = 7) -> list[dict]:
        """Parse configured RSS feeds and return events matching KEYWORDS.

        Best-effort: government feed formats vary and are not guaranteed
        machine-parseable for exact release dates. This surfaces headline
        + published date; exact release-time precision (EC-003's
        "pre-market vs post-market timing") should be cross-checked against
        the BLS/Fed release calendar pages, tracked as a backlog enhancement.
        """
        events: list[dict] = []
        cutoff = dt.datetime.utcnow() - dt.timedelta(days=lookback_days)

        for source, url in self.feeds.items():
            try:
                parsed = feedparser.parse(url)
            except Exception as e:
                logger.warning("Failed to parse feed %s (%s): %s", source, url, e)
                continue

            for entry in getattr(parsed, "entries", []):
                title = entry.get("title", "")
                if not any(kw.lower() in title.lower() for kw in KEYWORDS):
                    continue
                published = entry.get("published_parsed")
                published_dt = (
                    dt.datetime(*published[:6]) if published else dt.datetime.utcnow()
                )
                if published_dt < cutoff:
                    continue
                events.append(
                    {
                        "event_name": _classify(title),
                        "event_date": published_dt,
                        "timing": None,
                        "historical_reaction_notes": title,
                        "source": source.lower(),
                    }
                )
        return events


def _classify(title: str) -> str:
    title_lower = title.lower()
    if "fomc" in title_lower or "federal open market" in title_lower:
        return "FOMC"
    if "cpi" in title_lower or "consumer price index" in title_lower:
        return "CPI"
    if "ppi" in title_lower or "producer price index" in title_lower:
        return "PPI"
    if "nonfarm" in title_lower or "employment situation" in title_lower:
        return "NFP"
    if "gdp" in title_lower or "gross domestic product" in title_lower:
        return "GDP"
    return "OTHER"
