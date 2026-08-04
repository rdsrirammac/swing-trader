"""NewsAPI client — secondary source used only for 90-day historical news
backfill (yfinance .news has no history). Free tier: 100 requests/day.
"""
from __future__ import annotations

import datetime as dt

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from swing_trader.config import get_settings
from swing_trader.logging_setup import get_logger

logger = get_logger("data.newsapi")

BASE_URL = "https://newsapi.org/v2/everything"


class NewsApiClient:
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.secret("newsapi.api_key")
        self.enabled = bool(self.api_key) and "YOUR_NEWSAPI_KEY" not in (self.api_key or "")

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
    def _get(self, params: dict) -> dict:
        params = {**params, "apiKey": self.api_key}
        resp = requests.get(BASE_URL, params=params, timeout=15)
        if resp.status_code == 429:
            raise RuntimeError("NewsAPI rate limit hit")
        resp.raise_for_status()
        return resp.json()

    def get_historical_news(self, ticker: str, company_name: str | None = None, days: int = 90) -> list[dict]:
        """Fetch up to `days` of historical news for a ticker.

        Gracefully degrades (returns []) if no API key configured — the
        SRS refinement explicitly allows starting with empty news sentiment
        and building forward from day 1 (Section 7 of the refinement doc).
        """
        if not self.enabled:
            logger.info("NewsAPI not configured; skipping historical news backfill for %s", ticker)
            return []

        query = company_name or ticker
        from_date = (dt.datetime.utcnow() - dt.timedelta(days=days)).date().isoformat()
        try:
            data = self._get(
                {
                    "q": query,
                    "from": from_date,
                    "sortBy": "publishedAt",
                    "language": "en",
                    "pageSize": 100,
                }
            )
        except Exception as e:
            logger.warning("NewsAPI request failed for %s: %s", ticker, e)
            return []

        articles = data.get("articles", [])
        return [
            {
                "headline": a.get("title"),
                "publisher": (a.get("source") or {}).get("name"),
                "url": a.get("url"),
                "published_at": a.get("publishedAt"),
                "source": "newsapi",
            }
            for a in articles
        ]
