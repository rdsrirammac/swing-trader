"""SEC EDGAR client — free, no API key. Used for insider transactions (DC-002)
and as a fallback source for short-interest-adjacent filings.
"""
from __future__ import annotations

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from swing_trader.logging_setup import get_logger

logger = get_logger("data.sec_edgar")

HEADERS = {"User-Agent": "swing-trader personal-research contact@example.com"}
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


class SecEdgarClient:
    def __init__(self):
        self._ticker_to_cik: dict[str, int] | None = None

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
    def _get_json(self, url: str) -> dict:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _load_ticker_map(self) -> dict[str, int]:
        if self._ticker_to_cik is None:
            try:
                data = self._get_json(TICKER_MAP_URL)
                self._ticker_to_cik = {
                    row["ticker"].upper(): row["cik_str"] for row in data.values()
                }
            except Exception as e:
                logger.warning("Failed to load SEC ticker->CIK map: %s", e)
                self._ticker_to_cik = {}
        return self._ticker_to_cik

    def get_insider_transactions(self, ticker: str, limit: int = 20) -> list[dict]:
        """Best-effort insider transaction summary via EDGAR full-text search.

        SEC's raw Form 4 XML parsing is out of scope for a first cut; this
        returns recent filing metadata (form type + date) which is enough
        to drive an "insider activity spike" heuristic feature. Full XBRL
        parsing is tracked as a backlog enhancement (see ROADMAP.md).
        """
        cik_map = self._load_ticker_map()
        cik = cik_map.get(ticker.upper())
        if cik is None:
            logger.info("No CIK found for %s; skipping insider transactions", ticker)
            return []

        try:
            data = self._get_json(SUBMISSIONS_URL.format(cik=cik))
        except Exception as e:
            logger.warning("SEC EDGAR submissions fetch failed for %s: %s", ticker, e)
            return []

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accession = recent.get("accessionNumber", [])

        results = []
        for form, date, acc in zip(forms, dates, accession):
            if form in ("4", "4/A"):  # Form 4 = insider transaction
                results.append({"form": form, "filing_date": date, "accession_number": acc})
            if len(results) >= limit:
                break
        return results
