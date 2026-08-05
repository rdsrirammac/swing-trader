"""Thread-safe yfinance wrapper with disk caching, retry, and rate limiting.

Implements the design in SRS_Refinement_v1.1_yfinance.md Section 5, extended
with real retry/backoff (tenacity) and structured logging. Singleton per
process — one client, one rate limiter, one cache.
"""
from __future__ import annotations

import threading
import time
from collections import namedtuple
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from swing_trader.config import get_settings
from swing_trader.logging_setup import get_logger

logger = get_logger("data.yf_client")

try:
    import diskcache as dc
except ImportError:  # pragma: no cover
    dc = None


class YFinanceRateLimitError(Exception):
    pass


# yfinance's `Ticker.option_chain()` returns an `Options` namedtuple whose
# class isn't importable/picklable by diskcache ("Can't pickle
# yfinance.ticker.Options"). Callers here only ever touch `.calls`/`.puts`
# (both plain, picklable DataFrames), so we cache this equivalent, picklable
# stand-in instead of the raw yfinance object.
OptionsChain = namedtuple("OptionsChain", ["calls", "puts"])


class YFinanceClient:
    """Singleton yfinance wrapper. Use `YFinanceClient.instance()`."""

    _instance: "YFinanceClient | None" = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
        return cls._instance

    @classmethod
    def instance(cls) -> "YFinanceClient":
        return cls()

    def _init(self) -> None:
        settings = get_settings()
        cache_dir = Path(settings.get("cache.disk_cache_dir", "~/.swing-trader/cache/yfinance"))
        cache_dir = cache_dir.expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache = dc.Cache(str(cache_dir)) if dc else {}
        self._call_lock = threading.Lock()
        self.last_call = 0.0
        self.min_interval = 0.5  # ~120 calls/min, well under the ~2000/hr unofficial cap
        self.max_retries = 3

    # -- rate limiting -----------------------------------------------------
    def _rate_limit(self) -> None:
        with self._call_lock:
            elapsed = time.time() - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.time()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(Exception),
    )
    def _with_retry(self, fn, *args, **kwargs) -> Any:
        self._rate_limit()
        return fn(*args, **kwargs)

    def _cache_get(self, key: str):
        try:
            return self.cache[key] if key in self.cache else None
        except Exception:
            return None

    def _cache_set(self, key: str, value: Any, ttl: int) -> None:
        try:
            if dc:
                self.cache.set(key, value, expire=ttl)
            else:
                self.cache[key] = value
        except Exception as e:  # pragma: no cover
            logger.warning("cache write failed for %s: %s", key, e)

    @staticmethod
    def _is_empty(value: Any) -> bool:
        """True if `value` looks like a failed/empty response that should
        NOT be cached. yfinance occasionally returns an empty DataFrame,
        empty dict, or empty list on a transient hiccup *without* raising an
        exception -- if we cache that, every subsequent call (including
        retries) just re-reads the poisoned cache entry instead of hitting
        yfinance again, silently "confirming" a false failure for the full
        TTL (up to 24h). Only genuinely non-empty results get cached.
        """
        if value is None:
            return True
        if isinstance(value, pd.DataFrame):
            return value.empty
        if isinstance(value, (dict, list, tuple, str)):
            return len(value) == 0
        return False

    # -- price data (DC-001, TB-001 Phase 1) --------------------------------
    def get_history(
        self, ticker: str, period: str = "1y", interval: str = "1d", prepost: bool = False
    ) -> pd.DataFrame:
        cache_key = f"hist:{ticker}:{period}:{interval}:{prepost}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        t = yf.Ticker(ticker)
        df = self._with_retry(t.history, period=period, interval=interval, prepost=prepost)
        ttl = 3600 if interval in ("1m", "2m", "5m", "15m", "30m", "60m") else 86400
        if not self._is_empty(df):
            self._cache_set(cache_key, df, ttl)
        else:
            logger.warning("get_history(%s, %s, %s) returned empty; not caching", ticker, period, interval)
        return df

    def get_batch_history(
        self, tickers: list[str], period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        cache_key = f"batch:{','.join(sorted(tickers))}:{period}:{interval}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        data = self._with_retry(
            yf.download,
            tickers=" ".join(tickers),
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            prepost=False,
        )
        if not self._is_empty(data):
            self._cache_set(cache_key, data, 86400)
        else:
            logger.warning("get_batch_history(%s) returned empty; not caching", tickers)
        return data

    # -- fundamentals (DC-002, TB-001 Phase 2) ------------------------------
    def get_info(self, ticker: str) -> dict:
        cache_key = f"info:{ticker}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        t = yf.Ticker(ticker)
        info = self._with_retry(lambda: t.info)
        if not self._is_empty(info):
            self._cache_set(cache_key, info, 86400)
        else:
            logger.warning("get_info(%s) returned empty; not caching", ticker)
        return info

    def get_quarterly_financials(self, ticker: str) -> pd.DataFrame:
        t = yf.Ticker(ticker)
        return self._with_retry(lambda: t.quarterly_financials)

    def get_quarterly_balance_sheet(self, ticker: str) -> pd.DataFrame:
        t = yf.Ticker(ticker)
        return self._with_retry(lambda: t.quarterly_balance_sheet)

    def get_quarterly_earnings(self, ticker: str) -> pd.DataFrame:
        t = yf.Ticker(ticker)
        try:
            return self._with_retry(lambda: t.quarterly_earnings)
        except Exception:
            # Newer yfinance deprecates .quarterly_earnings in favor of .earnings_dates
            return self._with_retry(lambda: t.earnings_dates)

    def get_calendar(self, ticker: str) -> dict:
        t = yf.Ticker(ticker)
        return self._with_retry(lambda: t.calendar)

    # -- analyst data (TB-001 Phase 3) --------------------------------------
    def get_recommendations(self, ticker: str) -> pd.DataFrame:
        cache_key = f"rec:{ticker}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        t = yf.Ticker(ticker)
        rec = self._with_retry(lambda: t.recommendations)
        if not self._is_empty(rec):
            self._cache_set(cache_key, rec, 86400)
        else:
            logger.warning("get_recommendations(%s) returned empty; not caching", ticker)
        return rec

    def get_upgrades_downgrades(self, ticker: str) -> pd.DataFrame:
        t = yf.Ticker(ticker)
        return self._with_retry(lambda: t.upgrades_downgrades)

    # -- options (DC-004, TB-001 Phase 4) ------------------------------------
    def get_option_expirations(self, ticker: str) -> tuple[str, ...]:
        t = yf.Ticker(ticker)
        return self._with_retry(lambda: t.options)

    def get_options_chain(self, ticker: str, expiration: str):
        cache_key = f"opts:{ticker}:{expiration}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        t = yf.Ticker(ticker)
        chain = self._with_retry(t.option_chain, expiration)
        calls = getattr(chain, "calls", None)
        puts = getattr(chain, "puts", None)
        if (calls is None or calls.empty) and (puts is None or puts.empty):
            logger.warning("get_options_chain(%s, %s) returned empty; not caching", ticker, expiration)
        else:
            self._cache_set(cache_key, OptionsChain(calls=calls, puts=puts), 3600)
        return chain

    # -- news (DC-003, TB-001 Phase 5) ---------------------------------------
    def get_news(self, ticker: str) -> list[dict]:
        cache_key = f"news:{ticker}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        t = yf.Ticker(ticker)
        news = self._with_retry(lambda: t.news)
        if not self._is_empty(news):
            self._cache_set(cache_key, news, 900)
        else:
            logger.warning("get_news(%s) returned empty; not caching", ticker)
        return news

    # -- corporate actions / holders -----------------------------------------
    def get_actions(self, ticker: str) -> pd.DataFrame:
        t = yf.Ticker(ticker)
        return self._with_retry(lambda: t.actions)

    def get_institutional_holders(self, ticker: str) -> pd.DataFrame:
        t = yf.Ticker(ticker)
        return self._with_retry(lambda: t.institutional_holders)
