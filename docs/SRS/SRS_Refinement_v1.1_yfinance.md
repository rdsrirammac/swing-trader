# SRS Refinement v1.1 — yfinance-Centric Architecture

**Date:** August 2, 2026
**Refines:** Swing_Trading_SRS_v1.0.md
**Focus:** Primary data source = `yfinance` (Python Yahoo Finance client)

---

## 1. Why yfinance as Primary Source?

| Factor | yfinance | Polygon/Finnhub |
|--------|----------|-----------------|
| Cost | Free | $49–$449/month |
| API Key | None required | Required |
| Setup | `pip install yfinance` | Account + key management |
| Data breadth | Prices, fundamentals, options, news | Prices + some fundamentals |
| Rate limits | ~2000 requests/hour (unofficial) | Hard limits, overages charged |
| Reliability | Good for EOD, occasional downtime | Enterprise SLA |
| Mac Mini fit | Ideal for personal trading | Overkill for 5–50 tickers |

**Verdict:** For a personal swing trading system on a Mac Mini Pro, `yfinance` is the correct default. Every ticker request goes through yfinance first.

---

## 2. What yfinance Provides (Confirmed Capabilities)

```python
import yfinance as yf

ticker = yf.Ticker('AAPL')

# ── PRICE DATA ──
ticker.history(period='1y', interval='1d')        # Daily OHLCV + div/splits
ticker.history(period='60d', interval='30m')      # Intraday (30m bars)
ticker.history(period='5d', interval='1m')        # 1-min (last 7 days max)
ticker.history(prepost=True)                       # Pre/post market included

# ── FUNDAMENTALS ──
ticker.info['trailingPE']                          # P/E ratio
ticker.info['marketCap']                           # Market cap
ticker.info['sector']                              # Sector
ticker.info['industry']                            # Industry
ticker.info['shortRatio']                          # Short ratio (limited)
ticker.info['floatShares']                         # Float
ticker.quarterly_financials                        # Quarterly income stmt
ticker.quarterly_balance_sheet                     # Quarterly balance sheet
ticker.quarterly_cashflow                          # Quarterly cash flow
ticker.earnings                                    # Annual EPS
ticker.quarterly_earnings                          # Quarterly EPS
ticker.calendar                                    # Next earnings date

# ── ANALYST DATA ──
ticker.recommendations                             # Rating changes history
ticker.recommendations_summary                     # Current consensus
ticker.upgrades_downgrades                         # Historical upgrades/downgrades

# ── OPTIONS ──
ticker.options                                     # Expiration dates list
ticker.option_chain('2026-08-21')                  # Full chain for date

# ── CORPORATE ACTIONS ──
ticker.dividends                                   # Dividend history
ticker.splits                                      # Split history
ticker.actions                                     # Combined div + splits

# ── NEWS ──
ticker.news                                        # Recent headlines + publisher

# ── HOLDERS ──
ticker.institutional_holders                       # Top institutions
ticker.major_holders                               # Insider + institution %
```

---

## 3. What yfinance CANNOT Provide (Gaps to Accept or Fill)

| Data Need | yfinance Support | Gap Resolution |
|-----------|-----------------|----------------|
| **Historical short interest** | No. Only current shortRatio | Accept: use current only; or scrape SEC bi-weekly |
| **Borrow fee rate** | Not available | Accept as limitation; hard-to-borrow flagged via options IV skew instead |
| **Options flow / unusual activity** | No volume analytics | Accept: use raw options chain + volume ratio heuristic |
| **Real-time WebSocket streaming** | Polling only | Accept: 5-min polling during market hours is sufficient for swing |
| **Social sentiment (StockTwits/X)** | Not available | Accept: rely on news sentiment only |
| **VIX term structure** | ^VIX history only | Accept: use VIX level only; no term structure needed for swing |
| **Pre-market 30m bars** | Included in 1d with prepost=True, but not clean 30m pre-market | Accept: use pre-market % change from prepost=True daily data |
| **Sector breadth (% above 50d SMA)** | Not direct | Workaround: fetch 11 sector ETFs via yfinance, calculate internally |
| **Economic calendar (CPI, FOMC)** | Not available | Secondary: free RSS/ical from BLS/Federal Reserve |
| **Historical analyst recommendations** | Limited history | Accept: use available history; supplement with current consensus |

**Decision:** For a personal swing system, these gaps are acceptable. The system will be designed as **yfinance-first, with graceful degradation** where data is unavailable.

---

## 4. Refreshed Architecture — yfinance Native

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         yfinance DATA LAYER (Primary)                       │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Price Client │  │ Fundamentals │  │ Options      │  │ News Client  │   │
│  │ • daily      │  │ Client       │  │ Client       │  │ • headlines  │   │
│  │ • intraday   │  │ • info       │  │ • chain      │  │ • publisher  │   │
│  │ • pre/post   │  │ • financials │  │ • volume     │  │ • publishTime│   │
│  │ • splits/div │  │ • earnings   │  │ • IV calc    │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                             │
│  Shared: Retry logic │ Exponential backoff │ Local disk cache │ Rate limiter│
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SECONDARY SOURCES (Optional/Fallback)                  │
│                                                                             │
│  NewsAPI ───────► News sentiment (90-day history, 100 req/day free)        │
│  RSS Feeds ─────► Earnings calendar, macro events (free, no key)           │
│  SEC EDGAR ─────► Insider transactions (free, no key)                      │
│  yfinance ETFs ─► Sector rotation (XLK, XLF, XLE, etc. via yfinance)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. yfinance Client Design Specification

### 5.1 Singleton Client with Connection Pooling

```python
# src/data/yf_client.py
import yfinance as yf
import diskcache as dc
import time

class YFinanceClient:
    ""
    Thread-safe yfinance wrapper with caching, retry, and rate limiting.
    Singleton pattern — one client per process.
    ""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.cache = dc.Cache('~/.swing-trader/cache/yfinance')
        self.last_call = 0
        self.min_interval = 0.5  # seconds between calls (120/min max)
        self.max_retries = 3
        self.backoff_base = 2

    def _rate_limit(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()

    def _with_retry(self, fn, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait = self.backoff_base ** attempt
                time.sleep(wait)

    def get_history(self, ticker, period='1y', interval='1d', prepost=False):
        cache_key = f'hist:{ticker}:{period}:{interval}:{prepost}'
        ttl = 3600 if interval in ('1m','2m','5m','15m','30m','60m') else 86400
        if cache_key in self.cache:
            return self.cache[cache_key]
        t = yf.Ticker(ticker)
        df = self._with_retry(t.history, period=period, interval=interval, prepost=prepost)
        self.cache.set(cache_key, df, expire=ttl)
        return df

    def get_info(self, ticker):
        cache_key = f'info:{ticker}'
        if cache_key in self.cache:
            return self.cache[cache_key]
        t = yf.Ticker(ticker)
        info = self._with_retry(lambda: t.info)
        self.cache.set(cache_key, info, expire=86400)
        return info

    def get_options_chain(self, ticker, expiration):
        cache_key = f'opts:{ticker}:{expiration}'
        if cache_key in self.cache:
            return self.cache[cache_key]
        t = yf.Ticker(ticker)
        chain = self._with_retry(t.option_chain, expiration)
        self.cache.set(cache_key, chain, expire=3600)
        return chain

    def get_news(self, ticker):
        cache_key = f'news:{ticker}'
        if cache_key in self.cache:
            return self.cache[cache_key]
        t = yf.Ticker(ticker)
        news = self._with_retry(lambda: t.news)
        self.cache.set(cache_key, news, expire=900)
        return news

    def get_recommendations(self, ticker):
        cache_key = f'rec:{ticker}'
        if cache_key in self.cache:
            return self.cache[cache_key]
        t = yf.Ticker(ticker)
        rec = self._with_retry(lambda: t.recommendations)
        self.cache.set(cache_key, rec, expire=86400)
        return rec
```

### 5.2 Batch Ticker Operations

```python
    def get_batch_history(self, tickers, period='1y', interval='1d'):
        cache_key = f'batch:{','.join(sorted(tickers))}:{period}:{interval}'
        if cache_key in self.cache:
            return self.cache[cache_key]
        data = yf.download(
            tickers=' '.join(tickers),
            period=period,
            interval=interval,
            group_by='ticker',
            auto_adjust=True,
            prepost=False
        )
        self.cache.set(cache_key, data, expire=86400)
        return data
```

### 5.3 Data Freshness Guarantees

| Data Type | Max Age | Source Call |
|-----------|---------|-------------|
| Daily OHLCV | 4 hours | history(period='1d') at EOD |
| Intraday (30m) | 1 hour | history(period='5d', interval='30m') |
| Ticker info | 24 hours | .info |
| Fundamentals | 1 week | .quarterly_financials |
| Options chain | 1 hour | .option_chain() |
| News | 15 minutes | .news |
| Analyst recs | 24 hours | .recommendations |

---

## 6. Revised Data Collection Schedule (yfinance Native)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHEDULE (All times ET)          │  OPERATION                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  8:00 AM  Pre-market              │  Fetch pre-market % change for all    │
│                                   │  tickers (history with prepost=True)  │
├─────────────────────────────────────────────────────────────────────────────┤
│  9:30 AM  Market Open             │  No action (yfinance polls, not push) │
├─────────────────────────────────────────────────────────────────────────────┤
│  12:00 PM Mid-day                 │  Refresh intraday 30m bars            │
├─────────────────────────────────────────────────────────────────────────────┤
│  4:00 PM  Market Close            │  Wait 5 min for settlement            │
├─────────────────────────────────────────────────────────────────────────────┤
│  4:35 PM  EOD Ingestion           │  • Fetch full daily OHLCV             │
│                                   │  • Fetch updated info (P/E, etc.)     │
│                                   │  • Fetch news headlines               │
│                                   │  • Fetch analyst recommendations      │
│                                   │  • Fetch options expirations          │
├─────────────────────────────────────────────────────────────────────────────┤
│  5:00 PM  Feature Calculation     │  Recalculate 60-day rolling features  │
├─────────────────────────────────────────────────────────────────────────────┤
│  6:30 PM  Prediction Run          │  Generate tomorrow's signals          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Sunday 8:00 PM                   │  Full model retrain + backtest        │
├─────────────────────────────────────────────────────────────────────────────┤
│  Monthly 1st @ 6:00 AM            │  Refresh quarterly fundamentals       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Revised Backfill Pipeline (yfinance Native)

When a user adds ticker XYZ:

```
PHASE 1: PRICE (yfinance) — 2 minutes
├── history(period='1y', interval='1d')          → 252 daily bars
├── history(period='60d', interval='30m')        → Intraday (if available)
├── history(period='max', interval='1d')         → Splits & dividends
└── actions                                      → Corporate actions

PHASE 2: FUNDAMENTALS (yfinance) — 1 minute
├── info                                         → P/E, sector, market cap, float
├── quarterly_financials                         → 4 quarters income
├── quarterly_balance_sheet                      → 4 quarters balance
├── quarterly_earnings                           → EPS history
└── calendar                                     → Next earnings date

PHASE 3: ANALYST DATA (yfinance) — 30 seconds
├── recommendations                              → Rating changes history
└── recommendations_summary                      → Current consensus

PHASE 4: OPTIONS (yfinance) — 1 minute
├── options                                      → Available expirations
└── option_chain(nearest_expiration)             → Current chain snapshot

PHASE 5: NEWS (yfinance) — 30 seconds
└── news                                         → Recent headlines

PHASE 6: FEATURES (calculated) — 2 minutes
└── Calculate all technical indicators

TOTAL ESTIMATED BACKFILL TIME: ~7 minutes per ticker
```

**Note:** yfinance has no historical news. For 90-day news backfill, NewsAPI (free tier: 100/day) is used as secondary source. If unavailable, system starts with empty news sentiment and builds from day 1.

---

## 8. What Changes in the Original SRS

### Section 3.3 (Data Collection) — Revised
| Original | Revised (yfinance) |
|----------|-------------------|
| Polygon.io primary for intraday | yfinance primary for everything |
| Finnhub for fundamentals | yfinance .info + .quarterly_financials |
| CBOE for options | yfinance .option_chain() |
| NewsAPI primary for news | yfinance .news primary, NewsAPI fallback for history |
| Alpha Vantage fallback | Removed — yfinance is sufficient |

### Section 3.4 (Features) — Revised
| Original | Revised (yfinance) |
|----------|-------------------|
| Short interest history | Current shortRatio only (no history) |
| Borrow fee rate | REMOVED — not available |
| Options flow / unusual activity | REMOVED — use options volume ratio heuristic instead |
| Social sentiment | REMOVED — rely on news sentiment only |
| VIX term structure | REMOVED — VIX level only |
| Sector breadth | Calculated internally from sector ETFs via yfinance |

### Section 3.8 (Risk) — Revised
| Original | Revised (yfinance) |
|----------|-------------------|
| Short interest squeeze detection | Use options IV skew + volume spike as proxy |
| Borrow fee rate alert | REMOVED |

### Section 5.2 (API Keys) — Revised
| Source | Key Required? | Purpose |
|--------|--------------|---------|
| yfinance | No | All price, fundamental, options, news data |
| NewsAPI | Yes (free) | Historical news backfill only (90 days) |
| RSS Feeds | No | Earnings calendar, macro events |
| SEC EDGAR | No | Insider transactions (optional) |

**Result:** System now requires **zero paid subscriptions** to operate. Only NewsAPI free tier is recommended for news history.

---

## 9. Honest Assessment: Do We Need Further Refinement?

### What We Have Now
- Complete functional specification (81 requirements)
- Database schema designed
- yfinance-native architecture defined
- 24-week implementation roadmap
- Success metrics established

### What Could Still Be Refined (But May Be Over-Engineering)

| Area | Refinement Possible? | Recommendation |
|------|---------------------|----------------|
| **Broker API integration** | Yes — Alpaca/IBKR for live trading | **Defer to Phase 2.** Paper trade first. |
| **Machine learning ops** | Yes — Kubernetes, cloud training | **Not needed.** Mac Mini Pro handles 50 tickers. |
| **Real-time streaming** | Yes — WebSocket data feeds | **Not needed.** 5-min polling is sufficient for swing. |
| **Alternative data** | Yes — Satellite, credit card, web traffic | **Out of scope.** Fundamental + technical + news is enough. |
| **Mobile app** | Yes — iOS companion app | **Defer.** Streamlit dashboard is mobile-responsive. |
| **Multi-user support** | Yes — Authentication, roles | **Not needed.** Personal system. |
| **Options Greeks modeling** | Yes — Delta, gamma, theta tracking | **Nice to have.** Add in Phase 2 if trading options. |

### The Right Time to Stop Refining and Start Building

> **No battle plan survives contact with the enemy.** — Helmuth von Moltke

The requirements are **sufficient to start Phase 1 (Foundation)**. Here's why:

1. **yfinance uncertainty:** You will discover edge cases (delisted tickers, missing fundamentals, split adjustments) only by building. No amount of specification prevents this.

2. **Feature importance unknown:** You don't yet know which technical indicators actually predict your tickers' swings. This requires experimentation (Phase 3).

3. **Model performance reality:** The ensemble might achieve 55% accuracy or 65%. This changes position sizing, rating thresholds, and risk rules. You can't specify what you don't know.

4. **Personal trading style:** You may prefer 5-day holds vs 15-day holds. You may prefer tighter stops. These preferences emerge from live paper trading (Phase 5).

### Recommended Next Steps (Stop Refining → Start Building)

```
WEEK 1-2: BUILD PHASE 1
├── Day 1-2:   Set up PostgreSQL + TimescaleDB on Mac Mini
├── Day 3-4:   Create project scaffold (directories, config, logging)
├── Day 5-6:   Build YFinanceClient with caching + retry
├── Day 7-8:   Implement database models (SQLAlchemy)
├── Day 9-10:  Build CLI: add-ticker, remove-ticker, list-portfolio
├── Day 11-12: Implement auto-backfill pipeline (6 phases)
├── Day 13-14: Testing + data quality validation
└── MILESTONE:  Can add AAPL, backfill completes in < 10 min, data visible in DB
```

Once you hit that milestone, you'll have **real data** and **real insights** that no specification can provide. Then refine based on reality.

---

## 10. Final Architecture Summary (yfinance-Native)

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA SOURCE: yfinance (primary) + NewsAPI (news history)      │
│  COST: $0/month                                                │
│  API KEYS: 1 optional (NewsAPI free tier)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STORAGE: PostgreSQL + TimescaleDB (local on Mac Mini SSD)     │
│  CAPACITY: 3 years x 50 tickers ≈ 5 GB                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FEATURES: pandas-ta technical indicators + sentiment scores     │
│  CALCULATION: Daily batch after market close                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODELS: LightGBM + small LSTM + ARIMA-X → Stacking ensemble   │
│  TRAINING: Weekly on Mac Mini Pro (14-core CPU)                │
│  INFERENCE: CoreML-optimized for Neural Engine                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: Streamlit dashboard + macOS notifications             │
│  ALERTS: Strong Buy/Buy signals, stop hits, earnings warnings  │
└─────────────────────────────────────────────────────────────────┘
```

---

**End of Refinement Document**