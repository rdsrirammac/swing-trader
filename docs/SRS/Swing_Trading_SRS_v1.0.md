# Software Requirements Specification
## Swing Trading Stock Analysis & Prediction System

**Version:** 1.0  
**Date:** July 30, 2026  
**Platform:** macOS (Mac Mini Pro)  
**Author:** System Design Document  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Overview](#2-system-overview)
3. [Functional Requirements](#3-functional-requirements)
   - 3.1 Portfolio Management
   - 3.2 Ticker Universe & Auto-Backfill
   - 3.3 Data Collection & Ingestion
   - 3.4 Feature Engineering
   - 3.5 Market Regime Detection
   - 3.6 Predictive Modeling Engine
   - 3.7 Swing Signal Generation & Rating
   - 3.8 Risk Management & Position Sizing
   - 3.9 Trade Execution & Order Management
   - 3.10 Alert & Notification System
   - 3.11 Backtesting & Simulation
   - 3.12 Performance Analytics & Journaling
   - 3.13 Correlation & Concentration Analysis
   - 3.14 Earnings & Economic Calendar
   - 3.15 Dashboard & Visualization
4. [Non-Functional Requirements](#4-non-functional-requirements)
5. [Data Requirements](#5-data-requirements)
6. [System Architecture](#6-system-architecture)
7. [Missing Components Identified](#7-missing-components-identified)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Appendices](#9-appendices)

---

## 1. Introduction

### 1.1 Purpose
This document specifies the complete requirements for a Swing Trading Stock Analysis and Prediction System designed to run on a Mac Mini Pro. The system shall provide automated data collection, machine learning-based price prediction, swing trade signal generation, portfolio management, and performance analytics for a holding period of 3–21 days.

### 1.2 Scope
The system will:
- Maintain a dynamic portfolio of tickers with full historical backfill on addition
- Collect multi-source financial data (prices, fundamentals, news, options, macro)
- Engineer swing-specific technical and sentiment features
- Detect market regimes and adapt prediction models accordingly
- Generate probability-based swing signals (Strong Buy / Buy / Hold / Trim / Sell)
- Manage risk through position sizing, stop-losses, and portfolio heat monitoring
- Provide backtesting, paper trading, and performance journaling
- Deliver real-time alerts via desktop and mobile notifications

### 1.3 Definitions
| Term | Definition |
|------|------------|
| **Swing Trade** | A position held for 3–21 trading days targeting 3–15% moves |
| **Regime** | A classified market state (Trending, Range-Bound, Volatile, etc.) |
| **R-Multiple** | Return divided by initial risk (R = entry − stop) |
| **Portfolio Heat** | Total capital at risk across all open positions |
| **Backfill** | Historical data ingestion for a newly added ticker |
| **Hypertable** | TimescaleDB time-series optimized table partition |

---

## 2. System Overview

### 2.1 System Context
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL DATA SOURCES                               │
│  Yahoo Finance │ Polygon.io │ NewsAPI │ Finnhub │ CBOE │ RSS Feeds        │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA COLLECTION LAYER                               │
│  • Price ingestion (daily + intraday)                                       │
│  • News & sentiment scraping                                                │
│  • Fundamental & options data                                               │
│  • Macro & sector rotation                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STORAGE LAYER (PostgreSQL + TimescaleDB)           │
│  • stock_prices (hypertable)                                                │
│  • stock_features (hypertable)                                              │
│  • daily_metrics                                                            │
│  • news_sentiment                                                           │
│  • ticker_universe                                                          │
│  • portfolios / holdings / watchlist                                        │
│  • predictions / model_performance                                          │
│  • trades / trade_journal                                                   │
│  • alerts / notifications                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ANALYTICS & ML ENGINE                               │
│  • Feature engineering pipeline                                             │
│  • Market regime detector                                                   │
│  • Ensemble prediction models (LightGBM + LSTM + ARIMA-X)                   │
│  • Self-tuning hyperparameter optimization (Optuna)                         │
│  • Signal generation & rating engine                                        │
│  • Risk management calculator                                               │
│  • Backtesting framework                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                                  │
│  • Streamlit Dashboard (local browser)                                      │
│  • macOS Notification Center                                                │
│  • Email / Push notifications                                               │
│  • CLI tools for administration                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 User Roles
| Role | Description |
|------|-------------|
| **Trader** | Primary user who views signals, manages portfolio, receives alerts |
| **System** | Automated processes that collect data, train models, generate signals |
| **Administrator** | Manages API keys, monitors system health, tunes thresholds |

---

## 3. Functional Requirements

### 3.1 Portfolio Management (PF)

#### PF-001: Portfolio Creation
The system shall support multiple named portfolios, each with independent cash balance, risk settings, and holdings.

#### PF-002: Ticker Addition
When a user adds a ticker to the portfolio or watchlist:
- The system shall insert the ticker into `ticker_universe` with status `pending`
- The system shall automatically trigger a full historical backfill for 1 year
- The system shall notify the user when backfill completes and the ticker is prediction-ready
- The system shall reject tickers with insufficient liquidity (< 1M avg daily volume)

#### PF-003: Ticker Removal
When a user removes a ticker:
- The system shall close any open positions for that ticker
- The system shall optionally retain historical data for model training
- The system shall remove the ticker from active watchlists

#### PF-004: Position Entry
The system shall allow entry of a new position with:
- Ticker symbol
- Number of shares
- Entry price (or market order simulation)
- Stop-loss price (auto-calculated as 2× ATR if not provided)
- Take-profit levels (auto-calculated as 2× and 3.5× ATR)
- Notes / thesis

#### PF-005: Position Updates
The system shall support:
- Partial exits (trimming 50% at first target)
- Stop-loss adjustments (trailing stops based on ATR)
- Status changes: active → trimmed → closed
- Automatic closure when stop or target is hit

#### PF-006: Portfolio Summary
The system shall display real-time:
- All active holdings with unrealized P&L
- Cash balance and buying power
- Total portfolio value and daily change
- Portfolio heat (total risk exposure)
- Win/loss statistics for closed trades

#### PF-007: Watchlist Management
The system shall maintain a watchlist with:
- Trigger conditions (e.g., RSI < 30 AND prob_5pct_up_10d > 0.65)
- Alert status (triggered / not triggered)
- Latest prediction and rating
- Distance to trigger conditions

---

### 3.2 Ticker Universe & Auto-Backfill (TB)

#### TB-001: Backfill Trigger
Adding a ticker shall automatically initiate a 6-phase backfill pipeline:
1. **Price Data**: 1 year daily OHLCV + 60 days intraday (30-min)
2. **Fundamentals**: 4 quarters earnings, P/E history, short interest, market cap, sector
3. **News & Sentiment**: 90 days historical news with FinBERT sentiment scoring
4. **Options Data**: Current options chain, put/call ratio history, IV rank
5. **Feature Engineering**: Calculate all technical indicators and swing features
6. **Model Warm-up**: Train regime detector and generate first prediction

#### TB-002: Backfill Progress Tracking
The system shall track and display:
- Current phase and percentage complete
- Records ingested per data source
- Errors and retry attempts
- Estimated time remaining

#### TB-003: Data Quality Gate
A ticker shall not be marked `active` until:
- Minimum 200 daily price bars collected
- Feature completeness score ≥ 80%
- Data freshness: last price within 2 trading days
- No critical data source failures

#### TB-004: Failed Backfill Handling
If backfill fails:
- The system shall retry up to 3 times with exponential backoff
- After 3 failures, status shall be set to `failed` with error log
- The user shall be notified with specific failure reason
- Manual retry shall be available via CLI

#### TB-005: Daily Incremental Update
For all active tickers, the system shall:
- Fetch EOD prices at 4:35 PM ET
- Update fundamentals if earnings released
- Ingest daily news and calculate sentiment
- Recalculate rolling features for the last 60 days
- Generate updated predictions

#### TB-006: Ticker Screening
The system shall auto-screen tickers and reject those that don't meet swing criteria:
- Price between $10 and $500
- Average daily volume ≥ 1,000,000 shares
- ATR% between 1.5% and 8% of price
- Not in bankruptcy or delisting process

---

### 3.3 Data Collection & Ingestion (DC)

#### DC-001: Price Data Sources
| Source | Data | Frequency | Fallback |
|--------|------|-----------|----------|
| Yahoo Finance | Daily OHLCV, intraday, splits, dividends | 3× daily + EOD | Polygon.io |
| Polygon.io | Intraday (30-min), pre/post market | Real-time | Alpha Vantage |

#### DC-002: Fundamental Data
- Quarterly EPS, revenue, guidance (Yahoo Finance / Finnhub)
- P/E, PEG, P/S ratios (updated quarterly)
- Short interest % of float (bi-weekly from exchange data)
- Borrow fee rate (daily if available)
- Insider transactions (daily from SEC EDGAR)

#### DC-003: News & Sentiment Data
- NewsAPI: General financial news (90-day history)
- RSS feeds: Bloomberg, Reuters, Seeking Alpha, Benzinga
- Social sentiment: StockTwits, Twitter/X (if API available)
- Sentiment scoring: FinBERT model (positive/neutral/negative + confidence)

#### DC-004: Options Market Data
- Unusual options activity (volume > 2× average)
- Put/Call ratio (5-day rolling)
- Max pain price for nearest expiration
- Implied volatility rank and percentile
- Options flow sentiment (bullish/bearish skew)

#### DC-005: Macro & Sector Data
- VIX index (daily)
- SPY, QQQ, IWM performance (daily)
- Sector ETF performance (XLK, XLF, XLE, etc.)
- Treasury yields (10Y, 2Y)
- Economic calendar (CPI, FOMC, NFP, GDP)

#### DC-006: Data Validation
All ingested data shall be validated for:
- Missing values (reject if > 5% missing in a batch)
- Out-of-range values (e.g., negative prices, volume = 0)
- Timestamp consistency (no future dates, no duplicates)
- Cross-field validation (high ≥ low, close within high-low range)

#### DC-007: API Rate Limiting
The system shall:
- Respect rate limits of all data providers
- Implement request queuing and throttling
- Cache responses to minimize redundant calls
- Rotate API keys if multiple available

---

### 3.4 Feature Engineering (FE)

#### FE-001: Technical Indicators
| Indicator | Parameters | Purpose |
|-----------|------------|---------|
| RSI | 2, 14 | Mean reversion (RSI2) and momentum (RSI14) |
| MACD | 12, 26, 9 | Trend momentum and divergence |
| ATR | 14 | Stop-loss sizing, volatility regime |
| Bollinger Bands | 20, 2 | Mean reversion, squeeze detection |
| EMA | 20 | Short-term trend |
| SMA | 50 | Intermediate trend |
| ADX | 14 | Trend strength (not direction) |
| ROC | 5, 10, 21 | Rate of change across swing horizons |
| OBV | — | Volume-confirmed trend |
| Volume Ratio | vs 20-day SMA | Institutional participation |

#### FE-002: Relative Strength Features
- 5-day, 10-day, 21-day return vs SPY
- 5-day, 10-day, 21-day return vs sector ETF
- IBD-style RS Rating (percentile rank)
- Sector momentum rank (1-11 sectors)

#### FE-003: Volatility Features
- Realized volatility (20-day annualized)
- Realized volatility percentile (vs 252-day history)
- ATR as % of price
- Bollinger Bandwidth (squeeze expansion signal)
- Historical vs implied volatility spread

#### FE-004: Sentiment Features
- 3-day rolling average news sentiment score
- News volume velocity (rate of change)
- Social sentiment momentum
- Analyst rating change velocity (upgrades - downgrades)
- Options sentiment (put/call skew)

#### FE-005: Fundamental Features
- P/E percentile vs sector (0-100)
- P/E percentile vs 1-year history
- Short interest % of float
- Short interest trend (increasing/decreasing)
- Borrow fee rate (squeeze indicator)
- Earnings surprise history (beat/miss streak)

#### FE-006: Macro Features
- VIX level and percentile
- VIX term structure (contango/backwardation)
- SPY trend (above/below 20-day EMA)
- Sector breadth (% stocks above 50-day SMA)
- Yield curve spread (10Y - 2Y)

#### FE-007: Feature Store
All engineered features shall be stored in `stock_features` hypertable with:
- Timestamp + ticker primary key
- 30+ feature columns
- Automatic backfill on ticker addition
- Daily incremental update for last 60 days

---

### 3.5 Market Regime Detection (MR)

#### MR-001: Regime Classification
The system shall classify market regime daily into one of:
- **Strong Trend**: SPY ADX > 25, VIX < 20, sector breadth > 60%
- **Weak Trend**: SPY ADX 15-25, VIX 20-25
- **Range-Bound**: SPY ADX < 20, VIX < 22, BB width < 10%
- **High Volatility**: VIX > 25 or ATR expansion > 150% of 20-day avg
- **Earnings Season**: > 20% S&P 500 reporting in next 2 weeks

#### MR-002: Regime Transition Detection
The system shall detect regime changes using:
- VIX spike > 2 standard deviations
- SPY 20-day EMA crossover of 50-day SMA
- Sector rotation (top 3 sectors change within 5 days)
- Breadth thrust (90% volume on upside vs downside)

#### MR-003: Regime-Specific Model Selection
Based on detected regime, the system shall:
- Load the ensemble trained on historical similar regimes
- Adjust probability thresholds for signal generation
- Modify position sizing (reduce in high volatility)
- Update expected hold time (shorter in volatile regimes)

#### MR-004: Regime Performance Tracking
The system shall track model performance per regime:
- Win rate by regime
- Average R-multiple by regime
- Max drawdown by regime
- Sharpe ratio by regime

---

### 3.6 Predictive Modeling Engine (PM)

#### PM-001: Target Variables
Instead of predicting absolute price, the model shall predict:
| Target | Horizon | Description |
|--------|---------|-------------|
| prob_3pct_up_5d | 5 days | Probability of ≥ 3% gain from current price |
| prob_5pct_up_10d | 10 days | Probability of ≥ 5% gain |
| prob_10pct_up_21d | 21 days | Probability of ≥ 10% gain |
| expected_return_10d | 10 days | Mean predicted return (regression) |
| optimal_hold_days | — | Classification: 3, 5, 10, or 21 days |
| max_drawdown_10d | 10 days | Expected worst pullback from entry |

#### PM-002: Base Models
| Model | Type | Role |
|-------|------|------|
| LightGBM | Gradient Boosting | Primary tabular feature model |
| LSTM/GRU | Recurrent Neural Network | Sequence pattern recognition |
| ARIMA-X | Statistical | Baseline mean-reversion and trend |
| Random Forest | Ensemble | Robustness check and feature importance |

#### PM-003: Meta-Learner (Stacking)
A Ridge Regression or XGBoost meta-learner shall:
- Combine predictions from all base models
- Learn optimal weights per regime
- Output final probability and confidence interval
- Be retrained weekly with expanding window

#### PM-004: Self-Tuning Pipeline (Daily)
```
Every day at 6:30 PM ET:
1. INGEST: Load today's new data
2. REGIME UPDATE: Classify current market regime
3. DRIFT CHECK: Compare feature distributions (KL-divergence vs 30-day avg)
   - If drift > threshold: flag "regime change", use shorter lookback
4. WALK-FORWARD VALIDATION: Test model on last 60 days
5. HYPERPARAMETER SWEEP: Optuna (50 trials, Bayesian optimization)
6. MODEL SELECTION:
   - If new model MAPE improves > 2% vs current: deploy
   - Else: retain current model, log performance degradation
7. FEATURE SELECTION: Recursive elimination (drop importance < 0.5%)
8. THRESHOLD TUNE: Adjust signal probability thresholds based on recent win rate
9. PREDICT: Generate next-day predictions for all active tickers
10. LOG: Record predictions, model version, and confidence
```

#### PM-005: Model Performance Tracking
The system shall log to `model_performance`:
- MAPE (Mean Absolute Percentage Error)
- Directional accuracy (% of correct up/down calls)
- Sharpe ratio of model-based portfolio
- Max drawdown
- Calmar ratio

#### PM-006: Model Versioning
All models shall be versioned using MLflow:
- Model artifact storage
- Hyperparameter logging
- Training data hash
- Performance metrics
- A/B test capability (shadow mode)

---

### 3.7 Swing Signal Generation & Rating (SR)

#### SR-001: Signal Generation
For each active ticker, the system shall generate:
- Predicted probability of target move
- Confidence interval (5th–95th percentile)
- Expected return
- Optimal entry, stop-loss, and target prices
- Suggested position size (based on Kelly Criterion + risk limits)

#### SR-002: Rating Algorithm
```
Score = 0

# Price momentum (40% weight)
if predicted_return > 0.05:        score += 2.0
elif predicted_return > 0.02:      score += 1.0
elif predicted_return < -0.05:     score -= 2.0
elif predicted_return < -0.02:     score -= 1.0

# Confidence (20% weight)
ci_width = (upper - lower) / price
if ci_width < 0.03:                score *= 1.2
elif ci_width > 0.08:              score *= 0.8

# Sentiment (20% weight)
score += sentiment_score * 1.5

# Valuation (10% weight)
if pe_percentile > 0.8:            score -= 0.5
elif pe_percentile < 0.2:          score += 0.5

# Volatility regime (10% weight)
if regime == "high_vol" and predicted_return < 0: score -= 0.5

# Map to rating
score >= 1.5  → "Strong Buy"
score >= 0.5  → "Buy"
score <= -1.5 → "Sell"
score <= -0.5 → "Hold" (weak sell)
else           → "Hold"
```

#### SR-003: Rating Definitions
| Rating | Criteria | Position Size | Action |
|--------|----------|---------------|--------|
| **Strong Buy** | prob_5pct > 65%, ATR < 3%, RS top 20%, positive flow | 10% of portfolio | Full position |
| **Buy** | prob_3pct > 60%, R/R > 1:2, no earnings in 10 days | 5% of portfolio | Half position |
| **Hold** | Position active, not at target/stop, regime unchanged | — | Maintain stop |
| **Trim** | Position up > 50% of target, RSI > 75, regime shifting | — | Sell 50% |
| **Sell** | Stop hit, target reached, or regime flips to volatile | — | Close all |
| **Watch** | Meets 3 of 5 criteria but not enough for Buy | — | Monitor daily |

#### SR-004: Bracket Order Generation
For each Buy signal, the system shall auto-calculate:
```python
stop_loss = entry_price - (2.0 * atr_14)
take_profit_1 = entry_price + (2.0 * atr_14)   # Sell 50%
take_profit_2 = entry_price + (3.5 * atr_14)   # Sell 50%
risk_per_share = entry_price - stop_loss
position_shares = (portfolio_value * 0.02) / risk_per_share
```

#### SR-005: Earnings Blackout
The system shall:
- Flag tickers with earnings within 10 days
- Prevent new Buy signals 5 days before earnings
- Suggest closing active positions 2 days before earnings (configurable)
- Display earnings date prominently on dashboard

---

### 3.8 Risk Management & Position Sizing (RM)

#### RM-001: Per-Trade Risk
- Maximum risk per trade: 2% of portfolio value (configurable 1-3%)
- Position size = (Portfolio × Risk%) / (Entry − Stop)
- Never exceed 10% of portfolio in a single ticker

#### RM-002: Portfolio Heat
- Maximum portfolio heat: 20% of capital at risk (sum of all stop distances)
- If heat exceeds 20%, block new Buy signals
- Display heat gauge on dashboard (green < 10%, yellow 10-20%, red > 20%)

#### RM-003: Sector Concentration
- Maximum 30% of portfolio in any single sector
- Alert when sector exposure approaches limit
- Suggest diversification candidates

#### RM-004: Correlation Check
Before allowing a new position, the system shall:
- Calculate 60-day correlation with all existing holdings
- Reject if correlation > 0.80 with any existing position
- Suggest uncorrelated alternatives

#### RM-005: Drawdown Controls
- Daily max drawdown alert: 5% from peak portfolio value
- Weekly max drawdown alert: 10% from peak
- Auto-reduce position sizes by 50% after 10% drawdown
- Auto-reduce to 25% after 15% drawdown (circuit breaker)

#### RM-006: Volatility Adjusted Sizing
In high volatility regimes (VIX > 25):
- Reduce position sizes by 50%
- Widen stops to 2.5× ATR
- Reduce profit targets to 2× ATR

#### RM-007: Trailing Stops
For positions up > 3% from entry:
- Activate trailing stop at 1.5× ATR below highest close
- Update trailing stop daily after market close
- Notify when trailing stop is adjusted

---

### 3.9 Trade Execution & Order Management (TE)

#### TE-001: Trade Journal
Every entry and exit shall be logged with:
- Ticker, date, time, price, shares
- Entry thesis (auto-populated from signal)
- Stop loss and targets
- Screenshot of chart at entry (optional)
- Expected R-multiple at entry

#### TE-002: Exit Tracking
The system shall track:
- Actual exit price and date
- Realized P&L and R-multiple
- Exit reason (stop, target, manual, earnings, regime change)
- Holding period in days
- Slippage (expected vs actual fill)

#### TE-003: Paper Trading Mode
The system shall support a paper trading mode:
- Simulate fills at market open/close prices
- Track virtual P&L separately from real portfolio
- Allow A/B testing of model versions
- Generate paper trading performance reports

#### TE-004: Broker Integration (Future)
Architecture shall support future broker API integration:
- Alpaca, Interactive Brokers, TD Ameritrade
- Place bracket orders (entry + stop + target)
- Fetch real-time positions and buying power
- Webhook-based order status updates

---

### 3.10 Alert & Notification System (AL)

#### AL-001: Signal Alerts
Notify when:
- A ticker rating changes to Strong Buy or Buy
- A watchlist trigger condition is met
- A position hits stop loss or take profit
- A trailing stop is adjusted

#### AL-002: Risk Alerts
Notify when:
- Portfolio heat exceeds 20%
- Drawdown exceeds 5% daily or 10% weekly
- Sector concentration exceeds 30%
- Correlation threshold breached
- Volatility regime shifts to High

#### AL-003: Data Alerts
Notify when:
- Backfill completes (success or failure)
- Daily data update fails for a ticker
- API rate limit approaching
- Model performance degrades (> 5% MAPE increase)

#### AL-004: Earnings Alerts
Notify:
- 10 days before earnings (warning)
- 5 days before earnings (consider closing)
- 1 day before earnings (final warning)
- After earnings (surprise vs expected)

#### AL-005: Notification Channels
| Channel | Use Case | Priority |
|---------|----------|----------|
| macOS Notification Center | Signal alerts, stop hits | High |
| Email | Daily summary, weekly report | Medium |
| SMS / Push | Critical alerts (drawdown, circuit breaker) | Critical |
| Dashboard banner | All alerts with dismiss capability | All |

#### AL-006: Alert Rules Engine
Users shall configure:
- Minimum rating threshold for alerts (e.g., only Strong Buy)
- Time windows (no alerts pre-market unless critical)
- Quiet hours (configurable)
- Alert deduplication (same alert within 4 hours suppressed)

---

### 3.11 Backtesting & Simulation (BT)

#### BT-001: Backtest Engine
The system shall support walk-forward backtesting:
- Training window: expanding (min 6 months, max 3 years)
- Test window: 20 trading days (1 swing cycle)
- Step size: 5 days (rolling forward)
- Transaction costs: $0.01/share or configurable commission
- Slippage: 0.05% of price per trade

#### BT-002: Backtest Metrics
For each backtest, report:
- Total return and CAGR
- Win rate (%) and profit factor
- Average R-multiple (win and loss)
- Max drawdown and max consecutive losses
- Sharpe ratio and Sortino ratio
- Expectancy per trade
- Return by regime (trending vs range-bound vs volatile)

#### BT-003: Strategy Comparison
The system shall allow comparison of:
- Different model versions
- Different entry thresholds (60% vs 65% vs 70% prob)
- Different stop strategies (2× ATR vs 1.5× ATR)
- Different position sizing (Kelly vs fixed fractional)

#### BT-004: Monte Carlo Simulation
Run 1,000 randomized simulations to estimate:
- Probability of ruin
- Expected drawdown distribution
- Confidence intervals for annual return
- Optimal position sizing via Kelly Criterion

#### BT-005: Out-of-Sample Testing
- Reserve last 6 months of data for final validation
- Never use test data in training or hyperparameter tuning
- Report out-of-sample metrics separately

---

### 3.12 Performance Analytics & Journaling (PA)

#### PA-001: Trade Journal Dashboard
Display all closed trades with:
- Entry/exit dates and prices
- P&L in dollars and R-multiples
- Holding period
- Exit reason
- Screenshot link
- Notes field

#### PA-002: Performance Metrics
Calculate and display:
- **Overall**: Total return, CAGR, Sharpe, Sortino, Calmar
- **Win/Loss**: Win rate, avg win, avg loss, profit factor
- **R-Multiples**: Avg R, max R, R distribution histogram
- **Time**: Avg hold time, time in market %
- **Drawdown**: Max DD, DD duration, recovery time

#### PA-003: Attribution Analysis
Break down performance by:
- Ticker (best and worst performers)
- Sector (which sectors worked)
- Regime (performance in trending vs range-bound)
- Rating (did Strong Buy outperform Buy?)
- Month/Quarter (seasonality patterns)

#### PA-004: Behavioral Analytics
Track and report trader bias:
- Early exit rate (selling before target)
- Stop violation rate (moving stops to avoid loss)
- Revenge trading (increasing size after loss)
- Overtrading (trades per week vs plan)

#### PA-005: Weekly & Monthly Reports
Auto-generate and email:
- Weekly: Open positions, new signals, P&L summary
- Monthly: Full performance report, model accuracy, regime summary
- Quarterly: Strategy review, suggested adjustments

---

### 3.13 Correlation & Concentration Analysis (CC)

#### CC-001: Correlation Matrix
Display 60-day correlation matrix for:
- All holdings
- Holdings vs major indices (SPY, QQQ, VIX)
- Holdings vs sector ETFs

#### CC-002: Portfolio Diversification Score
Calculate a diversification score (0-100) based on:
- Number of holdings (ideal: 5-10)
- Sector spread (penalty for > 30% in one sector)
- Correlation average (penalty for avg correlation > 0.5)
- Market beta (penalty for beta > 1.5)

#### CC-003: Concentration Alerts
Alert when:
- Single position > 15% of portfolio
- Single sector > 30% of portfolio
- Portfolio beta > 1.3
- Avg correlation > 0.6

#### CC-004: Suggested Replacements
When correlation is too high, suggest:
- Uncorrelated tickers from watchlist
- Sector ETFs for diversification
- Inverse correlation pairs (e.g., XLU vs XLE)

---

### 3.14 Earnings & Economic Calendar (EC)

#### EC-001: Earnings Calendar
Track for all holdings and watchlist:
- Earnings date (confirmed vs estimated)
- EPS estimate vs whisper number
- Revenue estimate
- Historical earnings surprise rate
- Post-earnings drift history

#### EC-002: Earnings Impact Modeling
Predict post-earnings move probability:
- Based on historical reaction to beats/misses
- Options implied move (straddle price)
- Sentiment leading into earnings

#### EC-003: Economic Calendar
Track macro events:
- CPI, PPI, FOMC meetings, NFP, GDP
- Pre-market vs post-market timing
- Historical market reaction
- Alert 24 hours before major events

#### EC-004: Earnings Avoidance Rules
Configurable rules:
- No new positions 5 days before earnings
- Suggest closing 2 days before (configurable)
- Auto-close at market open on earnings day (optional)
- Resume trading day after earnings

---

### 3.15 Dashboard & Visualization (DV)

#### DV-001: Main Dashboard Layout
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PORTFOLIO SUMMARY          │  MARKET REGIME          │  ALERTS            │
│  • Total Value: $XXX,XXX    │  • Current: Trending    │  • 3 new signals   │
│  • Daily P&L: +$X,XXX       │  • VIX: 18.5            │  • 1 stop hit      │
│  • Heat: 12% (Green)        │  • Breadth: 72%         │  • Earnings warning│
├─────────────────────────────────────────────────────────────────────────────┤
│  ACTIVE POSITIONS           │  WATCHLIST / SIGNALS                         │
│  Ticker │ Shares │ P&L │ R  │  Ticker │ Rating │ Prob │ Entry │ Stop │ Tgt │
├─────────────────────────────────────────────────────────────────────────────┤
│  MODEL PERFORMANCE          │  SECTOR ROTATION        │  CORRELATION       │
│  • Win Rate: 62%            │  • Tech: +2.1%          │  • Avg: 0.42       │
│  • Avg R: 1.8               │  • Energy: -1.3%        │  • Max: 0.78       │
│  • Sharpe: 1.4              │  • Health: +0.8%        │  • Score: 78/100   │
├─────────────────────────────────────────────────────────────────────────────┤
│  BACKTEST RESULTS           │  TRADE JOURNAL          │  SYSTEM HEALTH     │
│  • Last 60 days: +8.2%      │  • Last 5 trades        │  • All systems OK  │
│  • vs Buy-Hold: +3.1%       │  • R-multiples chart    │  • Last update: now│
└─────────────────────────────────────────────────────────────────────────────┘
```

#### DV-002: Chart Visualization
For each ticker, display:
- Price chart with EMA 20, SMA 50, Bollinger Bands
- Volume bars with 20-day average
- RSI(14) subplot with overbought/oversold lines
- MACD histogram
- ATR subplot
- Entry/exit markers for historical trades
- Support/resistance levels (pivot points)

#### DV-003: Performance Charts
- Equity curve (portfolio value over time)
- Drawdown chart (underwater curve)
- R-multiple distribution histogram
- Monthly returns heatmap
- Win rate by regime bar chart

#### DV-004: Real-Time Updates
- Auto-refresh every 5 minutes during market hours
- WebSocket or polling for price updates
- Push updates for signal changes
- Last update timestamp displayed

---

## 4. Non-Functional Requirements

### 4.1 Performance
| Metric | Requirement |
|--------|-------------|
| Data ingestion | Complete EOD update for 20 tickers within 10 minutes |
| Prediction generation | All active tickers within 2 minutes |
| Dashboard load | < 3 seconds for full dashboard |
| Alert latency | < 30 seconds from trigger to notification |
| Backfill | Complete within 15 minutes per ticker |

### 4.2 Reliability
- Daily data pipeline success rate: > 99%
- Automatic retry on transient failures (3 attempts)
- Graceful degradation: if news fails, continue with price data only
- Data backup: Daily automated backup to external drive

### 4.3 Scalability
- Support up to 50 tickers in universe (Mac Mini Pro limit)
- Support up to 5 simultaneous open positions
- Database: Handle 3 years of data for 50 tickers (~5GB)

### 4.4 Security
- API keys stored in macOS Keychain or encrypted .env
- Database connection over SSL
- No sensitive data in logs
- Dashboard accessible only from localhost (default)

### 4.5 Maintainability
- Modular Python architecture (separate modules per layer)
- Comprehensive logging (rotated daily, 30-day retention)
- Configuration via YAML files (not hardcoded)
- Unit tests for all calculation modules
- Integration tests for data pipelines

### 4.6 Usability
- Single-command setup (`make install`)
- CLI for all admin operations
- Web dashboard for visualization
- Clear error messages with suggested fixes
- Documentation for all API endpoints and CLI commands

---

## 5. Data Requirements

### 5.1 Data Retention
| Data Type | Retention | Storage |
|-----------|-----------|---------|
| Intraday prices (30-min) | 60 days | ~500MB |
| Daily prices | 3 years | ~200MB |
| Features | 3 years | ~300MB |
| News / sentiment | 1 year | ~1GB |
| Predictions | 3 years | ~50MB |
| Trades / journal | Permanent | ~10MB |
| Model versions | Last 10 | ~500MB |

### 5.2 Data Quality Standards
- Price data: < 0.1% missing values
- Feature data: < 1% missing values (forward-filled acceptable)
- News sentiment: All articles scored within 24 hours
- Predictions: Confidence interval always provided

### 5.3 Data Sources & API Keys Required
| Source | API Key Required | Cost (2026) |
|--------|-----------------|-------------|
| Yahoo Finance | None | Free |
| Polygon.io | Yes | Free tier: 5 API calls/min |
| NewsAPI | Yes | Free tier: 100 requests/day |
| Finnhub | Yes | Free tier: 60 calls/min |
| CBOE Options | None | Free |
| SEC EDGAR | None | Free |

---

## 6. System Architecture

### 6.1 Technology Stack
| Layer | Technology | Version |
|-------|-----------|---------|
| OS | macOS | 14+ |
| Language | Python | 3.11+ |
| Database | PostgreSQL + TimescaleDB | 15+ |
| Cache | Redis | 7+ |
| ML Framework | LightGBM, PyTorch, Darts | Latest |
| Sentiment | Transformers (FinBERT) | Latest |
| Optimization | Optuna | Latest |
| Tracking | MLflow | Latest |
| Dashboard | Streamlit | Latest |
| Scheduling | launchd (macOS native) | — |
| Notifications | pync (macOS) | Latest |

### 6.2 Directory Structure
```
~/swing-trader/
├── config/
│   ├── settings.yaml           # Main configuration
│   ├── api_keys.yaml           # Encrypted API credentials
│   └── regimes.yaml            # Regime classification rules
├── src/
│   ├── data/
│   │   ├── collectors/         # Yahoo, Polygon, NewsAPI clients
│   │   ├── validators/         # Data quality checks
│   │   └── storage/            # Database models & queries
│   ├── features/
│   │   ├── technical.py        # Indicator calculations
│   │   ├── sentiment.py        # FinBERT scoring
│   │   └── engineering.py      # Feature pipeline
│   ├── models/
│   │   ├── regime_detector.py  # Market regime classification
│   │   ├── ensemble.py         # Stacked ensemble
│   │   ├── backtest.py         # Walk-forward engine
│   │   └── tuner.py            # Optuna hyperparameter tuning
│   ├── signals/
│   │   ├── generator.py        # Signal & rating generation
│   │   ├── risk_manager.py     # Position sizing & heat
│   │   └── alerts.py           # Notification engine
│   ├── portfolio/
│   │   ├── manager.py          # Portfolio CRUD operations
│   │   ├── backfill.py         # Auto-backfill orchestrator
│   │   └── journal.py          # Trade logging
│   └── dashboard/
│       ├── app.py              # Streamlit main app
│       ├── charts.py           # Plotly visualizations
│       └── pages/              # Dashboard pages
├── notebooks/
│   ├── exploration/            # Data analysis notebooks
│   └── backtests/              # Strategy research notebooks
├── tests/
│   ├── unit/                   # Unit tests
│   └── integration/            # Pipeline tests
├── logs/                       # Application logs
├── backups/                    # Database backups
├── Makefile                    # Setup & run commands
└── requirements.txt
```

### 6.3 Process Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│  launchd Scheduled Jobs (macOS)                                 │
├─────────────────────────────────────────────────────────────────┤
│  com.swingtrader.premarket   │  8:30 AM  │ Fetch pre-market    │
│  com.swingtrader.midday      │  12:00 PM │ Midday price update │
│  com.swingtrader.eod         │  4:35 PM  │ EOD data + features │
│  com.swingtrader.predict     │  6:30 PM  │ Model predictions   │
│  com.swingtrader.weekly      │  Sun 8PM  │ Full model retrain  │
│  com.swingtrader.backup      │  Daily 2AM│ Database backup     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Missing Components Identified

Based on comprehensive analysis, the following components were identified as critical additions beyond the initial scope:

### 7.1 Risk Management (Previously Under-Specified)
- **Portfolio heat monitoring** — Total capital at risk across all positions
- **Sector concentration limits** — Prevent overexposure to single sector
- **Correlation checking** — Prevent adding highly correlated positions
- **Drawdown circuit breakers** — Auto-reduce exposure after consecutive losses
- **Volatility-adjusted position sizing** — Reduce size in high VIX regimes
- **Trailing stop automation** — Dynamic stop adjustment as positions profit

### 7.2 Trade Execution & Journaling
- **Trade journal with screenshots** — Visual record of entry/exit decisions
- **Exit reason tracking** — Stop, target, manual, earnings, regime change
- **Slippage tracking** — Expected vs actual fill prices
- **Paper trading mode** — Virtual portfolio for strategy validation
- **Behavioral analytics** — Early exit rate, stop violation, revenge trading detection

### 7.3 Alert & Notification System
- **Multi-channel alerts** — macOS notifications, email, SMS for critical events
- **Configurable alert rules** — User-defined thresholds and quiet hours
- **Earnings alerts** — 10/5/1 day warnings before earnings
- **Risk alerts** — Heat, drawdown, concentration breach notifications
- **Signal alerts** — Real-time Buy/Strong Buy notifications

### 7.4 Backtesting Framework
- **Walk-forward backtesting** — Out-of-sample validation
- **Monte Carlo simulation** — Probability of ruin, drawdown distribution
- **Strategy comparison** — A/B test different thresholds and stops
- **Transaction cost modeling** — Commission and slippage inclusion
- **Regime-specific backtests** — Performance in trending vs range-bound markets

### 7.5 Correlation & Diversification Analysis
- **Real-time correlation matrix** — 60-day rolling correlation of all holdings
- **Diversification score** — Composite metric for portfolio balance
- **Concentration alerts** — Single position/sector limits
- **Replacement suggestions** — Uncorrelated alternatives when limits breached

### 7.6 Earnings & Economic Calendar Integration
- **Earnings date tracking** — Confirmed and estimated dates
- **Earnings impact prediction** — Historical reaction modeling
- **Earnings avoidance rules** — Auto-blackout periods before earnings
- **Economic event alerts** — CPI, FOMC, NFP with historical impact data

### 7.7 Performance Analytics
- **Attribution analysis** — Performance by ticker, sector, regime, month
- **R-multiple tracking** — Risk-adjusted return measurement
- **Behavioral bias detection** — Identify destructive trading patterns
- **Weekly/monthly auto-reports** — Automated performance summaries

### 7.8 System Health & Monitoring
- **Data pipeline monitoring** — Success/failure rates per source
- **Model drift detection** — KL-divergence monitoring for feature distributions
- **API rate limit tracking** — Prevent service interruptions
- **System resource monitoring** — CPU, memory, disk usage on Mac Mini

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)
- [ ] PostgreSQL + TimescaleDB setup
- [ ] Project scaffolding and directory structure
- [ ] Data collection layer (Yahoo Finance daily prices)
- [ ] Basic database schema (prices, tickers, portfolios)
- [ ] CLI tool for adding/removing tickers
- [ ] Auto-backfill pipeline (Phase 1: prices only)

### Phase 2: Features & Sentiment (Weeks 4-6)
- [ ] Technical indicator calculations (pandas-ta)
- [ ] Feature store implementation
- [ ] NewsAPI integration
- [ ] FinBERT sentiment scoring
- [ ] Auto-backfill Phase 2-3 (fundamentals + news)
- [ ] Data quality scoring

### Phase 3: Modeling (Weeks 7-10)
- [ ] LightGBM baseline model
- [ ] ARIMA-X statistical model
- [ ] LSTM/GRU sequence model
- [ ] Meta-learner stacking ensemble
- [ ] Walk-forward backtesting framework
- [ ] Model versioning with MLflow

### Phase 4: Signals & Risk (Weeks 11-13)
- [ ] Regime detection module
- [ ] Signal generation engine
- [ ] Rating algorithm (Strong Buy → Sell)
- [ ] Position sizing (Kelly + risk limits)
- [ ] Portfolio heat monitoring
- [ ] Stop-loss and target calculation

### Phase 5: Portfolio & Alerts (Weeks 14-16)
- [ ] Full portfolio management (entry/exit/trim)
- [ ] Watchlist with trigger conditions
- [ ] macOS notification system
- [ ] Email alerts
- [ ] Earnings calendar integration
- [ ] Correlation analysis

### Phase 6: Dashboard & Analytics (Weeks 17-19)
- [ ] Streamlit dashboard (main layout)
- [ ] Interactive charts (Plotly)
- [ ] Performance analytics page
- [ ] Trade journal page
- [ ] Backtest results visualization
- [ ] System health monitoring

### Phase 7: Polish & Automation (Weeks 20-22)
- [ ] launchd job configuration
- [ ] Paper trading mode
- [ ] Weekly/monthly auto-reports
- [ ] Comprehensive testing
- [ ] Documentation
- [ ] Backup and recovery procedures

### Phase 8: Optimization (Weeks 23-24)
- [ ] Performance profiling and optimization
- [ ] Mac Mini Pro specific tuning (CoreML conversion)
- [ ] Database query optimization
- [ ] Cache layer (Redis) implementation
- [ ] Load testing with 50 tickers

---

## 9. Appendices

### Appendix A: Glossary
| Term | Definition |
|------|------------|
| ATR | Average True Range — volatility measure |
| CAGR | Compound Annual Growth Rate |
| Calmar | CAGR / Max Drawdown ratio |
| DD | Drawdown — peak-to-trough decline |
| EMA | Exponential Moving Average |
| HTF | Higher Time Frame |
| IV | Implied Volatility |
| KL Divergence | Kullback-Leibler divergence — distribution difference measure |
| LTF | Lower Time Frame |
| MAPE | Mean Absolute Percentage Error |
| OBV | On-Balance Volume |
| P/C Ratio | Put/Call Ratio |
| RS | Relative Strength |
| RSI | Relative Strength Index |
| RV | Realized Volatility |
| SMA | Simple Moving Average |
| Sortino | Sharpe ratio using downside deviation only |

### Appendix B: API Rate Limits
| Provider | Free Tier | Paid Tier |
|----------|-----------|-----------|
| Yahoo Finance | Unlimited (unofficial) | N/A |
| Polygon.io | 5 calls/min | $49/mo unlimited |
| NewsAPI | 100/day | $449/mo 1M/day |
| Finnhub | 60 calls/min | $50/mo unlimited |

### Appendix C: Mac Mini Pro Resource Estimates
| Component | Requirement | Mac Mini Pro M4 Pro |
|-----------|-------------|---------------------|
| CPU | 4 cores for parallel training | 14 cores ✅ |
| RAM | 16GB for 50 tickers | 24-32GB ✅ |
| Storage | 10GB for 3 years data | 512GB-1TB ✅ |
| Neural Engine | LSTM inference acceleration | 16-core ✅ |

### Appendix D: Success Metrics
| Metric | Target |
|--------|--------|
| Model directional accuracy | > 60% |
| Win rate (swing trades) | > 55% |
| Average R-multiple | > 1.5 |
| Profit factor | > 1.3 |
| Max drawdown | < 15% |
| Sharpe ratio | > 1.0 |
| System uptime | > 99% |
| Data pipeline success | > 99% |

---

**End of Document**

*This SRS shall be reviewed and updated at the end of each implementation phase.*
