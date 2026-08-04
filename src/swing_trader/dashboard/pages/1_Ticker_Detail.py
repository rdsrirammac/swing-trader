"""Ticker Detail page (SRS DV-002/DV-003): candlestick + indicator chart
for a single selected ticker, with entry/exit markers from the trade
journal.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from swing_trader.db.base import session_scope
from swing_trader.db.models import StockPrice, TickerUniverse, Trade
from swing_trader.logging_setup import get_logger, setup_logging

try:
    from swing_trader.dashboard import charts
except ImportError:
    charts = None  # type: ignore[assignment]

setup_logging()
logger = get_logger("dashboard.pages.ticker_detail")

st.set_page_config(page_title="Ticker Detail", layout="wide")
st.title("Ticker Detail")

if charts is None:
    st.error("swing_trader.dashboard.charts failed to import; cannot render charts.")
    st.stop()

try:
    with session_scope() as db:
        tickers = [t.ticker for t in db.query(TickerUniverse).order_by(TickerUniverse.ticker).all()]
except Exception as exc:
    logger.exception("Failed to load ticker universe")
    st.error(f"Could not load ticker universe: {exc}")
    st.stop()

if not tickers:
    st.info("No tickers in the universe yet. Add one with `make add-ticker TICKER=...`.")
    st.stop()

col1, col2 = st.columns(2)
ticker = col1.selectbox("Ticker", tickers)
interval = col2.selectbox("Interval", ["1d", "30m"], index=0)

try:
    with session_scope() as db:
        rows = (
            db.query(StockPrice)
            .filter(StockPrice.ticker == ticker, StockPrice.interval == interval)
            .order_by(StockPrice.ts)
            .all()
        )
        price_df = pd.DataFrame(
            [
                {"ts": r.ts, "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume}
                for r in rows
            ]
        )

        trade_rows = db.query(Trade).filter(Trade.ticker == ticker).all()
        trades_df = pd.DataFrame(
            [
                {
                    "entry_date": t.entry_date, "entry_price": t.entry_price,
                    "exit_date": t.exit_date, "exit_price": t.exit_price,
                }
                for t in trade_rows
            ]
        )

    if price_df.empty:
        st.info(f"No stored price history for {ticker} at interval={interval}. Run `make backfill TICKER={ticker}` first.")
    else:
        fig = charts.price_chart(price_df, trades=trades_df if not trades_df.empty else None)
        st.plotly_chart(fig, use_container_width=True)
except Exception as exc:
    logger.exception("Ticker detail chart failed")
    st.error(f"Could not render chart for {ticker}: {exc}")
