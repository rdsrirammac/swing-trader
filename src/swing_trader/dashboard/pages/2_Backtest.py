"""Backtest page: form to run a backtest via `swing_trader.backtest.engine`
against stored StockPrice/SignalRating data, then plot the resulting equity
curve and drawdown (DV backtest results panel).
"""
from __future__ import annotations

import datetime as dt
import importlib

import pandas as pd
import streamlit as st

from swing_trader.db.base import session_scope
from swing_trader.db.models import SignalRating, StockPrice
from swing_trader.logging_setup import get_logger, setup_logging

try:
    from swing_trader.dashboard import charts
except ImportError:
    charts = None  # type: ignore[assignment]

setup_logging()
logger = get_logger("dashboard.pages.backtest")

st.set_page_config(page_title="Backtest", layout="wide")
st.title("Backtest")

try:
    engine = importlib.import_module("swing_trader.backtest.engine")
except ImportError as exc:
    engine = None
    st.info(
        f"Not yet available: swing_trader.backtest.engine ({exc}). "
        "You can still fill out the form; it will run once that module lands."
    )

with st.form("backtest_form"):
    col1, col2, col3 = st.columns(3)
    start = col1.date_input("Start date", value=dt.date.today() - dt.timedelta(days=365))
    end = col2.date_input("End date", value=dt.date.today())
    strategy = col3.text_input("Strategy", value="default")
    submitted = st.form_submit_button("Run Backtest")

if submitted:
    if engine is None or not hasattr(engine, "run_backtest"):
        st.warning("swing_trader.backtest.engine.run_backtest is not available yet.")
    elif start >= end:
        st.warning("Start date must be before end date.")
    else:
        try:
            with session_scope() as db:
                prices = (
                    db.query(StockPrice)
                    .filter(StockPrice.ts >= start, StockPrice.ts <= end, StockPrice.interval == "1d")
                    .all()
                )
                signals = (
                    db.query(SignalRating).filter(SignalRating.as_of >= start, SignalRating.as_of <= end).all()
                )

                if not prices:
                    st.warning(f"No stored price data between {start} and {end}. Run backfill first.")
                else:
                    price_df = pd.DataFrame(
                        [
                            {
                                "ticker": p.ticker, "ts": p.ts, "open": p.open, "high": p.high,
                                "low": p.low, "close": p.close, "volume": p.volume,
                            }
                            for p in prices
                        ]
                    )
                    signals_df = pd.DataFrame(
                        [{"ticker": s.ticker, "as_of": s.as_of, "score": s.score, "rating": s.rating.value} for s in signals]
                    )

                    with st.spinner("Running backtest..."):
                        result = engine.run_backtest(
                            price_df=price_df, signals_df=signals_df, start=start, end=end, strategy=strategy
                        )
                    st.success("Backtest complete.")

                    equity_series = None
                    if isinstance(result, dict):
                        equity_series = result.get("equity_curve")
                        display = {
                            k: v for k, v in result.items()
                            if k not in ("equity_curve", "trades", "trade_log")
                        }
                        if display:
                            st.json(display, expanded=False)
                        trade_log = result.get("trade_log")
                        if isinstance(trade_log, pd.DataFrame) and not trade_log.empty:
                            st.dataframe(trade_log, use_container_width=True)

                    if equity_series is not None and charts is not None:
                        if not isinstance(equity_series, pd.Series):
                            equity_series = pd.Series(equity_series)
                        st.plotly_chart(charts.equity_curve_chart(equity_series), use_container_width=True)
                        st.plotly_chart(charts.drawdown_chart(equity_series), use_container_width=True)
                    elif equity_series is None:
                        st.info("Backtest result did not include an 'equity_curve' series to plot.")
        except Exception as exc:
            logger.exception("Backtest run failed")
            st.error(f"Backtest failed: {exc}")
