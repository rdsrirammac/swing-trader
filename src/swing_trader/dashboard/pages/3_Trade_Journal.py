"""Trade Journal page: full trade history table + R-multiple distribution
(TE-001/002 journal, DV performance panel)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from swing_trader.db.base import session_scope
from swing_trader.db.models import Trade
from swing_trader.logging_setup import get_logger, setup_logging

try:
    from swing_trader.dashboard import charts
except ImportError:
    charts = None  # type: ignore[assignment]

setup_logging()
logger = get_logger("dashboard.pages.trade_journal")

st.set_page_config(page_title="Trade Journal", layout="wide")
st.title("Trade Journal")

try:
    with session_scope() as db:
        trades = db.query(Trade).order_by(Trade.entry_date.desc()).all()
        df = pd.DataFrame(
            [
                {
                    "ticker": t.ticker, "entry_date": t.entry_date, "entry_price": t.entry_price,
                    "shares": t.shares, "stop_loss": t.stop_loss, "target": t.target,
                    "exit_date": t.exit_date, "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason.value if t.exit_reason else None,
                    "realized_pnl": t.realized_pnl, "realized_r_multiple": t.realized_r_multiple,
                    "holding_period_days": t.holding_period_days, "is_paper": t.is_paper,
                    "thesis": t.thesis, "notes": t.notes,
                }
                for t in trades
            ]
        )

    if df.empty:
        st.info("No trades recorded yet.")
    else:
        st.subheader(f"All Trades ({len(df)})")
        st.dataframe(df, use_container_width=True, hide_index=True)

        r_multiples = df["realized_r_multiple"].dropna()
        st.subheader("R-Multiple Distribution")
        if r_multiples.empty:
            st.info("No closed trades with realized R-multiples yet.")
        elif charts is not None:
            st.plotly_chart(charts.r_multiple_histogram(r_multiples), use_container_width=True)
        else:
            st.error("swing_trader.dashboard.charts failed to import; cannot render histogram.")
except Exception as exc:
    logger.exception("Trade journal page failed")
    st.error(f"Could not load trade journal: {exc}")
