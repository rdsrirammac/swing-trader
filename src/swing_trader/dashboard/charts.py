"""Plotly chart builders for the Streamlit dashboard (SRS DV-002, DV-003).

Every function returns a `plotly.graph_objects.Figure` and is written to
degrade gracefully on empty/missing input (returns a figure with an
informational annotation instead of raising) so a caller in
`dashboard/app.py` or `dashboard/pages/*.py` never crashes the whole page
just because one series is empty. Callers should still wrap calls in
try/except for defense in depth against unexpected input shapes.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"size": 16})
    fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, height=300)
    return fig


def _ensure_ts_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a date/timestamp column to `ts` (datetime, sorted)."""
    df = df.copy()
    for candidate in ("ts", "date", "Date", "as_of"):
        if candidate in df.columns:
            df["ts"] = pd.to_datetime(df[candidate])
            return df.sort_values("ts").reset_index(drop=True)
    # Fall back to the index if it looks datetime-like.
    if isinstance(df.index, pd.DatetimeIndex):
        df["ts"] = df.index
        return df.sort_values("ts").reset_index(drop=True)
    return df


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA20/SMA50/Bollinger/RSI14/MACD-hist/ATR14/vol-avg20 with
    plain pandas for any column not already present.

    Kept dependency-free of `swing_trader.features` (which may not be ready
    yet / may not be merged into the df the caller passes in) so the price
    chart always has something to show once raw OHLCV is available.
    """
    df = df.copy()
    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close

    if "ema_20" not in df.columns:
        df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    if "sma_50" not in df.columns:
        df["sma_50"] = close.rolling(50, min_periods=1).mean()
    if "bb_upper" not in df.columns or "bb_lower" not in df.columns:
        mid = close.rolling(20, min_periods=1).mean()
        std = close.rolling(20, min_periods=1).std().fillna(0)
        df["bb_upper"] = mid + 2 * std
        df["bb_lower"] = mid - 2 * std
    if "rsi_14" not in df.columns:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, pd.NA)
        df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50)
    if "macd_hist" not in df.columns:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = macd - signal
    if "atr_14" not in df.columns:
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        df["atr_14"] = tr.rolling(14, min_periods=1).mean()
    if "volume" in df.columns and "volume_avg_20" not in df.columns:
        df["volume_avg_20"] = df["volume"].rolling(20, min_periods=1).mean()
    return df


def price_chart(df: pd.DataFrame, trades: pd.DataFrame | None = None) -> go.Figure:
    """Candlestick + EMA20/SMA50/Bollinger overlay with Volume, RSI(14),
    MACD histogram, and ATR(14) subplots (DV-002 ticker detail, DV-003
    indicator overlays). Optional `trades` dataframe with
    entry_date/entry_price/exit_date/exit_price columns adds entry/exit
    markers to the price panel.
    """
    if df is None or df.empty:
        return _empty_figure("No price data available")

    df = _ensure_ts_column(df)
    if "close" not in df.columns or "ts" not in df.columns:
        return _empty_figure("Price dataframe missing required 'close'/'ts' columns")
    df = _compute_indicators(df)

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.42, 0.14, 0.14, 0.15, 0.15],
        vertical_spacing=0.03,
        subplot_titles=("Price", "Volume", "RSI (14)", "MACD Histogram", "ATR (14)"),
    )

    if {"open", "high", "low"}.issubset(df.columns):
        fig.add_trace(
            go.Candlestick(
                x=df["ts"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(go.Scatter(x=df["ts"], y=df["close"], name="Close", line={"color": "black"}), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["ts"], y=df["ema_20"], name="EMA20", line={"width": 1}), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["ts"], y=df["sma_50"], name="SMA50", line={"width": 1}), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df["ts"], y=df["bb_upper"], name="BB Upper", line={"width": 1, "dash": "dot"}), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["ts"], y=df["bb_lower"], name="BB Lower", line={"width": 1, "dash": "dot"}, fill="tonexty"
        ),
        row=1,
        col=1,
    )

    if trades is not None and not trades.empty:
        if "entry_date" in trades.columns and "entry_price" in trades.columns:
            entries = trades.dropna(subset=["entry_date", "entry_price"])
            if not entries.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(entries["entry_date"]),
                        y=entries["entry_price"],
                        mode="markers",
                        marker={"symbol": "triangle-up", "color": "green", "size": 12},
                        name="Entry",
                    ),
                    row=1,
                    col=1,
                )
        if "exit_date" in trades.columns and "exit_price" in trades.columns:
            exits = trades.dropna(subset=["exit_date", "exit_price"])
            if not exits.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(exits["exit_date"]),
                        y=exits["exit_price"],
                        mode="markers",
                        marker={"symbol": "triangle-down", "color": "red", "size": 12},
                        name="Exit",
                    ),
                    row=1,
                    col=1,
                )

    if "volume" in df.columns:
        fig.add_trace(go.Bar(x=df["ts"], y=df["volume"], name="Volume", marker_color="lightslategray"), row=2, col=1)
        if "volume_avg_20" in df.columns:
            fig.add_trace(
                go.Scatter(x=df["ts"], y=df["volume_avg_20"], name="Vol Avg20", line={"width": 1, "color": "orange"}),
                row=2,
                col=1,
            )

    fig.add_trace(go.Scatter(x=df["ts"], y=df["rsi_14"], name="RSI14", line={"color": "purple"}), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    macd_colors = ["green" if v >= 0 else "red" for v in df["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df["ts"], y=df["macd_hist"], name="MACD Hist", marker_color=macd_colors), row=4, col=1)

    fig.add_trace(go.Scatter(x=df["ts"], y=df["atr_14"], name="ATR14", line={"color": "brown"}), row=5, col=1)

    fig.update_layout(height=1000, showlegend=True, xaxis_rangeslider_visible=False, margin={"t": 40, "b": 20})
    return fig


def equity_curve_chart(portfolio_value_series: pd.Series) -> go.Figure:
    """Line chart of portfolio equity over time."""
    if portfolio_value_series is None or portfolio_value_series.empty:
        return _empty_figure("No equity curve data available")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=portfolio_value_series.index,
            y=portfolio_value_series.values,
            mode="lines",
            name="Equity",
            line={"color": "royalblue"},
        )
    )
    fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Portfolio Value ($)")
    return fig


def drawdown_chart(portfolio_value_series: pd.Series) -> go.Figure:
    """Underwater equity curve (drawdown %) chart."""
    if portfolio_value_series is None or portfolio_value_series.empty:
        return _empty_figure("No drawdown data available")
    running_max = portfolio_value_series.cummax()
    drawdown = (portfolio_value_series - running_max) / running_max * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            name="Drawdown %",
            line={"color": "firebrick"},
            fill="tozeroy",
        )
    )
    fig.update_layout(title="Drawdown (Underwater Curve)", xaxis_title="Date", yaxis_title="Drawdown (%)")
    return fig


def r_multiple_histogram(r_multiples: pd.Series) -> go.Figure:
    """Histogram of realized R-multiples (trade journal DV panel)."""
    if r_multiples is None or r_multiples.empty:
        return _empty_figure("No R-multiple data available")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r_multiples.values, nbinsx=30, marker_color="teal"))
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(title="R-Multiple Distribution", xaxis_title="R-Multiple", yaxis_title="Count")
    return fig


def monthly_returns_heatmap(monthly_returns: pd.DataFrame) -> go.Figure:
    """Pivot heatmap of returns: rows=year, cols=month.

    Expects `monthly_returns` with columns ['year', 'month', 'return_pct']
    (month as int 1-12), or an already-pivoted dataframe (year index, month
    columns, return % values).
    """
    if monthly_returns is None or monthly_returns.empty:
        return _empty_figure("No monthly returns data available")

    df = monthly_returns.copy()
    if {"year", "month", "return_pct"}.issubset(df.columns):
        pivot = df.pivot(index="year", columns="month", values="return_pct")
    else:
        pivot = df

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    try:
        col_labels = [month_labels[int(c) - 1] for c in pivot.columns]
    except (ValueError, TypeError, IndexError):
        col_labels = [str(c) for c in pivot.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=col_labels,
            y=[str(y) for y in pivot.index],
            colorscale="RdYlGn",
            zmid=0,
            text=pivot.values,
            texttemplate="%{text:.1f}%",
        )
    )
    fig.update_layout(title="Monthly Returns (%)", xaxis_title="Month", yaxis_title="Year")
    return fig


def win_rate_by_regime_chart(regime_stats: pd.DataFrame) -> go.Figure:
    """Bar chart of win rate by market regime (MR-004 style breakdown).

    Expects columns ['regime', 'win_rate'] (win_rate as a 0-1 fraction or a
    0-100 percentage -- auto-detected).
    """
    if regime_stats is None or regime_stats.empty:
        return _empty_figure("No regime performance data available")
    df = regime_stats.copy()
    win_rate = df["win_rate"]
    if win_rate.max() <= 1.0:
        win_rate = win_rate * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["regime"], y=win_rate, marker_color="steelblue"))
    fig.update_layout(title="Win Rate by Market Regime", xaxis_title="Regime", yaxis_title="Win Rate (%)")
    return fig
