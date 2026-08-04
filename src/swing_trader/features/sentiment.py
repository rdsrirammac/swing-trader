"""News/analyst/options sentiment features (SRS FE-004, DC-003).

`SentimentScorer` prefers a FinBERT (ProsusAI/finbert) HuggingFace pipeline
for headline scoring, and gracefully falls back to VADER
(`vaderSentiment`) when `transformers`/`torch`/the model weights are
unavailable (e.g. no internet on first run, no torch installed on the Mac
Mini). This keeps the pipeline importable and runnable in constrained
environments per NFR guidance to degrade gracefully.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from swing_trader.logging_setup import get_logger

logger = get_logger("features.sentiment")


_SCORER_SINGLETON: "SentimentScorer | None" = None


def get_scorer() -> "SentimentScorer":
    """Process-wide lazy singleton so callers (e.g. the feature-engineering
    orchestrator) don't reload FinBERT/VADER on every ticker.
    """
    global _SCORER_SINGLETON
    if _SCORER_SINGLETON is None:
        _SCORER_SINGLETON = SentimentScorer()
    return _SCORER_SINGLETON


class SentimentScorer:
    """Headline sentiment scorer with FinBERT primary / VADER fallback."""

    def __init__(self) -> None:
        self._backend = "none"
        self._finbert_pipeline = None
        self._vader = None
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            from transformers import pipeline  # type: ignore

            self._finbert_pipeline = pipeline(
                "sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert"
            )
            self._backend = "finbert"
            logger.info("SentimentScorer: loaded FinBERT pipeline")
            return
        except Exception as e:
            logger.warning("SentimentScorer: FinBERT unavailable (%s); falling back to VADER", e)

        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self._vader = SentimentIntensityAnalyzer()
            self._backend = "vader"
            logger.info("SentimentScorer: loaded VADER fallback")
        except Exception as e:
            logger.error("SentimentScorer: no sentiment backend available (%s)", e)
            self._backend = "none"

    def score(self, text: str) -> dict:
        """Score a single headline/text. Returns {label, score, confidence}.

        `score` is signed in [-1, 1] (negative -> negative sentiment).
        `confidence` is the backend's confidence in [0, 1].
        """
        if not text:
            return {"label": "neutral", "score": 0.0, "confidence": 0.0}

        if self._backend == "finbert" and self._finbert_pipeline is not None:
            try:
                result = self._finbert_pipeline(text[:512])[0]
                label = str(result.get("label", "neutral")).lower()
                confidence = float(result.get("score", 0.0))
                if label == "positive":
                    signed = confidence
                elif label == "negative":
                    signed = -confidence
                else:
                    label = "neutral"
                    signed = 0.0
                return {"label": label, "score": float(signed), "confidence": confidence}
            except Exception as e:
                logger.warning("FinBERT scoring failed, falling back to VADER for this call: %s", e)

        if self._vader is not None:
            try:
                scores = self._vader.polarity_scores(text)
                compound = float(scores.get("compound", 0.0))
                if compound >= 0.05:
                    label = "positive"
                elif compound <= -0.05:
                    label = "negative"
                else:
                    label = "neutral"
                return {"label": label, "score": compound, "confidence": abs(compound)}
            except Exception as e:
                logger.error("VADER scoring failed: %s", e)

        return {"label": "neutral", "score": 0.0, "confidence": 0.0}

    def aggregate_news_features(self, news_rows: list[dict]) -> dict:
        """Compute news_sentiment_3d_avg and news_volume_velocity from scored
        news rows.

        Each row must have `published_at` (datetime or ISO string) and
        `sentiment_score` (float in [-1, 1], typically produced by `score()`).
        """
        result = {"news_sentiment_3d_avg": None, "news_volume_velocity": None}
        if not news_rows:
            return result

        try:
            df = pd.DataFrame(news_rows)
            if "published_at" not in df.columns:
                return result
            df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
            df = df.dropna(subset=["published_at"])
            if df.empty:
                return result

            now = df["published_at"].max()
            last_3d = df[df["published_at"] >= now - pd.Timedelta(days=3)]
            prior_3d = df[
                (df["published_at"] < now - pd.Timedelta(days=3))
                & (df["published_at"] >= now - pd.Timedelta(days=6))
            ]

            if "sentiment_score" in last_3d.columns and not last_3d.empty:
                avg = last_3d["sentiment_score"].dropna().mean()
                if pd.notna(avg):
                    result["news_sentiment_3d_avg"] = float(avg)

            last_count = len(last_3d)
            prior_count = len(prior_3d)
            if prior_count > 0:
                result["news_volume_velocity"] = float((last_count - prior_count) / prior_count)
            elif last_count > 0:
                # No prior baseline but articles exist now -> treat as maximal velocity.
                result["news_volume_velocity"] = float(last_count)
            else:
                result["news_volume_velocity"] = 0.0
        except Exception as e:
            logger.warning("aggregate_news_features failed: %s", e)

        return result


def analyst_rating_velocity(recommendations_df: Any) -> float:
    """(upgrades - downgrades) over the trailing 30 days from a yfinance
    recommendations dataframe. yfinance's schema for this endpoint changes
    frequently across versions, so this is intentionally best-effort and
    returns 0.0 (neutral) on any failure rather than raising.
    """
    try:
        if recommendations_df is None or len(recommendations_df) == 0:
            return 0.0

        df = recommendations_df.copy()

        # Normalize a date/index column.
        if not isinstance(df.index, pd.DatetimeIndex):
            date_col = next(
                (c for c in df.columns if str(c).lower() in ("date", "gradedate")), None
            )
            if date_col is not None:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
                df = df.dropna(subset=[date_col]).set_index(date_col)
            else:
                # No usable date info; cannot restrict to trailing 30 days.
                pass

        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=30)
            else:
                cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=30)
            df = df[df.index >= cutoff]

        action_col = next(
            (c for c in df.columns if str(c).lower() in ("action", "to grade", "tograde")), None
        )
        if action_col is None:
            return 0.0

        actions = df[action_col].astype(str).str.lower()
        upgrades = actions.str.contains("up").sum()
        downgrades = actions.str.contains("down").sum()
        return float(upgrades - downgrades)
    except Exception as e:
        logger.warning("analyst_rating_velocity failed: %s", e)
        return 0.0


def options_put_call_skew(calls_df: Any, puts_df: Any) -> float | None:
    """put volume / call volume ratio minus 1, from an options chain's
    .calls / .puts dataframes. Returns None if volumes are unavailable.
    """
    try:
        if calls_df is None or puts_df is None:
            return None
        if "volume" not in calls_df.columns or "volume" not in puts_df.columns:
            return None
        call_vol = calls_df["volume"].fillna(0).sum()
        put_vol = puts_df["volume"].fillna(0).sum()
        if call_vol <= 0:
            return None
        return float(put_vol / call_vol - 1.0)
    except Exception as e:
        logger.warning("options_put_call_skew failed: %s", e)
        return None
