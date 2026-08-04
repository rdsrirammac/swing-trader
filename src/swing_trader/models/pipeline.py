"""Daily self-tuning predictive-modeling pipeline (SRS PM-004).

`run_daily_self_tuning_pipeline` is the full daily regime + drift + retrain
+ predict cycle described in SRS 3.6 step-by-step:

  1. INGEST            -- NOT done here. See `DailyPipelineContext` below:
                           this function's job is regime detection, drift
                           checking, tuning, training, feature selection,
                           and prediction -- not fetching data. The caller
                           (an orchestrator in `swing_trader.data` /
                           `swing_trader.features`) pre-populates a
                           `DailyPipelineContext` with already-collected
                           price/feature history and hands it in.
  2. REGIME UPDATE      -- `regime_detector.classify_regime` + `record_regime`.
  3. DRIFT CHECK        -- KL-divergence of today's feature distribution vs
                           trailing-30-day average, per numeric feature.
  4. WALK-FORWARD VALIDATION -- rolling `TimeSeriesSplit` scoring of a
                           freshly-trained baseline over the last
                           `modeling.walk_forward_test_days`.
  5. HYPERPARAMETER SWEEP -- `tuner.tune_lightgbm`.
  6. MODEL SELECTION    -- deploy the new model only if its MAPE improves on
                           the currently-deployed `ModelPerformance` row by
                           more than `modeling.mape_improvement_threshold`.
  7. FEATURE SELECTION  -- drop features whose normalized LightGBM
                           importance falls below `modeling.feature_importance_floor`.
  8. THRESHOLD TUNE     -- best-effort call into `swing_trader.signals.rating`
                           (skipped gracefully if that sibling module isn't
                           built yet -- it's being written concurrently).
  9. PREDICT            -- write `Prediction` rows per ticker.
  10. LOG               -- write a `ModelPerformance` row with the MLflow run id.

Every external/cross-module call is defensive (try/except + log) because
sibling modules are being written concurrently by other agents and may not
exist, or may not match this exact interface, at any given time. This file
must remain syntactically valid and importable standalone regardless.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sqlalchemy import select

from swing_trader.config import get_settings
from swing_trader.db.base import session_scope
from swing_trader.db.models import ModelPerformance, Prediction, RegimeType
from swing_trader.logging_setup import get_logger

logger = get_logger("models.pipeline")


@dataclass
class DailyPipelineContext:
    """Pre-fetched inputs the daily self-tuning pipeline needs.

    Data collection/backfill/feature computation are owned by other modules
    (`swing_trader.data`, `swing_trader.features`); this dataclass is the
    hand-off point so `pipeline.py` never calls yfinance or re-derives
    features itself (step 1, INGEST, is satisfied by the caller building
    this object).
    """

    as_of: dt.date
    model_version: str
    # Panel training data: one row per (ticker, date), sorted by date
    # ascending, with a `ts` (date) column, `feature_columns`, and
    # `target_column`.
    training_frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    target_task: Literal["classification", "regression"]
    # ticker -> single-ticker feature frame (feature_columns present),
    # most-recent row last; used to build today's Prediction rows.
    latest_features: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Regime-classifier inputs (already computed upstream from SPY/VIX/etc).
    spy_adx: float | None = None
    vix_level: float | None = None
    sector_breadth_pct: float | None = None
    bb_width_pct: float | None = None
    atr_expansion_pct: float | None = None
    pct_sp500_reporting_next_2wk: float | None = None
    advance_decline_up_volume_pct: float | None = None


_FEATURE_COLUMN_NAMES = [
    "rsi_2", "rsi_14", "macd", "macd_signal", "macd_hist", "atr_14", "bb_upper", "bb_lower",
    "bb_bandwidth", "ema_20", "sma_50", "adx_14", "roc_5", "roc_10", "roc_21", "obv",
    "volume_ratio_20d", "ret_5d_vs_spy", "ret_10d_vs_spy", "ret_21d_vs_spy",
    "ret_5d_vs_sector", "ret_10d_vs_sector", "ret_21d_vs_sector", "rs_rating",
    "realized_vol_20d", "realized_vol_pctile", "atr_pct", "hv_iv_spread",
    "news_sentiment_3d_avg", "news_volume_velocity", "analyst_rating_velocity",
    "options_put_call_skew", "pe_percentile_sector", "pe_percentile_history",
    "short_interest_pct_float", "vix_level", "vix_percentile", "sector_breadth_pct",
    "yield_curve_10y_2y",
]


def build_context_from_db(
    session,
    tickers: list[str],
    as_of: dt.date | None = None,
    model_version: str = "lightgbm-v1",
    target_column: str = "expected_return_10d",
    target_horizon_days: int = 10,
    target_task: Literal["classification", "regression"] = "regression",
) -> "DailyPipelineContext | None":
    """Integration helper: assemble a `DailyPipelineContext` purely from
    what's already in the database (`StockFeature` + `StockPrice`), so CLI
    commands and scheduled jobs can call
    `run_daily_self_tuning_pipeline(tickers, build_context_from_db(db, tickers))`
    without needing their own data-assembly logic.

    This is a pragmatic, best-effort implementation added during final
    integration (not part of the original per-module build) -- it computes
    the regression target `expected_return_10d` (or a classification target
    like `prob_5pct_up_10d` if `target_task="classification"`, in which case
    `target_column` should be one of the boolean-derived columns below) as
    the forward N-day return computed from stored daily closes, joined
    against the feature history. Tickers with insufficient history are
    skipped with a warning rather than failing the whole build.

    Regime-classifier inputs (spy_adx, vix_level, sector_breadth_pct,
    bb_width_pct, atr_expansion_pct) are pulled from SPY's own
    `StockFeature` row if SPY is itself a tracked ticker (its per-ticker
    `adx_14`/`bb_bandwidth`/`atr_pct` double as SPY-level regime inputs in
    that case); if SPY isn't tracked, those fields are left None and
    `regime_detector.classify_regime` degrades to its documented default
    ordering. This is a known simplification -- a dedicated `MarketSnapshot`
    table (SPY/VIX-specific, independent of any single traded ticker) is a
    reasonable ROADMAP follow-up.
    """
    from sqlalchemy import select

    from swing_trader.db.models import StockFeature, StockPrice

    if not tickers:
        logger.warning("build_context_from_db called with no tickers")
        return None

    as_of = as_of or dt.date.today()
    frames: list[pd.DataFrame] = []
    latest_features: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        feat_rows = session.execute(
            select(StockFeature).where(StockFeature.ticker == ticker).order_by(StockFeature.ts)
        ).scalars().all()
        price_rows = session.execute(
            select(StockPrice)
            .where(StockPrice.ticker == ticker, StockPrice.interval == "1d")
            .order_by(StockPrice.ts)
        ).scalars().all()

        if not feat_rows or not price_rows:
            logger.info("[%s] insufficient feature/price history; skipping in training frame", ticker)
            continue

        feat_df = pd.DataFrame(
            [{**{c: getattr(r, c, None) for c in _FEATURE_COLUMN_NAMES}, "ts": r.ts} for r in feat_rows]
        )
        price_df = pd.DataFrame([{"ts": r.ts.date() if hasattr(r.ts, "date") else r.ts, "close": r.close} for r in price_rows])
        price_df = price_df.drop_duplicates(subset="ts").sort_values("ts").set_index("ts")

        # Forward N-day return target, computed from closes and aligned back onto feature rows.
        fwd_close = price_df["close"].shift(-target_horizon_days)
        fwd_return = (fwd_close - price_df["close"]) / price_df["close"]
        target_by_date = fwd_return.to_dict()

        feat_df["ticker"] = ticker
        feat_df[target_column] = feat_df["ts"].map(target_by_date)

        if target_task == "classification":
            # Interpret target_column as a "prob_Xpct_up_Nd"-style boolean target
            # derived from the same forward return, e.g. prob_5pct_up_10d -> return >= 0.05.
            try:
                pct_threshold = float(target_column.split("_")[1].replace("pct", "")) / 100.0
                feat_df[target_column] = feat_df["ts"].map(target_by_date).apply(
                    lambda r: (r >= pct_threshold) if pd.notna(r) else np.nan
                )
            except Exception:
                logger.warning(
                    "Could not parse classification threshold from target_column=%s; "
                    "falling back to regression-style raw forward return",
                    target_column,
                )

        frames.append(feat_df)
        latest_features[ticker] = feat_df[feat_df["ts"] <= as_of].tail(1)

    if not frames:
        logger.warning("build_context_from_db: no tickers had enough history to build a training frame")
        return None

    training_frame = pd.concat(frames, ignore_index=True)

    spy_adx = vix_level = sector_breadth_pct = bb_width_pct = atr_expansion_pct = None
    if "SPY" in latest_features and not latest_features["SPY"].empty:
        spy_row = latest_features["SPY"].iloc[-1]
        spy_adx = spy_row.get("adx_14")
        bb_width_pct = spy_row.get("bb_bandwidth")
        vix_level = spy_row.get("vix_level")
        sector_breadth_pct = spy_row.get("sector_breadth_pct")
    elif latest_features:
        # Fall back to any ticker's shared macro fields (vix_level/sector_breadth_pct
        # are macro-wide values duplicated onto every ticker's feature row).
        any_row = next(iter(latest_features.values()))
        if not any_row.empty:
            r = any_row.iloc[-1]
            vix_level = r.get("vix_level")
            sector_breadth_pct = r.get("sector_breadth_pct")

    return DailyPipelineContext(
        as_of=as_of,
        model_version=model_version,
        training_frame=training_frame,
        feature_columns=_FEATURE_COLUMN_NAMES,
        target_column=target_column,
        target_task=target_task,
        latest_features=latest_features,
        spy_adx=spy_adx,
        vix_level=vix_level,
        sector_breadth_pct=sector_breadth_pct,
        bb_width_pct=bb_width_pct,
        atr_expansion_pct=atr_expansion_pct,
    )


def run_daily_self_tuning_pipeline(tickers: list[str], context: DailyPipelineContext) -> None:
    """Run the PM-004 daily self-tuning cycle for `tickers` using the
    already-fetched data in `context`. Writes `RegimeHistory`, `Prediction`,
    and `ModelPerformance` rows. Returns None; failures in individual steps
    are logged and the pipeline continues in a degraded mode rather than
    raising, so a single bad step doesn't block the rest of the day's run.
    """
    if context is None:
        logger.error("run_daily_self_tuning_pipeline called without a DailyPipelineContext; nothing to do")
        return

    logger.info(
        "Starting daily self-tuning pipeline for %d tickers as_of=%s (model_version=%s)",
        len(tickers), context.as_of, context.model_version,
    )
    settings = get_settings()

    with session_scope() as db:
        # --- 2. REGIME UPDATE ---------------------------------------------
        regime = RegimeType.WEAK_TREND
        try:
            from swing_trader.models.regime_detector import (
                classify_regime,
                detect_transition,
                record_regime,
            )

            regime = classify_regime(
                spy_adx=context.spy_adx,
                vix=context.vix_level,
                sector_breadth_pct=context.sector_breadth_pct,
                bb_width_pct=context.bb_width_pct,
                atr_expansion_pct=context.atr_expansion_pct,
                pct_sp500_reporting_next_2wk=context.pct_sp500_reporting_next_2wk,
            )

            transitioned, reason = False, None
            try:
                transitioned, reason = detect_transition(
                    db, context.as_of, advance_decline_up_volume_pct=context.advance_decline_up_volume_pct
                )
            except Exception as e:
                logger.warning("detect_transition failed: %s", e)

            record_regime(
                db,
                as_of=context.as_of,
                regime=regime,
                vix=context.vix_level,
                spy_adx=context.spy_adx,
                sector_breadth_pct=context.sector_breadth_pct,
                transition_flag=transitioned,
                transition_reason=reason,
            )
            logger.info("Regime classified as %s (transition=%s, reason=%s)", regime.value, transitioned, reason)
        except Exception as e:
            logger.error("Regime update step failed, continuing with default regime: %s", e)

        # --- 3. DRIFT CHECK -------------------------------------------------
        lookback_days = settings.get("modeling.walk_forward_test_days", 60)
        try:
            drift_threshold = settings.get("modeling.drift_kl_threshold", 0.5)
            regime_change_suspected = _check_feature_drift(
                context.training_frame, context.feature_columns, context.as_of, drift_threshold
            )
            if regime_change_suspected:
                lookback_days = max(5, lookback_days // 2)
                logger.warning(
                    "Feature drift detected (KL > %.2f on >=1 feature); shortening lookback window to %d days",
                    drift_threshold, lookback_days,
                )
        except Exception as e:
            logger.warning("Drift check step failed, using default lookback_days=%d: %s", lookback_days, e)

        # --- 4. WALK-FORWARD VALIDATION -------------------------------------
        try:
            current_score = _walk_forward_score(context, lookback_days)
            logger.info("Walk-forward validation score (MAPE, lower is better): %s", current_score)
        except Exception as e:
            logger.warning("Walk-forward validation step failed: %s", e)

        # --- 5. HYPERPARAMETER SWEEP -----------------------------------------
        best_params: dict = {}
        try:
            from swing_trader.models.tuner import tune_lightgbm

            n_trials = settings.get("modeling.optuna_trials", 50)
            tune_df = context.training_frame.dropna(subset=[context.target_column]).tail(
                max(lookback_days * max(1, len(tickers)), 100)
            )
            if len(tune_df) >= 30:
                X_tune = tune_df[context.feature_columns].fillna(0.0)
                y_tune = tune_df[context.target_column]
                best_params = tune_lightgbm(X_tune, y_tune, task=context.target_task, n_trials=n_trials)
                logger.info("Optuna hyperparameter sweep complete: %s", best_params)
            else:
                logger.warning("Not enough rows (%d) for hyperparameter sweep; using default params", len(tune_df))
        except Exception as e:
            logger.warning("Hyperparameter sweep step failed, using default params: %s", e)

        # --- Train the candidate model with the swept (or default) params ---
        new_model = None
        new_mape: float | None = None
        try:
            new_model, new_mape = _train_candidate(context, best_params)
        except Exception as e:
            logger.warning("Candidate model training failed: %s", e)

        # --- 6. MODEL SELECTION ----------------------------------------------
        deployed = False
        try:
            deployed = _select_model(db, context.model_version, new_mape, settings)
        except Exception as e:
            logger.warning("Model selection step failed: %s", e)

        # --- 7. FEATURE SELECTION ---------------------------------------------
        selected_features = context.feature_columns
        try:
            if new_model is not None:
                selected_features = _select_features(new_model, context.feature_columns, settings)
                logger.info(
                    "Feature selection kept %d/%d features (importance floor=%.4f)",
                    len(selected_features), len(context.feature_columns),
                    settings.get("modeling.feature_importance_floor", 0.005),
                )
        except Exception as e:
            logger.warning("Feature selection step failed, keeping all features: %s", e)

        # --- 8. THRESHOLD TUNE ---------------------------------------------------
        try:
            from swing_trader.signals import rating  # sibling module, built concurrently

            if hasattr(rating, "tune_thresholds"):
                rating.tune_thresholds(db, context.as_of)
                logger.info("Rating thresholds tuned via swing_trader.signals.rating.tune_thresholds")
            else:
                logger.info(
                    "swing_trader.signals.rating exists but has no tune_thresholds(); skipping threshold tune"
                )
        except ImportError as e:
            logger.warning(
                "swing_trader.signals.rating not available yet (built concurrently); skipping threshold tune: %s", e
            )
        except Exception as e:
            logger.warning("Threshold tune step failed: %s", e)

        # --- 9. PREDICT -----------------------------------------------------------
        try:
            _write_predictions(db, tickers, context, new_model, regime)
        except Exception as e:
            logger.error("Prediction step failed: %s", e)

        # --- 10. LOG ----------------------------------------------------------------
        try:
            from swing_trader.models.tracking import log_model_run

            run_id = log_model_run(
                model_version=context.model_version,
                params=best_params,
                metrics={"mape": new_mape} if new_mape is not None else {},
                model=new_model,
            )
            perf = ModelPerformance(
                model_version=context.model_version,
                as_of=context.as_of,
                mape=new_mape,
                deployed=deployed,
                mlflow_run_id=run_id,
            )
            db.add(perf)
            logger.info(
                "Logged ModelPerformance: version=%s mape=%s deployed=%s mlflow_run_id=%s",
                context.model_version, new_mape, deployed, run_id,
            )
        except Exception as e:
            logger.error("Model performance logging step failed: %s", e)

    logger.info("Daily self-tuning pipeline complete for as_of=%s", context.as_of)


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _check_feature_drift(
    training_frame: pd.DataFrame, feature_columns: list[str], as_of: dt.date, threshold: float
) -> bool:
    """KL-divergence drift check: today's per-feature distribution vs the
    trailing-30-day average, using 10-bin histograms with add-one smoothing
    (avoids zero-probability blowups in `scipy.stats.entropy`). Flags drift
    (returns True) if ANY feature's KL divergence exceeds `threshold`.
    """
    from scipy.stats import entropy

    if training_frame is None or training_frame.empty or "ts" not in training_frame.columns:
        return False

    today_rows = training_frame[training_frame["ts"] == as_of]
    trailing_rows = training_frame[
        (training_frame["ts"] < as_of) & (training_frame["ts"] >= as_of - dt.timedelta(days=30))
    ]
    if today_rows.empty or trailing_rows.empty:
        return False

    n_bins = 10
    flagged = False
    for col in feature_columns:
        if col not in training_frame.columns:
            continue
        today_vals = today_rows[col].dropna()
        trailing_vals = trailing_rows[col].dropna()
        if len(today_vals) < 2 or len(trailing_vals) < 2:
            continue
        try:
            combined = pd.concat([today_vals, trailing_vals])
            if combined.nunique() < 2:
                continue
            bins = np.histogram_bin_edges(combined, bins=n_bins)
            today_hist, _ = np.histogram(today_vals, bins=bins)
            trailing_hist, _ = np.histogram(trailing_vals, bins=bins)
            today_dist = (today_hist + 1) / (today_hist.sum() + n_bins)
            trailing_dist = (trailing_hist + 1) / (trailing_hist.sum() + n_bins)
            kl = float(entropy(today_dist, trailing_dist))
            if kl > threshold:
                logger.warning("Feature drift on '%s': KL=%.3f > threshold %.3f", col, kl, threshold)
                flagged = True
        except Exception as e:
            logger.debug("Drift check skipped for column '%s': %s", col, e)

    return flagged


def _walk_forward_score(context: DailyPipelineContext, lookback_days: int) -> float | None:
    """Rolling walk-forward score of a freshly-trained baseline model over
    the trailing `lookback_days` window, using `TimeSeriesSplit` (3 folds).
    This is a self-contained proxy for "current model" performance (the
    pipeline doesn't load a persisted deployed model from disk here) --
    documented simplification; `_select_model` is what actually compares
    against the persisted `ModelPerformance.deployed` row.
    """
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.model_selection import TimeSeriesSplit

    from swing_trader.models.base_models import LightGBMModel

    df = context.training_frame.dropna(subset=[context.target_column]).tail(max(lookback_days * 5, 60))
    if len(df) < 30:
        return None

    X = df[context.feature_columns].fillna(0.0)
    y = df[context.target_column]

    n_splits = min(3, max(2, len(df) // 20))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []

    for train_idx, test_idx in tscv.split(X):
        model = LightGBMModel(task=context.target_task, verbosity=-1)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        if context.target_task == "regression":
            scores.append(float(mean_absolute_percentage_error(y.iloc[test_idx], preds)))

    return float(np.mean(scores)) if scores else None


def _train_candidate(context: DailyPipelineContext, best_params: dict):
    """Train the candidate LightGBM model on the full training frame."""
    from swing_trader.models.base_models import LightGBMModel

    train_df = context.training_frame.dropna(subset=[context.target_column])
    if train_df.empty:
        return None, None

    X_all = train_df[context.feature_columns].fillna(0.0)
    y_all = train_df[context.target_column]

    model = LightGBMModel(task=context.target_task, **{**best_params, "verbosity": -1})
    model.fit(X_all, y_all)

    mape = None
    if context.target_task == "regression":
        from sklearn.metrics import mean_absolute_percentage_error

        preds = model.predict(X_all)
        mape = float(mean_absolute_percentage_error(y_all, preds))

    return model, mape


def _select_model(db, model_version: str, new_mape: float | None, settings) -> bool:
    """MODEL SELECTION (step 6): deploy the new model if it improves MAPE
    over the currently-deployed row for this model_version family by more
    than `modeling.mape_improvement_threshold`. Returns True if the new
    model should be marked deployed.
    """
    if new_mape is None:
        logger.info("No MAPE available for candidate model (non-regression target or training failure); not deploying")
        return False

    threshold = settings.get("modeling.mape_improvement_threshold", 0.02)

    current = db.execute(
        select(ModelPerformance)
        .where(ModelPerformance.model_version == model_version, ModelPerformance.deployed.is_(True))
        .order_by(ModelPerformance.as_of.desc())
    ).scalars().first()

    if current is None or current.mape is None:
        logger.info("No currently-deployed model for %s; deploying new model (MAPE=%.4f)", model_version, new_mape)
        return True

    improvement = current.mape - new_mape  # lower MAPE is better
    if improvement > threshold:
        current.deployed = False
        logger.info(
            "New model improves MAPE by %.4f (> threshold %.4f); deploying (old=%.4f new=%.4f)",
            improvement, threshold, current.mape, new_mape,
        )
        return True

    logger.warning(
        "New model MAPE %.4f does not beat deployed %.4f by threshold %.4f; keeping current model deployed "
        "(degradation/no-improvement)",
        new_mape, current.mape, threshold,
    )
    return False


def _select_features(model, feature_columns: list[str], settings) -> list[str]:
    """FEATURE SELECTION (step 7): drop features whose normalized LightGBM
    importance is below `modeling.feature_importance_floor`. Never returns
    an empty list (falls back to the full feature set if everything would
    be dropped, e.g. due to a degenerate/untrained model).
    """
    floor = settings.get("modeling.feature_importance_floor", 0.005)

    importance_fn = getattr(model, "feature_importance", None)
    if importance_fn is None:
        return feature_columns

    series = importance_fn()
    if series is None or series.empty:
        return feature_columns

    total = series.sum()
    normalized = series / total if total else series

    kept = [f for f in feature_columns if normalized.get(f, 0.0) >= floor]
    return kept or feature_columns


def _write_predictions(db, tickers: list[str], context: DailyPipelineContext, model, regime: RegimeType) -> None:
    """PREDICT (step 9): write one `Prediction` row per ticker for
    `context.as_of`, using the latest feature row available for that ticker.
    """
    if model is None:
        logger.warning("No trained model available; skipping prediction writes for all tickers")
        return

    for ticker in tickers:
        try:
            latest = context.latest_features.get(ticker)
            if latest is None or latest.empty:
                logger.debug("[%s] no latest feature row available; skipping prediction", ticker)
                continue

            missing_cols = [c for c in context.feature_columns if c not in latest.columns]
            if missing_cols:
                logger.debug("[%s] latest feature row missing columns %s; skipping prediction", ticker, missing_cols)
                continue

            X = latest[context.feature_columns].fillna(0.0).tail(1)

            existing = db.execute(
                select(Prediction).where(
                    Prediction.ticker == ticker,
                    Prediction.as_of == context.as_of,
                    Prediction.model_version == context.model_version,
                )
            ).scalar_one_or_none()

            row = existing or Prediction(ticker=ticker, as_of=context.as_of, model_version=context.model_version)
            row.regime = regime

            if not hasattr(Prediction, context.target_column):
                logger.warning(
                    "[%s] target_column '%s' is not a Prediction column; skipping value assignment",
                    ticker, context.target_column,
                )
            elif context.target_task == "classification" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                proba_arr = np.asarray(proba)
                prob_positive = (
                    float(proba_arr[0][1]) if proba_arr.ndim == 2 and proba_arr.shape[1] > 1 else float(proba_arr[0])
                )
                setattr(row, context.target_column, prob_positive)
            else:
                pred_value = float(np.asarray(model.predict(X))[0])
                setattr(row, context.target_column, pred_value)

            if existing is None:
                db.add(row)
        except Exception as e:
            logger.warning("[%s] prediction write failed: %s", ticker, e)
