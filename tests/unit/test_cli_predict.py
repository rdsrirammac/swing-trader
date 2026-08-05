"""Tests for `python -m swing_trader.cli predict`'s success-reporting.

The underlying pipeline (`models.pipeline.run_daily_self_tuning_pipeline`)
deliberately swallows its own per-step exceptions and never raises -- by
design, so one bad step (e.g. candidate model training) doesn't block the
rest of a day's run. That means it can legitimately write zero `Prediction`
rows and still return normally. `predict_cmd` used to print an
unconditional "Prediction pipeline complete." regardless, which is exactly
what made a real "0 predictions written" failure indistinguishable from
success. These tests exercise the CLI's outcome-reporting logic (query
`Prediction` count after the pipeline call) using a fake, fast
`run_daily_self_tuning_pipeline` -- not the real LightGBM/Optuna pipeline.
"""
from __future__ import annotations

import datetime as dt

from click.testing import CliRunner

import swing_trader.cli as cli_module
import swing_trader.models.pipeline as pipeline_module
from swing_trader.db.models import Prediction, TickerStatus, TickerUniverse


class _PassthroughScope:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.session.commit()
        else:
            self.session.rollback()
        return False


def _seed_active_ticker(db_session, ticker="AAPL"):
    db_session.add(TickerUniverse(ticker=ticker, status=TickerStatus.ACTIVE))
    db_session.commit()


def _fake_context(as_of):
    return pipeline_module.DailyPipelineContext(
        as_of=as_of,
        model_version="lightgbm-v1",
        training_frame=None,
        feature_columns=[],
        target_column="expected_return_10d",
        target_task="regression",
    )


def test_predict_reports_zero_predictions_written(db_session, monkeypatch):
    """Simulates the exact scenario reported: the pipeline "completes" (no
    exception) but candidate model training failed internally, so
    `run_daily_self_tuning_pipeline` writes nothing. The CLI must say so,
    not print an unconditional success message.
    """
    monkeypatch.setattr(cli_module, "session_scope", lambda: _PassthroughScope(db_session))
    _seed_active_ticker(db_session, "AAPL")

    as_of = dt.date(2026, 8, 5)
    monkeypatch.setattr(pipeline_module, "build_context_from_db", lambda db, tickers: _fake_context(as_of))
    monkeypatch.setattr(pipeline_module, "run_daily_self_tuning_pipeline", lambda tickers, context: None)

    result = CliRunner().invoke(cli_module.cli, ["predict"])

    assert result.exit_code == 0, result.output
    assert "wrote 0 Prediction rows" in result.output
    assert "Candidate model training failed" in result.output or "No trained model available" in result.output


def test_predict_reports_success_count_when_predictions_written(db_session, monkeypatch):
    monkeypatch.setattr(cli_module, "session_scope", lambda: _PassthroughScope(db_session))
    _seed_active_ticker(db_session, "AAPL")

    as_of = dt.date(2026, 8, 5)

    def fake_run(tickers, context):
        # Simulates step 9 (_write_predictions) actually landing a row.
        db_session.add(Prediction(ticker="AAPL", as_of=as_of, model_version=context.model_version))

    monkeypatch.setattr(pipeline_module, "build_context_from_db", lambda db, tickers: _fake_context(as_of))
    monkeypatch.setattr(pipeline_module, "run_daily_self_tuning_pipeline", fake_run)

    result = CliRunner().invoke(cli_module.cli, ["predict"])

    assert result.exit_code == 0, result.output
    assert "1/1 ticker(s)" in result.output
    assert "wrote 0" not in result.output
