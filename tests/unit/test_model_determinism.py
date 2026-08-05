"""Regression tests for prediction reproducibility (`modeling.random_seed`).

Root cause of "different predictions for the same data in multiple runs":
every `predict` call retrains a brand-new LightGBM model from scratch
(Optuna hyperparameter search via `models.tuner.tune_lightgbm`, then a
final fit in `models.pipeline._train_candidate`). `optuna.create_study()`'s
default `TPESampler` seeds itself from system entropy when no `sampler` is
passed, so it explored a different sequence of hyperparameter trials on
every call -- confirmed empirically: two unseeded `create_study()` runs
against byte-identical data produced completely different `best_params`.
Both `tune_lightgbm` (seeded `TPESampler`) and `LightGBMModel` (default
`random_state`/`deterministic=True` injection) now read a fixed seed from
`modeling.random_seed` (config/settings.yaml, default 42).

These tests use small data/trial counts to stay fast, but they exercise
the real `tune_lightgbm` and `LightGBMModel` -- not a mock -- since the
whole point is verifying actual reproducibility, not just that a seed
argument got passed somewhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

lightgbm = pytest.importorskip("lightgbm")
pytest.importorskip("optuna")


@pytest.fixture()
def synthetic_data():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(150, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.normal(size=150))
    return X, y


def test_lightgbm_model_is_deterministic_across_fits(synthetic_data):
    from swing_trader.models.base_models import LightGBMModel

    X, y = synthetic_data
    predictions = []
    for _ in range(3):
        # subsample/colsample_bytree < 1 engages LightGBM's internal bagging
        # RNG -- the actual condition under which random_state matters.
        model = LightGBMModel(
            task="regression", subsample=0.7, subsample_freq=1, colsample_bytree=0.7, verbosity=-1
        )
        model.fit(X, y)
        predictions.append(model.predict(X))

    for i in range(1, len(predictions)):
        assert np.array_equal(predictions[0], predictions[i]), (
            f"run 0 and run {i} produced different predictions from identical data -- "
            "LightGBMModel's default random_state injection isn't working"
        )


def test_tune_lightgbm_is_deterministic_across_calls(synthetic_data):
    from swing_trader.models.tuner import tune_lightgbm

    X, y = synthetic_data
    results = [tune_lightgbm(X, y, task="regression", n_trials=5) for _ in range(2)]

    assert results[0] == results[1], (
        f"tune_lightgbm returned different best_params across two calls on identical data:\n"
        f"  run 0: {results[0]}\n  run 1: {results[1]}\n"
        "Optuna's TPESampler seeding isn't working."
    )
