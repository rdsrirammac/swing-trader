"""Stacking ensemble meta-learner (SRS PM-003).

Combines fitted base models (`base_models.py`) via a linear (Ridge, default)
or gradient-boosted (XGBoost) meta-learner trained on out-of-fold base-model
predictions, with a bootstrap-based prediction interval.
"""
from __future__ import annotations

import copy
import datetime as dt

import numpy as np
import pandas as pd

from swing_trader.logging_setup import get_logger

logger = get_logger("models.ensemble")


class StackingEnsemble:
    """Meta-learner over a set of already-fitted base models.

    `base_models` is kept for reference/traceability (e.g. so callers know
    which models produced the columns of `base_predictions`); the ensemble
    itself only ever operates on the base models' *output* predictions
    (`base_predictions`, a DataFrame of one column per base model), not the
    raw feature matrix.
    """

    def __init__(
        self,
        base_models: list | None = None,
        meta_learner=None,
        use_xgboost_meta: bool = False,
        n_folds: int = 5,
        random_state: int = 42,
    ):
        self.base_models = base_models or []
        self.n_folds = n_folds
        self.random_state = random_state
        self.feature_names_: list[str] | None = None
        self.residuals_: np.ndarray | None = None
        self.last_trained: dt.datetime | None = None

        if meta_learner is not None:
            self.meta_learner = meta_learner
        elif use_xgboost_meta:
            from xgboost import XGBRegressor

            self.meta_learner = XGBRegressor(n_estimators=100, max_depth=3, random_state=random_state)
        else:
            from sklearn.linear_model import Ridge

            self.meta_learner = Ridge()

    def fit(self, base_predictions: pd.DataFrame, y: pd.Series) -> "StackingEnsemble":
        """Fit the meta-learner.

        Uses k-fold (k=`n_folds`) cross-validation over `base_predictions`/`y`
        to produce out-of-fold predictions purely to obtain unbiased
        residuals for `predict_with_ci`'s bootstrap interval; the deployed
        meta-learner itself is then refit on the full dataset (standard
        stacking practice -- OOF folds estimate generalization error, the
        final model uses all available data).
        """
        from sklearn.model_selection import KFold

        self.feature_names_ = list(base_predictions.columns)
        X_arr = base_predictions.to_numpy()
        y_arr = np.asarray(y)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros(len(X_arr))

        for train_idx, val_idx in kf.split(X_arr):
            fold_model = copy.deepcopy(self.meta_learner)
            fold_model.fit(X_arr[train_idx], y_arr[train_idx])
            oof_preds[val_idx] = fold_model.predict(X_arr[val_idx])

        self.meta_learner.fit(X_arr, y_arr)
        self.residuals_ = y_arr - oof_preds
        self.last_trained = dt.datetime.utcnow()
        return self

    def predict(self, base_predictions: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.meta_learner.predict(base_predictions.to_numpy()))

    def predict_with_ci(
        self, base_predictions: pd.DataFrame, confidence: float = 0.90, n_bootstrap: int = 1000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (point_estimate, lower, upper) prediction bounds.

        Approach: bootstrap-resample the out-of-fold training residuals
        (from `.fit`) `n_bootstrap` times, take the mean of each resample,
        then use the `(1-confidence)/2` / `1-(1-confidence)/2` percentiles of
        that bootstrap distribution as symmetric offsets applied to every
        point estimate. This assumes residual variance is roughly constant
        across the prediction range (homoscedastic) -- a pragmatic
        simplification for a fast on-device pipeline, not a rigorous
        conformal-prediction guarantee.
        """
        point = self.predict(base_predictions)

        if self.residuals_ is None or len(self.residuals_) == 0:
            logger.warning("predict_with_ci called before fit (or with no residuals); returning point estimate as bounds")
            return point, point.copy(), point.copy()

        rng = np.random.default_rng(self.random_state)
        boot_means = np.array(
            [
                rng.choice(self.residuals_, size=len(self.residuals_), replace=True).mean()
                for _ in range(n_bootstrap)
            ]
        )
        alpha = 1.0 - confidence
        lower_offset = np.percentile(boot_means, 100 * (alpha / 2))
        upper_offset = np.percentile(boot_means, 100 * (1 - alpha / 2))

        lower = point + lower_offset
        upper = point + upper_offset
        return point, lower, upper

    @staticmethod
    def needs_retrain(last_trained: dt.datetime | None) -> bool:
        """True if the meta-learner should be retrained (weekly, expanding
        window per SRS PM-003): no training timestamp, or >7 days old.
        """
        if last_trained is None:
            return True
        return (dt.datetime.utcnow() - last_trained) > dt.timedelta(days=7)
