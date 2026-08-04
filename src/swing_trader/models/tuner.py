"""LightGBM hyperparameter tuning via Optuna (SRS PM-004 step 5).

Uses time-series-aware cross-validation (`TimeSeriesSplit`, NOT random
k-fold) since swing-trading feature panels are temporally ordered and random
folds would leak future information into training folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_trader.logging_setup import get_logger

logger = get_logger("models.tuner")


def tune_lightgbm(X: pd.DataFrame, y: pd.Series, task: str, n_trials: int = 50) -> dict:
    """Search LightGBM hyperparameters with Optuna, scored via
    `TimeSeriesSplit` cross-validation (5 splits). Returns `study.best_params`.

    `task` == "classification": scored by mean CV ROC-AUC (maximized).
    `task` == "regression": scored by mean CV MAPE (minimized).
    """
    import lightgbm as lgb
    import optuna
    from sklearn.model_selection import TimeSeriesSplit

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    is_classification = task == "classification"
    direction = "maximize" if is_classification else "minimize"

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        n_splits = min(5, max(2, len(X) // 30))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores: list[float] = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if is_classification:
                model = lgb.LGBMClassifier(**params, verbosity=-1)
                model.fit(X_train, y_train)
                try:
                    from sklearn.metrics import roc_auc_score

                    preds = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, preds)
                except (ValueError, IndexError):
                    score = 0.5
            else:
                model = lgb.LGBMRegressor(**params, verbosity=-1)
                model.fit(X_train, y_train)
                from sklearn.metrics import mean_absolute_percentage_error

                preds = model.predict(X_val)
                score = mean_absolute_percentage_error(y_val, preds)

            scores.append(float(score))

        return float(np.mean(scores)) if scores else (0.5 if is_classification else float("inf"))

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("tune_lightgbm: best value=%.5f params=%s", study.best_value, study.best_params)
    return study.best_params
