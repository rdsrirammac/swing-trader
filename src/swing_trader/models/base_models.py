"""Base predictive models (SRS PM-002).

Four interchangeable model wrappers used as ensemble members: LightGBM
(primary tabular model), an ARIMAX baseline for mean-reversion/trend,
a small LSTM for sequence pattern recognition, and a RandomForest used
for robustness checks / feature importance. Each (aside from ArimaXModel,
which is inherently per-series -- see its docstring) exposes a shared
`.fit(X, y)` / `.predict(X)` / `.predict_proba(X)` / `.save(path)` /
`.load(path)` interface so `ensemble.StackingEnsemble` can treat them
uniformly.
"""
from __future__ import annotations

from typing import Literal

import joblib
import numpy as np
import pandas as pd

from swing_trader.logging_setup import get_logger

logger = get_logger("models.base_models")

Task = Literal["classification", "regression"]


# ---------------------------------------------------------------------------
# LightGBM (primary tabular model)
# ---------------------------------------------------------------------------

class LightGBMModel:
    """Wraps `lightgbm.LGBMClassifier` / `LGBMRegressor`.

    Reproducibility: unless the caller explicitly overrides them,
    `random_state` and `deterministic=True` are always injected from
    `modeling.random_seed` (default 42). Without this, LightGBM's own
    internal RNG (bagging/feature subsampling) and multi-threaded histogram
    building are non-deterministic by default -- identical training data
    would otherwise produce a different model (and different predictions)
    every time `.fit()` is called, which is exactly what was happening
    across repeated `predict` runs before this was added.
    """

    def __init__(self, task: Task = "classification", **lgbm_params):
        import lightgbm as lgb

        from swing_trader.config import get_settings

        self.task = task
        self.params = dict(lgbm_params)
        self.params.setdefault("random_state", get_settings().get("modeling.random_seed", 42))
        self.params.setdefault("deterministic", True)
        self.model = (
            lgb.LGBMClassifier(**self.params)
            if task == "classification"
            else lgb.LGBMRegressor(**self.params)
        )
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else None
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            return self.predict(X)
        return np.asarray(self.model.predict_proba(X))

    def feature_importance(self) -> pd.Series:
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return pd.Series(dtype="float64")
        names = self.feature_names_ or [f"f{i}" for i in range(len(importances))]
        return pd.Series(importances, index=names).sort_values(ascending=False)

    def save(self, path: str) -> None:
        joblib.dump(
            {"task": self.task, "params": self.params, "model": self.model, "feature_names": self.feature_names_},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "LightGBMModel":
        payload = joblib.load(path)
        obj = cls(task=payload["task"], **payload.get("params", {}))
        obj.model = payload["model"]
        obj.feature_names_ = payload.get("feature_names")
        return obj


# ---------------------------------------------------------------------------
# ARIMAX (baseline mean-reversion/trend model)
# ---------------------------------------------------------------------------

class ArimaXModel:
    """Wraps `statsmodels.tsa.arima.model.ARIMA` with optional exogenous
    regressors, used as a simple mean-reversion/trend baseline.

    Unlike the sklearn-style wrappers in this module, ARIMA is inherently a
    per-series model: it doesn't accept a design matrix `X` of independent
    rows, it fits one time-ordered series `y` (optionally with an aligned
    `exog` regressor frame). To keep this pragmatic rather than forcing a
    poor-fit abstraction, `.fit`/`.predict` intentionally deviate from the
    shared `(X, y)` / `(X)` signature used elsewhere:

        model.fit(y: pd.Series, exog: pd.DataFrame | None = None)
        model.predict(steps: int = 1, exog: pd.DataFrame | None = None)

    Callers (e.g. the ensemble/pipeline) must branch on model type rather
    than calling `.fit(X, y)` uniformly for this class.
    """

    def __init__(self, order: tuple[int, int, int] = (1, 0, 0)):
        self.order = order
        self._result = None
        self.exog_columns: list[str] | None = None

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "ArimaXModel":
        from statsmodels.tsa.arima.model import ARIMA

        self.exog_columns = list(exog.columns) if exog is not None else None
        model = ARIMA(y, exog=exog, order=self.order)
        self._result = model.fit()
        return self

    def predict(self, steps: int = 1, exog: pd.DataFrame | None = None) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("ArimaXModel.predict called before fit")
        forecast = self._result.forecast(steps=steps, exog=exog)
        return np.asarray(forecast)

    def predict_proba(self, steps: int = 1, exog: pd.DataFrame | None = None) -> np.ndarray:
        # ARIMA is a regressor; alias to predict for interface symmetry with
        # the classifier-style models.
        return self.predict(steps=steps, exog=exog)

    def save(self, path: str) -> None:
        if self._result is None:
            raise RuntimeError("cannot save an unfit ArimaXModel")
        self._result.save(path)

    @classmethod
    def load(cls, path: str) -> "ArimaXModel":
        from statsmodels.tsa.arima.model import ARIMAResults

        obj = cls()
        obj._result = ARIMAResults.load(path)
        return obj


# ---------------------------------------------------------------------------
# LSTM (sequence pattern recognition)
# ---------------------------------------------------------------------------

def _build_lstm_net(input_size: int, hidden_size: int, num_layers: int):
    import torch.nn as nn

    class _LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            last_step = out[:, -1, :]
            return self.head(last_step)

    return _LSTMNet()


class LSTMModel:
    """Thin sklearn-like wrapper around a small 1-2 layer LSTM for sequence
    pattern recognition over the trailing `seq_len` days of features.

    Deliberately small/fast (default hidden_size=32, ~20 epochs, CPU-only
    Adam training loop) since this runs on a Mac Mini, not a GPU cluster.
    """

    def __init__(
        self,
        task: Task = "classification",
        seq_len: int = 20,
        hidden_size: int = 32,
        num_layers: int = 1,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int | None = 5,
    ):
        self.task = task
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.net = None
        self.feature_names_: list[str] | None = None

    def _make_sequences(self, X: pd.DataFrame) -> np.ndarray:
        """Build a (n_rows, seq_len, n_features) tensor of trailing windows,
        left-padding with zeros for rows without `seq_len` history yet.
        Assumes `X` is already sorted chronologically ascending.
        """
        values = X.to_numpy(dtype="float32")
        n_features = values.shape[1] if values.ndim == 2 else 1
        sequences = []
        for i in range(len(values)):
            start = max(0, i - self.seq_len + 1)
            window = values[start : i + 1]
            if len(window) < self.seq_len:
                pad = np.zeros((self.seq_len - len(window), n_features), dtype="float32")
                window = np.vstack([pad, window])
            sequences.append(window)
        return np.stack(sequences) if sequences else np.zeros((0, self.seq_len, n_features), dtype="float32")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMModel":
        import torch

        self.feature_names_ = list(X.columns)
        sequences = self._make_sequences(X)
        target = y.to_numpy(dtype="float32").reshape(-1, 1)

        self.net = _build_lstm_net(
            input_size=sequences.shape[-1], hidden_size=self.hidden_size, num_layers=self.num_layers
        )
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss() if self.task == "classification" else torch.nn.MSELoss()

        X_tensor = torch.from_numpy(sequences)
        y_tensor = torch.from_numpy(target)
        n = X_tensor.shape[0]

        best_loss = float("inf")
        epochs_no_improve = 0

        self.net.train()
        for epoch in range(self.epochs):
            permutation = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = permutation[start : start + self.batch_size]
                batch_X = X_tensor[idx]
                batch_y = y_tensor[idx]
                optimizer.zero_grad()
                preds = self.net(batch_X)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            epoch_loss /= max(n, 1)

            if self.patience is not None:
                if epoch_loss < best_loss - 1e-6:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        logger.info("LSTMModel: early stopping at epoch %d (loss=%.5f)", epoch, epoch_loss)
                        break
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        if self.net is None:
            raise RuntimeError("LSTMModel.predict called before fit")
        sequences = self._make_sequences(X)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(sequences)).numpy().reshape(-1)
        if self.task == "classification":
            probs = 1.0 / (1.0 + np.exp(-logits))
            return (probs > 0.5).astype(int)
        return logits

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        if self.task != "classification":
            return self.predict(X)
        if self.net is None:
            raise RuntimeError("LSTMModel.predict_proba called before fit")
        sequences = self._make_sequences(X)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(sequences)).numpy().reshape(-1)
        probs_pos = 1.0 / (1.0 + np.exp(-logits))
        return np.stack([1.0 - probs_pos, probs_pos], axis=1)

    def save(self, path: str) -> None:
        import torch

        torch.save(
            {
                "state_dict": self.net.state_dict() if self.net is not None else None,
                "task": self.task,
                "seq_len": self.seq_len,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "feature_names": self.feature_names_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        import torch

        payload = torch.load(path, map_location="cpu")
        obj = cls(
            task=payload.get("task", "classification"),
            seq_len=payload.get("seq_len", 20),
            hidden_size=payload.get("hidden_size", 32),
            num_layers=payload.get("num_layers", 1),
        )
        obj.feature_names_ = payload.get("feature_names")
        input_size = len(obj.feature_names_) if obj.feature_names_ else None
        if payload.get("state_dict") is not None and input_size:
            obj.net = _build_lstm_net(
                input_size=input_size, hidden_size=obj.hidden_size, num_layers=obj.num_layers
            )
            obj.net.load_state_dict(payload["state_dict"])
        return obj


# ---------------------------------------------------------------------------
# RandomForest (robustness checks / feature importance)
# ---------------------------------------------------------------------------

class RandomForestModel:
    """Wraps `sklearn.ensemble.RandomForestClassifier`/`Regressor`."""

    def __init__(self, task: Task = "classification", **rf_params):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        self.task = task
        self.params = dict(rf_params)
        self.model = (
            RandomForestClassifier(**self.params)
            if task == "classification"
            else RandomForestRegressor(**self.params)
        )
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else None
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            return self.predict(X)
        return np.asarray(self.model.predict_proba(X))

    def feature_importance(self) -> pd.Series:
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return pd.Series(dtype="float64")
        names = self.feature_names_ or [f"f{i}" for i in range(len(importances))]
        return pd.Series(importances, index=names).sort_values(ascending=False)

    def save(self, path: str) -> None:
        joblib.dump(
            {"task": self.task, "params": self.params, "model": self.model, "feature_names": self.feature_names_},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "RandomForestModel":
        payload = joblib.load(path)
        obj = cls(task=payload["task"], **payload.get("params", {}))
        obj.model = payload["model"]
        obj.feature_names_ = payload.get("feature_names")
        return obj
