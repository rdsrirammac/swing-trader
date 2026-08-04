"""Thin MLflow wrapper for experiment tracking (SRS PM-006).

All MLflow calls are wrapped in try/except: a missing or unreachable MLflow
tracking server must never crash the pipeline. On failure we log a warning
and return a locally-generated fallback run id.
"""
from __future__ import annotations

from uuid import uuid4

from swing_trader.config import get_settings
from swing_trader.logging_setup import get_logger

logger = get_logger("models.tracking")


def log_model_run(
    model_version: str,
    params: dict,
    metrics: dict,
    model=None,
    artifact_name: str = "model",
) -> str:
    """Log a training run to MLflow. Returns the MLflow run id, or a
    `local-<hex>` fallback id if MLflow is unavailable/unreachable.
    """
    settings = get_settings()
    tracking_uri = settings.get("mlflow.tracking_uri")
    experiment_name = settings.get("mlflow.experiment_name")

    try:
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=model_version) as run:
            try:
                mlflow.log_params(params or {})
            except Exception as e:
                logger.warning("log_model_run: failed to log params: %s", e)

            try:
                # mlflow.log_metrics rejects None values; drop them defensively.
                clean_metrics = {k: v for k, v in (metrics or {}).items() if v is not None}
                mlflow.log_metrics(clean_metrics)
            except Exception as e:
                logger.warning("log_model_run: failed to log metrics: %s", e)

            if model is not None:
                _log_model_artifact(model, artifact_name)

            run_id = run.info.run_id
            logger.info("log_model_run: logged MLflow run %s for %s", run_id, model_version)
            return run_id
    except Exception as e:
        fallback_id = f"local-{uuid4().hex[:12]}"
        logger.warning(
            "log_model_run: MLflow unavailable/unreachable (%s); using fallback run id %s",
            e,
            fallback_id,
        )
        return fallback_id


def _log_model_artifact(model, artifact_name: str) -> None:
    """Best-effort model artifact logging; tries sklearn flavor first, falls
    back to a generic joblib dump-and-log-artifact for wrapper classes
    (e.g. `base_models.LightGBMModel`) that aren't directly sklearn estimators.
    """
    import mlflow

    try:
        mlflow.sklearn.log_model(model, artifact_name)
        return
    except Exception:
        pass

    try:
        import os
        import tempfile

        import joblib

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, f"{artifact_name}.joblib")
            joblib.dump(model, path)
            mlflow.log_artifact(path)
    except Exception as e:
        logger.warning("log_model_run: failed to log model artifact: %s", e)
