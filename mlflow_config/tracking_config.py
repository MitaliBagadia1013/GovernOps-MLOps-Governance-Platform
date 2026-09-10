import mlflow
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MLflowConfig:

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.mlruns_dir = self.project_root / "mlruns"
        self.artifacts_dir = self.project_root / "mlflow_artifacts"
        self.mlruns_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)
        self.tracking_uri = f"file://{self.mlruns_dir}"
        mlflow.set_tracking_uri(self.tracking_uri)
        logger.info(f"MLflow tracking URI: {self.tracking_uri}")
        logger.info(f"MLflow artifacts dir: {self.artifacts_dir}")

    def set_experiment(self, experiment_name: str) -> str:
        experiment = mlflow.set_experiment(experiment_name)
        logger.info(
            f"Using experiment: {experiment_name} (ID: {experiment.experiment_id})"
        )
        return experiment.experiment_id

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ):
        return mlflow.start_run(run_name=run_name, tags=tags or {})

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics")

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        model_name: Optional[str] = None,
        registered_model_name: Optional[str] = None,
    ):
        model_type = type(model).__module__
        if "sklearn" in model_type:
            mlflow.sklearn.log_model(
                model, artifact_path, registered_model_name=registered_model_name
            )
        elif "xgboost" in model_type:
            mlflow.xgboost.log_model(
                model, artifact_path, registered_model_name=registered_model_name
            )
        elif "tensorflow" in model_type or "keras" in model_type:
            mlflow.keras.log_model(
                model, artifact_path, registered_model_name=registered_model_name
            )
        else:
            mlflow.pyfunc.log_model(
                artifact_path,
                python_model=model,
                registered_model_name=registered_model_name,
            )
        logger.info(f"Logged model to: {artifact_path}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        mlflow.log_artifact(local_path, artifact_path)
        logger.debug(f"Logged artifact: {local_path}")

    def log_dict(self, dictionary: Dict[str, Any], filename: str):
        mlflow.log_dict(dictionary, filename)
        logger.debug(f"Logged dictionary as: {filename}")

    def set_tags(self, tags: Dict[str, str]):
        mlflow.set_tags(tags)
        logger.debug(f"Set {len(tags)} tags")

    def end_run(self):
        mlflow.end_run()
        logger.debug("Ended MLflow run")


_mlflow_config = None


def get_mlflow_config() -> MLflowConfig:
    global _mlflow_config
    if _mlflow_config is None:
        _mlflow_config = MLflowConfig()
    return _mlflow_config


def configure_tracking(experiment_name: str = "governops"):
    config = get_mlflow_config()
    config.set_experiment(experiment_name)


def log_params(params: Dict[str, Any]):
    get_mlflow_config().log_params(params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    get_mlflow_config().log_metrics(metrics, step)


def log_model(model: Any, model_name: str, model_version: str, register: bool = True):
    artifact_path = f"models/{model_name}"
    registered_name = f"{model_name}" if register else None
    get_mlflow_config().log_model(
        model, artifact_path, registered_model_name=registered_name
    )
