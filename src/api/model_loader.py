import joblib
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not installed. MLflow model loading will be disabled.")
logger = logging.getLogger(__name__)


class ModelLoader:
    _instance = None
    _models_cache: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.project_root = Path(__file__).parent.parent.parent
            self.models_dir = self.project_root / "models"
            use_mlflow = os.getenv("USE_MLFLOW", "local").lower()
            self.use_mlflow = use_mlflow == "mlflow" and MLFLOW_AVAILABLE
            if self.use_mlflow:
                logger.info("ModelLoader initialized with MLflow backend")
            else:
                logger.info(f" ModelLoader initialized with local file backend")
                logger.info(f"   Models dir: {self.models_dir}")
            self.initialized = True

    def load_model(self, model_name: str, version: str) -> Dict[str, Any]:
        cache_key = f"{model_name}_v{version}"
        if cache_key in self._models_cache:
            logger.info(f" Loaded {model_name} v{version} from cache")
            return self._models_cache[cache_key]
        model_path = self.models_dir / model_name / f"v{version}"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_name} v{version} at {model_path}"
            )
        try:
            model_file = model_path / "model.pkl"
            model = joblib.load(model_file)
            features_file = model_path / "features.json"
            with open(features_file, "r") as f:
                features = json.load(f)
            metrics_file = model_path / "metrics.json"
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            model_bundle = {
                "model": model,
                "features": features,
                "metrics": metrics,
                "metadata": {
                    "name": model_name,
                    "version": version,
                    "loaded_at": datetime.now().isoformat(),
                    "path": str(model_path),
                },
            }
            self._models_cache[cache_key] = model_bundle
            logger.info(f" Loaded {model_name} v{version} from disk")
            logger.info(f"  Features: {len(features)}")
            logger.info(f"  Metrics: {list(metrics.keys())}")
            return model_bundle
        except Exception as e:
            logger.error(f"Failed to load {model_name} v{version}: {str(e)}")
            raise ValueError(f"Invalid model artifacts: {str(e)}")

    def load_model_from_mlflow(
        self, model_name: str, version: Optional[str] = None
    ) -> Dict[str, Any]:
        if not MLFLOW_AVAILABLE:
            raise RuntimeError("MLflow not installed. Run: pip install mlflow")
        cache_key = f"mlflow_{model_name}_{version or 'latest'}"
        if cache_key in self._models_cache:
            logger.info(f" Loaded {model_name} from cache (MLflow)")
            return self._models_cache[cache_key]
        try:
            from mlflow_config.tracking_config import get_mlflow_config

            mlflow_config = get_mlflow_config()
            mlflow.set_tracking_uri(mlflow_config.tracking_uri)
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            if version is None:
                model_versions = client.search_model_versions(f"name='{model_name}'")
                if not model_versions:
                    raise ValueError(f"No versions found for model '{model_name}'")
                latest = max(model_versions, key=lambda v: int(v.version))
                version = latest.version
                logger.info(f"Auto-detected latest version: {version}")
            model_uri = f"models:/{model_name}/{version}"
            logger.info(f"Loading from MLflow: {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)
            try:
                model_version = client.get_model_version(model_name, version)
                run_id = model_version.run_id
                run = client.get_run(run_id)
                metadata = {
                    "name": model_name,
                    "version": version,
                    "run_id": run_id,
                    "loaded_at": datetime.now().isoformat(),
                    "source": "mlflow",
                    "tags": dict(run.data.tags),
                }
                metrics = dict(run.data.metrics)
                try:
                    features_path = client.download_artifacts(run_id, "features.json")
                    with open(features_path, "r") as f:
                        features_data = json.load(f)
                        features = features_data.get("features", features_data)
                except:
                    features = []
            except Exception as e:
                logger.warning(f"Could not get MLflow metadata: {e}")
                metadata = {"name": model_name, "version": version, "source": "mlflow"}
                metrics = {}
                features = []
            model_bundle = {
                "model": model,
                "features": features,
                "metrics": metrics,
                "metadata": metadata,
            }
            self._models_cache[cache_key] = model_bundle
            logger.info(f" SUCCESS! Loaded {model_name} v{version} from MLflow")
            if metrics:
                logger.info(f"   Metrics: {list(metrics.keys())[:5]}")
            return model_bundle
        except Exception as e:
            logger.error(f"MLflow loading failed: {str(e)}")
            raise ValueError(
                f"Could not load {model_name} from MLflow.\nError: {str(e)}\nMake sure model is registered in MLflow Registry!"
            )

    def smart_load(
        self, model_name: str, version: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.use_mlflow:
            try:
                logger.info(f" Loading {model_name} from MLflow...")
                return self.load_model_from_mlflow(model_name, version)
            except Exception as e:
                logger.warning(f"MLflow loading failed, trying local files: {e}")
                if version:
                    local_version = version if "." in version else f"{version}.0.0"
                    return self.load_model(model_name, local_version)
                else:
                    raise ValueError(
                        f"MLflow loading failed and no version specified for local fallback.\nEither fix MLflow or provide a version number."
                    )
        else:
            if not version:
                raise ValueError(
                    f"Version required when loading from local files.\nProvide a version like: smart_load('{model_name}', '2.0.0')"
                )
            local_version = version if "." in version else f"{version}.0.0"
            logger.info(f" Loading {model_name} v{local_version} from local files...")
            return self.load_model(model_name, local_version)

    def get_available_models(self) -> Dict[str, list]:
        available = {}
        if not self.models_dir.exists():
            return available
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                versions = []
                for version_dir in model_dir.iterdir():
                    if version_dir.is_dir() and version_dir.name.startswith("v"):
                        version = version_dir.name[1:]
                        if (version_dir / "model.pkl").exists() and (
                            version_dir / "features.json"
                        ).exists():
                            versions.append(version)
                if versions:
                    available[model_name] = sorted(versions, reverse=True)
        return available

    def clear_cache(self):
        self._models_cache.clear()
        logger.info("Model cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        return {
            "cached_models": list(self._models_cache.keys()),
            "cache_size": len(self._models_cache),
            "models_dir": str(self.models_dir),
        }
