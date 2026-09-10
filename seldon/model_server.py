import joblib
import numpy as np
import json
import logging
from pathlib import Path
from typing import Union, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskModel:

    def __init__(self):
        self.model = None
        self.features = None
        self.metrics = None
        self.model_version = "2.0.0"
        self.loaded = False

    def load(self):
        try:
            model_dir = Path("/models/credit_risk/v2.0.0")
            model_path = model_dir / "model.pkl"
            self.model = joblib.load(model_path)
            logger.info(f" Loaded model from {model_path}")
            features_path = model_dir / "features.json"
            with open(features_path, "r") as f:
                self.features = json.load(f)
            logger.info(f" Loaded {len(self.features)} features")
            metrics_path = model_dir / "metrics.json"
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)
            logger.info(
                f" Loaded metrics: Accuracy={self.metrics.get('accuracy', 0):.4f}"
            )
            self.loaded = True
            logger.info("Model server ready!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(
        self, X: Union[np.ndarray, List], names: List[str] = None, meta: dict = None
    ) -> np.ndarray:
        if not self.loaded:
            self.load()
        if isinstance(X, list):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        logger.info(f"Predicting on {X.shape[0]} samples")
        try:
            prediction = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            result = probabilities
            logger.info(
                f"Prediction complete: {prediction[0]} (confidence: {probabilities[0].max():.4f})"
            )
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def health_ping(self) -> dict:
        return {
            "status": "ok",
            "model_loaded": self.loaded,
            "model_version": self.model_version,
        }

    def health_ready(self) -> dict:
        if not self.loaded:
            self.load()
        return {
            "ready": self.loaded,
            "model_version": self.model_version,
            "features_count": len(self.features) if self.features else 0,
            "accuracy": self.metrics.get("accuracy", 0) if self.metrics else 0,
        }

    def metadata(self) -> dict:
        return {
            "name": "credit_risk",
            "version": self.model_version,
            "platform": "scikit-learn",
            "inputs": [
                {"name": feat, "datatype": "FLOAT", "shape": [1]}
                for feat in self.features or []
            ],
            "outputs": [
                {"name": "probability_benign", "datatype": "FLOAT", "shape": [1]},
                {"name": "probability_malignant", "datatype": "FLOAT", "shape": [1]},
            ],
        }
