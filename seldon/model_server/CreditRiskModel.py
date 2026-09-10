import joblib
import json
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskModel:

    def __init__(self):
        logger.info("Initializing Credit Risk Model...")
        model_path = Path("/mnt/models/credit_risk/v2.0.0/model.pkl")
        features_path = Path("/mnt/models/credit_risk/v2.0.0/features.json")
        if not model_path.exists():
            model_path = Path("../../models/credit_risk/v2.0.0/model.pkl")
            features_path = Path("../../models/credit_risk/v2.0.0/features.json")
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        with open(features_path, "r") as f:
            self.feature_names = json.load(f)
        logger.info(f" Model loaded successfully!")
        logger.info(f" Features: {len(self.feature_names)}")
        logger.info(f" Model type: {type(self.model).__name__}")

    def predict(self, X, features_names=None):
        logger.info(f"Received prediction request with shape: {np.array(X).shape}")
        X = np.array(X)
        predictions = self.model.predict_proba(X)
        logger.info(f"Returning predictions with shape: {predictions.shape}")
        return predictions
