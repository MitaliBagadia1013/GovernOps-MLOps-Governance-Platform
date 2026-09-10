import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import logging
from src.pipeline.data_loader import load_fraud_detection_data
from src.models.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FraudDetectionTrainer:

    def __init__(
        self,
        model_name: str = "fraud_detection",
        version: str = "1.0.0",
        hyperparameters: dict = None,
    ):
        self.model_name = model_name
        self.version = version
        self.hyperparameters = hyperparameters or {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": "balanced",
        }
        self.model = None
        self.scaler = None
        self.metrics = {}
        self.feature_names = None
        self.artifacts_dir = Path(__file__).parent.parent.parent / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)
        self.model_dir = self.artifacts_dir / "models" / model_name / version
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_data(self, X_train, X_test):
        logger.info("Preprocessing data with StandardScaler...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        logger.info(f"  Training data shape: {X_train_scaled.shape}")
        logger.info(f"  Test data shape: {X_test_scaled.shape}")
        return (X_train_scaled, X_test_scaled)

    def train_model(self, X_train, y_train):
        logger.info(
            f"Training RandomForest with hyperparameters: {self.hyperparameters}"
        )
        self.model = RandomForestClassifier(**self.hyperparameters)
        self.model.fit(X_train, y_train)
        logger.info(f" Model trained successfully!")
        logger.info(f"   Number of trees: {self.model.n_estimators}")
        logger.info(
            f"   Feature importances computed: {len(self.model.feature_importances_)}"
        )

    def evaluate_model(self, X_test, y_test):
        logger.info("Evaluating model on test set...")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "test_samples": len(y_test),
            "fraud_detected": int(y_pred.sum()),
            "actual_frauds": int(y_test.sum()),
        }
        logger.info(f" Model Performance:")
        logger.info(f"   Accuracy:  {self.metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {self.metrics['precision']:.4f}")
        logger.info(f"   Recall:    {self.metrics['recall']:.4f}")
        logger.info(f"   F1 Score:  {self.metrics['f1_score']:.4f}")
        logger.info(f"   ROC AUC:   {self.metrics['roc_auc']:.4f}")
        logger.info(
            f"   Fraud Detection Rate: {self.metrics['fraud_detected']}/{self.metrics['actual_frauds']}"
        )
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = self.model_dir / "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"   Saved classification report to {report_path}")

    def save_model_artifacts(self):
        logger.info(f"Saving model artifacts to {self.model_dir}")
        model_path = self.model_dir / "model.pkl"
        joblib.dump(self.model, model_path)
        logger.info(f" Model saved: {model_path}")
        scaler_path = self.model_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f" Scaler saved: {scaler_path}")
        metadata = {
            "model_name": self.model_name,
            "version": self.version,
            "algorithm": "RandomForestClassifier",
            "framework": "scikit-learn",
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "training_date": datetime.now().isoformat(),
            "model_file": str(model_path.name),
            "scaler_file": str(scaler_path.name),
        }
        metadata_path = self.model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f" Metadata saved: {metadata_path}")
        return model_path

    def register_in_model_registry(self, model_path: Path):
        logger.info(f"Registering model in Model Registry...")
        try:
            registry = ModelRegistry()
            registered_model = registry.register_model(
                model_name=self.model_name,
                version=self.version,
                model_metadata={
                    "algorithm": "RandomForestClassifier",
                    "framework": "scikit-learn",
                    "model_path": str(model_path.absolute()),
                    "created_by": "training_pipeline",
                    "hyperparameters": self.hyperparameters,
                    "metrics": self.metrics,
                    "description": f"Fraud detection model trained on {self.metrics['test_samples']} samples",
                    "tags": ["fraud-detection", "production", "random-forest"],
                    "team": "data-science",
                    "project": "fraud-prevention",
                    "framework_version": "1.3.0",
                    "feature_names": self.feature_names,
                },
            )
            logger.info(f" Model registered successfully!")
            logger.info(f"   Model ID: {registered_model.model_id}")
            logger.info(f"   Database ID: {registered_model.id}")
            logger.info(f"   Status: {registered_model.status}")
            return registered_model
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def run_full_pipeline(self, n_samples: int = 100000):
        logger.info("=" * 80)
        logger.info(f"STARTING TRAINING PIPELINE: {self.model_name} v{self.version}")
        logger.info("=" * 80)
        logger.info("\nStep 1: Loading dataset...")
        X_train, X_test, y_train, y_test, feature_names = load_fraud_detection_data(
            n_samples
        )
        self.feature_names = feature_names
        logger.info("\nStep 2: Preprocessing data...")
        X_train_scaled, X_test_scaled = self.preprocess_data(X_train, X_test)
        logger.info("\nStep 3: Training model...")
        self.train_model(X_train_scaled, y_train)
        logger.info("\nStep 4: Evaluating model...")
        self.evaluate_model(X_test_scaled, y_test)
        logger.info("\nStep 5: Saving model artifacts...")
        model_path = self.save_model_artifacts()
        logger.info("\nStep 6: Registering in Model Registry...")
        registered_model = self.register_in_model_registry(model_path)
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"   Model: {self.model_name} v{self.version}")
        logger.info(f"   Accuracy: {self.metrics['accuracy']:.4f}")
        logger.info(f"   F1 Score: {self.metrics['f1_score']:.4f}")
        logger.info(f"   ROC AUC: {self.metrics['roc_auc']:.4f}")
        logger.info(f"   Model Path: {model_path}")
        logger.info(f"   Registry ID: {registered_model.id}")
        logger.info("=" * 80)
        return (self.model, self.metrics)


def train_fraud_detection_model(n_samples: int = 100000):
    trainer = FraudDetectionTrainer()
    return trainer.run_full_pipeline(n_samples=n_samples)


if __name__ == "__main__":
    train_fraud_detection_model(n_samples=100000)
