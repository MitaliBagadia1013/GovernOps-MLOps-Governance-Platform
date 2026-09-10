import sys
import os
from pathlib import Path
import logging
import warnings

warnings.filterwarnings("ignore")
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from typing import Dict, Any
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.datasets import (
    fetch_california_housing,
    load_digits,
    load_breast_cancer,
    load_wine,
    load_iris,
    load_diabetes,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    silhouette_score,
    confusion_matrix,
    r2_score,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingRegressor,
    IsolationForest,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from src.models.model_registry import ModelRegistry
from src.models.model_cards import create_model_card, save_model_card_to_file
from mlflow_config.tracking_config import get_mlflow_config, configure_tracking
import mlflow

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MultiModelTrainer:

    def __init__(self, use_mlflow: bool = True):
        self.project_root = project_root
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.model_cards_dir = self.project_root / "model_cards"
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.model_cards_dir.mkdir(exist_ok=True)
        self.registry = ModelRegistry()
        self.models_trained = []
        self.use_mlflow = use_mlflow
        if self.use_mlflow:
            self.mlflow_config = get_mlflow_config()
            configure_tracking("mlops-model-training")
            logger.info("MLflow tracking enabled")

    def train_credit_risk_model(self):
        logger.info("\n" + "=" * 80)
        logger.info("MODEL 1/9: CREDIT RISK PREDICTION (RandomForest on Breast Cancer)")
        logger.info("=" * 80)
        logger.info("Loading Wisconsin Breast Cancer dataset...")
        cancer = load_breast_cancer()
        X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        y = cancer.target
        logger.info(f" Loaded {len(X)} REAL diagnostic records")
        logger.info(f"  Features: {len(cancer.feature_names)}")
        logger.info(f"  Source: UCI Machine Learning Repository")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        logger.info("Training RandomForest model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        }
        logger.info(f" Accuracy: {metrics['accuracy'] * 100:.2f}%")
        logger.info(f" ROC-AUC: {metrics['roc_auc']:.4f}")
        self._save_and_register_model(
            model=model,
            model_name="credit_risk",
            version="2.0.0",
            algorithm="RandomForestClassifier",
            framework="scikit-learn",
            metrics=metrics,
            feature_names=list(cancer.feature_names),
            dataset_info={
                "name": "wisconsin_breast_cancer",
                "size": len(X),
                "source": "UCI ML Repository (sklearn.datasets.load_breast_cancer)",
                "description": "Real diagnostic data - binary classification",
                "features": list(cancer.feature_names)[:5],
            },
        )
        return metrics

    def train_churn_prediction_model(self):
        logger.info("\n" + "=" * 80)
        logger.info("MODEL 2/9: CHURN PREDICTION (Logistic Regression on Wine)")
        logger.info("=" * 80)
        logger.info("Loading Wine dataset...")
        wine = load_wine()
        X = pd.DataFrame(wine.data, columns=wine.feature_names)
        y = (wine.target > 0).astype(int)
        logger.info(f" Loaded {len(X)} REAL wine samples")
        logger.info(f"  Features: {len(wine.feature_names)}")
        logger.info(f"  Source: UCI Machine Learning Repository")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = LogisticRegression(max_iter=1000, random_state=42)
        logger.info("Training Logistic Regression model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        }
        logger.info(f" Accuracy: {metrics['accuracy'] * 100:.2f}%")
        logger.info(f" F1 Score: {metrics['f1_score']:.4f}")
        self._save_and_register_model(
            model=model,
            model_name="churn_prediction",
            version="1.0.0",
            algorithm="LogisticRegression",
            framework="scikit-learn",
            metrics=metrics,
            feature_names=list(wine.feature_names),
            dataset_info={
                "name": "wine_quality",
                "size": len(X),
                "source": "UCI ML Repository (sklearn.datasets.load_wine)",
                "description": "Real wine chemical analysis - binary classification proxy",
                "features": list(wine.feature_names)[:5],
            },
        )
        return metrics

    def train_price_prediction_model(self):
        logger.info("\n" + "=" * 80)
        logger.info("MODEL 3/9: PRICE PREDICTION (Gradient Boosting on Housing)")
        logger.info("=" * 80)
        logger.info("Loading California Housing dataset...")
        housing = fetch_california_housing()
        X = pd.DataFrame(housing.data, columns=housing.feature_names)
        y = housing.target * 100000
        logger.info(f" Loaded {len(X)} REAL housing records")
        logger.info(f"  Features: {list(housing.feature_names)}")
        logger.info(f"  Source: 1990 US Census via sklearn")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )
        logger.info("Training Gradient Boosting model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2_score": float(r2_score(y_test, y_pred)),
        }
        logger.info(f" RMSE: ${metrics['rmse']:,.0f}")
        logger.info(f" MAE: ${metrics['mae']:,.0f}")
        logger.info(f" R² Score: {metrics['r2_score']:.4f}")
        self._save_and_register_model(
            model=model,
            model_name="price_prediction",
            version="2.0.0",
            algorithm="GradientBoostingRegressor",
            framework="scikit-learn",
            metrics=metrics,
            feature_names=list(housing.feature_names),
            dataset_info={
                "name": "california_housing",
                "size": len(X),
                "source": "1990 US Census (sklearn.datasets.fetch_california_housing)",
                "description": "Real housing prices from California districts",
                "features": list(housing.feature_names),
            },
        )
        return metrics

    def train_anomaly_detection_model(self):
        logger.info("\n" + "=" * 80)
        logger.info("MODEL 4/9: ANOMALY DETECTION (Isolation Forest)")
        logger.info("=" * 80)
        logger.info("Loading Breast Cancer dataset for anomaly detection...")
        cancer = load_breast_cancer()
        X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        y = cancer.target
        logger.info(f" Loaded {len(X)} REAL records")
        logger.info(f"  Using unsupervised learning to detect outliers")
        model = IsolationForest(contamination=0.1, random_state=42)
        logger.info("Training Isolation Forest model...")
        model.fit(X)
        y_pred_raw = model.predict(X)
        y_pred = (y_pred_raw == -1).astype(int)
        metrics = {
            "anomaly_rate": float(y_pred.mean()),
            "n_anomalies": int(y_pred.sum()),
            "n_normal": int((y_pred == 0).sum()),
        }
        logger.info(
            f" Detected {metrics['n_anomalies']} anomalies ({metrics['anomaly_rate'] * 100:.1f}%)"
        )
        self._save_and_register_model(
            model=model,
            model_name="anomaly_detection",
            version="1.0.0",
            algorithm="IsolationForest",
            framework="scikit-learn",
            metrics=metrics,
            feature_names=list(cancer.feature_names),
            dataset_info={
                "name": "breast_cancer_anomaly",
                "size": len(X),
                "source": "UCI ML Repository (sklearn.datasets.load_breast_cancer)",
                "description": "Real diagnostic data for outlier detection",
                "features": list(cancer.feature_names)[:5],
            },
        )
        return metrics

    def train_customer_segmentation_model(self):
        logger.info("\n" + "=" * 80)
        logger.info("MODEL 5/9: CUSTOMER SEGMENTATION (KMeans on Iris)")
        logger.info("=" * 80)
        logger.info("Loading Iris dataset...")
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target
        logger.info(f" Loaded {len(X)} REAL flower samples")
        logger.info(f"  Features: {list(iris.feature_names)}")
        logger.info(f"  Source: Fisher's Iris dataset (1936)")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_clusters = 3
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        logger.info(f"Training KMeans model with {n_clusters} clusters...")
        model.fit(X_scaled)
        labels = model.labels_
        silhouette = silhouette_score(X_scaled, labels)
        metrics = {
            "silhouette_score": float(silhouette),
            "inertia": float(model.inertia_),
            "n_clusters": n_clusters,
        }
        logger.info(f" Silhouette Score: {silhouette:.4f}")
        logger.info(f" Inertia: {model.inertia_:.2f}")
        model_bundle = {"model": model, "scaler": scaler}
        self._save_and_register_model(
            model=model_bundle,
            model_name="customer_segmentation",
            version="2.0.0",
            algorithm="KMeans",
            framework="scikit-learn",
            metrics=metrics,
            feature_names=list(iris.feature_names),
            dataset_info={
                "name": "iris",
                "size": len(X),
                "source": "Fisher's Iris dataset (sklearn.datasets.load_iris)",
                "description": "Real flower measurements - clustering proxy for customer segments",
                "features": list(iris.feature_names),
            },
        )
        return metrics

    def train_image_classification_model(self):
        logger.info("\n" + "=" * 80)
        logger.info("MODEL 6/9: IMAGE CLASSIFICATION (Random Forest on Digits)")
        logger.info("=" * 80)
        logger.info("Loading Digits dataset...")
        digits = load_digits()
        X = digits.data
        y = digits.target
        logger.info(f" Loaded {len(X)} REAL handwritten digit images")
        logger.info(f"  Image size: 8x8 pixels")
        logger.info(f"  Classes: {len(np.unique(y))} digits (0-9)")
        logger.info(f"  Source: UCI ML Repository")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        )
        logger.info("Training Random Forest model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
            "f1_score_macro": float(f1_score(y_test, y_pred, average="macro")),
        }
        logger.info(f" Accuracy: {metrics['accuracy'] * 100:.2f}%")
        logger.info(f" F1 Score (macro): {metrics['f1_score_macro']:.4f}")
        self._save_and_register_model(
            model=model,
            model_name="image_classification",
            version="1.0.0",
            algorithm="RandomForestClassifier",
            framework="scikit-learn",
            metrics=metrics,
            feature_names=[f"pixel_{i}" for i in range(64)],
            dataset_info={
                "name": "digits",
                "size": len(X),
                "source": "UCI ML Repository (sklearn.datasets.load_digits)",
                "description": "Real handwritten digits (MNIST-like)",
                "image_size": "8x8 pixels",
                "num_classes": 10,
            },
        )
        return metrics

    def train_sentiment_analysis_model(self):
        logger.info("\n" + "=" * 80)
        logger.info("MODEL 7/9: SENTIMENT ANALYSIS (Gaussian NB on Iris)")
        logger.info("=" * 80)
        logger.info("Loading Iris dataset...")
        iris = load_iris()
        X = iris.data
        y = iris.target
        logger.info(f" Loaded {len(X)} REAL samples")
        logger.info(f"  Using as proxy for text classification")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = GaussianNB()
        logger.info("Training Gaussian Naive Bayes model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
            "f1_score_macro": float(f1_score(y_test, y_pred, average="macro")),
        }
        logger.info(f" Accuracy: {metrics['accuracy'] * 100:.2f}%")
        logger.info(f" F1 Score (macro): {metrics['f1_score_macro']:.4f}")
        self._save_and_register_model(
            model=model,
            model_name="sentiment_analysis",
            version="1.0.0",
            algorithm="GaussianNB",
            framework="scikit-learn",
            metrics=metrics,
            feature_names=list(iris.feature_names),
            dataset_info={
                "name": "iris_sentiment_proxy",
                "size": len(X),
                "source": "Fisher's Iris (sklearn.datasets.load_iris)",
                "description": "Real data used as multi-class classification proxy",
                "features": list(iris.feature_names),
            },
        )
        return metrics

    def train_demand_forecasting_model(self):
        logger.info("\n" + "=" * 80)
        logger.info("MODEL 8/9: DEMAND FORECASTING (Ridge Regression on Diabetes)")
        logger.info("=" * 80)
        logger.info("Loading Diabetes dataset...")
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        y = diabetes.target
        logger.info(f" Loaded {len(X)} REAL patient records")
        logger.info(f"  Features: {list(diabetes.feature_names)}")
        logger.info(f"  Source: sklearn diabetes dataset")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = Ridge(alpha=1.0, random_state=42)
        logger.info("Training Ridge Regression model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2_score": float(r2_score(y_test, y_pred)),
        }
        logger.info(f" RMSE: {metrics['rmse']:.2f}")
        logger.info(f" MAE: {metrics['mae']:.2f}")
        logger.info(f" R² Score: {metrics['r2_score']:.4f}")
        self._save_and_register_model(
            model=model,
            model_name="demand_forecasting",
            version="1.0.0",
            algorithm="Ridge",
            framework="scikit-learn",
            metrics=metrics,
            feature_names=list(diabetes.feature_names),
            dataset_info={
                "name": "diabetes",
                "size": len(X),
                "source": "sklearn diabetes dataset",
                "description": "Real patient data - regression proxy for demand forecasting",
                "features": list(diabetes.feature_names),
            },
        )
        return metrics

    def train_recommendation_model(self):
        logger.info("\n" + "=" * 80)
        logger.info("MODEL 9/9: RECOMMENDATION SYSTEM (KNN on Digits)")
        logger.info("=" * 80)
        logger.info("Loading Digits dataset...")
        digits = load_digits()
        X = digits.data
        y = digits.target
        logger.info(f" Loaded {len(X)} REAL digit images")
        logger.info(
            f"  Using similarity-based recommendation (content-based filtering)"
        )
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        model = NearestNeighbors(n_neighbors=10, metric="cosine")
        logger.info("Training KNN model...")
        model.fit(X_scaled)
        distances, indices = model.kneighbors(X_scaled[:100])
        avg_distance = distances.mean()
        metrics = {
            "n_items": len(X),
            "n_neighbors": 10,
            "avg_neighbor_distance": float(avg_distance),
        }
        distance_metric = "cosine"
        logger.info(f" Items: {len(X)}")
        logger.info(f" Avg neighbor distance: {avg_distance:.4f}")
        logger.info(f" Distance metric: {distance_metric}")
        model_bundle = {"model": model, "scaler": scaler}
        self._save_and_register_model(
            model=model_bundle,
            model_name="recommendation",
            version="3.1.0",
            algorithm="KNearestNeighbors",
            framework="scikit-learn",
            metrics=metrics,
            feature_names=[f"pixel_{i}" for i in range(64)],
            dataset_info={
                "name": "digits_recommendation",
                "size": len(X),
                "source": "UCI ML Repository (sklearn.datasets.load_digits)",
                "description": "Real digit images for similarity-based recommendation",
                "image_size": "8x8 pixels",
                "distance_metric": distance_metric,
            },
        )
        return metrics

    def _save_and_register_model(
        self,
        model,
        model_name: str,
        version: str,
        algorithm: str,
        framework: str,
        metrics: Dict[str, Any],
        feature_names: list,
        dataset_info: dict,
    ):
        if self.use_mlflow:
            with mlflow.start_run(run_name=f"{model_name}_v{version}"):
                tags = {
                    "model_name": model_name,
                    "model_version": version,
                    "algorithm": algorithm,
                    "framework": framework,
                    "dataset": dataset_info["name"],
                    "data_source": "real_public_dataset",
                    "environment": "production",
                }
                if "distance_metric" in dataset_info:
                    tags["distance_metric"] = dataset_info["distance_metric"]
                mlflow.set_tags(tags)
                params = {
                    "algorithm": algorithm,
                    "framework": framework,
                    "dataset_name": dataset_info["name"],
                    "dataset_size": dataset_info["size"],
                    "n_features": len(feature_names),
                }
                try:
                    if isinstance(model, dict) and "model" in model:
                        actual_model = model["model"]
                        if hasattr(actual_model, "get_params"):
                            model_params = actual_model.get_params()
                            important_params = {
                                k: v
                                for k, v in model_params.items()
                                if not k.startswith("_")
                                and (not callable(v))
                                and (v is not None)
                                and (not isinstance(v, (list, dict, np.ndarray)))
                            }
                            params.update(important_params)
                    elif hasattr(model, "get_params"):
                        model_params = model.get_params()
                        important_params = {
                            k: v
                            for k, v in model_params.items()
                            if not k.startswith("_")
                            and (not callable(v))
                            and (v is not None)
                            and (not isinstance(v, (list, dict, np.ndarray)))
                        }
                        params.update(important_params)
                except Exception as e:
                    logger.warning(f"Could not extract model params: {e}")
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_dict(dataset_info, "dataset_info.json")
                mlflow.log_dict({"features": feature_names}, "features.json")
                try:
                    model_to_log = (
                        model
                        if not isinstance(model, dict)
                        else model.get("model", model)
                    )
                    mlflow.sklearn.log_model(
                        model_to_log,
                        artifact_path="model",
                        registered_model_name=model_name,
                    )
                    logger.info(f" Logged to MLflow: {model_name} v{version}")
                except Exception as e:
                    logger.warning(f"Could not log model to MLflow: {e}")
        model_dir = self.models_dir / model_name / f"v{version}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)
        features_path = model_dir / "features.json"
        with open(features_path, "w") as f:
            json.dump(feature_names, f, indent=2)
        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f" Saved artifacts to {model_dir}")
        self.registry.register_model(
            model_name=model_name,
            version=version,
            model_metadata={
                "algorithm": algorithm,
                "framework": framework,
                "model_path": str(model_path),
                "created_by": "multi-model-training-pipeline-REAL-DATA",
                "metrics": metrics,
                "dataset_name": dataset_info["name"],
                "dataset_size": dataset_info["size"],
                "dataset_source": dataset_info["source"],
                "description": f"{algorithm} on REAL {dataset_info['name']} dataset",
                "tags": [model_name, "production", "real-data", "public-dataset"],
                "team": "data-science",
                "project": "governops-production",
            },
        )
        model_card = create_model_card(
            model_name=model_name,
            version=version,
            author="GovernOps - Real Data Pipeline",
            description=f"Production {algorithm} model trained on REAL public dataset: {dataset_info['name']}",
            training_data_info=dataset_info,
            hyperparameters={"algorithm": algorithm, "framework": framework},
            performance_metrics=metrics,
            deployment_info={
                "target_environment": "production",
                "serving_framework": "FastAPI",
            },
        )
        card_path = self.model_cards_dir / f"{model_name}_v{version}.json"
        save_model_card_to_file(model_card, str(card_path))
        self.models_trained.append(
            {"name": model_name, "version": version, "metrics": metrics}
        )
        logger.info(f" Model registered: {model_name} v{version}\n")

    def train_all_models(self):
        logger.info("\n" + "" + "=" * 78 + "")
        logger.info(
            ""
            + " " * 15
            + "MULTI-MODEL TRAINING - REAL PUBLIC DATASETS"
            + " " * 20
            + ""
        )
        logger.info("" + "=" * 78 + "\n")
        start_time = datetime.now()
        training_steps = [
            ("credit_risk", self.train_credit_risk_model),
            ("churn_prediction", self.train_churn_prediction_model),
            ("price_prediction", self.train_price_prediction_model),
            ("anomaly_detection", self.train_anomaly_detection_model),
            ("customer_segmentation", self.train_customer_segmentation_model),
            ("image_classification", self.train_image_classification_model),
            ("sentiment_analysis", self.train_sentiment_analysis_model),
            ("demand_forecasting", self.train_demand_forecasting_model),
            ("recommendation", self.train_recommendation_model),
        ]
        failed_models = []
        for model_name, train_fn in training_steps:
            try:
                train_fn()
            except Exception as e:
                logger.error(
                    f" {model_name} failed, continuing with remaining models: {e}"
                )
                failed_models.append(model_name)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("\n" + "=" * 80)
        if failed_models:
            logger.info(f"TRAINING COMPLETE WITH {len(failed_models)} FAILURE(S)")
        else:
            logger.info("ALL 9 MODELS TRAINED SUCCESSFULLY WITH REAL PUBLIC DATASETS!")
        logger.info("=" * 80)
        logger.info(f"Total Training Time: {duration:.2f} seconds")
        logger.info(f"Models Registered in Database: {len(self.models_trained)}")
        logger.info("\nModels Summary:")
        for model_info in self.models_trained:
            logger.info(f" {model_info['name']} v{model_info['version']}")
        if failed_models:
            logger.info("\nFailed Models:")
            for name in failed_models:
                logger.info(f" {name}")
        logger.info("\nDatasets Used (ALL REAL PUBLIC DATA):")
        logger.info("  • Wisconsin Breast Cancer (UCI ML Repository)")
        logger.info("  • Wine Quality (UCI ML Repository)")
        logger.info("  • California Housing (1990 US Census)")
        logger.info("  • Iris (Fisher's 1936 dataset)")
        logger.info("  • Digits (UCI ML Repository)")
        logger.info("  • Diabetes (sklearn medical dataset)")
        logger.info("=" * 80 + "\n")
        return len(failed_models) == 0


def main():
    trainer = MultiModelTrainer()
    success = trainer.train_all_models()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
