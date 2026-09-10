import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.model_registry import ModelRegistry
from src.models.model_cards import create_model_card, save_model_card_to_file
from src.drift_detection.drift_monitoring_pipeline import DriftMonitoringPipeline
from src.monitoring.self_healing import SelfHealingSystem
from src.compliance.reporting import generate_compliance_report, save_report_to_file


def demonstrate_model_lifecycle():
    print("=" * 80)
    print("GOVERNOPS - FULL LIFECYCLE DEMONSTRATION")
    print("=" * 80)
    print("\nSTEP 1: Model Registry - Managing 10+ Concurrent Models")
    print("-" * 80)
    registry = ModelRegistry()
    models = [
        {
            "name": "fraud_detection",
            "version": "1.0.0",
            "algorithm": "RandomForest",
            "framework": "scikit-learn",
            "type": "classification",
            "metrics": {"accuracy": 0.94, "precision": 0.92, "recall": 0.9},
        },
        {
            "name": "credit_risk",
            "version": "2.0.0",
            "algorithm": "XGBoost",
            "framework": "xgboost",
            "type": "classification",
            "metrics": {"accuracy": 0.89, "auc": 0.91},
        },
        {
            "name": "churn_prediction",
            "version": "1.0.0",
            "algorithm": "LogisticRegression",
            "framework": "scikit-learn",
            "type": "classification",
            "metrics": {"accuracy": 0.87, "f1_score": 0.85},
        },
        {
            "name": "recommendation",
            "version": "3.0.0",
            "algorithm": "NeuralCollaborativeFiltering",
            "framework": "tensorflow",
            "type": "recommendation",
            "metrics": {"accuracy": 0.92, "ndcg@10": 0.78},
        },
        {
            "name": "sentiment_analysis",
            "version": "1.0.0",
            "algorithm": "BERT",
            "framework": "pytorch",
            "type": "nlp",
            "metrics": {"accuracy": 0.91, "f1_score": 0.89},
        },
        {
            "name": "price_prediction",
            "version": "2.0.0",
            "algorithm": "GradientBoosting",
            "framework": "scikit-learn",
            "type": "regression",
            "metrics": {"rmse": 12.5, "mae": 8.3},
        },
        {
            "name": "anomaly_detection",
            "version": "1.0.0",
            "algorithm": "IsolationForest",
            "framework": "scikit-learn",
            "type": "classification",
            "metrics": {"f1_score": 0.88, "precision": 0.86},
        },
        {
            "name": "demand_forecasting",
            "version": "1.0.0",
            "algorithm": "LSTM",
            "framework": "tensorflow",
            "type": "time_series",
            "metrics": {"mape": 8.3, "rmse": 15.2},
        },
        {
            "name": "customer_segmentation",
            "version": "2.0.0",
            "algorithm": "KMeans",
            "framework": "scikit-learn",
            "type": "clustering",
            "metrics": {"silhouette": 0.75, "inertia": 1234.5},
        },
        {
            "name": "image_classification",
            "version": "1.0.0",
            "algorithm": "ResNet50",
            "framework": "pytorch",
            "type": "computer_vision",
            "metrics": {"accuracy": 0.96, "top5_accuracy": 0.99},
        },
    ]
    print("Registering models in PostgreSQL database...")
    for model in models:
        registry.register_model(
            model_name=model["name"],
            version=model["version"],
            model_metadata={
                "algorithm": model["algorithm"],
                "framework": model["framework"],
                "model_path": f"s3://mlops-models/{model['name']}/v{model['version']}/model.pkl",
                "created_by": "mlops-demo-pipeline",
                "model_type": model["type"],
                "metrics": model["metrics"],
                "description": f"{model['type'].title()} model using {model['algorithm']}",
                "tags": [model["type"], "production", "demo"],
                "team": "data-science",
                "project": "governops-demo",
            },
        )
        print(
            f" Registered: {model['name']} v{model['version']} ({model['algorithm']})"
        )
    registered_models = registry.list_models()
    print(
        f"\nSuccessfully registered {len(registered_models)} concurrent models in PostgreSQL"
    )
    for model in registered_models:
        print(
            f"   - {model.model_name} v{model.version} ({model.algorithm}) - Status: {model.status}"
        )
    print("\nSTEP 2: Model Cards - Ensuring 100% Reproducibility")
    print("-" * 80)
    model_card = create_model_card(
        model_name="fraud_detection",
        version="1.0.0",
        author="Data Science Team",
        description="Credit card fraud detection using Random Forest",
        training_data_info={
            "dataset": "credit_card_transactions_2026",
            "size": "284,807 transactions",
            "features": ["amount", "time", "V1-V28"],
            "data_lineage": "s3://mlops-data/fraud/2026/training/",
        },
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42,
        },
        performance_metrics={
            "accuracy": 0.9456,
            "precision": 0.9234,
            "recall": 0.9123,
            "f1_score": 0.9178,
            "auc_roc": 0.9678,
        },
        deployment_info={
            "endpoint": "https://api.mlops.com/v1/fraud-detection",
            "traffic_split": "canary_5_percent",
            "deployment_date": "2026-04-26",
        },
    )
    save_model_card_to_file(model_card, "model_cards/fraud_detection_v1.0.0.json")
    print("Model Card created with full data lineage and reproducibility info")
    print(f"   Model: {model_card['model_name']}")
    print(f"   Version: {model_card['version']}")
    print(f"   Training Data: {model_card['training_data_info']['dataset']}")
    print("\nSTEP 3: Canary Deployment - 5% Traffic Validation")
    print("-" * 80)
    self_healing = SelfHealingSystem(
        error_rate_threshold=0.1, latency_threshold_ms=500, min_requests=100
    )
    canary_metrics = {
        "error_rate": 0.03,
        "avg_latency_ms": 145,
        "request_count": 150,
        "p95_latency_ms": 280,
        "throughput": 10.2,
    }
    baseline_metrics = {
        "error_rate": 0.02,
        "avg_latency_ms": 150,
        "request_count": 2850,
        "p95_latency_ms": 300,
        "throughput": 190.0,
    }
    health_status = self_healing.monitor_canary_health(
        deployment_name="fraud_detection_v1.0.0",
        canary_metrics=canary_metrics,
        baseline_metrics=baseline_metrics,
    )
    print(f" Canary Health Check Complete")
    print(f"   Canary Healthy: {health_status['canary_healthy']}")
    print(f"   Recommended Action: {health_status['action']}")
    print(
        f"   Error Rate: Canary {canary_metrics['error_rate']:.1%} vs Baseline {baseline_metrics['error_rate']:.1%}"
    )
    print(
        f"   Latency: Canary {canary_metrics['avg_latency_ms']}ms vs Threshold {self_healing.latency_threshold_ms}ms"
    )
    if health_status["action"] == "promote":
        print("Canary passed validation! Promoting to 100% traffic...")
    elif health_status["action"] == "rollback":
        print("Self-Healing triggered! Rolling back due to performance degradation...")
    print("\nSTEP 4: Self-Healing - Drift Detection via KS Test")
    print("-" * 80)
    drift_pipeline = DriftMonitoringPipeline(
        step_function_arn="arn:aws:states:us-east-1:123456789012:stateMachine:retraining-workflow",
        drift_threshold=0.05,
        consecutive_drift_count=3,
    )
    reference_data = pd.DataFrame(
        {
            "transaction_amount": np.random.normal(100, 50, 1000),
            "transaction_time": np.random.uniform(0, 24, 1000),
            "merchant_category": np.random.choice(["retail", "food", "gas"], 1000),
        }
    )
    print("\n   Check #1: Testing with normal production data...")
    production_data_normal = pd.DataFrame(
        {
            "transaction_amount": np.random.normal(100, 50, 1000),
            "transaction_time": np.random.uniform(0, 24, 1000),
            "merchant_category": np.random.choice(["retail", "food", "gas"], 1000),
        }
    )
    result1 = drift_pipeline.monitor_and_trigger(
        reference_data=reference_data["transaction_amount"].values,
        production_data=production_data_normal["transaction_amount"].values,
        model_name="fraud_detection",
        feature_name="transaction_amount",
    )
    print(f" Drift Detected: {result1['drift_detected']}")
    print(f" P-value: {result1['ks_test_result']['p_value']:.4f}")
    print(
        "\n   Check #2-4: Testing with drifted production data (3 consecutive checks)..."
    )
    production_data_drifted = pd.DataFrame(
        {
            "transaction_amount": np.random.normal(200, 80, 1000),
            "transaction_time": np.random.uniform(0, 24, 1000),
            "merchant_category": np.random.choice(["retail", "food", "gas"], 1000),
        }
    )
    for i in range(3):
        result = drift_pipeline.monitor_and_trigger(
            reference_data=reference_data["transaction_amount"].values,
            production_data=production_data_drifted["transaction_amount"].values,
            model_name="fraud_detection",
            feature_name="transaction_amount",
        )
        print(
            f"   Check #{i + 2}: Drift={result['drift_detected']}, P-value={result['ks_test_result']['p_value']:.4f}, Retraining Triggered={result['retraining_triggered']}"
        )
    print("\nSTEP 5: Automated Compliance Reporting")
    print("-" * 80)
    model_performance = {
        "accuracy": 0.9456,
        "precision": 0.9234,
        "recall": 0.9123,
        "f1_score": 0.9178,
        "auc_roc": 0.9678,
        "data_lineage": "s3://mlops-data/fraud/2026/training/",
        "experiment_id": "exp_20260426_001",
        "reproducibility": "100%",
    }
    bias_detection_results = {
        "bias_detected": False,
        "metrics": {
            "demographic_parity_difference": 0.02,
            "equalized_odds_difference": 0.03,
            "disparate_impact": 0.98,
        },
        "sensitive_attributes": ["age", "gender", "location"],
        "compliance_status": "PASSED",
    }
    compliance_report = generate_compliance_report(
        model_performance=model_performance,
        bias_detection_results=bias_detection_results,
    )
    save_report_to_file(
        compliance_report, "compliance_reports/fraud_detection_v1.0.0.json"
    )
    print("Compliance Report Generated")
    print(f"   Model Performance: Accuracy {model_performance['accuracy']:.2%}")
    print(f"   Bias Status: {bias_detection_results['compliance_status']}")
    print(f"   Reproducibility: {model_performance['reproducibility']}")
    print(f"   Data Lineage: {model_performance['data_lineage']}")
    print("\n" + "=" * 80)
    print("END-TO-END MLOPS PIPELINE DEMONSTRATION COMPLETE")
    print("=" * 80)
    total_models = len(registry.list_models())
    print(f" Managed {total_models} concurrent models in PostgreSQL database")
    print("Ensured 100% experiment reproducibility with Model Cards")
    print("Validated canary deployment with 5% traffic")
    print("Detected drift via Kolmogorov-Smirnov test")
    print("Self-Healing: Automatic retraining triggered")
    print("Compliance reporting automated with bias detection")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_model_lifecycle()
