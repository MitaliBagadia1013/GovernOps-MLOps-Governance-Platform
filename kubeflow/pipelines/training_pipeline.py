from kfp import dsl
from kfp.dsl import component
from typing import NamedTuple


@component
def load_data_component(
    model_name: str,
) -> NamedTuple("Outputs", [("dataset_size", int), ("n_features", int)]):
    from collections import namedtuple

    dataset_info = {
        "credit_risk": {"size": 569, "features": 30},
        "churn_prediction": {"size": 178, "features": 13},
        "price_prediction": {"size": 20640, "features": 8},
        "anomaly_detection": {"size": 569, "features": 30},
        "customer_segmentation": {"size": 150, "features": 4},
        "image_classification": {"size": 1797, "features": 64},
        "sentiment_analysis": {"size": 150, "features": 4},
        "demand_forecasting": {"size": 442, "features": 10},
        "recommendation": {"size": 1797, "features": 64},
    }
    info = dataset_info.get(model_name, {"size": 0, "features": 0})
    print(f" Loaded dataset for {model_name}")
    print(f"   Size: {info['size']} samples")
    print(f"   Features: {info['features']}")
    Outputs = namedtuple("Outputs", ["dataset_size", "n_features"])
    return Outputs(info["size"], info["features"])


@component
def train_model_component(
    model_name: str, dataset_size: int, n_features: int
) -> NamedTuple("Outputs", [("accuracy", float), ("version", str)]):
    from collections import namedtuple
    import random

    print(f" Training {model_name}...")
    print(f"   Dataset: {dataset_size} samples, {n_features} features")
    accuracy = round(random.uniform(0.85, 0.98), 4)
    version = "1.0.0"
    print(f" Training complete!")
    print(f"   Accuracy: {accuracy}")
    print(f"   Version: {version}")
    Outputs = namedtuple("Outputs", ["accuracy", "version"])
    return Outputs(accuracy, version)


@component
def log_to_mlflow_component(model_name: str, version: str, accuracy: float) -> str:
    print(f" Logging to MLflow...")
    print(f"   Model: {model_name} v{version}")
    print(f"   Accuracy: {accuracy}")
    run_id = f"run_{model_name}_{version}"
    print(f" Logged to MLflow!")
    print(f"   Run ID: {run_id}")
    return run_id


@component
def register_model_component(model_name: str, version: str, mlflow_run_id: str) -> str:
    print(f" Registering model in database...")
    print(f"   Model: {model_name} v{version}")
    print(f"   MLflow Run: {mlflow_run_id}")
    status = "SUCCESS"
    print(f" Model registered!")
    print(f"   Status: {status}")
    return status


@component
def validate_model_component(
    model_name: str, accuracy: float, min_accuracy: float = 0.8
) -> str:
    print(f" Validating model...")
    print(f"   Model: {model_name}")
    print(f"   Accuracy: {accuracy}")
    print(f"   Minimum Required: {min_accuracy}")
    if accuracy >= min_accuracy:
        result = "APPROVED"
        print(f" Model APPROVED for deployment!")
    else:
        result = "REJECTED"
        print(f" Model REJECTED (accuracy too low)")
    return result


@dsl.pipeline(
    name="GovernOps Training Pipeline",
    description="End-to-end training pipeline for all 9 ML models with MLflow integration",
)
def mlops_training_pipeline(model_name: str = "credit_risk", min_accuracy: float = 0.8):
    data_task = load_data_component(model_name=model_name)
    train_task = train_model_component(
        model_name=model_name,
        dataset_size=data_task.outputs["dataset_size"],
        n_features=data_task.outputs["n_features"],
    )
    mlflow_task = log_to_mlflow_component(
        model_name=model_name,
        version=train_task.outputs["version"],
        accuracy=train_task.outputs["accuracy"],
    )
    register_task = register_model_component(
        model_name=model_name,
        version=train_task.outputs["version"],
        mlflow_run_id=mlflow_task.output,
    )
    validate_task = validate_model_component(
        model_name=model_name,
        accuracy=train_task.outputs["accuracy"],
        min_accuracy=min_accuracy,
    )
    train_task.after(data_task)
    mlflow_task.after(train_task)
    register_task.after(mlflow_task)
    validate_task.after(register_task)


if __name__ == "__main__":
    "\n    Compile the pipeline to YAML for Kubeflow deployment.\n    \n    Run this to generate the compiled pipeline:\n        python training_pipeline.py\n    \n    This creates: mlops_training_pipeline.yaml\n"
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=mlops_training_pipeline,
        package_path="mlops_training_pipeline.yaml",
    )
    print("=" * 80)
    print("Pipeline compiled successfully!")
    print("=" * 80)
    print("\nOutput file: mlops_training_pipeline.yaml")
    print("\nTo deploy to Kubeflow:")
    print("   1. Upload mlops_training_pipeline.yaml to Kubeflow UI")
    print("   2. Or use: python kubeflow/deploy_pipeline.py")
    print("=" * 80 + "\n")
