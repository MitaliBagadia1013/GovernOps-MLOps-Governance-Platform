from datetime import datetime
import json
from typing import Dict, Any, Optional


class ModelCard:

    def __init__(
        self, model_name, version, description, metrics, bias_report=None, **kwargs
    ):
        self.model_name = model_name
        self.version = version
        self.description = description
        self.metrics = metrics
        self.bias_report = bias_report or {}
        self.created_at = datetime.now().isoformat()
        self.author = kwargs.get("author", "Unknown")
        self.training_data_info = kwargs.get("training_data_info", {})
        self.hyperparameters = kwargs.get("hyperparameters", {})
        self.deployment_info = kwargs.get("deployment_info", {})
        self.performance_metrics = kwargs.get("performance_metrics", {})

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "metrics": self.metrics,
            "performance_metrics": self.performance_metrics,
            "bias_report": self.bias_report,
            "training_data_info": self.training_data_info,
            "hyperparameters": self.hyperparameters,
            "deployment_info": self.deployment_info,
            "created_at": self.created_at,
        }


def create_model_card(
    model_name: str,
    version: str,
    description: str = "",
    metrics: Optional[Dict[str, Any]] = None,
    bias_report: Optional[Dict[str, Any]] = None,
    author: str = "Unknown",
    training_data_info: Optional[Dict[str, Any]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    performance_metrics: Optional[Dict[str, Any]] = None,
    deployment_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model_card = ModelCard(
        model_name=model_name,
        version=version,
        description=description,
        metrics=metrics or {},
        bias_report=bias_report,
        author=author,
        training_data_info=training_data_info or {},
        hyperparameters=hyperparameters or {},
        performance_metrics=performance_metrics or {},
        deployment_info=deployment_info or {},
    )
    return model_card.to_dict()


def update_model_card(model_card, metrics=None, bias_report=None):
    if metrics:
        model_card["metrics"] = metrics
    if bias_report:
        model_card["bias_report"] = bias_report
    return model_card


def save_model_card_to_file(model_card, file_path):
    with open(file_path, "w") as f:
        json.dump(model_card, f, indent=4)


def load_model_card_from_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
