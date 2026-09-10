import json
import os
from datetime import datetime, timezone
from typing import Any, Dict
import numpy as np


def compute_bias_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive_feature: np.ndarray
) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "groups": {},
        "demographic_parity_difference": None,
        "disparate_impact_ratio": None,
        "bias_detected": False,
    }
    groups = np.unique(sensitive_feature)
    positive_rates = {}
    for group in groups:
        mask = sensitive_feature == group
        group_true = y_true[mask]
        group_pred = y_pred[mask]
        positive_rate = float(np.mean(group_pred))
        accuracy = float(np.mean(group_true == group_pred))
        results["groups"][str(group)] = {
            "sample_size": int(mask.sum()),
            "positive_rate": positive_rate,
            "accuracy": accuracy,
        }
        positive_rates[str(group)] = positive_rate
    if len(positive_rates) >= 2:
        rates = list(positive_rates.values())
        dpd = max(rates) - min(rates)
        dir_ = min(rates) / max(rates) if max(rates) > 0 else None
        results["demographic_parity_difference"] = round(dpd, 4)
        results["disparate_impact_ratio"] = round(dir_, 4) if dir_ is not None else None
        results["bias_detected"] = dpd > 0.1
    return results


def generate_bias_report(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_feature: np.ndarray,
    output_dir: str = "bias_reports",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    report = compute_bias_metrics(y_true, y_pred, sensitive_feature)
    report["model_name"] = model_name
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"{model_name}_bias_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report_path
