from typing import Any, Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


def split_traffic(
    X: np.ndarray, traffic_percentage: float = 0.05, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))
    split = max(1, int(len(X) * traffic_percentage))
    canary_idx = indices[:split]
    baseline_idx = indices[split:]
    return (X[canary_idx], X[baseline_idx], canary_idx, baseline_idx)


def evaluate_canary(
    canary_model: Any,
    baseline_model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    traffic_percentage: float = 0.05,
    task: str = "classification",
) -> Dict[str, Any]:
    canary_X, baseline_X, c_idx, b_idx = split_traffic(X, traffic_percentage)
    canary_y, baseline_y = (y_true[c_idx], y_true[b_idx])
    if len(canary_X) < 10:
        return {
            "recommendation": "insufficient_data",
            "canary_sample_size": len(canary_X),
            "message": "Need at least 10 canary samples before evaluating.",
        }
    canary_preds = canary_model.predict(canary_X)
    baseline_preds = baseline_model.predict(baseline_X)
    if task == "classification":
        canary_metric = accuracy_score(canary_y, canary_preds)
        baseline_metric = accuracy_score(baseline_y, baseline_preds)
        metric_name = "accuracy"
        higher_is_better = True
    else:
        canary_metric = float(np.sqrt(mean_squared_error(canary_y, canary_preds)))
        baseline_metric = float(np.sqrt(mean_squared_error(baseline_y, baseline_preds)))
        metric_name = "rmse"
        higher_is_better = False
    delta = canary_metric - baseline_metric
    canary_wins = delta > 0 if higher_is_better else delta < 0
    return {
        "canary": {metric_name: round(canary_metric, 4), "sample_size": len(canary_X)},
        "baseline": {
            metric_name: round(baseline_metric, 4),
            "sample_size": len(baseline_X),
        },
        "delta": round(delta, 4),
        "traffic_percentage": traffic_percentage,
        "recommendation": "promote" if canary_wins else "rollback",
    }


def canary_deployment(
    canary_model: Any,
    baseline_model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    traffic_percentage: float = 0.05,
    task: str = "classification",
) -> Dict[str, Any]:
    return evaluate_canary(
        canary_model, baseline_model, X, y_true, traffic_percentage, task
    )
