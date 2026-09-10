import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Union
import logging
from .ks_test import kolmogorov_smirnov_test
from .retraining_trigger import RetrainingTrigger


class DriftMonitoringPipeline:

    def __init__(
        self,
        step_function_arn: str,
        drift_threshold: float = 0.05,
        region_name: str = "us-east-1",
        consecutive_drift_count: int = 3,
    ):
        self.retraining_trigger = RetrainingTrigger(step_function_arn, region_name)
        self.drift_threshold = drift_threshold
        self.consecutive_drift_count = consecutive_drift_count
        self.logger = logging.getLogger(__name__)
        self.drift_history: Dict[str, list] = {}

    def monitor_and_trigger(
        self,
        reference_data: Union[np.ndarray, pd.Series],
        production_data: Union[np.ndarray, pd.Series],
        model_name: str,
        feature_name: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ks_result = kolmogorov_smirnov_test(reference_data, production_data)
        feature_label = f"{model_name}:{feature_name}" if feature_name else model_name
        self.logger.info(
            f"KS Test for {feature_label} - Statistic: {ks_result['ks_statistic']:.4f}, P-value: {ks_result['p_value']:.4f}"
        )
        if feature_label not in self.drift_history:
            self.drift_history[feature_label] = []
        drift_detected = ks_result["drift_detected"]
        self.drift_history[feature_label].append(
            {
                "timestamp": datetime.now().isoformat(),
                "drift_detected": drift_detected,
                "p_value": ks_result["p_value"],
                "statistic": ks_result["ks_statistic"],
            }
        )
        self.drift_history[feature_label] = self.drift_history[feature_label][-10:]
        result = {
            "model_name": model_name,
            "feature_name": feature_name,
            "drift_detected": drift_detected,
            "ks_test_result": ks_result,
            "retraining_triggered": False,
        }
        if drift_detected:
            self.logger.warning(f" Data Drift detected for {feature_label}!")
            recent_drifts = self.drift_history[feature_label][
                -self.consecutive_drift_count :
            ]
            consecutive_drifts = all((d["drift_detected"] for d in recent_drifts))
            if (
                consecutive_drifts
                and len(recent_drifts) == self.consecutive_drift_count
            ):
                self.logger.critical(
                    f" Self-Healing Triggered: {self.consecutive_drift_count} consecutive drift detections for {feature_label}! Automatically triggering retraining via AWS Step Functions..."
                )
                input_data = {
                    "model_name": model_name,
                    "feature_name": feature_name,
                    "drift_metrics": {
                        "p_value": ks_result["p_value"],
                        "statistic": ks_result["ks_statistic"],
                        "threshold": self.drift_threshold,
                    },
                    "timestamp": datetime.now().isoformat(),
                    "consecutive_drift_count": self.consecutive_drift_count,
                    "drift_history": self.drift_history[feature_label],
                }
                if additional_metadata:
                    input_data["metadata"] = additional_metadata
                execution_arn = self.retraining_trigger.trigger_retraining(
                    input_data=input_data,
                    execution_name=f"retraining-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                )
                self.logger.info(
                    f" Self-Healing: Retraining triggered! Execution ARN: {execution_arn}"
                )
                result["retraining_triggered"] = True
                result["execution_arn"] = execution_arn
                self.drift_history[feature_label] = []
            else:
                self.logger.info(
                    f"Drift detected but not triggering yet. Consecutive drifts: {sum((d['drift_detected'] for d in recent_drifts))}/{self.consecutive_drift_count}"
                )
        else:
            self.logger.info(f" No drift detected for {feature_label}")
        return result

    def monitor_multiple_features(
        self,
        reference_data: pd.DataFrame,
        production_data: pd.DataFrame,
        model_name: str,
        features: Optional[list] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if features is None:
            features = reference_data.columns.tolist()
        results = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "features_monitored": len(features),
            "features_with_drift": 0,
            "retraining_triggered": False,
            "feature_results": {},
        }
        for feature in features:
            if (
                feature not in reference_data.columns
                or feature not in production_data.columns
            ):
                self.logger.warning(f"Feature '{feature}' not found in data. Skipping.")
                continue
            feature_result = self.monitor_and_trigger(
                reference_data=reference_data[feature].values,
                production_data=production_data[feature].values,
                model_name=model_name,
                feature_name=feature,
                additional_metadata=additional_metadata,
            )
            results["feature_results"][feature] = feature_result
            if feature_result["drift_detected"]:
                results["features_with_drift"] += 1
            if feature_result["retraining_triggered"]:
                results["retraining_triggered"] = True
                results["execution_arn"] = feature_result["execution_arn"]
                break
        self.logger.info(
            f"Multi-feature monitoring complete: {results['features_with_drift']}/{results['features_monitored']} features with drift"
        )
        return results

    def get_drift_history(
        self, model_name: str, feature_name: Optional[str] = None
    ) -> list:
        feature_label = f"{model_name}:{feature_name}" if feature_name else model_name
        return self.drift_history.get(feature_label, [])

    def reset_drift_history(self, model_name: str, feature_name: Optional[str] = None):
        feature_label = f"{model_name}:{feature_name}" if feature_name else model_name
        if feature_label in self.drift_history:
            self.drift_history[feature_label] = []
            self.logger.info(f"Drift history reset for {feature_label}")

    def get_drift_summary(self) -> Dict[str, Any]:
        summary = {
            "total_monitored": len(self.drift_history),
            "models_with_recent_drift": 0,
            "drift_details": {},
        }
        for feature_label, history in self.drift_history.items():
            if not history:
                continue
            recent_drift = history[-1]["drift_detected"] if history else False
            drift_rate = sum((1 for h in history if h["drift_detected"])) / len(history)
            summary["drift_details"][feature_label] = {
                "recent_drift": recent_drift,
                "drift_rate": drift_rate,
                "total_checks": len(history),
                "last_check": history[-1]["timestamp"] if history else None,
            }
            if recent_drift:
                summary["models_with_recent_drift"] += 1
        return summary
