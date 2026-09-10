import logging
from typing import Dict, Any, Optional
from datetime import datetime


class SelfHealingSystem:

    def __init__(
        self,
        error_rate_threshold: float = 0.1,
        latency_threshold_ms: float = 500,
        min_requests: int = 100,
    ):
        self.error_rate_threshold = error_rate_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.min_requests = min_requests
        self.logger = logging.getLogger(__name__)
        self.health_history: Dict[str, list] = {}

    def monitor_canary_health(
        self,
        deployment_name: str,
        canary_metrics: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.logger.info(f"Monitoring canary health for: {deployment_name}")
        canary_error_rate = canary_metrics.get("error_rate", 0)
        baseline_error_rate = baseline_metrics.get("error_rate", 0)
        canary_latency = canary_metrics.get("avg_latency_ms", 0)
        canary_requests = canary_metrics.get("request_count", 0)
        error_rate_increase = canary_error_rate - baseline_error_rate
        health_status = {
            "deployment_name": deployment_name,
            "timestamp": datetime.now().isoformat(),
            "canary_healthy": True,
            "action": "continue",
            "reasons": [],
        }
        if canary_requests < self.min_requests:
            health_status["action"] = "wait"
            health_status["reasons"].append(
                f"Insufficient requests: {canary_requests}/{self.min_requests}"
            )
            self.logger.info(
                f"Waiting for more data: {canary_requests}/{self.min_requests} requests"
            )
            return health_status
        if error_rate_increase > self.error_rate_threshold:
            health_status["canary_healthy"] = False
            health_status["action"] = "rollback"
            health_status["reasons"].append(
                f"Error rate increase: {error_rate_increase:.2%} > {self.error_rate_threshold:.2%}"
            )
            self.logger.error(
                f" Self-Healing: Error rate increase detected! Canary: {canary_error_rate:.2%}, Baseline: {baseline_error_rate:.2%}"
            )
        if canary_latency > self.latency_threshold_ms:
            health_status["canary_healthy"] = False
            health_status["action"] = "rollback"
            health_status["reasons"].append(
                f"High latency: {canary_latency}ms > {self.latency_threshold_ms}ms"
            )
            self.logger.error(
                f" Self-Healing: High latency detected! Canary latency: {canary_latency}ms"
            )
        if health_status["canary_healthy"]:
            health_status["action"] = "promote"
            health_status["reasons"].append("All metrics within acceptable thresholds")
            self.logger.info(
                f" Canary is healthy. Ready for promotion to 100% traffic."
            )
        self._record_health_check(
            deployment_name, health_status, canary_metrics, baseline_metrics
        )
        return health_status

    def rollback_deployment(self, deployment_name: str, reason: str) -> Dict[str, Any]:
        self.logger.critical(
            f" Self-Healing: Performing automatic rollback for {deployment_name}. Reason: {reason}"
        )
        rollback_result = {
            "deployment_name": deployment_name,
            "action": "rollback",
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "status": "success",
        }
        self.logger.info(f" Rollback completed for {deployment_name}")
        return rollback_result

    def promote_deployment(self, deployment_name: str) -> Dict[str, Any]:
        self.logger.info(
            f" Self-Healing: Promoting canary to 100% traffic for {deployment_name}"
        )
        promotion_result = {
            "deployment_name": deployment_name,
            "action": "promote",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }
        self.logger.info(f" Promotion completed for {deployment_name}")
        return promotion_result

    def _record_health_check(
        self,
        deployment_name: str,
        health_status: Dict[str, Any],
        canary_metrics: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
    ):
        if deployment_name not in self.health_history:
            self.health_history[deployment_name] = []
        self.health_history[deployment_name].append(
            {
                "timestamp": health_status["timestamp"],
                "canary_healthy": health_status["canary_healthy"],
                "action": health_status["action"],
                "canary_metrics": canary_metrics,
                "baseline_metrics": baseline_metrics,
            }
        )
        self.health_history[deployment_name] = self.health_history[deployment_name][
            -100:
        ]

    def get_health_history(self, deployment_name: str) -> list:
        return self.health_history.get(deployment_name, [])

    def get_deployment_summary(self, deployment_name: str) -> Dict[str, Any]:
        history = self.health_history.get(deployment_name, [])
        if not history:
            return {
                "deployment_name": deployment_name,
                "total_checks": 0,
                "message": "No health checks recorded",
            }
        total_checks = len(history)
        healthy_checks = sum((1 for h in history if h["canary_healthy"]))
        rollback_count = sum((1 for h in history if h["action"] == "rollback"))
        promote_count = sum((1 for h in history if h["action"] == "promote"))
        return {
            "deployment_name": deployment_name,
            "total_checks": total_checks,
            "healthy_checks": healthy_checks,
            "health_rate": healthy_checks / total_checks,
            "rollback_count": rollback_count,
            "promote_count": promote_count,
            "last_check": history[-1]["timestamp"],
            "last_action": history[-1]["action"],
        }
