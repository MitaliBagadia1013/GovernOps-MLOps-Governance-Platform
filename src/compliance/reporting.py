import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


def generate_compliance_report(
    model_name: str,
    model_version: str,
    model_performance: Dict[str, Any],
    bias_detection_results: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "report_version": "1.0",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "model_performance": model_performance,
        "bias_detection": bias_detection_results,
        "compliance_status": _determine_compliance_status(
            model_performance, bias_detection_results
        ),
    }


def _determine_compliance_status(
    performance: Dict[str, Any], bias: Dict[str, Any]
) -> str:
    bias_detected = bias.get("bias_detected", False)
    accuracy = performance.get("accuracy", 1.0)
    if bias_detected or accuracy < 0.7:
        return "non_compliant"
    if accuracy < 0.85:
        return "review_required"
    return "compliant"


def save_report(report: Dict[str, Any], file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(report, f, indent=2)


def automate_compliance_reporting(
    model_name: str,
    model_version: str,
    model_performance: Dict[str, Any],
    bias_detection_results: Dict[str, Any],
    output_dir: str = "compliance_reports",
) -> str:
    report = generate_compliance_report(
        model_name, model_version, model_performance, bias_detection_results
    )
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(
        output_dir, f"{model_name}_v{model_version}_compliance_{timestamp}.json"
    )
    save_report(report, file_path)
    return file_path
