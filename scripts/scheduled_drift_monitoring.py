import sys
import os
import logging
from datetime import datetime
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src.drift_detection.drift_monitoring_pipeline import DriftMonitoringPipeline
import pandas as pd
import numpy as np
import boto3

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"logs/drift_monitoring_{datetime.now().strftime('%Y%m%d')}.log"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_reference_data(s3_path: str) -> pd.DataFrame:
    logger.info(f"Loading reference data from {s3_path}")
    try:
        s3_client = boto3.client("s3")
        bucket = s3_path.replace("s3://", "").split("/")[0]
        key = "/".join(s3_path.replace("s3://", "").split("/")[1:])
        logger.warning(
            "Using sample reference data - implement S3 loading for production"
        )
        return pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(5, 2, 1000),
                "feature3": np.random.exponential(1, 1000),
            }
        )
    except Exception as e:
        logger.error(f"Failed to load reference data: {str(e)}")
        raise


def load_production_data(s3_path: str, hours: int = 24) -> pd.DataFrame:
    logger.info(f"Loading production data from {s3_path} (last {hours} hours)")
    try:
        s3_client = boto3.client("s3")
        bucket = s3_path.replace("s3://", "").split("/")[0]
        key_prefix = "/".join(s3_path.replace("s3://", "").split("/")[1:])
        logger.warning(
            "Using sample production data - implement S3 loading for production"
        )
        return pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(5, 2, 1000),
                "feature3": np.random.exponential(1, 1000),
            }
        )
    except Exception as e:
        logger.error(f"Failed to load production data: {str(e)}")
        raise


def save_monitoring_results(results: dict, output_path: str):
    logger.info(f"Saving monitoring results to {output_path}")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")


def main():
    logger.info("=" * 80)
    logger.info("STARTING SCHEDULED DRIFT MONITORING JOB")
    logger.info("=" * 80)
    STEP_FUNCTION_ARN = os.getenv(
        "STEP_FUNCTION_ARN",
        "arn:aws:states:us-east-1:123456789012:stateMachine:retraining-workflow",
    )
    MODEL_NAME = os.getenv("MODEL_NAME", "production_model")
    REFERENCE_DATA_PATH = os.getenv(
        "REFERENCE_DATA_PATH", "s3://governops/reference/production_model/"
    )
    PRODUCTION_DATA_PATH = os.getenv(
        "PRODUCTION_DATA_PATH", "s3://governops/production/production_model/"
    )
    logger.info(f"Configuration:")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Step Function ARN: {STEP_FUNCTION_ARN}")
    logger.info(f"  Reference Data: {REFERENCE_DATA_PATH}")
    logger.info(f"  Production Data: {PRODUCTION_DATA_PATH}")
    pipeline = DriftMonitoringPipeline(
        step_function_arn=STEP_FUNCTION_ARN,
        drift_threshold=0.05,
        consecutive_drift_count=3,
    )
    try:
        logger.info("\nLoading data...")
        reference_data = load_reference_data(REFERENCE_DATA_PATH)
        logger.info(f" Reference data loaded: {reference_data.shape}")
        production_data = load_production_data(PRODUCTION_DATA_PATH, hours=24)
        logger.info(f" Production data loaded: {production_data.shape}")
        logger.info(f"\nMonitoring drift for model: {MODEL_NAME}")
        result = pipeline.monitor_multiple_features(
            reference_data=reference_data,
            production_data=production_data,
            model_name=MODEL_NAME,
            additional_metadata={
                "environment": "production",
                "reference_data_path": REFERENCE_DATA_PATH,
                "production_data_path": PRODUCTION_DATA_PATH,
                "monitoring_timestamp": datetime.now().isoformat(),
            },
        )
        logger.info("\n" + "=" * 80)
        logger.info("DRIFT MONITORING RESULTS")
        logger.info("=" * 80)
        logger.info(f"Model: {result['model_name']}")
        logger.info(f"Features Monitored: {result['features_monitored']}")
        logger.info(f"Features with Drift: {result['features_with_drift']}")
        logger.info(f"Retraining Triggered: {result['retraining_triggered']}")
        if result["retraining_triggered"]:
            logger.critical(f" SELF-HEALING ACTIVATED!")
            logger.critical(f"Retraining job triggered via AWS Step Functions")
            logger.critical(f"Execution ARN: {result['execution_arn']}")
        logger.info("\nFeature-level results:")
        for feature, feature_result in result["feature_results"].items():
            drift_status = "DRIFT" if feature_result["drift_detected"] else "OK"
            logger.info(
                f"  {feature}: {drift_status} (p-value: {feature_result['ks_test_result']['p_value']:.4f})"
            )
        summary = pipeline.get_drift_summary()
        logger.info(f"\nDrift Summary:")
        logger.info(f"  Total models monitored: {summary['total_monitored']}")
        logger.info(
            f"  Models with recent drift: {summary['models_with_recent_drift']}"
        )
        output_path = f"monitoring_results/{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_monitoring_results(result, output_path)
        logger.info("\n" + "=" * 80)
        logger.info("DRIFT MONITORING JOB COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        return 0
    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("DRIFT MONITORING JOB FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
