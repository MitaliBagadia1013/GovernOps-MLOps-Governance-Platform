import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
try:
    import kfp
    from kfp import compiler
    from kfp.client import Client
except ImportError:
    print("ERROR: Kubeflow Pipelines SDK not installed!")
    print("Install with: pip install kfp==2.4.0")
    sys.exit(1)
from kubeflow.pipelines.training_pipeline import mlops_training_pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KubeflowDeployer:

    def __init__(
        self,
        host: Optional[str] = None,
        namespace: str = "kubeflow",
        output_dir: str = "./pipeline_builds",
    ):
        self.host = host or os.getenv("KFP_ENDPOINT", "http://localhost:8080")
        self.namespace = namespace
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = None

    def compile_pipeline(
        self,
        pipeline_name: str = "ml-training-pipeline",
        output_filename: Optional[str] = None,
    ) -> Path:
        logger.info("=" * 60)
        logger.info("COMPILING KUBEFLOW PIPELINE")
        logger.info("=" * 60)
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{pipeline_name}_{timestamp}.yaml"
        output_path = self.output_dir / output_filename
        try:
            pipeline_func = mlops_training_pipeline
            logger.info(f"Compiling pipeline to: {output_path}")
            compiler.Compiler().compile(
                pipeline_func=pipeline_func, package_path=str(output_path)
            )
            logger.info(f" Pipeline compiled successfully!")
            logger.info(f" Output file: {output_path}")
            logger.info(f" File size: {output_path.stat().st_size / 1024:.2f} KB")
            return output_path
        except Exception as e:
            logger.error(f" Pipeline compilation failed: {str(e)}")
            raise

    def connect_to_server(self) -> bool:
        logger.info(f"Connecting to Kubeflow server: {self.host}")
        try:
            self.client = Client(host=self.host, namespace=self.namespace)
            experiments = self.client.list_experiments(page_size=1)
            logger.info(f" Connected successfully!")
            logger.info(f" Server has {experiments.total_size or 0} experiments")
            return True
        except Exception as e:
            logger.error(f" Failed to connect to Kubeflow server: {str(e)}")
            logger.error(f"Make sure Kubeflow is running at: {self.host}")
            self.client = None
            return False

    def upload_pipeline(
        self,
        pipeline_path: Path,
        pipeline_name: str = "ML Training Pipeline",
        description: Optional[str] = None,
    ) -> str:
        logger.info("=" * 60)
        logger.info("UPLOADING PIPELINE TO KUBEFLOW")
        logger.info("=" * 60)
        if self.client is None:
            if not self.connect_to_server():
                raise RuntimeError("Cannot upload: Not connected to Kubeflow server")
        if description is None:
            description = (
                f"ML Training Pipeline - Deployed at {datetime.now().isoformat()}"
            )
        try:
            logger.info(f"Uploading: {pipeline_path}")
            pipeline = self.client.upload_pipeline(
                pipeline_package_path=str(pipeline_path),
                pipeline_name=pipeline_name,
                description=description,
            )
            logger.info(f" Pipeline uploaded successfully!")
            logger.info(f" Pipeline ID: {pipeline.pipeline_id}")
            logger.info(f" Pipeline Name: {pipeline.display_name}")
            return pipeline.pipeline_id
        except Exception as e:
            logger.error(f" Pipeline upload failed: {str(e)}")
            raise

    def create_or_get_experiment(self, experiment_name: str) -> str:
        if self.client is None:
            if not self.connect_to_server():
                raise RuntimeError("Cannot create experiment: Not connected to server")
        try:
            experiment = self.client.get_experiment(experiment_name=experiment_name)
            logger.info(f"Using existing experiment: {experiment_name}")
            return experiment.id
        except:
            logger.info(f"Creating new experiment: {experiment_name}")
            experiment = self.client.create_experiment(name=experiment_name)
            return experiment.id

    def run_pipeline(
        self,
        pipeline_id: str,
        experiment_name: str = "Default",
        run_name: Optional[str] = None,
        parameters: Optional[dict] = None,
    ) -> str:
        logger.info("=" * 60)
        logger.info("STARTING PIPELINE RUN")
        logger.info("=" * 60)
        if self.client is None:
            if not self.connect_to_server():
                raise RuntimeError("Cannot run pipeline: Not connected to server")
        experiment_id = self.create_or_get_experiment(experiment_name)
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"ml-training-run-{timestamp}"
        default_params = {
            "models_to_train": "credit_risk,churn_prediction,sentiment_analysis",
            "mlflow_tracking_uri": os.getenv(
                "MLFLOW_TRACKING_URI", "http://mlflow-server:5000"
            ),
            "data_version": "latest",
            "enable_validation": True,
        }
        if parameters:
            default_params.update(parameters)
        try:
            logger.info(f"Experiment: {experiment_name}")
            logger.info(f"Run Name: {run_name}")
            logger.info(f"Parameters: {default_params}")
            run = self.client.run_pipeline(
                experiment_id=experiment_id,
                job_name=run_name,
                pipeline_id=pipeline_id,
                params=default_params,
            )
            logger.info(f" Pipeline run started successfully!")
            logger.info(f" Run ID: {run.id}")
            logger.info(f" View in UI: {self.host}/#/runs/details/{run.id}")
            return run.id
        except Exception as e:
            logger.error(f" Pipeline run failed: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Deploy ML Training Pipeline to Kubeflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\nExamples:\n  # Just compile the pipeline\n  python deploy_pipeline.py --compile-only\n  \n  # Compile and upload to server\n  python deploy_pipeline.py --upload\n  \n  # Compile, upload, and run immediately\n  python deploy_pipeline.py --run --experiment-name "production-training"\n  \n  # Use custom Kubeflow endpoint\n  python deploy_pipeline.py --upload --host http://kubeflow.example.com:8080\n        ',
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile the pipeline, do not upload",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Compile and upload the pipeline to Kubeflow",
    )
    parser.add_argument(
        "--run", action="store_true", help="Compile, upload, and execute the pipeline"
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Kubeflow Pipelines API endpoint (default: from KFP_ENDPOINT env var or http://localhost:8080)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="kubeflow",
        help="Kubernetes namespace (default: kubeflow)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ml-training-experiment",
        help="Experiment name for pipeline runs (default: ml-training-experiment)",
    )
    parser.add_argument(
        "--run-name", type=str, help="Custom name for this pipeline run"
    )
    parser.add_argument(
        "--pipeline-name",
        type=str,
        default="ml-training-pipeline",
        help="Name for the pipeline (default: ml-training-pipeline)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./pipeline_builds",
        help="Directory for compiled pipeline files (default: ./pipeline_builds)",
    )
    args = parser.parse_args()
    if not any([args.compile_only, args.upload, args.run]):
        args.compile_only = True
    try:
        deployer = KubeflowDeployer(
            host=args.host, namespace=args.namespace, output_dir=args.output_dir
        )
        logger.info("Step 1: Compiling pipeline...")
        compiled_path = deployer.compile_pipeline(pipeline_name=args.pipeline_name)
        pipeline_id = None
        if args.upload or args.run:
            logger.info("\nStep 2: Uploading pipeline to Kubeflow...")
            pipeline_id = deployer.upload_pipeline(
                pipeline_path=compiled_path, pipeline_name=args.pipeline_name
            )
        if args.run:
            if pipeline_id is None:
                logger.error("Cannot run: Pipeline not uploaded")
                sys.exit(1)
            logger.info("\nStep 3: Starting pipeline run...")
            run_id = deployer.run_pipeline(
                pipeline_id=pipeline_id,
                experiment_name=args.experiment_name,
                run_name=args.run_name,
            )
        logger.info("\n" + "=" * 60)
        logger.info("DEPLOYMENT SUCCESSFUL!")
        logger.info("=" * 60)
        logger.info(f"Compiled file: {compiled_path}")
        if pipeline_id:
            logger.info(f"Pipeline ID: {pipeline_id}")
        if args.run:
            logger.info(f"Run ID: {run_id}")
            logger.info(f"\nView run: {deployer.host}/#/runs/details/{run_id}")
        logger.info("\nNext Steps:")
        if args.compile_only:
            logger.info("  1. Review the compiled YAML file")
            logger.info("  2. Upload with: python deploy_pipeline.py --upload")
        elif args.upload:
            logger.info("  1. Open Kubeflow UI to view the pipeline")
            logger.info(
                "  2. Run manually from UI or use: python deploy_pipeline.py --run"
            )
        else:
            logger.info("  1. Monitor the run in Kubeflow UI")
            logger.info("  2. Check MLflow for logged models and metrics")
    except KeyboardInterrupt:
        logger.warning("\nDeployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nDeployment failed: {str(e)}")
        logger.exception("Full error traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
