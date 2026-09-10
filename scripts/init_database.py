import os
import sys
import argparse
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.database.connection import get_db_connection
from src.database.models import Base, ModelMetadata, DeploymentHistory, DriftRecord
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_database_if_not_exists():
    from sqlalchemy import create_engine
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME", "mlops_registry")
    user = os.getenv("DB_USER", "mlops_user")
    password = os.getenv("DB_PASSWORD", "mlops_password")
    try:
        conn = psycopg2.connect(
            host=host, port=port, user=user, password=password, database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
        exists = cursor.fetchone()
        if not exists:
            cursor.execute(f"CREATE DATABASE {database}")
            logger.info(f"Created database: {database}")
        else:
            logger.info(f"Database already exists: {database}")
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error creating database: {str(e)}")
        raise


def init_database(reset: bool = False):
    logger.info("Initializing database...")
    try:
        db = get_db_connection()
        if not db.health_check():
            raise Exception("Database health check failed")
        logger.info("Database connection successful")
        if reset:
            logger.warning("RESETTING DATABASE - Dropping all tables...")
            Base.metadata.drop_all(db.engine)
            logger.warning("All tables dropped")
        logger.info("Creating database tables...")
        Base.metadata.create_all(db.engine)
        with db.engine.connect() as conn:
            result = conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            )
            tables = [row[0] for row in result]
            logger.info(f"Created tables: {', '.join(tables)}")
        logger.info("Database initialization complete!")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise


def seed_example_data():
    logger.info("Seeding example data...")
    try:
        from datetime import datetime, timedelta
        from src.models.model_registry import ModelRegistry

        registry = ModelRegistry()
        models_data = [
            {
                "name": "fraud_detection",
                "version": "1.0.0",
                "metadata": {
                    "algorithm": "XGBoost",
                    "framework": "xgboost",
                    "framework_version": "1.7.0",
                    "model_path": "s3://mlops-models/fraud_detection/v1.0.0/model.pkl",
                    "created_by": "data-science-team",
                    "team": "risk-ml",
                    "project": "fraud-prevention",
                    "hyperparameters": {
                        "max_depth": 6,
                        "learning_rate": 0.1,
                        "n_estimators": 100,
                    },
                    "metrics": {
                        "accuracy": 0.95,
                        "precision": 0.93,
                        "recall": 0.92,
                        "f1_score": 0.925,
                    },
                    "description": "Fraud detection model for credit card transactions",
                    "tags": ["fraud", "classification", "production"],
                },
            },
            {
                "name": "churn_prediction",
                "version": "2.1.0",
                "metadata": {
                    "algorithm": "RandomForest",
                    "framework": "scikit-learn",
                    "framework_version": "1.3.0",
                    "model_path": "s3://mlops-models/churn_prediction/v2.1.0/model.pkl",
                    "created_by": "ml-platform",
                    "team": "growth-ml",
                    "project": "customer-retention",
                    "hyperparameters": {
                        "n_estimators": 200,
                        "max_depth": 10,
                        "min_samples_split": 5,
                    },
                    "metrics": {
                        "accuracy": 0.88,
                        "precision": 0.86,
                        "recall": 0.84,
                        "auc": 0.91,
                    },
                    "description": "Customer churn prediction model",
                    "tags": ["churn", "classification", "staging"],
                },
            },
            {
                "name": "recommendation",
                "version": "1.5.2",
                "metadata": {
                    "algorithm": "Neural Collaborative Filtering",
                    "framework": "tensorflow",
                    "framework_version": "2.13.0",
                    "model_path": "s3://mlops-models/recommendation/v1.5.2/model.h5",
                    "created_by": "recommendations-team",
                    "team": "personalization",
                    "project": "product-recommendations",
                    "hyperparameters": {
                        "embedding_dim": 64,
                        "layers": [128, 64, 32],
                        "dropout": 0.2,
                    },
                    "metrics": {
                        "ndcg@10": 0.76,
                        "precision@10": 0.65,
                        "recall@10": 0.58,
                    },
                    "description": "Product recommendation neural network",
                    "tags": ["recommendations", "deep-learning", "production"],
                },
            },
        ]
        created_models = []
        for model_data in models_data:
            model = registry.register_model(
                model_name=model_data["name"],
                version=model_data["version"],
                model_metadata=model_data["metadata"],
            )
            created_models.append(model)
            logger.info(f"Created example model: {model.model_name} v{model.version}")
        registry.update_model_status(
            model_name="fraud_detection", version="1.0.0", new_status="production"
        )
        logger.info("Updated fraud_detection to production status")
        logger.info(f"Seeded {len(created_models)} example models")
        logger.info("Example data seeding complete!")
    except Exception as e:
        logger.error(f"Error seeding example data: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Initialize GovernOps database")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate all tables (WARNING: deletes all data)",
    )
    parser.add_argument(
        "--seed", action="store_true", help="Seed database with example data"
    )
    parser.add_argument(
        "--create-db", action="store_true", help="Create database if it does not exist"
    )
    args = parser.parse_args()
    try:
        if args.create_db:
            create_database_if_not_exists()
        init_database(reset=args.reset)
        if args.seed:
            seed_example_data()
        logger.info("=" * 60)
        logger.info("Database setup complete!")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Verify tables: SELECT * FROM model_metadata;")
        logger.info(
            "2. Check connection: python -c 'from src.database import get_db_connection; get_db_connection().health_check()'"
        )
        logger.info("3. Start using ModelRegistry in your code")
    except Exception as e:
        logger.error("Database setup failed!")
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
