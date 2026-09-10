from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from src.database.connection import get_db_connection
from src.database.models import (
    ModelMetadata,
    DeploymentHistory,
    DriftRecord,
    PerformanceMetric,
    ModelLineage,
)

logger = logging.getLogger(__name__)


class ModelRegistry:

    def __init__(self):
        self.db = get_db_connection()
        logger.info("ModelRegistry initialized with database backend")

    def register_model(
        self, model_name: str, version: str, model_metadata: Dict[str, Any]
    ) -> ModelMetadata:
        required_fields = ["algorithm", "framework", "model_path", "created_by"]
        missing_fields = [f for f in required_fields if f not in model_metadata]
        if missing_fields:
            raise KeyError(f"Missing required fields: {', '.join(missing_fields)}")
        model_id = f"{model_name}_v{version}"
        try:
            with self.db.get_session() as session:
                existing = (
                    session.query(ModelMetadata)
                    .filter_by(model_name=model_name, version=version)
                    .first()
                )
                if existing:
                    raise ValueError(
                        f"Model '{model_name}' version '{version}' already exists. Use update_model() to modify or increment version."
                    )
                model = ModelMetadata(
                    model_name=model_name,
                    version=version,
                    model_id=model_id,
                    algorithm=model_metadata["algorithm"],
                    framework=model_metadata["framework"],
                    framework_version=model_metadata.get("framework_version"),
                    model_path=model_metadata["model_path"],
                    created_by=model_metadata["created_by"],
                    hyperparameters=model_metadata.get("hyperparameters", {}),
                    metrics=model_metadata.get("metrics", {}),
                    description=model_metadata.get("description"),
                    tags=model_metadata.get("tags", []),
                    team=model_metadata.get("team"),
                    project=model_metadata.get("project"),
                    dataset_name=model_metadata.get("dataset_name"),
                    dataset_version=model_metadata.get("dataset_version"),
                    dataset_size=model_metadata.get("dataset_size"),
                    artifact_location=model_metadata.get("artifact_location"),
                    model_size_mb=model_metadata.get("model_size_mb"),
                    training_duration_seconds=model_metadata.get(
                        "training_duration_seconds"
                    ),
                    bias_report=model_metadata.get("bias_report"),
                    status="registered",
                    is_active=True,
                )
                session.add(model)
                session.flush()
                logger.info(
                    f"Registered model: {model_name} v{version} (ID: {model.id})"
                )
                return model
        except IntegrityError as e:
            logger.error(f"Integrity error registering model: {str(e)}")
            raise ValueError(
                f"Model registration failed: duplicate entry or constraint violation"
            )
        except SQLAlchemyError as e:
            logger.error(f"Database error registering model: {str(e)}")
            raise

    def get_model(
        self, model_name: str, version: Optional[str] = None
    ) -> ModelMetadata:
        try:
            with self.db.get_session() as session:
                query = session.query(ModelMetadata).filter_by(model_name=model_name)
                if version:
                    query = query.filter_by(version=version)
                else:
                    query = query.filter_by(is_active=True).order_by(
                        desc(ModelMetadata.created_at)
                    )
                model = query.first()
                if not model:
                    version_str = f" version '{version}'" if version else ""
                    raise ValueError(f"Model '{model_name}'{version_str} not found")
                logger.debug(f"Retrieved model: {model_name} v{model.version}")
                return model
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving model: {str(e)}")
            raise

    def list_models(
        self, status: Optional[str] = None, active_only: bool = True, limit: int = 100
    ) -> List[ModelMetadata]:
        try:
            with self.db.get_session() as session:
                query = session.query(ModelMetadata)
                if active_only:
                    query = query.filter_by(is_active=True)
                if status:
                    query = query.filter_by(status=status)
                query = query.order_by(desc(ModelMetadata.created_at)).limit(limit)
                models = query.all()
                logger.debug(f"Listed {len(models)} models")
                return models
        except SQLAlchemyError as e:
            logger.error(f"Database error listing models: {str(e)}")
            raise

    def update_model(
        self, model_name: str, version: str, updates: Dict[str, Any]
    ) -> ModelMetadata:
        try:
            with self.db.get_session() as session:
                model = (
                    session.query(ModelMetadata)
                    .filter_by(model_name=model_name, version=version)
                    .first()
                )
                if not model:
                    raise ValueError(f"Model '{model_name}' v{version} not found")
                allowed_updates = [
                    "status",
                    "description",
                    "tags",
                    "metrics",
                    "is_active",
                    "compliance_status",
                    "bias_report",
                ]
                for key, value in updates.items():
                    if key in allowed_updates:
                        setattr(model, key, value)
                    else:
                        logger.warning(f"Ignoring update to non-updatable field: {key}")
                logger.info(f"Updated model: {model_name} v{version}")
                return model
        except SQLAlchemyError as e:
            logger.error(f"Database error updating model: {str(e)}")
            raise

    def update_model_status(
        self, model_name: str, version: str, new_status: str
    ) -> ModelMetadata:
        valid_statuses = [
            "registered",
            "staging",
            "production",
            "archived",
            "deprecated",
        ]
        if new_status not in valid_statuses:
            raise ValueError(
                f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
            )
        return self.update_model(model_name, version, {"status": new_status})

    def get_production_models(self) -> List[ModelMetadata]:
        return self.list_models(status="production", active_only=True)

    def search_models(self, search_criteria: Dict[str, Any]) -> List[ModelMetadata]:
        try:
            with self.db.get_session() as session:
                query = session.query(ModelMetadata).filter_by(is_active=True)
                if "team" in search_criteria:
                    query = query.filter_by(team=search_criteria["team"])
                if "project" in search_criteria:
                    query = query.filter_by(project=search_criteria["project"])
                if "framework" in search_criteria:
                    query = query.filter_by(framework=search_criteria["framework"])
                if "algorithm" in search_criteria:
                    query = query.filter_by(algorithm=search_criteria["algorithm"])
                if "tags" in search_criteria:
                    for tag in search_criteria["tags"]:
                        query = query.filter(ModelMetadata.tags.contains([tag]))
                models = query.order_by(desc(ModelMetadata.created_at)).all()
                logger.debug(f"Search found {len(models)} models")
                return models
        except SQLAlchemyError as e:
            logger.error(f"Database error searching models: {str(e)}")
            raise

    def delete_model(
        self, model_name: str, version: str, hard_delete: bool = False
    ) -> None:
        try:
            with self.db.get_session() as session:
                model = (
                    session.query(ModelMetadata)
                    .filter_by(model_name=model_name, version=version)
                    .first()
                )
                if not model:
                    raise ValueError(f"Model '{model_name}' v{version} not found")
                if hard_delete:
                    session.delete(model)
                    logger.warning(f"Hard deleted model: {model_name} v{version}")
                else:
                    model.status = "archived"
                    model.is_active = False
                    logger.info(f"Archived model: {model_name} v{version}")
        except SQLAlchemyError as e:
            logger.error(f"Database error deleting model: {str(e)}")
            raise

    def get_model_count(self) -> Dict[str, int]:
        try:
            with self.db.get_session() as session:
                from sqlalchemy import func

                counts = (
                    session.query(ModelMetadata.status, func.count(ModelMetadata.id))
                    .filter_by(is_active=True)
                    .group_by(ModelMetadata.status)
                    .all()
                )
                result = {status: count for status, count in counts}
                logger.debug(f"Model counts: {result}")
                return result
        except SQLAlchemyError as e:
            logger.error(f"Database error getting model count: {str(e)}")
            raise
