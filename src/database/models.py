from datetime import datetime
from typing import Optional
import json
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Boolean,
    Text,
    JSON,
    ForeignKey,
    Index,
    UniqueConstraint,
    CheckConstraint,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class ModelMetadata(Base):
    __tablename__ = "model_metadata"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    model_id = Column(String(255), unique=True, nullable=False, index=True)
    algorithm = Column(String(100), nullable=False)
    framework = Column(String(50), nullable=False)
    framework_version = Column(String(50))
    training_date = Column(DateTime, nullable=False, default=func.now())
    training_duration_seconds = Column(Float)
    dataset_name = Column(String(255))
    dataset_version = Column(String(50))
    dataset_size = Column(Integer)
    hyperparameters = Column(JSON, nullable=False, default={})
    model_path = Column(String(500), nullable=False)
    artifact_location = Column(String(500))
    model_size_mb = Column(Float)
    metrics = Column(JSON, default={})
    status = Column(String(50), nullable=False, default="registered", index=True)
    is_active = Column(Boolean, default=True, index=True)
    created_by = Column(String(100), nullable=False)
    team = Column(String(100))
    project = Column(String(100))
    bias_report = Column(JSON)
    compliance_status = Column(String(50), default="pending")
    created_at = Column(DateTime, nullable=False, default=func.now(), index=True)
    updated_at = Column(
        DateTime, nullable=False, default=func.now(), onupdate=func.now()
    )
    description = Column(Text)
    tags = Column(JSON, default=[])
    deployments = relationship(
        "DeploymentHistory", back_populates="model", cascade="all, delete-orphan"
    )
    drift_records = relationship(
        "DriftRecord", back_populates="model", cascade="all, delete-orphan"
    )
    performance_metrics = relationship(
        "PerformanceMetric", back_populates="model", cascade="all, delete-orphan"
    )
    lineage = relationship(
        "ModelLineage",
        back_populates="model",
        foreign_keys="ModelLineage.model_id",
        cascade="all, delete-orphan",
    )
    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_model_version"),
        Index("idx_model_status_active", "status", "is_active"),
        Index("idx_model_created_at", "created_at"),
        CheckConstraint(
            "status IN ('registered', 'staging', 'production', 'archived', 'deprecated')",
            name="ck_model_status",
        ),
    )

    def __repr__(self):
        return f"<ModelMetadata(id={self.id}, name={self.model_name}, version={self.version}, status={self.status})>"


class DeploymentHistory(Base):
    __tablename__ = "deployment_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(
        Integer,
        ForeignKey("model_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    deployment_id = Column(String(255), unique=True, nullable=False)
    deployment_type = Column(String(50), nullable=False)
    environment = Column(String(50), nullable=False)
    canary_traffic_percent = Column(Float, default=0.0)
    baseline_traffic_percent = Column(Float, default=100.0)
    deployment_status = Column(String(50), nullable=False)
    is_current = Column(Boolean, default=False, index=True)
    deployed_at = Column(DateTime, nullable=False, default=func.now())
    promoted_at = Column(DateTime)
    rolled_back_at = Column(DateTime)
    ended_at = Column(DateTime)
    total_requests = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    error_rate = Column(Float, default=0.0)
    avg_latency_ms = Column(Float)
    deployment_config = Column(JSON, default={})
    rollback_reason = Column(Text)
    rolled_back_by = Column(String(100))
    deployed_by = Column(String(100), nullable=False)
    model = relationship("ModelMetadata", back_populates="deployments")
    __table_args__ = (
        Index("idx_deployment_status", "deployment_status"),
        Index("idx_deployment_model_current", "model_id", "is_current"),
    )

    def __repr__(self):
        return f"<DeploymentHistory(id={self.id}, model_id={self.model_id}, status={self.deployment_status})>"


class DriftRecord(Base):
    __tablename__ = "drift_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(
        Integer,
        ForeignKey("model_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    feature_name = Column(String(255), nullable=False, index=True)
    drift_detected = Column(Boolean, nullable=False, index=True)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False, default=0.05)
    reference_data_path = Column(String(500))
    production_data_path = Column(String(500))
    sample_size = Column(Integer)
    detected_at = Column(DateTime, nullable=False, default=func.now(), index=True)
    consecutive_drift_count = Column(Integer, default=0)
    retraining_triggered = Column(Boolean, default=False)
    retraining_triggered_at = Column(DateTime)
    drift_metadata = Column(JSON, default={})
    model = relationship("ModelMetadata", back_populates="drift_records")
    __table_args__ = (
        Index("idx_drift_model_feature", "model_id", "feature_name"),
        Index("idx_drift_detected_at", "detected_at"),
    )

    def __repr__(self):
        return f"<DriftRecord(id={self.id}, model_id={self.model_id}, feature={self.feature_name}, drift={self.drift_detected})>"


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(
        Integer,
        ForeignKey("model_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(50), nullable=False)
    environment = Column(String(50), nullable=False)
    data_source = Column(String(100))
    recorded_at = Column(DateTime, nullable=False, default=func.now(), index=True)
    metric_metadata = Column(JSON, default={})
    model = relationship("ModelMetadata", back_populates="performance_metrics")
    __table_args__ = (
        Index("idx_metric_model_name", "model_id", "metric_name"),
        Index("idx_metric_recorded_at", "recorded_at"),
    )

    def __repr__(self):
        return f"<PerformanceMetric(id={self.id}, model_id={self.model_id}, metric={self.metric_name}, value={self.metric_value})>"


class ModelLineage(Base):
    __tablename__ = "model_lineage"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(
        Integer,
        ForeignKey("model_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parent_model_id = Column(
        Integer, ForeignKey("model_metadata.id", ondelete="SET NULL")
    )
    training_data_sources = Column(JSON, default=[])
    data_versions = Column(JSON, default={})
    training_script_path = Column(String(500))
    git_commit_hash = Column(String(40))
    git_branch = Column(String(100))
    git_repository = Column(String(500))
    mlflow_run_id = Column(String(255), index=True)
    experiment_id = Column(String(255))
    created_at = Column(DateTime, nullable=False, default=func.now())
    model = relationship(
        "ModelMetadata", back_populates="lineage", foreign_keys=[model_id]
    )
    __table_args__ = (
        Index("idx_lineage_model", "model_id"),
        Index("idx_lineage_parent", "parent_model_id"),
    )

    def __repr__(self):
        return f"<ModelLineage(id={self.id}, model_id={self.model_id}, parent_id={self.parent_model_id})>"
