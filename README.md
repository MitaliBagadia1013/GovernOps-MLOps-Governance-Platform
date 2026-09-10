# GovernOps - MLOps Governance Platform

A production-grade MLOps platform that manages the full lifecycle of 9 concurrent ML models — from experiment tracking and versioning to canary deployment, data drift detection, and automated compliance reporting.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI REST API                         │
│              9 model endpoints · Swagger UI · Health checks     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌─────────────┐  ┌────────────┐  ┌─────────────────┐
   │ PostgreSQL  │  │   MLflow   │  │    Kubeflow     │
   │  Registry  │  │  Tracking  │  │    Pipelines    │
   │ (SQLAlchemy)│  │  Server   │  │                 │
   └─────────────┘  └────────────┘  └─────────────────┘
          │
          ▼
   ┌──────────────────────────────────────┐
   │         Seldon Core (Kubernetes)     │
   │   95% baseline · 5% canary traffic  │
   └──────────────────────────────────────┘
          │
          ▼
   ┌──────────────────────────────────────┐
   │      Self-Healing Monitor            │
   │  KS-test drift detection →          │
   │  AWS Step Functions retraining      │
   └──────────────────────────────────────┘
```

## Key Features

### Model Lifecycle Management
- **9 production models** registered and served concurrently: credit risk, churn prediction, price prediction, anomaly detection, customer segmentation, image classification, sentiment analysis, demand forecasting, and recommendation
- Full versioning with semantic version strings (`v1.0.0`, `v2.0.0`) stored in PostgreSQL via SQLAlchemy ORM
- Model cards (JSON) for each model capturing algorithm, dataset, hyperparameters, and performance metrics

### Experiment Tracking & Reproducibility
- MLflow integration for 100% reproducible experiments — all hyperparameters, metrics, and artifacts logged per run
- Data lineage tracked through `ModelLineage` table (git commit hash, training script path, dataset version)
- Local and remote (S3) artifact storage supported via environment config

### Canary Deployment (Seldon Core)
- Risk-mitigating rollout: **5% traffic → canary**, **95% → stable baseline**
- `SelfHealingSystem` monitors canary health in real time and auto-promotes or auto-rolls back based on configurable error-rate and latency thresholds

### Data Drift Detection & Self-Healing
- Kolmogorov-Smirnov (KS) test per feature, configurable significance threshold (default `p < 0.05`)
- After N consecutive drift detections, `DriftMonitoringPipeline` automatically triggers an AWS Step Functions retraining workflow — no human intervention required

### Compliance & Bias Reporting
- Per-model bias detection: computes **demographic parity difference** and **disparate impact ratio** across a protected attribute
- Automated compliance reports (JSON) classify each model as `compliant`, `review_required`, or `non_compliant`

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Pydantic v2, Uvicorn |
| ML | scikit-learn, MLflow |
| Pipelines | Kubeflow Pipelines |
| Deployment | Seldon Core, Kubernetes |
| Database | PostgreSQL 15, SQLAlchemy 2 |
| Orchestration | AWS Step Functions |
| CI/CD | GitHub Actions |
| Containerisation | Docker, Docker Compose |

## Project Structure

```
governops/
├── src/
│   ├── api/                  # FastAPI app — 9 prediction endpoints
│   ├── canary/               # Canary deployment strategy
│   ├── compliance/           # Bias detection & compliance reports
│   ├── database/             # SQLAlchemy models, connection pooling
│   ├── drift_detection/      # KS-test pipeline + retraining trigger
│   ├── models/               # Model registry & model cards
│   ├── monitoring/           # Self-healing canary monitor
│   └── pipeline/             # Multi-model training pipeline
├── kubeflow/                 # Kubeflow pipeline definitions
├── seldon/                   # Seldon Core canary deployment configs
├── mlflow_config/            # MLflow tracking configuration
├── model_cards/              # JSON model cards for all 9 models
├── models/                   # Model metadata (features.json, metrics.json)
├── scripts/                  # Drift demo, DB init, scheduled monitoring
├── step_functions/           # AWS Step Functions retraining workflow
├── config/                   # Platform configuration (YAML)
├── tests/                    # CI test suite
├── Dockerfile
├── docker-compose.yml        # Full stack: API + Postgres + MLflow
└── requirements.txt
```

## Quick Start

**Prerequisites:** Docker, Docker Compose

```bash
git clone <your-repo-url>
cd governops

# Configure environment
cp .env.example .env
# Edit .env with your DB credentials and AWS keys (if using Step Functions)

# Start the full stack
docker-compose up -d

# API is live at http://localhost:8000
# Swagger UI:    http://localhost:8000/docs
# MLflow UI:     http://localhost:5000
# PgAdmin:       http://localhost:5050
```

**Train models locally:**

```bash
pip install -r requirements.txt
python src/pipeline/train_all_models_real.py
```

**Run drift detection demo:**

```bash
python scripts/drift_detection_demo.py
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service health + loaded models |
| GET | `/models` | List all registered models |
| GET | `/models/{name}/{version}` | Model metadata, metrics, features |
| POST | `/predict/credit_risk` | RandomForest · 95.6% accuracy |
| POST | `/predict/churn` | Logistic Regression · Wine dataset |
| POST | `/predict/price` | GradientBoosting · R²=0.81 |
| POST | `/predict/anomaly` | IsolationForest · outlier detection |
| POST | `/predict/segment` | KMeans · 3-cluster segmentation |
| POST | `/predict/image` | RandomForest · 96.1% on Digits |
| POST | `/predict/sentiment` | GaussianNB · 96.7% accuracy |
| POST | `/predict/demand` | Ridge Regression · Diabetes dataset |
| POST | `/predict/recommend` | KNN collaborative filtering |

## Database Schema

Five PostgreSQL tables managed by SQLAlchemy + Alembic migrations:

- `model_metadata` — core registry with versioning, status lifecycle, compliance fields
- `deployment_history` — canary traffic splits, rollback events, promotion timestamps
- `drift_records` — per-feature KS statistics, consecutive drift counts, retraining flags
- `performance_metrics` — time-series metric storage per model per environment
- `model_lineage` — git commit hash, MLflow run ID, parent model chain for reproducibility

## CI/CD

GitHub Actions pipeline (`.github/workflows/ci-cd.yml`) runs on every push:
- Installs dependencies
- Runs the test suite (`pytest tests/`)
- Validates imports and directory structure

## License

This project is for educational purposes only. It was built as a personal/portfolio exploration of MLOps platform design and model governance. Please review the terms of service for AWS, and the licenses for MLflow, Kubeflow, and Seldon Core, before reusing this code with your own data or in production.
