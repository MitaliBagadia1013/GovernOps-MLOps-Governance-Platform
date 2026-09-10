# Kubeflow Pipeline Container Image
# This Docker image contains all dependencies needed to run pipeline components

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional Kubeflow Pipeline SDK components
RUN pip install --no-cache-dir \
    kfp==2.4.0 \
    kfp-server-api==2.0.5 \
    kfp-pipeline-spec==0.2.2

# Copy the entire project
COPY . /app/

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
ENV DATABASE_URL=postgresql://mlops:password@postgres:5432/mlops_db

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/mlruns

# Default command (can be overridden in pipeline components)
CMD ["python", "-c", "print('Kubeflow Pipeline Container Ready')"]
