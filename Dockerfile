# ============================================================================
# Production Dockerfile for GovernOps
# ============================================================================
# This creates a Docker "container" - a packaged version of your app
# Think of it as: "Packaging your app with everything it needs to run"

# ============================================================================
# STAGE 1: Base Image
# ============================================================================
FROM python:3.13-slim as base
# Explanation:
# - "FROM python:3.13-slim" = Start with Python 3.13 (lightweight version)
# - "slim" = Smaller size (no unnecessary tools)
# - Like choosing a foundation for a house

# Set working directory
WORKDIR /app
# Explanation:
# - Creates /app folder inside container
# - All commands will run from here
# - Like setting up your workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*
# Explanation:
# - Installs tools needed to compile Python packages
# - gcc/g++: Compilers for C/C++ code
# - curl: For health checks
# - "rm -rf /var/lib/apt/lists/*" = Clean up to reduce image size

# ============================================================================
# STAGE 2: Dependencies
# ============================================================================
FROM base as dependencies

# Copy requirements first (for better caching)
COPY requirements.txt .
# Explanation:
# - Copies requirements.txt from your computer into container
# - Docker caches this layer
# - If requirements.txt doesn't change, Docker reuses cached layer
# - Speeds up builds!

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
# Explanation:
# - Installs all Python packages listed in requirements.txt
# - "--no-cache-dir" = Don't save package cache (saves space)
# - Like installing all ingredients before cooking

# ============================================================================
# STAGE 3: Application
# ============================================================================
FROM dependencies as application

# Copy application code
COPY . .
# Explanation:
# - Copies ALL your code into container
# - "." means current directory
# - First "." = your computer
# - Second "." = container's /app folder

# Create necessary directories
RUN mkdir -p models model_cards mlruns logs
# Explanation:
# - Creates folders your app needs
# - models: Trained model files
# - model_cards: Model metadata
# - mlruns: MLflow experiment data
# - logs: Application logs

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://localhost:5000
# Explanation:
# - PYTHONUNBUFFERED=1: Print output immediately (good for logs)
# - PYTHONDONTWRITEBYTECODE=1: Don't create .pyc files (cleaner)
# - MLFLOW_TRACKING_URI: Where MLflow is running

# Expose ports
EXPOSE 8000 5000
# Explanation:
# - Port 8000: FastAPI (your API server)
# - Port 5000: MLflow UI
# - Like opening doors for visitors

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
# Explanation:
# - Automatically checks if app is healthy
# - Every 30 seconds, curl http://localhost:8000/health
# - If fails 3 times, marks container as unhealthy
# - Like a doctor checking your pulse

# ============================================================================
# STAGE 4: Production (Final)
# ============================================================================
FROM application as production

# Run as non-root user (security best practice)
RUN useradd -m -u 1000 mlops && \
    chown -R mlops:mlops /app
USER mlops
# Explanation:
# - Creates a user "mlops" instead of running as root
# - Root = admin with all permissions (dangerous!)
# - mlops = regular user (safer)
# - Like giving keys to an employee, not the master key

# Default command to run
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
# Explanation:
# - When container starts, run this command
# - uvicorn: Python web server
# - src.api.main:app: Your FastAPI application
# - --host 0.0.0.0: Accept connections from anywhere
# - --port 8000: Run on port 8000
# - Like: "When someone turns on this machine, start the web server"

# ============================================================================
# How to build and run this:
# 
# Build:
#   docker build -t governops:latest .
# 
# Run:
#   docker run -p 8000:8000 governops:latest
# 
# Then open: http://localhost:8000
# ============================================================================
