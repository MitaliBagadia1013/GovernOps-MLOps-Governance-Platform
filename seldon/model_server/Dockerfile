# Seldon Model Server - Credit Risk Model

FROM python:3.9-slim

# Install dependencies
RUN pip install --no-cache-dir \
    joblib==1.3.2 \
    scikit-learn==1.3.2 \
    numpy==1.24.3 \
    pandas==2.0.3

# Copy model server code
WORKDIR /app
COPY CreditRiskModel.py /app/

# Create mount point for models
RUN mkdir -p /mnt/models

# Expose port for Seldon
EXPOSE 9000

# Seldon will run this automatically
CMD ["python", "-c", "import CreditRiskModel; print('Model server ready')"]
