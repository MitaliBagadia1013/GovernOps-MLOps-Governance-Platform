# Seldon Core Canary Deployment Guide

## 🎯 Goal
Deploy a canary deployment strategy for credit_risk model with:
- **Baseline (v1.0.0)**: 95% of traffic
- **Canary (v2.0.0)**: 5% of traffic
- Automated monitoring and rollback

---

## 📋 Prerequisites

✅ Minikube running (already confirmed)
✅ kubectl installed (already confirmed)
⏳ Seldon Core operator (installing now)
⏳ Model server Docker images (creating now)

---

## 🚀 Step-by-Step Deployment

### Step 1: Install Seldon Core Operator
```bash
# Install Seldon Core using kubectl
kubectl create namespace seldon-system
kubectl apply -f https://github.com/SeldonIO/seldon-core/releases/download/v1.17.1/seldon-core-operator.yaml
```

### Step 2: Verify Seldon Installation
```bash
# Check if Seldon operator is running
kubectl get pods -n seldon-system
```

### Step 3: Create Model Namespace
```bash
kubectl create namespace mlops-models
```

### Step 4: Build Model Server Docker Image
```bash
# Build custom model server
docker build -t credit-risk-server:v1.0.0 -f seldon/Dockerfile.model .
docker build -t credit-risk-server:v2.0.0 -f seldon/Dockerfile.model .
```

### Step 5: Deploy Canary Configuration
```bash
kubectl apply -f seldon/credit_risk_canary_deployment.yaml
```

### Step 6: Test Traffic Split
```bash
# Send 100 requests and verify ~5% go to canary
for i in {1..100}; do
  curl -X POST http://$(minikube ip):30000/predict \
    -H "Content-Type: application/json" \
    -d @seldon/sample_request.json
done
```

---

## 📊 Expected Results

- **Baseline traffic**: ~95 requests
- **Canary traffic**: ~5 requests
- **Latency**: < 100ms p95
- **Success rate**: > 99%

---

## 🔄 Rollback Procedure

If canary fails:
```bash
kubectl patch seldondeployment credit-risk \
  --type='json' \
  -p='[{"op": "replace", "path": "/spec/predictors/1/traffic", "value": 0}]'
```

---

## 📸 Screenshot Evidence

Take screenshots of:
1. `kubectl get pods -n mlops-models`
2. `kubectl get seldondeployment credit-risk -o yaml`
3. Grafana dashboard showing 5%/95% split
4. Test results showing traffic distribution
