#!/bin/bash
# ============================================================================
# Seldon Canary Deployment Setup Script
# ============================================================================
# This script sets up the complete Seldon canary deployment on Minikube

set -e  # Exit on error

echo "========================================================================"
echo "🚀 SELDON CANARY DEPLOYMENT SETUP"
echo "========================================================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ----------------------------------------------------------------------
# Step 1: Verify Prerequisites
# ----------------------------------------------------------------------
echo -e "\n${YELLOW}Step 1: Verifying prerequisites...${NC}"

if ! command -v minikube &> /dev/null; then
    echo -e "${RED}❌ Minikube not found. Please install it first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Minikube installed${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl not found. Please install it first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ kubectl installed${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install it first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker installed${NC}"

# ----------------------------------------------------------------------
# Step 2: Start Minikube (if not running)
# ----------------------------------------------------------------------
echo -e "\n${YELLOW}Step 2: Checking Minikube status...${NC}"

if ! minikube status | grep -q "Running"; then
    echo "Starting Minikube..."
    minikube start --cpus=4 --memory=8192
else
    echo -e "${GREEN}✓ Minikube already running${NC}"
fi

# Configure Docker to use Minikube's Docker daemon
echo "Configuring Docker environment..."
eval $(minikube docker-env)

MINIKUBE_IP=$(minikube ip)
echo -e "${GREEN}✓ Minikube IP: ${MINIKUBE_IP}${NC}"

# ----------------------------------------------------------------------
# Step 3: Install Seldon Core Operator
# ----------------------------------------------------------------------
echo -e "\n${YELLOW}Step 3: Installing Seldon Core...${NC}"

# Create namespace
kubectl create namespace seldon-system --dry-run=client -o yaml | kubectl apply -f -

# Install Seldon Core operator
echo "Installing Seldon operator (this may take a few minutes)..."
kubectl apply -f https://github.com/SeldonIO/seldon-core/releases/download/v1.17.1/seldon-core-operator.yaml

# Wait for Seldon operator to be ready
echo "Waiting for Seldon operator to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=seldon-core -n seldon-system --timeout=300s

echo -e "${GREEN}✓ Seldon Core installed${NC}"

# ----------------------------------------------------------------------
# Step 4: Create Model Namespace
# ----------------------------------------------------------------------
echo -e "\n${YELLOW}Step 4: Creating model namespace...${NC}"

kubectl create namespace mlops-models --dry-run=client -o yaml | kubectl apply -f -
echo -e "${GREEN}✓ Namespace 'mlops-models' created${NC}"

# ----------------------------------------------------------------------
# Step 5: Build Model Server Docker Images
# ----------------------------------------------------------------------
echo -e "\n${YELLOW}Step 5: Building model server Docker images...${NC}"

cd "$(dirname "$0")/.."

# Build baseline version
echo "Building baseline image (v2.0.0)..."
docker build \
  -t credit-risk-model:v2.0.0 \
  -f seldon/Dockerfile.model \
  .

# Tag for canary
echo "Creating canary image tag..."
docker tag credit-risk-model:v2.0.0 credit-risk-model:v2.0.0-canary

echo -e "${GREEN}✓ Docker images built${NC}"

# List images
echo -e "\nBuilt images:"
docker images | grep credit-risk-model

# ----------------------------------------------------------------------
# Step 6: Deploy Canary Configuration
# ----------------------------------------------------------------------
echo -e "\n${YELLOW}Step 6: Deploying canary configuration...${NC}"

# Apply the deployment
kubectl apply -f seldon/credit_risk_canary_deployment.yaml

echo "Waiting for deployments to be ready..."
sleep 10

# Wait for pods
echo "Waiting for pods to start..."
kubectl wait --for=condition=ready pod -l seldon-deployment-id=credit-risk-canary -n mlops-models --timeout=300s || true

echo -e "${GREEN}✓ Canary deployment applied${NC}"

# ----------------------------------------------------------------------
# Step 7: Verify Deployment
# ----------------------------------------------------------------------
echo -e "\n${YELLOW}Step 7: Verifying deployment...${NC}"

echo -e "\nPods in mlops-models namespace:"
kubectl get pods -n mlops-models

echo -e "\nSeldon deployments:"
kubectl get seldondeployment -n mlops-models

echo -e "\nServices:"
kubectl get svc -n mlops-models

# ----------------------------------------------------------------------
# Step 8: Setup Port Forwarding (Alternative to NodePort)
# ----------------------------------------------------------------------
echo -e "\n${YELLOW}Step 8: Setting up access...${NC}"

# Get the service name
SERVICE_NAME=$(kubectl get svc -n mlops-models -l seldon-deployment-id=credit-risk-canary -o jsonpath='{.items[0].metadata.name}')

if [ -n "$SERVICE_NAME" ]; then
    echo -e "${GREEN}✓ Service found: ${SERVICE_NAME}${NC}"
    echo -e "\n${YELLOW}To access the model, run:${NC}"
    echo "kubectl port-forward -n mlops-models svc/${SERVICE_NAME} 8000:8000"
    echo -e "\nThen test with:"
    echo "curl -X POST http://localhost:8000/api/v1.0/predictions -H 'Content-Type: application/json' -d @seldon/sample_request.json"
else
    echo -e "${RED}⚠️  Service not found yet. It may still be creating.${NC}"
fi

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
echo -e "\n========================================================================"
echo -e "${GREEN}✅ SELDON CANARY DEPLOYMENT SETUP COMPLETE!${NC}"
echo "========================================================================"

echo -e "\n📋 Summary:"
echo "   Minikube IP: ${MINIKUBE_IP}"
echo "   Namespace: mlops-models"
echo "   Deployment: credit-risk-canary"
echo "   Traffic Split: 95% baseline, 5% canary"

echo -e "\n🎯 Next Steps:"
echo "   1. Port forward: kubectl port-forward -n mlops-models svc/\${SERVICE_NAME} 8000:8000"
echo "   2. Test canary: python seldon/test_canary.py"
echo "   3. Monitor: kubectl logs -n mlops-models -l seldon-deployment-id=credit-risk-canary --tail=50"
echo "   4. View dashboard: minikube dashboard"

echo -e "\n💡 Useful Commands:"
echo "   - Check pods: kubectl get pods -n mlops-models"
echo "   - Check deployment: kubectl get seldondeployment -n mlops-models"
echo "   - View logs: kubectl logs -n mlops-models <pod-name>"
echo "   - Delete deployment: kubectl delete -f seldon/credit_risk_canary_deployment.yaml"

echo -e "\n========================================================================"
