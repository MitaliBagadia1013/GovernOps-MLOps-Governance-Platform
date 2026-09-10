from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager
from .model_loader import ModelLoader
from .schemas import (
    HealthResponse,
    ModelsListResponse,
    ModelInfo,
    ErrorResponse,
    CreditRiskRequest,
    ChurnRequest,
    PricePredictionRequest,
    AnomalyDetectionRequest,
    SegmentationRequest,
    ImageClassificationRequest,
    SentimentRequest,
    DemandForecastRequest,
    RecommendationRequest,
    ClassificationResponse,
    RegressionResponse,
    AnomalyResponse,
    ClusteringResponse,
    RecommendationResponse,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 80)
    logger.info("Starting MLOps Model Serving Platform")
    logger.info("=" * 80)
    global model_loader
    model_loader = ModelLoader()
    models_to_load = [
        ("credit_risk", "2.0.0"),
        ("churn_prediction", "1.0.0"),
        ("price_prediction", "2.0.0"),
        ("anomaly_detection", "1.0.0"),
        ("customer_segmentation", "2.0.0"),
        ("image_classification", "1.0.0"),
        ("sentiment_analysis", "1.0.0"),
        ("demand_forecasting", "1.0.0"),
        ("recommendation", "3.0.0"),
    ]
    loaded_count = 0
    for model_name, version in models_to_load:
        try:
            model_loader.load_model(model_name, version)
            loaded_count += 1
            logger.info(f" Preloaded {model_name} v{version}")
        except Exception as e:
            logger.error(f" Failed to preload {model_name} v{version}: {str(e)}")
    logger.info(f" Preloaded {loaded_count}/{len(models_to_load)} models")
    logger.info("=" * 80)
    logger.info("Application ready to serve predictions!")
    logger.info("=" * 80)
    yield
    logger.info("Shutting down GovernOps...")


app = FastAPI(
    title="MLOps Model Governance Platform",
    description="\n    Production-ready ML model serving platform with 9 trained models.\n    \n    ## Available Models:\n    \n    1. **Credit Risk** (v2.0.0) - RandomForest classification on Wisconsin Breast Cancer data\n    2. **Churn Prediction** (v1.0.0) - LogisticRegression on Wine Quality data\n    3. **Price Prediction** (v2.0.0) - GradientBoosting regression on California Housing data\n    4. **Anomaly Detection** (v1.0.0) - IsolationForest for outlier detection\n    5. **Customer Segmentation** (v2.0.0) - KMeans clustering on Iris data\n    6. **Image Classification** (v1.0.0) - RandomForest on Digits dataset\n    7. **Sentiment Analysis** (v1.0.0) - GaussianNB classifier\n    8. **Demand Forecasting** (v1.0.0) - Ridge regression on Diabetes data\n    9. **Recommendation** (v3.0.0) - KNN collaborative filtering\n    \n    ## Features:\n    - Model versioning and lineage tracking\n    - Request/response validation with Pydantic\n    - In-memory model caching for fast predictions\n    - Comprehensive error handling\n    - OpenAPI/Swagger documentation\n    ",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "MLOps Model Governance Platform",
        "status": "operational",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    try:
        cache_info = model_loader.get_cache_info()
        available = model_loader.get_available_models()
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            database="postgresql",
            models_loaded=cache_info["cache_size"],
            available_models=available,
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}",
        )


@app.get("/models", response_model=ModelsListResponse, tags=["Models"])
async def list_models():
    try:
        available = model_loader.get_available_models()
        total = sum((len(versions) for versions in available.values()))
        return ModelsListResponse(total_models=total, models=available)
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
        )


@app.get("/models/{model_name}/{version}", response_model=ModelInfo, tags=["Models"])
async def get_model_info(model_name: str, version: str):
    try:
        model_bundle = model_loader.load_model(model_name, version)
        return ModelInfo(
            name=model_name,
            version=version,
            metrics=model_bundle["metrics"],
            features=model_bundle["features"],
            loaded_at=model_bundle["metadata"]["loaded_at"],
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_name} v{version} not found",
        )
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post(
    "/predict/credit_risk", response_model=ClassificationResponse, tags=["Predictions"]
)
async def predict_credit_risk(request: CreditRiskRequest):
    try:
        model_bundle = model_loader.smart_load("credit_risk", version="2")
        features = [
            request.mean_radius,
            request.mean_texture,
            request.mean_perimeter,
            request.mean_area,
            request.mean_smoothness,
            request.mean_compactness,
            request.mean_concavity,
            request.mean_concave_points,
            request.mean_symmetry,
            request.mean_fractal_dimension,
            request.radius_error,
            request.texture_error,
            request.perimeter_error,
            request.area_error,
            request.smoothness_error,
            request.compactness_error,
            request.concavity_error,
            request.concave_points_error,
            request.symmetry_error,
            request.fractal_dimension_error,
            request.worst_radius,
            request.worst_texture,
            request.worst_perimeter,
            request.worst_area,
            request.worst_smoothness,
            request.worst_compactness,
            request.worst_concavity,
            request.worst_concave_points,
            request.worst_symmetry,
            request.worst_fractal_dimension,
        ]
        X = np.array([features])
        prediction = int(model_bundle["model"].predict(X)[0])
        probabilities = model_bundle["model"].predict_proba(X)[0].tolist()
        return ClassificationResponse(
            prediction=prediction,
            probability=float(probabilities[prediction]),
            probabilities=probabilities,
            model_name="credit_risk",
            model_version="2.0.0",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Credit risk prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/churn", response_model=ClassificationResponse, tags=["Predictions"])
async def predict_churn(request: ChurnRequest):
    try:
        model_bundle = model_loader.smart_load("churn_prediction", version="1")
        features = [
            request.alcohol,
            request.malic_acid,
            request.ash,
            request.alcalinity_of_ash,
            request.magnesium,
            request.total_phenols,
            request.flavanoids,
            request.nonflavanoid_phenols,
            request.proanthocyanins,
            request.color_intensity,
            request.hue,
            request.od280_od315_of_diluted_wines,
            request.proline,
        ]
        X = np.array([features])
        prediction = int(model_bundle["model"].predict(X)[0])
        probabilities = model_bundle["model"].predict_proba(X)[0].tolist()
        return ClassificationResponse(
            prediction=prediction,
            probability=float(probabilities[prediction]),
            probabilities=probabilities,
            model_name="churn_prediction",
            model_version="1.0.0",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Churn prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/image", response_model=ClassificationResponse, tags=["Predictions"])
async def predict_image_class(request: ImageClassificationRequest):
    try:
        model_bundle = model_loader.smart_load("image_classification", version="1")
        X = np.array([request.pixels])
        prediction = int(model_bundle["model"].predict(X)[0])
        probabilities = model_bundle["model"].predict_proba(X)[0].tolist()
        return ClassificationResponse(
            prediction=prediction,
            probability=float(probabilities[prediction]),
            probabilities=probabilities,
            model_name="image_classification",
            model_version="1.0.0",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Image classification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/predict/sentiment", response_model=ClassificationResponse, tags=["Predictions"]
)
async def predict_sentiment(request: SentimentRequest):
    try:
        model_bundle = model_loader.smart_load("sentiment_analysis", version="1")
        features = [
            request.feature_1,
            request.feature_2,
            request.feature_3,
            request.feature_4,
        ]
        X = np.array([features])
        prediction = int(model_bundle["model"].predict(X)[0])
        probabilities = model_bundle["model"].predict_proba(X)[0].tolist()
        return ClassificationResponse(
            prediction=prediction,
            probability=float(probabilities[prediction]),
            probabilities=probabilities,
            model_name="sentiment_analysis",
            model_version="1.0.0",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/price", response_model=RegressionResponse, tags=["Predictions"])
async def predict_price(request: PricePredictionRequest):
    try:
        model_bundle = model_loader.smart_load("price_prediction", version="2")
        features = [
            request.MedInc,
            request.HouseAge,
            request.AveRooms,
            request.AveBedrms,
            request.Population,
            request.AveOccup,
            request.Latitude,
            request.Longitude,
        ]
        X = np.array([features])
        prediction = float(model_bundle["model"].predict(X)[0])
        return RegressionResponse(
            prediction=prediction,
            model_name="price_prediction",
            model_version="2.0.0",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Price prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/demand", response_model=RegressionResponse, tags=["Predictions"])
async def predict_demand(request: DemandForecastRequest):
    try:
        model_bundle = model_loader.smart_load("demand_forecasting", version="1")
        features = [
            request.age,
            request.sex,
            request.bmi,
            request.bp,
            request.s1,
            request.s2,
            request.s3,
            request.s4,
            request.s5,
            request.s6,
        ]
        X = np.array([features])
        prediction = float(model_bundle["model"].predict(X)[0])
        return RegressionResponse(
            prediction=prediction,
            model_name="demand_forecasting",
            model_version="1.0.0",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Demand forecasting failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/anomaly", response_model=AnomalyResponse, tags=["Predictions"])
async def detect_anomaly(request: AnomalyDetectionRequest):
    try:
        model_bundle = model_loader.smart_load("anomaly_detection", version="1")
        X = np.array([request.features])
        prediction = int(model_bundle["model"].predict(X)[0])
        anomaly_score = float(model_bundle["model"].score_samples(X)[0])
        return AnomalyResponse(
            is_anomaly=prediction == -1,
            anomaly_score=anomaly_score,
            model_name="anomaly_detection",
            model_version="1.0.0",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}",
        )


@app.post("/predict/segment", response_model=ClusteringResponse, tags=["Predictions"])
async def predict_segment(request: SegmentationRequest):
    try:
        model_bundle = model_loader.smart_load("customer_segmentation", version="2")
        features = [
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width,
        ]
        X = np.array([features])
        model_obj = model_bundle["model"]
        if isinstance(model_obj, dict):
            kmeans = model_obj["model"]
            scaler = model_obj["scaler"]
            X = scaler.transform(X)
        else:
            kmeans = model_obj
        cluster = int(kmeans.predict(X)[0])
        distances = kmeans.transform(X)[0].tolist()
        return ClusteringResponse(
            cluster=cluster,
            cluster_distances=distances,
            model_name="customer_segmentation",
            model_version="2.0.0",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Segmentation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/predict/recommend", response_model=RecommendationResponse, tags=["Predictions"]
)
async def get_recommendations(request: RecommendationRequest):
    try:
        model_bundle = model_loader.smart_load("recommendation", version="3")
        X = np.array([request.item_features])
        model_obj = model_bundle["model"]
        if isinstance(model_obj, dict):
            knn = model_obj["model"]
            scaler = model_obj["scaler"]
            X = scaler.transform(X)
        else:
            knn = model_obj
        distances, indices = knn.kneighbors(
            X, n_neighbors=request.n_recommendations + 1
        )
        recommendations = []
        for i, (dist, idx) in enumerate(zip(distances[0][1:], indices[0][1:])):
            recommendations.append(
                {
                    "rank": i + 1,
                    "item_index": int(idx),
                    "similarity_score": float(1 - dist),
                    "distance": float(dist),
                }
            )
        return RecommendationResponse(
            recommendations=recommendations,
            model_name="recommendation",
            model_version="3.0.0",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Recommendation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation failed: {str(e)}",
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
