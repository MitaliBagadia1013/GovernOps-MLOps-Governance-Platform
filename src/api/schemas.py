from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    database: str
    models_loaded: int
    available_models: Dict[str, List[str]]


class ModelInfo(BaseModel):
    name: str
    version: str
    metrics: Dict[str, float]
    features: List[str]
    loaded_at: Optional[str] = None


class ModelsListResponse(BaseModel):
    total_models: int
    models: Dict[str, List[str]]


class CreditRiskRequest(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float


class ChurnRequest(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


class PricePredictionRequest(BaseModel):
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms")
    AveBedrms: float = Field(..., description="Average number of bedrooms")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average house occupancy")
    Latitude: float = Field(..., description="Latitude")
    Longitude: float = Field(..., description="Longitude")


class AnomalyDetectionRequest(BaseModel):
    features: List[float] = Field(..., min_items=30, max_items=30)


class SegmentationRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class ImageClassificationRequest(BaseModel):
    pixels: List[float] = Field(..., min_items=64, max_items=64)


class SentimentRequest(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float


class DemandForecastRequest(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


class RecommendationRequest(BaseModel):
    item_features: List[float] = Field(..., min_items=64, max_items=64)
    n_recommendations: int = Field(default=5, ge=1, le=20)


class ClassificationResponse(BaseModel):
    prediction: int
    probability: Optional[float] = None
    probabilities: Optional[List[float]] = None
    model_name: str
    model_version: str
    timestamp: str


class RegressionResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: str
    timestamp: str


class AnomalyResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    model_name: str
    model_version: str
    timestamp: str


class ClusteringResponse(BaseModel):
    cluster: int
    cluster_distances: Optional[List[float]] = None
    model_name: str
    model_version: str
    timestamp: str


class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    model_name: str
    model_version: str
    timestamp: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str


class BatchPredictionRequest(BaseModel):
    instances: List[Dict[str, Any]] = Field(..., min_items=1, max_items=1000)


class BatchPredictionResponse(BaseModel):
    predictions: List[Any]
    total_instances: int
    model_name: str
    model_version: str
    timestamp: str
