"""
Pydantic Response Models
Định nghĩa schemas cho tất cả responses từ API
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class PredictionResponse(BaseModel):
    """Standard prediction response cho tất cả modes"""
    prediction: int = Field(
        ..., 
        description="Predicted class: 0=Poor, 1=Moderate, 2=Good",
        example=2
    )
    
    prediction_label: str = Field(
        ..., 
        description="Human-readable prediction label",
        example="Good"
    )
    
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score of the prediction (0-1)",
        example=0.85
    )
    
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across all classes",
        example={
            "Poor": 0.05,
            "Moderate": 0.10,
            "Good": 0.85
        }
    )
    
    message: str = Field(
        default="",
        description="Additional information or context about the prediction",
        example="Prediction based on auto-collected metrics"
    )
    
    mode: Optional[str] = Field(
        default=None,
        description="Prediction mode used: auto, scenario, or simple",
        example="auto"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata (scenario info, estimated features, etc.)"
    )


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(
        ..., 
        description="Service status: healthy, degraded, or unhealthy",
        example="healthy"
    )
    
    model_loaded: bool = Field(
        ..., 
        description="Whether ML model is loaded and ready",
        example=True
    )
    
    scenarios_loaded: int = Field(
        ..., 
        description="Number of scenarios loaded",
        example=10
    )
    
    version: str = Field(
        ..., 
        description="API version",
        example="1.0.0"
    )
    
    uptime_seconds: Optional[float] = Field(
        default=None,
        description="Server uptime in seconds"
    )


class ScenarioInfo(BaseModel):
    """Information about a single scenario"""
    id: int = Field(..., description="Scenario ID", example=1)
    name: str = Field(..., description="Scenario name", example="Home WiFi - Good Connection")
    description: str = Field(..., description="Scenario description", example="Ideal home network conditions")
    category: str = Field(..., description="Scenario category", example="home")
    expected_quality: str = Field(..., description="Expected quality", example="Good")
    parameters: Optional[Dict[str, float]] = Field(
        default=None,
        description="Network parameters for this scenario"
    )


class ScenarioListResponse(BaseModel):
    """Response for listing all scenarios"""
    count: int = Field(..., description="Total number of scenarios", example=10)
    scenarios: List[ScenarioInfo] = Field(..., description="List of available scenarios")
    categories: Optional[List[str]] = Field(
        default=None,
        description="Available scenario categories",
        example=["home", "office", "public", "mobile"]
    )


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error type", example="ValidationError")
    message: str = Field(..., description="Error message", example="Invalid input parameters")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )


class RootResponse(BaseModel):
    """Root endpoint response"""
    message: str = Field(
        default="Network Quality Prediction API",
        description="API welcome message"
    )
    version: str = Field(default="1.0.0", description="API version")
    endpoints: Dict[str, str] = Field(
        ...,
        description="Available API endpoints"
    )
    modes: Dict[str, str] = Field(
        ...,
        description="Available prediction modes"
    )