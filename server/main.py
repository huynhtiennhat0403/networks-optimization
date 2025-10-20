"""
FastAPI Server - Network Quality Prediction API
Supports 3 prediction modes: Auto, Scenario, Simple
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import uvicorn
import logging
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_wrapper import ModelWrapper
from services.scenario_manager import ScenarioManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== PYDANTIC MODELS ====================

class PredictionResponse(BaseModel):
    """Standard prediction response"""
    prediction: int
    prediction_label: str
    confidence: float
    probabilities: Dict[str, float]
    message: str = ""

class AutoPredictRequest(BaseModel):
    """Request for Mode 1: Auto Collect"""
    metrics: Dict[str, float] = Field(..., description="Network metrics collected automatically")

class ScenarioPredictRequest(BaseModel):
    """Request for Mode 2: Scenario Selection"""
    scenario_id: int = Field(..., ge=1, description="ID of selected scenario")

class SimplePredictRequest(BaseModel):
    """Request for Mode 3: Simplified Input"""
    throughput: float = Field(..., ge=1, le=100, description="Throughput in Mbps")
    latency: float = Field(..., ge=1, le=100, description="Latency in ms")
    signal_strength: float = Field(..., ge=-100, le=-40, description="Signal strength in dBm")
    user_activity: str = Field(..., description="Current activity: browsing, streaming, gaming")
    device_type: str = Field(default="laptop", description="Device type: laptop, phone, tablet")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    scenarios_loaded: int
    version: str

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Network Quality Prediction API",
    description="Predicts network quality using Random Forest model with 3 modes",
    version="1.0.0"
)

# CORS middleware - Allow client to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify client IPs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GLOBAL INSTANCES ====================

model_wrapper = None
scenario_manager = None
smart_estimator = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global model_wrapper, scenario_manager, smart_estimator
    
    try:
        logger.info("🚀 Starting server...")
        
        # Load model
        logger.info("Loading ML model...")
        model_wrapper = ModelWrapper()
        logger.info("✅ Model loaded successfully")
        
        # Load scenarios
        logger.info("Loading scenarios...")
        scenario_manager = ScenarioManager()
        logger.info(f"✅ Loaded {len(scenario_manager.scenarios)} scenarios")
        
        # Initialize estimator
        logger.info("Initializing smart estimator...")
        smart_estimator = SmartEstimator()
        logger.info("✅ Smart estimator ready")
        
        logger.info("🎉 Server ready to accept requests!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

# ==================== ENDPOINTS ====================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Network Quality Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "scenarios": "/scenarios/list",
            "predict_auto": "/predict/auto",
            "predict_scenario": "/predict/scenario",
            "predict_simple": "/predict/simple"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        model_health = model_wrapper.health_check() if model_wrapper else {}
        
        return HealthResponse(
            status="healthy" if model_health.get('ready', False) else "unhealthy",
            model_loaded=model_health.get('model_loaded', False),
            scenarios_loaded=len(scenario_manager.scenarios) if scenario_manager else 0,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/scenarios/list", tags=["Scenarios"])
async def list_scenarios():
    """Get list of available scenarios"""
    try:
        if not scenario_manager:
            raise HTTPException(status_code=500, detail="Scenario manager not initialized")
        
        scenarios = scenario_manager.get_all_scenarios()
        return {
            "count": len(scenarios),
            "scenarios": scenarios
        }
    except Exception as e:
        logger.error(f"Error listing scenarios: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scenarios/{scenario_id}", tags=["Scenarios"])
async def get_scenario(scenario_id: int):
    """Get details of a specific scenario"""
    try:
        if not scenario_manager:
            raise HTTPException(status_code=500, detail="Scenario manager not initialized")
        
        scenario = scenario_manager.get_scenario(scenario_id)
        if not scenario:
            raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found")
        
        return scenario
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting scenario: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/auto", response_model=PredictionResponse, tags=["Prediction"])
async def predict_auto(request: AutoPredictRequest):
    """
    Mode 1: Auto Collect
    Client sends automatically collected network metrics
    """
    try:
        if not model_wrapper:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        logger.info(f"[AUTO] Received metrics: {len(request.metrics)} features")
        
        # Predict
        result = model_wrapper.predict(request.metrics)
        
        logger.info(f"[AUTO] Prediction: {result['prediction_label']} ({result['confidence']:.2%})")
        
        return PredictionResponse(
            prediction=result['prediction'],
            prediction_label=result['prediction_label'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            message="Prediction based on auto-collected metrics"
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/scenario", response_model=PredictionResponse, tags=["Prediction"])
async def predict_scenario(request: ScenarioPredictRequest):
    """
    Mode 2: Scenario Selection
    Client sends scenario ID, server loads pre-defined parameters
    """
    try:
        if not model_wrapper or not scenario_manager:
            raise HTTPException(status_code=500, detail="Services not initialized")
        
        logger.info(f"[SCENARIO] Requested scenario ID: {request.scenario_id}")
        
        # Get scenario parameters
        scenario = scenario_manager.get_scenario(request.scenario_id)
        if not scenario:
            raise HTTPException(status_code=404, detail=f"Scenario {request.scenario_id} not found")
        
        # Predict using scenario parameters
        result = model_wrapper.predict(scenario['parameters'])
        
        logger.info(f"[SCENARIO] '{scenario['name']}' → {result['prediction_label']} ({result['confidence']:.2%})")
        
        return PredictionResponse(
            prediction=result['prediction'],
            prediction_label=result['prediction_label'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            message=f"Prediction for scenario: {scenario['name']}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scenario prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/simple", response_model=PredictionResponse, tags=["Prediction"])
async def predict_simple(request: SimplePredictRequest):
    """
    Mode 3: Simplified Input
    Client sends 3-5 key parameters, server estimates the rest
    """
    try:
        if not model_wrapper or not smart_estimator:
            raise HTTPException(status_code=500, detail="Services not initialized")
        
        logger.info(f"[SIMPLE] Input: throughput={request.throughput}, latency={request.latency}, signal={request.signal_strength}")
        
        # Convert simple input to full parameters
        simple_input = {
            'throughput': request.throughput,
            'latency': request.latency,
            'signal_strength': request.signal_strength,
            'user_activity': request.user_activity,
            'device_type': request.device_type
        }
        
        full_params = smart_estimator.estimate(simple_input)
        
        # Predict
        result = model_wrapper.predict(full_params)
        
        logger.info(f"[SIMPLE] Prediction: {result['prediction_label']} ({result['confidence']:.2%})")
        
        return PredictionResponse(
            prediction=result['prediction'],
            prediction_label=result['prediction_label'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            message="Prediction based on simplified input with AI estimation"
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Simple prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    print("="*80)
    print("🚀 NETWORK QUALITY PREDICTION SERVER")
    print("="*80)
    print("\n📍 Server will start on: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔄 Alternative Docs: http://localhost:8000/redoc")
    print("\n⏳ Starting server...\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Listen on all interfaces (allows connection from other machines)
        port=8000,
        reload=True,  # Auto-reload on code changes (disable in production)
        log_level="info"
    )