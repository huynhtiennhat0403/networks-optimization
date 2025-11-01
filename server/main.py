"""
FastAPI Server - Network Quality Prediction API
Supports 3 prediction modes: Auto, Scenario, Simple
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import uvicorn
import logging
import sys
import os
import time

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Define global base directory for consistent file paths ---
BASE_DIR = PROJECT_ROOT

# Import services
from utils.model_wrapper import ModelWrapper
from .services.scenario_manager import ScenarioManager
from .services.smart_estimator import SmartEstimator

# Import models from models folder
from .models.request_models import (
    AutoPredictRequest,
    ScenarioPredictRequest,
    SimplePredictRequest
)
from .models.response_models import (
    PredictionResponse,
    HealthResponse,
    ScenarioInfo,
    ScenarioListResponse,
    ErrorResponse,
    RootResponse
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# ==================== GLOBAL ERROR HANDLERS ====================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation/value errors"""
    logger.error(f"ValueError: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "ValidationError",
            "message": str(exc),
            "details": None
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": str(exc) if logger.level == logging.DEBUG else None
        }
    )

# ==================== GLOBAL INSTANCES ====================

model_wrapper = None
scenario_manager = None
smart_estimator = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global model_wrapper, scenario_manager, smart_estimator
    
    start_time = time.time()
    
    try:
        logger.info("=" * 80)
        logger.info("🚀 Starting Network Quality Prediction API Server...")
        logger.info("=" * 80)
        logger.info(f"📁 Project root: {PROJECT_ROOT}")
        logger.info(f"📁 Base directory: {BASE_DIR}")
        
        # Load model
        logger.info("\n📦 Loading ML model...")
        try:
            model_wrapper = ModelWrapper()
            logger.info("✅ Model loaded successfully")
            # Get model details if available
            if hasattr(model_wrapper, 'feature_names'):
                logger.info(f"   - Number of features: {len(model_wrapper.feature_names)}")
            if hasattr(model_wrapper, 'model'):
                logger.info(f"   - Model type: {type(model_wrapper.model).__name__}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise
        
        # Load scenarios
        logger.info("\n🎬 Loading scenarios...")
        try:
            scenario_manager = ScenarioManager()
            logger.info(f"✅ Loaded {len(scenario_manager.scenarios)} scenarios")
            for scenario_id, scenario in scenario_manager.scenarios.items():
                logger.info(f"   - Scenario {scenario_id}: {scenario.get('name', 'Unknown')}")
        except Exception as e:
            logger.error(f"❌ Failed to load scenarios: {str(e)}")
            raise
        
        # Initialize estimator
        logger.info("\n🤖 Initializing smart estimator...")
        try:
            smart_estimator = SmartEstimator()
            logger.info("✅ Smart estimator ready")
        except Exception as e:
            logger.error(f"❌ Failed to initialize estimator: {str(e)}")
            raise
        
        elapsed_time = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info(f"🎉 Server ready to accept requests! (took {elapsed_time:.2f}s)")
        logger.info(f"📚 API Docs: http://localhost:8000/docs")
        logger.info(f"📚 Alternative Docs: http://localhost:8000/redoc")
        logger.info("=" * 80 + "\n")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.critical("=" * 80)
        logger.critical(f"❌ Startup failed after {elapsed_time:.2f}s: {str(e)}")
        logger.critical("=" * 80)
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
async def health_check(detailed: bool = False):
    """
    Health check endpoint
    
    Parameters:
    - detailed: If true, return detailed component status
    """
    try:
        if not model_wrapper or not scenario_manager:
            return HealthResponse(
                status="unhealthy",
                model_loaded=False,
                scenarios_loaded=0,
                version="1.0.0"
            )
        
        model_health = model_wrapper.health_check()
        
        # Determine overall status
        all_ready = all([
            model_health.get('ready', False),
            scenario_manager.scenarios,
            smart_estimator is not None
        ])
        
        status = "healthy" if all_ready else ("degraded" if model_health.get('model_loaded') else "unhealthy")
        
        health_response = HealthResponse(
            status=status,
            model_loaded=model_health.get('model_loaded', False),
            scenarios_loaded=len(scenario_manager.scenarios),
            version="1.0.0"
        )
        
        logger.info(f"Health check - Status: {status}, Model: {health_response.model_loaded}, Scenarios: {health_response.scenarios_loaded}")
        
        return health_response
        
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