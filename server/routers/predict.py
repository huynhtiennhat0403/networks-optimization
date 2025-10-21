"""
Prediction Router
Handles all prediction endpoints for 3 modes
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict
import logging

from models.request_models import AutoPredictRequest, ScenarioPredictRequest, SimplePredictRequest
from models.response_models import PredictionResponse
from services.scenario_manager import ScenarioManager
from services.smart_estimator import SmartEstimator

logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
    responses={404: {"description": "Not found"}}
)

# Dependencies (will be injected from main.py)
_model_wrapper = None
_scenario_manager = None
_smart_estimator = None


def set_dependencies(model_wrapper, scenario_manager, smart_estimator):
    """Set global dependencies (called from main.py)"""
    global _model_wrapper, _scenario_manager, _smart_estimator
    _model_wrapper = model_wrapper
    _scenario_manager = scenario_manager
    _smart_estimator = smart_estimator


def get_model_wrapper():
    """Dependency to get model wrapper"""
    if _model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model_wrapper


def get_scenario_manager():
    """Dependency to get scenario manager"""
    if _scenario_manager is None:
        raise HTTPException(status_code=503, detail="Scenario manager not initialized")
    return _scenario_manager


def get_smart_estimator():
    """Dependency to get smart estimator"""
    if _smart_estimator is None:
        raise HTTPException(status_code=503, detail="Smart estimator not initialized")
    return _smart_estimator


# ==================== ENDPOINTS ====================

@router.post("/auto", response_model=PredictionResponse)
async def predict_auto(
    request: AutoPredictRequest,
    model=Depends(get_model_wrapper)
):
    """
    Mode 1: Auto Collect
    
    Client sends automatically collected network metrics.
    Server uses them directly for prediction.
    
    **Use case:** Desktop/mobile apps with full network access
    """
    try:
        logger.info(f"[AUTO] Received {len(request.metrics)} features")
        
        # Validate feature count (should match training data)
        expected_features = model.expected_features
        if len(request.metrics) != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, got {len(request.metrics)}"
            )
        
        # Predict
        result = model.predict(request.metrics)
        
        logger.info(
            f"[AUTO] Prediction: {result['prediction_label']} "
            f"(confidence: {result['confidence']:.2%})"
        )
        
        return PredictionResponse(
            prediction=result['prediction'],
            prediction_label=result['prediction_label'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            message="Prediction based on auto-collected metrics",
            mode="auto"
        )
        
    except ValueError as e:
        logger.error(f"[AUTO] Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[AUTO] Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.post("/scenario", response_model=PredictionResponse)
async def predict_scenario(
    request: ScenarioPredictRequest,
    model=Depends(get_model_wrapper),
    scenario_mgr=Depends(get_scenario_manager)
):
    """
    Mode 2: Scenario Selection
    
    Client selects a predefined scenario by ID.
    Server loads scenario parameters and predicts.
    
    **Use case:** Quick testing, demos, or guided experiences
    """
    try:
        logger.info(f"[SCENARIO] Requested scenario ID: {request.scenario_id}")
        
        # Get scenario
        scenario = scenario_mgr.get_scenario(request.scenario_id)
        if not scenario:
            raise HTTPException(
                status_code=404, 
                detail=f"Scenario {request.scenario_id} not found"
            )
        
        # Predict using scenario parameters
        result = model.predict(scenario['parameters'])
        
        logger.info(
            f"[SCENARIO] '{scenario['name']}' → {result['prediction_label']} "
            f"(confidence: {result['confidence']:.2%})"
        )
        
        return PredictionResponse(
            prediction=result['prediction'],
            prediction_label=result['prediction_label'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            message=f"Prediction for scenario: {scenario['name']}",
            mode="scenario",
            metadata={
                "scenario_id": scenario['id'],
                "scenario_name": scenario['name'],
                "scenario_category": scenario['category']
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SCENARIO] Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simple", response_model=PredictionResponse)
async def predict_simple(
    request: SimplePredictRequest,
    model=Depends(get_model_wrapper),
    estimator=Depends(get_smart_estimator)
):
    """
    Mode 3: Simplified Input
    
    Client sends 3-5 key parameters.
    Server estimates missing features using smart heuristics.
    
    **Use case:** User-friendly interface, limited network access
    """
    try:
        logger.info(
            f"[SIMPLE] Input - throughput: {request.throughput}Mbps, "
            f"latency: {request.latency}ms, signal: {request.signal_strength}dBm"
        )
        
        # Convert request to dict
        simple_input = {
            'throughput': request.throughput,
            'latency': request.latency,
            'signal_strength': request.signal_strength,
            'user_activity': request.user_activity,
            'device_type': request.device_type,
            'location': request.location,
            'connection_type': request.connection_type
        }
        
        # Estimate full parameters
        full_params = estimator.estimate(simple_input)
        
        # Predict
        result = model.predict(full_params)
        
        logger.info(
            f"[SIMPLE] Prediction: {result['prediction_label']} "
            f"(confidence: {result['confidence']:.2%})"
        )
        
        return PredictionResponse(
            prediction=result['prediction'],
            prediction_label=result['prediction_label'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            message="Prediction based on simplified input with AI estimation",
            mode="simple",
            metadata={
                "estimated_features": [k for k in full_params.keys() 
                                      if k not in simple_input],
                "user_provided": list(simple_input.keys())
            }
        )
        
    except ValueError as e:
        logger.error(f"[SIMPLE] Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[SIMPLE] Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/modes")
async def list_prediction_modes():
    """
    List all available prediction modes and their descriptions
    """
    return {
        "modes": [
            {
                "name": "auto",
                "endpoint": "/predict/auto",
                "description": "Auto-collect full network metrics",
                "required_features": "All network parameters",
                "use_case": "Desktop/mobile apps with full network access"
            },
            {
                "name": "scenario",
                "endpoint": "/predict/scenario",
                "description": "Select from predefined scenarios",
                "required_features": "scenario_id only",
                "use_case": "Quick testing, demos, guided experiences"
            },
            {
                "name": "simple",
                "endpoint": "/predict/simple",
                "description": "Simplified input with AI estimation",
                "required_features": "throughput, latency, signal_strength + optional context",
                "use_case": "User-friendly interface, limited access"
            }
        ]
    }