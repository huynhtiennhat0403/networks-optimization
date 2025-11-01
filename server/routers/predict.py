"""
Prediction Router
Handles all prediction endpoints for 3 modes
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict
import logging

from ..models.request_models import  ScenarioPredictRequest, SimplePredictRequest
from ..models.response_models import PredictionResponse
from ..services.scenario_manager import ScenarioManager
from ..services.smart_estimator import SmartEstimator

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
                "scenario_name": scenario['name']
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
    
    Client sends 5 key parameters (all user inputs):
    - **user_speed** (km/h) - User manually inputs their speed
    - **battery_level** (%) - Device battery percentage
    - **signal_strength** (dBm) - Estimated from signal bars (1-4 bars)
    - **latency** (ms) - User inputs latency (can use speedtest apps)
    - **throughput** (Mbps) - User inputs throughput (can use speedtest apps)
    
    Plus optional contexts:
    - **user_activity**: browsing, streaming, gaming, downloading
    - **device_type**: phone, laptop, tablet
    - **location**: home, office, public, outdoor
    - **connection_type**: 4g, 5g
    
    Server estimates missing features using smart heuristics.
    
    **Use case:** User-friendly interface for manual input
    """
    try:
        logger.info(
            f"[SIMPLE] User inputs - speed: {request.user_speed}km/h, "
            f"battery: {request.battery_level}%, signal: {request.signal_strength}dBm"
        )
        logger.info(
            f"[SIMPLE] User inputs - latency: {request.latency}ms, "
            f"throughput: {request.throughput}Mbps"
        )
        logger.info(
            f"[SIMPLE] Context - activity: {request.user_activity}, "
            f"device: {request.device_type}, location: {request.location}, "
            f"connection: {request.connection_type}"
        )
        
        # Convert request to dict for estimator
        simple_input = {
            'user_speed': request.user_speed,
            'battery_level': request.battery_level,
            'signal_strength': request.signal_strength,
            'latency': request.latency,
            'throughput': request.throughput,
            'user_activity': request.user_activity,
            'device_type': request.device_type,
            'location': request.location,
            'connection_type': request.connection_type
        }
        
        # Estimate full parameters using smart estimator
        full_params = estimator.estimate(simple_input)
        
        logger.info(f"[SIMPLE] Estimated {len(full_params)} parameters from smart estimator")
        
        # Predict using estimated parameters
        result = model.predict(full_params)
        
        logger.info(
            f"[SIMPLE] Prediction: {result['prediction_label']} "
            f"(confidence: {result['confidence']:.2%})"
        )
        
        # Identify which features were user-provided vs estimated
        user_provided_features = [
            'User Speed (m/s)',  # converted from km/h
            'Battery Level (%)',
            'Signal Strength (dBm)',
            'Latency (ms)',
            'Throughput (Mbps)',
        ]
        
        estimated_features = [k for k in full_params.keys() if k not in user_provided_features]
        
        return PredictionResponse(
            prediction=result['prediction'],
            prediction_label=result['prediction_label'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            message="Prediction based on user inputs with AI estimation",
            mode="simple",
            metadata={
                "user_provided_count": len(user_provided_features),
                "estimated_count": len(estimated_features),
                "estimated_features": estimated_features,
                "contexts_used": {
                    "user_activity": request.user_activity,
                    "device_type": request.device_type,
                    "location": request.location,
                    "connection_type": request.connection_type
                }
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
                "required_features": "All network parameters (auto-collected)",
                "use_case": "Desktop/mobile apps with full network access",
                "input_count": "~20 features"
            },
            {
                "name": "scenario",
                "endpoint": "/predict/scenario",
                "description": "Select from predefined scenarios",
                "required_features": "scenario_id only",
                "use_case": "Quick testing, demos, guided experiences",
                "input_count": "1 parameter (scenario ID)"
            },
            {
                "name": "simple",
                "endpoint": "/predict/simple",
                "description": "Simplified input with AI estimation",
                "required_features": "5 user inputs + optional contexts",
                "use_case": "User-friendly manual input interface",
                "input_count": "5 required + 4 optional contexts",
                "details": {
                    "user_inputs": [
                        "user_speed (km/h) - Manual input",
                        "battery_level (%) - From device",
                        "signal_strength (dBm) - From signal bars (1-4)",
                        "latency (ms) - From speedtest apps",
                        "throughput (Mbps) - From speedtest apps"
                    ],
                    "optional_contexts": [
                        "user_activity (browsing/streaming/gaming/downloading)",
                        "device_type (phone/laptop/tablet)",
                        "location (home/office/public/outdoor/vehicle)",
                        "connection_type (4g/5g)"
                    ],
                    "estimated_by_ai": [
                        "SNR, BER, PDR, Modulation Scheme",
                        "User Direction, Handover Events",
                        "Distance from Base Station",
                        "Network Congestion, Retransmission Count",
                        "Power Consumption, Transmission Power"
                    ]
                }
            }
        ],
        "recommendation": {
            "for_developers": "Use 'auto' mode for maximum accuracy",
            "for_users": "Use 'simple' mode for easy manual input",
            "for_testing": "Use 'scenario' mode for quick demos"
        }
    }