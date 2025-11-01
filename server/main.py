"""
FastAPI Server - Network Quality Prediction API
Hỗ trợ Mode 2 (Scenario) và Mode 3 (Simple)
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import sys
import os
import time

# --- Thêm project root vào sys.path ---

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Định nghĩa BASE_DIR ---

BASE_DIR = PROJECT_ROOT

# ==================== IMPORTS ====================
# 1. Import các routers
from .routers import predict as predict_router
from .routers import scenarios as scenarios_router

# 2. Import các services

from utils.model_wrapper import ModelWrapper
from .services.scenario_manager import ScenarioManager
from .services.smart_estimator import SmartEstimator

# 3. Import các response models 
from .models.response_models import HealthResponse

# Setup logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Network Quality Prediction API",
    description="Dự đoán chất lượng mạng với Mode 2 (Scenario) và Mode 3 (Simple)",
    version="1.1.0" # Cập nhật version
)

# CORS middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GLOBAL ERROR HANDLERS ====================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.error(f"ValueError: {str(exc)}")
    return JSONResponse(status_code=400, content={"error": "ValidationError", "message": str(exc)})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(status_code=500, content={"error": "InternalServerError", "message": "An unexpected error occurred"})

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
        logger.info("🚀 Starting Network Quality Prediction API Server (v1.1.0)...")
        logger.info("=" * 80)
        logger.info(f"📁 Project root: {PROJECT_ROOT}")
        logger.info(f"📁 Base directory: {BASE_DIR}")
        
        # --- 1. Load Model ---
        logger.info("\n📦 Loading ML model...")
        try:
            # Truyền BASE_DIR vào ModelWrapper
            model_wrapper = ModelWrapper(base_dir=BASE_DIR)
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise
        
        # --- 2. Load Scenarios ---
        logger.info("\n🎬 Loading scenarios...")
        try:
            scenario_manager = ScenarioManager()
            logger.info(f"✅ Loaded {len(scenario_manager.scenarios)} scenarios")
        except Exception as e:
            logger.error(f"❌ Failed to load scenarios: {str(e)}")
            raise
        
        # --- 3. Initialize Estimator ---
        logger.info("\n🤖 Initializing smart estimator...")
        try:
            smart_estimator = SmartEstimator()
            logger.info("✅ Smart estimator ready")
        except Exception as e:
            logger.error(f"❌ Failed to initialize estimator: {str(e)}")
            raise
            
        # --- 4. Inject dependencies vào Routers ---
        # Gửi các services đã khởi tạo vào các file router
        logger.info("\n🔗 Injecting dependencies into routers...")
        predict_router.set_dependencies(model_wrapper, scenario_manager, smart_estimator)
        scenarios_router.set_dependencies(scenario_manager)
        logger.info("✅ Dependencies injected")

        elapsed_time = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info(f"🎉 Server ready to accept requests! (took {elapsed_time:.2f}s)")
        logger.info(f"📚 API Docs: http://localhost:8000/docs")
        logger.info("=" * 80 + "\n")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.critical("=" * 80)
        logger.critical(f"❌ Startup failed after {elapsed_time:.2f}s: {str(e)}")
        logger.critical("=" * 80)
        raise

# ==================== ENDPOINTS ====================

app.include_router(predict_router.router)
app.include_router(scenarios_router.router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Network Quality Prediction API (Modes: Scenario, Simple)",
        "version": "1.1.0",
        "endpoints": {
            "health": "/health",
            "scenarios": "/scenarios/list",
            "predict_scenario": "/predict/scenario",
            "predict_simple": "/predict/simple",
            "modes_info": "/predict/modes" # Endpoint mới từ predict.py
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(detailed: bool = False):
    """Health check endpoint"""
    # (Phần này giữ nguyên logic cũ)
    try:
        if not model_wrapper or not scenario_manager:
            raise HTTPException(status_code=503, detail="Services not initialized")
        
        model_health = model_wrapper.health_check()
        
        all_ready = all([
            model_health.get('ready', False),
            scenario_manager.scenarios,
            smart_estimator is not None
        ])
        
        status = "healthy" if all_ready else "degraded"
        
        health_response = HealthResponse(
            status=status,
            model_loaded=model_health.get('model_loaded', False),
            scenarios_loaded=len(scenario_manager.scenarios),
            version="1.1.0"
        )
        
        logger.info(f"Health check - Status: {status}")
        return health_response
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    print("="*80)
    print("🚀 NETWORK QUALITY PREDICTION SERVER (v1.1.0)")
    print("="*80)
    print("\n📍 Server will start on: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\n⏳ Starting server...\n")
    
    uvicorn.run(
        # Sửa ở đây: trỏ đến app trong file main.py của thư mục 'server'
        "server.main:app", 
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )