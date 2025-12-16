"""
FastAPI Server - Network Quality Prediction API
Hỗ trợ Mode 2 (Scenario) và Mode 3 (Simple)
FIXED: TCP Thread -> Socket.IO Bridge
FIXED: 'NoneType' object is not a mapping error
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import sys
import os
import time
import asyncio 

from .tcp_server import TCPServer

# --- Thêm project root vào sys.path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
BASE_DIR = PROJECT_ROOT

# ==================== IMPORTS ====================
from .routers import predict as predict_router
from .routers import scenarios as scenarios_router
from utils.model_wrapper import ModelWrapper
from .services.scenario_manager import ScenarioManager
from .services.smart_estimator import SmartEstimator
from .services.recommendation_service import RecommendationService
from .models.response_models import HealthResponse, PredictionResponse
import socketio

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
    version="1.1.0"
)

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
    return JSONResponse(status_code=400, content={"error": "ValidationError", "message": str(exc)})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "InternalServerError", "message": "An unexpected error occurred"})

# ==================== GLOBAL INSTANCES ====================

model_wrapper = None
scenario_manager = None
smart_estimator = None
recommendation_service = None
server_loop = None 

# Socket.IO Setup
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=False, 
    engineio_logger=False
)
socket_app = socketio.ASGIApp(sio, other_asgi_app=app, socketio_path="/ws/socket.io")

# Global State
global_state = {
    "metrics": None,
    "context": None, # Ban đầu là None
    "react_sid": None 
}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global model_wrapper, scenario_manager, smart_estimator, recommendation_service
    global server_loop
    
    # BẮT LẤY EVENT LOOP HIỆN TẠI
    server_loop = asyncio.get_running_loop()
    
    try:
        logger.info("🚀 Starting Network Quality Prediction API Server...")
        
        # Load Model
        model_wrapper = ModelWrapper(base_dir=BASE_DIR)
        
        # Load Scenarios
        scenario_manager = ScenarioManager()
        
        # Initialize Estimator
        smart_estimator = SmartEstimator()

        # Init Recommendation
        recommendation_service = RecommendationService(base_dir=BASE_DIR)
            
        # Inject dependencies
        predict_router.set_dependencies(model_wrapper, scenario_manager, smart_estimator, recommendation_service)
        scenarios_router.set_dependencies(scenario_manager)

        # Start TCP Server
        try:
            # Lưu ý port phải khớp với worker (9500)
            tcp_server = TCPServer(host="0.0.0.0", port=9500, callback_function=process_tcp_data)
            tcp_server.start() 
            logger.info("✅ TCP Server started on port 9500")
        except Exception as e:
            logger.error(f"❌ Failed to start TCP Server: {e}")

    except Exception as e:
        logger.critical(f"❌ Startup failed: {str(e)}")
        raise

# === HÀM CALLBACK XỬ LÝ DỮ LIỆU TỪ TCP ===
def process_tcp_data(payload: dict):
    """
    Hàm này chạy trong Thread riêng của TCP Server.
    """
    global smart_estimator, model_wrapper, recommendation_service
    global global_state, server_loop, sio 
    
    try:
        if payload.get("type") != "worker_data":
            return "Invalid data type"

        metrics_data = payload.get("data", {})
        
        # Lấy context hiện tại từ React. Nếu None thì dùng dict rỗng {}
        current_context = global_state.get("context") or {}
        
        # Gộp metrics từ Worker + Context từ React
        combined_input = {**metrics_data, **current_context}
        
        # 1. Estimate
        try:
            full_params = smart_estimator.estimate(combined_input)
        except Exception as e:
            logger.warning(f"Lỗi estimate: {e}")
            return "Estimation Error"

        # 2. Predict
        result = model_wrapper.predict(full_params)
        
        # 3. Recommendation
        recommendation = recommendation_service.get_recommendation(
            full_params, 
            result['prediction_label']
        )

        logger.info(f"🔮 TCP Prediction: {result['prediction_label']}")
        
        # --- B. CẦU NỐI SANG REACTJS (BRIDGE) ---
        react_sid = global_state.get("react_sid")
        
        if react_sid and server_loop:
            response_data = PredictionResponse(
                prediction=result['prediction'],
                prediction_label=result['prediction_label'],
                confidence=result['confidence'],
                probabilities=result['probabilities'],
                message="Real-time update from TCP Worker",
                mode="realtime",
                metadata={
                    "estimated_features_dict": full_params,
                    "contexts_used": current_context
                },
                insight=recommendation
            ).model_dump()

            asyncio.run_coroutine_threadsafe(
                sio.emit("prediction_update", response_data, to=react_sid),
                server_loop
            )
            logger.info(f"⚡ Đã bắn tín hiệu update sang React (SID: {react_sid})")
        else:
            # Không log debug liên tục để đỡ rác log
            pass

        return result['prediction_label']

    except Exception as e:
        logger.error(f"Lỗi xử lý TCP data: {e}")
        return "Error"

# ==================== ENDPOINTS ====================

app.include_router(predict_router.router)
app.include_router(scenarios_router.router)

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Network Quality Prediction API",
        "version": "1.2.0"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(detailed: bool = False):
    status = "healthy" if model_wrapper else "degraded"
    return HealthResponse(
        status=status,
        model_loaded=True,
        scenarios_loaded=len(scenario_manager.scenarios),
        version="1.1.0"
    )

# ==================== SOCKET.IO EVENTS ====================

@sio.event
async def connect(sid, environ, auth):
    logger.info(f"📡 Client Socket kết nối: {sid}")

@sio.event
async def disconnect(sid):
    if sid == global_state.get("react_sid"):
        global_state["react_sid"] = None
        logger.warning(f"🔌 React Dashboard ngắt kết nối: {sid}")

@sio.event
async def start_prediction(sid, data):
    """
    React gửi context và bắt đầu phiên đo
    """
    logger.info(f"[{sid}] React bắt đầu phiên đo Real-time")
    global_state["context"] = data
    global_state["react_sid"] = sid

if __name__ == "__main__":
    uvicorn.run(
        "server.main:socket_app", 
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )