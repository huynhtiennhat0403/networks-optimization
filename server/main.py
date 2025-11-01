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
from .models.response_models import HealthResponse, PredictionResponse

# 4. --- Import Socket.IO ---
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
        "message": "Network Quality Prediction API (Modes: Scenario, Simple, Real-time)",
        "version": "1.2.0",
        "rest_api_docs": "/docs",
        "websocket_path": "/ws/socket.io"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(detailed: bool = False):
    """Health check endpoint"""
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

# ==========================================================
# --- CÀI ĐẶT SOCKET.IO (WEBSOCKET) ---
# ==========================================================

# 1. Tạo đối tượng Socket.IO Server (sio)
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*", # Cho phép React và Worker kết nối
    logger=True,
    engineio_logger=True
)

# 2. Tạo ứng dụng ASGI cho Socket.IO
socket_app = socketio.ASGIApp(
    sio,
    socketio_path="/ws/socket.io"
)

# 3. Mount ứng dụng Socket.IO vào FastAPI
# Bất kỳ request nào tới /ws đều sẽ do socket_app xử lý
app.mount("/ws", socket_app)

# 4. --- Bộ lưu trữ trạng thái ---
# Dùng để lưu trữ dữ liệu tạm thời từ Worker và React
# Key là `sid` (session ID) của client, value là dict chứa metrics và context
client_state = {}

async def trigger_prediction(sid: str):
    """
    Hàm lõi: Khi có đủ 2 phần dữ liệu (metrics + context),
    gọi SmartEstimator và Model, sau đó gửi trả kết quả.
    """
    global model_wrapper, smart_estimator
    
    state = client_state.get(sid)
    
    # Kiểm tra xem đã đủ 2 phần dữ liệu chưa
    if not state or "metrics" not in state or "context" not in state:
        logger.debug(f"[{sid}] Chưa đủ dữ liệu, đang chờ...")
        return

    try:
        logger.info(f"[{sid}] Đã đủ 2 phần dữ liệu, bắt đầu dự đoán...")
        
        # 1. Gom 9 thông số
        # 4 từ worker, 5 từ react
        simple_input = {
            **state["metrics"], # Gồm: latency, throughput, battery_level, signal_strength
            **state["context"]  # Gồm: user_speed, user_activity, device_type, location, connection_type
        }
        
        # 2. Ước tính (Giống hệt predict_simple)
        full_params = smart_estimator.estimate(simple_input)
        logger.info(f"[{sid}] Đã ước tính {len(full_params)} thông số.")
        
        # 3. Dự đoán (Giống hệt predict_simple)
        result = model_wrapper.predict(full_params)
        logger.info(f"[{sid}] Kết quả: {result['prediction_label']}")

        # 4. Chuẩn bị response (Giống hệt predict_simple)
        response = PredictionResponse(
            prediction=result['prediction'],
            prediction_label=result['prediction_label'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            message="Prediction based on real-time auto-collected metrics",
            mode="realtime", # Mode mới
            metadata={
                "estimated_features_dict": full_params, # Gửi toàn bộ params đã ước tính
                "contexts_used": state["context"]
            }
        )
        
        # 5. Gửi kết quả NGƯỢC LẠI cho React Dashboard
        await sio.emit(
            "prediction_update",  # Tên sự kiện
            response.model_dump(),  # Chuyển Pydantic model về dict
            to=sid                   # Chỉ gửi cho client này
        )
        logger.info(f"[{sid}] Đã gửi 'prediction_update' cho client.")

    except Exception as e:
        logger.error(f"[{sid}] Lỗi khi dự đoán real-time: {e}")
        # Gửi lỗi về cho React
        await sio.emit("prediction_error", {"error": str(e)}, to=sid)

# 5. --- Các trình xử lý sự kiện (Event Handlers) ---

@sio.event
async def connect(sid, environ, auth):
    """Client (Worker hoặc React) kết nối"""
    logger.info(f"📡 Client đã kết nối: {sid}")
    # Khởi tạo bộ lưu trữ trạng thái rỗng cho client này
    client_state[sid] = {}

@sio.event
async def disconnect(sid):
    """Client ngắt kết nối"""
    logger.warning(f"🔌 Client đã ngắt kết nối: {sid}")
    # Xóa trạng thái của client này
    if sid in client_state:
        del client_state[sid]

@sio.event
async def worker_metrics(sid, data):
    """
    Nhận dữ liệu từ 'worker.py' (4 thông số)
    """
    logger.info(f"[{sid}] Nhận 'worker_metrics': {data}")
    if sid in client_state:
        client_state[sid]["metrics"] = data
        # Gọi hàm lõi để kiểm tra và dự đoán
        await trigger_prediction(sid)

@sio.event
async def context_update(sid, data):
    """
    Nhận dữ liệu từ 'React Dashboard' (5 thông số bối cảnh)
    """
    logger.info(f"[{sid}] Nhận 'context_update': {data}")
    if sid in client_state:
        client_state[sid]["context"] = data
        # Gọi hàm lõi để kiểm tra và dự đoán
        await trigger_prediction(sid)

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    print("="*80)
    print("🚀 NETWORK QUALITY PREDICTION SERVER (v1.2.0 - Real-time)")
    print("="*80)
    print("\n📍 Server sẽ start on: http://localhost:8000")
    print("📚 API Docs (REST): http://localhost:8000/docs")
    print(f"📡 Socket.IO (WS) listening on: /ws/socket.io")
    print("\n⏳ Starting server...\n")
    
    # uvicorn.run sẽ tự động chạy 'app' (đã bao gồm cả FastAPI và Socket.IO)
    uvicorn.run(
        "server.main:app", 
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )