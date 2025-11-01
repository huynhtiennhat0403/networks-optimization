import socketio
import time
import speedtest
import psutil
import subprocess
import re
import platform
import logging

# --- Cấu hình logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Worker")

# --- Địa chỉ Server ---
SERVER_URL = "http://localhost:8000"
SOCKETIO_PATH = "/ws/socket.io" 

# --- 1. Đo Speedtest (Latency & Throughput) ---
def get_speed_metrics():
    """
    Sử dụng speedtest-cli để đo lường
    - Latency (ping)
    - Throughput (download)
    """
    try:
        logger.info("Đang chạy speedtest...")
        s = speedtest.Speedtest()
        s.get_best_server()
        s.download()
        
        results = s.results.dict()
        
        latency_ms = results['ping']
        # Chuyển đổi từ bits/s sang Megabits/s
        throughput_mbps = results['download'] / 1_000_000 
        
        logger.info(f"Speedtest thành công: Latency={latency_ms:.2f} ms, Throughput={throughput_mbps:.2f} Mbps")
        return latency_ms, throughput_mbps
        
    except Exception as e:
        logger.warning(f"Không thể đo speedtest: {e}")
        return None, None

# --- 2. Đo Pin ---
def get_battery_level():
    """
    Sử dụng psutil để lấy % pin hiện tại
    """
    battery = psutil.sensors_battery()
    if battery:
        logger.info(f"Đo pin thành công: {battery.percent}%")
        return battery.percent
    else:
        logger.info("Không phát hiện thấy pin (có thể là máy bàn), mặc định 100%")
        return 100 

# --- 3. Đo Cường độ Sóng (Signal Strength - dBm) ---
def get_signal_strength():
    """
    Sử dụng các lệnh hệ thống để lấy cường độ sóng (dBm).
    Đây là phần phức tạp nhất vì khác nhau trên mỗi HĐH.
    """
    os_type = platform.system()
    signal_dbm = -75.0 # Giá trị mặc định (tương đương 4 vạch sóng)
    
    try:
        if os_type == "Windows":
            cmd_output = subprocess.check_output("netsh wlan show interfaces", shell=True).decode('utf-8')
            # Windows thường chỉ trả về %
            match = re.search(r"Signal\s*:\s*(\d+)%", cmd_output)
            if match:
                signal_percent = int(match.group(1))
                # Đây là một phép nội suy RẤT thô từ % sang dBm (dựa trên 4 vạch)
                # (Bạn có thể cải thiện logic này)
                if signal_percent > 90: signal_dbm = -55.0 # Rất mạnh
                elif signal_percent > 80: signal_dbm = -65.0
                elif signal_percent > 70: signal_dbm = -75.0 # 4 vạch
                elif signal_percent > 50: signal_dbm = -85.0 # 3 vạch
                elif signal_percent > 30: signal_dbm = -95.0 # 2 vạch
                else: signal_dbm = -105.0 # 1 vạch
                logger.info(f"Đo sóng (Windows): {signal_percent}% -> {signal_dbm} dBm")
                
        elif os_type == "Darwin": # macOS
            cmd_output = subprocess.check_output(
                "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -I",
                shell=True
            ).decode('utf-8')
            # macOS trả về dBm trực tiếp (ví dụ: CtlRSSI: -54)
            match = re.search(r"CtlRSSI:\s*(-?\d+)", cmd_output)
            if match:
                signal_dbm = float(match.group(1))
                logger.info(f"Đo sóng (macOS): {signal_dbm} dBm")

        elif os_type == "Linux":
            cmd_output = subprocess.check_output("iwconfig", shell=True).decode('utf-8')
            # Linux trả về dBm trực tiếp (ví dụ: Signal level=-47 dBm)
            match = re.search(r"Signal level=(-?\d+)\s*dBm", cmd_output)
            if match:
                signal_dbm = float(match.group(1))
                logger.info(f"Đo sóng (Linux): {signal_dbm} dBm")
        
        else:
            logger.warning(f"Hệ điều hành {os_type} không hỗ trợ đo sóng, dùng mặc định.")

    except Exception as e:
        logger.warning(f"Lỗi khi đo sóng, dùng mặc định: {e}")

    # Đảm bảo giá trị nằm trong phạm vi mô hình của bạn
    return max(-120.0, min(-50.0, signal_dbm))


# --- 4. Khởi chạy Client Socket ---
def start_worker():
    sio = socketio.Client(logger=True, engineio_logger=True)

    @sio.event
    def connect():
        logger.info("✅ Đã kết nối thành công tới Server!")

    @sio.event
    def connect_error(data):
        logger.error(f"❌ Kết nối thất bại: {data}")

    @sio.event
    def disconnect():
        logger.warning("🔌 Đã ngắt kết nối khỏi Server.")

    try:
        logger.info(f"Đang kết nối tới server {SERVER_URL}...")
        sio.connect(SERVER_URL, socketio_path=SOCKETIO_PATH)
        
        # Vòng lặp chính của Worker
        while True:
            # 1. Đo các thông số
            latency, throughput = get_speed_metrics()
            battery = get_battery_level()
            signal = get_signal_strength()
            
            # 2. Tạo payload
            # Đây là 4/5 thông số bắt buộc của "Simplified Input"
            if latency is not None and throughput is not None:
                payload = {
                    "latency": latency,
                    "throughput": throughput,
                    "battery_level": battery,
                    "signal_strength": signal
                }
                
                # 3. Gửi dữ liệu lên server
                logger.info(f"Gửi 'worker_metrics': {payload}")
                sio.emit("worker_metrics", payload)
            
            # 4. Chờ 15 giây cho lần đo tiếp theo
            logger.info("Chờ 15 giây cho lần đo tiếp theo...")
            time.sleep(15)
            
    except socketio.exceptions.ConnectionError as e:
        logger.critical(f"Không thể kết nối tới server. Server có đang chạy không? {e}")
    except KeyboardInterrupt:
        logger.info("Ngắt bởi người dùng.")
    finally:
        if sio.connected:
            sio.disconnect()
            logger.info("Đã ngắt kết nối.")

# --- Điểm vào ---
if __name__ == "__main__":
    start_worker()