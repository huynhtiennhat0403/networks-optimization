import socket
import json
import time
import speedtest
import psutil
import subprocess
import re
import platform
import logging

# --- Cấu hình logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TCP_Worker")

# --- Cấu hình kết nối TCP ---
SERVER_IP = '127.0.0.1'
SERVER_PORT = 9500       
BUFFER_SIZE = 4096

# --- 1. Các hàm đo đạc (Giữ nguyên logic cũ) ---
def get_speed_metrics():
    try:
        logger.info("Dang chay speedtest...")
        s = speedtest.Speedtest()
        s.get_best_server()
        s.download()
        results = s.results.dict()
        return results['ping'], results['download'] / 1_000_000
    except Exception as e:
        logger.warning(f"Loi speedtest: {e}")
        # Trả về giá trị giả lập nếu lỗi (để test code mạng)
        return 45.0, 50.0 

def get_battery_level():
    battery = psutil.sensors_battery()
    return battery.percent if battery else 100

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

# --- 2. Gửi dữ liệu qua TCP ---
def send_data_via_tcp(data):
    """
    Hàm cốt lõi của Lập trình mạng: Tạo socket, connect, send, recv
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        logger.info(f"Dang ket noi toi TCP Server {SERVER_IP}:{SERVER_PORT}...")
        client_socket.connect((SERVER_IP, SERVER_PORT))
        
        # Serialize JSON và encode sang bytes
        message = json.dumps(data).encode('utf-8')
        
        # Gửi dữ liệu
        client_socket.sendall(message)
        logger.info(f"Da gui {len(message)} bytes du lieu.")

        # Nhận phản hồi từ Server (ACK hoặc Kết quả dự đoán)
        response = client_socket.recv(BUFFER_SIZE)
        logger.info(f"Phan hoi tu Server: {response.decode('utf-8')}")

    except ConnectionRefusedError:
        logger.error("Khong the ket noi. Server TCP co dang chay khong?")
    except Exception as e:
        logger.error(f"Loi TCP: {e}")
    finally:
        client_socket.close()
        logger.info("Da dong ket noi.")

# --- 3. Vòng lặp chính ---
def start_worker():
    logger.info("Worker khoi dong (TCP Mode)...")
    
    while True:
        # Thu thập dữ liệu
        latency, throughput = get_speed_metrics()
        
        payload = {
            "type": "worker_data", # Định danh loại tin nhắn
            "data": {
                "latency": latency,
                "throughput": throughput,
                "battery_level": get_battery_level(),
                "signal_strength": get_signal_strength(),
                # Thêm context mặc định nếu Worker chạy tự động
                "user_speed": 10.0, # Giả lập vận tốc
                "user_activity": "streaming",
                "device_type": "laptop",
                "location": "home",
                "connection_type": "4g"
            }
        }
        
        # Gửi qua TCP
        send_data_via_tcp(payload)
        
        # Nghỉ 10 giây trước khi đo lại
        time.sleep(10)

if __name__ == "__main__":
    start_worker()