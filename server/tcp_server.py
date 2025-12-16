import socket
import threading
import json
import logging
from typing import Callable

logger = logging.getLogger("TCP_Server")

class TCPServer(threading.Thread):
    def __init__(self, host: str, port: int, callback_function: Callable):
        super().__init__()
        self.host = host
        self.port = port
        self.callback = callback_function # Hàm callback để xử lý data nhận được
        self.running = True
        self.server_socket = None

    def run(self):
        """Logic chính của Thread"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Cho phép reuse port ngay lập tức sau khi restart
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5) # Backlog = 5
            logger.info(f"🚀 TCP Server dang lang nghe tai {self.host}:{self.port}")

            while self.running:
                try:
                    conn, addr = self.server_socket.accept()
                    # Xử lý mỗi client trong một thread con (nếu muốn handle nhiều worker đồng thời)
                    client_handler = threading.Thread(
                        target=self.handle_client, 
                        args=(conn, addr)
                    )
                    client_handler.start()
                except OSError:
                    break # Socket bị đóng

        except Exception as e:
            logger.error(f"Loi khoi tao TCP Server: {e}")

    def handle_client(self, conn, addr):
        logger.info(f"🔗 Chap nhan ket noi tu: {addr}")
        try:
            # Nhận dữ liệu
            data = conn.recv(4096)
            if not data:
                return

            # Decode & Parse JSON
            message = data.decode('utf-8')
            json_data = json.loads(message)
            
            logger.info(f"📥 Nhan du lieu tu Worker: {json_data}")

            # Gọi callback để xử lý Logic nghiệp vụ (Dự đoán AI)
            # Hàm này được truyền từ main.py vào
            result = self.callback(json_data)

            # Gửi phản hồi lại cho Worker
            response = json.dumps({"status": "success", "prediction": result}).encode('utf-8')
            conn.sendall(response)

        except json.JSONDecodeError:
            logger.error("Dữ liệu không phải JSON hợp lệ")
            conn.sendall(b'{"status": "error", "message": "Invalid JSON"}')
        except Exception as e:
            logger.error(f"Loi xu ly client: {e}")
            conn.sendall(f'{{"status": "error", "message": "{str(e)}"}}'.encode())
        finally:
            conn.close()

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()