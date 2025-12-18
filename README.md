# 🚀 Data Augmentation for Network Optimization

Dự án môn học: Lập trình mạng / Đồ án chuyên ngành
**Sinh viên:** Huỳnh Ngọc Tiến Nhật / Mai Thanh Tân
**MSSV:** 102230363 / 102230372
**GitHub:** [https://github.com/huynhtiennhat0403/networks-optimization](https://github.com/huynhtiennhat0403/networks-optimization)

---

## 📖 Tổng quan
Hệ thống dự đoán chất lượng mạng (Quality of Service - QoS) sử dụng AI (Random Forest) và kỹ thuật Data Augmentation (SMOTE/GAN). Hệ thống hoạt động theo mô hình **Client-Server** kết hợp với **Worker** thu thập dữ liệu thời gian thực qua giao thức TCP/Socket.

### Kiến trúc hệ thống:
1.  **Server (Python/FastAPI):** Trung tâm xử lý, chạy mô hình AI, quản lý kết nối WebSocket.
2.  **Client (ReactJS/Vite):** Dashboard hiển thị kết quả và tương tác người dùng.
3.  **Worker (Python):** Sensor mô phỏng, đo đạc thông số mạng máy tính (Ping, Signal, Battery) và gửi về Server.

---

## ⚙️ Yêu cầu cài đặt (Prerequisites)

Để chạy được dự án, máy tính cần cài đặt sẵn:

1.  **Python (3.8 trở lên):** [Tải tại đây](https://www.python.org/downloads/) (Đảm bảo đã tích chọn *"Add Python to PATH"* khi cài đặt).
2.  **Node.js (v16 trở lên):** [Tải tại đây](https://nodejs.org/) (Bắt buộc để chạy giao diện Client).

---

## 🚀 Hướng dẫn chạy dự án (Quick Start)

### Cách 1: Chạy tự động (Khuyên dùng)
Chỉ cần click đúp vào file **`run_project.bat`** ở thư mục gốc. Hệ thống sẽ:
1.  Tự động kiểm tra và cài đặt thư viện Python cần thiết.
2.  Khởi động Server (Port 8000).
3.  Khởi động Worker.
4.  Cài đặt thư viện Node.js (nếu chạy lần đầu) và bật Client.

### Cách 2: Chạy thủ công (Manual)
Nếu file `.bat` không hoạt động, Thầy/Cô vui lòng mở 3 cửa sổ Terminal (CMD/PowerShell) tại thư mục gốc của dự án:

**Terminal 1 - Server:**
```bash
pip install -r requirements.txt
python -m uvicorn server.main:socket_app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Worker:**
```bash
python worker/worker.py
```

**Terminal 3 - Client:**
```bash
cd client
npm install
npm run dev
```

Sau đó truy cập trình duyệt tại: [http://localhost:5173](http://localhost:5173)