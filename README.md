# 🚀 Data Augmentation for Network Optimization

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![React](https://img.shields.io/badge/React-18-61DAFB.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688.svg)
![Socket.IO](https://img.shields.io/badge/Socket.IO-Realtime-black)
![AI Model](https://img.shields.io/badge/AI-RandomForest%20%2B%20GAN-orange)

## 📖 Giới thiệu

Dự án nghiên cứu ứng dụng **Generative AI (GANs)** để sinh dữ liệu giả định (Data Augmentation), nhằm tối ưu hóa hiệu suất mạng SAGSINs. Hệ thống tích hợp mô hình học máy để dự đoán chất lượng mạng (QoS) dựa trên các điều kiện môi trường đa dạng.

Dự án bao gồm một hệ thống hoàn chỉnh: **Client (Dashboard)**, **Server (FastAPI)**, và **Worker (Sensor Simulation)** hoạt động theo kiến trúc Event-Driven.

---

## ✨ Tính năng chính

Hệ thống hỗ trợ 3 chế độ dự đoán thông minh:

### 1. 🤖 Mode 1: Smart Input (Nhập liệu thông minh)
- Người dùng chỉ cần nhập **5 chỉ số cơ bản** (Speed, Battery, Signal Bar, Ping, Throughput).
- Hệ thống AI (**Smart Estimator**) tự động suy luận các thông số kỹ thuật ẩn (SNR, BER, Modulation Scheme...) dựa trên ngữ cảnh thiết bị và hành vi người dùng.

### 2. 🎬 Mode 2: Scenario Simulation (Mô phỏng kịch bản)
- Tích hợp sẵn các kịch bản mạng thực tế tại Việt Nam:
  - 🚌 Di chuyển xe bus/Grab tại TP.HCM.
  - 🏢 Văn phòng cao tầng (Bitexco/Landmark 81).
  - 🛣️ Cao tốc TP.HCM - Long Thành (Tín hiệu kém, Handover cao).
  - 🎉 Sự kiện đông người (Phố đi bộ Nguyễn Huệ).

### 3. ⚡ Mode 3: Real-time Monitoring (Giám sát thời gian thực)
- **Worker** tự động đo đạc thông số máy tính (Pin, Wifi Signal, Speedtest) và gửi về Server.
- **Server** kết hợp dữ liệu đo đạc + ngữ cảnh người dùng để dự đoán chất lượng mạng ngay lập tức.
- Kết quả hiển thị trực quan trên Dashboard mà không cần tải lại trang.

---

## 🛠️ Công nghệ sử dụng

- **Frontend:** ReactJS, TailwindCSS, Lucide Icons.
- **Backend:** Python FastAPI, Socket.IO (WebSockets).
- **AI/ML Core:**
  - **Conditional GAN (CGAN):** Sinh dữ liệu mạng tổng hợp để cân bằng tập dữ liệu.
  - **Random Forest:** Phân loại chất lượng mạng (Good/Moderate/Poor).
  - **Scikit-learn & Pandas:** Xử lý dữ liệu.
- **Worker:** Python `psutil`, `speedtest-cli`, `requests`.

---

## 📂 Cấu trúc dự án

```text
NetworkOptimization_DoAn/
├── client/                 # Giao diện ReactJS (Dashboard)
├── server/                 # Backend FastAPI & Logic AI
│   ├── main.py             # Entry point của Server
│   ├── services/           # SmartEstimator, ScenarioManager
│   └── data/recommendations/ # File cấu hình lời khuyên AI
├── worker/                 # Script đo đạc thông số thực tế
│   └── worker.py
├── utils/                  # Các hàm tiện ích (Model Wrapper, Preprocessor)
├── config/                 # Cấu hình giới hạn thông số mạng (network_ranges.json)
├── models/                 # Chứa Model đã huấn luyện (.pkl)
├── notebooks/              # Jupyter Notebooks (Data Analysis, Training GAN)
├── reports/                # Báo cáo, biểu đồ đánh giá model
├── requirements.txt        # Danh sách thư viện Python
├── run_project.bat         # Script chạy tự động (Windows)
└── README.md               # Tài liệu hướng dẫn

## 🚀 Hướng dẫn Cài đặt và Chạy Demo

Để chạy dự án trên máy tính mới, vui lòng thực hiện theo quy trình sau:

### 📋 1. Yêu cầu hệ thống (Prerequisites)
Trước khi bắt đầu, hãy đảm bảo máy tính đã cài đặt:
* **Python (v3.8 trở lên)**: Đã cài đặt và thêm vào biến môi trường (PATH).
* **Node.js (v14 trở lên)**: Để chạy giao diện Dashboard (Client).

### 📦 2. Cài đặt thư viện (Chỉ cần làm 1 lần đầu)
Mở **Command Prompt (CMD)** hoặc Terminal tại thư mục gốc của dự án (`NetworkOptimization_DoAn`) và chạy lệnh:

```bash
pip install -r requirements.txt