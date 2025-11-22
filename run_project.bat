@echo off
title Network Optimization Project Launcher

echo ========================================================
echo   DANG KHOI DONG HE THONG NETWORK OPTIMIZATION...
echo ========================================================

:: 1. Cài đặt thư viện Python (Chỉ chạy nếu cần, bỏ comment nếu muốn tự động cài)
:: echo Dang cai dat thu vien Python...
:: pip install -r requirements.txt

:: 2. Khởi chạy SERVER
echo [1/3] Dang khoi dong Server (FastAPI)...
start "SERVER - Network Optimization" cmd /k "cd server && python main.py"

:: Chờ 5 giây để Server kịp khởi động trước khi bật Client/Worker
timeout /t 5 /nobreak >nul

:: 3. Khởi chạy CLIENT
echo [2/3] Dang khoi dong Client (ReactJS)...
:: Lưu ý: Máy thầy cần có Node.js. Lệnh này sẽ tự mở trình duyệt.
start "CLIENT - Dashboard" cmd /k "cd client && npm install && npm run dev"

:: 4. Khởi chạy WORKER
echo [3/3] Dang khoi dong Worker (Sensor)...
start "WORKER - Data Collector" cmd /k "cd worker && python worker.py"

echo ========================================================
echo   HE THONG DA DUOC KHOI DONG THANH CONG!
echo   Hay kiem tra cac cua so CMD va Trinh duyet.
echo ========================================================
pause