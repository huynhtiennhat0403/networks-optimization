@echo off
title Network Optimization Project Launcher - Student ID: [Mã_SV_Của_Bạn]

:: Chuyen huong ve thu muc chua file .bat de dam bao duong dan luon dung
cd /d "%~dp0"

echo ========================================================
echo   HE THONG DU DOAN CHAT LUONG MANG (NETWORK OPTIMIZATION)
echo   Sinh vien thuc hien: Huynh Tien Nhat
echo ========================================================

:: 1. KIEM TRA VA CAI DAT THU VIEN PYTHON
echo [1/4] Kiem tra thu vien Python...
python -c "import fastapi" 2>nul
if %errorlevel% neq 0 (
    echo     Phat hien thieu thu vien. Dang cai dat tu requirements.txt...
    pip install -r requirements.txt
) else (
    echo     Thu vien Python da san sang.
)

:: 2. KHOI CHAY SERVER (BACKEND)
echo.
echo [2/4] Dang khoi dong Server (FastAPI)...
:: Da bo ngoac don trong title de tranh loi syntax
start "SERVER - Backend - Port 8000" cmd /k "python -m uvicorn server.main:socket_app --host 0.0.0.0 --port 8000 --reload"

:: Cho 5 giay de Server on dinh
timeout /t 5 /nobreak >nul

:: 3. KHOI CHAY WORKER (DATA COLLECTOR)
echo.
echo [3/4] Dang khoi dong Worker (Sensor)...
start "WORKER - Data Collector" cmd /k "cd worker && python worker.py"

:: 4. KHOI CHAY CLIENT (FRONTEND)
echo.
echo [4/4] Dang khoi dong Client (ReactJS)...
echo     Luu y: Buoc nay can Node.js.
echo.

:: Kiem tra node_modules
if not exist "client\node_modules" (
    echo     Lan dau chay - Dang cai dat dependencies...
    start "CLIENT - Dashboard - Port 5173" cmd /k "cd client && npm install && npm run dev"
) else (
    echo     Da tim thay node_modules - Dang khoi dong...
    start "CLIENT - Dashboard - Port 5173" cmd /k "cd client && npm run dev"
)

echo.
echo ========================================================
echo   HE THONG DA DUOC KHOI DONG!
echo   Hay truy cap: http://localhost:5173
echo ========================================================
pause