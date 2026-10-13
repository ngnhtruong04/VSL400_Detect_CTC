@echo off
title VSL Realtime Recognition - Backend Server
echo ======================================================================
echo STARTING VSL BACKEND SERVER (FastAPI + PyTorch TCN-Transformer)
echo ======================================================================

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Please ensure the project environment is set up.
    pause
    exit /b 1
)

echo Starting Uvicorn server on http://localhost:8000 ...
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

pause

