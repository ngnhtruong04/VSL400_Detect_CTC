@echo off
title VSL Realtime Recognition System - Launcher
echo ======================================================================
echo LAUNCHING COMPLETE VSL REALTIME RECOGNITION SYSTEM
echo ======================================================================

cd /d "%~dp0"

echo [1/2] Launching Backend Server in new window...
start "VSL Backend" cmd /k "start_backend.bat"

echo [2/2] Launching Frontend App in new window...
start "VSL Frontend" cmd /k "start_frontend.bat"

echo.
echo Waiting 3 seconds for servers to initialize...
timeout /t 3 /nobreak >nul

echo Opening browser at http://localhost:5173 ...
start http://localhost:5173

echo ======================================================================
echo SYSTEM LAUNCHED SUCCESSFULLY!
echo Backend API docs: http://localhost:8000/docs
echo Frontend Web UI:  http://localhost:5173
echo ======================================================================

