@echo off
title VSL Realtime Recognition - Frontend Web App
echo ======================================================================
echo STARTING VSL FRONTEND CLIENT (React + Vite + Tailwind CSS)
echo ======================================================================

cd /d "%~dp0frontend"

if not exist "node_modules" (
    echo [INFO] Installing npm dependencies...
    call npm install
)

echo Starting Vite development server on http://localhost:5173 ...
npm run dev

pause

