@echo off
title SOLOMOND AI Auto Start
echo Starting SOLOMOND AI System...
echo.
echo [1/3] Starting AI Analysis Server...
cd /d "C:\Users\PC_58410\solomond-ai-system"
start /B python simple_analysis.py
echo.
echo [2/3] Waiting for server to start...
timeout /t 3 /nobreak >nul
echo.
echo [3/3] Opening browser...
start "" "http://localhost:8000"
echo.
echo SOLOMOND AI started successfully!
echo Browser should open automatically
echo If not, go to: http://localhost:8000
echo.
echo Press any key to close this window...
pause >nul
