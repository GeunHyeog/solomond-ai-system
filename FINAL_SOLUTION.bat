@echo off
title SOLOMOND AI EMERGENCY RECOVERY
color 0A

echo ========================================
echo    SOLOMOND AI EMERGENCY RECOVERY
echo ========================================
echo.

cd /d "C:\Users\PC_58410\solomond-ai-system"

echo [STEP 1] Terminating all processes...
taskkill /f /im streamlit.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
echo OK - All processes terminated

echo.
echo [STEP 2] Starting Main Dashboard (Port 8511)...
start "Main-Dashboard" cmd /c "streamlit run solomond_ai_main_dashboard.py --server.port 8511"

echo.
echo [STEP 3] Starting Module1 Analysis (Port 8501)...
start "Module1-Analysis" cmd /c "streamlit run modules\module1_conference\conference_analysis.py --server.port 8501"

echo.
echo [STEP 4] Waiting for services to start...
timeout /t 15 /nobreak >nul

echo.
echo [STEP 5] Opening browsers...
start "" "http://localhost:8511"
start "" "http://localhost:8501"

echo.
echo ========================================
echo   RECOVERY COMPLETED! 
echo   - Main Dashboard: http://localhost:8511
echo   - Module1 Analysis: http://localhost:8501
echo ========================================
echo.
echo Press any key to exit...
pause >nul