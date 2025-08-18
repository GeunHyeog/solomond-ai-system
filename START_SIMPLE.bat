@echo off
title SOLOMOND AI Start

echo Starting SOLOMOND AI...
echo.

cd /d "C:\Users\PC_58410\solomond-ai-system"

echo [1] Trying dashboard.html...
if exist "dashboard.html" (
    start "" "dashboard.html"
    echo SUCCESS: dashboard.html opened
    goto end
)

echo [2] Trying simple_dashboard.html...  
if exist "simple_dashboard.html" (
    start "" "simple_dashboard.html"
    echo SUCCESS: simple_dashboard.html opened
    goto end
)

echo [3] Starting local server...
python --version >nul 2>&1
if not errorlevel 1 (
    echo Python found, starting server...
    start /b python -m http.server 8899
    timeout /t 3 /nobreak >nul
    start "" "http://localhost:8899"
    echo SUCCESS: Local server started at port 8899
    goto end
)

echo ERROR: All methods failed
echo Please manually open dashboard.html

:end
echo.
echo SOLOMOND AI is now running!
pause