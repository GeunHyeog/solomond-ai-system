@echo off
chcp 65001 > nul
title SOLOMOND AI - 안정적 시스템 시작

echo.
echo 🎯 SOLOMOND AI - 안정적 통합 시스템
echo ════════════════════════════════════════
echo.
echo 📋 시스템 상태 확인 중...

REM 현재 디렉토리 이동
cd /d "C:\Users\PC_58410\solomond-ai-system"

echo ✅ 작업 디렉토리: %CD%
echo.

REM 기존 Streamlit 프로세스 정리
echo 🧹 기존 프로세스 정리 중...
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im streamlit.exe >nul 2>&1

timeout /t 2 /nobreak >nul

REM Python 환경 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python이 설치되지 않았습니다.
    pause
    exit /b 1
)

echo ✅ Python 환경 확인 완료
echo.

REM 필수 패키지 설치 확인
echo 📦 필수 패키지 확인 중...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo 📦 FastAPI 설치 중...
    pip install fastapi uvicorn
)

pip show torch >nul 2>&1
if errorlevel 1 (
    echo 📦 PyTorch 설치 중...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

echo ✅ 패키지 확인 완료
echo.

REM 백엔드 서버 시작 (백그라운드)
echo 🚀 백엔드 서버 시작 중...
start "SOLOMOND Backend" /min python stable_backend.py

REM 서버 시작 대기
echo ⏰ 서버 시작 대기 중 (5초)...
timeout /t 5 /nobreak >nul

REM HTML 프론트엔드 열기
echo 🌐 프론트엔드 시스템 시작 중...
start "" "SOLOMOND_AI_STABLE_SYSTEM.html"

echo.
echo ✅ SOLOMOND AI 안정적 시스템이 시작되었습니다!
echo.
echo 📋 시스템 정보:
echo    🌐 프론트엔드: SOLOMOND_AI_STABLE_SYSTEM.html
echo    🔧 백엔드 API: http://localhost:8080
echo    📊 상태 확인: http://localhost:8080/health
echo.
echo 💡 사용법:
echo    1. 웹 페이지에서 파일을 드래그앤드롭
echo    2. "통합 분석 시작" 또는 "듀얼 브레인 분석" 클릭
echo    3. 결과 확인 및 다운로드
echo.
echo 🔄 시스템을 종료하려면 이 창을 닫으세요.
echo.

REM 백그라운드에서 실행 유지
:loop
timeout /t 30 /nobreak >nul
goto loop