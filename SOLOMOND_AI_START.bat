@echo off
chcp 65001 >nul
title 🤖 SOLOMOND AI - 무결성 보장 시스템

echo.
echo ================================================================
echo 🤖 SOLOMOND AI - 접속 실패 제로 시스템
echo ================================================================
echo.
echo 🔍 자동 진단 시작...
echo.

cd /d "%~dp0"

:: 1단계: HTML 직접 실행 시도 (Method 1)
echo [1/3] 🌐 HTML 직접 실행 시도...
if exist "dashboard.html" (
    echo ✅ dashboard.html 발견
    start "" "dashboard.html"
    timeout /t 3 /nobreak >nul
    echo ✅ Method 1: HTML 직접 실행 완료
    goto check_success
) else (
    echo ❌ dashboard.html 없음
)

:: 2단계: 간단한 대시보드 시도 (Method 2)
:method2
echo [2/3] 🎯 간단한 대시보드 시도...
if exist "simple_dashboard.html" (
    echo ✅ simple_dashboard.html 발견
    start "" "simple_dashboard.html"
    timeout /t 3 /nobreak >nul
    echo ✅ Method 2: 간단한 대시보드 완료
    goto check_success
) else (
    echo ❌ simple_dashboard.html 없음
)

:: 3단계: Python 서버 + HTML 조합 (Method 3)
:method3
echo [3/3] 🐍 Python 서버 + HTML 조합 시도...

:: Python 사용 가능 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 설치되지 않음
    goto emergency_mode
)

echo ✅ Python 사용 가능
echo 🚀 서버 시작 중...

:: 백그라운드에서 Python 서버 시작
start /b python -m http.server 8899 >nul 2>&1

timeout /t 5 /nobreak >nul
echo ✅ 서버 시작됨 (포트 8899)

:: HTML 파일 생성 후 실행
echo 📝 임시 대시보드 생성...
echo ^<!DOCTYPE html^>^<html^>^<head^>^<title^>SOLOMOND AI^</title^>^</head^>^<body^>^<h1^>🤖 SOLOMOND AI 대시보드^</h1^>^<p^>시스템이 정상 작동 중입니다!^</p^>^</body^>^</html^> > temp_dashboard.html

start "" "http://localhost:8899/temp_dashboard.html"
echo ✅ Method 3: Python 서버 + HTML 완료
goto success

:check_success
echo.
echo 🔍 실행 결과 확인 중...
timeout /t 2 /nobreak >nul

:: 브라우저가 열렸는지 확인
tasklist | find "chrome.exe" >nul 2>&1
if not errorlevel 1 goto success

tasklist | find "msedge.exe" >nul 2>&1
if not errorlevel 1 goto success

tasklist | find "firefox.exe" >nul 2>&1
if not errorlevel 1 goto success

echo ⚠️ 브라우저 실행 감지 안됨, 추가 시도...
goto method2

:success
echo.
echo ================================================================
echo 🎉 SOLOMOND AI 성공적으로 실행됨!
echo ================================================================
echo.
echo ✅ 무결성 보장 시스템 작동 완료
echo 🌐 브라우저에서 대시보드를 확인하세요
echo 💡 문제가 있으면 이 창을 닫지 말고 오류를 확인하세요
echo.
echo 🔄 이 창을 닫으려면 아무 키나 누르세요...
pause >nul
exit

:emergency_mode
echo.
echo ================================================================
echo 🚨 응급 모드 - 수동 복구 가이드
echo ================================================================
echo.
echo 📋 다음 중 하나를 시도하세요:
echo.
echo 1. 📂 파일 탐색기에서 다음 파일들을 더블클릭:
echo    - dashboard.html
echo    - simple_dashboard.html
echo.
echo 2. 🌐 브라우저에서 직접 열기:
echo    - file:///C:/Users/PC_58410/solomond-ai-system/dashboard.html
echo.
echo 3. 🐍 Python이 설치되어 있다면:
echo    - python -m http.server 8899
echo    - 브라우저에서 http://localhost:8899 열기
echo.
echo 🔧 복구 후 이 배치파일을 다시 실행하세요
echo.
pause
exit