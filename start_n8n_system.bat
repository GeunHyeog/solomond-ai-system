@echo off
chcp 65001 > nul
echo.
echo ========================================
echo  SOLOMOND AI - n8n 자동화 시스템 v2.0
echo ========================================
echo.

echo 🔍 시스템 상태 확인 중...

:: n8n 설치 확인
where n8n >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ n8n이 설치되지 않았습니다.
    echo.
    echo 📥 자동 설치를 시작하시겠습니까? (Y/N)
    set /p "install_choice=선택: "
    if /i "%install_choice%"=="Y" (
        echo 🚀 완전 자동 설치 시작...
        python setup_n8n_automation.py
        echo.
        echo ✅ 자동 설치 완료! 다시 시작합니다...
        timeout /t 5 > nul
        goto :start_n8n
    ) else (
        echo.
        echo 수동 설치 방법:
        echo 1. npm install -g n8n
        echo 2. 이 스크립트 다시 실행
        echo.
        pause
        exit /b 1
    )
)

:start_n8n
echo [1단계] n8n 서버 시작 중...
start "SOLOMOND n8n Server" cmd /k "n8n start"

echo [2단계] 서버 초기화 대기 중... (30초)
timeout /t 30 > nul

echo [3단계] n8n 상태 확인...
curl -s http://localhost:5678/healthz >nul 2>nul
if %errorlevel% equ 0 (
    echo ✅ n8n 서버 정상 작동 중
) else (
    echo ⚠️ n8n 서버 시작 확인 중... 브라우저에서 확인하세요.
)

echo [4단계] 워크플로우 자동 배포...
python -c "import asyncio; from n8n_connector import N8nConnector; asyncio.run(N8nConnector().setup_solomond_workflows())" 2>nul
if %errorlevel% equ 0 (
    echo ✅ SOLOMOND AI 워크플로우 자동 배포 완료
) else (
    echo ⚠️ 워크플로우 수동 설정이 필요할 수 있습니다.
)

echo [5단계] 웹 브라우저에서 n8n 대시보드 열기...
start "" "http://localhost:5678"

echo.
echo 🎉 SOLOMOND AI - n8n 자동화 시스템 시작 완료!
echo.
echo 📋 시스템 정보:
echo    🌐 n8n 대시보드: http://localhost:5678
echo    🎯 SOLOMOND AI 분석: http://localhost:8501
echo    📊 메인 대시보드: http://localhost:8500
echo.
echo 🔗 자동화 워크플로우:
echo    ✅ 컨퍼런스 분석 완료 → 듀얼 브레인 시스템 자동 실행
echo    ✅ AI 인사이트 생성 → Google Calendar 이벤트 자동 생성
echo    ✅ 시스템 모니터링 → 상태 알림 자동 발송
echo.
echo 💡 사용법: 컨퍼런스 분석을 완료하면 자동으로 n8n 워크플로우가 실행됩니다!
echo.
pause