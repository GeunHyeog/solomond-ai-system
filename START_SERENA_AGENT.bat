@echo off
echo 🧠 SOLOMOND AI Serena 코딩 에이전트 시작
echo ================================================

cd /d "%~dp0"

echo 📋 1단계: Serena 통합 테스트 실행...
python test_serena_integration.py
if errorlevel 1 (
    echo ❌ 테스트 실패. 문제를 확인해주세요.
    pause
    exit /b 1
)

echo ✅ 테스트 통과!
echo.

echo 🔧 2단계: ThreadPool 이슈 자동 수정...
python serena_auto_fixer.py
echo.

echo 🎯 3단계: Serena 대시보드 시작...
echo 브라우저에서 http://localhost:8520 으로 접속하세요.
echo.

streamlit run solomond_serena_dashboard.py --server.port 8520 --server.headless true

pause