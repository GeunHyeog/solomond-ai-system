@echo off
echo =================================
echo 🚨 응급 시스템 자동 시작 🚨
echo =================================
cd /d "C:\Users\PC_58410\solomond-ai-system"

echo [1단계] 기존 프로세스 완전 종료...
taskkill /f /im streamlit.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
echo 완료!

echo [2단계] 메인 대시보드 시작... (포트 8511)
start "Main-Dashboard" cmd /k "streamlit run solomond_ai_main_dashboard.py --server.port 8511"
ping -n 8 127.0.0.1 >nul

echo [3단계] Module1 컨퍼런스 분석 시작... (포트 8501)
start "Module1-Conference" cmd /k "streamlit run modules\module1_conference\conference_analysis.py --server.port 8501"
ping -n 8 127.0.0.1 >nul

echo [4단계] 자동 헬스체크 시작...
:healthcheck
echo 헬스체크 중...
curl -s -I http://localhost:8511 | findstr "200 OK" >nul
if %errorlevel%==0 (
    echo ✅ 메인 대시보드 정상!
    curl -s -I http://localhost:8501 | findstr "200 OK" >nul
    if %errorlevel%==0 (
        echo ✅ Module1 정상!
        goto success
    )
)
echo ⏳ 시작 중... 5초 후 재확인
ping -n 6 127.0.0.1 >nul
goto healthcheck

:success
echo =================================
echo ✅ 시스템 시작 완료!
echo - 메인 대시보드: http://localhost:8511
echo - Module1 분석: http://localhost:8501
echo =================================
start "" "http://localhost:8511"
start "" "http://localhost:8501"
pause