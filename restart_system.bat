@echo off
echo 🔄 솔로몬드 AI 시스템 재시작 스크립트
echo.

echo 1. 기존 Streamlit 프로세스 종료 중...
taskkill /f /im python.exe /fi "WINDOWTITLE eq Streamlit*" 2>nul
timeout /t 2 /nobreak >nul

echo 2. 메인 대시보드 시작 중... (포트 8511)
start "메인 대시보드" cmd /k "cd /d %~dp0 && streamlit run solomond_ai_main_dashboard.py --server.port 8511"
timeout /t 3 /nobreak >nul

echo 3. 통합 컨퍼런스 분석 시스템 시작 중... (포트 8550)
start "통합 시스템" cmd /k "cd /d %~dp0 && streamlit run modules/module1_conference/conference_analysis_unified.py --server.port 8550"
timeout /t 3 /nobreak >nul

echo.
echo ✅ 시스템 재시작 완료!
echo 📊 메인 대시보드: http://localhost:8511
echo 🏆 통합 시스템: http://localhost:8550
echo.
pause