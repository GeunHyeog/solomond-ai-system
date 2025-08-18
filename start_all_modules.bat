@echo off
echo 모든 모듈 시작 중...
start cmd /k "streamlit run solomond_ai_main_dashboard.py --server.port 8500"
timeout /t 3 /nobreak >nul
start cmd /k "streamlit run conference_analysis_COMPLETE_WORKING.py --server.port 8501"  
echo 시작 완료! http://localhost:8500 접속하세요
pause
