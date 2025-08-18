@echo off
echo ============================================
echo 솔로몬드 AI 시스템 자동 재시작 스크립트
echo ============================================
echo.

REM 프로젝트 디렉토리로 이동
cd /d "C:\Users\PC_58410\solomond-ai-system"
echo 현재 디렉토리: %CD%
echo.

REM Git 상태 확인
echo [1/4] Git 상태 확인...
git status --short
echo 최신 커밋:
git log --oneline -1
echo.

REM Python/Streamlit 환경 확인
echo [2/4] Python 환경 확인...
python --version
echo.

REM 기존 Streamlit 프로세스 종료
echo [3/4] 기존 Streamlit 프로세스 정리...
taskkill /F /IM python.exe /T 2>nul
timeout /t 2 /nobreak >nul
echo.

REM Streamlit 앱 시작
echo [4/4] Streamlit 앱 시작...
echo 브라우저에서 http://localhost:8503 접속하세요
echo Ctrl+C로 종료할 수 있습니다
echo.
start http://localhost:8503
python -m streamlit run jewelry_stt_ui_v23_real.py --server.port 8503

pause