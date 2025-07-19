@echo off
echo 🖥️ 솔로몬드 AI - 윈도우 데모 모니터
echo ============================================
echo.

REM 현재 디렉토리를 WSL 경로로 설정
cd /d \\wsl$\Ubuntu\home\solomond\claude\solomond-ai-system

REM Python 의존성 확인 및 설치
echo 📦 의존성 확인 중...
python -c "import psutil, pyautogui, requests" 2>nul
if errorlevel 1 (
    echo ❌ 필요한 패키지가 설치되지 않았습니다.
    echo 📥 의존성 설치 중...
    pip install psutil pyautogui requests pillow
    echo.
)

REM 윈도우 모니터 실행
echo 🚀 윈도우 데모 모니터 시작...
echo 💡 브라우저에서 http://localhost:8503 접속 후 시연하세요!
echo.
python windows_demo_monitor.py

REM 완료 후 잠시 대기
echo.
echo ✅ 모니터링 완료! 
echo 📁 결과는 windows_captures 폴더에 저장되었습니다.
pause