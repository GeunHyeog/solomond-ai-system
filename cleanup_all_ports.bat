@echo off
echo 🧹 모든 Streamlit 포트 완전 정리 스크립트
echo.

echo 1단계: 모든 Python 프로세스 강제 종료...
taskkill /f /im python.exe /t 2>nul
timeout /t 2 /nobreak >nul

echo 2단계: Streamlit 관련 프로세스 정리...
taskkill /f /im streamlit.exe /t 2>nul
timeout /t 1 /nobreak >nul

echo 3단계: 포트별 프로세스 강제 종료...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8540') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8541') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8542') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8544') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8545') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8546') do taskkill /f /pid %%a 2>nul

echo 4단계: 잠시 대기...
timeout /t 3 /nobreak >nul

echo 5단계: 포트 상태 확인...
echo 8540: & curl -s -o nul -w "%%{http_code}" http://localhost:8540
echo 8541: & curl -s -o nul -w "%%{http_code}" http://localhost:8541
echo 8542: & curl -s -o nul -w "%%{http_code}" http://localhost:8542
echo 8544: & curl -s -o nul -w "%%{http_code}" http://localhost:8544
echo 8545: & curl -s -o nul -w "%%{http_code}" http://localhost:8545
echo 8546: & curl -s -o nul -w "%%{http_code}" http://localhost:8546
echo 8550: & curl -s -o nul -w "%%{http_code}" http://localhost:8550

echo.
echo ✅ 포트 정리 완료! 이제 새로운 시스템만 시작하세요.
echo 💡 다음 단계: restart_system.bat 실행
echo.
pause