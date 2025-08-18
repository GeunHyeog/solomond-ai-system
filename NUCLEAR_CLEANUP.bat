@echo off
echo ⚠️  NUCLEAR CLEANUP - 모든 Streamlit 프로세스 완전 제거
echo 이 스크립트는 시스템의 모든 Python/Streamlit 프로세스를 강제 종료합니다.
echo.
pause

echo 🚀 1단계: 모든 Python 프로세스 강제 종료 (서브프로세스 포함)...
wmic process where "name='python.exe'" delete
timeout /t 3 /nobreak >nul

echo 🔥 2단계: Streamlit 프로세스 강제 종료...
wmic process where "name='streamlit.exe'" delete
timeout /t 2 /nobreak >nul

echo 💀 3단계: 포트 점유 프로세스 개별 제거...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8540 "') do (
    echo 포트 8540 프로세스 %%a 종료 중...
    taskkill /f /pid %%a 2>nul
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8541 "') do (
    echo 포트 8541 프로세스 %%a 종료 중...
    taskkill /f /pid %%a 2>nul
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8542 "') do (
    echo 포트 8542 프로세스 %%a 종료 중...
    taskkill /f /pid %%a 2>nul
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8544 "') do (
    echo 포트 8544 프로세스 %%a 종료 중...
    taskkill /f /pid %%a 2>nul
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8545 "') do (
    echo 포트 8545 프로세스 %%a 종료 중...
    taskkill /f /pid %%a 2>nul
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8546 "') do (
    echo 포트 8546 프로세스 %%a 종료 중...
    taskkill /f /pid %%a 2>nul
)

echo 🧹 4단계: Streamlit 캐시 정리...
if exist "%USERPROFILE%\.streamlit" (
    rd /s /q "%USERPROFILE%\.streamlit\cache" 2>nul
    echo Streamlit 캐시 정리 완료
)

echo ⏰ 5단계: 시스템 안정화 대기...
timeout /t 5 /nobreak >nul

echo 🔍 6단계: 최종 상태 확인...
echo.
echo === 포트 상태 확인 ===
echo 8540: & curl -s -o nul -w "%%{http_code}" http://localhost:8540 --connect-timeout 2
echo 8541: & curl -s -o nul -w "%%{http_code}" http://localhost:8541 --connect-timeout 2
echo 8542: & curl -s -o nul -w "%%{http_code}" http://localhost:8542 --connect-timeout 2
echo 8544: & curl -s -o nul -w "%%{http_code}" http://localhost:8544 --connect-timeout 2
echo 8545: & curl -s -o nul -w "%%{http_code}" http://localhost:8545 --connect-timeout 2
echo 8546: & curl -s -o nul -w "%%{http_code}" http://localhost:8546 --connect-timeout 2
echo 8550: & curl -s -o nul -w "%%{http_code}" http://localhost:8550 --connect-timeout 2
echo.

echo === 실행 중인 Python 프로세스 ===
tasklist | findstr python.exe | wc -l
echo.

echo ✅ NUCLEAR CLEANUP 완료!
echo 💡 이제 restart_system.bat를 실행하여 깨끗한 시스템을 시작하세요.
echo.
pause