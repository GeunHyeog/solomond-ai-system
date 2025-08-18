@echo off
echo 🚀 솔로몬드 AI 자동 시작 시스템
echo 이 스크립트는 PC 재시작 후 항상 깨끗한 상태로 시스템을 시작합니다.
echo.

echo === 1단계: 레거시 프로세스 완전 정리 ===
echo 이전 세션의 좀비 프로세스들을 모두 제거합니다...

REM 모든 Python 프로세스 강제 종료
wmic process where "name='python.exe'" delete 2>nul

REM Streamlit 프로세스 강제 종료  
wmic process where "name='streamlit.exe'" delete 2>nul

REM 레거시 포트들 개별 정리
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8540 "') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8541 "') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8542 "') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8544 "') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8545 "') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8546 "') do taskkill /f /pid %%a 2>nul

echo ✅ 레거시 프로세스 정리 완료

echo.
echo === 2단계: 시스템 안정화 대기 ===
timeout /t 3 /nobreak >nul

echo.
echo === 3단계: 정확한 시스템만 시작 ===

echo 🎯 메인 대시보드 시작 중... (포트 8511)
start "SOLOMOND 메인 대시보드" cmd /k "cd /d "%~dp0" && title SOLOMOND 메인 대시보드 && streamlit run solomond_ai_main_dashboard.py --server.port 8511 --server.headless true"

echo ⏰ 메인 대시보드 로딩 대기 중...
timeout /t 8 /nobreak >nul

echo 🏆 통합 컨퍼런스 분석 시작 중... (포트 8550)  
start "SOLOMOND 통합 시스템" cmd /k "cd /d "%~dp0" && title SOLOMOND 통합 시스템 && streamlit run modules/module1_conference/conference_analysis_unified.py --server.port 8550 --server.headless true"

echo ⏰ 통합 시스템 로딩 대기 중...
timeout /t 8 /nobreak >nul

echo.
echo === 4단계: 시스템 상태 검증 ===
echo 목표 시스템 상태 확인 중...

echo 메인 대시보드 (8511): & curl -s -o nul -w "%%{http_code}" http://localhost:8511 --connect-timeout 5
echo 통합 시스템 (8550): & curl -s -o nul -w "%%{http_code}" http://localhost:8550 --connect-timeout 5

echo.
echo 레거시 포트 정리 확인 중... (모두 000이어야 함)
echo 레거시 8542: & curl -s -o nul -w "%%{http_code}" http://localhost:8542 --connect-timeout 2
echo 레거시 8544: & curl -s -o nul -w "%%{http_code}" http://localhost:8544 --connect-timeout 2
echo 레거시 8545: & curl -s -o nul -w "%%{http_code}" http://localhost:8545 --connect-timeout 2

echo.
echo === ✅ 솔로몬드 AI 자동 시작 완료! ===
echo.
echo 🎯 사용 방법:
echo 1. 웹브라우저에서 http://localhost:8511 접속
echo 2. "통합 컨퍼런스 분석 🏆" 클릭  
echo 3. http://localhost:8550 자동 연결
echo 4. 모드 선택 후 분석 시작!
echo.
echo 💡 이 창은 시스템 모니터링용이므로 닫지 마세요.
echo    시스템 종료 시에만 이 창을 닫으시기 바랍니다.
echo.

REM 시스템 상태를 주기적으로 체크
:MONITOR_LOOP
timeout /t 300 /nobreak >nul
echo [%date% %time%] 시스템 상태 체크...
curl -s -o nul -w "메인(8511): %%{http_code} " http://localhost:8511 --connect-timeout 3
curl -s -o nul -w "통합(8550): %%{http_code}" http://localhost:8550 --connect-timeout 3
echo.
goto MONITOR_LOOP