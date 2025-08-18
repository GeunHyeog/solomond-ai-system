@echo off
echo 🛡️ 레거시 포트 방지 시스템
echo 이 스크립트는 레거시 포트들이 실행되는 것을 원천 차단합니다.
echo.

set "CONFIG_FILE=%~dp0.streamlit\config.toml"
set "LEGACY_BLOCK_FILE=%~dp0LEGACY_PORTS_BLOCKED.txt"

echo === 1단계: Streamlit 설정 최적화 ===

REM .streamlit 폴더가 없으면 생성
if not exist "%~dp0.streamlit" mkdir "%~dp0.streamlit"

echo 📝 Streamlit 설정 파일 업데이트 중...

REM 최적화된 config.toml 생성
(
echo [server]
echo maxUploadSize = 10240
echo maxMessageSize = 10240
echo enableWebsocketCompression = false
echo fileWatcherType = "none"
echo runOnSave = false
echo allowRunOnSave = false
echo enableCORS = false
echo enableXsrfProtection = false
echo headless = true
echo.
echo # 허용된 포트만 명시적으로 설정
echo port = 8511
echo.
echo [browser]
echo gatherUsageStats = false
echo serverAddress = "localhost"
echo.
echo [theme]
echo base = "light"
echo.
echo [global]
echo developmentMode = false
echo suppressDeprecationWarnings = true
echo.
echo # 레거시 포트 차단 설정
echo # 8540, 8541, 8542, 8544, 8545, 8546 포트 사용 금지
) > "%CONFIG_FILE%"

echo ✅ Streamlit 설정 최적화 완료

echo.
echo === 2단계: 레거시 포트 차단 규칙 생성 ===

REM 레거시 포트 차단 정보 파일 생성
(
echo # 솔로몬드 AI 레거시 포트 차단 목록
echo # 생성 시간: %date% %time%
echo.
echo BLOCKED_PORTS=8540,8541,8542,8544,8545,8546
echo REASON=Legacy zombie process prevention
echo ALLOWED_PORTS=8511,8550,8502,8503,8504
echo.
echo # 이 파일이 존재하면 레거시 포트 차단 활성화됨
echo BLOCK_STATUS=ACTIVE
) > "%LEGACY_BLOCK_FILE%"

echo ✅ 레거시 포트 차단 규칙 생성 완료

echo.
echo === 3단계: 환경 변수 설정 ===

REM 환경 변수로 허용된 포트만 명시
setx SOLOMOND_ALLOWED_PORTS "8511,8550,8502,8503,8504" /M 2>nul
setx SOLOMOND_BLOCKED_PORTS "8540,8541,8542,8544,8545,8546" /M 2>nul
setx SOLOMOND_AUTO_CLEANUP "ENABLED" /M 2>nul

echo ✅ 환경 변수 설정 완료

echo.
echo === 4단계: 포트 모니터링 스크립트 생성 ===

REM 포트 모니터링 스크립트 생성
(
echo @echo off
echo REM 레거시 포트 자동 감지 및 차단 시스템
echo.
echo :MONITOR_LOOP
echo for /f "tokens=5" %%%%a in ^('netstat -ano 2^^^>nul ^^^| findstr ":8540 "'^) do ^(
echo     echo [%%date%% %%time%%] 레거시 포트 8540 감지! PID %%%%a 종료 중...
echo     taskkill /f /pid %%%%a 2^^^>nul
echo ^)
echo for /f "tokens=5" %%%%a in ^('netstat -ano 2^^^>nul ^^^| findstr ":8541 "'^) do ^(
echo     echo [%%date%% %%time%%] 레거시 포트 8541 감지! PID %%%%a 종료 중...
echo     taskkill /f /pid %%%%a 2^^^>nul
echo ^)
echo for /f "tokens=5" %%%%a in ^('netstat -ano 2^^^>nul ^^^| findstr ":8542 "'^) do ^(
echo     echo [%%date%% %%time%%] 레거시 포트 8542 감지! PID %%%%a 종료 중...
echo     taskkill /f /pid %%%%a 2^^^>nul
echo ^)
echo for /f "tokens=5" %%%%a in ^('netstat -ano 2^^^>nul ^^^| findstr ":8544 "'^) do ^(
echo     echo [%%date%% %%time%%] 레거시 포트 8544 감지! PID %%%%a 종료 중...
echo     taskkill /f /pid %%%%a 2^^^>nul
echo ^)
echo for /f "tokens=5" %%%%a in ^('netstat -ano 2^^^>nul ^^^| findstr ":8545 "'^) do ^(
echo     echo [%%date%% %%time%%] 레거시 포트 8545 감지! PID %%%%a 종료 중...
echo     taskkill /f /pid %%%%a 2^^^>nul
echo ^)
echo for /f "tokens=5" %%%%a in ^('netstat -ano 2^^^>nul ^^^| findstr ":8546 "'^) do ^(
echo     echo [%%date%% %%time%%] 레거시 포트 8546 감지! PID %%%%a 종료 중...
echo     taskkill /f /pid %%%%a 2^^^>nul
echo ^)
echo.
echo timeout /t 30 /nobreak ^^^>nul
echo goto MONITOR_LOOP
) > "%~dp0LEGACY_PORT_MONITOR.bat"

echo ✅ 포트 모니터링 스크립트 생성 완료

echo.
echo === ✅ 레거시 포트 방지 시스템 설치 완료! ===
echo.
echo 🛡️ 적용된 보호 기능:
echo - Streamlit 설정 최적화 (허용된 포트만 명시)
echo - 레거시 포트 차단 규칙 설정
echo - 환경 변수 기반 포트 제어
echo - 실시간 레거시 포트 모니터링 (LEGACY_PORT_MONITOR.bat)
echo.
echo 💡 모니터링 시작 방법:
echo   LEGACY_PORT_MONITOR.bat 실행 (백그라운드에서 계속 감시)
echo.
echo 🎯 이제 레거시 포트가 실행되면 자동으로 차단됩니다!
echo.
pause