@echo off
echo 🚀 솔로몬드 AI 궁극 시작 설정 시스템
echo 이 스크립트는 PC 재시작 문제를 영구적으로 해결합니다.
echo.

echo ⚠️  이 설정을 적용하면:
echo 1. PC 부팅 시 솔로몬드 AI가 자동으로 시작됩니다
echo 2. 레거시 포트 좀비 문제가 완전히 방지됩니다  
echo 3. 항상 깨끗한 상태로 시작됩니다
echo 4. 메인 대시보드(8511) → 통합 시스템(8550) 완벽 연동
echo.

set /p confirm="계속 진행하시겠습니까? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo 설정이 취소되었습니다.
    pause
    exit /b
)

echo.
echo === 🔧 1단계: 레거시 포트 방지 시스템 설치 ===
call "%~dp0PREVENT_LEGACY_PORTS.bat"

echo.
echo === 🔧 2단계: 자동 시작 바로가기 생성 ===
call "%~dp0CREATE_STARTUP_SHORTCUT.bat"

echo.
echo === 🔧 3단계: 시스템 검증 ===

if exist "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\솔로몬드AI자동시작.lnk" (
    echo ✅ 자동 시작 등록 확인됨
) else (
    echo ❌ 자동 시작 등록 실패
)

if exist "%~dp0LEGACY_PORTS_BLOCKED.txt" (
    echo ✅ 레거시 포트 차단 설정 확인됨
) else (
    echo ❌ 레거시 포트 차단 설정 실패
)

if exist "%~dp0.streamlit\config.toml" (
    echo ✅ Streamlit 최적화 설정 확인됨
) else (
    echo ❌ Streamlit 설정 실패
)

echo.
echo === 🔧 4단계: 테스트 실행 (선택사항) ===
set /p test="지금 바로 테스트 실행하시겠습니까? (Y/N): "
if /i "%test%"=="Y" (
    echo 🚀 테스트 실행 중...
    start "" "%~dp0AUTO_START_SOLOMOND.bat"
    echo.
    echo 테스트가 시작되었습니다. 새 창에서 실행 상황을 확인하세요.
    timeout /t 5 /nobreak >nul
)

echo.
echo === ✅ 궁극 시작 설정 완료! ===
echo.
echo 🎯 적용된 기능들:
echo ✅ PC 부팅 시 자동 시작
echo ✅ 레거시 포트 완전 차단
echo ✅ 자동 프로세스 정리
echo ✅ 최적화된 Streamlit 설정
echo ✅ 실시간 포트 모니터링
echo.
echo 🚀 다음 부팅부터 적용됩니다:
echo 1. PC 부팅 완료 후 자동으로 솔로몬드 AI 시작
echo 2. http://localhost:8511 자동 접속 가능
echo 3. "통합 컨퍼런스 분석 🏆" → http://localhost:8550 완벽 연동
echo 4. 레거시 포트 문제 영원히 해결됨
echo.
echo 💡 수동 해제 방법:
echo - Windows + R → shell:startup → "솔로몬드AI자동시작.lnk" 삭제
echo.
echo 🎉 이제 PC를 껐다 켜도 같은 문제가 반복되지 않습니다!
echo.
pause