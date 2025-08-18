@echo off
echo 🔧 윈도우 시작프로그램 등록 스크립트
echo 이 스크립트는 솔로몬드 AI가 PC 부팅 시 자동으로 시작되도록 설정합니다.
echo.

set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SCRIPT_PATH=%~dp0AUTO_START_SOLOMOND.bat"
set "SHORTCUT_PATH=%STARTUP_FOLDER%\솔로몬드AI자동시작.lnk"

echo 📁 시작프로그램 폴더: %STARTUP_FOLDER%
echo 📄 스크립트 경로: %SCRIPT_PATH%
echo 🔗 바로가기 경로: %SHORTCUT_PATH%
echo.

echo ✅ 바로가기 생성 중...

REM PowerShell을 사용해서 바로가기 생성
powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%SCRIPT_PATH%'; $Shortcut.WorkingDirectory = '%~dp0'; $Shortcut.Description = '솔로몬드 AI 자동 시작 시스템'; $Shortcut.Save()"

if exist "%SHORTCUT_PATH%" (
    echo ✅ 시작프로그램 등록 완료!
    echo.
    echo 🎯 등록 결과:
    echo - PC 부팅 시 솔로몬드 AI가 자동으로 시작됩니다
    echo - 항상 깨끗한 상태로 시작됩니다 (레거시 프로세스 자동 정리)
    echo - 메인 대시보드(8511)와 통합 시스템(8550)이 자동 실행됩니다
    echo.
    echo 💡 해제 방법:
    echo - Windows + R → shell:startup → "솔로몬드AI자동시작.lnk" 삭제
    echo.
    echo 🚀 다음 부팅부터 자동으로 솔로몬드 AI가 시작됩니다!
) else (
    echo ❌ 바로가기 생성 실패
    echo 관리자 권한으로 다시 실행해보세요.
)

echo.
pause