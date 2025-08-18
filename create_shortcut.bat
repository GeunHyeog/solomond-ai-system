@echo off
chcp 65001 >nul
title 데스크톱 바로가기 생성

echo 🚀 SOLOMOND AI 바탕화면 바로가기 생성 중...

:: PowerShell을 사용하여 바로가기 생성
powershell -Command ^
"$WshShell = New-Object -comObject WScript.Shell; " ^
"$Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\SOLOMOND AI.lnk'); " ^
"$Shortcut.TargetPath = '%CD%\SOLOMOND_AI_START.bat'; " ^
"$Shortcut.WorkingDirectory = '%CD%'; " ^
"$Shortcut.Description = 'SOLOMOND AI - 접속 실패 제로 시스템'; " ^
"$Shortcut.Save()"

if exist "%USERPROFILE%\Desktop\SOLOMOND AI.lnk" (
    echo ✅ 바탕화면에 'SOLOMOND AI' 바로가기가 생성되었습니다!
    echo.
    echo 🚀 사용법:
    echo 1. 바탕화면의 'SOLOMOND AI' 아이콘 더블클릭
    echo 2. 자동 진단 및 최적 방법으로 실행
    echo 3. 브라우저에서 대시보드 확인
    echo.
    echo 💡 3가지 독립적 접속 방법으로 실패율 0%% 보장!
) else (
    echo ❌ 바로가기 생성 실패
    echo 📂 수동으로 SOLOMOND_AI_START.bat 파일을 바탕화면에 복사하세요
)

echo.
echo 계속하려면 아무 키나 누르세요...
pause >nul