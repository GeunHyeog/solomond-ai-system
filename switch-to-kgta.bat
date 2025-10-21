@echo off
echo.
echo ========================================
echo   KGTA 주얼리마스터 프로젝트로 전환
echo ========================================
echo.

cd /d "C:\Users\PC_58410\KGTA-System"

echo ✅ 작업 디렉토리 변경: %CD%
echo.

if exist "KGTA-System.code-workspace" (
    echo 🚀 VSCode 워크스페이스 열기...
    code "KGTA-System.code-workspace"
) else (
    echo 🚀 VSCode 폴더 열기...
    code .
)

echo.
echo 💡 KGTA 시스템 환경 준비 완료!
echo    Claude Code에서 "주얼리마스터 이어가자" 라고 말해보세요.
echo.
pause