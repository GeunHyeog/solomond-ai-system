@echo off
echo ========================================
echo    SOLOMOND AI GitHub Push Ready
echo ========================================
echo.
echo GitHub 서버 복구 후 즉시 실행용 스크립트
echo.

REM 1. 현재 상태 확인
echo [1] 현재 Git 상태 확인...
git status --short
echo.

REM 2. 푸시할 커밋 수 확인
echo [2] 푸시 대기 중인 커밋 수...
git rev-list --count origin/main..HEAD
echo.

REM 3. 최신 커밋 목록 (최근 5개)
echo [3] 최신 커밋 목록...
git log --oneline -5
echo.

REM 4. 원격 저장소 연결 테스트
echo [4] GitHub 연결 테스트...
git ls-remote origin HEAD
if errorlevel 1 (
    echo ERROR: GitHub 연결 실패 - 아직 서버 문제 지속
    pause
    exit /b 1
)
echo SUCCESS: GitHub 연결 정상!
echo.

REM 5. 메인 푸시 실행
echo [5] 메인 브랜치 푸시 시작...
echo 푸시 중... (시간이 걸릴 수 있습니다)
git push origin main

if errorlevel 1 (
    echo.
    echo ERROR: 푸시 실패 - 다시 시도하거나 분할 푸시 고려
    echo.
    echo 대안: SSH 프로토콜 시도
    echo git remote set-url origin git@github.com:GeunHyeog/solomond-ai-system.git
    echo git push origin main
    pause
    exit /b 1
) else (
    echo.
    echo ========================================
    echo     🎉 푸시 성공! 🎉
    echo ========================================
    echo.
    echo GitHub 저장소가 최신 상태로 업데이트되었습니다.
    echo 브라우저에서 확인: https://github.com/GeunHyeog/solomond-ai-system
    echo.
)

pause