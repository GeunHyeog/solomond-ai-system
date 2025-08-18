@echo off
chcp 65001 >nul

echo.
echo ========================================
echo 🚀 SOLOMOND AI 하이브리드 설치 시작
echo ========================================
echo.

echo 📋 설치 전략:
echo    🚀 uv: 일반 패키지 (39.8배 빠름)
echo    🛡️ pip: AI 패키지 (안정성 보장)
echo.

:: uv 설치 확인
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ uv가 설치되지 않았습니다.
    echo 💡 'winget install astral-sh.uv' 실행 후 다시 시도하세요.
    pause
    exit /b 1
)

:: Phase 1: uv로 고속 설치
echo.
echo 🚀 Phase 1: uv로 일반 패키지 고속 설치 중...
echo.
uv pip install -r requirements-uv-fast.txt
if %errorlevel% neq 0 (
    echo ⚠️ uv 설치 중 일부 패키지 실패, pip로 폴백합니다...
    pip install -r requirements-uv-fast.txt
)

:: Phase 2: pip으로 안정 설치  
echo.
echo 🛡️ Phase 2: pip으로 AI 패키지 안정 설치 중...
echo.
pip install -r requirements-pip-stable.txt

:: 설치 검증
echo.
echo 🔍 설치 검증 중...
echo.
python -c "
try:
    import streamlit, torch, transformers, whisper, easyocr
    print('✅ 모든 핵심 패키지 로드 성공!')
    print('🎯 SOLOMOND AI 시스템 준비 완료!')
except ImportError as e:
    print(f'❌ 패키지 로드 실패: {e}')
    exit(1)
"

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo ✅ 하이브리드 설치 완료!
    echo ========================================
    echo.
    echo 📊 설치된 패키지 확인: uv pip list
    echo 🚀 시스템 시작: streamlit run conference_analysis_WORKING_FIXED.py --server.port 8550
    echo.
) else (
    echo ❌ 설치 검증 실패
    echo 💡 개별 패키지 확인 필요
)

pause