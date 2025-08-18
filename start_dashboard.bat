@echo off
chcp 65001 >nul
echo 🎯 솔로몬드 AI v3.0 메인 대시보드 시작
echo ====================================

cd /d "C:\Users\PC_58410\solomond-ai-system"

echo 📊 메인 대시보드 실행 중...
echo 브라우저에서 http://localhost:8500 을 열어주세요.
echo.
echo 💡 오류 해결 완료:
echo    - KeyError 'health_score' 수정
echo    - 안전한 세션 상태 관리
echo    - 모든 모듈 정상 작동
echo.
echo 종료하려면 Ctrl+C를 누르세요.
echo.

python -m streamlit run solomond_ai_main_dashboard.py --server.port 8500

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ 실행 오류 발생! 대체 포트로 시도...
    python -m streamlit run solomond_ai_main_dashboard.py --server.port 8510
)

pause