@echo off
chcp 65001 >nul
echo 🏆 ULTIMATE 통합 시스템 시작
echo ========================================
echo.

cd /d "C:\Users\PC_58410\solomond-ai-system"

echo 🔧 시스템 환경 설정 중...

REM Windows 방화벽 포트 8510 허용 (관리자 권한 필요)
echo 📡 방화벽 포트 8510 설정...
netsh advfirewall firewall add rule name="ULTIMATE_System_8510" dir=in action=allow protocol=TCP localport=8510 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ 방화벽 설정 완료
) else (
    echo ⚠️  방화벽 설정 실패 (관리자 권한 필요)
)

echo.
echo 🚀 ULTIMATE 시스템 구성요소:
echo    🏆 5D 멀티모달 분석 엔진
echo    🤖 Ollama AI 통합 (qwen2.5:7b)
echo    ⚡ 터보 업로드 시스템
echo    🧠 ComprehensiveMessageExtractor
echo    💎 주얼리 도메인 특화
echo.

echo 📊 포트 8510에서 ULTIMATE UI 시작...
echo 브라우저에서 http://localhost:8510 을 열어주세요.
echo.
echo 🔄 시스템이 준비되면 자동으로 브라우저가 열립니다.
echo 종료하려면 Ctrl+C를 누르세요.
echo.

REM Streamlit 실행 (네트워크 바인딩 최적화)
python -m streamlit run modules/module1_conference/conference_analysis_ultimate_integrated.py ^
    --server.port=8510 ^
    --server.address=0.0.0.0 ^
    --server.headless=true ^
    --server.runOnSave=false ^
    --browser.gatherUsageStats=false ^
    --server.maxUploadSize=5000

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ 포트 8510 실행 실패! 대체 포트로 시도...
    echo 📊 포트 8512에서 재시도...
    
    python -m streamlit run modules/module1_conference/conference_analysis_ultimate_integrated.py ^
        --server.port=8512 ^
        --server.address=0.0.0.0 ^
        --server.headless=true ^
        --server.runOnSave=false ^
        --browser.gatherUsageStats=false ^
        --server.maxUploadSize=5000
    
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ❌ 포트 8512도 실패! 포트 8513으로 재시도...
        
        python -m streamlit run modules/module1_conference/conference_analysis_ultimate_integrated.py ^
            --server.port=8513 ^
            --server.address=0.0.0.0 ^
            --server.headless=true ^
            --server.runOnSave=false ^
            --browser.gatherUsageStats=false ^
            --server.maxUploadSize=5000
    )
)

echo.
echo 🔄 시스템 종료됨. 아무 키나 누르면 창이 닫힙니다.
pause