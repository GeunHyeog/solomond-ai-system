@echo off
echo 🚀 깨끗한 시스템 시작 스크립트
echo 이 스크립트는 오직 필요한 2개 포트만 시작합니다.
echo.

echo 📋 시작할 서비스:
echo - 메인 대시보드: 포트 8511
echo - 통합 컨퍼런스 분석: 포트 8550
echo.
pause

echo 🔧 1단계: 메인 대시보드 시작 중... (포트 8511)
start "메인 대시보드" cmd /k "cd /d "%~dp0" && echo 메인 대시보드 시작 중... && streamlit run solomond_ai_main_dashboard.py --server.port 8511"

echo ⏰ 대기 중... (메인 대시보드 로딩)
timeout /t 8 /nobreak >nul

echo 🏆 2단계: 통합 컨퍼런스 분석 시스템 시작 중... (포트 8550)
start "통합 시스템" cmd /k "cd /d "%~dp0" && echo 통합 시스템 시작 중... && streamlit run modules/module1_conference/conference_analysis_unified.py --server.port 8550"

echo ⏰ 대기 중... (통합 시스템 로딩)
timeout /t 8 /nobreak >nul

echo 🔍 3단계: 시스템 상태 확인...
echo.
echo === 최종 시스템 상태 ===
echo 메인 대시보드 (8511): & curl -s -o nul -w "%%{http_code}" http://localhost:8511 --connect-timeout 5
echo 통합 시스템 (8550): & curl -s -o nul -w "%%{http_code}" http://localhost:8550 --connect-timeout 5
echo.

echo === 불필요한 포트 확인 (모두 000이어야 함) ===
echo 레거시 8542: & curl -s -o nul -w "%%{http_code}" http://localhost:8542 --connect-timeout 2
echo 레거시 8544: & curl -s -o nul -w "%%{http_code}" http://localhost:8544 --connect-timeout 2
echo 레거시 8545: & curl -s -o nul -w "%%{http_code}" http://localhost:8545 --connect-timeout 2
echo.

echo ✅ 깨끗한 시스템 시작 완료!
echo.
echo 🎯 사용 방법:
echo 1. http://localhost:8511 접속 (메인 대시보드)
echo 2. "통합 컨퍼런스 분석 🏆" 클릭
echo 3. http://localhost:8550 자동 연결 (통합 시스템)
echo 4. 모드 선택 후 분석 시작!
echo.
pause