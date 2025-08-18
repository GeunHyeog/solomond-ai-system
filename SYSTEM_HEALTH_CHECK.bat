@echo off
echo 🏥 시스템 건강도 체크 스크립트
echo.

echo === 현재 시스템 상태 진단 ===
echo.

echo 🎯 목표 시스템 (정상이어야 함):
echo 메인 대시보드 (8511): & curl -s -o nul -w "%%{http_code}" http://localhost:8511 --connect-timeout 3
echo 통합 시스템 (8550): & curl -s -o nul -w "%%{http_code}" http://localhost:8550 --connect-timeout 3
echo.

echo ❌ 레거시 시스템 (000이어야 함):
echo 레거시 8540: & curl -s -o nul -w "%%{http_code}" http://localhost:8540 --connect-timeout 2
echo 레거시 8541: & curl -s -o nul -w "%%{http_code}" http://localhost:8541 --connect-timeout 2
echo 레거시 8542: & curl -s -o nul -w "%%{http_code}" http://localhost:8542 --connect-timeout 2
echo 레거시 8544: & curl -s -o nul -w "%%{http_code}" http://localhost:8544 --connect-timeout 2
echo 레거시 8545: & curl -s -o nul -w "%%{http_code}" http://localhost:8545 --connect-timeout 2
echo 레거시 8546: & curl -s -o nul -w "%%{http_code}" http://localhost:8546 --connect-timeout 2
echo.

echo 📊 추가 모듈들:
echo 웹 크롤러 (8502): & curl -s -o nul -w "%%{http_code}" http://localhost:8502 --connect-timeout 2
echo 보석 분석 (8503): & curl -s -o nul -w "%%{http_code}" http://localhost:8503 --connect-timeout 2
echo 3D CAD (8504): & curl -s -o nul -w "%%{http_code}" http://localhost:8504 --connect-timeout 2
echo.

echo 🔧 실행 중인 Python 프로세스:
tasklist | findstr python.exe | find /c "python.exe"
echo.

echo 💡 건강도 판정:
echo - 8511, 8550이 200이면 ✅ 정상
echo - 8542, 8544, 8545가 000이면 ✅ 레거시 정리됨
echo - Python 프로세스가 2-4개면 ✅ 적정 수준
echo.

echo === 권장 조치 ===
echo 🔴 레거시 포트가 200이면: NUCLEAR_CLEANUP.bat 실행
echo 🟡 목표 포트가 000이면: START_CLEAN_SYSTEM.bat 실행
echo 🟢 모든 상태가 정상이면: 시스템 정상 작동 중
echo.
pause