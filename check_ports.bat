@echo off
echo 🔍 포트 상태 확인 스크립트
echo.

echo 📊 메인 대시보드 (포트 8511):
curl -s -o nul -w "HTTP 상태: %%{http_code}" http://localhost:8511
echo.

echo 🏆 통합 시스템 (포트 8550):
curl -s -o nul -w "HTTP 상태: %%{http_code}" http://localhost:8550
echo.

echo 🕷️ 웹 크롤러 (포트 8502):
curl -s -o nul -w "HTTP 상태: %%{http_code}" http://localhost:8502
echo.

echo 💎 보석 분석 (포트 8503):
curl -s -o nul -w "HTTP 상태: %%{http_code}" http://localhost:8503
echo.

echo 🏗️ 3D CAD 변환 (포트 8504):
curl -s -o nul -w "HTTP 상태: %%{http_code}" http://localhost:8504
echo.

echo.
echo 💡 참고: HTTP 200 = 정상, 000 = 중지됨
echo.
pause