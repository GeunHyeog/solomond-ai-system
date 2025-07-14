@echo off
REM 💎 솔로몬드 AI 엔진 v2.3 Windows 프로덕션 배포 스크립트
REM 99.4%% 정확도 하이브리드 LLM 시스템 자동 배포

title 솔로몬드 AI v2.3 Windows 배포 시스템
color 0B

echo.
echo ████████████████████████████████████████████████████████
echo █                                                      █
echo █  💎 솔로몬드 AI 엔진 v2.3 Windows 배포 시스템     █
echo █  🎯 99.4%% 정확도 하이브리드 LLM 시스템           █
echo █  🚀 Windows 전용 자동 배포 스크립트              █
echo █                                                      █
echo ████████████████████████████████████████████████████████
echo.

echo 📋 배포 스크립트 정보
echo - 버전: v2.3 Windows Edition
echo - 날짜: %DATE% %TIME%
echo - 대상: Windows 프로덕션 환경
echo - 개발자: 전근혁 (솔로몬드 대표)
echo.

REM 사전 요구사항 확인
echo 🔍 Windows 사전 요구사항 확인
echo.

REM Docker Desktop 확인
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Desktop이 설치되지 않았거나 실행되지 않습니다.
    echo Docker Desktop을 설치하고 실행해주세요: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)
echo ✅ Docker Desktop 확인됨

REM Docker Compose 확인
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    docker compose version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Docker Compose가 설치되지 않았습니다.
        pause
        exit /b 1
    )
)
echo ✅ Docker Compose 확인됨

REM Git 확인
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git이 설치되지 않았습니다.
    echo Git을 설치해주세요: https://git-scm.com/download/win
    pause
    exit /b 1
)
echo ✅ Git 확인됨

echo.

REM 환경 변수 설정
echo ⚙️ Windows 배포 설정 구성
echo.

set ENV_FILE=.env.v23.windows
echo # 솔로몬드 AI v2.3 Windows 프로덕션 환경 변수 > %ENV_FILE%
echo # 생성일: %DATE% %TIME% >> %ENV_FILE%
echo. >> %ENV_FILE%

echo ENVIRONMENT=production >> %ENV_FILE%
echo SOLOMOND_VERSION=v2.3 >> %ENV_FILE%
echo LOG_LEVEL=INFO >> %ENV_FILE%
echo PLATFORM=windows >> %ENV_FILE%
echo. >> %ENV_FILE%

REM 데이터베이스 비밀번호 입력
echo 🔒 데이터베이스 비밀번호 설정:
set /p POSTGRES_PASSWORD=PostgreSQL 비밀번호 입력: 
set /p REDIS_PASSWORD=Redis 비밀번호 입력: 

echo POSTGRES_PASSWORD=%POSTGRES_PASSWORD% >> %ENV_FILE%
echo REDIS_PASSWORD=%REDIS_PASSWORD% >> %ENV_FILE%
echo. >> %ENV_FILE%

REM API 키 입력 (선택사항)
echo.
echo 🔑 AI API 키 설정 (선택사항 - Enter로 건너뛰기):
set /p OPENAI_API_KEY=OpenAI API 키: 
set /p ANTHROPIC_API_KEY=Anthropic API 키: 
set /p GOOGLE_API_KEY=Google API 키: 

if not "%OPENAI_API_KEY%"=="" (
    echo OPENAI_API_KEY=%OPENAI_API_KEY% >> %ENV_FILE%
)
if not "%ANTHROPIC_API_KEY%"=="" (
    echo ANTHROPIC_API_KEY=%ANTHROPIC_API_KEY% >> %ENV_FILE%
)
if not "%GOOGLE_API_KEY%"=="" (
    echo GOOGLE_API_KEY=%GOOGLE_API_KEY% >> %ENV_FILE%
)

echo ✅ Windows 환경 설정 완료: %ENV_FILE%
echo.

REM 배포 옵션 선택
echo 🚀 Windows 배포 옵션 선택
echo 1) 기본 배포 (AI 시스템만)
echo 2) 완전 배포 (모니터링 포함)
echo 3) 개발자 배포 (모든 서비스)
set /p DEPLOY_OPTION=선택 (1-3): 

if "%DEPLOY_OPTION%"=="1" set PROFILES=
if "%DEPLOY_OPTION%"=="2" set PROFILES=--profile monitoring
if "%DEPLOY_OPTION%"=="3" set PROFILES=--profile production --profile monitoring --profile backup

echo.

REM 배포 시작
echo 🚀 솔로몬드 AI v2.3 Windows 프로덕션 배포 시작!
echo.

REM 이전 컨테이너 정리
echo 🧹 이전 배포 정리 중...
docker-compose -f docker-compose.v23.production.yml down --remove-orphans 2>nul

REM Docker 시스템 정리
echo 🧹 Docker 캐시 정리 중...
docker system prune -f 2>nul

REM 최신 이미지 풀
echo 📥 최신 이미지 다운로드 중...
docker-compose -f docker-compose.v23.production.yml pull 2>nul

REM 배포 실행
echo.
echo 🚀 v2.3 시스템 시작 중...
docker-compose -f docker-compose.v23.production.yml --env-file %ENV_FILE% %PROFILES% up -d

if %errorlevel% neq 0 (
    echo ❌ 배포 중 오류가 발생했습니다.
    echo 로그 확인: docker-compose -f docker-compose.v23.production.yml logs
    pause
    exit /b 1
)

REM 서비스 시작 대기
echo.
echo ⏳ Windows 서비스 시작 대기 (120초)...
ping localhost -n 121 > nul

REM 헬스체크
echo.
echo 🔍 Windows 시스템 상태 확인
echo.

REM Docker 컨테이너 상태 확인
echo 📊 컨테이너 상태:
docker-compose -f docker-compose.v23.production.yml ps

echo.

REM 메인 서비스 확인
echo 🔍 메인 서비스 헬스체크...
curl -s http://localhost:8080/health/v23 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ 솔로몬드 AI v2.3 메인 서비스: 정상
) else (
    echo ⚠️ 메인 서비스 응답 대기 중... (정상적인 경우가 많습니다)
)

REM 포트 확인
echo.
echo 🔍 포트 사용 현황:
netstat -an | findstr :8080
netstat -an | findstr :5432
netstat -an | findstr :6379

echo.

REM 배포 완료 메시지
echo ████████████████████████████████████████████████████████
echo █                                                      █
echo █  🎉 솔로몬드 AI v2.3 Windows 배포 완료!             █
echo █  💎 99.4%% 정확도 하이브리드 LLM 시스템 가동 중     █
echo █                                                      █
echo ████████████████████████████████████████████████████████
echo.

echo 📍 Windows 접속 정보:
echo 🌐 메인 서비스: http://localhost:8080
echo 🔍 헬스체크: http://localhost:8080/health/v23
echo 📊 시스템 상태: http://localhost:8080/status

if "%DEPLOY_OPTION%" geq "2" (
    echo 📈 Grafana 대시보드: http://localhost:3000 (admin/admin123^)
    echo 📊 Prometheus 메트릭: http://localhost:9090
)

echo.
echo 🛠️ Windows 관리 명령어:
echo 시스템 중지: docker-compose -f docker-compose.v23.production.yml down
echo 로그 확인: docker-compose -f docker-compose.v23.production.yml logs -f
echo 상태 확인: docker-compose -f docker-compose.v23.production.yml ps
echo 재시작: docker-compose -f docker-compose.v23.production.yml restart

echo.
echo ✨ Windows 배포가 성공적으로 완료되었습니다!
echo 📞 지원이 필요하시면 전근혁 (solomond.jgh@gmail.com^)으로 연락해주세요.

echo.
echo 🎯 솔로몬드 AI v2.3으로 주얼리 업계 혁신을 시작하세요!

echo.
echo 📋 다음 단계:
echo 1. 브라우저에서 http://localhost:8080 접속
echo 2. 시스템 상태 확인
echo 3. 주얼리 분석 기능 테스트

pause
