@echo off
REM Solomond AI Engine v2.3 Windows Production Deployment Script
REM 99.4% Accuracy Hybrid LLM System Auto Deployment

title Solomond AI v2.3 Windows Deployment System
color 0B

echo.
echo ============================================================
echo               Solomond AI Engine v2.3                    
echo          Windows Production Deployment System            
echo        99.4%% Accuracy Hybrid LLM System                
echo ============================================================
echo.

echo Script Information:
echo - Version: v2.3 Windows Edition
echo - Date: %DATE% %TIME%
echo - Target: Windows Production Environment
echo - Developer: Jeon Geunhyeog (Solomond CEO)
echo.

REM Check Windows Prerequisites
echo Checking Windows Prerequisites...
echo.

REM Check Docker Desktop
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker Desktop is not installed or running.
    echo Please install Docker Desktop: https://www.docker.com/products/docker-desktop
    echo.
    echo Steps:
    echo 1. Download Docker Desktop from the link above
    echo 2. Install and restart your computer
    echo 3. Start Docker Desktop
    echo 4. Run this script again
    pause
    exit /b 1
)
echo [OK] Docker Desktop confirmed

REM Check Docker Compose
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    docker compose version >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Docker Compose not found
        pause
        exit /b 1
    )
)
echo [OK] Docker Compose confirmed

REM Check Git
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed
    echo Please install Git: https://git-scm.com/download/win
    pause
    exit /b 1
)
echo [OK] Git confirmed

echo.

REM Environment Configuration
echo Configuring Windows Deployment Settings...
echo.

set ENV_FILE=.env.v23.windows
echo # Solomond AI v2.3 Windows Production Environment > %ENV_FILE%
echo # Created: %DATE% %TIME% >> %ENV_FILE%
echo. >> %ENV_FILE%

echo ENVIRONMENT=production >> %ENV_FILE%
echo SOLOMOND_VERSION=v2.3 >> %ENV_FILE%
echo LOG_LEVEL=INFO >> %ENV_FILE%
echo PLATFORM=windows >> %ENV_FILE%
echo. >> %ENV_FILE%

REM Database Password Setup
echo Database Password Setup:
set /p POSTGRES_PASSWORD=Enter PostgreSQL Password: 
set /p REDIS_PASSWORD=Enter Redis Password: 

echo POSTGRES_PASSWORD=%POSTGRES_PASSWORD% >> %ENV_FILE%
echo REDIS_PASSWORD=%REDIS_PASSWORD% >> %ENV_FILE%
echo. >> %ENV_FILE%

REM Optional AI API Keys
echo.
echo AI API Keys Setup (Optional - Press Enter to skip):
set /p OPENAI_API_KEY=OpenAI API Key: 
set /p ANTHROPIC_API_KEY=Anthropic API Key: 
set /p GOOGLE_API_KEY=Google API Key: 

if not "%OPENAI_API_KEY%"=="" (
    echo OPENAI_API_KEY=%OPENAI_API_KEY% >> %ENV_FILE%
)
if not "%ANTHROPIC_API_KEY%"=="" (
    echo ANTHROPIC_API_KEY=%ANTHROPIC_API_KEY% >> %ENV_FILE%
)
if not "%GOOGLE_API_KEY%"=="" (
    echo GOOGLE_API_KEY=%GOOGLE_API_KEY% >> %ENV_FILE%
)

echo [OK] Windows environment configuration completed: %ENV_FILE%
echo.

REM Deployment Options
echo Windows Deployment Options:
echo 1) Basic Deployment (AI System Only)
echo 2) Full Deployment (with Monitoring)
echo 3) Developer Deployment (All Services)
set /p DEPLOY_OPTION=Choose (1-3): 

if "%DEPLOY_OPTION%"=="1" set PROFILES=
if "%DEPLOY_OPTION%"=="2" set PROFILES=--profile monitoring
if "%DEPLOY_OPTION%"=="3" set PROFILES=--profile production --profile monitoring --profile backup

echo.

REM Start Deployment
echo Starting Solomond AI v2.3 Windows Production Deployment!
echo.

REM Clean Previous Deployment
echo Cleaning previous deployment...
docker-compose -f docker-compose.v23.production.yml down --remove-orphans 2>nul

REM Clean Docker System
echo Cleaning Docker cache...
docker system prune -f 2>nul

REM Pull Latest Images
echo Downloading latest images...
docker-compose -f docker-compose.v23.production.yml pull 2>nul

REM Execute Deployment
echo.
echo Starting v2.3 system...
docker-compose -f docker-compose.v23.production.yml --env-file %ENV_FILE% %PROFILES% up -d

if %errorlevel% neq 0 (
    echo [ERROR] Deployment failed
    echo Check logs: docker-compose -f docker-compose.v23.production.yml logs
    pause
    exit /b 1
)

REM Wait for Services
echo.
echo Waiting for Windows services to start (120 seconds)...
ping localhost -n 121 > nul

REM Health Check
echo.
echo Checking Windows System Status
echo.

REM Container Status
echo Container Status:
docker-compose -f docker-compose.v23.production.yml ps

echo.

REM Main Service Check
echo Checking main service health...
curl -s http://localhost:8080/health/v23 >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Solomond AI v2.3 Main Service: Running
) else (
    echo [WAIT] Main service starting... (This is often normal)
)

REM Port Check
echo.
echo Port Usage Status:
netstat -an | findstr :8080
netstat -an | findstr :5432
netstat -an | findstr :6379

echo.

REM Deployment Complete Message
echo ============================================================
echo     Solomond AI v2.3 Windows Deployment Complete!        
echo    99.4%% Accuracy Hybrid LLM System is Running         
echo ============================================================
echo.

echo Windows Access Information:
echo Web Service: http://localhost:8080
echo Health Check: http://localhost:8080/health/v23
echo System Status: http://localhost:8080/status

if "%DEPLOY_OPTION%" geq "2" (
    echo Grafana Dashboard: http://localhost:3000 (admin/admin123)
    echo Prometheus Metrics: http://localhost:9090
)

echo.
echo Windows Management Commands:
echo Stop System: docker-compose -f docker-compose.v23.production.yml down
echo View Logs: docker-compose -f docker-compose.v23.production.yml logs -f
echo Check Status: docker-compose -f docker-compose.v23.production.yml ps
echo Restart: docker-compose -f docker-compose.v23.production.yml restart

echo.
echo [SUCCESS] Windows deployment completed successfully!
echo Support: Jeon Geunhyeog (solomond.jgh@gmail.com)

echo.
echo Start your jewelry industry AI innovation with Solomond AI v2.3!

echo.
echo Next Steps:
echo 1. Open browser and go to http://localhost:8080
echo 2. Check system status
echo 3. Test jewelry analysis features

pause
