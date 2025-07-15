@echo off
chcp 65001 >nul
cls

echo ============================================================
echo         Solomond AI Engine v2.3 PLOTLY HOTFIX
echo        Windows Compatible Deployment System
echo       PLOTLY DEPENDENCY ISSUE HOTFIX PATCH
echo ============================================================
echo.

echo PLOTLY HOTFIX Information:
echo - Version: v2.3 Windows PLOTLY HOTFIX Edition  
echo - Date: %date% %time%
echo - Issue: Fixed "No module named 'plotly'" error
echo - Added: plotly, matplotlib, seaborn packages
echo - Status: Critical dependency fix applied
echo.

echo Stopping current containers...
docker-compose -f docker-compose.v23.windows.ultimate.final.yml down

echo.
echo Applying PLOTLY HOTFIX...
echo Updating requirements with plotly package...

echo Rebuilding with PLOTLY fix...
docker-compose -f docker-compose.v23.windows.ultimate.final.yml build --no-cache solomond-ai

echo.
echo Starting system with PLOTLY fix...
docker-compose -f docker-compose.v23.windows.ultimate.final.yml up -d

echo.
echo ============================================================
echo              🔧 PLOTLY HOTFIX APPLIED! 🔧  
echo ============================================================
echo.

echo ✅ plotly>=5.15.0 package added
echo ✅ matplotlib>=3.7.0 package added  
echo ✅ seaborn>=0.12.0 package added
echo ✅ Container rebuilt with new dependencies
echo.

echo 🔍 Checking system status...
timeout /t 10 /nobreak >nul

docker-compose -f docker-compose.v23.windows.ultimate.final.yml ps

echo.
echo 🌐 Try accessing: http://localhost:8080
echo.
echo 📊 To check logs: 
echo docker-compose -f docker-compose.v23.windows.ultimate.final.yml logs -f solomond-ai
echo.
echo Press any key to continue...
pause >nul
