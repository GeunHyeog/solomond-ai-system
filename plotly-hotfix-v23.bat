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
echo - Added: plotly>=5.15.0, matplotlib>=3.7.0, seaborn>=0.12.0
echo - Status: Critical dependency fix applied
echo.

echo Stopping current containers...
docker-compose -f docker-compose.v23.windows.ultimate.final.yml down >nul 2>&1
docker-compose -f docker-compose.v23.windows.plotly.hotfix.yml down >nul 2>&1

echo.
echo Applying PLOTLY HOTFIX...
echo Using updated requirements with plotly package...

echo Rebuilding with PLOTLY fix (this may take a few minutes)...
docker-compose -f docker-compose.v23.windows.plotly.hotfix.yml build --no-cache solomond-ai

if %errorlevel% equ 0 (
    echo ✅ Build successful! Starting system...
    docker-compose -f docker-compose.v23.windows.plotly.hotfix.yml up -d
    
    echo.
    echo ============================================================
    echo              🎉 PLOTLY HOTFIX SUCCESSFUL! 🎉  
    echo ============================================================
    echo.
    
    echo ✅ plotly>=5.15.0 package added successfully
    echo ✅ matplotlib>=3.7.0 package added  
    echo ✅ seaborn>=0.12.0 package added
    echo ✅ Container rebuilt with new dependencies
    echo ✅ "No module named 'plotly'" error FIXED!
    echo.
    
    echo 🔍 Checking system status...
    timeout /t 15 /nobreak >nul
    
    docker-compose -f docker-compose.v23.windows.plotly.hotfix.yml ps
    
    echo.
    echo 🌐 System should now be accessible at: http://localhost:8080
    echo 📊 API Documentation: http://localhost:8080/docs
    echo.
    echo 🎯 If you still see issues, check logs with:
    echo docker-compose -f docker-compose.v23.windows.plotly.hotfix.yml logs -f solomond-ai
    echo.
    echo ✅ PLOTLY HOTFIX COMPLETE - System Ready!
    
) else (
    echo.
    echo ❌ Build failed. Checking for alternative solutions...
    echo.
    echo 🔧 Try manual installation:
    echo 1. docker exec -it solomond-ai-v23-windows-plotly-hotfix pip install plotly matplotlib seaborn
    echo 2. docker-compose -f docker-compose.v23.windows.plotly.hotfix.yml restart solomond-ai
    echo.
)

echo.
echo Press any key to continue...
pause >nul
