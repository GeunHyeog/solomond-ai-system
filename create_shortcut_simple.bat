@echo off
title Create Desktop Shortcut

echo Creating SOLOMOND AI desktop shortcut...

powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\SOLOMOND_AI.lnk'); $Shortcut.TargetPath = '%CD%\SOLOMOND_AI_START.bat'; $Shortcut.WorkingDirectory = '%CD%'; $Shortcut.Save()"

echo.
if exist "%USERPROFILE%\Desktop\SOLOMOND_AI.lnk" (
    echo SUCCESS: Desktop shortcut created!
    echo Double-click 'SOLOMOND_AI' icon on desktop to start
) else (
    echo FAILED: Please manually copy SOLOMOND_AI_START.bat to desktop
)

echo.
pause