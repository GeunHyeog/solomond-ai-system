@echo off
title Fix Desktop Shortcut

echo Fixing desktop shortcut...

:: Delete old shortcut
if exist "%USERPROFILE%\Desktop\SOLOMOND_AI.lnk" (
    del "%USERPROFILE%\Desktop\SOLOMOND_AI.lnk"
    echo Old shortcut removed
)

:: Create new shortcut pointing to working BAT file
powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\SOLOMOND_AI_FIXED.lnk'); $Shortcut.TargetPath = '%CD%\START_SIMPLE.bat'; $Shortcut.WorkingDirectory = '%CD%'; $Shortcut.Save()"

if exist "%USERPROFILE%\Desktop\SOLOMOND_AI_FIXED.lnk" (
    echo SUCCESS: New shortcut created!
    echo Double-click 'SOLOMOND_AI_FIXED' on desktop
) else (
    echo FAILED: Please use START_SIMPLE.bat directly
)

echo.
pause