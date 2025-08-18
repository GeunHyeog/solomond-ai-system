Set WshShell = CreateObject("WScript.Shell")
DesktopPath = WshShell.SpecialFolders("Desktop")
Set oShellLink = WshShell.CreateShortcut(DesktopPath & "\SOLOMOND AI.lnk")

CurrentPath = "C:\Users\PC_58410\solomond-ai-system"
oShellLink.TargetPath = CurrentPath & "\SOLOMOND_AI_START.bat"
oShellLink.WorkingDirectory = CurrentPath
oShellLink.Description = "SOLOMOND AI - 접속 실패 제로 시스템"
oShellLink.IconLocation = "C:\Windows\System32\shell32.dll,13"

oShellLink.Save

WScript.Echo "바탕화면에 SOLOMOND AI 바로가기가 생성되었습니다!"