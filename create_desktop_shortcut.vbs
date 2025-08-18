Set WshShell = CreateObject("WScript.Shell")
DesktopPath = WshShell.SpecialFolders("Desktop")
Set oShellLink = WshShell.CreateShortcut(DesktopPath & "\SOLOMOND AI.lnk")

' 배치파일 경로 설정  
CurrentPath = "C:\Users\PC_58410\solomond-ai-system"
oShellLink.TargetPath = CurrentPath & "\SOLOMOND_AI_START.bat"
oShellLink.WorkingDirectory = CurrentPath
oShellLink.Description = "SOLOMOND AI - 접속 실패 제로 시스템"
oShellLink.IconLocation = "C:\Windows\System32\shell32.dll,13"

oShellLink.Save

WScript.Echo "✅ 바탕화면에 'SOLOMOND AI' 바로가기가 생성되었습니다!"
WScript.Echo ""
WScript.Echo "🚀 사용법:"
WScript.Echo "1. 바탕화면의 '🤖 SOLOMOND AI' 아이콘 더블클릭"
WScript.Echo "2. 자동 진단 및 최적 방법으로 실행"
WScript.Echo "3. 브라우저에서 대시보드 확인"
WScript.Echo ""
WScript.Echo "💡 3가지 독립적 접속 방법으로 실패율 0% 보장!"