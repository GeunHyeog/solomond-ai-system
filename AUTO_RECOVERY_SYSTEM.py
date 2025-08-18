#!/usr/bin/env python3
"""
🚨 솔로몬드 AI 자동 복구 시스템 🚨
- 접속 문제 자동 진단 및 해결
- 재발방지 시스템 구축
- 실시간 헬스체크
"""
import subprocess
import time
import requests
import os
import sys
from pathlib import Path

class AutoRecoverySystem:
    def __init__(self):
        self.base_path = Path(r"C:\Users\PC_58410\solomond-ai-system")
        self.streamlit_path = r"C:\Users\PC_58410\AppData\Roaming\Python\Python313\Scripts\streamlit.exe"
        self.target_ports = [8511, 8501]
        
    def kill_existing_processes(self):
        """기존 프로세스 완전 종료"""
        print("Terminating existing processes...")
        try:
            subprocess.run(["taskkill", "/f", "/im", "streamlit.exe"], 
                          capture_output=True, check=False)
            subprocess.run(["taskkill", "/f", "/im", "python.exe"], 
                          capture_output=True, check=False)
            time.sleep(2)
        except Exception as e:
            print(f"프로세스 종료 중 오류: {e}")
    
    def start_streamlit_service(self, script_path, port, name):
        """Streamlit 서비스 시작"""
        print(f"Starting {name}... (port {port})")
        
        cmd = [
            self.streamlit_path, "run", str(script_path),
            "--server.port", str(port),
            "--server.address", "localhost"
        ]
        
        try:
            # 백그라운드에서 실행
            subprocess.Popen(
                cmd, 
                cwd=str(self.base_path),
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            time.sleep(5)  # 시작 대기
            return True
        except Exception as e:
            print(f"Failed to start {name}: {e}")
            return False
    
    def health_check(self, port, timeout=30):
        """포트 헬스체크"""
        print(f"Health checking port {port}...")
        
        for i in range(timeout):
            try:
                response = requests.get(f"http://localhost:{port}", timeout=2)
                if response.status_code == 200:
                    print(f"Port {port} OK!")
                    return True
            except:
                if i % 5 == 0:  # 5초마다 진행상황 출력
                    print(f"Waiting for port {port}... ({i}/{timeout})")
                time.sleep(1)
        
        print(f"Port {port} health check failed!")
        return False
    
    def emergency_recovery(self):
        """응급 복구 실행"""
        print("=== EMERGENCY RECOVERY START ===")
        
        # 1단계: 기존 프로세스 종료
        self.kill_existing_processes()
        
        # 2단계: 서비스 시작
        services = [
            (self.base_path / "solomond_ai_main_dashboard.py", 8511, "메인 대시보드"),
            (self.base_path / "modules" / "module1_conference" / "conference_analysis.py", 8501, "Module1 분석")
        ]
        
        started_services = []
        for script_path, port, name in services:
            if script_path.exists():
                if self.start_streamlit_service(script_path, port, name):
                    started_services.append((port, name))
            else:
                print(f"File not found: {script_path}")
        
        # 3단계: 헬스체크
        success_count = 0
        for port, name in started_services:
            if self.health_check(port):
                success_count += 1
        
        # 4단계: 결과 보고
        if success_count == len(started_services):
            print("=== COMPLETE RECOVERY SUCCESS ===")
            print("Main Dashboard: http://localhost:8511")
            print("Module1 Analysis: http://localhost:8501")
            
            # 브라우저 자동 열기
            os.system("start http://localhost:8511")
            os.system("start http://localhost:8501")
            return True
        else:
            print(f"Partial recovery: {success_count}/{len(started_services)} success")
            return False
    
    def install_watchdog(self):
        """재발방지 감시 시스템 설치"""
        watchdog_script = '''
@echo off
:watch
echo [%time%] 헬스체크 실행 중...
curl -s http://localhost:8511 >nul
if %errorlevel% neq 0 (
    echo [%time%] 포트 8511 다운! 자동 복구 중...
    python "C:\\Users\\PC_58410\\solomond-ai-system\\AUTO_RECOVERY_SYSTEM.py"
)
timeout /t 60 /nobreak >nul
goto watch
'''
        watchdog_path = self.base_path / "WATCHDOG_SYSTEM.bat"
        with open(watchdog_path, 'w', encoding='cp949') as f:
            f.write(watchdog_script)
        
        print(f"Watchdog system installed: {watchdog_path}")

if __name__ == "__main__":
    # Windows 콘솔 UTF-8 설정
    os.system("chcp 65001 >nul")
    
    recovery = AutoRecoverySystem()
    
    print("SOLOMOND AI AUTO RECOVERY SYSTEM")
    print("=" * 40)
    
    # 응급 복구 실행
    if recovery.emergency_recovery():
        print("\nInstalling watchdog system...")
        recovery.install_watchdog()
        print("All tasks completed!")
    else:
        print("Recovery failed - manual check required")
        sys.exit(1)