#!/usr/bin/env python3
"""
🚀 자동 서버 시작 스크립트 - SOLOMOND AI
사용자 친화적 서버 자동 시작 및 관리
"""

import subprocess
import time
import sys
import os
from pathlib import Path
import webbrowser
import signal
import psutil

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent

def check_port_available(port):
    """포트 사용 가능 여부 확인"""
    try:
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                return False
        return True
    except:
        return True

def start_conference_analysis():
    """컨퍼런스 분석 모듈 시작"""
    print("🎤 컨퍼런스 분석 모듈 시작 중...")
    
    if not check_port_available(8501):
        print("⚠️ 포트 8501이 이미 사용 중입니다.")
        return None
    
    try:
        # Streamlit 명령어
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(PROJECT_ROOT / "modules" / "module1_conference" / "conference_analysis.py"),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "true"
        ]
        
        # 백그라운드에서 실행
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=PROJECT_ROOT
        )
        
        # 시작 대기
        time.sleep(3)
        
        # 브라우저에서 열기
        webbrowser.open("http://localhost:8501")
        
        print("✅ 컨퍼런스 분석 모듈이 http://localhost:8501 에서 실행 중입니다.")
        return process
        
    except Exception as e:
        print(f"❌ 컨퍼런스 분석 모듈 시작 실패: {e}")
        return None

def start_web_crawler():
    """웹 크롤러 모듈 시작"""
    print("🕷️ 웹 크롤러 모듈 시작 중...")
    
    if not check_port_available(8502):
        print("⚠️ 포트 8502가 이미 사용 중입니다.")
        return None
    
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(PROJECT_ROOT / "modules" / "module2_crawler" / "web_crawler_main.py"),
            "--server.port", "8502",
            "--server.address", "localhost", 
            "--server.headless", "true"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=PROJECT_ROOT
        )
        
        time.sleep(3)
        webbrowser.open("http://localhost:8502")
        
        print("✅ 웹 크롤러 모듈이 http://localhost:8502 에서 실행 중입니다.")
        return process
        
    except Exception as e:
        print(f"❌ 웹 크롤러 모듈 시작 실패: {e}")
        return None

def start_gemstone_analyzer():
    """보석 분석 모듈 시작"""
    print("💎 보석 분석 모듈 시작 중...")
    
    if not check_port_available(8503):
        print("⚠️ 포트 8503이 이미 사용 중입니다.")
        return None
    
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(PROJECT_ROOT / "modules" / "module3_gemstone" / "gemstone_analyzer.py"),
            "--server.port", "8503",
            "--server.address", "localhost",
            "--server.headless", "true"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=PROJECT_ROOT
        )
        
        time.sleep(3)
        webbrowser.open("http://localhost:8503")
        
        print("✅ 보석 분석 모듈이 http://localhost:8503 에서 실행 중입니다.")
        return process
        
    except Exception as e:
        print(f"❌ 보석 분석 모듈 시작 실패: {e}")
        return None

def start_cad_converter():
    """3D CAD 변환 모듈 시작"""
    print("🏗️ 3D CAD 변환 모듈 시작 중...")
    
    if not check_port_available(8504):
        print("⚠️ 포트 8504가 이미 사용 중입니다.")
        return None
    
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(PROJECT_ROOT / "modules" / "module4_3d_cad" / "image_to_cad.py"),
            "--server.port", "8504",
            "--server.address", "localhost",
            "--server.headless", "true"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=PROJECT_ROOT
        )
        
        time.sleep(3)
        webbrowser.open("http://localhost:8504")
        
        print("✅ 3D CAD 변환 모듈이 http://localhost:8504 에서 실행 중입니다.")
        return process
        
    except Exception as e:
        print(f"❌ 3D CAD 변환 모듈 시작 실패: {e}")
        return None

def main():
    """메인 함수"""
    print("=" * 60)
    print("🚀 SOLOMOND AI - 자동 서버 시작")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\n📋 사용법:")
        print("python auto_server_start.py [module]")
        print("\n📦 사용 가능한 모듈:")
        print("  conference  - 🎤 컨퍼런스 분석 (포트 8501)")
        print("  crawler     - 🕷️ 웹 크롤러 (포트 8502)")
        print("  gemstone    - 💎 보석 분석 (포트 8503)")
        print("  cad         - 🏗️ 3D CAD 변환 (포트 8504)")
        print("  all         - 🎯 모든 모듈 시작")
        return
    
    module = sys.argv[1].lower()
    processes = []
    
    try:
        if module == "conference":
            process = start_conference_analysis()
            if process:
                processes.append(process)
                
        elif module == "crawler":
            process = start_web_crawler()
            if process:
                processes.append(process)
                
        elif module == "gemstone":
            process = start_gemstone_analyzer()
            if process:
                processes.append(process)
                
        elif module == "cad":
            process = start_cad_converter()
            if process:
                processes.append(process)
                
        elif module == "all":
            print("🎯 모든 모듈 시작 중...")
            
            for mod_func in [start_conference_analysis, start_web_crawler, 
                           start_gemstone_analyzer, start_cad_converter]:
                process = mod_func()
                if process:
                    processes.append(process)
                time.sleep(2)  # 순차 시작
            
            print(f"\n🎉 {len(processes)}개 모듈이 성공적으로 시작되었습니다!")
            
        else:
            print(f"❌ 알 수 없는 모듈: {module}")
            return
        
        if processes:
            print(f"\n✅ 성공적으로 {len(processes)}개 프로세스가 시작되었습니다.")
            print("🔄 서버들이 백그라운드에서 실행 중입니다.")
            print("💡 종료하려면 Ctrl+C를 누르거나 브라우저에서 서버를 종료하세요.")
            
            # 메인 대시보드 열기
            time.sleep(1)
            webbrowser.open("file://" + str(PROJECT_ROOT / "simple_dashboard.html"))
            print("🎯 메인 대시보드가 열렸습니다.")
            
            # 프로세스 모니터링
            try:
                while True:
                    time.sleep(5)
                    # 프로세스 상태 확인
                    alive = [p for p in processes if p.poll() is None]
                    if not alive:
                        print("⚠️ 모든 서버가 종료되었습니다.")
                        break
            except KeyboardInterrupt:
                print("\n🛑 종료 신호 받음. 서버들을 종료하는 중...")
                for process in processes:
                    try:
                        process.terminate()
                    except:
                        pass
                print("✅ 모든 서버가 종료되었습니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()