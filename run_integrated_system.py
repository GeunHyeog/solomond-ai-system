#!/usr/bin/env python3
"""
🚀 SOLOMOND AI 통합 시스템 실행
사용자 친화적 웹 UI + 고급 Module1 API 동시 실행
"""

import subprocess
import time
import requests
import sys
import threading
from pathlib import Path

def start_service_thread(script_name, port, service_name):
    """백그라운드 스레드에서 서비스 시작"""
    print(f"{service_name} 시작 중... (포트 {port})")
    try:
        process = subprocess.Popen([
            sys.executable, script_name
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 잠시 대기 후 상태 확인
        time.sleep(5)
        
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                print(f"OK {service_name} 정상 시작됨 (포트 {port})")
            else:
                print(f"ERROR {service_name} 응답 오류: {response.status_code}")
        except requests.exceptions.RequestException:
            print(f"WARNING {service_name} 연결 확인 실패 (서비스는 시작됨)")
        
        return process
    except Exception as e:
        print(f"ERROR {service_name} 시작 실패: {e}")
        return None

def check_system_status():
    """시스템 전체 상태 확인"""
    print("\n시스템 상태 확인...")
    
    services = [
        {"name": "Module1 Advanced API", "url": "http://localhost:8001/health"},
        {"name": "통합 웹 UI", "url": "http://localhost:8080/health"},
    ]
    
    for service in services:
        try:
            response = requests.get(service["url"], timeout=3)
            if response.status_code == 200:
                print(f"OK {service['name']}: 정상 작동")
            else:
                print(f"ERROR {service['name']}: 응답 오류 ({response.status_code})")
        except requests.exceptions.RequestException:
            print(f"ERROR {service['name']}: 연결 실패")

def display_access_info():
    """접속 정보 표시"""
    print("\n" + "="*60)
    print("SOLOMOND AI 통합 시스템이 시작되었습니다!")
    print("="*60)
    print()
    print("메인 접속 주소:")
    print("   통합 웹 UI: http://localhost:8080")
    print("      여기서 파일을 드래그하여 AI 분석을 받으세요!")
    print()
    print("개발자용 API:")
    print("   Module1 API 문서: http://localhost:8001/docs")
    print("   API Gateway: http://localhost:8000")
    print()
    print("주요 기능:")
    print("   - 드래그 앤 드롭 파일 업로드")
    print("   - 실시간 분석 진행률 표시") 
    print("   - AI 기반 종합 분석")
    print("   - 화자별 분석 및 요약")
    print("   - 비동기 대용량 파일 처리")
    print()
    print("지원 파일 형식:")
    print("   텍스트: .txt, .md")
    print("   이미지: .jpg, .png (OCR 준비)")
    print("   오디오: .mp3, .wav, .m4a (STT 준비)")
    print("   비디오: .mp4, .mov (처리 준비)")
    print()
    print("종료하려면 Ctrl+C를 누르세요...")

def main():
    """메인 실행 함수"""
    print("SOLOMOND AI 통합 시스템 시작")
    print("="*50)
    
    processes = []
    
    # 서비스들을 스레드로 시작
    services = [
        ("module1_api_advanced.py", 8001, "Module1 Advanced API"),
        ("integrated_web_ui.py", 8080, "통합 웹 UI"),
    ]
    
    threads = []
    for script, port, name in services:
        thread = threading.Thread(
            target=start_service_thread,
            args=(script, port, name)
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # 모든 서비스가 시작될 때까지 대기
    print("\n서비스 시작 대기 중...")
    time.sleep(10)
    
    # 시스템 상태 확인
    check_system_status()
    
    # 접속 정보 표시
    display_access_info()
    
    # 브라우저 자동 열기
    try:
        import webbrowser
        print("기본 브라우저에서 메인 UI를 열고 있습니다...")
        webbrowser.open("http://localhost:8080")
    except:
        print("브라우저를 수동으로 열어 http://localhost:8080 에 접속하세요.")
    
    # 메인 루프 - 사용자 입력 대기
    try:
        while True:
            user_input = input("\n명령어를 입력하세요 (status/help/quit): ").lower().strip()
            
            if user_input == 'quit' or user_input == 'q':
                print("시스템을 종료합니다...")
                break
            elif user_input == 'status' or user_input == 's':
                check_system_status()
            elif user_input == 'help' or user_input == 'h':
                print("\n사용 가능한 명령어:")
                print("  status (s) - 시스템 상태 확인")
                print("  help (h)   - 도움말 표시")
                print("  quit (q)   - 시스템 종료")
            else:
                print("알 수 없는 명령어입니다. 'help'를 입력하세요.")
                
    except KeyboardInterrupt:
        print("\n\n시스템 종료 중...")
    
    print("SOLOMOND AI 시스템이 종료되었습니다.")

if __name__ == "__main__":
    main()