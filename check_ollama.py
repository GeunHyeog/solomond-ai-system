#!/usr/bin/env python3
"""
간단한 Ollama 상태 확인 스크립트
"""

import subprocess
import platform
import requests
import json

def check_ollama_installation():
    """Ollama 설치 상태 확인"""
    
    print("=== Ollama 상태 확인 ===")
    
    # 1. 명령어 존재 확인
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"✓ Ollama 설치됨: {result.stdout.strip()}")
            installed = True
        else:
            print("✗ Ollama 명령어 실행 실패")
            installed = False
            
    except FileNotFoundError:
        print("✗ Ollama 설치되지 않음")
        installed = False
    except Exception as e:
        print(f"✗ 확인 중 오류: {str(e)}")
        installed = False
    
    # 2. 서버 상태 확인
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            print("✓ Ollama 서버 실행 중")
            
            # 설치된 모델 확인
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            
            print(f"✓ 설치된 모델 ({len(models)}개):")
            for model in models:
                print(f"  - {model}")
                
            server_running = True
        else:
            print("✗ 서버 응답 오류")
            server_running = False
            
    except requests.exceptions.ConnectionError:
        print("✗ Ollama 서버 실행되지 않음")
        server_running = False
    except Exception as e:
        print(f"✗ 서버 확인 중 오류: {str(e)}")
        server_running = False
    
    # 3. 권장사항 제시
    print("\n=== 권장사항 ===")
    
    if not installed:
        system = platform.system()
        if system == "Windows":
            print("Windows 설치방법:")
            print("1. https://ollama.ai/download 방문")
            print("2. Windows 설치파일 다운로드")
            print("3. 설치 후 'ollama serve' 실행")
        elif system == "Linux":
            print("Linux 설치방법:")
            print("curl -fsSL https://ollama.ai/install.sh | sh")
        elif system == "Darwin":
            print("macOS 설치방법:")
            print("brew install ollama")
    
    elif not server_running:
        print("서버 시작방법:")
        print("ollama serve")
        
    else:
        print("✓ Ollama 정상 작동 중!")
        print("\n추천 모델 설치:")
        print("ollama pull llama3.2:3b  # 경량 모델")
        print("ollama pull mistral:7b   # 감정 분석")
        
    return installed and server_running

def test_korean_analysis():
    """간단한 한국어 분석 테스트"""
    
    if not check_ollama_installation():
        return
    
    print("\n=== 한국어 분석 테스트 ===")
    
    try:
        # 간단한 텍스트 분석 요청
        payload = {
            "model": "llama3.2:3b",
            "prompt": "다음 한국어 대화를 요약해주세요: 고객: 결혼반지 보러 왔어요. 상담사: 어떤 스타일을 선호하시나요?",
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "응답 없음")
            print(f"✓ 분석 결과: {answer[:100]}...")
        else:
            print(f"✗ API 호출 실패: {response.status_code}")
            
    except Exception as e:
        print(f"✗ 테스트 실패: {str(e)}")

if __name__ == "__main__":
    print("Ollama 상태 확인 도구")
    print("=" * 40)
    
    try:
        is_working = check_ollama_installation()
        
        if is_working:
            print("\n한국어 분석 테스트를 실행하시겠습니까? (y/N): ", end="")
            if input().lower() == 'y':
                test_korean_analysis()
                
    except KeyboardInterrupt:
        print("\n사용자가 중단했습니다.")
    except Exception as e:
        print(f"\n예상치 못한 오류: {str(e)}")