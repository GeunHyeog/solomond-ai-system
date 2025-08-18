#!/usr/bin/env python3
"""
GEMMA3 자동 설치 스크립트 (단순화)
"""

import subprocess
import time
import sys
import os

def check_ollama():
    """Ollama 설치 및 실행 상태 확인"""
    
    print("=== Ollama 상태 확인 ===")
    
    # 1. 설치 확인
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"Ollama 설치됨: {result.stdout.strip()}")
        else:
            print("Ollama 설치되지 않음")
            return False
    except:
        print("Ollama 명령어 실행 실패")
        return False
    
    # 2. 서비스 실행 확인
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            print("Ollama 서비스 실행 중")
            return True
        else:
            print("Ollama 서비스 응답 없음")
    except:
        print("Ollama 서비스 연결 실패")
    
    return False

def start_ollama():
    """Ollama 서비스 시작"""
    
    print("\n=== Ollama 서비스 시작 ===")
    
    try:
        # 백그라운드로 ollama serve 시작
        print("ollama serve 시작 중...")
        subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
        
        # 10초 대기
        print("서비스 시작 대기... (10초)")
        time.sleep(10)
        
        # 연결 확인
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("Ollama 서비스 시작 성공")
                return True
        except:
            pass
        
        print("서비스 시작 실패")
        return False
        
    except Exception as e:
        print(f"서비스 시작 오류: {str(e)}")
        return False

def install_gemma_models():
    """GEMMA 모델 설치"""
    
    print("\n=== GEMMA 모델 설치 ===")
    
    models = [
        ("gemma2:2b", "경량 버전"),
        ("gemma2:9b", "권장 버전")
    ]
    
    installed = []
    
    for model_id, desc in models:
        print(f"\n{desc} 설치: {model_id}")
        
        try:
            # ollama pull 실행
            result = subprocess.run(["ollama", "pull", model_id],
                                  capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(f"설치 완료: {model_id}")
                installed.append(model_id)
            else:
                print(f"설치 실패: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"설치 시간 초과: {model_id}")
        except Exception as e:
            print(f"설치 오류: {str(e)}")
    
    return installed

def test_gemma_simple():
    """GEMMA 간단 테스트"""
    
    print("\n=== GEMMA 간단 테스트 ===")
    
    try:
        import requests
        
        # 사용 가능한 모델 확인
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code != 200:
            print("API 연결 실패")
            return False
        
        models = response.json().get('models', [])
        gemma_models = [m for m in models if 'gemma' in m['name'].lower()]
        
        if not gemma_models:
            print("GEMMA 모델이 설치되지 않음")
            return False
        
        test_model = gemma_models[0]['name']
        print(f"테스트 모델: {test_model}")
        
        # 간단한 한국어 테스트
        prompt = "안녕하세요, 한국어로 답변해주세요: 결혼반지 추천해주세요."
        
        payload = {
            "model": test_model,
            "prompt": prompt,
            "stream": False
        }
        
        print("테스트 실행 중...")
        response = requests.post("http://localhost:11434/api/generate", 
                               json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '')
            print(f"응답 길이: {len(answer)} 글자")
            print(f"응답 일부: {answer[:100]}...")
            
            # 한국어 키워드 체크
            korean_words = ['결혼', '반지', '추천', '선택']
            found = [w for w in korean_words if w in answer]
            print(f"한국어 이해도: {len(found)}/{len(korean_words)} 키워드")
            
            return True
        else:
            print(f"API 오류: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"테스트 실패: {str(e)}")
        return False

def main():
    """메인 실행"""
    
    print("GEMMA3 자동 설치 시작")
    print("="*40)
    
    # 1. Ollama 확인
    if not check_ollama():
        # Ollama 서비스 시작 시도
        if not start_ollama():
            print("\n설치 필요:")
            print("1. https://ollama.ai/download 에서 Ollama 다운로드")
            print("2. 설치 후 'ollama serve' 실행")
            print("3. 다시 이 스크립트 실행")
            return False
    
    # 2. GEMMA 모델 설치
    installed_models = install_gemma_models()
    
    if not installed_models:
        print("\n모델 설치 실패")
        return False
    
    print(f"\n설치된 모델: {installed_models}")
    
    # 3. 간단 테스트
    if test_gemma_simple():
        print("\n성공! GEMMA3 설치 및 테스트 완료")
        print("\n다음 단계:")
        print("1. python simple_gemma3_test.py - 상세 테스트")
        print("2. 솔로몬드 AI 시스템에 통합")
        return True
    else:
        print("\n테스트 실패 - 수동 확인 필요")
        return False

if __name__ == "__main__":
    main()