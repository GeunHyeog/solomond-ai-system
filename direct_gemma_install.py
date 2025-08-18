#!/usr/bin/env python3
"""
GEMMA 모델 직접 설치
"""

import subprocess
import os
import time
import requests

def find_ollama_executable():
    """Ollama 실행 파일 찾기"""
    
    # 일반적인 설치 경로들
    possible_paths = [
        r"C:\Users\{}\AppData\Local\Programs\Ollama\ollama.exe".format(os.getenv("USERNAME")),
        r"C:\Program Files\Ollama\ollama.exe",
        r"C:\Program Files (x86)\Ollama\ollama.exe",
        r"C:\Users\{}\AppData\Local\Ollama\ollama.exe".format(os.getenv("USERNAME"))
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Ollama 발견: {path}")
            return path
    
    # PATH에서 찾기
    try:
        result = subprocess.run(["where", "ollama"], capture_output=True, text=True)
        if result.returncode == 0:
            path = result.stdout.strip().split('\n')[0]
            print(f"PATH에서 발견: {path}")
            return path
    except:
        pass
    
    return None

def install_gemma_models(ollama_path):
    """GEMMA 모델 설치"""
    
    print("=== GEMMA 모델 설치 ===")
    
    models = [
        ("gemma2:2b", "경량 버전"),
        ("gemma2:9b", "권장 버전")
    ]
    
    installed = []
    
    for model_id, description in models:
        print(f"\n{description} 설치: {model_id}")
        print("다운로드 중... (몇 분 소요)")
        
        try:
            # ollama pull 실행
            result = subprocess.run([ollama_path, "pull", model_id],
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

def test_installation():
    """설치 확인 테스트"""
    
    print("\n=== 설치 확인 ===")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            
            gemma_models = [m for m in models if 'gemma' in m['name'].lower()]
            
            print(f"전체 모델: {len(models)}개")
            print(f"GEMMA 모델: {len(gemma_models)}개")
            
            for model in gemma_models:
                print(f"  - {model['name']}")
            
            return len(gemma_models) > 0
        else:
            print(f"API 오류: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"확인 실패: {str(e)}")
        return False

def quick_performance_test():
    """빠른 성능 테스트"""
    
    print("\n=== 빠른 성능 테스트 ===")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        data = response.json()
        models = data.get('models', [])
        
        gemma_models = [m for m in models if 'gemma' in m['name'].lower()]
        
        if not gemma_models:
            print("테스트할 GEMMA 모델 없음")
            return False
        
        # 가장 작은 모델로 테스트
        test_model = sorted(gemma_models, key=lambda x: x['name'])[0]['name']
        print(f"테스트 모델: {test_model}")
        
        # 간단한 한국어 테스트
        prompt = "안녕하세요. 결혼반지 추천해주세요."
        
        payload = {
            "model": test_model,
            "prompt": prompt,
            "stream": False
        }
        
        print("테스트 실행 중...")
        start_time = time.time()
        
        response = requests.post("http://localhost:11434/api/generate", 
                               json=payload, timeout=30)
        
        if response.status_code == 200:
            end_time = time.time()
            result = response.json()
            answer = result.get('response', '')
            
            print(f"✓ 처리시간: {end_time - start_time:.2f}초")
            print(f"✓ 응답길이: {len(answer)} 글자")
            
            # 한국어 키워드 확인
            korean_keywords = ['결혼', '반지', '추천']
            found = sum(1 for k in korean_keywords if k in answer)
            
            print(f"✓ 한국어 이해: {found}/{len(korean_keywords)} 키워드")
            print(f"✓ 응답 샘플: {answer[:80]}...")
            
            if found >= 1 and len(answer) > 20:
                print("\n🎉 테스트 성공! GEMMA 모델이 정상 작동합니다.")
                return True
            else:
                print("\n⚠️ 테스트 부분 성공")
                return True
        else:
            print(f"✗ API 오류: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ 테스트 실패: {str(e)}")
        return False

def main():
    """메인 실행"""
    
    print("GEMMA 모델 직접 설치")
    print("=" * 30)
    
    # 1. Ollama 실행파일 찾기
    ollama_path = find_ollama_executable()
    
    if not ollama_path:
        print("Ollama 실행 파일을 찾을 수 없습니다.")
        print("Ollama를 먼저 설치해주세요:")
        print("https://ollama.ai/download")
        return False
    
    # 2. GEMMA 모델 설치
    installed_models = install_gemma_models(ollama_path)
    
    if not installed_models:
        print("\n모델 설치에 실패했습니다.")
        return False
    
    print(f"\n설치된 모델: {installed_models}")
    
    # 3. 설치 확인
    if test_installation():
        print("\n모델 설치 확인 완료")
        
        # 4. 성능 테스트
        if quick_performance_test():
            print(f"\n다음 단계:")
            print(f"1. python check_gemma3_status.py - 상세 확인")
            print(f"2. python simple_gemma3_test.py - 성능 분석")
            print(f"3. 솔로몬드 AI 시스템 통합")
            return True
        else:
            print(f"\n성능 테스트에 문제가 있지만 설치는 완료되었습니다.")
            return True
    else:
        print(f"\n설치 확인에 실패했습니다.")
        return False

if __name__ == "__main__":
    main()