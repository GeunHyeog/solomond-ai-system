#!/usr/bin/env python3
"""
GEMMA3 빠른 시작 및 상태 확인
"""

import subprocess
import time
import requests

def quick_check():
    """빠른 상태 확인"""
    
    print("=== GEMMA3 빠른 상태 확인 ===")
    
    # 1. Ollama 실행 확인
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            gemma_models = [m for m in models if 'gemma' in m['name'].lower()]
            
            print(f"✓ Ollama 서비스: 정상")
            print(f"✓ 전체 모델: {len(models)}개")
            print(f"✓ GEMMA 모델: {len(gemma_models)}개")
            
            if gemma_models:
                for model in gemma_models:
                    print(f"  - {model['name']}")
                return True
            else:
                print("✗ GEMMA 모델 없음")
                return False
                
        else:
            print(f"✗ 서버 오류: {response.status_code}")
            return False
            
    except:
        print("✗ Ollama 서버 연결 실패")
        return False

def quick_test():
    """빠른 성능 테스트"""
    
    print("\n=== GEMMA3 빠른 테스트 ===")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = response.json().get('models', [])
        gemma_models = [m for m in models if 'gemma' in m['name'].lower()]
        
        if not gemma_models:
            print("✗ 테스트할 GEMMA 모델 없음")
            return False
        
        test_model = gemma_models[0]['name']
        print(f"테스트 모델: {test_model}")
        
        # 간단한 한국어 테스트
        prompt = "한국어로 답변하세요: 결혼반지 추천해주세요."
        
        payload = {
            "model": test_model,
            "prompt": prompt,
            "stream": False
        }
        
        print("테스트 요청 중...")
        start_time = time.time()
        
        response = requests.post("http://localhost:11434/api/generate", 
                               json=payload, timeout=30)
        
        if response.status_code == 200:
            end_time = time.time()
            result = response.json()
            answer = result.get('response', '')
            
            print(f"✓ 응답 시간: {end_time - start_time:.2f}초")
            print(f"✓ 응답 길이: {len(answer)} 글자")
            
            # 한국어 키워드 체크
            keywords = ['결혼', '반지', '추천', '선택', '다이아']
            found = sum(1 for k in keywords if k in answer)
            
            print(f"✓ 한국어 이해도: {found}/{len(keywords)} ({found/len(keywords)*100:.1f}%)")
            print(f"✓ 응답 샘플: {answer[:100]}...")
            
            if found >= 2 and len(answer) > 50:
                print("✓ 테스트 성공!")
                return True
            else:
                print("△ 테스트 부분 성공")
                return True
        else:
            print(f"✗ API 오류: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ 테스트 실패: {str(e)}")
        return False

def integration_status():
    """솔로몬드 AI 통합 상태 확인"""
    
    print("\n=== 솔로몬드 AI 통합 상태 ===")
    
    # ollama_integration_engine.py 확인
    try:
        with open("core/ollama_integration_engine.py", 'r', encoding='utf-8') as f:
            content = f.read()
            
        if "gemma2" in content.lower():
            print("✓ GEMMA2 이미 통합됨")
        else:
            print("△ GEMMA2 통합 필요")
            print("  실행: python integrate_gemma3.py")
            
    except:
        print("△ 통합 파일 확인 실패")

def next_steps():
    """다음 단계 안내"""
    
    print("\n=== 다음 실행 단계 ===")
    print("1. python simple_gemma3_test.py     # 상세 성능 테스트")
    print("2. python demo_integrated_system.py # 통합 시스템 테스트")
    print("3. 메인 UI에서 GEMMA3 성능 확인")
    print("4. 기존 eeve-korean 모델과 비교")

def main():
    """메인 실행"""
    
    print("GEMMA3 빠른 시작 점검")
    print("="*30)
    
    # 상태 확인
    if quick_check():
        # 성능 테스트
        if quick_test():
            # 통합 상태 확인
            integration_status()
            
            # 다음 단계 안내
            next_steps()
            
            print("\n🎉 GEMMA3 준비 완료!")
            return True
        else:
            print("\n⚠️ 성능 테스트에 문제가 있습니다.")
    else:
        print("\n❌ GEMMA3 설정이 완료되지 않았습니다.")
        print("\n해결 방법:")
        print("1. python install_ollama_windows.py 재실행")
        print("2. 수동으로 ollama serve 실행")
        print("3. ollama pull gemma2:9b 실행")
    
    return False

if __name__ == "__main__":
    main()