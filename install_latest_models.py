#!/usr/bin/env python3
"""
2025년 최신 고성능 모델 설치
"""

import subprocess
import os
import time
import requests

def find_ollama():
    """Ollama 실행파일 찾기"""
    
    paths = [
        r"C:\Users\{}\AppData\Local\Programs\Ollama\ollama.exe".format(os.getenv("USERNAME")),
        r"C:\Program Files\Ollama\ollama.exe",
        r"C:\Program Files (x86)\Ollama\ollama.exe"
    ]
    
    for path in paths:
        if os.path.exists(path):
            return path
    
    return "ollama"  # PATH에 있다고 가정

def check_available_models():
    """사용 가능한 최신 모델 확인"""
    
    print("=== 2025년 최신 모델 확인 ===")
    
    # 추천 모델들 (성능 순)
    latest_models = [
        ("qwen2.5:14b", "Qwen2.5 14B - 한국어 특화 최신 모델 (강력 추천!)"),
        ("qwen2.5:7b", "Qwen2.5 7B - 한국어 특화 경량 버전"),
        ("llama3.3:8b", "Llama 3.3 8B - Meta 최신 모델"),
        ("llama3.2:8b", "Llama 3.2 8B - Meta 안정 버전"),  
        ("mistral-nemo:12b", "Mistral-Nemo 12B - 효율성 최강"),
        ("phi3.5:3.8b", "Phi 3.5 - Microsoft 경량 고성능")
    ]
    
    print("추천 최신 모델:")
    for i, (model_id, description) in enumerate(latest_models, 1):
        print(f"  {i}. {description}")
    
    return latest_models

def install_recommended_models(ollama_path):
    """추천 모델 설치"""
    
    print("\n=== 최신 모델 자동 설치 ===")
    
    # 1순위 모델들 (한국어 주얼리에 최적)
    priority_models = [
        ("qwen2.5:7b", "Qwen2.5 7B - 한국어 최적화 (1순위)"),
        ("llama3.2:8b", "Llama 3.2 8B - 범용 고성능 (2순위)")
    ]
    
    installed = []
    
    for model_id, description in priority_models:
        print(f"\n{description} 설치 중...")
        print(f"모델: {model_id}")
        print("다운로드 진행 중... (5-10분 소요)")
        
        try:
            # 실시간 진행 상황 표시
            process = subprocess.Popen(
                [ollama_path, "pull", model_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )
            
            # 진행 상황 모니터링
            dots = 0
            while process.poll() is None:
                print("." * (dots % 10 + 1), end="\r")
                dots += 1
                time.sleep(1)
            
            if process.returncode == 0:
                print(f"\n✓ {model_id} 설치 완료!")
                installed.append(model_id)
            else:
                error = process.stderr.read()
                print(f"\n✗ {model_id} 설치 실패: {error}")
                
        except Exception as e:
            print(f"\n✗ 설치 오류: {str(e)}")
    
    return installed

def performance_benchmark(models):
    """최신 모델 성능 벤치마크"""
    
    print(f"\n=== 최신 모델 성능 벤치마크 ===")
    
    # 한국어 주얼리 전문 테스트
    test_prompt = """한국어로 전문적으로 답변해주세요:

고객 상황: "25살 직장인이 결혼 예정이에요. 약혼반지와 결혼반지 세트로 300만원 예산인데, 어떤 걸 추천하시나요? 다이아몬드 크기보다는 디자인이 예쁜 게 좋겠어요."

전문 상담사로서 이 고객에게 어떤 조언과 제품을 추천하시겠습니까?"""

    results = []
    
    for model in models:
        print(f"\n--- {model} 성능 테스트 ---")
        
        payload = {
            "model": model,
            "prompt": test_prompt,
            "stream": False
        }
        
        try:
            start_time = time.time()
            
            response = requests.post("http://localhost:11434/api/generate",
                                   json=payload, timeout=60)
            
            if response.status_code == 200:
                end_time = time.time()
                result = response.json()
                answer = result.get('response', '')
                
                processing_time = end_time - start_time
                
                # 품질 분석
                keywords = ['약혼반지', '결혼반지', '300만원', '예산', '디자인', '추천', '다이아몬드', '25살']
                found_keywords = [k for k in keywords if k in answer]
                
                quality_score = len(found_keywords) / len(keywords) * 100
                
                print(f"  처리시간: {processing_time:.2f}초")
                print(f"  응답길이: {len(answer)} 글자")
                print(f"  키워드 인식: {len(found_keywords)}/{len(keywords)}개")
                print(f"  품질점수: {quality_score:.1f}%")
                
                # 응답 품질 평가
                if len(answer) > 200 and quality_score > 50:
                    print(f"  평가: 우수 ✓")
                elif len(answer) > 100 and quality_score > 30:
                    print(f"  평가: 양호 △")
                else:
                    print(f"  평가: 개선필요 ✗")
                
                results.append({
                    'model': model,
                    'time': processing_time,
                    'quality': quality_score,
                    'length': len(answer),
                    'keywords': len(found_keywords)
                })
                
                print(f"  응답 샘플: {answer[:100]}...")
                
            else:
                print(f"  API 오류: {response.status_code}")
                
        except Exception as e:
            print(f"  테스트 실패: {str(e)}")
    
    # 최고 성능 모델 선정
    if results:
        best_model = max(results, 
                        key=lambda x: x['quality'] * 0.4 + (100/max(x['time'], 1)) * 0.3 + min(x['length']/10, 10) * 0.3)
        
        print(f"\n🏆 최고 성능 모델: {best_model['model']}")
        print(f"   품질점수: {best_model['quality']:.1f}%")
        print(f"   처리속도: {best_model['time']:.2f}초")
        print(f"   응답품질: {best_model['length']} 글자")
    
    return results

def integrate_to_solomond():
    """솔로몬드 AI 시스템에 통합"""
    
    print(f"\n=== 솔로몬드 AI 통합 준비 ===")
    print(f"1. core/ollama_integration_engine.py 업데이트")
    print(f"2. 최신 모델을 기본 모델로 설정")
    print(f"3. 한국어 프롬프트 최적화")
    print(f"4. demo_integrated_system.py에서 테스트")

def main():
    """메인 실행"""
    
    print("🚀 2025년 최신 AI 모델 설치")
    print("="*40)
    
    # 1. Ollama 찾기
    ollama_path = find_ollama()
    
    # 2. 사용 가능한 최신 모델 확인
    latest_models = check_available_models()
    
    print(f"\n추천: Qwen2.5가 한국어 주얼리 상담에 최적입니다!")
    
    # 3. 최신 모델 설치
    installed = install_recommended_models(ollama_path)
    
    if not installed:
        print(f"\n설치 실패. 수동 설치 해보세요:")
        print(f"  ollama pull qwen2.5:7b")
        print(f"  ollama pull llama3.2:8b")
        return
    
    print(f"\n✓ 설치된 최신 모델: {installed}")
    
    # 4. 성능 벤치마크
    results = performance_benchmark(installed)
    
    # 5. 시스템 통합 안내
    integrate_to_solomond()
    
    print(f"\n🎉 최신 모델 설치 완료!")
    print(f"다음 단계:")
    print(f"1. python check_gemma3_status.py - 설치 확인")
    print(f"2. 솔로몬드 AI 메인 시스템에서 테스트")
    print(f"3. 기존 eeve-korean 모델과 성능 비교")

if __name__ == "__main__":
    main()