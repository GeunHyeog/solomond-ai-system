#!/usr/bin/env python3
"""
GEMMA3 상태 확인 (인코딩 문제 해결)
"""

import subprocess
import time
import sys

def check_ollama_status():
    """Ollama 상태 확인"""
    
    print("=== GEMMA3 상태 확인 ===")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            
            print(f"Ollama 서비스: 정상")
            print(f"설치된 모델: {len(models)}개")
            
            # 모델 목록 출력
            gemma_models = []
            for model in models:
                model_name = model['name']
                print(f"  - {model_name}")
                if 'gemma' in model_name.lower():
                    gemma_models.append(model_name)
            
            if gemma_models:
                print(f"\nGEMMA 모델: {len(gemma_models)}개 설치됨")
                return True, gemma_models
            else:
                print("\nGEMMA 모델: 설치되지 않음")
                return True, []
        else:
            print(f"서버 오류: {response.status_code}")
            return False, []
            
    except Exception as e:
        print(f"연결 실패: {str(e)}")
        return False, []

def test_gemma_performance(model_name):
    """GEMMA 성능 테스트"""
    
    print(f"\n=== {model_name} 성능 테스트 ===")
    
    try:
        import requests
        
        # 한국어 테스트 프롬프트
        prompt = """한국어로 답변해주세요:

고객이 "결혼반지 200만원 예산으로 심플한 디자인 찾고 있어요"라고 말했습니다.
이 고객에게 어떤 조언을 해주시겠어요?"""

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        
        print("분석 중...")
        start_time = time.time()
        
        response = requests.post("http://localhost:11434/api/generate", 
                               json=payload, timeout=45)
        
        if response.status_code == 200:
            end_time = time.time()
            result = response.json()
            answer = result.get('response', '')
            
            processing_time = end_time - start_time
            
            print(f"처리 시간: {processing_time:.2f}초")
            print(f"응답 길이: {len(answer)} 글자")
            
            # 한국어 키워드 분석
            keywords = ['결혼반지', '200만원', '예산', '심플', '디자인', '추천', '선택']
            found_keywords = [k for k in keywords if k in answer]
            
            print(f"키워드 인식: {len(found_keywords)}/{len(keywords)}개")
            print(f"인식된 키워드: {found_keywords}")
            
            print(f"\n응답 내용:")
            print("-" * 40)
            print(answer[:300] + "..." if len(answer) > 300 else answer)
            print("-" * 40)
            
            # 성능 평가
            speed_score = 10 if processing_time < 3 else (7 if processing_time < 5 else 5)
            keyword_score = (len(found_keywords) / len(keywords)) * 10
            length_score = min(10, len(answer) / 50)
            
            total_score = (speed_score + keyword_score + length_score) / 3
            
            print(f"\n성능 평가:")
            print(f"  속도: {speed_score}/10")
            print(f"  이해도: {keyword_score:.1f}/10")
            print(f"  응답품질: {length_score:.1f}/10")
            print(f"  종합점수: {total_score:.1f}/10")
            
            if total_score >= 7:
                print("판정: 우수")
            elif total_score >= 5:
                print("판정: 양호")
            else:
                print("판정: 개선필요")
            
            return {
                'success': True,
                'processing_time': processing_time,
                'response_length': len(answer),
                'keywords_found': len(found_keywords),
                'total_keywords': len(keywords),
                'score': total_score
            }
            
        else:
            print(f"API 오류: {response.status_code}")
            return {'success': False, 'error': f'API error {response.status_code}'}
            
    except Exception as e:
        print(f"테스트 실패: {str(e)}")
        return {'success': False, 'error': str(e)}

def compare_with_current():
    """기존 시스템과 비교"""
    
    print(f"\n=== 기존 시스템과 비교 ===")
    print(f"기존 eeve-korean-instruct-10.8b:")
    print(f"  예상 처리시간: 3-5초")
    print(f"  예상 한국어 이해도: 90%+")
    print(f"  메모리 사용량: 높음 (10.8B)")
    
    print(f"\nGEMMA2 9B:")
    print(f"  예상 처리시간: 2-4초")
    print(f"  예상 한국어 이해도: 80-90%")
    print(f"  메모리 사용량: 중간 (9B)")
    
    print(f"\n권장사항:")
    print(f"  - 속도 우선: GEMMA2 2B")
    print(f"  - 균형: GEMMA2 9B")
    print(f"  - 정확도 우선: 기존 eeve-korean")

def installation_guide():
    """설치 가이드"""
    
    print(f"\n=== GEMMA 모델 설치 가이드 ===")
    print(f"자동 설치:")
    print(f"  python install_ollama_windows.py")
    
    print(f"\n수동 설치:")
    print(f"  1. ollama serve")
    print(f"  2. ollama pull gemma2:2b")
    print(f"  3. ollama pull gemma2:9b")
    
    print(f"\n설치 확인:")
    print(f"  python check_gemma3_status.py")

def main():
    """메인 실행"""
    
    print("GEMMA3 상태 및 성능 확인")
    print("=" * 40)
    
    # 1. 상태 확인
    server_ok, gemma_models = check_ollama_status()
    
    if not server_ok:
        print("\nOllama 서버가 실행되지 않습니다.")
        installation_guide()
        return
    
    if not gemma_models:
        print("\nGEMMA 모델이 설치되지 않았습니다.")
        installation_guide()
        return
    
    # 2. 성능 테스트
    test_results = []
    
    for model in gemma_models[:2]:  # 최대 2개 모델만 테스트
        result = test_gemma_performance(model)
        if result['success']:
            test_results.append((model, result))
    
    # 3. 결과 요약
    if test_results:
        print(f"\n=== 테스트 결과 요약 ===")
        
        best_model = None
        best_score = 0
        
        for model_name, result in test_results:
            score = result['score']
            print(f"{model_name}: {score:.1f}/10점")
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        if best_model:
            print(f"\n추천 모델: {best_model} ({best_score:.1f}점)")
        
        # 4. 기존 시스템과 비교
        compare_with_current()
        
        print(f"\n다음 단계:")
        print(f"1. python simple_gemma3_test.py - 상세 테스트")
        print(f"2. 솔로몬드 AI 시스템에 통합")
        print(f"3. demo_integrated_system.py에서 성능 확인")
    
    else:
        print(f"\n성능 테스트를 완료할 수 없습니다.")
        installation_guide()

if __name__ == "__main__":
    main()