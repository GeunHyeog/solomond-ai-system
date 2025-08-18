#!/usr/bin/env python3
"""
솔로몬드 AI 시스템 개선점 분석 및 업그레이드
현재 GEMMA3:4b + Qwen3:8b를 활용한 성능 최적화
"""

import requests
import time
import json
from datetime import datetime

def check_current_system_status():
    """현재 시스템 상태 분석"""
    print("=== 솔로몬드 AI 시스템 현재 상태 분석 ===")
    print(f"분석 시간: {datetime.now()}")
    print()
    
    # 1. Ollama 모델 상태 확인
    print("1. 사용 가능한 AI 모델들:")
    try:
        r = requests.get('http://localhost:11434/api/tags')
        if r.status_code == 200:
            models = r.json()['models']
            
            latest_models = []
            standard_models = []
            
            for model in models:
                name = model['name']
                size_gb = model['size'] / (1024**3)
                
                if 'qwen3' in name or 'gemma3' in name:
                    latest_models.append((name, size_gb, "3세대 최신"))
                else:
                    standard_models.append((name, size_gb, "2세대/표준"))
            
            print("   🥇 3세대 최신 모델들:")
            for name, size, gen in latest_models:
                print(f"      - {name} ({size:.1f}GB)")
            
            print("   🥈 2세대/표준 모델들:")
            for name, size, gen in standard_models:
                print(f"      - {name} ({size:.1f}GB)")
                
        else:
            print("   ❌ Ollama 연결 실패")
            
    except Exception as e:
        print(f"   ❌ 모델 상태 확인 실패: {e}")
    
    print()
    
    # 2. 시스템 서비스 상태 확인
    print("2. 서비스 상태:")
    services = [
        ("Streamlit UI", "http://localhost:8504"),
        ("Ollama API", "http://localhost:11434/api/tags"),
    ]
    
    for service_name, url in services:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print(f"   ✅ {service_name}: 정상 작동")
            else:
                print(f"   ⚠️ {service_name}: 응답 오류 ({r.status_code})")
        except:
            print(f"   ❌ {service_name}: 연결 실패")
    
    print()

def identify_improvement_areas():
    """개선 필요 영역 식별"""
    print("=== 개선 필요 영역 분석 ===")
    
    improvements = {
        "성능 최적화": [
            "GEMMA3:4b를 메인 분석 엔진으로 전환",
            "Qwen3:8b를 한국어 전문 분석용으로 특화",
            "모델별 역할 분담 최적화",
            "배치 처리 성능 향상"
        ],
        "분석 정확도 향상": [
            "최신 모델 기반 프롬프트 엔지니어링",
            "다중 모델 앙상블 분석",
            "컨텍스트 인식 개선",
            "도메인 특화 fine-tuning 적용"
        ],
        "사용자 경험 개선": [
            "실시간 진행 상황 표시 강화",
            "에러 핸들링 및 복구 자동화",
            "결과 시각화 개선",
            "반응 속도 최적화"
        ],
        "시스템 안정성": [
            "메모리 사용량 최적화",
            "에러 로깅 및 모니터링 강화",
            "백업 및 복구 시스템",
            "자동 재시작 메커니즘"
        ]
    }
    
    for category, items in improvements.items():
        print(f"{category}:")
        for i, item in enumerate(items, 1):
            print(f"   {i}. {item}")
        print()

def test_latest_models_performance():
    """최신 모델들 성능 테스트"""
    print("=== 최신 모델 성능 비교 테스트 ===")
    
    models_to_test = ["qwen3:8b", "gemma3:4b", "gemma2:9b"]
    test_prompt = "한국어로 답변해주세요: 결혼반지 200만원 예산으로 어떤 제품을 추천하시나요?"
    
    results = []
    
    for model in models_to_test:
        print(f"\n테스트 중: {model}")
        
        try:
            payload = {
                "model": model,
                "prompt": test_prompt,
                "stream": False
            }
            
            start_time = time.time()
            response = requests.post("http://localhost:11434/api/generate",
                                   json=payload, timeout=60)
            
            if response.status_code == 200:
                end_time = time.time()
                result_data = response.json()
                answer = result_data.get('response', '')
                
                processing_time = end_time - start_time
                
                # 품질 분석
                keywords = ['결혼반지', '200만원', '예산', '추천', '다이아몬드', '디자인']
                found_keywords = [k for k in keywords if k in answer]
                quality_score = len(found_keywords) / len(keywords) * 100
                
                results.append({
                    'model': model,
                    'time': processing_time,
                    'quality': quality_score,
                    'length': len(answer),
                    'keywords_found': len(found_keywords)
                })
                
                print(f"   ⏱️ 처리시간: {processing_time:.2f}초")
                print(f"   📊 품질점수: {quality_score:.1f}%")
                print(f"   📝 응답길이: {len(answer)} 글자")
                print(f"   🔍 키워드 인식: {len(found_keywords)}/{len(keywords)}개")
                
            else:
                print(f"   ❌ API 오류: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ 테스트 실패: {str(e)}")
    
    # 결과 비교
    if results:
        print(f"\n=== 성능 비교 결과 ===")
        
        # 최고 성능 모델 선정
        best_quality = max(results, key=lambda x: x['quality'])
        fastest = min(results, key=lambda x: x['time'])
        
        print(f"🏆 최고 품질: {best_quality['model']} ({best_quality['quality']:.1f}%)")
        print(f"⚡ 최고 속도: {fastest['model']} ({fastest['time']:.2f}초)")
        
        # 종합 점수 (품질 70% + 속도 30%)
        for result in results:
            speed_score = (10 / max(result['time'], 1)) * 10  # 속도를 점수로 변환
            total_score = result['quality'] * 0.7 + min(speed_score, 100) * 0.3
            result['total_score'] = total_score
        
        best_overall = max(results, key=lambda x: x['total_score'])
        print(f"🎯 종합 최고: {best_overall['model']} ({best_overall['total_score']:.1f}점)")
        
        return best_overall['model']
    
    return None

def generate_improvement_plan(best_model):
    """개선 계획 생성"""
    print(f"\n=== 개선 실행 계획 ===")
    print(f"최적 모델: {best_model}")
    print()
    
    plan = {
        "단기 개선 (즉시 적용)": [
            f"ollama_integration_engine.py에서 korean_chat를 {best_model}로 설정",
            "GEMMA3:4b를 감정 분석 전용으로 활용",
            "프롬프트 템플릿 최신 모델에 맞게 최적화",
            "응답 속도 개선을 위한 타임아웃 조정"
        ],
        "중기 개선 (1-2주)": [
            "다중 모델 앙상블 시스템 구축",
            "실시간 모델 성능 모니터링 추가",
            "자동 모델 선택 알고리즘 개발",
            "분석 결과 캐싱 시스템 구현"
        ],
        "장기 개선 (1개월+)": [
            "도메인 특화 fine-tuning 데이터 구축",
            "한국어 주얼리 전문 모델 훈련",
            "A/B 테스트 프레임워크 구축",
            "성능 벤치마크 자동화 시스템"
        ]
    }
    
    for phase, tasks in plan.items():
        print(f"{phase}:")
        for i, task in enumerate(tasks, 1):
            print(f"   {i}. {task}")
        print()
    
    return plan

def main():
    """메인 실행"""
    print("솔로몬드 AI 시스템 개선점 분석")
    print("=" * 50)
    print()
    
    # 1. 현재 상태 분석
    check_current_system_status()
    
    # 2. 개선 영역 식별
    identify_improvement_areas()
    
    # 3. 최신 모델 성능 테스트
    best_model = test_latest_models_performance()
    
    # 4. 개선 계획 생성
    if best_model:
        improvement_plan = generate_improvement_plan(best_model)
        
        print("=== 다음 단계 ===")
        print("1. 위 개선 계획에 따라 시스템 업그레이드")
        print("2. 성능 테스트 및 검증")
        print("3. 사용자 피드백 수집 및 반영")
        print()
        print("🚀 최신 AI 모델을 활용한 성능 향상 준비 완료!")
    
    else:
        print("⚠️ 모델 테스트에 문제가 있습니다. 연결 상태를 확인해주세요.")

if __name__ == "__main__":
    main()