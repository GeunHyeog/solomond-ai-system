#!/usr/bin/env python3
"""
GEMMA3 및 2025년 최신 모델 설치
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
    
    return "ollama"

def check_available_latest_models():
    """2025년 최신 모델 확인"""
    
    print("=== 2025년 7월 최신 모델 목록 ===")
    
    # 실제 사용 가능한 최신 모델들
    latest_models = [
        # Google 최신
        ("gemma2:9b", "Gemma 2 9B - Google 최신 (현재 최신)"),
        ("gemma2:27b", "Gemma 2 27B - Google 고성능"),
        ("gemma2:2b", "Gemma 2 2B - Google 경량"),
        
        # 참고: Gemma 3는 아직 공식 출시 전이거나 제한적
        # ("gemma3:9b", "Gemma 3 9B - Google 차세대 (출시시)"),
        
        # Meta 최신
        ("llama3.2:8b", "Llama 3.2 8B - Meta 최신 안정 버전"),
        ("llama3.2:3b", "Llama 3.2 3B - Meta 경량"),
        ("llama3.1:8b", "Llama 3.1 8B - Meta 검증된 버전"),
        
        # Alibaba 최신 (한국어 우수)
        ("qwen2.5:14b", "Qwen 2.5 14B - 아시아 언어 특화 최강"),
        ("qwen2.5:7b", "Qwen 2.5 7B - 아시아 언어 특화"),
        ("qwen2.5:3b", "Qwen 2.5 3B - 아시아 언어 경량"),
        
        # Microsoft 최신
        ("phi3.5:3.8b", "Phi 3.5 - Microsoft 소형 고성능"),
        
        # Mistral 최신
        ("mistral-nemo:12b", "Mistral Nemo 12B - 효율성 최강"),
        
        # DeepSeek 최신
        ("deepseek-r1:7b", "DeepSeek R1 7B - 추론 특화 (있다면)"),
        
        # 코드 특화
        ("codellama:7b", "CodeLlama 7B - 코드 생성 특화"),
    ]
    
    print("🎯 한국어 주얼리 상담용 추천 순위:")
    print("1. Qwen 2.5 (아시아 언어 최강)")
    print("2. Llama 3.2 (범용 안정성)")
    print("3. Gemma 2 (구글 최신)")
    print("4. Mistral Nemo (효율성)")
    
    return latest_models

def install_top_models(ollama_path):
    """상위 추천 모델 설치"""
    
    print("\n=== 최신 최강 모델 설치 ===")
    
    # 성능 + 한국어 + 최신성 기준 TOP 모델들
    top_models = [
        ("qwen2.5:7b", "🥇 Qwen 2.5 7B - 한국어 최강 (1순위)"),
        ("llama3.2:8b", "🥈 Llama 3.2 8B - 범용 최신 (2순위)"),
        ("gemma2:9b", "🥉 Gemma 2 9B - Google 최신 (3순위)")
    ]
    
    installed = []
    
    for model_id, description in top_models:
        print(f"\n{description}")
        print(f"모델 설치 중: {model_id}")
        print("다운로드 진행... (크기에 따라 3-15분 소요)")
        
        try:
            # 진행 상황 표시
            process = subprocess.Popen(
                [ollama_path, "pull", model_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 간단한 진행 표시
            dots = 0
            while process.poll() is None:
                print(f"진행 중{'.' * (dots % 4)}", end="\r")
                dots += 1
                time.sleep(2)
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print(f"\n✅ {model_id} 설치 완료!")
                installed.append(model_id)
            else:
                print(f"\n❌ {model_id} 설치 실패")
                print(f"오류: {stderr}")
                
        except Exception as e:
            print(f"\n❌ 설치 중 오류: {str(e)}")
    
    return installed

def comprehensive_benchmark(models):
    """종합 성능 벤치마크"""
    
    print(f"\n=== 최신 모델 종합 성능 테스트 ===")
    
    # 실제 주얼리 상담 시나리오
    test_scenarios = [
        {
            "name": "복잡한_주얼리_상담",
            "prompt": """전문 주얼리 상담사로서 답변해주세요:

고객: "안녕하세요. 결혼 5주년 기념으로 아내에게 목걸이를 선물하려고 해요. 
아내는 30대 중반이고 평소에 심플하고 클래식한 스타일을 좋아해요. 
예산은 150만원 정도인데, 다이아몬드가 들어간 것과 안 들어간 것 중 어떤 게 더 좋을까요? 
그리고 금 재질별로 차이점도 알려주세요."

위 상황을 바탕으로 전문적이고 자세한 상담을 해주세요.""",
            "keywords": ["5주년", "목걸이", "30대", "심플", "클래식", "150만원", "다이아몬드", "금", "재질"]
        },
        {
            "name": "긴급_제품_추천",
            "prompt": """빠르고 정확하게 답변해주세요:

"내일이 여자친구 생일인데 깜빡했어요! 20대 초반 대학생이고 귀여운 걸 좋아해요. 
30만원 이하로 귀걸이나 목걸이 추천해주세요. 어떤 게 좋을까요?"

간단명료하면서도 실용적인 추천을 해주세요.""",
            "keywords": ["생일", "20대", "대학생", "귀여운", "30만원", "귀걸이", "목걸이", "추천"]
        }
    ]
    
    results = []
    
    for model in models:
        print(f"\n{'='*50}")
        print(f"🧪 {model} 성능 테스트")
        print(f"{'='*50}")
        
        model_results = {
            "model": model,
            "scenarios": [],
            "avg_time": 0,
            "avg_quality": 0,
            "total_score": 0
        }
        
        total_time = 0
        total_quality = 0
        
        for scenario in test_scenarios:
            print(f"\n--- {scenario['name']} ---")
            
            payload = {
                "model": model,
                "prompt": scenario["prompt"],
                "stream": False
            }
            
            try:
                start_time = time.time()
                
                response = requests.post("http://localhost:11434/api/generate",
                                       json=payload, timeout=120)
                
                if response.status_code == 200:
                    end_time = time.time()
                    result = response.json()
                    answer = result.get('response', '')
                    
                    processing_time = end_time - start_time
                    total_time += processing_time
                    
                    # 품질 분석
                    keywords = scenario["keywords"]
                    found_keywords = [k for k in keywords if k in answer]
                    keyword_score = len(found_keywords) / len(keywords) * 100
                    
                    # 응답 길이 점수 (너무 짧거나 길면 감점)
                    ideal_length = 300  # 이상적인 응답 길이
                    length_score = min(100, (len(answer) / ideal_length) * 100)
                    if len(answer) > ideal_length * 2:  # 너무 길면 감점
                        length_score *= 0.8
                    
                    # 전문성 점수 (전문 용어 포함 여부)
                    professional_terms = ["주얼리", "다이아몬드", "금", "은", "백금", "캐럿", "컷", "클래리티"]
                    found_terms = [t for t in professional_terms if t in answer]
                    professional_score = min(100, len(found_terms) * 20)
                    
                    # 종합 품질 점수
                    quality_score = (keyword_score * 0.4 + length_score * 0.3 + professional_score * 0.3)
                    total_quality += quality_score
                    
                    print(f"처리시간: {processing_time:.2f}초")
                    print(f"응답길이: {len(answer)} 글자")
                    print(f"키워드 인식: {len(found_keywords)}/{len(keywords)}개 ({keyword_score:.1f}%)")
                    print(f"전문용어: {len(found_terms)}개")
                    print(f"품질점수: {quality_score:.1f}/100")
                    
                    # 응답 샘플
                    print(f"응답 샘플: {answer[:120]}...")
                    
                    scenario_result = {
                        "scenario": scenario['name'],
                        "time": processing_time,
                        "quality": quality_score,
                        "keywords_found": len(found_keywords),
                        "professional_terms": len(found_terms)
                    }
                    
                    model_results["scenarios"].append(scenario_result)
                    
                else:
                    print(f"API 오류: {response.status_code}")
                    
            except Exception as e:
                print(f"테스트 실패: {str(e)}")
        
        # 모델별 종합 점수 계산
        if model_results["scenarios"]:
            model_results["avg_time"] = total_time / len(test_scenarios)
            model_results["avg_quality"] = total_quality / len(test_scenarios)
            
            # 속도 점수 (빠를수록 높음, 최대 100점)
            speed_score = max(0, 100 - (model_results["avg_time"] * 10))
            
            # 종합 점수 (품질 70%, 속도 30%)
            model_results["total_score"] = (model_results["avg_quality"] * 0.7 + speed_score * 0.3)
            
            print(f"\n📊 {model} 종합 결과:")
            print(f"   평균 처리시간: {model_results['avg_time']:.2f}초")
            print(f"   평균 품질점수: {model_results['avg_quality']:.1f}/100")
            print(f"   종합 점수: {model_results['total_score']:.1f}/100")
        
        results.append(model_results)
    
    # 최종 순위 발표
    if results:
        print(f"\n🏆 최종 성능 순위")
        print(f"{'='*60}")
        
        # 종합 점수 순으로 정렬
        ranked_results = sorted(results, key=lambda x: x['total_score'], reverse=True)
        
        for i, result in enumerate(ranked_results, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}위"
            print(f"{medal} {result['model']}")
            print(f"   종합점수: {result['total_score']:.1f}/100")
            print(f"   품질: {result['avg_quality']:.1f}/100")
            print(f"   속도: {result['avg_time']:.2f}초")
            print()
        
        # 추천
        best_model = ranked_results[0]
        print(f"🎯 솔로몬드 AI 추천 모델: {best_model['model']}")
        print(f"   이유: 종합 성능 {best_model['total_score']:.1f}점으로 최고 성능")
    
    return results

def main():
    """메인 실행"""
    
    print("🚀 GEMMA3 & 2025년 최신 모델 설치")
    print("="*50)
    
    # 1. Ollama 찾기
    ollama_path = find_ollama()
    
    # 2. 최신 모델 확인
    latest_models = check_available_latest_models()
    
    print(f"\n💡 참고: Gemma 3는 아직 정식 출시 전이거나 제한적입니다.")
    print(f"현재는 Gemma 2가 Google의 최신 안정 버전입니다.")
    
    # 3. TOP 모델 설치
    installed = install_top_models(ollama_path)
    
    if not installed:
        print(f"\n❌ 자동 설치 실패")
        print(f"수동 설치 시도:")
        print(f"  ollama pull qwen2.5:7b")
        print(f"  ollama pull llama3.2:8b")
        print(f"  ollama pull gemma2:9b")
        return
    
    print(f"\n✅ 설치 완료: {len(installed)}개 모델")
    for model in installed:
        print(f"  ✓ {model}")
    
    # 4. 종합 성능 벤치마크
    print(f"\n🧪 이제 성능 테스트를 시작합니다...")
    time.sleep(2)
    
    benchmark_results = comprehensive_benchmark(installed)
    
    print(f"\n🎉 설치 및 벤치마크 완료!")
    print(f"다음 단계:")
    print(f"1. 최고 성능 모델을 솔로몬드 AI 시스템에 통합")
    print(f"2. demo_integrated_system.py에서 실제 테스트")
    print(f"3. 기존 eeve-korean 모델과 성능 비교")

if __name__ == "__main__":
    main()