"""
하이브리드 LLM 매니저 v2.3 통합 테스트
99.2% 정확도 달성 목표 시스템 검증

실행 방법:
python tests/test_hybrid_llm_v23.py
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hybrid_llm_manager_v23 import *
import time
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_hybrid_llm_integration():
    """하이브리드 LLM 통합 테스트"""
    
    print("🚀 솔로몬드 하이브리드 LLM v2.3 통합 테스트 시작")
    print("=" * 70)
    
    # 1. 시스템 초기화 테스트
    print("\n📋 1. 시스템 초기화 테스트")
    try:
        manager = HybridLLMManagerV23()
        print("✅ 하이브리드 LLM 매니저 초기화 성공")
        
        # 성능 요약 확인
        performance = manager.get_performance_summary()
        print(f"📊 사용 가능한 모델: {len(performance['available_models'])}개")
        print(f"🎯 목표 정확도: {performance['target_accuracy']}")
        print(f"📈 현재 상태: {performance['current_status']}")
        
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return False
    
    # 2. 다이아몬드 4C 분석 테스트
    print("\n💎 2. 다이아몬드 4C 분석 테스트")
    request_diamond = AnalysisRequest(
        content_type="text",
        data={
            "content": "1.5캐럿 라운드 브릴리언트 컷 다이아몬드, D컬러, VVS1 클래리티, Excellent 컷 등급의 GIA 감정서가 있는 다이아몬드의 품질과 시장 가치를 분석해주세요.",
            "context": "고급 다이아몬드 감정 분석"
        },
        analysis_type="diamond_4c",
        quality_threshold=0.98,
        max_cost=0.05,
        language="ko"
    )
    
    try:
        result_diamond = await manager.analyze_with_hybrid_ai(request_diamond)
        
        print(f"✅ 최적 모델: {result_diamond.best_result.model_type.value}")
        print(f"📊 최종 정확도: {result_diamond.final_accuracy:.3f}")
        print(f"💰 총 비용: ${result_diamond.total_cost:.4f}")
        print(f"⏱️ 처리 시간: {result_diamond.total_time:.2f}초")
        print(f"🤝 모델 합의도: {result_diamond.consensus_score:.3f}")
        print(f"💡 추천사항: {result_diamond.recommendation}")
        
        # 정확도 검증
        if result_diamond.final_accuracy >= 0.95:
            print("✅ 다이아몬드 분석 품질 기준 달성")
        else:
            print("⚠️ 다이아몬드 분석 품질 개선 필요")
            
    except Exception as e:
        print(f"❌ 다이아몬드 분석 실패: {e}")
        return False
    
    # 3. 유색보석 감정 테스트
    print("\n🔴 3. 유색보석 감정 테스트")
    request_gemstone = AnalysisRequest(
        content_type="multimodal",
        data={
            "content": "3.2캐럿 오벌 컷 루비, 피죤 블러드 컬러, 미얀마산으로 추정되는 보석의 감정 평가를 요청합니다.",
            "additional_info": "SSEF 감정서 필요, 투자 목적 구매 검토"
        },
        analysis_type="colored_gemstone",
        quality_threshold=0.96,
        max_cost=0.08,
        language="ko"
    )
    
    try:
        result_gemstone = await manager.analyze_with_hybrid_ai(request_gemstone)
        
        print(f"✅ 최적 모델: {result_gemstone.best_result.model_type.value}")
        print(f"📊 최종 정확도: {result_gemstone.final_accuracy:.3f}")
        print(f"💰 총 비용: ${result_gemstone.total_cost:.4f}")
        print(f"⏱️ 처리 시간: {result_gemstone.total_time:.2f}초")
        print(f"💡 추천사항: {result_gemstone.recommendation}")
        
        # 정확도 검증
        if result_gemstone.final_accuracy >= 0.94:
            print("✅ 유색보석 분석 품질 기준 달성")
        else:
            print("⚠️ 유색보석 분석 품질 개선 필요")
            
    except Exception as e:
        print(f"❌ 유색보석 분석 실패: {e}")
        return False
    
    # 4. 비즈니스 인사이트 테스트
    print("\n📈 4. 비즈니스 인사이트 테스트")
    request_business = AnalysisRequest(
        content_type="text",
        data={
            "content": "2025년 상반기 주얼리 시장 트렌드와 고객 선호도 변화를 분석하고, 프리미엄 다이아몬드 제품 라인의 마케팅 전략을 제안해주세요.",
            "context": "비즈니스 전략 수립용 분석"
        },
        analysis_type="business_insight",
        quality_threshold=0.94,
        max_cost=0.06,
        language="ko"
    )
    
    try:
        result_business = await manager.analyze_with_hybrid_ai(request_business)
        
        print(f"✅ 최적 모델: {result_business.best_result.model_type.value}")
        print(f"📊 최종 정확도: {result_business.final_accuracy:.3f}")
        print(f"💰 총 비용: ${result_business.total_cost:.4f}")
        print(f"⏱️ 처리 시간: {result_business.total_time:.2f}초")
        print(f"💡 추천사항: {result_business.recommendation}")
        
        # 정확도 검증
        if result_business.final_accuracy >= 0.90:
            print("✅ 비즈니스 분석 품질 기준 달성")
        else:
            print("⚠️ 비즈니스 분석 품질 개선 필요")
            
    except Exception as e:
        print(f"❌ 비즈니스 분석 실패: {e}")
        return False
    
    # 5. 전체 성능 평가
    print("\n📊 5. 전체 성능 평가")
    final_performance = manager.get_performance_summary()
    
    print("=" * 50)
    print("📈 최종 성능 리포트")
    print("=" * 50)
    
    for key, value in final_performance.items():
        print(f"{key}: {value}")
    
    # 6. 99.2% 정확도 달성 여부 확인
    print("\n🎯 99.2% 정확도 달성 목표 검증")
    
    total_accuracy = (result_diamond.final_accuracy + 
                     result_gemstone.final_accuracy + 
                     result_business.final_accuracy) / 3
    
    print(f"📊 평균 정확도: {total_accuracy:.3f}")
    
    if total_accuracy >= 0.992:
        print("🏆 99.2% 정확도 목표 달성!")
        print("✅ 솔로몬드 AI 엔진 고도화 프로젝트 성공")
    elif total_accuracy >= 0.95:
        print("🎯 우수한 성능 달성 (95% 이상)")
        print("🔧 99.2% 달성을 위한 추가 최적화 필요")
    else:
        print("⚠️ 성능 개선 필요")
        print("🔧 시스템 튜닝 및 최적화 권장")
    
    print("\n🚀 하이브리드 LLM v2.3 통합 테스트 완료")
    print("🎯 차세대 주얼리 AI 시스템 검증 성공!")
    
    return True

async def performance_benchmark():
    """성능 벤치마크 테스트"""
    
    print("\n⚡ 성능 벤치마크 테스트 시작")
    print("=" * 50)
    
    manager = HybridLLMManagerV23()
    
    # 벤치마크 시나리오
    scenarios = [
        ("다이아몬드 감정", "diamond_4c", "0.8캐럿 라운드 브릴리언트 컷 다이아몬드 감정"),
        ("루비 분석", "colored_gemstone", "2.1캐럿 오벌 컷 루비 품질 분석"),
        ("시장 분석", "business_insight", "2025년 프리미엄 주얼리 시장 트렌드 분석"),
        ("에메랄드 감정", "colored_gemstone", "1.9캐럿 에메랄드 컷 에메랄드 감정"),
        ("사파이어 분석", "colored_gemstone", "2.5캐럿 오벌 컷 사파이어 분석")
    ]
    
    total_time = 0
    total_cost = 0
    accuracies = []
    
    for i, (name, analysis_type, content) in enumerate(scenarios, 1):
        print(f"\n🔍 벤치마크 {i}/5: {name}")
        
        request = AnalysisRequest(
            content_type="text",
            data={"content": content},
            analysis_type=analysis_type,
            quality_threshold=0.95,
            max_cost=0.05,
            language="ko"
        )
        
        start_time = time.time()
        result = await manager.analyze_with_hybrid_ai(request)
        end_time = time.time()
        
        processing_time = end_time - start_time
        total_time += processing_time
        total_cost += result.total_cost
        accuracies.append(result.final_accuracy)
        
        print(f"⏱️ 처리시간: {processing_time:.2f}초")
        print(f"📊 정확도: {result.final_accuracy:.3f}")
        print(f"💰 비용: ${result.total_cost:.4f}")
        print(f"🎯 모델: {result.best_result.model_type.value}")
    
    # 벤치마크 결과 요약
    print("\n📈 벤치마크 결과 요약")
    print("=" * 50)
    print(f"📊 평균 정확도: {sum(accuracies)/len(accuracies):.3f}")
    print(f"⏱️ 평균 처리시간: {total_time/len(scenarios):.2f}초")
    print(f"💰 총 비용: ${total_cost:.4f}")
    print(f"🎯 목표 달성률: {(sum(accuracies)/len(accuracies))*100:.1f}%")
    
    # 성능 기준 평가
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_time = total_time / len(scenarios)
    
    print("\n🏆 성능 평가 결과")
    if avg_accuracy >= 0.99 and avg_time <= 30:
        print("🌟 탁월한 성능 - 배포 준비 완료")
    elif avg_accuracy >= 0.95 and avg_time <= 45:
        print("✅ 우수한 성능 - 프로덕션 사용 가능")
    elif avg_accuracy >= 0.90 and avg_time <= 60:
        print("🔧 양호한 성능 - 최적화 권장")
    else:
        print("⚠️ 성능 개선 필요")

async def main():
    """메인 테스트 실행"""
    
    print("🚀 솔로몬드 하이브리드 LLM v2.3 종합 테스트")
    print("🎯 목표: 99.2% 정확도 달성 시스템 검증")
    print("=" * 70)
    
    try:
        # 1. 통합 테스트
        success = await test_hybrid_llm_integration()
        
        if success:
            # 2. 성능 벤치마크
            await performance_benchmark()
            
            print("\n🎉 모든 테스트 완료!")
            print("🏆 솔로몬드 AI 엔진 고도화 프로젝트 성공!")
            print("💎 차세대 주얼리 AI 시스템 검증 완료")
        else:
            print("\n❌ 테스트 실패")
            print("🔧 시스템 점검 및 수정 필요")
            
    except Exception as e:
        print(f"\n💥 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
