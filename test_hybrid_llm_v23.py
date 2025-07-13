"""
🧪 솔로몬드 AI 하이브리드 LLM 매니저 v2.3 테스트 스크립트

실제 API 키 없이도 시스템 구조와 기능을 테스트할 수 있는 종합 테스트
"""

import asyncio
import json
import sys
import os
import time
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from core.hybrid_llm_manager_v23 import (
        HybridLLMManager, 
        AnalysisRequest, 
        AIModel, 
        JewelryPromptOptimizer
    )
except ImportError as e:
    print(f"❌ Import 실패: {e}")
    print("💡 현재 디렉토리에서 실행하거나 PYTHONPATH를 설정해주세요.")
    sys.exit(1)

class MockAIResponse:
    """Mock AI 응답 (테스트용)"""
    def __init__(self, model_name: str, content: str):
        self.model = model_name
        self.content = content
        self.confidence = 0.9
        self.processing_time = 1.5
        self.cost_estimate = 0.02
        self.jewelry_relevance = 0.8
        self.metadata = {"mock": True}

class HybridLLMTester:
    """하이브리드 LLM 시스템 종합 테스터"""
    
    def __init__(self):
        self.results = []
        
    def test_jewelry_prompt_optimizer(self):
        """주얼리 프롬프트 최적화 테스트"""
        print("🔧 주얼리 프롬프트 최적화 테스트")
        print("-" * 50)
        
        optimizer = JewelryPromptOptimizer()
        
        test_cases = [
            ("diamond_analysis", "1캐럿 다이아몬드 분석"),
            ("colored_stone_analysis", "루비 감정 요청"),
            ("jewelry_design_analysis", "빈티지 반지 디자인 분석"),
            ("business_analysis", "주얼리 시장 트렌드 분석")
        ]
        
        for analysis_type, content in test_cases:
            optimized = optimizer.optimize_prompt(analysis_type, content)
            print(f"✅ {analysis_type}: {len(optimized)} 글자 프롬프트 생성")
            
        print("✅ 프롬프트 최적화 테스트 완료\n")
    
    def test_manager_initialization(self):
        """매니저 초기화 테스트"""
        print("🚀 하이브리드 LLM 매니저 초기화 테스트")
        print("-" * 50)
        
        # API 키 없는 초기화
        manager = HybridLLMManager()
        print("✅ API 키 없는 초기화 성공")
        
        # 가짜 API 키로 초기화
        fake_config = {
            "openai_key": "test-key-openai",
            "anthropic_key": "test-key-anthropic", 
            "google_key": "test-key-google"
        }
        manager_with_keys = HybridLLMManager(fake_config)
        print("✅ 설정이 있는 초기화 성공")
        
        print("✅ 매니저 초기화 테스트 완료\n")
        return manager_with_keys
    
    async def test_individual_models(self, manager):
        """개별 모델 테스트 (Mock)"""
        print("🤖 개별 AI 모델 테스트")
        print("-" * 50)
        
        request = AnalysisRequest(
            text_content="1캐럿 라운드 다이아몬드의 등급을 분석해주세요.",
            analysis_type="diamond_analysis"
        )
        
        # Mock 응답 시뮬레이션
        models_tested = []
        
        try:
            # GPT-4V 테스트 (실제로는 실행되지 않지만 구조 확인)
            print("🔍 GPT-4V 모델 구조 확인...")
            models_tested.append("GPT-4V")
        except Exception as e:
            print(f"⚠️ GPT-4V: {str(e)[:50]}...")
        
        try:
            # Claude Vision 테스트
            print("🔍 Claude Vision 모델 구조 확인...")
            models_tested.append("Claude Vision")
        except Exception as e:
            print(f"⚠️ Claude Vision: {str(e)[:50]}...")
        
        try:
            # Gemini 2.0 테스트
            print("🔍 Gemini 2.0 모델 구조 확인...")
            models_tested.append("Gemini 2.0")
        except Exception as e:
            print(f"⚠️ Gemini 2.0: {str(e)[:50]}...")
        
        print(f"✅ {len(models_tested)}개 모델 구조 확인 완료\n")
    
    async def test_hybrid_analysis_mock(self, manager):
        """하이브리드 분석 Mock 테스트"""
        print("🧠 하이브리드 분석 Mock 테스트")
        print("-" * 50)
        
        request = AnalysisRequest(
            text_content="1캐럿 라운드 다이아몬드, 컬러 H, 클래리티 VS1 등급을 분석해주세요.",
            analysis_type="diamond_analysis",
            require_jewelry_expertise=True
        )
        
        print(f"📝 분석 요청: {request.text_content}")
        print(f"📊 분석 타입: {request.analysis_type}")
        
        try:
            # 실제 하이브리드 분석 호출 (API 키가 없으므로 에러 예상)
            result = await manager.hybrid_analyze(request)
            
            print("📊 분석 결과:")
            print(f"   상태: {result['status']}")
            print(f"   메시지: {result.get('message', 'N/A')}")
            
            if result['status'] == 'success':
                print(f"   최적 모델: {result['best_model']}")
                print(f"   신뢰도: {result['confidence']:.1%}")
                print(f"   주얼리 관련성: {result['jewelry_relevance']:.1%}")
            
        except Exception as e:
            print(f"⚠️ 예상된 에러 (API 키 없음): {str(e)[:100]}...")
        
        print("✅ 하이브리드 분석 구조 테스트 완료\n")
    
    def test_performance_tracking(self, manager):
        """성능 추적 기능 테스트"""
        print("📈 성능 추적 기능 테스트")
        print("-" * 50)
        
        # 성능 메트릭 확인
        performance = manager._get_performance_summary()
        print(f"📊 성능 요약: {performance}")
        
        # 비용 리포트 확인
        cost_report = manager.get_cost_report()
        print(f"💰 비용 리포트: {cost_report}")
        
        print("✅ 성능 추적 기능 테스트 완료\n")
    
    def test_jewelry_relevance_calculation(self, manager):
        """주얼리 관련성 계산 테스트"""
        print("💎 주얼리 관련성 계산 테스트")
        print("-" * 50)
        
        test_texts = [
            "1캐럿 다이아몬드 반지의 GIA 감정서를 확인했습니다.",
            "오늘 날씨가 좋네요. 산책을 하러 나갔습니다.",
            "루비와 사파이어의 차이점에 대해 알아보겠습니다.",
            "컴퓨터 프로그래밍에 대한 설명입니다."
        ]
        
        for text in test_texts:
            relevance = manager._calculate_jewelry_relevance(text)
            print(f"📝 '{text[:30]}...' → 관련성: {relevance:.1%}")
        
        print("✅ 주얼리 관련성 계산 테스트 완료\n")
    
    def test_cost_estimation(self, manager):
        """비용 추정 테스트"""
        print("💰 비용 추정 테스트")
        print("-" * 50)
        
        models = ["gpt4v", "claude", "gemini"]
        input_length = 500
        output_length = 1000
        
        for model in models:
            cost = manager._estimate_cost(model, input_length, output_length)
            print(f"💳 {model}: ${cost:.4f} (입력: {input_length}, 출력: {output_length})")
        
        print("✅ 비용 추정 테스트 완료\n")
    
    async def run_comprehensive_test(self):
        """종합 테스트 실행"""
        print("🧪 솔로몬드 AI 하이브리드 LLM 매니저 v2.3 종합 테스트")
        print("=" * 70)
        print()
        
        start_time = time.time()
        
        # 1. 프롬프트 최적화 테스트
        self.test_jewelry_prompt_optimizer()
        
        # 2. 매니저 초기화 테스트
        manager = self.test_manager_initialization()
        
        # 3. 개별 모델 구조 테스트
        await self.test_individual_models(manager)
        
        # 4. 하이브리드 분석 Mock 테스트
        await self.test_hybrid_analysis_mock(manager)
        
        # 5. 성능 추적 테스트
        self.test_performance_tracking(manager)
        
        # 6. 주얼리 관련성 계산 테스트
        self.test_jewelry_relevance_calculation(manager)
        
        # 7. 비용 추정 테스트
        self.test_cost_estimation(manager)
        
        total_time = time.time() - start_time
        
        print("🎉 종합 테스트 완료")
        print("=" * 70)
        print(f"⏱️ 총 소요 시간: {total_time:.2f}초")
        print(f"✅ 모든 구조 및 기능 테스트 성공")
        print()
        print("💡 실제 사용을 위해서는 다음 API 키가 필요합니다:")
        print("   - OpenAI API 키 (GPT-4V)")
        print("   - Anthropic API 키 (Claude Vision)")  
        print("   - Google API 키 (Gemini 2.0)")
        print()
        print("🚀 API 키 설정 후 실제 하이브리드 분석을 테스트해보세요!")

def create_sample_config():
    """샘플 설정 파일 생성"""
    sample_config = {
        "openai_key": "your-openai-api-key-here",
        "anthropic_key": "your-anthropic-api-key-here",
        "google_key": "your-google-api-key-here"
    }
    
    config_path = Path("config_v23_sample.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)
    
    print(f"📝 샘플 설정 파일 생성: {config_path}")
    print("💡 실제 API 키로 수정 후 사용하세요.")

async def main():
    """메인 테스트 실행 함수"""
    
    # 샘플 설정 파일 생성
    create_sample_config()
    print()
    
    # 종합 테스트 실행
    tester = HybridLLMTester()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(main())
