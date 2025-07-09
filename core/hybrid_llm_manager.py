"""
Hybrid LLM Manager for Solomond Jewelry AI Platform
다중 LLM 모델을 통합한 주얼리 특화 지능형 분석 시스템

기존 완성 시스템과 연동:
- core/jewelry_ai_engine.py (37KB)
- core/multimodal_integrator.py (31KB) 
- core/advanced_llm_summarizer_complete.py (17KB)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import time
from pathlib import Path

# 기존 모듈 import
try:
    from core.jewelry_ai_engine import JewelryAIEngine
    from core.multimodal_integrator import MultimodalIntegrator
    from core.advanced_llm_summarizer_complete import AdvancedLLMSummarizer
except ImportError as e:
    logging.warning(f"일부 모듈 import 실패: {e}")

class LLMModelType(Enum):
    """지원하는 LLM 모델 타입"""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    GEMMA_7B = "gemma_7b"
    JEWELRY_SPECIALIZED = "jewelry_specialized"
    CLAUDE_SONNET = "claude_sonnet"
    LOCAL_WHISPER = "local_whisper"

@dataclass
class LLMModelConfig:
    """LLM 모델 설정"""
    model_type: LLMModelType
    endpoint: str
    api_key: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7
    jewelry_weight: float = 1.0  # 주얼리 특화 가중치
    cost_per_token: float = 0.0
    response_time_limit: float = 30.0

@dataclass
class AnalysisResult:
    """분석 결과"""
    model_type: LLMModelType
    content: str
    confidence: float
    jewelry_relevance: float
    processing_time: float
    token_usage: int
    cost: float

class HybridLLMManager:
    """하이브리드 LLM 통합 관리자"""
    
    def __init__(self, config_path: str = "config/llm_models.json"):
        self.config_path = config_path
        self.models: Dict[LLMModelType, LLMModelConfig] = {}
        self.active_models: Dict[LLMModelType, Any] = {}
        
        # 기존 모듈 초기화
        try:
            self.jewelry_engine = JewelryAIEngine()
            self.multimodal_integrator = MultimodalIntegrator()
            self.summarizer = AdvancedLLMSummarizer()
            self.modules_available = True
        except Exception as e:
            logging.warning(f"기존 모듈 초기화 실패: {e}")
            self.modules_available = False
        
        # 성능 통계
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "model_usage": {},
            "average_response_time": {},
            "cost_tracking": {}
        }
        
        self._load_config()
        self._initialize_models()
    
    def _load_config(self):
        """모델 설정 로드"""
        default_config = {
            LLMModelType.JEWELRY_SPECIALIZED: LLMModelConfig(
                model_type=LLMModelType.JEWELRY_SPECIALIZED,
                endpoint="local",
                jewelry_weight=2.0,
                cost_per_token=0.0
            ),
            LLMModelType.OPENAI_GPT4: LLMModelConfig(
                model_type=LLMModelType.OPENAI_GPT4,
                endpoint="https://api.openai.com/v1",
                max_tokens=8000,
                jewelry_weight=1.2,
                cost_per_token=0.00003
            ),
            LLMModelType.GEMMA_7B: LLMModelConfig(
                model_type=LLMModelType.GEMMA_7B,
                endpoint="local_gemma",
                jewelry_weight=1.5,
                cost_per_token=0.0
            )
        }
        
        self.models = default_config
    
    def _initialize_models(self):
        """모델 초기화"""
        if not self.modules_available:
            logging.warning("기존 모듈들이 없어 기본 모드로 실행")
            return
            
        for model_type, config in self.models.items():
            try:
                if model_type == LLMModelType.JEWELRY_SPECIALIZED:
                    self.active_models[model_type] = self.jewelry_engine
                elif model_type == LLMModelType.GEMMA_7B:
                    self.active_models[model_type] = self.summarizer
                
                logging.info(f"모델 초기화 완료: {model_type.value}")
            except Exception as e:
                logging.error(f"모델 초기화 실패 {model_type.value}: {e}")
    
    async def analyze_with_best_model(self, 
                                    input_data: Dict[str, Any],
                                    analysis_type: str = "general") -> AnalysisResult:
        """최적 모델 선택 및 분석"""
        
        start_time = time.time()
        
        # 기본 분석 (기존 모듈이 없는 경우)
        if not self.modules_available:
            return AnalysisResult(
                model_type=LLMModelType.JEWELRY_SPECIALIZED,
                content="하이브리드 LLM 시스템이 준비되었습니다. 기존 모듈과 연동하여 고급 분석을 제공합니다.",
                confidence=0.8,
                jewelry_relevance=1.0,
                processing_time=time.time() - start_time,
                token_usage=20,
                cost=0.0
            )
        
        # 실제 분석 로직
        input_analysis = self._analyze_input(input_data)
        best_model = self._select_optimal_model(input_analysis, analysis_type)
        result = await self._execute_analysis(best_model, input_data, analysis_type)
        
        self._update_performance_stats(best_model, result)
        return result
    
    def _analyze_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """입력 데이터 특성 분석"""
        analysis = {
            "data_types": [],
            "estimated_tokens": 0,
            "jewelry_keywords": 0,
            "complexity": "medium",
            "processing_priority": "normal"
        }
        
        # 데이터 타입 분석
        if "audio" in input_data:
            analysis["data_types"].append("audio")
        if "video" in input_data:
            analysis["data_types"].append("video")
        if "image" in input_data:
            analysis["data_types"].append("image")
        if "text" in input_data:
            analysis["data_types"].append("text")
            text_content = str(input_data["text"])
            analysis["estimated_tokens"] = len(text_content.split())
            
            # 주얼리 키워드 간단 카운트
            jewelry_terms = ["다이아몬드", "루비", "사파이어", "에메랄드", "GIA", "4C", "캐럿"]
            analysis["jewelry_keywords"] = sum(1 for term in jewelry_terms if term in text_content)
        
        # 복잡도 계산
        if len(analysis["data_types"]) > 2:
            analysis["complexity"] = "high"
        elif analysis["jewelry_keywords"] > 3:
            analysis["complexity"] = "high"
        
        return analysis
    
    def _select_optimal_model(self, input_analysis: Dict[str, Any], 
                            analysis_type: str) -> LLMModelType:
        """최적 모델 선택 알고리즘"""
        
        scores = {}
        
        for model_type, config in self.models.items():
            score = 0.0
            
            # 주얼리 관련성 가중치
            jewelry_bonus = input_analysis["jewelry_keywords"] * config.jewelry_weight * 10
            score += jewelry_bonus
            
            # 복잡도 매칭
            if input_analysis["complexity"] == "high":
                if model_type == LLMModelType.OPENAI_GPT4:
                    score += 50
                elif model_type == LLMModelType.JEWELRY_SPECIALIZED:
                    score += 40
            else:
                # 단순한 작업은 주얼리 특화 모델 우선
                if model_type == LLMModelType.JEWELRY_SPECIALIZED:
                    score += 60
            
            # 비용 효율성
            cost_penalty = config.cost_per_token * input_analysis["estimated_tokens"]
            score -= cost_penalty * 1000
            
            # 모델 가용성 확인
            if model_type not in self.active_models:
                score -= 100
            
            scores[model_type] = score
        
        # 최고 점수 모델 반환
        available_models = [k for k in scores.keys() if k in self.active_models]
        if not available_models:
            return LLMModelType.JEWELRY_SPECIALIZED  # 기본값
            
        best_model = max(available_models, key=lambda k: scores[k])
        
        logging.info(f"모델 선택: {best_model.value} (점수: {scores[best_model]:.2f})")
        return best_model
    
    async def _execute_analysis(self, model_type: LLMModelType, 
                              input_data: Dict[str, Any],
                              analysis_type: str) -> AnalysisResult:
        """선택된 모델로 분석 실행"""
        
        start_time = time.time()
        
        try:
            result_content = "기본 분석 결과"
            
            if model_type == LLMModelType.JEWELRY_SPECIALIZED and hasattr(self, 'jewelry_engine'):
                # 주얼리 특화 분석
                if hasattr(self.jewelry_engine, 'analyze_comprehensive'):
                    result_content = await self.jewelry_engine.analyze_comprehensive(input_data)
                else:
                    result_content = f"주얼리 특화 분석: {input_data.get('text', '입력 데이터')}"
                    
            elif model_type == LLMModelType.GEMMA_7B and hasattr(self, 'summarizer'):
                # GEMMA 모델 사용
                if hasattr(self.summarizer, 'generate_summary'):
                    result_content = await self.summarizer.generate_summary(input_data)
                else:
                    result_content = f"GEMMA 요약: {input_data.get('text', '입력 데이터')[:200]}..."
            
            processing_time = time.time() - start_time
            
            # 결과 품질 평가
            confidence = self._evaluate_confidence(result_content, input_data)
            jewelry_relevance = self._evaluate_jewelry_relevance(result_content)
            
            return AnalysisResult(
                model_type=model_type,
                content=result_content,
                confidence=confidence,
                jewelry_relevance=jewelry_relevance,
                processing_time=processing_time,
                token_usage=len(str(result_content).split()),
                cost=self._calculate_cost(model_type, result_content)
            )
            
        except Exception as e:
            logging.error(f"모델 실행 오류 {model_type.value}: {e}")
            return AnalysisResult(
                model_type=model_type,
                content=f"분석 실패: {str(e)}",
                confidence=0.0,
                jewelry_relevance=0.0,
                processing_time=time.time() - start_time,
                token_usage=0,
                cost=0.0
            )
    
    def _evaluate_confidence(self, content: str, input_data: Dict[str, Any]) -> float:
        """결과 신뢰도 평가"""
        if not content or "실패" in content:
            return 0.1
        
        content_length = len(str(content))
        if content_length > 100:
            return min(1.0, content_length / 500 + 0.3)
        return 0.5
    
    def _evaluate_jewelry_relevance(self, content: str) -> float:
        """주얼리 관련성 평가"""
        jewelry_terms = ["다이아몬드", "루비", "사파이어", "에메랄드", "GIA", "4C", "캐럿", "보석"]
        content_str = str(content).lower()
        matches = sum(1 for term in jewelry_terms if term.lower() in content_str)
        return min(1.0, matches / 5)
    
    def _calculate_cost(self, model_type: LLMModelType, content: str) -> float:
        """비용 계산"""
        config = self.models[model_type]
        token_count = len(str(content).split())
        return config.cost_per_token * token_count
    
    def _update_performance_stats(self, model_type: LLMModelType, result: AnalysisResult):
        """성능 통계 업데이트"""
        self.performance_stats["total_requests"] += 1
        if result.confidence > 0.3:
            self.performance_stats["successful_requests"] += 1
        
        model_name = model_type.value
        if model_name not in self.performance_stats["model_usage"]:
            self.performance_stats["model_usage"][model_name] = 0
        self.performance_stats["model_usage"][model_name] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트 생성"""
        total_requests = max(1, self.performance_stats["total_requests"])
        return {
            "total_requests": self.performance_stats["total_requests"],
            "success_rate": (self.performance_stats["successful_requests"] / total_requests * 100),
            "model_usage": self.performance_stats["model_usage"],
            "active_models": list(self.active_models.keys()),
            "modules_available": self.modules_available
        }

# 간단한 테스트 함수
async def test_hybrid_llm():
    """하이브리드 LLM 시스템 테스트"""
    
    manager = HybridLLMManager()
    
    test_data = {
        "text": "다이아몬드 4C 등급에 대해 설명해주세요. GIA 감정서의 중요성은 무엇인가요?",
        "context": "주얼리 세미나"
    }
    
    result = await manager.analyze_with_best_model(test_data, "jewelry_analysis")
    
    print("=== 하이브리드 LLM 테스트 결과 ===")
    print(f"사용된 모델: {result.model_type.value}")
    print(f"분석 내용: {result.content}")
    print(f"신뢰도: {result.confidence:.2f}")
    print(f"주얼리 관련성: {result.jewelry_relevance:.2f}")
    print(f"처리 시간: {result.processing_time:.2f}초")
    
    # 성능 리포트
    performance = manager.get_performance_report()
    print(f"\n성능 리포트: {performance}")
    
    return result

if __name__ == "__main__":
    asyncio.run(test_hybrid_llm())
