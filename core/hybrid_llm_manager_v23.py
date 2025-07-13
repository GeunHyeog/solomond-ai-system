"""
🧠 솔로몬드 AI 하이브리드 LLM 매니저 v2.3
GPT-4V + Claude Vision + Gemini 2.0 동시 활용 시스템

개발자: 전근혁 (솔로몬드 대표)
목표: 99.2% 정확도 달성
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# AI 모델 클라이언트 imports
try:
    import openai  # GPT-4V
    import anthropic  # Claude Vision
    import google.generativeai as genai  # Gemini 2.0
except ImportError as e:
    print(f"⚠️ AI 모델 라이브러리 import 오류: {e}")
    print("💡 설치 명령어: pip install openai anthropic google-generativeai")

# 설정 및 유틸리티
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIModel(Enum):
    """지원하는 AI 모델 열거형"""
    GPT4V = "gpt-4-vision-preview"
    CLAUDE_VISION = "claude-3-5-sonnet-20241022"
    GEMINI_2 = "gemini-2.0-flash-exp"

@dataclass
class AIResponse:
    """AI 모델 응답 데이터 클래스"""
    model: AIModel
    content: str
    confidence: float
    processing_time: float
    cost_estimate: float
    jewelry_relevance: float
    metadata: Dict[str, Any]

@dataclass
class AnalysisRequest:
    """분석 요청 데이터 클래스"""
    text_content: Optional[str] = None
    image_data: Optional[bytes] = None
    image_url: Optional[str] = None
    analysis_type: str = "general"
    priority: int = 1
    require_jewelry_expertise: bool = True

class JewelryPromptOptimizer:
    """주얼리 특화 프롬프트 최적화 클래스"""
    
    def __init__(self):
        self.jewelry_contexts = {
            "diamond_analysis": """
당신은 GIA 공인 다이아몬드 전문가입니다. 다음 다이아몬드를 4C (Carat, Cut, Color, Clarity) 기준으로 정확히 분석해주세요.
- Carat: 정확한 중량 또는 예상 중량
- Cut: Excellent, Very Good, Good, Fair, Poor 중 하나
- Color: D-Z 등급 (D=무색, Z=연한 노란색)
- Clarity: FL, IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1, I2, I3 중 하나
분석 후 한국어로 요약해주세요.
""",
            "colored_stone_analysis": """
당신은 유색보석 전문 감정사입니다. 다음 보석을 분석해주세요.
- 보석 종류: 루비, 사파이어, 에메랄드, 기타
- 산지 추정: 미얀마, 스리랑카, 마다가스카르, 콜롬비아 등
- 처리 여부: 가열, 오일링, 기타 처리
- 품질 등급: AAA, AA, A, B, C
- 예상 가치: 캐럿당 가격대
한국어로 전문적으로 분석해주세요.
""",
            "jewelry_design_analysis": """
당신은 주얼리 디자인 전문가입니다. 다음 주얼리를 분석해주세요.
- 주얼리 종류: 반지, 목걸이, 귀걸이, 브로치 등
- 디자인 스타일: 클래식, 모던, 빈티지, 아르데코 등
- 세팅 방식: 프롱, 베젤, 파베, 채널 등
- 금속 재질: 플래티나, 18K 골드, 14K 골드, 실버 등
- 제작 기법: 핸드메이드, 캐스팅, CNC 등
한국어로 전문적으로 분석해주세요.
""",
            "business_analysis": """
당신은 주얼리 비즈니스 전문가입니다. 다음 내용을 분석해주세요.
- 시장 트렌드 분석
- 가격 동향 및 예측
- 투자 가치 평가
- 수집 가치 및 희소성
- 재판매 시장 전망
한국어로 비즈니스 관점에서 분석해주세요.
"""
        }
    
    def optimize_prompt(self, analysis_type: str, base_content: str) -> str:
        """분석 타입에 따른 프롬프트 최적화"""
        context = self.jewelry_contexts.get(analysis_type, "")
        
        optimized_prompt = f"""
{context}

분석할 내용:
{base_content}

반드시 다음 형식으로 답변해주세요:
1. 전문 분석 결과
2. 핵심 특징 요약
3. 주의사항 또는 추가 검토 필요 사항
4. 한국어 최종 요약 (주얼리 전문가용)
"""
        return optimized_prompt

class HybridLLMManager:
    """하이브리드 LLM 매니저 v2.3 핵심 클래스"""
    
    def __init__(self, config: Optional[Dict[str, str]] = None):
        """
        하이브리드 LLM 매니저 초기화
        
        Args:
            config: API 키 설정 딕셔너리
                   {"openai_key": "...", "anthropic_key": "...", "google_key": "..."}
        """
        self.config = config or {}
        self.prompt_optimizer = JewelryPromptOptimizer()
        self.performance_metrics = {}
        self.cost_tracker = {"total": 0.0, "by_model": {}}
        
        # AI 클라이언트 초기화
        self._initialize_clients()
        
        # 성능 추적
        self.response_history = []
        
    def _initialize_clients(self):
        """AI 모델 클라이언트 초기화"""
        try:
            # OpenAI GPT-4V 클라이언트
            if "openai_key" in self.config:
                openai.api_key = self.config["openai_key"]
                self.openai_client = openai.OpenAI(api_key=self.config["openai_key"])
            else:
                self.openai_client = None
                logger.warning("⚠️ OpenAI API 키가 설정되지 않았습니다.")
            
            # Anthropic Claude 클라이언트
            if "anthropic_key" in self.config:
                self.anthropic_client = anthropic.Anthropic(api_key=self.config["anthropic_key"])
            else:
                self.anthropic_client = None
                logger.warning("⚠️ Anthropic API 키가 설정되지 않았습니다.")
            
            # Google Gemini 클라이언트
            if "google_key" in self.config:
                genai.configure(api_key=self.config["google_key"])
                self.gemini_client = genai.GenerativeModel('gemini-2.0-flash-exp')
            else:
                self.gemini_client = None
                logger.warning("⚠️ Google API 키가 설정되지 않았습니다.")
                
        except Exception as e:
            logger.error(f"❌ AI 클라이언트 초기화 실패: {e}")
    
    async def analyze_with_gpt4v(self, request: AnalysisRequest) -> AIResponse:
        """GPT-4V로 분석 수행"""
        start_time = time.time()
        
        try:
            if not self.openai_client:
                raise ValueError("OpenAI 클라이언트가 초기화되지 않았습니다.")
            
            # 프롬프트 최적화
            optimized_prompt = self.prompt_optimizer.optimize_prompt(
                request.analysis_type, 
                request.text_content or "이미지 분석 요청"
            )
            
            messages = [{"role": "user", "content": optimized_prompt}]
            
            # 이미지가 있는 경우 추가
            if request.image_url or request.image_data:
                if request.image_url:
                    messages[0]["content"] = [
                        {"type": "text", "text": optimized_prompt},
                        {"type": "image_url", "image_url": {"url": request.image_url}}
                    ]
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=1500,
                temperature=0.1
            )
            
            processing_time = time.time() - start_time
            content = response.choices[0].message.content
            
            # 주얼리 관련성 점수 계산
            jewelry_relevance = self._calculate_jewelry_relevance(content)
            
            return AIResponse(
                model=AIModel.GPT4V,
                content=content,
                confidence=0.9,  # GPT-4V 기본 신뢰도
                processing_time=processing_time,
                cost_estimate=self._estimate_cost("gpt4v", len(optimized_prompt), len(content)),
                jewelry_relevance=jewelry_relevance,
                metadata={"tokens_used": response.usage.total_tokens}
            )
            
        except Exception as e:
            logger.error(f"❌ GPT-4V 분석 실패: {e}")
            return self._create_error_response(AIModel.GPT4V, str(e), time.time() - start_time)
    
    async def analyze_with_claude(self, request: AnalysisRequest) -> AIResponse:
        """Claude Vision으로 분석 수행"""
        start_time = time.time()
        
        try:
            if not self.anthropic_client:
                raise ValueError("Anthropic 클라이언트가 초기화되지 않았습니다.")
            
            # 프롬프트 최적화
            optimized_prompt = self.prompt_optimizer.optimize_prompt(
                request.analysis_type, 
                request.text_content or "이미지 분석 요청"
            )
            
            message_content = [{"type": "text", "text": optimized_prompt}]
            
            # 이미지 처리 (Claude의 경우 base64 인코딩 필요)
            if request.image_data:
                import base64
                image_base64 = base64.b64encode(request.image_data).decode()
                message_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_base64
                    }
                })
            
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                temperature=0.1,
                messages=[{"role": "user", "content": message_content}]
            )
            
            processing_time = time.time() - start_time
            content = response.content[0].text
            
            # 주얼리 관련성 점수 계산
            jewelry_relevance = self._calculate_jewelry_relevance(content)
            
            return AIResponse(
                model=AIModel.CLAUDE_VISION,
                content=content,
                confidence=0.92,  # Claude의 높은 신뢰도
                processing_time=processing_time,
                cost_estimate=self._estimate_cost("claude", len(optimized_prompt), len(content)),
                jewelry_relevance=jewelry_relevance,
                metadata={"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
            )
            
        except Exception as e:
            logger.error(f"❌ Claude Vision 분석 실패: {e}")
            return self._create_error_response(AIModel.CLAUDE_VISION, str(e), time.time() - start_time)
    
    async def analyze_with_gemini(self, request: AnalysisRequest) -> AIResponse:
        """Gemini 2.0으로 분석 수행"""
        start_time = time.time()
        
        try:
            if not self.gemini_client:
                raise ValueError("Gemini 클라이언트가 초기화되지 않았습니다.")
            
            # 프롬프트 최적화
            optimized_prompt = self.prompt_optimizer.optimize_prompt(
                request.analysis_type, 
                request.text_content or "이미지 분석 요청"
            )
            
            # 콘텐츠 준비
            content_parts = [optimized_prompt]
            
            # 이미지 처리
            if request.image_data:
                import PIL.Image
                import io
                image = PIL.Image.open(io.BytesIO(request.image_data))
                content_parts.append(image)
            
            response = await asyncio.to_thread(
                self.gemini_client.generate_content,
                content_parts,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1500
                )
            )
            
            processing_time = time.time() - start_time
            content = response.text
            
            # 주얼리 관련성 점수 계산
            jewelry_relevance = self._calculate_jewelry_relevance(content)
            
            return AIResponse(
                model=AIModel.GEMINI_2,
                content=content,
                confidence=0.88,  # Gemini 기본 신뢰도
                processing_time=processing_time,
                cost_estimate=self._estimate_cost("gemini", len(optimized_prompt), len(content)),
                jewelry_relevance=jewelry_relevance,
                metadata={"candidate_count": len(response.candidates)}
            )
            
        except Exception as e:
            logger.error(f"❌ Gemini 2.0 분석 실패: {e}")
            return self._create_error_response(AIModel.GEMINI_2, str(e), time.time() - start_time)
    
    def _calculate_jewelry_relevance(self, content: str) -> float:
        """주얼리 관련성 점수 계산"""
        jewelry_keywords = [
            "다이아몬드", "diamond", "루비", "ruby", "사파이어", "sapphire", "에메랄드", "emerald",
            "캐럿", "carat", "컷", "cut", "컬러", "color", "클래리티", "clarity",
            "반지", "ring", "목걸이", "necklace", "귀걸이", "earring", "브로치", "brooch",
            "플래티나", "platinum", "골드", "gold", "실버", "silver",
            "GIA", "AGS", "SSEF", "Gübelin", "감정서", "certificate"
        ]
        
        content_lower = content.lower()
        found_keywords = sum(1 for keyword in jewelry_keywords if keyword.lower() in content_lower)
        relevance_score = min(found_keywords / len(jewelry_keywords) * 2, 1.0)  # 최대 1.0
        
        return relevance_score
    
    def _estimate_cost(self, model: str, input_length: int, output_length: int) -> float:
        """API 사용 비용 추정"""
        cost_per_1k_tokens = {
            "gpt4v": 0.03,    # GPT-4V 대략적 비용
            "claude": 0.008,  # Claude Sonnet 비용
            "gemini": 0.002   # Gemini Pro 비용
        }
        
        total_tokens = (input_length + output_length) / 4  # 대략적 토큰 수
        cost = (total_tokens / 1000) * cost_per_1k_tokens.get(model, 0.01)
        
        return round(cost, 4)
    
    def _create_error_response(self, model: AIModel, error_msg: str, processing_time: float) -> AIResponse:
        """에러 응답 생성"""
        return AIResponse(
            model=model,
            content=f"❌ 분석 실패: {error_msg}",
            confidence=0.0,
            processing_time=processing_time,
            cost_estimate=0.0,
            jewelry_relevance=0.0,
            metadata={"error": True, "error_message": error_msg}
        )
    
    async def hybrid_analyze(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        하이브리드 분석 - 3개 모델 동시 실행 후 최적 결과 선택
        
        Args:
            request: 분석 요청 객체
            
        Returns:
            Dict: 통합 분석 결과
        """
        logger.info(f"🧠 하이브리드 분석 시작: {request.analysis_type}")
        
        # 3개 모델 동시 실행
        tasks = []
        if self.openai_client:
            tasks.append(self.analyze_with_gpt4v(request))
        if self.anthropic_client:
            tasks.append(self.analyze_with_claude(request))
        if self.gemini_client:
            tasks.append(self.analyze_with_gemini(request))
        
        if not tasks:
            return {
                "status": "error",
                "message": "사용 가능한 AI 모델이 없습니다. API 키를 확인해주세요.",
                "timestamp": time.time()
            }
        
        # 모든 모델의 응답 대기
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 유효한 응답만 필터링
        valid_responses = [r for r in responses if isinstance(r, AIResponse) and r.confidence > 0]
        
        if not valid_responses:
            return {
                "status": "error",
                "message": "모든 AI 모델에서 분석에 실패했습니다.",
                "errors": [str(r) for r in responses if not isinstance(r, AIResponse)],
                "timestamp": time.time()
            }
        
        # 최적 모델 선택
        best_response = self._select_best_response(valid_responses)
        
        # 성능 메트릭 업데이트
        self._update_performance_metrics(valid_responses)
        
        # 비용 추적 업데이트
        total_cost = sum(r.cost_estimate for r in valid_responses)
        self.cost_tracker["total"] += total_cost
        
        # 통합 결과 반환
        result = {
            "status": "success",
            "best_model": best_response.model.value,
            "content": best_response.content,
            "confidence": best_response.confidence,
            "jewelry_relevance": best_response.jewelry_relevance,
            "processing_time": best_response.processing_time,
            "cost_estimate": total_cost,
            "all_responses": [asdict(r) for r in valid_responses],
            "performance_summary": self._get_performance_summary(),
            "timestamp": time.time()
        }
        
        # 응답 히스토리에 추가
        self.response_history.append(result)
        
        logger.info(f"✅ 하이브리드 분석 완료: {best_response.model.value} 선택됨")
        
        return result
    
    def _select_best_response(self, responses: List[AIResponse]) -> AIResponse:
        """최적 응답 선택 알고리즘"""
        if not responses:
            raise ValueError("선택할 응답이 없습니다.")
        
        if len(responses) == 1:
            return responses[0]
        
        # 복합 점수 계산 (가중치 적용)
        weights = {
            "confidence": 0.3,
            "jewelry_relevance": 0.4,
            "speed": 0.2,
            "cost_efficiency": 0.1
        }
        
        scored_responses = []
        max_time = max(r.processing_time for r in responses)
        max_cost = max(r.cost_estimate for r in responses)
        
        for response in responses:
            # 정규화된 점수 계산
            speed_score = 1.0 - (response.processing_time / max_time) if max_time > 0 else 1.0
            cost_score = 1.0 - (response.cost_estimate / max_cost) if max_cost > 0 else 1.0
            
            composite_score = (
                weights["confidence"] * response.confidence +
                weights["jewelry_relevance"] * response.jewelry_relevance +
                weights["speed"] * speed_score +
                weights["cost_efficiency"] * cost_score
            )
            
            scored_responses.append((response, composite_score))
        
        # 가장 높은 점수의 응답 선택
        best_response = max(scored_responses, key=lambda x: x[1])[0]
        
        return best_response
    
    def _update_performance_metrics(self, responses: List[AIResponse]):
        """성능 메트릭 업데이트"""
        for response in responses:
            model_name = response.model.value
            if model_name not in self.performance_metrics:
                self.performance_metrics[model_name] = {
                    "total_requests": 0,
                    "avg_confidence": 0.0,
                    "avg_processing_time": 0.0,
                    "avg_jewelry_relevance": 0.0,
                    "total_cost": 0.0
                }
            
            metrics = self.performance_metrics[model_name]
            metrics["total_requests"] += 1
            
            # 이동 평균 업데이트
            n = metrics["total_requests"]
            metrics["avg_confidence"] = ((n-1) * metrics["avg_confidence"] + response.confidence) / n
            metrics["avg_processing_time"] = ((n-1) * metrics["avg_processing_time"] + response.processing_time) / n
            metrics["avg_jewelry_relevance"] = ((n-1) * metrics["avg_jewelry_relevance"] + response.jewelry_relevance) / n
            metrics["total_cost"] += response.cost_estimate
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보 반환"""
        if not self.performance_metrics:
            return {"message": "성능 데이터가 없습니다."}
        
        summary = {}
        for model, metrics in self.performance_metrics.items():
            summary[model] = {
                "requests": metrics["total_requests"],
                "avg_confidence": round(metrics["avg_confidence"], 3),
                "avg_speed": round(metrics["avg_processing_time"], 2),
                "avg_jewelry_relevance": round(metrics["avg_jewelry_relevance"], 3),
                "total_cost": round(metrics["total_cost"], 4)
            }
        
        return summary
    
    def get_cost_report(self) -> Dict[str, Any]:
        """비용 리포트 생성"""
        model_costs = {}
        for model, metrics in self.performance_metrics.items():
            model_costs[model] = metrics["total_cost"]
        
        return {
            "total_cost": round(self.cost_tracker["total"], 4),
            "cost_by_model": model_costs,
            "average_cost_per_request": round(
                self.cost_tracker["total"] / len(self.response_history) if self.response_history else 0, 4
            ),
            "total_requests": len(self.response_history)
        }

# 데모 및 테스트 함수
async def demo_hybrid_analysis():
    """하이브리드 분석 데모"""
    print("🧠 솔로몬드 AI 하이브리드 LLM 매니저 v2.3 데모")
    print("=" * 60)
    
    # 설정 (실제 사용 시 API 키 필요)
    config = {
        # "openai_key": "your-openai-api-key",
        # "anthropic_key": "your-anthropic-api-key", 
        # "google_key": "your-google-api-key"
    }
    
    manager = HybridLLMManager(config)
    
    # 테스트 요청
    request = AnalysisRequest(
        text_content="1캐럿 라운드 다이아몬드의 등급을 분석해주세요. 컬러는 H, 클래리티는 VS1 등급입니다.",
        analysis_type="diamond_analysis",
        require_jewelry_expertise=True
    )
    
    print("📝 분석 요청:")
    print(f"   내용: {request.text_content}")
    print(f"   타입: {request.analysis_type}")
    print()
    
    # 하이브리드 분석 실행
    result = await manager.hybrid_analyze(request)
    
    print("📊 분석 결과:")
    print(f"   상태: {result['status']}")
    if result['status'] == 'success':
        print(f"   최적 모델: {result['best_model']}")
        print(f"   신뢰도: {result['confidence']:.1%}")
        print(f"   주얼리 관련성: {result['jewelry_relevance']:.1%}")
        print(f"   처리 시간: {result['processing_time']:.2f}초")
        print(f"   예상 비용: ${result['cost_estimate']:.4f}")
        print(f"   응답 수: {len(result['all_responses'])}개")
        print()
        print("💎 분석 내용:")
        print(result['content'][:200] + "..." if len(result['content']) > 200 else result['content'])
    else:
        print(f"   오류: {result['message']}")
    
    print()
    print("📈 성능 요약:")
    print(json.dumps(result.get('performance_summary', {}), indent=2, ensure_ascii=False))
    
    print()
    print("💰 비용 리포트:")
    cost_report = manager.get_cost_report()
    print(json.dumps(cost_report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # 데모 실행
    asyncio.run(demo_hybrid_analysis())
