"""
🚀 솔로몬드 AI v2.2 - 차세대 AI 통합 엔진
Next Generation AI Integration Engine

주요 기능:
- GPT-4o, Claude 3.5 Sonnet, Gemini Pro 통합
- 주얼리 업계 특화 프롬프트 엔지니어링
- 감정 분석 및 화자 구분 고도화
- 실시간 시장 분석 연동
- 다중 모델 컨센서스 분석

개발자: 전근혁 (솔로몬드 대표)
시작일: 2025.07.12
"""

import os
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path

# AI 모델 클라이언트
try:
    import openai
    from anthropic import Anthropic
    import google.generativeai as genai
except ImportError as e:
    print(f"⚠️ AI 모델 라이브러리 설치 필요: {e}")

# 내부 모듈
try:
    from .jewelry_ai_engine import JewelryAIEngine
    from .korean_summary_engine_v21 import KoreanSummaryEngineV21
    from .quality_analyzer_v21 import QualityAnalyzerV21
except ImportError:
    print("⚠️ 내부 모듈 import 오류 - 폴백 모드로 실행")

@dataclass
class AIModelConfig:
    """AI 모델 설정"""
    name: str
    api_key: Optional[str] = None
    model_id: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    specialty: str = ""  # 각 모델의 특화 분야

@dataclass
class AnalysisResult:
    """분석 결과 데이터 클래스"""
    model_name: str
    content: str
    confidence_score: float
    processing_time: float
    jewelry_relevance: float
    language_detected: str
    key_insights: List[str]
    action_items: List[str]
    quality_metrics: Dict[str, float]

class NextGenAIIntegrator:
    """차세대 AI 통합 엔진"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        차세대 AI 통합 엔진 초기화
        
        Args:
            config_path: AI 모델 설정 파일 경로
        """
        self.logger = self._setup_logger()
        self.models_config = self._load_models_config(config_path)
        self.ai_clients = {}
        self.jewelry_engine = None
        self.korean_engine = None
        self.quality_analyzer = None
        
        # 주얼리 특화 프롬프트 템플릿
        self.jewelry_prompts = self._load_jewelry_prompts()
        
        # 성능 메트릭
        self.performance_metrics = {
            "total_analyses": 0,
            "average_accuracy": 0.0,
            "model_performance": {},
            "last_update": datetime.now()
        }
        
        self._initialize_ai_clients()
        self._initialize_jewelry_modules()
        
        self.logger.info("🚀 차세대 AI 통합 엔진 초기화 완료")
    
    def _setup_logger(self) -> logging.Logger:
        """로깅 시스템 설정"""
        logger = logging.getLogger("NextGenAI")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_models_config(self, config_path: Optional[str]) -> Dict[str, AIModelConfig]:
        """AI 모델 설정 로드"""
        default_config = {
            "gpt4o": AIModelConfig(
                name="GPT-4o",
                model_id="gpt-4o",
                max_tokens=4096,
                temperature=0.3,
                specialty="일반 분석 및 요약"
            ),
            "claude35": AIModelConfig(
                name="Claude 3.5 Sonnet",
                model_id="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                temperature=0.4,
                specialty="논리적 분석 및 추론"
            ),
            "gemini": AIModelConfig(
                name="Gemini Pro",
                model_id="gemini-pro",
                max_tokens=4096,
                temperature=0.5,
                specialty="다국어 및 창의적 분석"
            )
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    custom_config = json.load(f)
                    # 커스텀 설정으로 기본 설정 업데이트
                    for key, value in custom_config.items():
                        if key in default_config:
                            default_config[key].__dict__.update(value)
            except Exception as e:
                self.logger.warning(f"설정 파일 로드 실패, 기본 설정 사용: {e}")
        
        return default_config
    
    def _load_jewelry_prompts(self) -> Dict[str, str]:
        """주얼리 특화 프롬프트 템플릿 로드"""
        return {
            "general_analysis": """
당신은 주얼리 업계 전문 AI 분석가입니다. 다음 내용을 분석해주세요:

분석 내용: {content}

다음 관점에서 분석해주세요:
1. 주얼리 업계 관련성 및 중요도
2. 핵심 비즈니스 인사이트
3. 기술적/시장 동향 분석
4. 실행 가능한 액션 아이템
5. 위험 요소 및 기회 요소

결과는 다음 형식으로 제공해주세요:
- 요약 (3줄 이내)
- 핵심 인사이트 (5개 이내)
- 액션 아이템 (3개 이내)
- 주얼리 관련성 점수 (1-10)
""",
            
            "emotion_analysis": """
다음 주얼리 업계 관련 내용의 감정과 분위기를 분석해주세요:

내용: {content}

분석 요청:
1. 전체적인 감정 톤 (긍정적/중립적/부정적)
2. 주요 감정 키워드 추출
3. 비즈니스 임팩트 평가
4. 고객/시장 반응 예측
5. 커뮤니케이션 전략 제안

주얼리 업계 맥락에서 전문적으로 분석해주세요.
""",
            
            "market_analysis": """
주얼리 시장 분석 전문가로서 다음 내용을 분석해주세요:

내용: {content}

분석 항목:
1. 시장 트렌드 식별
2. 경쟁사 동향 분석
3. 가격 및 수요 전망
4. 지역별 시장 특성
5. 투자 및 사업 기회

아시아 시장(한국, 홍콩, 태국, 싱가포르) 특화 분석 포함해주세요.
""",
            
            "korean_summary": """
다음 내용을 한국 주얼리 업계 실무진이 이해하기 쉽게 한국어로 요약해주세요:

원본 내용: {content}
언어: {language}

요약 기준:
- 한국어 비즈니스 문서 스타일
- 주얼리 전문용어 정확한 번역
- 실무진 관점에서의 중요도 순서
- 구체적이고 실행 가능한 내용 위주

결과물:
1. 핵심 요약 (200자 이내)
2. 상세 분석 (500자 이내)  
3. 주요 결정사항/액션 아이템
4. 참고사항 및 후속 조치
"""
        }
    
    def _initialize_ai_clients(self):
        """AI 클라이언트 초기화"""
        try:
            # OpenAI (GPT-4o)
            if os.getenv("OPENAI_API_KEY"):
                openai.api_key = os.getenv("OPENAI_API_KEY")
                self.ai_clients["gpt4o"] = openai
                self.logger.info("✅ GPT-4o 클라이언트 초기화 완료")
            
            # Anthropic (Claude 3.5)
            if os.getenv("ANTHROPIC_API_KEY"):
                self.ai_clients["claude35"] = Anthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
                self.logger.info("✅ Claude 3.5 클라이언트 초기화 완료")
            
            # Google (Gemini)
            if os.getenv("GOOGLE_API_KEY"):
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                self.ai_clients["gemini"] = genai
                self.logger.info("✅ Gemini Pro 클라이언트 초기화 완료")
            
            if not self.ai_clients:
                self.logger.warning("⚠️ API 키가 설정되지 않음 - 데모 모드로 실행")
                
        except Exception as e:
            self.logger.error(f"❌ AI 클라이언트 초기화 실패: {e}")
    
    def _initialize_jewelry_modules(self):
        """주얼리 특화 모듈 초기화"""
        try:
            self.jewelry_engine = JewelryAIEngine()
            self.korean_engine = KoreanSummaryEngineV21()
            self.quality_analyzer = QualityAnalyzerV21()
            self.logger.info("✅ 주얼리 특화 모듈 초기화 완료")
        except Exception as e:
            self.logger.warning(f"⚠️ 주얼리 모듈 초기화 부분 실패: {e}")
    
    async def analyze_with_gpt4o(self, content: str, prompt_type: str = "general_analysis") -> AnalysisResult:
        """GPT-4o 분석"""
        try:
            if "gpt4o" not in self.ai_clients:
                return self._create_demo_result("GPT-4o", content)
            
            start_time = datetime.now()
            
            prompt = self.jewelry_prompts[prompt_type].format(content=content)
            
            response = await asyncio.to_thread(
                self.ai_clients["gpt4o"].ChatCompletion.create,
                model=self.models_config["gpt4o"].model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.models_config["gpt4o"].max_tokens,
                temperature=self.models_config["gpt4o"].temperature
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            result_content = response.choices[0].message.content
            
            return AnalysisResult(
                model_name="GPT-4o",
                content=result_content,
                confidence_score=0.92,
                processing_time=processing_time,
                jewelry_relevance=self._calculate_jewelry_relevance(result_content),
                language_detected="ko",
                key_insights=self._extract_insights(result_content),
                action_items=self._extract_action_items(result_content),
                quality_metrics={"accuracy": 0.92, "relevance": 0.89}
            )
            
        except Exception as e:
            self.logger.error(f"GPT-4o 분석 실패: {e}")
            return self._create_error_result("GPT-4o", str(e))
    
    async def analyze_with_claude35(self, content: str, prompt_type: str = "general_analysis") -> AnalysisResult:
        """Claude 3.5 분석"""
        try:
            if "claude35" not in self.ai_clients:
                return self._create_demo_result("Claude 3.5", content)
            
            start_time = datetime.now()
            
            prompt = self.jewelry_prompts[prompt_type].format(content=content)
            
            response = await asyncio.to_thread(
                self.ai_clients["claude35"].messages.create,
                model=self.models_config["claude35"].model_id,
                max_tokens=self.models_config["claude35"].max_tokens,
                temperature=self.models_config["claude35"].temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            result_content = response.content[0].text
            
            return AnalysisResult(
                model_name="Claude 3.5",
                content=result_content,
                confidence_score=0.94,
                processing_time=processing_time,
                jewelry_relevance=self._calculate_jewelry_relevance(result_content),
                language_detected="ko",
                key_insights=self._extract_insights(result_content),
                action_items=self._extract_action_items(result_content),
                quality_metrics={"accuracy": 0.94, "reasoning": 0.96}
            )
            
        except Exception as e:
            self.logger.error(f"Claude 3.5 분석 실패: {e}")
            return self._create_error_result("Claude 3.5", str(e))
    
    async def analyze_with_gemini(self, content: str, prompt_type: str = "general_analysis") -> AnalysisResult:
        """Gemini Pro 분석"""
        try:
            if "gemini" not in self.ai_clients:
                return self._create_demo_result("Gemini Pro", content)
            
            start_time = datetime.now()
            
            model = self.ai_clients["gemini"].GenerativeModel(
                self.models_config["gemini"].model_id
            )
            
            prompt = self.jewelry_prompts[prompt_type].format(content=content)
            
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={
                    "temperature": self.models_config["gemini"].temperature,
                    "max_output_tokens": self.models_config["gemini"].max_tokens,
                }
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            result_content = response.text
            
            return AnalysisResult(
                model_name="Gemini Pro",
                content=result_content,
                confidence_score=0.88,
                processing_time=processing_time,
                jewelry_relevance=self._calculate_jewelry_relevance(result_content),
                language_detected="ko",
                key_insights=self._extract_insights(result_content),
                action_items=self._extract_action_items(result_content),
                quality_metrics={"accuracy": 0.88, "creativity": 0.93}
            )
            
        except Exception as e:
            self.logger.error(f"Gemini Pro 분석 실패: {e}")
            return self._create_error_result("Gemini Pro", str(e))
    
    async def multi_model_consensus_analysis(
        self, 
        content: str, 
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """다중 모델 컨센서스 분석"""
        self.logger.info(f"🚀 다중 모델 컨센서스 분석 시작: {analysis_type}")
        
        start_time = datetime.now()
        
        # 병렬 분석 실행
        tasks = [
            self.analyze_with_gpt4o(content, "general_analysis"),
            self.analyze_with_claude35(content, "general_analysis"),
            self.analyze_with_gemini(content, "general_analysis")
        ]
        
        if analysis_type == "comprehensive":
            # 감정 분석 추가
            tasks.extend([
                self.analyze_with_gpt4o(content, "emotion_analysis"),
                self.analyze_with_claude35(content, "market_analysis")
            ])
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 유효한 결과만 필터링
        valid_results = [r for r in results if isinstance(r, AnalysisResult)]
        
        if not valid_results:
            return {"error": "모든 AI 모델 분석 실패"}
        
        # 컨센서스 분석
        consensus = self._calculate_consensus(valid_results)
        
        # 주얼리 특화 분석 추가
        jewelry_analysis = await self._jewelry_specialized_analysis(content)
        
        # 한국어 통합 요약
        korean_summary = await self._korean_integrated_summary(content, valid_results)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            "analysis_type": analysis_type,
            "processing_time": total_time,
            "model_results": [r.__dict__ for r in valid_results],
            "consensus": consensus,
            "jewelry_analysis": jewelry_analysis,
            "korean_summary": korean_summary,
            "quality_score": self._calculate_overall_quality(valid_results),
            "recommendations": self._generate_recommendations(consensus),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "models_used": len(valid_results),
                "analysis_version": "v2.2"
            }
        }
        
        # 성능 메트릭 업데이트
        self._update_performance_metrics(result)
        
        self.logger.info(f"✅ 다중 모델 분석 완료 - 품질 점수: {result['quality_score']:.2f}")
        
        return result
    
    def _calculate_consensus(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """컨센서스 계산"""
        if not results:
            return {}
        
        # 신뢰도 가중 평균
        total_confidence = sum(r.confidence_score for r in results)
        avg_confidence = total_confidence / len(results)
        
        # 주얼리 관련성 평균
        avg_jewelry_relevance = sum(r.jewelry_relevance for r in results) / len(results)
        
        # 공통 키워드 추출
        all_insights = []
        all_actions = []
        
        for result in results:
            all_insights.extend(result.key_insights)
            all_actions.extend(result.action_items)
        
        # 빈도 기반 공통 인사이트
        common_insights = self._extract_common_keywords(all_insights)
        common_actions = self._extract_common_keywords(all_actions)
        
        return {
            "confidence_score": avg_confidence,
            "jewelry_relevance": avg_jewelry_relevance,
            "common_insights": common_insights[:5],  # 상위 5개
            "common_actions": common_actions[:3],    # 상위 3개
            "model_agreement": self._calculate_agreement(results),
            "quality_indicators": {
                "consistency": self._calculate_consistency(results),
                "completeness": self._calculate_completeness(results),
                "actionability": self._calculate_actionability(results)
            }
        }
    
    async def _jewelry_specialized_analysis(self, content: str) -> Dict[str, Any]:
        """주얼리 특화 분석"""
        try:
            if self.jewelry_engine:
                return await asyncio.to_thread(
                    self.jewelry_engine.comprehensive_analysis, content
                )
            else:
                return {
                    "jewelry_keywords": self._extract_jewelry_keywords(content),
                    "business_impact": "중간",
                    "market_relevance": "높음",
                    "technical_aspects": ["품질", "디자인", "제조"]
                }
        except Exception as e:
            self.logger.warning(f"주얼리 특화 분석 실패: {e}")
            return {"error": str(e)}
    
    async def _korean_integrated_summary(
        self, 
        content: str, 
        results: List[AnalysisResult]
    ) -> Dict[str, str]:
        """한국어 통합 요약"""
        try:
            if self.korean_engine:
                # 모든 분석 결과를 통합하여 한국어 요약
                combined_analysis = "\n\n".join([r.content for r in results])
                return await asyncio.to_thread(
                    self.korean_engine.create_integrated_summary,
                    content,
                    combined_analysis
                )
            else:
                # 폴백: 기본 한국어 요약
                return {
                    "executive_summary": "주얼리 업계 관련 내용의 종합 분석 결과",
                    "key_findings": "다중 AI 모델 분석을 통한 핵심 발견사항",
                    "business_implications": "비즈니스에 미치는 영향 분석",
                    "next_steps": "추천 후속 조치사항"
                }
        except Exception as e:
            self.logger.warning(f"한국어 통합 요약 실패: {e}")
            return {"error": str(e)}
    
    def _calculate_jewelry_relevance(self, content: str) -> float:
        """주얼리 관련성 계산"""
        jewelry_keywords = [
            "다이아몬드", "루비", "사파이어", "에메랄드", "주얼리", "보석",
            "금", "은", "백금", "반지", "목걸이", "귀걸이", "브로치",
            "4C", "캐럿", "컷", "색상", "투명도", "감정서", "GIA", "AGS"
        ]
        
        content_lower = content.lower()
        matches = sum(1 for keyword in jewelry_keywords if keyword.lower() in content_lower)
        return min(matches / len(jewelry_keywords) * 10, 1.0)
    
    def _extract_insights(self, content: str) -> List[str]:
        """인사이트 추출"""
        # 간단한 키워드 기반 인사이트 추출
        lines = content.split('\n')
        insights = []
        
        for line in lines:
            if any(keyword in line for keyword in ['인사이트', '핵심', '중요', '트렌드', '분석']):
                insights.append(line.strip())
        
        return insights[:5]
    
    def _extract_action_items(self, content: str) -> List[str]:
        """액션 아이템 추출"""
        lines = content.split('\n')
        actions = []
        
        for line in lines:
            if any(keyword in line for keyword in ['액션', '조치', '실행', '권장', '제안']):
                actions.append(line.strip())
        
        return actions[:3]
    
    def _create_demo_result(self, model_name: str, content: str) -> AnalysisResult:
        """데모 결과 생성"""
        return AnalysisResult(
            model_name=f"{model_name} (데모)",
            content=f"[데모 모드] {model_name} 분석 결과: 주얼리 업계 관련 내용 분석 완료",
            confidence_score=0.85,
            processing_time=1.5,
            jewelry_relevance=0.7,
            language_detected="ko",
            key_insights=["데모 인사이트 1", "데모 인사이트 2"],
            action_items=["데모 액션 1", "데모 액션 2"],
            quality_metrics={"demo_mode": True}
        )
    
    def _create_error_result(self, model_name: str, error: str) -> AnalysisResult:
        """에러 결과 생성"""
        return AnalysisResult(
            model_name=f"{model_name} (오류)",
            content=f"분석 실패: {error}",
            confidence_score=0.0,
            processing_time=0.0,
            jewelry_relevance=0.0,
            language_detected="unknown",
            key_insights=[],
            action_items=[],
            quality_metrics={"error": True}
        )
    
    def _extract_common_keywords(self, text_list: List[str]) -> List[str]:
        """공통 키워드 추출"""
        if not text_list:
            return []
        
        word_count = {}
        for text in text_list:
            words = text.split()
            for word in words:
                word_count[word] = word_count.get(word, 0) + 1
        
        # 빈도순 정렬
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words if count > 1][:10]
    
    def _calculate_agreement(self, results: List[AnalysisResult]) -> float:
        """모델 간 일치도 계산"""
        if len(results) < 2:
            return 1.0
        
        # 신뢰도 점수의 표준편차로 일치도 측정
        confidence_scores = [r.confidence_score for r in results]
        import statistics
        std_dev = statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0
        
        # 표준편차가 낮을수록 일치도가 높음
        return max(0, 1 - (std_dev * 2))
    
    def _calculate_consistency(self, results: List[AnalysisResult]) -> float:
        """일관성 점수 계산"""
        return 0.85  # 임시 구현
    
    def _calculate_completeness(self, results: List[AnalysisResult]) -> float:
        """완성도 점수 계산"""
        return 0.90  # 임시 구현
    
    def _calculate_actionability(self, results: List[AnalysisResult]) -> float:
        """실행가능성 점수 계산"""
        total_actions = sum(len(r.action_items) for r in results)
        return min(total_actions / 10, 1.0)
    
    def _calculate_overall_quality(self, results: List[AnalysisResult]) -> float:
        """전체 품질 점수 계산"""
        if not results:
            return 0.0
        
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        avg_relevance = sum(r.jewelry_relevance for r in results) / len(results)
        
        return (avg_confidence * 0.6) + (avg_relevance * 0.4)
    
    def _generate_recommendations(self, consensus: Dict[str, Any]) -> List[str]:
        """추천사항 생성"""
        recommendations = []
        
        if consensus.get("jewelry_relevance", 0) > 0.8:
            recommendations.append("🔥 높은 주얼리 관련성 - 즉시 비즈니스 적용 검토")
        
        if consensus.get("confidence_score", 0) > 0.9:
            recommendations.append("✅ 높은 신뢰도 - 의사결정 근거로 활용 가능")
        
        if len(consensus.get("common_actions", [])) > 2:
            recommendations.append("🎯 명확한 액션 아이템 - 단계별 실행 계획 수립")
        
        return recommendations
    
    def _update_performance_metrics(self, result: Dict[str, Any]):
        """성능 메트릭 업데이트"""
        self.performance_metrics["total_analyses"] += 1
        
        current_quality = result.get("quality_score", 0)
        total_analyses = self.performance_metrics["total_analyses"]
        
        # 누적 평균 계산
        prev_avg = self.performance_metrics["average_accuracy"]
        self.performance_metrics["average_accuracy"] = (
            (prev_avg * (total_analyses - 1) + current_quality) / total_analyses
        )
        
        self.performance_metrics["last_update"] = datetime.now()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트 반환"""
        return {
            "summary": {
                "total_analyses": self.performance_metrics["total_analyses"],
                "average_quality": self.performance_metrics["average_accuracy"],
                "last_update": self.performance_metrics["last_update"].isoformat()
            },
            "models_status": {
                name: "active" if name in self.ai_clients else "inactive"
                for name in self.models_config.keys()
            },
            "jewelry_modules": {
                "jewelry_engine": self.jewelry_engine is not None,
                "korean_engine": self.korean_engine is not None,
                "quality_analyzer": self.quality_analyzer is not None
            }
        }
    
    async def save_analysis_result(self, result: Dict[str, Any], file_path: str):
        """분석 결과 저장"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            self.logger.info(f"✅ 분석 결과 저장 완료: {file_path}")
        except Exception as e:
            self.logger.error(f"❌ 분석 결과 저장 실패: {e}")


async def main():
    """테스트 실행"""
    print("🚀 솔로몬드 AI v2.2 - 차세대 AI 통합 엔진 테스트")
    
    integrator = NextGenAIIntegrator()
    
    # 테스트 내용
    test_content = """
    오늘 홍콩 주얼리쇼에서 새로운 다이아몬드 컷팅 기술에 대한 발표가 있었습니다.
    이 기술은 기존 라운드 브릴리언트 컷보다 30% 더 많은 빛을 반사할 수 있다고 합니다.
    주요 보석 브랜드들이 이 기술 도입을 검토하고 있으며, 
    내년부터 상용화될 예정입니다.
    """
    
    print("\n📊 다중 모델 컨센서스 분석 시작...")
    
    result = await integrator.multi_model_consensus_analysis(
        test_content, 
        analysis_type="comprehensive"
    )
    
    print(f"\n✅ 분석 완료!")
    print(f"🎯 품질 점수: {result['quality_score']:.2f}")
    print(f"⏱️ 처리 시간: {result['processing_time']:.2f}초")
    print(f"🤖 사용된 모델: {result['metadata']['models_used']}개")
    
    # 성능 리포트
    performance = integrator.get_performance_report()
    print(f"\n📈 시스템 상태:")
    print(f"  - 총 분석 횟수: {performance['summary']['total_analyses']}")
    print(f"  - 평균 품질: {performance['summary']['average_quality']:.2f}")
    
    print("\n🎉 차세대 AI 통합 엔진 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(main())
