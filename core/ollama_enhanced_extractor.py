#!/usr/bin/env python3
"""
🤖 Ollama 강화 메시지 추출기
ComprehensiveMessageExtractor + Ollama AI 완전 통합
"이 사람들이 무엇을 말하는지" → "핵심 비즈니스 인사이트" 완벽 변환
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

try:
    from shared.ollama_interface import OllamaInterface
    from core.comprehensive_message_extractor import ComprehensiveMessageExtractor
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False

class OllamaEnhancedExtractor:
    """Ollama AI 강화 메시지 추출기"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        if COMPONENTS_AVAILABLE:
            self.ollama = OllamaInterface()
            self.base_extractor = ComprehensiveMessageExtractor()
            self.logger.info("✅ Ollama 강화 추출기 초기화 완료")
        else:
            self.ollama = None
            self.base_extractor = None
            self.logger.warning("⚠️ 컴포넌트 누락 - 기본 모드로 실행")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def extract_ultimate_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """ULTIMATE 종합 인사이트 추출"""
        if not COMPONENTS_AVAILABLE:
            return self._basic_insights(analysis_results)
        
        try:
            # 1. 기본 추출기로 1차 분석
            base_insights = self._extract_base_insights(analysis_results)
            
            # 2. Ollama AI로 고급 분석
            enhanced_insights = self._ollama_deep_analysis(analysis_results, base_insights)
            
            # 3. 주얼리 비즈니스 특화 분석
            business_insights = self._jewelry_business_analysis(analysis_results)
            
            # 4. 최종 통합
            return self._synthesize_ultimate_insights(
                base_insights, enhanced_insights, business_insights, analysis_results
            )
            
        except Exception as e:
            self.logger.error(f"Ultimate 인사이트 추출 실패: {e}")
            return self._basic_insights(analysis_results)
    
    def _extract_base_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """기본 추출기를 활용한 1차 분석"""
        base_insights = {}
        
        # 오디오 분석에서 텍스트 추출
        if "audio_analysis" in results:
            audio = results["audio_analysis"]
            transcript = audio.get("transcript", "")
            
            if transcript and self.base_extractor:
                # 기본 메시지 추출
                key_messages = self.base_extractor.extract_key_messages(transcript)
                base_insights["key_messages"] = key_messages
        
        return base_insights
    
    def _ollama_deep_analysis(self, results: Dict[str, Any], base_insights: Dict) -> Dict[str, Any]:
        """Ollama AI 심층 분석"""
        if not self.ollama:
            return {}
        
        # 분석 데이터 준비
        analysis_context = self._prepare_ollama_context(results, base_insights)
        
        # Ollama 프롬프트 구성
        deep_analysis_prompt = f"""
당신은 주얼리 업계 전문 컨설턴트입니다. 다음 컨퍼런스/회의 분석 결과를 바탕으로 핵심 비즈니스 인사이트를 도출해주세요.

### 분석 데이터:
{analysis_context}

### 심층 분석 요청:

#### 1. 🎯 핵심 메시지 (Core Messages)
- 이 대화/회의에서 전달된 가장 중요한 메시지 3가지는?
- 각 메시지의 비즈니스 임팩트는?

#### 2. 👥 참여자 인사이트 (Participant Insights)
- 각 화자의 역할과 관심사는?
- 누가 결정권자이고, 누가 영향력을 가지고 있는가?

#### 3. 💎 주얼리 비즈니스 기회 (Business Opportunities)
- 발견된 비즈니스 기회는?
- 고객의 실제 니즈는 무엇인가?
- 어떤 제품/서비스에 관심이 있는가?

#### 4. 🔥 긴급 액션 아이템 (Urgent Actions)
- 즉시 실행해야 할 일은?
- 후속 미팅이나 연락이 필요한가?

#### 5. 📊 감정 및 만족도 (Emotional Analysis)
- 전체적인 분위기와 만족도는?
- 우려사항이나 저항 포인트는?

한국어로 구체적이고 실행 가능한 인사이트를 제공해주세요.
"""
        
        try:
            # Ollama AI 분석 실행
            deep_analysis = self.ollama.generate_response(
                deep_analysis_prompt,
                model="qwen2.5:7b",
                context_type="conference_analysis"
            )
            
            return {
                "deep_analysis": deep_analysis,
                "analysis_timestamp": datetime.now().isoformat(),
                "model_used": "qwen2.5:7b"
            }
            
        except Exception as e:
            self.logger.error(f"Ollama 심층 분석 실패: {e}")
            return {}
    
    def _prepare_ollama_context(self, results: Dict[str, Any], base_insights: Dict) -> str:
        """Ollama 분석용 컨텍스트 준비"""
        context_parts = []
        
        # 오디오 분석 정보
        if "audio_analysis" in results:
            audio = results["audio_analysis"]
            
            # 기본 정보
            context_parts.append(f"### 음성 분석 결과:")
            context_parts.append(f"- 총 발화 시간: {audio.get('duration', 0):.1f}초")
            context_parts.append(f"- 화자 수: {len(audio.get('speaker_segments', []))}명")
            context_parts.append(f"- 음성 품질: {audio.get('audio_quality', {}).get('quality_level', 'Unknown')}")
            
            # 전체 대화 내용
            transcript = audio.get("transcript", "")
            if transcript:
                context_parts.append(f"### 대화 내용:")
                context_parts.append(transcript[:1000] + ("..." if len(transcript) > 1000 else ""))
            
            # 화자별 분석
            if "speaker_segments" in audio:
                context_parts.append(f"### 화자별 분석:")
                for i, speaker_info in enumerate(audio["speaker_segments"][:3]):  # 최대 3명
                    speaker_id = speaker_info.get("speaker", f"화자{i+1}")
                    segments = speaker_info.get("segments", [])
                    total_time = sum(seg.get("end", 0) - seg.get("start", 0) for seg in segments)
                    
                    context_parts.append(f"- {speaker_id}: {len(segments)}회 발화, {total_time:.1f}초")
                    
                    # 주요 발언
                    if segments:
                        main_speech = max(segments, key=lambda x: len(x.get("text", "")))
                        context_parts.append(f"  주요 발언: {main_speech.get('text', '')[:200]}")
        
        # 비주얼 분석 정보
        if "visual_analysis" in results:
            visual = results["visual_analysis"]
            context_parts.append(f"### 시각 자료 분석:")
            context_parts.append(f"- 텍스트 블록: {visual.get('total_text_blocks', 0)}개")
            
            full_text = visual.get("full_text", "")
            if full_text:
                context_parts.append(f"- 추출된 텍스트: {full_text[:500]}")
        
        return "\n".join(context_parts)
    
    def _jewelry_business_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """주얼리 비즈니스 특화 분석"""
        business_insights = {
            "jewelry_focus": False,
            "product_categories": [],
            "customer_profile": {},
            "sales_stage": "정보수집",
            "business_potential": "보통"
        }
        
        # 텍스트 데이터 수집
        all_text = ""
        if "audio_analysis" in results:
            all_text += results["audio_analysis"].get("transcript", "") + " "
        if "visual_analysis" in results:
            all_text += results["visual_analysis"].get("full_text", "") + " "
        
        if not all_text:
            return business_insights
        
        # 주얼리 키워드 분석
        jewelry_keywords = {
            "제품": ["반지", "목걸이", "귀걸이", "팔찌", "펜던트", "브로치", "시계", "다이아몬드"],
            "재료": ["금", "은", "백금", "플래티넘", "루비", "사파이어", "에메랄드"],
            "상황": ["결혼", "약혼", "선물", "기념일", "생일"],
            "가격": ["가격", "비용", "얼마", "할인", "이벤트"],
            "구매": ["구매", "사고싶", "주문", "예약", "결정"]
        }
        
        # 카테고리별 키워드 발견
        for category, keywords in jewelry_keywords.items():
            found = [kw for kw in keywords if kw in all_text]
            if found:
                business_insights[f"{category}_mentioned"] = found
                if category == "제품":
                    business_insights["product_categories"] = found
                    business_insights["jewelry_focus"] = True
        
        # 영업 단계 판단
        if any(kw in all_text for kw in ["구매", "주문", "결정"]):
            business_insights["sales_stage"] = "구매결정"
            business_insights["business_potential"] = "높음"
        elif any(kw in all_text for kw in ["가격", "비용", "할인"]):
            business_insights["sales_stage"] = "가격협상"
            business_insights["business_potential"] = "높음"
        elif any(kw in all_text for kw in ["상담", "문의", "추천"]):
            business_insights["sales_stage"] = "상담단계"
            business_insights["business_potential"] = "보통"
        
        # 고객 프로필 추정
        if any(kw in all_text for kw in ["결혼", "약혼"]):
            business_insights["customer_profile"]["occasion"] = "결혼/약혼"
            business_insights["customer_profile"]["urgency"] = "높음"
        elif any(kw in all_text for kw in ["생일", "기념일"]):
            business_insights["customer_profile"]["occasion"] = "기념일"
        
        return business_insights
    
    def _synthesize_ultimate_insights(self, base: Dict, enhanced: Dict, 
                                    business: Dict, raw_results: Dict) -> Dict[str, Any]:
        """최종 통합 인사이트 생성"""
        
        ultimate_insights = {
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_version": "ULTIMATE_v1.0",
            "confidence_score": self._calculate_overall_confidence(raw_results),
            
            # 핵심 결과
            "executive_summary": self._generate_executive_summary(enhanced, business),
            "key_findings": self._extract_key_findings(base, enhanced, business),
            "business_recommendations": self._generate_business_recommendations(business),
            "next_actions": self._suggest_next_actions(business, enhanced),
            
            # 상세 분석
            "detailed_analysis": {
                "base_insights": base,
                "ai_enhanced": enhanced,
                "business_analysis": business
            },
            
            # 메타데이터
            "analysis_metadata": {
                "total_processing_time": raw_results.get("analysis_metadata", {}).get("processing_time", 0),
                "data_sources": self._identify_data_sources(raw_results),
                "quality_metrics": self._calculate_quality_metrics(raw_results)
            }
        }
        
        return ultimate_insights
    
    def _generate_executive_summary(self, enhanced: Dict, business: Dict) -> str:
        """경영진 요약 생성"""
        summary_parts = []
        
        # AI 분석 요약
        if "deep_analysis" in enhanced:
            ai_summary = enhanced["deep_analysis"][:200] + "..."
            summary_parts.append(f"🤖 AI 분석: {ai_summary}")
        
        # 비즈니스 요약
        if business.get("jewelry_focus"):
            products = ", ".join(business.get("product_categories", [])[:3])
            stage = business.get("sales_stage", "정보수집")
            potential = business.get("business_potential", "보통")
            summary_parts.append(f"💎 비즈니스: {products} 관심, {stage} 단계, {potential} 잠재력")
        
        return " | ".join(summary_parts) if summary_parts else "종합적인 분석이 완료되었습니다."
    
    def _extract_key_findings(self, base: Dict, enhanced: Dict, business: Dict) -> List[str]:
        """핵심 발견사항 추출"""
        findings = []
        
        # 비즈니스 발견사항
        if business.get("jewelry_focus"):
            findings.append(f"주얼리 관련 대화 (집중도: {business.get('business_potential', '보통')})")
        
        if business.get("sales_stage") == "구매결정":
            findings.append("구매 의사결정 단계 - 높은 전환 가능성")
        
        # AI 분석 발견사항
        if enhanced and "deep_analysis" in enhanced:
            findings.append("AI 심층 분석 완료 - 상세 인사이트 확보")
        
        return findings[:5]  # 최대 5개
    
    def _generate_business_recommendations(self, business: Dict) -> List[str]:
        """비즈니스 권장사항 생성"""
        recommendations = []
        
        sales_stage = business.get("sales_stage", "정보수집")
        
        if sales_stage == "구매결정":
            recommendations.append("즉시 후속 연락 - 구매 지원 및 상담 제공")
            recommendations.append("맞춤형 제품 제안서 발송")
        elif sales_stage == "가격협상":
            recommendations.append("경쟁력 있는 가격 제안 준비")
            recommendations.append("할인 혜택 또는 패키지 상품 검토")
        elif sales_stage == "상담단계":
            recommendations.append("전문 상담 일정 조율")
            recommendations.append("고객 니즈 세부 파악을 위한 질문지 준비")
        else:
            recommendations.append("관심 제품 카테고리 기반 정보 제공")
            recommendations.append("정기적 팔로업 계획 수립")
        
        return recommendations[:3]  # 최대 3개
    
    def _suggest_next_actions(self, business: Dict, enhanced: Dict) -> List[Dict[str, Any]]:
        """다음 액션 제안"""
        actions = []
        
        # 우선순위별 액션
        if business.get("sales_stage") == "구매결정":
            actions.append({
                "priority": "긴급",
                "action": "고객 연락",
                "description": "24시간 내 구매 지원 연락",
                "deadline": "1일"
            })
        
        if business.get("jewelry_focus"):
            actions.append({
                "priority": "높음", 
                "action": "제품 카탈로그 발송",
                "description": "관심 제품 카테고리 기반 자료 제공",
                "deadline": "3일"
            })
        
        if enhanced and "deep_analysis" in enhanced:
            actions.append({
                "priority": "보통",
                "action": "AI 분석 보고서 검토",
                "description": "상세 인사이트 내부 공유 및 전략 수립",
                "deadline": "1주일"
            })
        
        return actions
    
    def _calculate_overall_confidence(self, results: Dict[str, Any]) -> float:
        """전체 신뢰도 계산"""
        confidence_factors = []
        
        # 오디오 품질
        if "audio_analysis" in results:
            audio_quality = results["audio_analysis"].get("audio_quality", {})
            if "quality_score" in audio_quality:
                confidence_factors.append(audio_quality["quality_score"] / 100)
        
        # 비주얼 신뢰도
        if "visual_analysis" in results:
            visual_confidence = results["visual_analysis"].get("avg_confidence", 0)
            if visual_confidence > 0:
                confidence_factors.append(visual_confidence)
        
        # AI 분석 가능 여부
        if COMPONENTS_AVAILABLE:
            confidence_factors.append(0.9)  # AI 분석 가능
        else:
            confidence_factors.append(0.5)  # 기본 분석만
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _identify_data_sources(self, results: Dict[str, Any]) -> List[str]:
        """데이터 소스 식별"""
        sources = []
        
        if "audio_analysis" in results:
            sources.append("음성 분석")
        if "visual_analysis" in results:
            sources.append("시각 자료 분석")
        if COMPONENTS_AVAILABLE:
            sources.append("AI 심층 분석")
        
        return sources
    
    def _calculate_quality_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """품질 메트릭 계산"""
        metrics = {}
        
        # 데이터 완성도
        data_completeness = 0
        if "audio_analysis" in results:
            data_completeness += 0.5
        if "visual_analysis" in results:
            data_completeness += 0.3
        if COMPONENTS_AVAILABLE:
            data_completeness += 0.2
        
        metrics["data_completeness"] = min(1.0, data_completeness)
        
        # 분석 깊이
        metrics["analysis_depth"] = 0.9 if COMPONENTS_AVAILABLE else 0.6
        
        # 신뢰도
        metrics["reliability"] = self._calculate_overall_confidence(results)
        
        return metrics
    
    def _basic_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """컴포넌트 없을 때 기본 인사이트"""
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_version": "BASIC_v1.0",
            "confidence_score": 0.5,
            "executive_summary": "기본 분석이 완료되었습니다.",
            "key_findings": ["분석 데이터 확보 완료"],
            "business_recommendations": ["상세 분석을 위해 AI 컴포넌트 설치 권장"],
            "next_actions": [
                {
                    "priority": "보통",
                    "action": "시스템 업그레이드",
                    "description": "AI 분석 기능 활성화"
                }
            ]
        }

# 전역 인스턴스
enhanced_extractor = OllamaEnhancedExtractor()

def get_ultimate_insights(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """편의 함수: Ultimate 인사이트 추출"""
    return enhanced_extractor.extract_ultimate_insights(analysis_results)