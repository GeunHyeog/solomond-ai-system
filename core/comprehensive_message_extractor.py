#!/usr/bin/env python3
"""
종합 메시지 추출 엔진
"이 사람들이 무엇을 말하는지" 명확하게 파악하는 시스템
클로바 노트 + ChatGPT 수준의 요약 품질 제공
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

class ComprehensiveMessageExtractor:
    """종합 메시지 추출기"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 주얼리 도메인 키워드
        self.jewelry_keywords = {
            "제품": ["반지", "목걸이", "귀걸이", "팔찌", "펜던트", "브로치", "시계"],
            "재료": ["금", "은", "백금", "플래티넘", "다이아몬드", "루비", "사파이어", "에메랄드"],
            "상황": ["결혼", "약혼", "선물", "기념일", "생일", "졸업", "승진"],
            "감정": ["좋아", "예쁘", "마음에", "고민", "망설", "결정", "선택"],
            "비즈니스": ["가격", "할인", "이벤트", "상담", "문의", "구매", "주문"]
        }
        
        # 대화 패턴 분석
        self.conversation_patterns = {
            "정보_문의": ["얼마", "가격", "비용", "언제", "어디서", "어떻게"],
            "구매_의향": ["사고싶", "구매", "주문", "예약", "결정"],
            "비교_검토": ["다른", "비교", "차이", "어떤게", "뭐가"],
            "고민_상담": ["고민", "망설", "모르겠", "어떨까", "추천"]
        }
        
        self.logger.info("🎯 종합 메시지 추출 엔진 초기화 완료")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.ComprehensiveMessageExtractor')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def extract_key_messages(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """핵심 메시지 추출 - "이 사람들이 무엇을 말하는지" 명확하게"""
        
        if not text or len(text.strip()) < 10:
            return self._create_empty_result()
        
        # 1. 텍스트 전처리 및 정제
        cleaned_text = self._clean_and_enhance_text(text)
        
        # 2. 화자 구분 및 대화 플로우 분석
        speakers_analysis = self._analyze_speakers_and_flow(cleaned_text)
        
        # 3. 핵심 메시지 추출
        main_messages = self._extract_main_messages(cleaned_text, speakers_analysis)
        
        # 4. 대화 의도 및 감정 분석
        intent_analysis = self._analyze_conversation_intent(cleaned_text)
        
        # 5. 실행 가능한 인사이트 생성
        actionable_insights = self._generate_actionable_insights(
            main_messages, intent_analysis, context
        )
        
        # 6. 사용자 친화적 요약 생성
        user_friendly_summary = self._create_user_friendly_summary(
            main_messages, intent_analysis, actionable_insights
        )
        
        return {
            "status": "success",
            "main_summary": user_friendly_summary,
            "key_messages": main_messages,
            "conversation_analysis": {
                "speakers": speakers_analysis,
                "intent": intent_analysis,
                "insights": actionable_insights
            },
            "original_text_length": len(text),
            "processed_text_length": len(cleaned_text),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _clean_and_enhance_text(self, text: str) -> str:
        """텍스트 정제 및 품질 향상"""
        
        # 1. 기본 정제
        text = re.sub(r'\s+', ' ', text)  # 연속 공백 제거
        text = re.sub(r'[^\w\s가-힣.,!?]', '', text)  # 특수문자 정리
        
        # 2. 한국어 맞춤법 기본 보정
        corrections = {
            "에요": "예요", "구매할게요": "구매하겠어요", "좋겠네요": "좋겠어요",
            "반지가": "반지가", "다이야": "다이아", "플래티늄": "플래티넘"
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        # 3. 주얼리 전문용어 보정
        jewelry_corrections = {
            "다이야몬드": "다이아몬드", "골드": "금", "실버": "은",
            "링": "반지", "네클리스": "목걸이", "이어링": "귀걸이"
        }
        
        for wrong, correct in jewelry_corrections.items():
            text = re.sub(f'\\b{wrong}\\b', correct, text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _analyze_speakers_and_flow(self, text: str) -> Dict[str, Any]:
        """화자 구분 및 대화 플로우 분석"""
        
        # 화자 구분 키워드
        customer_indicators = ["고객", "구매자", "아", "음", "그럼", "저는", "제가"]
        staff_indicators = ["안녕하세요", "추천", "설명", "가격은", "이 제품"]
        
        sentences = re.split(r'[.!?]\s*', text)
        
        speakers = []
        current_speaker = "unknown"
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # 화자 추정
            if any(word in sentence for word in customer_indicators):
                current_speaker = "고객"
            elif any(word in sentence for word in staff_indicators):
                current_speaker = "직원"
            
            speakers.append({
                "speaker": current_speaker,
                "content": sentence.strip(),
                "type": self._classify_sentence_type(sentence)
            })
        
        return {
            "total_speakers": len(set(s["speaker"] for s in speakers)),
            "speaker_distribution": self._get_speaker_distribution(speakers),
            "conversation_flow": speakers[:10],  # 처음 10개 문장
            "dominant_speaker": self._get_dominant_speaker(speakers)
        }
    
    def _classify_sentence_type(self, sentence: str) -> str:
        """문장 유형 분류"""
        if "?" in sentence or any(word in sentence for word in ["얼마", "언제", "어디"]):
            return "질문"
        elif any(word in sentence for word in ["추천", "설명", "소개"]):
            return "설명"
        elif any(word in sentence for word in ["구매", "사겠", "결정"]):
            return "결정"
        elif any(word in sentence for word in ["고민", "망설", "어떨까"]):
            return "고민"
        else:
            return "일반"
    
    def _extract_main_messages(self, text: str, speakers_analysis: Dict) -> List[Dict[str, Any]]:
        """핵심 메시지 추출"""
        
        messages = []
        
        # 1. 주요 제품/서비스 언급
        for category, keywords in self.jewelry_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    context = self._extract_context_around_keyword(text, keyword)
                    if context:
                        messages.append({
                            "type": f"{category}_언급",
                            "keyword": keyword,
                            "context": context,
                            "importance": "high" if category in ["제품", "비즈니스"] else "medium"
                        })
        
        # 2. 고객 의도 및 니즈
        customer_needs = self._extract_customer_needs(text)
        messages.extend(customer_needs)
        
        # 3. 비즈니스 기회 및 액션 포인트
        business_opportunities = self._extract_business_opportunities(text)
        messages.extend(business_opportunities)
        
        # 중요도 순으로 정렬
        messages.sort(key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(x.get("importance", "low"), 1), reverse=True)
        
        return messages[:10]  # 상위 10개만
    
    def _extract_context_around_keyword(self, text: str, keyword: str, window: int = 30) -> str:
        """키워드 주변 컨텍스트 추출"""
        index = text.find(keyword)
        if index == -1:
            return ""
        
        start = max(0, index - window)
        end = min(len(text), index + len(keyword) + window)
        
        return text[start:end].strip()
    
    def _extract_customer_needs(self, text: str) -> List[Dict[str, Any]]:
        """고객 니즈 추출"""
        needs = []
        
        # 가격 관심도
        if any(word in text for word in ["가격", "얼마", "비용", "저렴", "비싸"]):
            price_context = self._extract_price_context(text)
            needs.append({
                "type": "가격_관심",
                "context": price_context,
                "importance": "high",
                "insight": "고객이 가격 정보를 중요하게 생각하고 있습니다"
            })
        
        # 제품 선택 고민
        if any(word in text for word in ["고민", "선택", "어떤", "추천"]):
            needs.append({
                "type": "선택_고민",
                "context": self._extract_decision_context(text),
                "importance": "high",
                "insight": "고객이 제품 선택에 대해 도움을 필요로 합니다"
            })
        
        # 특별한 목적
        occasions = ["결혼", "약혼", "기념일", "선물", "생일"]
        for occasion in occasions:
            if occasion in text:
                needs.append({
                    "type": f"{occasion}_목적",
                    "context": self._extract_context_around_keyword(text, occasion),
                    "importance": "medium",
                    "insight": f"{occasion} 관련 구매를 고려하고 있습니다"
                })
        
        return needs
    
    def _extract_price_context(self, text: str) -> str:
        """가격 관련 컨텍스트 추출"""
        price_patterns = [
            r'[가-힣\s]*[0-9,]+원[가-힣\s]*',
            r'[가-힣\s]*얼마[가-힣\s]*',
            r'[가-힣\s]*가격[가-힣\s]*'
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return '; '.join(matches[:3])
        
        return "가격에 대한 관심을 보이고 있음"
    
    def _extract_decision_context(self, text: str) -> str:
        """의사결정 관련 컨텍스트 추출"""
        decision_keywords = ["고민", "선택", "결정", "추천", "어떤"]
        contexts = []
        
        for keyword in decision_keywords:
            if keyword in text:
                context = self._extract_context_around_keyword(text, keyword, 40)
                if context:
                    contexts.append(context)
        
        return '; '.join(contexts[:2])
    
    def _extract_business_opportunities(self, text: str) -> List[Dict[str, Any]]:
        """비즈니스 기회 추출"""
        opportunities = []
        
        # 구매 신호
        buy_signals = ["사고 싶", "구매", "주문", "예약", "결정했"]
        if any(signal in text for signal in buy_signals):
            opportunities.append({
                "type": "구매_신호",
                "context": "고객이 구매 의향을 보이고 있음",
                "importance": "high",
                "action": "즉시 상담 진행 및 구매 절차 안내"
            })
        
        # 추가 정보 요청
        info_requests = ["자세히", "더 알고", "설명", "보여주"]
        if any(request in text for request in info_requests):
            opportunities.append({
                "type": "정보_요청",
                "context": "고객이 더 많은 정보를 원하고 있음",
                "importance": "medium",
                "action": "상세 제품 정보 및 카탈로그 제공"
            })
        
        return opportunities
    
    def _analyze_conversation_intent(self, text: str) -> Dict[str, Any]:
        """대화 의도 분석"""
        
        intent_scores = {}
        
        # 각 패턴별 점수 계산
        for intent, keywords in self.conversation_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                intent_scores[intent] = score
        
        # 주요 의도 결정
        if not intent_scores:
            primary_intent = "일반_대화"
            confidence = 0.3
        else:
            primary_intent = max(intent_scores, key=intent_scores.get)
            total_signals = sum(intent_scores.values())
            confidence = intent_scores[primary_intent] / total_signals if total_signals > 0 else 0
        
        # 의도별 설명
        intent_descriptions = {
            "정보_문의": "제품이나 서비스에 대한 정보를 알고 싶어합니다",
            "구매_의향": "실제 구매를 고려하고 있습니다",
            "비교_검토": "여러 옵션을 비교하여 최선의 선택을 하려고 합니다",
            "고민_상담": "구매 결정에 도움이 필요합니다",
            "일반_대화": "일반적인 대화를 나누고 있습니다"
        }
        
        return {
            "primary_intent": primary_intent,
            "confidence": round(confidence, 2),
            "description": intent_descriptions.get(primary_intent, "알 수 없는 의도"),
            "all_detected_intents": intent_scores,
            "urgency_level": self._assess_urgency_level(primary_intent, confidence)
        }
    
    def _assess_urgency_level(self, intent: str, confidence: float) -> str:
        """긴급도 평가"""
        if intent == "구매_의향" and confidence > 0.7:
            return "높음"
        elif intent in ["정보_문의", "비교_검토"] and confidence > 0.5:
            return "보통"
        elif intent == "고민_상담":
            return "보통"
        else:
            return "낮음"
    
    def _generate_actionable_insights(self, messages: List[Dict], intent_analysis: Dict, 
                                    context: Dict = None) -> List[Dict[str, Any]]:
        """실행 가능한 인사이트 생성"""
        
        insights = []
        
        # 1. 즉시 실행 가능한 액션
        if intent_analysis["primary_intent"] == "구매_의향":
            insights.append({
                "type": "즉시_액션",
                "title": "🔥 구매 의향 고객 - 즉시 대응 필요",
                "description": "고객이 구매 결정 단계에 있습니다. 지금이 성사 기회입니다.",
                "action": "즉시 상담 연결, 특별 할인 제안, 구매 절차 안내",
                "priority": "최우선"
            })
        
        # 2. 고객 세그먼트 분석
        segment = self._identify_customer_segment(messages)
        if segment:
            insights.append({
                "type": "고객_세그먼트",
                "title": f"👤 고객 유형: {segment['type']}",
                "description": segment['description'],
                "action": segment['recommended_action'],
                "priority": "높음"
            })
        
        # 3. 제품 추천 기회
        product_opportunities = self._identify_product_opportunities(messages)
        insights.extend(product_opportunities)
        
        # 4. 비즈니스 리스크 및 기회
        risks_and_opportunities = self._assess_risks_and_opportunities(intent_analysis, messages)
        insights.extend(risks_and_opportunities)
        
        return insights[:5]  # 상위 5개만
    
    def _identify_customer_segment(self, messages: List[Dict]) -> Optional[Dict[str, Any]]:
        """고객 세그먼트 식별"""
        
        # 제품 관심도 분석
        product_interests = []
        for msg in messages:
            if msg.get("type", "").endswith("_언급"):
                product_interests.append(msg["keyword"])
        
        # 목적 분석
        purposes = []
        for msg in messages:
            if "목적" in msg.get("type", ""):
                purposes.append(msg["type"].replace("_목적", ""))
        
        # 세그먼트 결정
        if "결혼" in purposes or "약혼" in purposes:
            return {
                "type": "브라이덜 고객",
                "description": "결혼 관련 주얼리를 찾고 있는 고객",
                "recommended_action": "브라이덜 컬렉션 추천, 커플 할인 제안, 맞춤 서비스 안내"
            }
        elif "선물" in purposes:
            return {
                "type": "선물 구매 고객",
                "description": "누군가를 위한 선물을 찾고 있는 고객",
                "recommended_action": "선물 포장 서비스, 가격대별 추천, 교환/반품 정책 안내"
            }
        elif any("가격" in msg.get("type", "") for msg in messages):
            return {
                "type": "가격 민감 고객",
                "description": "가격을 중요하게 고려하는 고객",
                "recommended_action": "할인 이벤트 안내, 분할 결제 옵션 제시, 가성비 제품 추천"
            }
        
        return None
    
    def _identify_product_opportunities(self, messages: List[Dict]) -> List[Dict[str, Any]]:
        """제품 추천 기회 식별"""
        opportunities = []
        
        mentioned_products = []
        for msg in messages:
            if msg.get("type", "").startswith("제품_"):
                mentioned_products.append(msg["keyword"])
        
        if mentioned_products:
            opportunities.append({
                "type": "제품_추천",
                "title": f"💎 관심 제품: {', '.join(mentioned_products)}",
                "description": f"고객이 {', '.join(mentioned_products)}에 관심을 보이고 있습니다",
                "action": f"관련 제품 라인업 소개, 시착 기회 제공, 세트 할인 제안",
                "priority": "높음"
            })
        
        return opportunities
    
    def _assess_risks_and_opportunities(self, intent_analysis: Dict, 
                                      messages: List[Dict]) -> List[Dict[str, Any]]:
        """리스크 및 기회 평가"""
        assessments = []
        
        # 긴급도가 높은 경우
        if intent_analysis["urgency_level"] == "높음":
            assessments.append({
                "type": "기회",
                "title": "⚡ 고전환 기회",
                "description": "지금이 성사 확률이 가장 높은 시점입니다",
                "action": "최고 수준의 서비스 제공, 의사결정권자 즉시 배정",
                "priority": "최우선"
            })
        
        # 고민하고 있는 경우
        if any("고민" in msg.get("type", "") for msg in messages):
            assessments.append({
                "type": "리스크",
                "title": "🤔 이탈 위험",
                "description": "고객이 구매를 망설이고 있어 이탈 가능성이 있습니다",
                "action": "추가 혜택 제공, 전문 상담사 배정, 체험 기회 확대",
                "priority": "높음"
            })
        
        return assessments
    
    def _create_user_friendly_summary(self, messages: List[Dict], intent_analysis: Dict, 
                                    insights: List[Dict]) -> Dict[str, Any]:
        """사용자 친화적 요약 생성"""
        
        # 핵심 한 줄 요약
        main_message = self._generate_one_line_summary(intent_analysis, messages)
        
        # 주요 포인트 (3-5개)
        key_points = self._extract_key_points(messages, insights)
        
        # 추천 액션 (실행 가능한 것들)
        recommended_actions = self._extract_recommended_actions(insights)
        
        # 고객 상태 요약
        customer_status = self._summarize_customer_status(intent_analysis, messages)
        
        return {
            "one_line_summary": main_message,
            "key_points": key_points,
            "customer_status": customer_status,
            "recommended_actions": recommended_actions,
            "urgency_indicator": intent_analysis["urgency_level"],
            "confidence_score": intent_analysis["confidence"]
        }
    
    def _generate_one_line_summary(self, intent_analysis: Dict, messages: List[Dict]) -> str:
        """핵심 한 줄 요약 생성"""
        
        intent = intent_analysis["primary_intent"]
        
        if intent == "구매_의향":
            return "🔥 고객이 구매 의사를 명확히 표현했습니다 - 즉시 상담 진행 필요"
        elif intent == "정보_문의":
            products = [msg["keyword"] for msg in messages if msg.get("type", "").startswith("제품_")]
            if products:
                return f"📋 고객이 {', '.join(products[:2])}에 대한 정보를 요청하고 있습니다"
            else:
                return "📋 고객이 제품 정보를 문의하고 있습니다"
        elif intent == "고민_상담":
            return "🤔 고객이 구매 결정에 도움을 필요로 하고 있습니다 - 상담 지원 필요"
        elif intent == "비교_검토":
            return "⚖️ 고객이 여러 옵션을 비교 검토하고 있습니다 - 차별화 포인트 어필 필요"
        else:
            return "💬 고객과의 일반적인 상담이 진행되고 있습니다"
    
    def _extract_key_points(self, messages: List[Dict], insights: List[Dict]) -> List[str]:
        """주요 포인트 추출"""
        points = []
        
        # 메시지에서 핵심 포인트
        for msg in messages[:3]:  # 상위 3개
            if msg.get("insight"):
                points.append(msg["insight"])
            elif msg.get("context"):
                points.append(f"{msg['type'].replace('_', ' ')}: {msg['context'][:50]}...")
        
        # 인사이트에서 핵심 포인트
        for insight in insights[:2]:  # 상위 2개
            if insight.get("description"):
                points.append(insight["description"])
        
        return points[:5]  # 최대 5개
    
    def _extract_recommended_actions(self, insights: List[Dict]) -> List[str]:
        """추천 액션 추출"""
        actions = []
        
        for insight in insights:
            if insight.get("action"):
                actions.append(f"• {insight['action']}")
        
        return actions[:3]  # 최대 3개
    
    def _summarize_customer_status(self, intent_analysis: Dict, messages: List[Dict]) -> str:
        """고객 상태 요약"""
        
        intent = intent_analysis["primary_intent"]
        confidence = intent_analysis["confidence"]
        
        status_map = {
            "구매_의향": f"🟢 구매 준비 상태 (확신도: {confidence*100:.0f}%)",
            "정보_문의": f"🟡 정보 수집 단계 (관심도: {confidence*100:.0f}%)",
            "고민_상담": f"🟠 의사결정 고민 중 (지원 필요도: {confidence*100:.0f}%)",
            "비교_검토": f"🔵 옵션 비교 검토 중 (검토 깊이: {confidence*100:.0f}%)",
            "일반_대화": f"⚪ 일반 상담 진행 중"
        }
        
        return status_map.get(intent, "⚪ 상태 파악 중")
    
    def _get_speaker_distribution(self, speakers: List[Dict]) -> Dict[str, int]:
        """화자별 발언 비율"""
        distribution = {}
        for speaker_info in speakers:
            speaker = speaker_info["speaker"]
            distribution[speaker] = distribution.get(speaker, 0) + 1
        return distribution
    
    def _get_dominant_speaker(self, speakers: List[Dict]) -> str:
        """주요 화자 식별"""
        distribution = self._get_speaker_distribution(speakers)
        if not distribution:
            return "unknown"
        return max(distribution, key=distribution.get)
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """빈 결과 생성"""
        return {
            "status": "no_content",
            "main_summary": {
                "one_line_summary": "분석할 충분한 내용이 없습니다",
                "key_points": [],
                "customer_status": "⚪ 내용 부족",
                "recommended_actions": ["더 많은 대화 내용 필요"],
                "urgency_indicator": "낮음",
                "confidence_score": 0.0
            },
            "key_messages": [],
            "conversation_analysis": {},
            "analysis_timestamp": datetime.now().isoformat()
        }

# 전역 인스턴스
global_message_extractor = ComprehensiveMessageExtractor()

def extract_comprehensive_messages(text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """간편 메시지 추출 함수"""
    return global_message_extractor.extract_key_messages(text, context)

if __name__ == "__main__":
    # 테스트 코드
    test_text = """
    안녕하세요. 다이아몬드 반지 가격이 궁금해서요. 
    1캐럿 정도로 생각하고 있는데 얼마 정도 할까요?
    결혼 예정이라서 예쁜 걸로 찾고 있어요.
    예산은 500만원 정도 생각하고 있습니다.
    """
    
    extractor = ComprehensiveMessageExtractor()
    result = extractor.extract_key_messages(test_text)
    
    print("=== 메시지 추출 결과 ===")
    summary = result["main_summary"]
    print(f"핵심 요약: {summary['one_line_summary']}")
    print(f"고객 상태: {summary['customer_status']}")
    print("주요 포인트:")
    for point in summary['key_points']:
        print(f"  - {point}")
    print("추천 액션:")
    for action in summary['recommended_actions']:
        print(f"  {action}")