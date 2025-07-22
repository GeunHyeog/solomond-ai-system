#!/usr/bin/env python3
"""
분석 결과 품질 향상 엔진
사용자가 "이 사람들이 무엇을 말하는 것인지 알고 싶다"고 할 때 실제로 의미있는 인사이트를 제공
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import difflib

# 고급 NLP 라이브러리들
try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    transformers_available = True
except ImportError:
    transformers_available = False

try:
    import spacy
    spacy_available = True
except ImportError:
    spacy_available = False

try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    sentence_transformers_available = False

try:
    import google.generativeai as genai
    gemini_available = True
except ImportError:
    gemini_available = False

class AnalysisQualityEnhancer:
    """분석 결과 품질을 대폭 개선하는 엔진"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 고급 모델들 지연 로딩
        self.sentiment_analyzer = None
        self.ner_model = None
        self.semantic_model = None
        self.advanced_summarizer = None
        self.conversation_classifier = None
        
        # 주얼리 도메인 특화 지식베이스
        self.jewelry_knowledge_base = self._build_jewelry_knowledge_base()
        
        # 대화 패턴 분석을 위한 키워드
        self.conversation_patterns = self._build_conversation_patterns()
        
        self.logger.info("🚀 분석 품질 향상 엔진 초기화 완료")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _build_jewelry_knowledge_base(self) -> Dict[str, Any]:
        """주얼리 도메인 특화 지식베이스 구축"""
        return {
            "materials": {
                "precious_metals": ["금", "금도금", "골드", "gold", "은", "실버", "silver", "백금", "플래티넘", "platinum"],
                "gemstones": ["다이아몬드", "diamond", "루비", "ruby", "사파이어", "sapphire", "에메랄드", "emerald", 
                             "진주", "pearl", "오팔", "opal", "토파즈", "topaz", "가넷", "garnet"],
                "other_materials": ["크리스탈", "crystal", "스와로브스키", "swarovski", "큐빅", "cubic", "지르코니아", "zirconia"]
            },
            "products": {
                "jewelry_types": ["반지", "ring", "목걸이", "necklace", "귀걸이", "earring", "팔찌", "bracelet", 
                                "펜던트", "pendant", "브로치", "brooch", "시계", "watch", "체인", "chain"],
                "categories": ["웨딩", "wedding", "약혼", "engagement", "일상", "daily", "파티", "party", 
                              "정장", "formal", "캐주얼", "casual"]
            },
            "quality_terms": {
                "characteristics": ["투명도", "clarity", "컬러", "color", "캐럿", "carat", "커팅", "cut", 
                                  "광택", "luster", "브릴리언스", "brilliance"],
                "certifications": ["GIA", "지아", "인증서", "certificate", "감정서", "appraisal", "보증서", "warranty"]
            },
            "business_terms": {
                "price_related": ["가격", "price", "할인", "discount", "세일", "sale", "프로모션", "promotion", 
                                "비용", "cost", "예산", "budget"],
                "service_related": ["수리", "repair", "리폼", "reform", "맞춤", "custom", "주문제작", "order", 
                                  "배송", "delivery", "A/S", "서비스", "service"]
            }
        }
    
    def _build_conversation_patterns(self) -> Dict[str, List[str]]:
        """대화 패턴 분석을 위한 키워드"""
        return {
            "inquiry": ["문의", "질문", "궁금", "알고 싶", "어떻게", "무엇", "언제", "어디서", "얼마"],
            "purchase_intent": ["구매", "사고 싶", "주문", "예약", "결제", "계산", "카드", "현금"],
            "comparison": ["비교", "차이", "다른", "더 좋은", "추천", "선택", "고민"],
            "complaint": ["불만", "문제", "이상", "안 좋", "실망", "환불", "교환", "AS"],
            "satisfaction": ["만족", "좋다", "예쁘다", "마음에 들", "감사", "추천하고 싶"],
            "technical": ["사양", "규격", "크기", "무게", "재질", "성분", "제조", "원산지"],
            "negotiation": ["할인", "깎아", "가격 조정", "흥정", "더 싸게", "특가", "이벤트"],
            "relationship": ["선물", "기념일", "생일", "결혼", "약혼", "부모님", "연인", "친구"]
        }
    
    def enhance_stt_result(self, original_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """STT 결과 품질 향상"""
        self.logger.info("🎤 STT 결과 품질 향상 시작")
        
        if not original_text or len(original_text.strip()) < 10:
            return {
                "enhanced_text": original_text,
                "improvements": ["텍스트가 너무 짧아 개선할 수 없음"],
                "confidence_score": 0.1
            }
        
        enhanced_text = original_text
        improvements = []
        
        # 1. 한국어 맞춤법 및 문법 보정
        enhanced_text, grammar_improvements = self._improve_korean_grammar(enhanced_text)
        improvements.extend(grammar_improvements)
        
        # 2. 주얼리 전문용어 보정
        enhanced_text, term_improvements = self._correct_jewelry_terms(enhanced_text)
        improvements.extend(term_improvements)
        
        # 3. 컨텍스트 기반 인명/장소명 보정
        if context:
            enhanced_text, context_improvements = self._apply_context_corrections(enhanced_text, context)
            improvements.extend(context_improvements)
        
        # 4. 문장 구조 개선
        enhanced_text, structure_improvements = self._improve_sentence_structure(enhanced_text)
        improvements.extend(structure_improvements)
        
        # 5. 신뢰도 점수 계산
        confidence_score = self._calculate_text_confidence(original_text, enhanced_text)
        
        return {
            "enhanced_text": enhanced_text,
            "original_text": original_text,
            "improvements": improvements,
            "confidence_score": confidence_score,
            "enhancement_type": "stt_quality_boost"
        }
    
    def enhance_ocr_result(self, detected_blocks: List[Dict], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """OCR 결과 품질 향상"""
        self.logger.info("🖼️ OCR 결과 품질 향상 시작")
        
        if not detected_blocks:
            return {
                "enhanced_text": "",
                "improvements": ["OCR 블록이 없음"],
                "confidence_score": 0.0
            }
        
        improvements = []
        enhanced_blocks = []
        
        # 1. 신뢰도 기반 필터링
        filtered_blocks = [block for block in detected_blocks if block.get('confidence', 0) > 0.3]
        if len(filtered_blocks) < len(detected_blocks):
            improvements.append(f"낮은 신뢰도 블록 {len(detected_blocks) - len(filtered_blocks)}개 제거")
        
        # 2. 공간적 정렬 (상하좌우 순서로 정렬)
        sorted_blocks = self._sort_ocr_blocks_spatially(filtered_blocks)
        if sorted_blocks != filtered_blocks:
            improvements.append("텍스트 블록 공간적 순서 정렬")
        
        # 3. 텍스트 병합 및 정제
        for block in sorted_blocks:
            enhanced_text = block.get('text', '')
            
            # 주얼리 용어 보정
            enhanced_text, term_improvements = self._correct_jewelry_terms(enhanced_text)
            
            # 숫자/가격 정보 보정
            enhanced_text, number_improvements = self._correct_numbers_and_prices(enhanced_text)
            
            enhanced_blocks.append({
                **block,
                'enhanced_text': enhanced_text,
                'improvements': term_improvements + number_improvements
            })
            
            improvements.extend(term_improvements + number_improvements)
        
        # 4. 전체 텍스트 조합
        full_enhanced_text = ' '.join([block['enhanced_text'] for block in enhanced_blocks if block.get('enhanced_text')])
        
        # 5. 컨텍스트 적용
        if context:
            full_enhanced_text, context_improvements = self._apply_context_corrections(full_enhanced_text, context)
            improvements.extend(context_improvements)
        
        # 6. 신뢰도 계산
        original_text = ' '.join([block.get('text', '') for block in detected_blocks])
        confidence_score = self._calculate_text_confidence(original_text, full_enhanced_text)
        
        return {
            "enhanced_text": full_enhanced_text,
            "enhanced_blocks": enhanced_blocks,
            "improvements": improvements,
            "confidence_score": confidence_score,
            "enhancement_type": "ocr_quality_boost"
        }
    
    def generate_meaningful_summary(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """의미있는 요약 생성 - "이 사람들이 무엇을 말하는지" 이해할 수 있는 수준"""
        self.logger.info("🧠 의미있는 요약 생성 시작")
        
        if not text or len(text.strip()) < 50:
            return {
                "executive_summary": "분석할 내용이 너무 짧습니다.",
                "key_points": [],
                "main_topics": [],
                "insights": [],
                "confidence": 0.1
            }
        
        # 1. 대화 패턴 분석
        conversation_analysis = self._analyze_conversation_patterns(text)
        
        # 2. 핵심 주제 추출
        main_topics = self._extract_main_topics(text, context)
        
        # 3. 핵심 인사이트 도출
        key_insights = self._derive_key_insights(text, conversation_analysis, main_topics, context)
        
        # 4. 실용적 요약 생성
        executive_summary = self._generate_executive_summary(text, main_topics, key_insights, context)
        
        # 5. 액션 아이템 추출
        action_items = self._extract_action_items(text, conversation_analysis)
        
        return {
            "executive_summary": executive_summary,
            "main_topics": main_topics,
            "key_insights": key_insights,
            "conversation_patterns": conversation_analysis,
            "action_items": action_items,
            "confidence": self._calculate_summary_confidence(text, main_topics, key_insights),
            "enhancement_type": "meaningful_summary"
        }
    
    def _improve_korean_grammar(self, text: str) -> Tuple[str, List[str]]:
        """한국어 맞춤법 및 문법 보정"""
        improvements = []
        enhanced_text = text
        
        # 자주 틀리는 한국어 표현 보정
        corrections = {
            # 맞춤법 오류
            "됬다": "됐다", "됬습니다": "됐습니다", "됬어요": "됐어요",
            "맞추다": "맞추다", "맞춘다": "맞춘다",
            "어떻해": "어떻게", "어떻케": "어떻게",
            "얼마에요": "얼마예요", "뭐에요": "뭐예요",
            
            # 주얼리 관련 자주 틀리는 표현
            "다이아": "다이아몬드", "다이야": "다이아몬드",
            "골드": "금", "실버": "은",
            "캐롯": "캐럿", "케럿": "캐럿",
            "펜덴트": "펜던트", "펜던드": "펜던트"
        }
        
        for wrong, correct in corrections.items():
            if wrong in enhanced_text:
                enhanced_text = enhanced_text.replace(wrong, correct)
                improvements.append(f"맞춤법 보정: '{wrong}' → '{correct}'")
        
        return enhanced_text, improvements
    
    def _correct_jewelry_terms(self, text: str) -> Tuple[str, List[str]]:
        """주얼리 전문용어 보정"""
        improvements = []
        enhanced_text = text
        
        # 주얼리 용어 정규화
        jewelry_corrections = {
            # 재질 관련
            "14k": "14K", "18k": "18K", "24k": "24K",
            "골드필드": "골드 필드", "로즈골드": "로즈 골드",
            "화이트골드": "화이트 골드", "옐로우골드": "옐로우 골드",
            
            # 보석 관련
            "cz": "CZ", "큐빅지르코니아": "큐빅 지르코니아",
            "스와로브스키": "스와로브스키", "크리스털": "크리스탈",
            
            # 제품 관련
            "이어링": "귀걸이", "네클리스": "목걸이",
            "브레이슬릿": "팔찌", "링": "반지",
            
            # 품질 관련
            "vvs": "VVS", "vs": "VS", "si": "SI",
            "gia": "GIA", "지아인증": "GIA 인증"
        }
        
        for wrong, correct in jewelry_corrections.items():
            if wrong.lower() in enhanced_text.lower():
                enhanced_text = re.sub(rf"\b{re.escape(wrong)}\b", correct, enhanced_text, flags=re.IGNORECASE)
                improvements.append(f"전문용어 보정: '{wrong}' → '{correct}'")
        
        return enhanced_text, improvements
    
    def _apply_context_corrections(self, text: str, context: Dict[str, Any]) -> Tuple[str, List[str]]:
        """컨텍스트 기반 보정"""
        improvements = []
        enhanced_text = text
        
        # 참석자/발표자 이름 보정
        if context.get('participants') or context.get('speakers'):
            names = []
            if context.get('speakers'):
                names.extend([n.strip() for n in context['speakers'].split(',')])
            if context.get('participants'):
                participant_names = re.findall(r'([가-힣a-zA-Z\s]+?)(?:\s*\([^)]*\))?(?:,|$)', context['participants'])
                names.extend([n.strip() for n in participant_names if n.strip()])
            
            # 이름 유사도 기반 보정
            for name in names:
                if name and len(name) > 1:
                    words = enhanced_text.split()
                    for i, word in enumerate(words):
                        if difflib.SequenceMatcher(None, word, name).ratio() > 0.7:
                            words[i] = name
                            improvements.append(f"인명 보정: '{word}' → '{name}'")
                    enhanced_text = ' '.join(words)
        
        # 주제 키워드 강화
        if context.get('topic_keywords'):
            keywords = [k.strip() for k in context['topic_keywords'].split(',')]
            for keyword in keywords:
                if keyword and len(keyword) > 2:
                    words = enhanced_text.split()
                    for i, word in enumerate(words):
                        if difflib.SequenceMatcher(None, word.lower(), keyword.lower()).ratio() > 0.8:
                            words[i] = keyword
                            improvements.append(f"키워드 보정: '{word}' → '{keyword}'")
                    enhanced_text = ' '.join(words)
        
        return enhanced_text, improvements
    
    def _improve_sentence_structure(self, text: str) -> Tuple[str, List[str]]:
        """문장 구조 개선"""
        improvements = []
        enhanced_text = text
        
        # 불완전한 문장 보완
        sentences = re.split(r'[.!?]\s*', enhanced_text)
        improved_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 주어 없는 문장 보완
            if len(sentence) > 10 and not re.search(r'[가-힣]+[은는이가]', sentence):
                if any(term in sentence for term in ['가격', '비용', '할인']):
                    sentence = "가격이 " + sentence
                    improvements.append("주어 보완: 가격 관련")
                elif any(term in sentence for term in ['제품', '상품', '주얼리']):
                    sentence = "제품이 " + sentence
                    improvements.append("주어 보완: 제품 관련")
            
            improved_sentences.append(sentence)
        
        enhanced_text = '. '.join(improved_sentences)
        if enhanced_text and not enhanced_text.endswith('.'):
            enhanced_text += '.'
        
        return enhanced_text, improvements
    
    def _sort_ocr_blocks_spatially(self, blocks: List[Dict]) -> List[Dict]:
        """OCR 블록을 공간적 순서로 정렬"""
        # bbox가 있는 경우 상하좌우 순서로 정렬
        def get_position(block):
            bbox = block.get('bbox', [])
            if len(bbox) >= 4:
                # bbox가 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 형태인 경우
                if isinstance(bbox[0], list):
                    y_center = sum(point[1] for point in bbox) / len(bbox)
                    x_center = sum(point[0] for point in bbox) / len(bbox)
                else:
                    # bbox가 [x1, y1, x2, y2] 형태인 경우
                    y_center = (bbox[1] + bbox[3]) / 2
                    x_center = (bbox[0] + bbox[2]) / 2
                return (int(y_center // 50) * 1000 + x_center)  # 세로 우선, 가로 보조 정렬
            return 0
        
        try:
            return sorted(blocks, key=get_position)
        except:
            return blocks  # 정렬 실패시 원본 반환
    
    def _correct_numbers_and_prices(self, text: str) -> Tuple[str, List[str]]:
        """숫자 및 가격 정보 보정"""
        improvements = []
        enhanced_text = text
        
        # 가격 표기 정규화
        price_patterns = [
            (r'(\d+)원', r'\1원'),  # 숫자+원
            (r'(\d+),(\d{3})', r'\1,\2'),  # 천 단위 쉼표
            (r'(\d+)만원', r'\1만원'),  # 만원 단위
            (r'(\d+)k', r'\1K'),  # K 표기 통일
        ]
        
        for pattern, replacement in price_patterns:
            if re.search(pattern, enhanced_text):
                enhanced_text = re.sub(pattern, replacement, enhanced_text)
                improvements.append("가격 표기 정규화")
        
        return enhanced_text, improvements
    
    def _analyze_conversation_patterns(self, text: str) -> Dict[str, Any]:
        """대화 패턴 분석"""
        analysis = {
            "dominant_patterns": [],
            "conversation_type": "general",
            "intent_scores": {},
            "emotional_tone": "neutral"
        }
        
        text_lower = text.lower()
        
        # 각 패턴별 점수 계산
        for pattern_name, keywords in self.conversation_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                analysis["intent_scores"][pattern_name] = score
        
        # 주요 패턴 결정
        if analysis["intent_scores"]:
            dominant_pattern = max(analysis["intent_scores"], key=analysis["intent_scores"].get)
            analysis["dominant_patterns"] = [dominant_pattern]
            analysis["conversation_type"] = dominant_pattern
        
        # 감정 톤 분석 (간단한 규칙 기반)
        positive_words = ["좋다", "예쁘다", "만족", "감사", "추천"]
        negative_words = ["안 좋", "실망", "문제", "불만", "환불"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            analysis["emotional_tone"] = "positive"
        elif negative_count > positive_count:
            analysis["emotional_tone"] = "negative"
        
        return analysis
    
    def _extract_main_topics(self, text: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """핵심 주제 추출"""
        topics = []
        text_lower = text.lower()
        
        # 주얼리 관련 주제 식별
        for category, terms in self.jewelry_knowledge_base.items():
            if isinstance(terms, dict):
                for subcategory, term_list in terms.items():
                    mentions = [term for term in term_list if term.lower() in text_lower]
                    if mentions:
                        topics.append({
                            "category": category,
                            "subcategory": subcategory,
                            "mentioned_terms": mentions,
                            "relevance_score": len(mentions)
                        })
        
        # 관련도 기준으로 정렬
        topics.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return topics[:5]  # 상위 5개 주제만 반환
    
    def _derive_key_insights(self, text: str, conversation_analysis: Dict, main_topics: List, context: Dict = None) -> List[str]:
        """핵심 인사이트 도출"""
        insights = []
        
        # 대화 패턴 기반 인사이트
        dominant_pattern = conversation_analysis.get("conversation_type", "general")
        
        if dominant_pattern == "inquiry":
            insights.append("💡 고객이 제품에 대한 구체적인 정보를 원하고 있습니다.")
        elif dominant_pattern == "purchase_intent":
            insights.append("🛒 구매 의향이 높은 상태입니다. 구매 결정을 도울 정보 제공이 필요합니다.")
        elif dominant_pattern == "comparison":
            insights.append("⚖️ 여러 옵션을 비교 검토 중입니다. 차별점 설명이 중요합니다.")
        elif dominant_pattern == "complaint":
            insights.append("⚠️ 불만사항이나 문제점에 대한 해결이 필요합니다.")
        elif dominant_pattern == "satisfaction":
            insights.append("😊 제품에 만족하고 있으며, 재구매나 추천 가능성이 높습니다.")
        
        # 주제 기반 인사이트
        if main_topics:
            top_topic = main_topics[0]
            if top_topic["category"] == "materials":
                insights.append(f"🔍 {', '.join(top_topic['mentioned_terms'])} 재질에 대한 관심이 높습니다.")
            elif top_topic["category"] == "products":
                insights.append(f"👑 {', '.join(top_topic['mentioned_terms'])} 제품군이 주요 관심사입니다.")
            elif top_topic["category"] == "business_terms":
                insights.append("💰 가격이나 서비스 조건이 중요한 결정 요소입니다.")
        
        # 컨텍스트 기반 인사이트
        if context and context.get('objective'):
            insights.append(f"🎯 '{context['objective']}' 목적으로 진행된 대화입니다.")
        
        return insights
    
    def _generate_executive_summary(self, text: str, main_topics: List, key_insights: List, context: Dict = None) -> str:
        """실무진이 이해할 수 있는 경영진 요약"""
        
        # 기본 요약 템플릿
        summary_parts = []
        
        # 1. 상황 요약
        if context and context.get('event_context'):
            summary_parts.append(f"📍 **상황**: {context['event_context']}")
        
        # 2. 주요 논의 사항
        if main_topics:
            top_categories = list(set([topic["category"] for topic in main_topics[:3]]))
            category_names = {
                "materials": "재질 및 소재",
                "products": "제품 종류",
                "quality_terms": "품질 기준",
                "business_terms": "비즈니스 조건"
            }
            discussed_items = [category_names.get(cat, cat) for cat in top_categories]
            summary_parts.append(f"💼 **주요 논의**: {', '.join(discussed_items)}")
        
        # 3. 핵심 인사이트 (상위 2개)
        if key_insights:
            summary_parts.append(f"🔑 **핵심 포인트**: {key_insights[0]}")
            if len(key_insights) > 1:
                summary_parts.append(f"💡 **추가 인사이트**: {key_insights[1]}")
        
        # 4. 결론 또는 다음 액션
        text_lower = text.lower()
        if any(word in text_lower for word in ["결정", "선택", "구매", "주문"]):
            summary_parts.append("✅ **상태**: 구매 결정 단계 진입")
        elif any(word in text_lower for word in ["고민", "검토", "생각"]):
            summary_parts.append("🤔 **상태**: 추가 검토 필요")
        elif any(word in text_lower for word in ["문의", "질문", "궁금"]):
            summary_parts.append("❓ **상태**: 추가 정보 제공 필요")
        
        return "\n".join(summary_parts) if summary_parts else "대화 내용을 분석했으나 명확한 패턴을 찾기 어렵습니다."
    
    def _extract_action_items(self, text: str, conversation_analysis: Dict) -> List[str]:
        """실행 가능한 액션 아이템 추출"""
        actions = []
        text_lower = text.lower()
        
        # 문의 사항 기반 액션
        if "문의" in text_lower or "질문" in text_lower:
            actions.append("📞 고객 문의사항에 대한 상세 답변 제공")
        
        # 가격 관련 액션
        if any(word in text_lower for word in ["가격", "할인", "비용"]):
            actions.append("💰 가격 정보 및 할인 혜택 안내")
        
        # 제품 정보 관련 액션
        if any(word in text_lower for word in ["사양", "규격", "재질"]):
            actions.append("📋 제품 상세 스펙 및 인증서 제공")
        
        # 구매 관련 액션
        if conversation_analysis.get("conversation_type") == "purchase_intent":
            actions.append("🛒 구매 프로세스 안내 및 결제 방법 설명")
        
        # 비교 관련 액션
        if conversation_analysis.get("conversation_type") == "comparison":
            actions.append("⚖️ 제품 비교표 및 차별점 자료 준비")
        
        return actions
    
    def _calculate_text_confidence(self, original: str, enhanced: str) -> float:
        """텍스트 신뢰도 계산"""
        if not original or not enhanced:
            return 0.0
        
        # 기본 신뢰도는 개선 정도에 따라 결정
        similarity = difflib.SequenceMatcher(None, original, enhanced).ratio()
        
        # 너무 많이 바뀌면 신뢰도 낮음, 적절히 개선되면 신뢰도 높음
        if similarity > 0.9:
            confidence = 0.95  # 거의 변화 없음 = 원본이 좋았음
        elif similarity > 0.7:
            confidence = 0.85  # 적절한 개선
        elif similarity > 0.5:
            confidence = 0.70  # 상당한 개선
        else:
            confidence = 0.50  # 대폭 수정 = 원본 품질 낮음
        
        return confidence
    
    def _calculate_summary_confidence(self, text: str, main_topics: List, key_insights: List) -> float:
        """요약 신뢰도 계산"""
        confidence = 0.5  # 기본값
        
        # 텍스트 길이에 따른 신뢰도
        if len(text) > 500:
            confidence += 0.2
        elif len(text) > 200:
            confidence += 0.1
        
        # 주제 식별 성공도
        if len(main_topics) >= 3:
            confidence += 0.2
        elif len(main_topics) >= 1:
            confidence += 0.1
        
        # 인사이트 도출 성공도
        if len(key_insights) >= 2:
            confidence += 0.1
        
        return min(1.0, confidence)

# 전역 품질 향상 엔진 인스턴스
global_quality_enhancer = AnalysisQualityEnhancer()

def enhance_analysis_quality(analysis_result: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """분석 결과 품질 통합 개선"""
    if analysis_result.get('status') != 'success':
        return analysis_result
    
    enhanced_result = analysis_result.copy()
    
    # STT 결과 개선
    if 'full_text' in analysis_result:
        stt_enhancement = global_quality_enhancer.enhance_stt_result(
            analysis_result['full_text'], context
        )
        enhanced_result['enhanced_text'] = stt_enhancement['enhanced_text']
        enhanced_result['text_improvements'] = stt_enhancement['improvements']
        enhanced_result['text_confidence'] = stt_enhancement['confidence_score']
        
        # 개선된 텍스트로 요약 재생성
        summary_enhancement = global_quality_enhancer.generate_meaningful_summary(
            stt_enhancement['enhanced_text'], context
        )
        enhanced_result['meaningful_summary'] = summary_enhancement
    
    # OCR 결과 개선
    elif 'detailed_results' in analysis_result:
        ocr_enhancement = global_quality_enhancer.enhance_ocr_result(
            analysis_result['detailed_results'], context
        )
        enhanced_result['enhanced_text'] = ocr_enhancement['enhanced_text']
        enhanced_result['enhanced_blocks'] = ocr_enhancement['enhanced_blocks']
        enhanced_result['ocr_improvements'] = ocr_enhancement['improvements']
        enhanced_result['ocr_confidence'] = ocr_enhancement['confidence_score']
        
        # 개선된 텍스트로 요약 재생성
        summary_enhancement = global_quality_enhancer.generate_meaningful_summary(
            ocr_enhancement['enhanced_text'], context
        )
        enhanced_result['meaningful_summary'] = summary_enhancement
    
    # 기존 요약이 있다면 의미있는 요약으로 교체
    if 'summary' in analysis_result and 'meaningful_summary' in enhanced_result:
        enhanced_result['original_summary'] = analysis_result['summary']
        enhanced_result['summary'] = enhanced_result['meaningful_summary']['executive_summary']
    
    enhanced_result['quality_enhanced'] = True
    enhanced_result['enhancement_timestamp'] = datetime.now().isoformat()
    
    return enhanced_result

if __name__ == "__main__":
    # 테스트 실행
    print("🧪 분석 품질 향상 엔진 테스트")
    
    # 테스트 데이터
    test_stt = "어 이게 다이아 반지인가요? 가격이 얼마에요? 할인도 되나요?"
    test_context = {
        "event_context": "주얼리 매장 상담",
        "objective": "반지 구매 상담",
        "participants": "고객(김영희), 직원(이철수)"
    }
    
    enhancer = AnalysisQualityEnhancer()
    
    # STT 개선 테스트
    stt_result = enhancer.enhance_stt_result(test_stt, test_context)
    print(f"원본: {test_stt}")
    print(f"개선: {stt_result['enhanced_text']}")
    print(f"개선사항: {stt_result['improvements']}")
    
    # 요약 생성 테스트
    summary_result = enhancer.generate_meaningful_summary(stt_result['enhanced_text'], test_context)
    print(f"\n요약: {summary_result['executive_summary']}")
    print(f"인사이트: {summary_result['key_insights']}")