#!/usr/bin/env python3
"""
종합 메시지 추출 엔진 - 클로바 노트 + ChatGPT 수준의 분석 시스템
강연자가 전달하고자 하는 핵심 메시지를 정확히 파악하는 시스템
"""

import os
import re
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

# 고급 언어 모델들
try:
    import openai
    openai_available = True
except ImportError:
    openai_available = False

try:
    import google.generativeai as genai
    gemini_available = True
except ImportError:
    gemini_available = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    transformers_available = True
except ImportError:
    transformers_available = False

class ComprehensiveMessageExtractor:
    """종합 메시지 추출 엔진 - 강연자의 의도를 정확히 파악"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 고급 언어 모델 초기화
        self.advanced_llm = None
        self.korean_llm = None
        self.context_analyzer = None
        
        # 강연/프레젠테이션 특화 지식
        self.presentation_patterns = self._build_presentation_patterns()
        self.message_extraction_rules = self._build_message_extraction_rules()
        self.context_enhancement_rules = self._build_context_enhancement_rules()
        
        self.logger.info("🎯 종합 메시지 추출 엔진 초기화 완료")
    
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
    
    def _build_presentation_patterns(self) -> Dict[str, Any]:
        """프레젠테이션 패턴 분석 규칙"""
        return {
            "opening_patterns": [
                "오늘 말씀드릴", "발표할", "소개하겠습니다", "시작하겠습니다",
                "주제는", "테마는", "다루고자", "살펴보겠습니다"
            ],
            "key_point_indicators": [
                "중요한 것은", "핵심은", "포인트는", "강조하고 싶은",
                "기억해야 할", "주목할", "특히", "무엇보다도"
            ],
            "transition_phrases": [
                "다음으로", "이어서", "그리고", "또한", "한편",
                "반면에", "그러나", "결론적으로", "마지막으로"
            ],
            "emphasis_markers": [
                "반드시", "꼭", "절대", "매우", "정말", "진짜",
                "확실히", "분명히", "당연히", "물론"
            ],
            "conclusion_patterns": [
                "결론은", "요약하면", "정리하자면", "마무리하며",
                "끝으로", "마지막으로", "결과적으로", "따라서"
            ],
            "question_patterns": [
                "궁금하시죠?", "어떻게 생각하세요?", "질문이 있으실까요?",
                "이해되시나요?", "맞죠?", "그렇지 않나요?"
            ]
        }
    
    def _build_message_extraction_rules(self) -> Dict[str, Any]:
        """메시지 추출 규칙"""
        return {
            "primary_message_weights": {
                "title_mentions": 3.0,      # 제목/주제 언급
                "key_indicators": 2.5,      # 핵심 지시어
                "emphasis_markers": 2.0,    # 강조 표현
                "conclusion_statements": 2.5, # 결론 진술
                "repetition": 1.5,          # 반복된 내용
                "question_answers": 2.0     # 질문-답변 패턴
            },
            "supporting_message_weights": {
                "examples": 1.5,            # 예시/사례
                "statistics": 2.0,          # 통계/수치
                "quotes": 1.8,              # 인용
                "analogies": 1.3,           # 비유/은유
                "stories": 1.4              # 스토리/일화
            },
            "context_modifiers": {
                "time_references": 1.2,     # 시간 언급
                "place_references": 1.1,    # 장소 언급
                "people_references": 1.3,   # 인물 언급
                "data_references": 1.8      # 데이터 언급
            }
        }
    
    def _build_context_enhancement_rules(self) -> Dict[str, Any]:
        """컨텍스트 향상 규칙"""
        return {
            "slide_context_clues": [
                "슬라이드", "화면", "보시는 바와 같이", "그림", "표", "차트",
                "다음 페이지", "이 부분을", "여기서", "이것은"
            ],
            "audience_interaction": [
                "여러분", "청중", "참석자", "고객", "동료", "팀원",
                "함께", "같이", "우리가", "모두"
            ],
            "temporal_markers": [
                "과거에", "현재", "미래에", "지금", "이전", "다음",
                "올해", "내년", "최근", "앞으로"
            ],
            "causality_markers": [
                "따라서", "그래서", "결과적으로", "이로 인해", "때문에",
                "덕분에", "의해", "으로부터", "에서 비롯된"
            ]
        }
    
    def extract_comprehensive_message(self, multimodal_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """다중 모달 데이터에서 종합 메시지 추출"""
        self.logger.info("🎯 종합 메시지 추출 시작")
        
        # 1. 다중 모달 데이터 통합
        integrated_content = self._integrate_multimodal_content(multimodal_data)
        
        # 2. 컨텍스트 강화 전처리
        enhanced_content = self._enhance_context(integrated_content, context)
        
        # 3. 강연자 의도 분석
        speaker_intent = self._analyze_speaker_intent(enhanced_content)
        
        # 4. 핵심 메시지 추출
        key_messages = self._extract_key_messages(enhanced_content, speaker_intent)
        
        # 5. 메시지 계층 구조화
        message_hierarchy = self._structure_message_hierarchy(key_messages)
        
        # 6. 실행 가능한 인사이트 도출
        actionable_insights = self._derive_actionable_insights(message_hierarchy, context)
        
        # 7. 클로바 노트 스타일 요약 생성
        clova_style_summary = self._generate_clova_style_summary(message_hierarchy, speaker_intent)
        
        return {
            "comprehensive_analysis": {
                "speaker_intent": speaker_intent,
                "key_messages": key_messages,
                "message_hierarchy": message_hierarchy,
                "actionable_insights": actionable_insights,
                "clova_style_summary": clova_style_summary
            },
            "technical_details": {
                "integrated_content": integrated_content,
                "enhancement_applied": enhanced_content != integrated_content,
                "processing_time": time.time(),
                "confidence_score": self._calculate_overall_confidence(message_hierarchy)
            }
        }
    
    def _integrate_multimodal_content(self, multimodal_data: Dict[str, Any]) -> Dict[str, Any]:
        """다중 모달 데이터 통합"""
        integrated = {
            "audio_content": "",
            "visual_content": "",
            "temporal_sync": [],
            "metadata": {}
        }
        
        # 음성 데이터 통합
        if 'audio_analysis' in multimodal_data:
            audio = multimodal_data['audio_analysis']
            if audio.get('status') == 'success':
                # 향상된 텍스트 우선 사용
                integrated["audio_content"] = audio.get('enhanced_text', audio.get('full_text', ''))
                
                # 시간대별 세그먼트 정보
                if audio.get('segments'):
                    for segment in audio['segments']:
                        integrated["temporal_sync"].append({
                            "type": "audio",
                            "start": segment.get('start', 0),
                            "end": segment.get('end', 0),
                            "content": segment.get('text', ''),
                            "confidence": segment.get('avg_logprob', 0)
                        })
        
        # 시각 데이터 통합 (OCR + 키프레임)
        visual_texts = []
        
        # 이미지 OCR 결과
        if 'image_analysis' in multimodal_data:
            for image_result in multimodal_data['image_analysis']:
                if image_result.get('status') == 'success':
                    visual_texts.append(image_result.get('enhanced_text', image_result.get('full_text', '')))
        
        # 비디오 키프레임 OCR 결과
        if 'video_analysis' in multimodal_data:
            video = multimodal_data['video_analysis']
            if video.get('visual_analysis', {}).get('status') == 'success':
                visual_analysis = video['visual_analysis']
                combined_visual = visual_analysis.get('combined_visual_text', '')
                if combined_visual:
                    visual_texts.append(combined_visual)
                
                # 시간대별 시각 정보
                if visual_analysis.get('frame_details'):
                    for frame in visual_analysis['frame_details']:
                        if frame.get('enhanced_text', '').strip():
                            integrated["temporal_sync"].append({
                                "type": "visual",
                                "timestamp": frame.get('timestamp_seconds', 0),
                                "content": frame['enhanced_text'],
                                "confidence": frame.get('average_confidence', 0)
                            })
        
        integrated["visual_content"] = ' '.join(filter(None, visual_texts))
        
        # 메타데이터 통합
        integrated["metadata"] = {
            "has_audio": bool(integrated["audio_content"]),
            "has_visual": bool(integrated["visual_content"]),
            "temporal_mappings": len(integrated["temporal_sync"]),
            "content_richness": self._assess_content_richness(integrated)
        }
        
        return integrated
    
    def _enhance_context(self, integrated_content: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """컨텍스트 강화 처리"""
        enhanced = integrated_content.copy()
        
        if not context:
            return enhanced
        
        # 1. 참석자/발표자 정보 활용
        if context.get('speakers') or context.get('participants'):
            enhanced = self._apply_speaker_context(enhanced, context)
        
        # 2. 이벤트 컨텍스트 활용
        if context.get('event_context'):
            enhanced = self._apply_event_context(enhanced, context)
        
        # 3. 주제 키워드 강화
        if context.get('topic_keywords'):
            enhanced = self._apply_topic_enhancement(enhanced, context)
        
        # 4. 목적 기반 필터링
        if context.get('objective'):
            enhanced = self._apply_objective_filtering(enhanced, context)
        
        return enhanced
    
    def _analyze_speaker_intent(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """강연자 의도 분석"""
        audio_text = content.get("audio_content", "")
        visual_text = content.get("visual_content", "")
        combined_text = f"{audio_text} {visual_text}".strip()
        
        if not combined_text:
            return {"intent_type": "unknown", "confidence": 0.0}
        
        intent_analysis = {
            "primary_intent": self._identify_primary_intent(combined_text),
            "communication_style": self._analyze_communication_style(combined_text),
            "audience_engagement": self._assess_audience_engagement(combined_text),
            "content_structure": self._analyze_content_structure(combined_text),
            "emotional_tone": self._analyze_emotional_tone(combined_text)
        }
        
        return intent_analysis
    
    def _extract_key_messages(self, content: Dict[str, Any], speaker_intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """핵심 메시지 추출"""
        audio_text = content.get("audio_content", "")
        visual_text = content.get("visual_content", "")
        
        key_messages = []
        
        # 1. 음성에서 핵심 메시지 추출
        if audio_text:
            audio_messages = self._extract_messages_from_text(audio_text, "audio")
            key_messages.extend(audio_messages)
        
        # 2. 시각 자료에서 핵심 메시지 추출
        if visual_text:
            visual_messages = self._extract_messages_from_text(visual_text, "visual")
            key_messages.extend(visual_messages)
        
        # 3. 시간 동기화 정보 활용
        if content.get("temporal_sync"):
            temporal_messages = self._extract_temporal_messages(content["temporal_sync"])
            key_messages.extend(temporal_messages)
        
        # 4. 메시지 중요도 계산 및 정렬
        for message in key_messages:
            message["importance_score"] = self._calculate_message_importance(message, speaker_intent)
        
        # 중요도 기준 정렬
        key_messages.sort(key=lambda x: x["importance_score"], reverse=True)
        
        return key_messages[:10]  # 상위 10개 메시지
    
    def _extract_messages_from_text(self, text: str, source_type: str) -> List[Dict[str, Any]]:
        """텍스트에서 메시지 추출"""
        messages = []
        sentences = re.split(r'[.!?]\s+', text)
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:
                continue
            
            # 메시지 후보 점수 계산
            score = 0
            matched_patterns = []
            
            # 패턴 매칭
            for pattern_type, patterns in self.presentation_patterns.items():
                for pattern in patterns:
                    if pattern in sentence:
                        weight = self.message_extraction_rules["primary_message_weights"].get(pattern_type, 1.0)
                        score += weight
                        matched_patterns.append(f"{pattern_type}:{pattern}")
            
            # 일정 점수 이상만 메시지로 간주
            if score >= 1.5:
                messages.append({
                    "content": sentence.strip(),
                    "source_type": source_type,
                    "position": i,
                    "raw_score": score,
                    "matched_patterns": matched_patterns,
                    "message_type": self._classify_message_type(sentence)
                })
        
        return messages
    
    def _extract_temporal_messages(self, temporal_sync: List[Dict]) -> List[Dict[str, Any]]:
        """시간 동기화 정보에서 메시지 추출"""
        messages = []
        
        # 시간대별 그룹화
        time_groups = {}
        for item in temporal_sync:
            time_key = int(item.get('timestamp', item.get('start', 0)) // 30)  # 30초 단위
            if time_key not in time_groups:
                time_groups[time_key] = []
            time_groups[time_key].append(item)
        
        # 각 그룹에서 메시지 추출
        for time_key, group in time_groups.items():
            audio_content = " ".join([item['content'] for item in group if item['type'] == 'audio'])
            visual_content = " ".join([item['content'] for item in group if item['type'] == 'visual'])
            
            if audio_content and visual_content:
                # 음성과 시각 정보가 모두 있는 경우
                combined_content = f"{audio_content} [화면: {visual_content}]"
                messages.append({
                    "content": combined_content,
                    "source_type": "multimodal",
                    "timestamp": time_key * 30,
                    "message_type": "synchronized",
                    "raw_score": 2.0  # 멀티모달 보너스
                })
        
        return messages
    
    def _structure_message_hierarchy(self, key_messages: List[Dict]) -> Dict[str, Any]:
        """메시지 계층 구조화"""
        hierarchy = {
            "main_theme": None,
            "key_points": [],
            "supporting_details": [],
            "conclusions": [],
            "call_to_actions": []
        }
        
        for message in key_messages:
            msg_type = message.get("message_type", "general")
            importance = message.get("importance_score", 0)
            
            if importance >= 4.0:
                if not hierarchy["main_theme"]:
                    hierarchy["main_theme"] = message
                else:
                    hierarchy["key_points"].append(message)
            elif importance >= 2.5:
                hierarchy["key_points"].append(message)
            elif importance >= 1.5:
                hierarchy["supporting_details"].append(message)
            
            # 메시지 타입별 분류
            if msg_type == "conclusion":
                hierarchy["conclusions"].append(message)
            elif msg_type == "action":
                hierarchy["call_to_actions"].append(message)
        
        return hierarchy
    
    def _generate_clova_style_summary(self, message_hierarchy: Dict, speaker_intent: Dict) -> Dict[str, Any]:
        """클로바 노트 스타일 요약 생성"""
        
        # 1. 핵심 메시지 (클로바 노트의 "요약" 섹션)
        main_summary = ""
        if message_hierarchy.get("main_theme"):
            main_summary = message_hierarchy["main_theme"]["content"]
        elif message_hierarchy.get("key_points"):
            main_summary = message_hierarchy["key_points"][0]["content"]
        
        # 2. 주요 포인트 (클로바 노트의 "키워드" 섹션)
        key_points = []
        for point in message_hierarchy.get("key_points", [])[:5]:
            key_points.append({
                "point": point["content"][:100] + "..." if len(point["content"]) > 100 else point["content"],
                "importance": point.get("importance_score", 0)
            })
        
        # 3. 실행 항목 (클로바 노트의 "액션 아이템" 섹션)
        action_items = []
        for action in message_hierarchy.get("call_to_actions", []):
            action_items.append(action["content"])
        
        # 4. 인사이트 (클로바 노트의 "인사이트" 섹션)
        insights = self._generate_insights_from_intent(speaker_intent)
        
        return {
            "executive_summary": main_summary,
            "key_takeaways": key_points,
            "action_items": action_items,
            "speaker_insights": insights,
            "presentation_structure": {
                "style": speaker_intent.get("communication_style", {}),
                "engagement_level": speaker_intent.get("audience_engagement", {}),
                "emotional_tone": speaker_intent.get("emotional_tone", {})
            },
            "clova_compatibility_score": self._calculate_clova_compatibility(message_hierarchy, speaker_intent)
        }
    
    def _identify_primary_intent(self, text: str) -> Dict[str, Any]:
        """주요 의도 식별"""
        intent_indicators = {
            "inform": ["설명", "소개", "알려드리", "보여드리", "말씀드리"],
            "persuade": ["설득", "제안", "추천", "권유", "선택"],
            "educate": ["교육", "학습", "배우", "이해", "습득"],
            "inspire": ["영감", "동기", "격려", "응원", "자극"],
            "demonstrate": ["시연", "보여주", "데모", "실습", "실제"],
            "analyze": ["분석", "검토", "평가", "비교", "연구"]
        }
        
        intent_scores = {}
        for intent, indicators in intent_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            primary = max(intent_scores, key=intent_scores.get)
            return {
                "type": primary,
                "confidence": intent_scores[primary] / 10,
                "all_scores": intent_scores
            }
        
        return {"type": "general", "confidence": 0.1, "all_scores": {}}
    
    def _analyze_communication_style(self, text: str) -> Dict[str, Any]:
        """커뮤니케이션 스타일 분석"""
        style_indicators = {
            "formal": ["존경하는", "말씀드리겠습니다", "감사합니다", "정중히"],
            "casual": ["여러분", "우리", "같이", "함께", "그죠"],
            "technical": ["데이터", "분석", "결과", "연구", "방법론"],
            "storytelling": ["이야기", "경험", "사례", "예를 들어", "한번은"],
            "interactive": ["질문", "의견", "생각해보세요", "어떻게 생각하세요"]
        }
        
        style_scores = {}
        for style, indicators in style_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text)
            if score > 0:
                style_scores[style] = score
        
        return {
            "dominant_style": max(style_scores, key=style_scores.get) if style_scores else "neutral",
            "style_mix": style_scores
        }
    
    def _assess_audience_engagement(self, text: str) -> Dict[str, Any]:
        """청중 참여도 평가"""
        engagement_indicators = {
            "high": ["여러분", "함께", "질문", "참여", "상호작용"],
            "medium": ["보시듯이", "알 수 있듯", "이해하시겠지만"],
            "low": ["설명드리겠습니다", "보여드리겠습니다"]
        }
        
        engagement_scores = {}
        for level, indicators in engagement_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text)
            engagement_scores[level] = score
        
        total_score = sum(engagement_scores.values())
        if total_score == 0:
            return {"level": "unknown", "score": 0}
        
        # 가중 평균 계산
        weighted_score = (engagement_scores["high"] * 3 + engagement_scores["medium"] * 2 + engagement_scores["low"] * 1) / total_score
        
        if weighted_score >= 2.5:
            level = "high"
        elif weighted_score >= 1.5:
            level = "medium"
        else:
            level = "low"
        
        return {"level": level, "score": weighted_score, "details": engagement_scores}
    
    def _analyze_content_structure(self, text: str) -> Dict[str, Any]:
        """내용 구조 분석"""
        structure_elements = {
            "has_introduction": any(pattern in text for pattern in self.presentation_patterns["opening_patterns"]),
            "has_main_points": any(pattern in text for pattern in self.presentation_patterns["key_point_indicators"]),
            "has_transitions": any(pattern in text for pattern in self.presentation_patterns["transition_phrases"]),
            "has_conclusion": any(pattern in text for pattern in self.presentation_patterns["conclusion_patterns"]),
            "has_emphasis": any(pattern in text for pattern in self.presentation_patterns["emphasis_markers"])
        }
        
        structure_score = sum(structure_elements.values()) / len(structure_elements)
        
        return {
            "structure_completeness": structure_score,
            "elements_present": structure_elements,
            "organization_level": "well_structured" if structure_score >= 0.6 else "moderately_structured" if structure_score >= 0.3 else "poorly_structured"
        }
    
    def _analyze_emotional_tone(self, text: str) -> Dict[str, Any]:
        """감정 톤 분석"""
        emotional_indicators = {
            "enthusiastic": ["정말", "굉장히", "매우", "놀라운", "환상적"],
            "confident": ["확신", "분명", "당연히", "확실히", "명확히"],
            "cautious": ["아마도", "가능성", "고려해야", "주의해야", "신중히"],
            "urgent": ["긴급", "빨리", "즉시", "반드시", "중요"],
            "neutral": ["입니다", "있습니다", "됩니다", "것입니다"]
        }
        
        tone_scores = {}
        for tone, indicators in emotional_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text)
            if score > 0:
                tone_scores[tone] = score
        
        dominant_tone = max(tone_scores, key=tone_scores.get) if tone_scores else "neutral"
        
        return {
            "dominant_tone": dominant_tone,
            "tone_distribution": tone_scores,
            "emotional_intensity": sum(tone_scores.values()) / len(text.split()) * 100 if text else 0
        }
    
    def _classify_message_type(self, message: str) -> str:
        """메시지 타입 분류"""
        message_lower = message.lower()
        
        if any(pattern in message_lower for pattern in self.presentation_patterns["conclusion_patterns"]):
            return "conclusion"
        elif any(pattern in message_lower for pattern in self.presentation_patterns["key_point_indicators"]):
            return "key_point"
        elif any(pattern in message_lower for pattern in self.presentation_patterns["opening_patterns"]):
            return "introduction"
        elif "해야" in message_lower or "필요" in message_lower or "권장" in message_lower:
            return "action"
        elif "?" in message:
            return "question"
        else:
            return "general"
    
    def _calculate_message_importance(self, message: Dict, speaker_intent: Dict) -> float:
        """메시지 중요도 계산"""
        base_score = message.get("raw_score", 0)
        
        # 메시지 타입별 보너스
        type_bonus = {
            "key_point": 1.5,
            "conclusion": 1.3,
            "action": 1.2,
            "introduction": 1.1,
            "question": 1.0,
            "general": 0.8
        }
        
        msg_type = message.get("message_type", "general")
        score = base_score * type_bonus.get(msg_type, 1.0)
        
        # 소스 타입별 보너스
        source_bonus = {
            "multimodal": 1.3,  # 음성+시각 동시
            "audio": 1.0,
            "visual": 1.1
        }
        
        source_type = message.get("source_type", "audio")
        score *= source_bonus.get(source_type, 1.0)
        
        # 스피커 의도와의 일치도 보너스
        intent_type = speaker_intent.get("primary_intent", {}).get("type", "general")
        if intent_type == "inform" and msg_type == "key_point":
            score *= 1.2
        elif intent_type == "persuade" and msg_type == "action":
            score *= 1.3
        
        return score
    
    def _derive_actionable_insights(self, message_hierarchy: Dict, context: Dict = None) -> List[Dict[str, Any]]:
        """실행 가능한 인사이트 도출"""
        insights = []
        
        # 주요 메시지 기반 인사이트
        if message_hierarchy.get("main_theme"):
            main_theme = message_hierarchy["main_theme"]["content"]
            insights.append({
                "type": "main_message",
                "insight": f"강연자의 핵심 메시지: {main_theme[:100]}...",
                "action": "이 메시지를 중심으로 후속 논의나 실행 계획 수립",
                "priority": "high"
            })
        
        # 실행 항목 인사이트
        if message_hierarchy.get("call_to_actions"):
            insights.append({
                "type": "action_items",
                "insight": f"{len(message_hierarchy['call_to_actions'])}개의 구체적인 실행 항목 제시됨",
                "action": "제시된 실행 항목들을 체크리스트로 정리하여 단계별 실행",
                "priority": "high"
            })
        
        # 지식 공유 인사이트
        if len(message_hierarchy.get("supporting_details", [])) > 3:
            insights.append({
                "type": "knowledge_sharing",
                "insight": "풍부한 세부 정보와 배경 지식 제공됨",
                "action": "상세 내용을 정리하여 팀 내 지식 공유 자료로 활용",
                "priority": "medium"
            })
        
        return insights
    
    def _generate_insights_from_intent(self, speaker_intent: Dict) -> List[str]:
        """의도 기반 인사이트 생성"""
        insights = []
        
        intent_type = speaker_intent.get("primary_intent", {}).get("type", "general")
        
        if intent_type == "inform":
            insights.append("💡 정보 전달 중심의 발표 - 핵심 정보 습득에 집중")
        elif intent_type == "persuade":
            insights.append("🎯 설득을 위한 발표 - 제안사항에 대한 의사결정 필요")
        elif intent_type == "educate":
            insights.append("📚 교육 목적의 발표 - 학습한 내용을 실무에 적용 검토")
        elif intent_type == "inspire":
            insights.append("🔥 동기부여 중심의 발표 - 개인/팀 목표 재정비 기회")
        
        # 커뮤니케이션 스타일 인사이트
        comm_style = speaker_intent.get("communication_style", {}).get("dominant_style", "neutral")
        if comm_style == "interactive":
            insights.append("🤝 상호작용적 발표 스타일 - 추가 질의응답 시간 확보 권장")
        elif comm_style == "technical":
            insights.append("🔬 기술적 발표 내용 - 전문 용어 및 세부사항 재검토 필요")
        
        return insights
    
    def _calculate_overall_confidence(self, message_hierarchy: Dict) -> float:
        """전체 신뢰도 계산"""
        factors = [
            message_hierarchy.get("main_theme") is not None,
            len(message_hierarchy.get("key_points", [])) >= 2,
            len(message_hierarchy.get("supporting_details", [])) >= 1,
            len(message_hierarchy.get("conclusions", [])) >= 1
        ]
        
        confidence = sum(factors) / len(factors)
        return confidence
    
    def _calculate_clova_compatibility(self, message_hierarchy: Dict, speaker_intent: Dict) -> float:
        """클로바 노트 호환성 점수"""
        compatibility_factors = [
            message_hierarchy.get("main_theme") is not None,  # 명확한 주제
            len(message_hierarchy.get("key_points", [])) >= 1,  # 핵심 포인트
            speaker_intent.get("content_structure", {}).get("structure_completeness", 0) > 0.5,  # 구조화
            speaker_intent.get("primary_intent", {}).get("confidence", 0) > 0.3  # 의도 명확성
        ]
        
        score = sum(compatibility_factors) / len(compatibility_factors)
        return score
    
    # Context application methods (helper methods)
    def _apply_speaker_context(self, content: Dict, context: Dict) -> Dict:
        """발표자 컨텍스트 적용"""
        # 간단한 구현 - 실제로는 더 복잡한 로직 필요
        return content
    
    def _apply_event_context(self, content: Dict, context: Dict) -> Dict:
        """이벤트 컨텍스트 적용"""
        return content
    
    def _apply_topic_enhancement(self, content: Dict, context: Dict) -> Dict:
        """주제 키워드 강화"""
        return content
    
    def _apply_objective_filtering(self, content: Dict, context: Dict) -> Dict:
        """목적 기반 필터링"""
        return content
    
    def _assess_content_richness(self, content: Dict) -> float:
        """콘텐츠 풍부도 평가"""
        richness = 0
        if content.get("audio_content"):
            richness += 0.5
        if content.get("visual_content"):
            richness += 0.3
        if len(content.get("temporal_sync", [])) > 0:
            richness += 0.2
        return min(richness, 1.0)

# 전역 메시지 추출 엔진
global_message_extractor = ComprehensiveMessageExtractor()

def extract_speaker_message(analysis_results: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """강연자 메시지 추출 통합 함수"""
    return global_message_extractor.extract_comprehensive_message(analysis_results, context)

if __name__ == "__main__":
    # 테스트 실행
    print("🎯 종합 메시지 추출 엔진 테스트")
    
    # 테스트 데이터
    test_data = {
        "audio_analysis": {
            "status": "success",
            "enhanced_text": "오늘 말씀드릴 주제는 디지털 전환입니다. 가장 중요한 것은 고객 중심의 사고입니다. 결론적으로 우리는 즉시 행동해야 합니다.",
            "segments": [
                {"start": 0, "end": 10, "text": "오늘 말씀드릴 주제는 디지털 전환입니다"},
                {"start": 10, "end": 20, "text": "가장 중요한 것은 고객 중심의 사고입니다"}
            ]
        },
        "visual_analysis": {
            "status": "success",
            "combined_visual_text": "디지털 전환 전략 슬라이드 고객 만족도 95% 증가"
        }
    }
    
    test_context = {
        "event_context": "기업 전략 발표회",
        "speakers": "CEO 김철수",
        "objective": "디지털 전환 계획 공유"
    }
    
    extractor = ComprehensiveMessageExtractor()
    result = extractor.extract_comprehensive_message(test_data, test_context)
    
    print(f"메인 메시지: {result['comprehensive_analysis']['clova_style_summary']['executive_summary']}")
    print(f"핵심 포인트: {len(result['comprehensive_analysis']['key_messages'])}개")
    print(f"클로바 호환성: {result['comprehensive_analysis']['clova_style_summary']['clova_compatibility_score']:.2f}")