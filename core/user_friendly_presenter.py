#!/usr/bin/env python3
"""
사용자 친화적 결과 표시 엔진
"아, 이 사람들이 이런 얘기를 했구나"라고 바로 이해할 수 있는 수준의 결과 표시
"""

import streamlit as st
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import re

class UserFriendlyPresenter:
    """사용자가 쉽게 이해할 수 있는 분석 결과 표시"""
    
    def __init__(self):
        self.conversation_icons = {
            "inquiry": "🤔",
            "purchase_intent": "🛒", 
            "comparison": "⚖️",
            "complaint": "😤",
            "satisfaction": "😊",
            "technical": "🔧",
            "negotiation": "💰",
            "relationship": "💝",
            "general": "💬"
        }
        
        self.insight_icons = {
            "high_interest": "🔥",
            "decision_ready": "✅",
            "needs_info": "❓",
            "price_sensitive": "💰",
            "quality_focused": "💎",
            "service_issue": "⚠️",
            "satisfied": "😊"
        }
    
    def present_enhanced_result(self, analysis_result: Dict[str, Any], context: Dict[str, Any] = None):
        """향상된 분석 결과를 사용자 친화적으로 표시"""
        
        if analysis_result.get('status') != 'success':
            self._show_error_result(analysis_result)
            return
        
        # 메인 헤더
        st.markdown("## 📊 분석 결과 요약")
        
        # 품질 향상 적용 여부 표시
        if analysis_result.get('quality_enhancement_applied'):
            st.success("🚀 **AI 품질 향상 기술 적용됨** - 더 정확하고 의미있는 분석 결과")
        
        # 1. 핵심 요약 (가장 중요)
        self._show_executive_summary(analysis_result, context)
        
        # 2. 대화 패턴 및 인사이트
        self._show_conversation_insights(analysis_result)
        
        # 3. 주요 내용 (원문 + 개선된 버전)
        self._show_content_comparison(analysis_result)
        
        # 4. 실행 가능한 액션 아이템
        self._show_action_items(analysis_result)
        
        # 5. 기술적 세부사항 (접을 수 있는 형태)
        self._show_technical_details(analysis_result)
    
    def _show_executive_summary(self, result: Dict[str, Any], context: Dict[str, Any] = None):
        """경영진/실무진이 이해할 수 있는 핵심 요약"""
        st.markdown("### 🎯 **핵심 요약** - 이 사람들이 한 얘기")
        
        # 의미있는 요약이 있는 경우
        if 'meaningful_summary' in result:
            summary_data = result['meaningful_summary']
            
            # 경영진 요약
            if summary_data.get('executive_summary'):
                st.markdown(f"**📋 상황 요약:**")
                st.info(summary_data['executive_summary'])
            
            # 핵심 인사이트
            if summary_data.get('key_insights'):
                st.markdown("**🔑 핵심 인사이트:**")
                for insight in summary_data['key_insights'][:3]:  # 상위 3개만
                    st.markdown(f"• {insight}")
            
            # 대화 유형 표시
            if summary_data.get('conversation_patterns'):
                conv_type = summary_data['conversation_patterns'].get('conversation_type', 'general')
                icon = self.conversation_icons.get(conv_type, '💬')
                st.markdown(f"**{icon} 대화 유형:** {self._get_conversation_type_description(conv_type)}")
        
        # 기본 요약 (fallback)
        else:
            if result.get('summary'):
                st.info(f"📝 {result['summary']}")
            elif result.get('enhanced_text'):
                # 긴 텍스트는 요약해서 표시
                text = result['enhanced_text']
                if len(text) > 200:
                    summary = text[:200] + "..."
                else:
                    summary = text
                st.info(f"📝 {summary}")
    
    def _show_conversation_insights(self, result: Dict[str, Any]):
        """대화 패턴 및 인사이트 표시"""
        if 'meaningful_summary' not in result:
            return
        
        summary_data = result['meaningful_summary']
        
        # 대화 패턴 분석
        if summary_data.get('conversation_patterns'):
            patterns = summary_data['conversation_patterns']
            
            with st.expander("🧠 **대화 패턴 분석** - 고객이 원하는 것"):
                
                # 주요 의도 점수
                if patterns.get('intent_scores'):
                    st.markdown("**🎯 고객 의도 분석:**")
                    for intent, score in sorted(patterns['intent_scores'].items(), key=lambda x: x[1], reverse=True)[:3]:
                        intent_desc = self._get_intent_description(intent)
                        progress_value = min(score * 20, 100)  # 점수를 0-100으로 변환
                        st.progress(progress_value / 100, text=f"{intent_desc}: {score}점")
                
                # 감정 톤
                if patterns.get('emotional_tone'):
                    tone = patterns['emotional_tone']
                    tone_icon = "😊" if tone == "positive" else "😔" if tone == "negative" else "😐"
                    st.markdown(f"**{tone_icon} 감정 톤:** {self._get_emotional_tone_description(tone)}")
        
        # 주요 주제들
        if summary_data.get('main_topics'):
            st.markdown("### 🏷️ **주요 관심 주제**")
            
            cols = st.columns(min(3, len(summary_data['main_topics'])))
            for i, topic in enumerate(summary_data['main_topics'][:3]):
                with cols[i]:
                    category_icon = self._get_topic_icon(topic['category'])
                    st.metric(
                        label=f"{category_icon} {self._get_category_description(topic['category'])}",
                        value=f"{topic['relevance_score']}개 언급",
                        delta=f"{', '.join(topic['mentioned_terms'][:2])}"
                    )
    
    def _show_content_comparison(self, result: Dict[str, Any]):
        """원본 vs 개선된 내용 비교"""
        with st.expander("📝 **분석 내용 상세** - 실제 대화 내용"):
            
            # 탭으로 구분
            if result.get('enhanced_text') and result.get('enhanced_text') != result.get('full_text', ''):
                tab1, tab2 = st.tabs(["🚀 개선된 내용", "📋 원본 내용"])
                
                with tab1:
                    st.markdown("**AI가 정제하고 보정한 내용:**")
                    self._display_formatted_text(result.get('enhanced_text', ''))
                    
                    if result.get('text_improvements'):
                        st.markdown("**🔧 적용된 개선사항:**")
                        for improvement in result['text_improvements'][:5]:
                            st.caption(f"• {improvement}")
                
                with tab2:
                    st.markdown("**원본 추출 내용:**")
                    self._display_formatted_text(result.get('full_text', result.get('original_text', '')))
            
            else:
                # 개선이 없는 경우 원본만 표시
                text_to_show = result.get('enhanced_text') or result.get('full_text') or result.get('original_text', '')
                self._display_formatted_text(text_to_show)
    
    def _show_action_items(self, result: Dict[str, Any]):
        """실행 가능한 액션 아이템"""
        if 'meaningful_summary' in result and result['meaningful_summary'].get('action_items'):
            st.markdown("### ✅ **추천 액션 아이템**")
            
            action_items = result['meaningful_summary']['action_items']
            
            for i, action in enumerate(action_items[:5]):  # 최대 5개
                col1, col2 = st.columns([0.1, 0.9])
                with col1:
                    st.checkbox("", key=f"action_{i}")
                with col2:
                    st.markdown(action)
    
    def _show_technical_details(self, result: Dict[str, Any]):
        """기술적 세부사항 (접을 수 있는 형태)"""
        with st.expander("🔧 **기술적 세부사항** (개발자/관리자용)"):
            
            # 처리 성능
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("처리 시간", f"{result.get('processing_time', 0):.1f}초")
            with col2:
                st.metric("파일 크기", f"{result.get('file_size_mb', 0):.1f}MB")
            with col3:
                confidence = result.get('text_confidence', result.get('average_confidence', 0))
                st.metric("신뢰도", f"{confidence:.1%}" if confidence else "N/A")
            
            # 품질 향상 정보
            if result.get('quality_enhancement_applied'):
                st.success("✅ 품질 향상 엔진 적용됨")
                
                if result.get('text_improvements'):
                    st.markdown("**텍스트 개선사항:**")
                    for improvement in result.get('text_improvements', []):
                        st.caption(f"• {improvement}")
                
                if result.get('ocr_improvements'):
                    st.markdown("**OCR 개선사항:**")
                    for improvement in result.get('ocr_improvements', []):
                        st.caption(f"• {improvement}")
            
            # 주얼리 키워드
            if result.get('jewelry_keywords'):
                st.markdown("**감지된 주얼리 키워드:**")
                keywords_text = ", ".join(result['jewelry_keywords'][:10])
                st.caption(keywords_text)
            
            # 상세 기술 정보
            tech_info = {
                "분석 타입": result.get('analysis_type', 'unknown'),
                "파일명": result.get('file_name', 'unknown'),
                "타임스탬프": result.get('timestamp', 'unknown')
            }
            
            if result.get('detected_language'):
                tech_info["감지된 언어"] = result['detected_language']
            if result.get('blocks_detected'):
                tech_info["OCR 블록 수"] = result['blocks_detected']
            if result.get('segments_count'):
                tech_info["음성 세그먼트 수"] = result['segments_count']
            
            for key, value in tech_info.items():
                st.caption(f"**{key}:** {value}")
    
    def _show_error_result(self, result: Dict[str, Any]):
        """에러 결과 표시"""
        st.error("❌ **분석 실패**")
        
        error_msg = result.get('error', '알 수 없는 오류')
        st.error(f"오류 내용: {error_msg}")
        
        # 해결 방안 제시
        st.markdown("### 🔧 **해결 방안**")
        
        if "M4A" in error_msg or "m4a" in error_msg:
            st.info("""
            **M4A 파일 처리 문제:**
            1. 파일을 WAV 형식으로 변환해보세요
            2. 파일이 손상되지 않았는지 확인하세요
            3. 다른 음성 파일로 테스트해보세요
            """)
        elif "OCR" in error_msg or "이미지" in error_msg:
            st.info("""
            **이미지 분석 문제:**
            1. 이미지 해상도가 충분한지 확인하세요
            2. 텍스트가 선명하게 보이는지 확인하세요
            3. JPG, PNG 등 지원 형식인지 확인하세요
            """)
        else:
            st.info("""
            **일반적인 해결 방안:**
            1. 파일 형식이 지원되는지 확인하세요
            2. 파일이 손상되지 않았는지 확인하세요
            3. 다른 파일로 테스트해보세요
            """)
    
    def _display_formatted_text(self, text: str):
        """텍스트를 보기 좋게 포맷팅해서 표시"""
        if not text:
            st.caption("추출된 텍스트가 없습니다.")
            return
        
        # 긴 텍스트는 줄바꿈 처리
        formatted_text = text.replace('. ', '.\n\n').replace('? ', '?\n\n').replace('! ', '!\n\n')
        
        # 주얼리 관련 단어 강조
        jewelry_terms = ['다이아몬드', '금', '은', '백금', '반지', '목걸이', '귀걸이', '팔찌', '주얼리', 'diamond', 'gold', 'silver']
        for term in jewelry_terms:
            if term in formatted_text:
                formatted_text = formatted_text.replace(term, f"**{term}**")
        
        st.markdown(formatted_text)
    
    def _get_conversation_type_description(self, conv_type: str) -> str:
        """대화 유형 설명"""
        descriptions = {
            "inquiry": "정보 문의 - 제품에 대해 알고 싶어함",
            "purchase_intent": "구매 의도 - 구매를 고려 중",
            "comparison": "비교 검토 - 여러 옵션을 비교 중",
            "complaint": "불만 제기 - 문제점이나 불만 표출",
            "satisfaction": "만족 표현 - 제품/서비스에 만족",
            "technical": "기술 문의 - 사양이나 기술적 정보 요청",
            "negotiation": "가격 협상 - 할인이나 조건 협상",
            "relationship": "관계 관련 - 선물이나 기념품 목적",
            "general": "일반 대화 - 특별한 패턴 없음"
        }
        return descriptions.get(conv_type, conv_type)
    
    def _get_intent_description(self, intent: str) -> str:
        """의도 설명"""
        descriptions = {
            "inquiry": "🤔 궁금해함",
            "purchase_intent": "🛒 구매하고 싶어함",
            "comparison": "⚖️ 비교하고 있음",
            "complaint": "😤 불만이 있음",
            "satisfaction": "😊 만족해함",
            "technical": "🔧 기술정보 원함",
            "negotiation": "💰 가격 협상 중",
            "relationship": "💝 선물/기념품 목적"
        }
        return descriptions.get(intent, intent)
    
    def _get_emotional_tone_description(self, tone: str) -> str:
        """감정 톤 설명"""
        descriptions = {
            "positive": "긍정적 - 좋은 반응",
            "negative": "부정적 - 불만이나 문제 있음",
            "neutral": "중립적 - 평범한 대화"
        }
        return descriptions.get(tone, tone)
    
    def _get_topic_icon(self, category: str) -> str:
        """주제 카테고리 아이콘"""
        icons = {
            "materials": "💎",
            "products": "👑", 
            "quality_terms": "⭐",
            "business_terms": "💼"
        }
        return icons.get(category, "📋")
    
    def _get_category_description(self, category: str) -> str:
        """카테고리 설명"""
        descriptions = {
            "materials": "재질/소재",
            "products": "제품 종류",
            "quality_terms": "품질 기준", 
            "business_terms": "비즈니스"
        }
        return descriptions.get(category, category)

# 전역 표시 엔진 인스턴스
global_presenter = UserFriendlyPresenter()

def show_enhanced_analysis_result(analysis_result: Dict[str, Any], context: Dict[str, Any] = None):
    """향상된 분석 결과를 사용자 친화적으로 표시하는 통합 함수"""
    global_presenter.present_enhanced_result(analysis_result, context)

if __name__ == "__main__":
    # 테스트용 스트림릿 앱
    st.title("🧪 사용자 친화적 결과 표시 테스트")
    
    # 테스트 데이터
    test_result = {
        "status": "success",
        "file_name": "test_audio.m4a",
        "processing_time": 12.3,
        "file_size_mb": 2.5,
        "enhanced_text": "이 다이아몬드 반지 가격이 얼마인가요? 할인도 가능한지 궁금합니다.",
        "full_text": "이게 다이아 반지 가격이 얼마에요? 할인도 되나요?",
        "text_improvements": ["맞춤법 보정: '다이아' → '다이아몬드'", "문법 보정 적용"],
        "jewelry_keywords": ["다이아몬드", "반지", "가격", "할인"],
        "meaningful_summary": {
            "executive_summary": "💼 주요 논의: 다이아몬드 반지 가격 문의\n💡 핵심 포인트: 고객이 제품에 대한 구체적인 정보를 원하고 있습니다.",
            "key_insights": ["💡 고객이 제품에 대한 구체적인 정보를 원하고 있습니다.", "💰 가격이나 서비스 조건이 중요한 결정 요소입니다."],
            "conversation_patterns": {
                "conversation_type": "inquiry",
                "intent_scores": {"inquiry": 3, "negotiation": 1},
                "emotional_tone": "neutral"
            },
            "action_items": ["📞 고객 문의사항에 대한 상세 답변 제공", "💰 가격 정보 및 할인 혜택 안내"]
        },
        "quality_enhancement_applied": True
    }
    
    test_context = {
        "event_context": "주얼리 매장 상담",
        "objective": "반지 구매 상담"
    }
    
    show_enhanced_analysis_result(test_result, test_context)