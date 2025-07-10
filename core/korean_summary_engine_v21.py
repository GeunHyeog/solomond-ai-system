"""
🎯 Solomond AI v2.1 - 한국어 통합 분석 엔진
모든 다국어 입력을 한국어로 통합 분석, 주얼리 업계 특화 요약 및 인사이트 생성

Author: 전근혁 (Solomond)
Created: 2025.07.11
Version: 2.1.0
"""

import re
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
import numpy as np
from collections import Counter, defaultdict
import openai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

@dataclass
class KoreanAnalysisResult:
    """한국어 분석 결과"""
    final_korean_summary: str       # 최종 한국어 요약
    business_insights: List[str]    # 비즈니스 인사이트
    technical_insights: List[str]   # 기술적 인사이트
    market_insights: List[str]      # 시장 인사이트
    action_items: List[str]         # 액션 아이템
    key_decisions: List[str]        # 주요 결정사항
    follow_up_tasks: List[str]      # 후속 작업
    executive_summary: str          # 경영진 요약
    detailed_analysis: str          # 상세 분석
    jewelry_terminology: Dict[str, int]  # 주얼리 용어 빈도
    confidence_score: float         # 분석 신뢰도
    processing_details: Dict[str, Any]

@dataclass
class SummaryStyle:
    """요약 스타일 설정"""
    target_audience: str    # executive, technical, business, comprehensive
    length_preference: str  # brief, standard, detailed
    focus_areas: List[str]  # business, technical, market, trends
    language_tone: str      # formal, professional, casual
    include_examples: bool  # 예시 포함 여부

class JewelryKnowledgeBase:
    """주얼리 업계 지식 베이스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 주얼리 업계 특화 지식
        self.business_contexts = {
            '4C': {
                'description': '다이아몬드 품질 평가 기준',
                'components': ['캐럿(Carat)', '투명도(Clarity)', '컬러(Color)', '커팅(Cut)'],
                'importance': '다이아몬드 가치 결정의 핵심 요소'
            },
            'GIA': {
                'description': '국제보석학회 인증',
                'significance': '전 세계적으로 인정받는 보석 감정 기관',
                'impact': '보석의 신뢰성과 가치를 보장'
            },
            '시장_트렌드': {
                'areas': ['디자인 트렌드', '소비자 선호도', '기술 혁신', '지속가능성'],
                'impact_factors': ['경제 상황', '문화적 변화', '기술 발전', '환경 의식']
            },
            '제조_공정': {
                'stages': ['원석 선별', '커팅', '연마', '세팅', '완성'],
                'quality_factors': ['기술력', '장비', '경험', '품질 관리']
            }
        }
        
        # 비즈니스 키워드 매핑
        self.business_keywords = {
            '매출': ['revenue', 'sales', '수익', '매출액', '판매'],
            '마진': ['margin', 'profit', '이익', '수익률', '마진율'],
            '고객': ['customer', 'client', '고객', '소비자', '바이어'],
            '시장': ['market', '시장', '마켓', '시세', '수요'],
            '경쟁': ['competition', 'competitor', '경쟁', '경쟁사', '라이벌'],
            '브랜드': ['brand', 'branding', '브랜드', '브랜딩', '마케팅'],
            '품질': ['quality', '품질', '퀄리티', '등급', '기준'],
            '혁신': ['innovation', 'technology', '혁신', '기술', '신기술']
        }
        
    def get_context_for_term(self, term: str) -> Optional[Dict[str, Any]]:
        """용어에 대한 컨텍스트 정보 반환"""
        try:
            term_lower = term.lower()
            
            # 직접 매칭
            for key, context in self.business_contexts.items():
                if key.lower() in term_lower or any(comp.lower() in term_lower 
                                                   for comp in str(context).lower().split()):
                    return context
            
            # 키워드 매핑 검색
            for category, keywords in self.business_keywords.items():
                if any(keyword.lower() in term_lower for keyword in keywords):
                    return {
                        'category': category,
                        'related_terms': keywords,
                        'business_relevance': 'high'
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"컨텍스트 조회 실패: {e}")
            return None

class KoreanSummaryGenerator:
    """한국어 요약 생성기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.knowledge_base = JewelryKnowledgeBase()
        
        # 요약 템플릿
        self.summary_templates = {
            'executive': {
                'title': '경영진 요약',
                'structure': ['핵심 결과', '주요 결정사항', '비즈니스 영향', '다음 단계'],
                'tone': 'formal',
                'length': 'brief'
            },
            'technical': {
                'title': '기술적 분석',
                'structure': ['기술 현황', '품질 분석', '공정 개선', '기술 권장사항'],
                'tone': 'professional',
                'length': 'detailed'
            },
            'business': {
                'title': '비즈니스 분석',
                'structure': ['시장 현황', '경쟁 분석', '기회 요소', '위험 요소'],
                'tone': 'professional',
                'length': 'standard'
            },
            'comprehensive': {
                'title': '종합 분석',
                'structure': ['전체 개요', '세부 분석', '핵심 인사이트', '실행 계획'],
                'tone': 'professional',
                'length': 'detailed'
            }
        }
        
    def generate_korean_summary(self, content: str, style: SummaryStyle) -> str:
        """한국어 요약 생성"""
        try:
            template = self.summary_templates.get(style.target_audience, 
                                                 self.summary_templates['comprehensive'])
            
            # 요약 구조 생성
            summary_sections = []
            
            # 제목
            summary_sections.append(f"# {template['title']}")
            summary_sections.append("")
            
            # 날짜 및 기본 정보
            summary_sections.append(f"**분석 일시**: {datetime.now().strftime('%Y년 %m월 %d일')}")
            summary_sections.append("")
            
            # 구조별 내용 생성
            for section_title in template['structure']:
                section_content = self._generate_section_content(content, section_title, style)
                if section_content:
                    summary_sections.append(f"## {section_title}")
                    summary_sections.append(section_content)
                    summary_sections.append("")
            
            return '\n'.join(summary_sections)
            
        except Exception as e:
            self.logger.error(f"한국어 요약 생성 실패: {e}")
            return "요약 생성 중 오류가 발생했습니다."
    
    def _generate_section_content(self, content: str, section_title: str, style: SummaryStyle) -> str:
        """섹션별 내용 생성"""
        try:
            # 섹션별 키워드 추출
            section_keywords = self._get_section_keywords(section_title)
            
            # 관련 내용 추출
            relevant_sentences = self._extract_relevant_sentences(content, section_keywords)
            
            if not relevant_sentences:
                return "관련 내용이 식별되지 않았습니다."
            
            # 섹션별 특화 요약 생성
            if section_title == '핵심 결과':
                return self._summarize_key_results(relevant_sentences)
            elif section_title == '주요 결정사항':
                return self._summarize_decisions(relevant_sentences)
            elif section_title == '비즈니스 영향':
                return self._summarize_business_impact(relevant_sentences)
            elif section_title == '기술 현황':
                return self._summarize_technical_status(relevant_sentences)
            elif section_title == '시장 현황':
                return self._summarize_market_status(relevant_sentences)
            else:
                return self._general_summarize(relevant_sentences, section_title)
                
        except Exception as e:
            self.logger.error(f"섹션 내용 생성 실패 ({section_title}): {e}")
            return "내용 분석 중 오류가 발생했습니다."
    
    def _get_section_keywords(self, section_title: str) -> List[str]:
        """섹션별 키워드 반환"""
        keyword_map = {
            '핵심 결과': ['결과', '성과', '달성', '완료', '결론'],
            '주요 결정사항': ['결정', '결정사항', '합의', '승인', '채택'],
            '비즈니스 영향': ['영향', '효과', '변화', '개선', '증가', '감소'],
            '다음 단계': ['다음', '향후', '계획', '예정', '단계'],
            '기술 현황': ['기술', '품질', '공정', '개발', '혁신'],
            '시장 현황': ['시장', '트렌드', '수요', '공급', '경쟁'],
            '품질 분석': ['품질', '등급', '기준', '평가', '검사'],
            '경쟁 분석': ['경쟁', '경쟁사', '비교', '우위', '차별화']
        }
        
        return keyword_map.get(section_title, [section_title])
    
    def _extract_relevant_sentences(self, content: str, keywords: List[str]) -> List[str]:
        """키워드 관련 문장 추출"""
        try:
            sentences = re.split(r'[.!?]\s+', content)
            relevant_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:  # 너무 짧은 문장 제외
                    continue
                
                # 키워드 포함 확인
                for keyword in keywords:
                    if keyword in sentence:
                        relevant_sentences.append(sentence)
                        break
            
            return relevant_sentences[:10]  # 상위 10개 문장만
            
        except Exception as e:
            self.logger.error(f"관련 문장 추출 실패: {e}")
            return []
    
    def _summarize_key_results(self, sentences: List[str]) -> str:
        """핵심 결과 요약"""
        try:
            if not sentences:
                return "핵심 결과를 확인할 수 없습니다."
            
            # 결과 관련 문장 중요도 분석
            result_indicators = ['달성', '완료', '성공', '개선', '증가', '향상']
            important_results = []
            
            for sentence in sentences:
                for indicator in result_indicators:
                    if indicator in sentence:
                        important_results.append(f"• {sentence}")
                        break
            
            if important_results:
                return '\n'.join(important_results[:5])
            else:
                return f"• {sentences[0]}" if sentences else "분석 가능한 결과가 없습니다."
                
        except Exception as e:
            self.logger.error(f"핵심 결과 요약 실패: {e}")
            return "핵심 결과 요약 중 오류가 발생했습니다."
    
    def _summarize_decisions(self, sentences: List[str]) -> str:
        """결정사항 요약"""
        try:
            decision_patterns = [
                r'결정.*?했다',
                r'합의.*?했다', 
                r'승인.*?했다',
                r'채택.*?했다',
                r'정하기로.*?했다'
            ]
            
            decisions = []
            for sentence in sentences:
                for pattern in decision_patterns:
                    if re.search(pattern, sentence):
                        decisions.append(f"• {sentence}")
                        break
            
            return '\n'.join(decisions[:5]) if decisions else "명확한 결정사항이 확인되지 않았습니다."
            
        except Exception as e:
            self.logger.error(f"결정사항 요약 실패: {e}")
            return "결정사항 분석 중 오류가 발생했습니다."
    
    def _summarize_business_impact(self, sentences: List[str]) -> str:
        """비즈니스 영향 요약"""
        try:
            impact_keywords = ['매출', '수익', '비용', '효율', '고객', '시장', '경쟁력']
            impact_sentences = []
            
            for sentence in sentences:
                for keyword in impact_keywords:
                    if keyword in sentence:
                        impact_sentences.append(f"• {sentence}")
                        break
            
            return '\n'.join(impact_sentences[:5]) if impact_sentences else "비즈니스 영향을 분석할 수 있는 내용이 부족합니다."
            
        except Exception as e:
            self.logger.error(f"비즈니스 영향 요약 실패: {e}")
            return "비즈니스 영향 분석 중 오류가 발생했습니다."
    
    def _summarize_technical_status(self, sentences: List[str]) -> str:
        """기술 현황 요약"""
        try:
            tech_keywords = ['기술', '품질', '공정', '제작', '가공', '혁신', '개발']
            tech_sentences = []
            
            for sentence in sentences:
                for keyword in tech_keywords:
                    if keyword in sentence:
                        tech_sentences.append(f"• {sentence}")
                        break
            
            return '\n'.join(tech_sentences[:5]) if tech_sentences else "기술 관련 내용이 충분하지 않습니다."
            
        except Exception as e:
            self.logger.error(f"기술 현황 요약 실패: {e}")
            return "기술 현황 분석 중 오류가 발생했습니다."
    
    def _summarize_market_status(self, sentences: List[str]) -> str:
        """시장 현황 요약"""
        try:
            market_keywords = ['시장', '트렌드', '수요', '공급', '가격', '고객', '소비자']
            market_sentences = []
            
            for sentence in sentences:
                for keyword in market_keywords:
                    if keyword in sentence:
                        market_sentences.append(f"• {sentence}")
                        break
            
            return '\n'.join(market_sentences[:5]) if market_sentences else "시장 관련 정보가 제한적입니다."
            
        except Exception as e:
            self.logger.error(f"시장 현황 요약 실패: {e}")
            return "시장 현황 분석 중 오류가 발생했습니다."
    
    def _general_summarize(self, sentences: List[str], section_title: str) -> str:
        """일반 요약"""
        try:
            if not sentences:
                return f"{section_title} 관련 내용이 부족합니다."
            
            # 상위 3-5개 문장을 요약으로 제공
            summary_sentences = sentences[:5]
            formatted_summary = '\n'.join([f"• {sentence}" for sentence in summary_sentences])
            
            return formatted_summary
            
        except Exception as e:
            self.logger.error(f"일반 요약 실패: {e}")
            return f"{section_title} 요약 중 오류가 발생했습니다."

class InsightExtractor:
    """인사이트 추출기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.knowledge_base = JewelryKnowledgeBase()
        
    def extract_business_insights(self, content: str) -> List[str]:
        """비즈니스 인사이트 추출"""
        try:
            insights = []
            
            # 비즈니스 패턴 분석
            business_patterns = [
                (r'매출.*?증가.*?(\d+%)', '매출 성장률'),
                (r'시장.*?확대.*?', '시장 확장 기회'),
                (r'고객.*?만족.*?', '고객 만족도 개선'),
                (r'경쟁.*?우위.*?', '경쟁 우위 확보'),
                (r'브랜드.*?가치.*?', '브랜드 가치 향상'),
                (r'수익.*?개선.*?', '수익성 개선')
            ]
            
            for pattern, insight_type in business_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    insights.append(f"💼 {insight_type}: {matches[0] if isinstance(matches[0], str) else '관련 내용 확인됨'}")
            
            # 키워드 기반 비즈니스 인사이트
            business_keywords = ['매출', '수익', '시장', '고객', '브랜드', '경쟁']
            for keyword in business_keywords:
                sentences_with_keyword = [s for s in content.split('.') if keyword in s]
                if sentences_with_keyword:
                    insight = f"📊 {keyword} 관련: {sentences_with_keyword[0].strip()[:100]}..."
                    insights.append(insight)
            
            return list(set(insights))[:5]  # 중복 제거 후 상위 5개
            
        except Exception as e:
            self.logger.error(f"비즈니스 인사이트 추출 실패: {e}")
            return ["비즈니스 인사이트 분석 중 오류가 발생했습니다."]
    
    def extract_technical_insights(self, content: str) -> List[str]:
        """기술적 인사이트 추출"""
        try:
            insights = []
            
            # 기술 관련 패턴
            tech_patterns = [
                (r'품질.*?향상.*?', '품질 개선'),
                (r'공정.*?개선.*?', '제조 공정 최적화'),
                (r'기술.*?혁신.*?', '기술 혁신'),
                (r'자동화.*?', '자동화 도입'),
                (r'디지털.*?', '디지털 전환'),
                (r'AI.*?머신러닝.*?', 'AI/ML 활용')
            ]
            
            for pattern, insight_type in tech_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    insights.append(f"🔧 {insight_type} 기회 확인")
            
            # 주얼리 특화 기술 키워드
            jewelry_tech_keywords = ['커팅', '연마', '세팅', '가공', 'CAD', '3D프린팅']
            for keyword in jewelry_tech_keywords:
                if keyword in content:
                    insights.append(f"💎 {keyword} 기술 관련 내용 포함")
            
            return list(set(insights))[:5]
            
        except Exception as e:
            self.logger.error(f"기술적 인사이트 추출 실패: {e}")
            return ["기술적 인사이트 분석 중 오류가 발생했습니다."]
    
    def extract_market_insights(self, content: str) -> List[str]:
        """시장 인사이트 추출"""
        try:
            insights = []
            
            # 시장 트렌드 패턴
            market_patterns = [
                (r'트렌드.*?변화.*?', '시장 트렌드 변화'),
                (r'수요.*?증가.*?', '수요 증가 추세'),
                (r'가격.*?상승.*?', '가격 상승 압력'),
                (r'경쟁.*?심화.*?', '경쟁 환경 변화'),
                (r'신제품.*?출시.*?', '신제품 출시 동향'),
                (r'소비자.*?선호.*?', '소비자 선호도 변화')
            ]
            
            for pattern, insight_type in market_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    insights.append(f"📈 {insight_type} 감지")
            
            # 지역별 시장 키워드
            regional_keywords = ['아시아', '중국', '홍콩', '일본', '유럽', '미국']
            for region in regional_keywords:
                if region in content:
                    insights.append(f"🌏 {region} 시장 관련 정보 포함")
            
            return list(set(insights))[:5]
            
        except Exception as e:
            self.logger.error(f"시장 인사이트 추출 실패: {e}")
            return ["시장 인사이트 분석 중 오류가 발생했습니다."]

class ActionItemExtractor:
    """액션 아이템 추출기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_action_items(self, content: str) -> List[str]:
        """액션 아이템 추출"""
        try:
            action_patterns = [
                r'해야\s*할\s*일.*?[\.!?]',
                r'준비.*?해야.*?[\.!?]',
                r'검토.*?필요.*?[\.!?]',
                r'확인.*?요청.*?[\.!?]',
                r'개선.*?계획.*?[\.!?]',
                r'다음.*?단계.*?[\.!?]',
                r'follow.*?up.*?[\.!?]',
                r'action.*?item.*?[\.!?]',
                r'to.*?do.*?[\.!?]'
            ]
            
            action_items = []
            
            for pattern in action_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                action_items.extend(matches[:2])  # 각 패턴에서 최대 2개
            
            # 정리 및 포맷팅
            formatted_actions = []
            for action in action_items:
                cleaned_action = action.strip()
                if len(cleaned_action) > 10:
                    formatted_actions.append(f"📋 {cleaned_action}")
            
            return list(set(formatted_actions))[:5]  # 중복 제거 후 상위 5개
            
        except Exception as e:
            self.logger.error(f"액션 아이템 추출 실패: {e}")
            return []
    
    def extract_key_decisions(self, content: str) -> List[str]:
        """주요 결정사항 추출"""
        try:
            decision_patterns = [
                r'결정.*?했다.*?[\.!?]',
                r'합의.*?했다.*?[\.!?]',
                r'승인.*?받았다.*?[\.!?]',
                r'채택.*?했다.*?[\.!?]',
                r'선택.*?했다.*?[\.!?]',
                r'정하기로.*?했다.*?[\.!?]'
            ]
            
            decisions = []
            
            for pattern in decision_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                decisions.extend(matches[:2])
            
            # 포맷팅
            formatted_decisions = []
            for decision in decisions:
                cleaned_decision = decision.strip()
                if len(cleaned_decision) > 10:
                    formatted_decisions.append(f"✅ {cleaned_decision}")
            
            return list(set(formatted_decisions))[:5]
            
        except Exception as e:
            self.logger.error(f"주요 결정사항 추출 실패: {e}")
            return []
    
    def extract_follow_up_tasks(self, content: str) -> List[str]:
        """후속 작업 추출"""
        try:
            followup_patterns = [
                r'향후.*?계획.*?[\.!?]',
                r'다음.*?회의.*?[\.!?]',
                r'후속.*?작업.*?[\.!?]',
                r'예정.*?사항.*?[\.!?]',
                r'스케줄.*?예약.*?[\.!?]'
            ]
            
            followups = []
            
            for pattern in followup_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                followups.extend(matches[:2])
            
            # 포맷팅
            formatted_followups = []
            for followup in followups:
                cleaned_followup = followup.strip()
                if len(cleaned_followup) > 10:
                    formatted_followups.append(f"⏭️ {cleaned_followup}")
            
            return list(set(formatted_followups))[:5]
            
        except Exception as e:
            self.logger.error(f"후속 작업 추출 실패: {e}")
            return []

class KoreanSummaryEngineV21:
    """v2.1 한국어 통합 분석 엔진"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.summary_generator = KoreanSummaryGenerator()
        self.insight_extractor = InsightExtractor()
        self.action_extractor = ActionItemExtractor()
        self.knowledge_base = JewelryKnowledgeBase()
        
    def analyze_korean_content(self, integrated_content: str, analysis_style: str = "comprehensive") -> KoreanAnalysisResult:
        """한국어 통합 분석 수행"""
        try:
            start_time = time.time()
            
            # 1. 분석 스타일 설정
            style = SummaryStyle(
                target_audience=analysis_style,
                length_preference="standard",
                focus_areas=["business", "technical", "market"],
                language_tone="professional",
                include_examples=True
            )
            
            # 2. 최종 한국어 요약 생성
            final_summary = self.summary_generator.generate_korean_summary(integrated_content, style)
            
            # 3. 각종 인사이트 추출
            business_insights = self.insight_extractor.extract_business_insights(integrated_content)
            technical_insights = self.insight_extractor.extract_technical_insights(integrated_content)
            market_insights = self.insight_extractor.extract_market_insights(integrated_content)
            
            # 4. 액션 아이템 및 결정사항 추출
            action_items = self.action_extractor.extract_action_items(integrated_content)
            key_decisions = self.action_extractor.extract_key_decisions(integrated_content)
            follow_up_tasks = self.action_extractor.extract_follow_up_tasks(integrated_content)
            
            # 5. 경영진 요약 생성
            executive_style = SummaryStyle(
                target_audience="executive",
                length_preference="brief",
                focus_areas=["business"],
                language_tone="formal",
                include_examples=False
            )
            executive_summary = self.summary_generator.generate_korean_summary(integrated_content, executive_style)
            
            # 6. 상세 분석 생성
            detailed_style = SummaryStyle(
                target_audience="technical",
                length_preference="detailed",
                focus_areas=["technical", "business", "market"],
                language_tone="professional",
                include_examples=True
            )
            detailed_analysis = self.summary_generator.generate_korean_summary(integrated_content, detailed_style)
            
            # 7. 주얼리 용어 분석
            jewelry_terminology = self._analyze_jewelry_terminology(integrated_content)
            
            # 8. 신뢰도 점수 계산
            confidence_score = self._calculate_analysis_confidence(
                integrated_content, business_insights, technical_insights, market_insights
            )
            
            processing_time = time.time() - start_time
            
            return KoreanAnalysisResult(
                final_korean_summary=final_summary,
                business_insights=business_insights,
                technical_insights=technical_insights,
                market_insights=market_insights,
                action_items=action_items,
                key_decisions=key_decisions,
                follow_up_tasks=follow_up_tasks,
                executive_summary=executive_summary,
                detailed_analysis=detailed_analysis,
                jewelry_terminology=jewelry_terminology,
                confidence_score=confidence_score,
                processing_details={
                    'processing_time': processing_time,
                    'content_length': len(integrated_content),
                    'analysis_style': analysis_style,
                    'insights_count': len(business_insights) + len(technical_insights) + len(market_insights),
                    'action_items_count': len(action_items),
                    'decisions_count': len(key_decisions)
                }
            )
            
        except Exception as e:
            self.logger.error(f"한국어 통합 분석 실패: {e}")
            return KoreanAnalysisResult(
                final_korean_summary="분석 중 오류가 발생했습니다.",
                business_insights=[],
                technical_insights=[],
                market_insights=[],
                action_items=[],
                key_decisions=[],
                follow_up_tasks=[],
                executive_summary="",
                detailed_analysis="",
                jewelry_terminology={},
                confidence_score=0.0,
                processing_details={'error': str(e)}
            )
    
    def _analyze_jewelry_terminology(self, content: str) -> Dict[str, int]:
        """주얼리 전문용어 분석"""
        try:
            # 주얼리 주요 용어 목록
            jewelry_terms = [
                # 보석 종류
                '다이아몬드', '루비', '사파이어', '에메랄드', '진주', '오팔', '터키석',
                # 금속
                '금', '은', '플래티넘', '화이트골드', '로즈골드', '옐로우골드',
                # 4C 관련
                '캐럿', '투명도', '컬러', '커팅', '클래리티',
                # 세팅 및 기술
                '프롱', '베젤', '세팅', '마운팅', '연마', '가공',
                # 인증 및 품질
                'GIA', '감정서', '인증서', '등급', '품질',
                # 비즈니스
                '도매', '소매', '수입', '수출', '시세', '마진'
            ]
            
            # 용어 빈도 계산
            terminology_count = {}
            content_lower = content.lower()
            
            for term in jewelry_terms:
                count = content_lower.count(term.lower())
                if count > 0:
                    terminology_count[term] = count
            
            # 빈도순 정렬
            sorted_terms = dict(sorted(terminology_count.items(), key=lambda x: x[1], reverse=True))
            
            return sorted_terms
            
        except Exception as e:
            self.logger.error(f"주얼리 용어 분석 실패: {e}")
            return {}
    
    def _calculate_analysis_confidence(self, content: str, business_insights: List[str], 
                                     technical_insights: List[str], market_insights: List[str]) -> float:
        """분석 신뢰도 계산"""
        try:
            # 기본 신뢰도 (내용 길이 기반)
            content_length = len(content)
            length_score = min(1.0, content_length / 2000)  # 2000자 기준
            
            # 인사이트 개수 기반 보정
            total_insights = len(business_insights) + len(technical_insights) + len(market_insights)
            insight_score = min(1.0, total_insights / 10)  # 10개 인사이트 기준
            
            # 주얼리 용어 포함 보너스
            jewelry_terms = self._analyze_jewelry_terminology(content)
            jewelry_bonus = min(0.2, len(jewelry_terms) * 0.05)
            
            # 구조화된 내용 여부 (섹션, 번호 등)
            structure_indicators = ['1.', '2.', '•', '-', '=', '#']
            structure_score = sum(1 for indicator in structure_indicators if indicator in content) / 10
            structure_score = min(0.2, structure_score)
            
            # 종합 신뢰도 계산
            final_confidence = (
                length_score * 0.3 +
                insight_score * 0.3 +
                structure_score * 0.2 +
                0.2  # 기본 점수
            ) + jewelry_bonus
            
            return max(0.0, min(1.0, final_confidence))
            
        except Exception as e:
            self.logger.error(f"신뢰도 계산 실패: {e}")
            return 0.5
    
    def generate_comprehensive_report(self, analysis_result: KoreanAnalysisResult) -> str:
        """종합 리포트 생성"""
        try:
            report_sections = []
            
            # 헤더
            report_sections.append("# 🏆 주얼리 업계 종합 분석 리포트")
            report_sections.append("")
            report_sections.append(f"**생성 일시**: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}")
            report_sections.append(f"**분석 신뢰도**: {analysis_result.confidence_score:.1%}")
            report_sections.append("")
            
            # 경영진 요약
            if analysis_result.executive_summary:
                report_sections.append("## 📊 경영진 요약")
                report_sections.append(analysis_result.executive_summary)
                report_sections.append("")
            
            # 핵심 인사이트
            if any([analysis_result.business_insights, analysis_result.technical_insights, analysis_result.market_insights]):
                report_sections.append("## 💡 핵심 인사이트")
                
                if analysis_result.business_insights:
                    report_sections.append("### 📈 비즈니스 인사이트")
                    report_sections.extend(analysis_result.business_insights)
                    report_sections.append("")
                
                if analysis_result.technical_insights:
                    report_sections.append("### 🔧 기술적 인사이트")
                    report_sections.extend(analysis_result.technical_insights)
                    report_sections.append("")
                
                if analysis_result.market_insights:
                    report_sections.append("### 🌍 시장 인사이트")
                    report_sections.extend(analysis_result.market_insights)
                    report_sections.append("")
            
            # 액션 아이템 및 결정사항
            if analysis_result.action_items or analysis_result.key_decisions:
                report_sections.append("## 🎯 실행 계획")
                
                if analysis_result.key_decisions:
                    report_sections.append("### ✅ 주요 결정사항")
                    report_sections.extend(analysis_result.key_decisions)
                    report_sections.append("")
                
                if analysis_result.action_items:
                    report_sections.append("### 📋 액션 아이템")
                    report_sections.extend(analysis_result.action_items)
                    report_sections.append("")
                
                if analysis_result.follow_up_tasks:
                    report_sections.append("### ⏭️ 후속 작업")
                    report_sections.extend(analysis_result.follow_up_tasks)
                    report_sections.append("")
            
            # 주얼리 전문용어 분석
            if analysis_result.jewelry_terminology:
                report_sections.append("## 💎 주얼리 전문용어 분석")
                top_terms = list(analysis_result.jewelry_terminology.items())[:10]
                for term, count in top_terms:
                    report_sections.append(f"• **{term}**: {count}회 언급")
                report_sections.append("")
            
            # 상세 분석
            if analysis_result.detailed_analysis:
                report_sections.append("## 📋 상세 분석")
                report_sections.append(analysis_result.detailed_analysis)
                report_sections.append("")
            
            # 처리 통계
            details = analysis_result.processing_details
            report_sections.append("## 📊 분석 통계")
            report_sections.append(f"• **처리 시간**: {details.get('processing_time', 0):.2f}초")
            report_sections.append(f"• **분석 내용 길이**: {details.get('content_length', 0):,}자")
            report_sections.append(f"• **추출된 인사이트**: {details.get('insights_count', 0)}개")
            report_sections.append(f"• **액션 아이템**: {details.get('action_items_count', 0)}개")
            report_sections.append(f"• **주요 결정사항**: {details.get('decisions_count', 0)}개")
            
            return '\n'.join(report_sections)
            
        except Exception as e:
            self.logger.error(f"종합 리포트 생성 실패: {e}")
            return "리포트 생성 중 오류가 발생했습니다."

# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 한국어 통합 분석 엔진 초기화
    engine = KoreanSummaryEngineV21()
    
    # 샘플 분석
    # sample_content = "다이아몬드 시장 분석 결과..."
    # result = engine.analyze_korean_content(sample_content, "comprehensive")
    # report = engine.generate_comprehensive_report(result)
    # print(report)
    
    print("✅ 한국어 통합 분석 엔진 v2.1 로드 완료!")
