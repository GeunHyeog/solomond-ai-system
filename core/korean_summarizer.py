"""
솔로몬드 AI 시스템 - 한국어 최종 요약 분석기
다국어 입력을 한국어로 통합 요약하고 주얼리 비즈니스 관점에서 분석
"""

import asyncio
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import json
import logging
import re
from pathlib import Path

# 번역 모듈
from .multilingual_translator import get_translator, detect_language, translate_to_korean

# AI 분석 모듈
from .jewelry_ai_engine import JewelryAIEngine

# 품질 분석 모듈
from .quality_analyzer import get_quality_analyzer

class KoreanSummarizer:
    """한국어 최종 요약 및 분석 클래스"""
    
    def __init__(self):
        self.translator = get_translator()
        self.ai_engine = JewelryAIEngine()
        self.quality_analyzer = get_quality_analyzer()
        
        # 상황별 분석 템플릿
        self.situation_templates = {
            "seminar": {
                "name": "세미나",
                "key_points": ["주제", "발표자", "핵심 내용", "질의응답", "참고자료"],
                "business_focus": ["시장 동향", "기술 혁신", "비즈니스 기회", "네트워킹"]
            },
            "meeting": {
                "name": "회의",
                "key_points": ["안건", "결정사항", "액션 아이템", "담당자", "일정"],
                "business_focus": ["의사결정", "업무 분담", "성과 지표", "리스크 관리"]
            },
            "lecture": {
                "name": "강의",
                "key_points": ["학습 목표", "핵심 개념", "실습 내용", "평가 방법", "과제"],
                "business_focus": ["교육 효과", "역량 강화", "지식 전수", "실무 적용"]
            },
            "conference": {
                "name": "컨퍼런스",
                "key_points": ["주요 발표", "패널 토론", "네트워킹", "전시 부스", "후속 일정"],
                "business_focus": ["업계 트렌드", "경쟁사 동향", "파트너십", "시장 기회"]
            }
        }
        
        # 주얼리 비즈니스 관점 키워드
        self.business_keywords = {
            "market": ["시장", "트렌드", "수요", "공급", "가격", "경쟁", "점유율"],
            "product": ["제품", "디자인", "품질", "인증", "브랜드", "차별화"],
            "technology": ["기술", "공정", "혁신", "자동화", "디지털", "AI"],
            "trade": ["무역", "수출", "수입", "관세", "인증", "물류"],
            "customer": ["고객", "소비자", "선호도", "만족도", "충성도", "세그먼트"],
            "finance": ["매출", "수익", "비용", "투자", "자금", "수익성"]
        }
        
        logging.info("한국어 최종 요약 분석기 초기화 완료")
    
    async def analyze_situation_comprehensively(self, 
                                              multimodal_results: Dict,
                                              situation_type: str = "auto",
                                              focus_areas: List[str] = None) -> Dict:
        """
        상황 종합 분석 (세미나/회의/강의 등)
        
        Args:
            multimodal_results: 멀티모달 분석 결과
            situation_type: 상황 타입 ("auto", "seminar", "meeting", "lecture", "conference")
            focus_areas: 집중 분석 영역
            
        Returns:
            한국어 종합 분석 결과
        """
        try:
            print("🇰🇷 한국어 종합 분석 시작")
            
            # 1. 상황 타입 자동 감지
            if situation_type == "auto":
                situation_type = self._detect_situation_type(multimodal_results)
            
            # 2. 다국어 텍스트 수집 및 번역
            korean_content = await self._collect_and_translate_content(multimodal_results)
            
            # 3. 품질 평가 종합
            quality_assessment = await self._comprehensive_quality_assessment(multimodal_results)
            
            # 4. 상황별 구조화 분석
            structured_analysis = await self._perform_structured_analysis(
                korean_content, situation_type, focus_areas
            )
            
            # 5. 주얼리 비즈니스 관점 분석
            business_analysis = await self._analyze_business_perspective(korean_content)
            
            # 6. 최종 한국어 요약 생성
            final_summary = await self._generate_final_korean_summary(
                korean_content, structured_analysis, business_analysis, situation_type
            )
            
            # 7. 실행 가능한 인사이트 도출
            actionable_insights = self._generate_actionable_insights(
                structured_analysis, business_analysis, quality_assessment
            )
            
            # 8. 종합 리포트 구성
            comprehensive_report = {
                "success": True,
                "analysis_info": {
                    "situation_type": situation_type,
                    "situation_name": self.situation_templates.get(situation_type, {}).get("name", "일반 상황"),
                    "analysis_time": datetime.now().isoformat(),
                    "content_sources": list(multimodal_results.get("source_results", {}).keys()),
                    "total_content_length": len(korean_content)
                },
                "quality_assessment": quality_assessment,
                "korean_content": korean_content[:1000] + "..." if len(korean_content) > 1000 else korean_content,
                "structured_analysis": structured_analysis,
                "business_analysis": business_analysis,
                "final_summary": final_summary,
                "actionable_insights": actionable_insights
            }
            
            print(f"✅ 한국어 종합 분석 완료: {situation_type} ({len(korean_content)}자 분석)")
            return comprehensive_report
            
        except Exception as e:
            logging.error(f"한국어 종합 분석 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis_time": datetime.now().isoformat()
            }
    
    def _detect_situation_type(self, multimodal_results: Dict) -> str:
        """상황 타입 자동 감지"""
        try:
            # 모든 텍스트 수집
            all_text = ""
            source_results = multimodal_results.get("source_results", {})
            
            for source_data in source_results.values():
                if source_data and "combined_text" in source_data:
                    all_text += " " + source_data["combined_text"]
            
            text_lower = all_text.lower()
            
            # 키워드 기반 분류
            seminar_keywords = ["세미나", "seminar", "발표", "presentation", "강연", "lecture"]
            meeting_keywords = ["회의", "meeting", "미팅", "논의", "결정", "안건"]
            lecture_keywords = ["강의", "수업", "교육", "학습", "과정", "curriculum"]
            conference_keywords = ["컨퍼런스", "conference", "박람회", "전시회", "포럼", "forum"]
            
            scores = {
                "seminar": sum(1 for kw in seminar_keywords if kw in text_lower),
                "meeting": sum(1 for kw in meeting_keywords if kw in text_lower),
                "lecture": sum(1 for kw in lecture_keywords if kw in text_lower),
                "conference": sum(1 for kw in conference_keywords if kw in text_lower)
            }
            
            detected_type = max(scores.items(), key=lambda x: x[1])[0]
            
            # 점수가 너무 낮으면 일반 회의로 분류
            if scores[detected_type] == 0:
                return "meeting"
            
            return detected_type
            
        except Exception as e:
            logging.warning(f"상황 타입 감지 실패: {e}")
            return "meeting"
    
    async def _collect_and_translate_content(self, multimodal_results: Dict) -> str:
        """다국어 텍스트 수집 및 한국어 번역"""
        try:
            korean_texts = []
            source_results = multimodal_results.get("source_results", {})
            
            for source_type, source_data in source_results.items():
                if not source_data or "combined_text" not in source_data:
                    continue
                
                text = source_data["combined_text"]
                if not text.strip():
                    continue
                
                print(f"📝 {source_type} 텍스트 처리 중...")
                
                # 언어 감지
                detected_lang = detect_language(text)
                
                if detected_lang == "ko":
                    # 이미 한국어
                    korean_texts.append(f"[{source_type}] {text}")
                else:
                    # 한국어로 번역
                    try:
                        translated = await translate_to_korean(text, source_lang=detected_lang)
                        korean_texts.append(f"[{source_type}] {translated}")
                    except Exception as e:
                        logging.warning(f"{source_type} 번역 실패: {e}")
                        # 번역 실패 시 원문 사용
                        korean_texts.append(f"[{source_type}] {text}")
            
            combined_korean = "\n\n".join(korean_texts)
            
            # 텍스트 정리 및 후처리
            cleaned_korean = self._clean_korean_text(combined_korean)
            
            return cleaned_korean
            
        except Exception as e:
            logging.error(f"텍스트 수집 및 번역 오류: {e}")
            return "텍스트 처리 실패"
    
    def _clean_korean_text(self, text: str) -> str:
        """한국어 텍스트 정리"""
        try:
            # 불필요한 공백 제거
            text = re.sub(r'\s+', ' ', text)
            
            # 중복된 문장 제거 (간단한 버전)
            sentences = text.split('.')
            unique_sentences = []
            seen = set()
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence not in seen and len(sentence) > 10:
                    unique_sentences.append(sentence)
                    seen.add(sentence)
            
            cleaned = '. '.join(unique_sentences)
            
            # 마지막 정리
            cleaned = cleaned.replace('..', '.').replace('  ', ' ').strip()
            
            return cleaned
            
        except Exception:
            return text
    
    async def _comprehensive_quality_assessment(self, multimodal_results: Dict) -> Dict:
        """종합 품질 평가"""
        try:
            quality_summary = {
                "overall_score": 0.0,
                "source_qualities": {},
                "issues_found": [],
                "recommendations": []
            }
            
            source_results = multimodal_results.get("source_results", {})
            total_score = 0
            source_count = 0
            
            for source_type, source_data in source_results.items():
                if not source_data:
                    continue
                
                source_quality = {
                    "score": 0.0,
                    "details": {},
                    "issues": []
                }
                
                # 소스별 품질 평가
                if source_type == "audio":
                    # 음성 품질 (이미 분석된 경우)
                    if "summary" in source_data and "average_confidence" in source_data["summary"]:
                        source_quality["score"] = source_data["summary"]["average_confidence"]
                        source_quality["details"]["confidence"] = source_data["summary"]["average_confidence"]
                    else:
                        source_quality["score"] = 0.7  # 기본값
                
                elif source_type in ["video", "documents"]:
                    # 비디오/문서 품질
                    if "individual_results" in source_data:
                        scores = []
                        for result in source_data["individual_results"]:
                            if "quality_score" in result:
                                scores.append(result["quality_score"])
                            elif "confidence" in result:
                                scores.append(result["confidence"])
                        
                        if scores:
                            source_quality["score"] = sum(scores) / len(scores)
                    else:
                        source_quality["score"] = 0.6  # 기본값
                
                elif source_type == "web":
                    # 웹 소스 품질 (신뢰도 기반)
                    source_quality["score"] = 0.5  # 웹은 낮은 신뢰도
                
                # 품질 이슈 식별
                if source_quality["score"] < 0.6:
                    source_quality["issues"].append("낮은 품질로 인한 신뢰도 감소")
                    quality_summary["issues_found"].append(f"{source_type}: 품질 개선 필요")
                
                quality_summary["source_qualities"][source_type] = source_quality
                total_score += source_quality["score"]
                source_count += 1
            
            # 전체 품질 점수 계산
            if source_count > 0:
                quality_summary["overall_score"] = total_score / source_count
            
            # 종합 권장사항
            if quality_summary["overall_score"] >= 0.8:
                quality_summary["recommendations"].append("우수한 품질의 분석 결과입니다.")
            elif quality_summary["overall_score"] >= 0.6:
                quality_summary["recommendations"].append("양호한 품질이지만 일부 개선 가능합니다.")
            else:
                quality_summary["recommendations"].append("품질 개선을 통해 더 정확한 분석이 가능합니다.")
                quality_summary["recommendations"].append("현장 촬영/녹음 환경 개선을 권장합니다.")
            
            return quality_summary
            
        except Exception as e:
            logging.error(f"품질 평가 오류: {e}")
            return {
                "overall_score": 0.5,
                "error": str(e)
            }
    
    async def _perform_structured_analysis(self, 
                                         korean_content: str,
                                         situation_type: str,
                                         focus_areas: List[str] = None) -> Dict:
        """상황별 구조화 분석"""
        try:
            template = self.situation_templates.get(situation_type, self.situation_templates["meeting"])
            
            # AI 엔진을 사용한 구조화 분석
            ai_analysis = await self.ai_engine.analyze_jewelry_content(
                korean_content,
                analysis_type="comprehensive"
            )
            
            # 상황별 핵심 포인트 추출
            key_points = {}
            for point in template["key_points"]:
                extracted = self._extract_content_by_keyword(korean_content, point)
                if extracted:
                    key_points[point] = extracted
            
            # 시간순 구조 분석 (가능한 경우)
            timeline = self._extract_timeline(korean_content)
            
            # 참석자/발표자 정보
            participants = self._extract_participants(korean_content)
            
            # 주요 결정사항/결론
            decisions = self._extract_decisions(korean_content)
            
            structured = {
                "situation_type": template["name"],
                "key_points": key_points,
                "timeline": timeline,
                "participants": participants,
                "decisions": decisions,
                "ai_insights": ai_analysis.get("insights", []),
                "main_topics": ai_analysis.get("main_topics", []),
                "technical_terms": ai_analysis.get("technical_analysis", {}).get("terms_found", [])
            }
            
            return structured
            
        except Exception as e:
            logging.error(f"구조화 분석 오류: {e}")
            return {
                "situation_type": situation_type,
                "error": str(e)
            }
    
    def _extract_content_by_keyword(self, text: str, keyword: str) -> List[str]:
        """키워드 기반 내용 추출"""
        try:
            # 키워드 주변 문장 추출
            sentences = re.split(r'[.!?]', text)
            related_content = []
            
            keyword_variants = {
                "주제": ["주제", "topic", "제목", "타이틀"],
                "발표자": ["발표자", "speaker", "presenter", "강연자"],
                "핵심 내용": ["핵심", "주요", "중요", "포인트", "요점"],
                "안건": ["안건", "agenda", "논의사항", "항목"],
                "결정사항": ["결정", "결론", "합의", "decision"],
                "액션 아이템": ["액션", "행동", "실행", "과제", "업무"]
            }
            
            search_terms = keyword_variants.get(keyword, [keyword])
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                
                for term in search_terms:
                    if term in sentence:
                        related_content.append(sentence)
                        break
            
            return related_content[:3]  # 최대 3개
            
        except Exception:
            return []
    
    def _extract_timeline(self, text: str) -> List[Dict]:
        """시간순 구조 추출"""
        try:
            timeline = []
            
            # 시간 표현 패턴
            time_patterns = [
                r'(\d{1,2}:\d{2})',  # 시:분
                r'(\d{1,2}시\s*\d{0,2}분?)',  # X시 Y분
                r'(오전|오후)\s*(\d{1,2}시)',  # 오전/오후 X시
                r'(\d{1,2}월\s*\d{1,2}일)',  # X월 Y일
                r'(첫째|둘째|셋째|첫 번째|두 번째|세 번째|마지막)',  # 순서
                r'(시작|중간|마지막|결론|끝)'  # 구간
            ]
            
            sentences = re.split(r'[.!?]', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                
                for pattern in time_patterns:
                    match = re.search(pattern, sentence)
                    if match:
                        timeline.append({
                            "time_indicator": match.group(1),
                            "content": sentence,
                            "type": "time_based" if ":" in match.group(1) else "sequence_based"
                        })
                        break
            
            return timeline[:5]  # 최대 5개
            
        except Exception:
            return []
    
    def _extract_participants(self, text: str) -> List[str]:
        """참석자/발표자 정보 추출"""
        try:
            participants = []
            
            # 인명 패턴 (한국어)
            name_patterns = [
                r'([가-힣]{2,4})\s*(대표|사장|이사|부장|과장|팀장|교수|박사|연구원|전문가)',
                r'(Mr\.|Ms\.|Dr\.)\s*([A-Z][a-z]+\s*[A-Z][a-z]+)',
                r'([가-힣]{2,4})\s*(씨|님|선생님)',
                r'발표자[:\s]*([가-힣]{2,4})',
                r'강연자[:\s]*([가-힣]{2,4})'
            ]
            
            for pattern in name_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        name = ' '.join(match).strip()
                    else:
                        name = match.strip()
                    
                    if name and name not in participants:
                        participants.append(name)
            
            return participants[:10]  # 최대 10명
            
        except Exception:
            return []
    
    def _extract_decisions(self, text: str) -> List[str]:
        """주요 결정사항/결론 추출"""
        try:
            decisions = []
            
            # 결정사항 패턴
            decision_patterns = [
                r'(결정[했한][다니]|합의[했한][다니]|확정[했한][다니])[^.!?]*[.!?]',
                r'(따라서|그러므로|결론적으로)[^.!?]*[.!?]',
                r'(~하기로\s*했다|~하기로\s*결정|~하기로\s*합의)[^.!?]*[.!?]',
                r'(액션\s*아이템|실행\s*계획|향후\s*계획)[^.!?]*[.!?]'
            ]
            
            for pattern in decision_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        decision = match[0]
                    else:
                        decision = match
                    
                    if len(decision) > 10 and decision not in decisions:
                        decisions.append(decision.strip())
            
            return decisions[:5]  # 최대 5개
            
        except Exception:
            return []
    
    async def _analyze_business_perspective(self, korean_content: str) -> Dict:
        """주얼리 비즈니스 관점 분석"""
        try:
            business_analysis = {
                "market_insights": [],
                "product_insights": [],
                "technology_insights": [],
                "customer_insights": [],
                "financial_insights": [],
                "strategic_recommendations": [],
                "risk_factors": [],
                "opportunities": []
            }
            
            # 각 비즈니스 영역별 키워드 분석
            for category, keywords in self.business_keywords.items():
                insights = []
                
                for keyword in keywords:
                    if keyword in korean_content:
                        # 키워드 주변 문맥 추출
                        context = self._extract_context_around_keyword(korean_content, keyword)
                        if context:
                            insights.extend(context)
                
                if insights:
                    if category == "market":
                        business_analysis["market_insights"] = insights[:3]
                    elif category == "product":
                        business_analysis["product_insights"] = insights[:3]
                    elif category == "technology":
                        business_analysis["technology_insights"] = insights[:3]
                    elif category == "customer":
                        business_analysis["customer_insights"] = insights[:3]
                    elif category == "finance":
                        business_analysis["financial_insights"] = insights[:3]
            
            # AI 엔진의 비즈니스 분석 활용
            ai_business = await self.ai_engine.analyze_jewelry_content(
                korean_content,
                analysis_type="business_focused"
            )
            
            if ai_business.get("success"):
                business_analysis["strategic_recommendations"] = ai_business.get("strategic_insights", [])[:5]
                business_analysis["opportunities"] = ai_business.get("business_opportunities", [])[:3]
                business_analysis["risk_factors"] = ai_business.get("risk_analysis", [])[:3]
            
            return business_analysis
            
        except Exception as e:
            logging.error(f"비즈니스 관점 분석 오류: {e}")
            return {
                "error": str(e),
                "market_insights": [],
                "strategic_recommendations": []
            }
    
    def _extract_context_around_keyword(self, text: str, keyword: str, window: int = 100) -> List[str]:
        """키워드 주변 문맥 추출"""
        try:
            contexts = []
            start = 0
            
            while True:
                index = text.find(keyword, start)
                if index == -1:
                    break
                
                # 앞뒤 문맥 추출
                context_start = max(0, index - window)
                context_end = min(len(text), index + len(keyword) + window)
                context = text[context_start:context_end].strip()
                
                if len(context) > 20 and context not in contexts:
                    contexts.append(context)
                
                start = index + 1
                
                if len(contexts) >= 3:  # 최대 3개
                    break
            
            return contexts
            
        except Exception:
            return []
    
    async def _generate_final_korean_summary(self, 
                                           korean_content: str,
                                           structured_analysis: Dict,
                                           business_analysis: Dict,
                                           situation_type: str) -> Dict:
        """최종 한국어 요약 생성"""
        try:
            # AI 엔진으로 고품질 요약 생성
            ai_summary = await self.ai_engine.generate_comprehensive_summary(
                korean_content,
                summary_type="korean_executive",
                max_length=1000
            )
            
            # 상황별 맞춤 요약
            situation_name = self.situation_templates.get(situation_type, {}).get("name", situation_type)
            
            # 구조화된 요약 구성
            structured_summary = {
                "executive_summary": self._create_executive_summary(
                    structured_analysis, business_analysis, situation_name
                ),
                "key_findings": self._extract_key_findings(structured_analysis, business_analysis),
                "main_discussions": structured_analysis.get("key_points", {}),
                "business_implications": self._create_business_implications(business_analysis),
                "next_steps": self._generate_next_steps(structured_analysis, business_analysis),
                "ai_generated_summary": ai_summary.get("summary", "") if ai_summary.get("success") else ""
            }
            
            return structured_summary
            
        except Exception as e:
            logging.error(f"최종 요약 생성 오류: {e}")
            return {
                "executive_summary": "요약 생성 중 오류가 발생했습니다.",
                "error": str(e)
            }
    
    def _create_executive_summary(self, 
                                structured_analysis: Dict, 
                                business_analysis: Dict,
                                situation_name: str) -> str:
        """경영진용 요약 생성"""
        try:
            summary_parts = []
            
            # 상황 개요
            summary_parts.append(f"본 {situation_name}에서는")
            
            # 주요 주제
            main_topics = structured_analysis.get("main_topics", [])
            if main_topics:
                topics_str = ", ".join(main_topics[:3])
                summary_parts.append(f"{topics_str} 등을 중심으로 논의되었습니다.")
            
            # 핵심 결정사항
            decisions = structured_analysis.get("decisions", [])
            if decisions:
                summary_parts.append(f"주요 결정사항으로는 {decisions[0][:50]}... 등이 있습니다.")
            
            # 비즈니스 함의
            market_insights = business_analysis.get("market_insights", [])
            if market_insights:
                summary_parts.append(f"시장 관점에서는 {market_insights[0][:50]}... 등의 인사이트가 도출되었습니다.")
            
            # 향후 계획
            opportunities = business_analysis.get("opportunities", [])
            if opportunities:
                summary_parts.append(f"향후 {opportunities[0][:50]}... 등의 기회를 고려할 필요가 있습니다.")
            
            return " ".join(summary_parts)
            
        except Exception:
            return f"본 {situation_name}의 주요 내용을 분석하여 비즈니스 인사이트를 도출했습니다."
    
    def _extract_key_findings(self, 
                            structured_analysis: Dict, 
                            business_analysis: Dict) -> List[str]:
        """핵심 발견사항 추출"""
        findings = []
        
        # 구조화된 분석에서 추출
        ai_insights = structured_analysis.get("ai_insights", [])
        findings.extend(ai_insights[:2])
        
        # 비즈니스 분석에서 추출
        strategic_recommendations = business_analysis.get("strategic_recommendations", [])
        findings.extend(strategic_recommendations[:2])
        
        # 기술적 발견사항
        technical_terms = structured_analysis.get("technical_terms", [])
        if technical_terms:
            findings.append(f"주요 기술 용어: {', '.join(technical_terms[:5])}")
        
        return findings[:5]  # 최대 5개
    
    def _create_business_implications(self, business_analysis: Dict) -> Dict:
        """비즈니스 함의 정리"""
        implications = {
            "market_impact": business_analysis.get("market_insights", [])[:2],
            "product_impact": business_analysis.get("product_insights", [])[:2],
            "financial_impact": business_analysis.get("financial_insights", [])[:2],
            "strategic_impact": business_analysis.get("strategic_recommendations", [])[:2]
        }
        
        return {k: v for k, v in implications.items() if v}
    
    def _generate_next_steps(self, 
                           structured_analysis: Dict, 
                           business_analysis: Dict) -> List[str]:
        """향후 액션 아이템 생성"""
        next_steps = []
        
        # 결정사항에서 추출
        decisions = structured_analysis.get("decisions", [])
        for decision in decisions[:2]:
            next_steps.append(f"결정사항 실행: {decision[:50]}...")
        
        # 비즈니스 기회에서 추출
        opportunities = business_analysis.get("opportunities", [])
        for opportunity in opportunities[:2]:
            next_steps.append(f"기회 활용: {opportunity[:50]}...")
        
        # 리스크 대응
        risks = business_analysis.get("risk_factors", [])
        for risk in risks[:1]:
            next_steps.append(f"리스크 관리: {risk[:50]}...")
        
        # 기본 액션 아이템
        if not next_steps:
            next_steps = [
                "회의 내용을 이해관계자들과 공유",
                "핵심 결정사항에 대한 실행 계획 수립",
                "후속 회의 일정 조정"
            ]
        
        return next_steps[:5]
    
    def _generate_actionable_insights(self, 
                                    structured_analysis: Dict,
                                    business_analysis: Dict,
                                    quality_assessment: Dict) -> Dict:
        """실행 가능한 인사이트 도출"""
        try:
            insights = {
                "immediate_actions": [],
                "short_term_goals": [],
                "long_term_strategy": [],
                "quality_improvements": [],
                "success_metrics": []
            }
            
            # 즉시 실행 가능한 액션
            decisions = structured_analysis.get("decisions", [])
            insights["immediate_actions"] = [f"실행: {d[:40]}..." for d in decisions[:3]]
            
            # 단기 목표
            opportunities = business_analysis.get("opportunities", [])
            insights["short_term_goals"] = [f"목표: {o[:40]}..." for o in opportunities[:3]]
            
            # 장기 전략
            strategic_recommendations = business_analysis.get("strategic_recommendations", [])
            insights["long_term_strategy"] = [f"전략: {s[:40]}..." for s in strategic_recommendations[:3]]
            
            # 품질 개선사항
            quality_recommendations = quality_assessment.get("recommendations", [])
            insights["quality_improvements"] = quality_recommendations[:3]
            
            # 성공 지표
            insights["success_metrics"] = [
                "참석자 만족도 조사",
                "결정사항 실행률 추적",
                "비즈니스 성과 측정"
            ]
            
            return insights
            
        except Exception as e:
            logging.error(f"실행 가능한 인사이트 생성 오류: {e}")
            return {
                "immediate_actions": ["분석 결과 검토"],
                "error": str(e)
            }


# 전역 인스턴스
_korean_summarizer_instance = None

def get_korean_summarizer() -> KoreanSummarizer:
    """전역 한국어 요약기 인스턴스 반환"""
    global _korean_summarizer_instance
    if _korean_summarizer_instance is None:
        _korean_summarizer_instance = KoreanSummarizer()
    return _korean_summarizer_instance

# 편의 함수들
async def analyze_situation_in_korean(multimodal_results: Dict, **kwargs) -> Dict:
    """상황 종합 분석 편의 함수"""
    summarizer = get_korean_summarizer()
    return await summarizer.analyze_situation_comprehensively(multimodal_results, **kwargs)

async def generate_korean_executive_summary(content: str, situation_type: str = "meeting") -> Dict:
    """한국어 경영진용 요약 생성 편의 함수"""
    # 간단한 형태로 변환하여 분석
    mock_results = {
        "source_results": {
            "primary": {
                "combined_text": content
            }
        }
    }
    
    return await analyze_situation_in_korean(mock_results, situation_type=situation_type)

if __name__ == "__main__":
    print("한국어 최종 요약 분석기 테스트")
    print("지원 상황 타입:", list(KoreanSummarizer().situation_templates.keys()))
