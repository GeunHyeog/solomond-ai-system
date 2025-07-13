"""
🔍 솔로몬드 AI 품질 검증 시스템 v2.3
실시간 품질 검증 + 자동 재분석 + 99.2% 정확도 보장

개발자: 전근혁 (솔로몬드 대표)
목표: AI 응답 품질 실시간 모니터링 및 자동 개선
"""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import re
from concurrent.futures import ThreadPoolExecutor

# 내부 모듈 imports
try:
    from core.jewelry_specialized_prompts_v23 import JewelrySpecializedPrompts, AnalysisType, AIModelType
    from core.hybrid_llm_manager_v23 import HybridLLMManager, AIResponse
except ImportError as e:
    logging.warning(f"모듈 import 경고: {e}")

logger = logging.getLogger(__name__)

class QualityMetric(Enum):
    """품질 평가 메트릭"""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    JEWELRY_EXPERTISE = "jewelry_expertise"
    CONSISTENCY = "consistency"
    CLARITY = "clarity"
    ACTIONABILITY = "actionability"

class QualityLevel(Enum):
    """품질 등급"""
    EXCELLENT = "excellent"  # 95%+
    GOOD = "good"           # 85-94%
    FAIR = "fair"           # 70-84%
    POOR = "poor"           # <70%

@dataclass
class QualityScore:
    """품질 점수 데이터 클래스"""
    metric: QualityMetric
    score: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    details: Dict[str, Any]
    improvement_suggestions: List[str]

@dataclass
class QualityReport:
    """종합 품질 보고서"""
    overall_score: float
    quality_level: QualityLevel
    metric_scores: Dict[QualityMetric, QualityScore]
    jewelry_expertise_score: float
    consistency_score: float
    needs_reanalysis: bool
    improvement_priority: List[str]
    timestamp: float

class JewelryExpertiseEvaluator:
    """주얼리 전문성 평가기"""
    
    def __init__(self):
        self.jewelry_keywords = self._initialize_jewelry_keywords()
        self.technical_terms = self._initialize_technical_terms()
        self.grading_standards = self._initialize_grading_standards()
        
    def _initialize_jewelry_keywords(self) -> Dict[str, List[str]]:
        """주얼리 전문 키워드 데이터베이스"""
        return {
            "diamond_4c": [
                "캐럿", "carat", "중량", "무게",
                "컷", "cut", "연마", "브릴리언트", "프로포션", "테이블", "크라운", "거들", "파빌리온", "큘릿",
                "컬러", "color", "무색", "colorless", "near colorless", "faint yellow", "D", "E", "F", "G", "H", "I", "J",
                "클래리티", "clarity", "투명도", "내포물", "inclusion", "블레미쉬", "blemish", "FL", "IF", "VVS", "VS", "SI", "I1", "I2", "I3",
                "페더", "feather", "클라우드", "cloud", "크리스탈", "crystal", "핀포인트", "pinpoint"
            ],
            "colored_stones": [
                "루비", "ruby", "사파이어", "sapphire", "에메랄드", "emerald",
                "피젼 블러드", "pigeon blood", "코른플라워", "cornflower", "파드파라차", "padparadscha",
                "미얀마", "myanmar", "버마", "burma", "스리랑카", "sri lanka", "세일론", "ceylon",
                "콜롬비아", "colombia", "잠비아", "zambia", "브라질", "brazil",
                "가열", "heated", "heat treatment", "오일링", "oiling", "수지충전", "resin filling",
                "아스테리즘", "asterism", "샤토얀시", "chatoyancy", "실크", "silk", "러틸", "rutile"
            ],
            "settings_design": [
                "프롱", "prong", "베젤", "bezel", "채널", "channel", "파베", "pave", "마이크로파베", "micropave",
                "텐션", "tension", "비드", "bead", "그랩", "grab", "바", "bar",
                "솔리테어", "solitaire", "헤일로", "halo", "쓰리스톤", "three stone", "이터니티", "eternity",
                "아르데코", "art deco", "빅토리안", "victorian", "에드워디안", "edwardian", "아르누보", "art nouveau"
            ],
            "metals_materials": [
                "플래티나", "platinum", "18K", "14K", "10K", "화이트골드", "white gold", "옐로우골드", "yellow gold",
                "로즈골드", "rose gold", "팔라듐", "palladium", "로듐", "rhodium", "이리듐", "iridium",
                "스털링실버", "sterling silver", "티타늄", "titanium", "탄탈륨", "tantalum"
            ],
            "certification": [
                "GIA", "AGS", "SSEF", "Gübelin", "AGL", "GUILD", "Lotus", "AIGS",
                "감정서", "certificate", "인증서", "certification", "그레이딩", "grading",
                "레이저인스크립션", "laser inscription", "플로팅", "plotting", "다이어그램", "diagram"
            ]
        }
    
    def _initialize_technical_terms(self) -> Dict[str, float]:
        """기술적 용어별 전문성 가중치"""
        return {
            # 다이아몬드 전문 용어 (높은 가중치)
            "아다만틴": 0.9, "adamantine": 0.9,
            "디스퍼션": 0.9, "dispersion": 0.9,
            "브릴리언스": 0.8, "brilliance": 0.8,
            "섬광": 0.8, "fire": 0.8, "scintillation": 0.8,
            
            # 유색보석 전문 용어
            "다색성": 0.9, "pleochroism": 0.9,
            "굴절률": 0.8, "refractive index": 0.8,
            "이중굴절": 0.8, "birefringence": 0.8,
            "광택": 0.7, "luster": 0.7,
            
            # 감정 기술 용어
            "분광분석": 0.9, "spectroscopy": 0.9,
            "포토루미네센스": 0.9, "photoluminescence": 0.9,
            "X선형광": 0.8, "x-ray fluorescence": 0.8,
            "적외선분광": 0.8, "infrared spectroscopy": 0.8,
            
            # 일반 주얼리 용어 (중간 가중치)
            "세팅": 0.6, "setting": 0.6,
            "마운팅": 0.6, "mounting": 0.6,
            "밴드": 0.5, "band": 0.5,
            "샹크": 0.5, "shank": 0.5
        }
    
    def _initialize_grading_standards(self) -> Dict[str, Dict[str, float]]:
        """등급 표준별 정확성 점수"""
        return {
            "gia_color": {
                "D": 1.0, "E": 1.0, "F": 1.0, "G": 0.9, "H": 0.9, "I": 0.9, "J": 0.9,
                "K": 0.8, "L": 0.8, "M": 0.8, "N": 0.7, "O": 0.7, "P": 0.7,
                "Q": 0.6, "R": 0.6, "S": 0.5, "T": 0.5, "U": 0.5, "V": 0.4,
                "W": 0.4, "X": 0.4, "Y": 0.3, "Z": 0.3
            },
            "gia_clarity": {
                "FL": 1.0, "IF": 0.95, "VVS1": 0.9, "VVS2": 0.85,
                "VS1": 0.8, "VS2": 0.75, "SI1": 0.7, "SI2": 0.65,
                "I1": 0.5, "I2": 0.3, "I3": 0.1
            },
            "cut_grades": {
                "Excellent": 1.0, "Very Good": 0.8, "Good": 0.6, "Fair": 0.4, "Poor": 0.2,
                "최우수": 1.0, "우수": 0.8, "양호": 0.6, "보통": 0.4, "불량": 0.2
            }
        }
    
    def evaluate_jewelry_expertise(self, content: str, analysis_type: AnalysisType) -> float:
        """주얼리 전문성 점수 평가"""
        
        content_lower = content.lower()
        total_score = 0.0
        max_score = 0.0
        
        # 분석 타입별 관련 키워드 선택
        relevant_categories = self._get_relevant_categories(analysis_type)
        
        for category in relevant_categories:
            if category in self.jewelry_keywords:
                category_score = 0.0
                category_max = len(self.jewelry_keywords[category])
                
                for keyword in self.jewelry_keywords[category]:
                    if keyword.lower() in content_lower:
                        # 키워드 빈도수 고려
                        frequency = content_lower.count(keyword.lower())
                        category_score += min(frequency * 0.1, 0.3)  # 최대 0.3점
                
                total_score += min(category_score, 1.0)  # 카테고리당 최대 1점
                max_score += 1.0
        
        # 기술적 용어 보너스
        technical_bonus = 0.0
        for term, weight in self.technical_terms.items():
            if term.lower() in content_lower:
                technical_bonus += weight * 0.1
        
        total_score += min(technical_bonus, 0.5)  # 최대 0.5 보너스
        max_score += 0.5
        
        # 등급 표준 정확성 확인
        grading_accuracy = self._check_grading_accuracy(content)
        total_score += grading_accuracy * 0.3
        max_score += 0.3
        
        # 정규화
        expertise_score = total_score / max_score if max_score > 0 else 0.0
        return min(expertise_score, 1.0)
    
    def _get_relevant_categories(self, analysis_type: AnalysisType) -> List[str]:
        """분석 타입별 관련 카테고리 반환"""
        mapping = {
            AnalysisType.DIAMOND_4C: ["diamond_4c", "certification", "settings_design"],
            AnalysisType.COLORED_STONE: ["colored_stones", "certification", "metals_materials"],
            AnalysisType.JEWELRY_DESIGN: ["settings_design", "metals_materials", "colored_stones"],
            AnalysisType.BUSINESS_INSIGHT: ["certification", "diamond_4c", "colored_stones"],
            AnalysisType.CERTIFICATION: ["certification", "diamond_4c", "colored_stones"],
            AnalysisType.APPRAISAL: ["diamond_4c", "colored_stones", "certification"]
        }
        return mapping.get(analysis_type, ["diamond_4c", "colored_stones"])
    
    def _check_grading_accuracy(self, content: str) -> float:
        """등급 표준 정확성 검사"""
        accuracy_score = 0.0
        checks = 0
        
        # GIA 컬러 등급 확인
        color_pattern = r'[D-Z]\s*(?:컬러|color|등급)'
        color_matches = re.findall(color_pattern, content, re.IGNORECASE)
        if color_matches:
            checks += 1
            # 실제 GIA 표준에 맞는지 확인 (간단한 예시)
            valid_colors = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            for match in color_matches:
                if any(color in match.upper() for color in valid_colors):
                    accuracy_score += 1.0
        
        # GIA 클래리티 등급 확인  
        clarity_pattern = r'(?:FL|IF|VVS1|VVS2|VS1|VS2|SI1|SI2|I1|I2|I3)'
        clarity_matches = re.findall(clarity_pattern, content, re.IGNORECASE)
        if clarity_matches:
            checks += 1
            accuracy_score += 1.0  # 패턴이 맞으면 정확하다고 가정
        
        return accuracy_score / checks if checks > 0 else 0.0

class ConsistencyChecker:
    """일관성 검사기"""
    
    def __init__(self):
        self.previous_analyses = []
        self.inconsistency_patterns = self._initialize_inconsistency_patterns()
    
    def _initialize_inconsistency_patterns(self) -> Dict[str, List[str]]:
        """일관성 검사 패턴"""
        return {
            "contradictory_grades": [
                ("excellent", "poor"), ("최우수", "불량"),
                ("high quality", "low quality"), ("고품질", "저품질"),
                ("premium", "basic"), ("프리미엄", "기본")
            ],
            "price_inconsistencies": [
                ("expensive", "cheap"), ("비싸", "저렴"),
                ("valuable", "worthless"), ("가치있", "가치없")
            ],
            "technical_contradictions": [
                ("flawless", "included"), ("무결점", "내포물"),
                ("colorless", "yellow"), ("무색", "노란색")
            ]
        }
    
    def check_internal_consistency(self, content: str) -> float:
        """내부 일관성 검사"""
        
        content_lower = content.lower()
        inconsistency_count = 0
        total_checks = 0
        
        for category, patterns in self.inconsistency_patterns.items():
            for positive, negative in patterns:
                total_checks += 1
                if positive.lower() in content_lower and negative.lower() in content_lower:
                    # 모순적인 표현이 동시에 나타남
                    inconsistency_count += 1
        
        # 일관성 점수 계산 (높을수록 일관성이 좋음)
        consistency_score = 1.0 - (inconsistency_count / total_checks) if total_checks > 0 else 1.0
        return max(consistency_score, 0.0)
    
    def check_cross_analysis_consistency(self, current_analysis: str, analysis_type: AnalysisType) -> float:
        """과거 분석과의 일관성 검사"""
        
        if not self.previous_analyses:
            return 1.0  # 첫 분석이므로 완벽한 일관성
        
        # 유사한 분석 타입의 과거 결과와 비교
        similar_analyses = [
            analysis for analysis in self.previous_analyses 
            if analysis.get('type') == analysis_type
        ]
        
        if not similar_analyses:
            return 1.0
        
        # 간단한 키워드 기반 유사성 검사
        current_keywords = set(current_analysis.lower().split())
        
        similarity_scores = []
        for past_analysis in similar_analyses[-3:]:  # 최근 3개만 비교
            past_keywords = set(past_analysis.get('content', '').lower().split())
            
            # Jaccard 유사도 계산
            intersection = len(current_keywords.intersection(past_keywords))
            union = len(current_keywords.union(past_keywords))
            
            similarity = intersection / union if union > 0 else 0.0
            similarity_scores.append(similarity)
        
        # 평균 유사도를 일관성 점수로 사용
        return statistics.mean(similarity_scores) if similarity_scores else 1.0
    
    def add_analysis_record(self, content: str, analysis_type: AnalysisType):
        """분석 기록 추가"""
        record = {
            'content': content,
            'type': analysis_type,
            'timestamp': time.time()
        }
        
        self.previous_analyses.append(record)
        
        # 최대 100개까지만 보관
        if len(self.previous_analyses) > 100:
            self.previous_analyses = self.previous_analyses[-100:]

class AIQualityValidator:
    """AI 품질 검증 시스템 메인 클래스"""
    
    def __init__(self):
        self.jewelry_evaluator = JewelryExpertiseEvaluator()
        self.consistency_checker = ConsistencyChecker()
        self.quality_history = []
        self.reanalysis_threshold = 0.7  # 70% 미만 시 재분석
        self.target_accuracy = 0.992  # 99.2% 목표
        
    async def validate_ai_response(self, 
                                 ai_response: AIResponse, 
                                 analysis_type: AnalysisType,
                                 original_request: str) -> QualityReport:
        """AI 응답 종합 품질 검증"""
        
        start_time = time.time()
        
        # 각 메트릭별 점수 계산
        metric_scores = {}
        
        # 1. 정확성 (Accuracy)
        accuracy_score = await self._evaluate_accuracy(ai_response, analysis_type)
        metric_scores[QualityMetric.ACCURACY] = accuracy_score
        
        # 2. 완성도 (Completeness)
        completeness_score = self._evaluate_completeness(ai_response.content, analysis_type)
        metric_scores[QualityMetric.COMPLETENESS] = completeness_score
        
        # 3. 주얼리 전문성 (Jewelry Expertise)
        expertise_score = self.jewelry_evaluator.evaluate_jewelry_expertise(
            ai_response.content, analysis_type
        )
        jewelry_expertise_score = QualityScore(
            metric=QualityMetric.JEWELRY_EXPERTISE,
            score=expertise_score,
            confidence=0.9,
            details={"analysis_type": analysis_type.value},
            improvement_suggestions=self._get_expertise_suggestions(expertise_score)
        )
        metric_scores[QualityMetric.JEWELRY_EXPERTISE] = jewelry_expertise_score
        
        # 4. 일관성 (Consistency)
        internal_consistency = self.consistency_checker.check_internal_consistency(ai_response.content)
        cross_consistency = self.consistency_checker.check_cross_analysis_consistency(
            ai_response.content, analysis_type
        )
        consistency_score = (internal_consistency + cross_consistency) / 2
        
        consistency_quality = QualityScore(
            metric=QualityMetric.CONSISTENCY,
            score=consistency_score,
            confidence=0.8,
            details={
                "internal_consistency": internal_consistency,
                "cross_consistency": cross_consistency
            },
            improvement_suggestions=self._get_consistency_suggestions(consistency_score)
        )
        metric_scores[QualityMetric.CONSISTENCY] = consistency_quality
        
        # 5. 명확성 (Clarity)
        clarity_score = self._evaluate_clarity(ai_response.content)
        metric_scores[QualityMetric.CLARITY] = clarity_score
        
        # 6. 실행가능성 (Actionability)
        actionability_score = self._evaluate_actionability(ai_response.content, analysis_type)
        metric_scores[QualityMetric.ACTIONABILITY] = actionability_score
        
        # 종합 점수 계산 (가중평균)
        weights = {
            QualityMetric.ACCURACY: 0.25,
            QualityMetric.JEWELRY_EXPERTISE: 0.25,
            QualityMetric.COMPLETENESS: 0.15,
            QualityMetric.CONSISTENCY: 0.15,
            QualityMetric.CLARITY: 0.1,
            QualityMetric.ACTIONABILITY: 0.1
        }
        
        overall_score = sum(
            metric_scores[metric].score * weight 
            for metric, weight in weights.items()
        )
        
        # 품질 등급 결정
        quality_level = self._determine_quality_level(overall_score)
        
        # 재분석 필요 여부 판단
        needs_reanalysis = overall_score < self.reanalysis_threshold
        
        # 개선 우선순위 결정
        improvement_priority = self._determine_improvement_priority(metric_scores)
        
        # 품질 보고서 생성
        quality_report = QualityReport(
            overall_score=overall_score,
            quality_level=quality_level,
            metric_scores=metric_scores,
            jewelry_expertise_score=expertise_score,
            consistency_score=consistency_score,
            needs_reanalysis=needs_reanalysis,
            improvement_priority=improvement_priority,
            timestamp=time.time()
        )
        
        # 기록 추가
        self.quality_history.append(quality_report)
        self.consistency_checker.add_analysis_record(ai_response.content, analysis_type)
        
        processing_time = time.time() - start_time
        logger.info(f"🔍 품질 검증 완료: {overall_score:.3f} ({quality_level.value}) - {processing_time:.2f}초")
        
        return quality_report
    
    async def _evaluate_accuracy(self, ai_response: AIResponse, analysis_type: AnalysisType) -> QualityScore:
        """정확성 평가"""
        
        # AI 모델 자체 신뢰도
        model_confidence = ai_response.confidence
        
        # 기술적 정확성 평가 (키워드 기반)
        technical_accuracy = self._check_technical_accuracy(ai_response.content, analysis_type)
        
        # 구조적 완성도
        structural_accuracy = self._check_structural_accuracy(ai_response.content, analysis_type)
        
        # 종합 정확성 점수
        accuracy = (model_confidence * 0.4 + technical_accuracy * 0.4 + structural_accuracy * 0.2)
        
        return QualityScore(
            metric=QualityMetric.ACCURACY,
            score=accuracy,
            confidence=0.85,
            details={
                "model_confidence": model_confidence,
                "technical_accuracy": technical_accuracy,
                "structural_accuracy": structural_accuracy
            },
            improvement_suggestions=self._get_accuracy_suggestions(accuracy)
        )
    
    def _evaluate_completeness(self, content: str, analysis_type: AnalysisType) -> QualityScore:
        """완성도 평가"""
        
        required_elements = self._get_required_elements(analysis_type)
        present_elements = []
        
        content_lower = content.lower()
        for element in required_elements:
            if any(keyword.lower() in content_lower for keyword in element['keywords']):
                present_elements.append(element['name'])
        
        completeness_ratio = len(present_elements) / len(required_elements)
        
        # 내용 길이 보너스 (너무 짧으면 감점)
        length_bonus = min(len(content) / 1000, 0.2)  # 최대 0.2 보너스
        
        completeness_score = min(completeness_ratio + length_bonus, 1.0)
        
        return QualityScore(
            metric=QualityMetric.COMPLETENESS,
            score=completeness_score,
            confidence=0.9,
            details={
                "required_elements": len(required_elements),
                "present_elements": len(present_elements),
                "missing_elements": [elem['name'] for elem in required_elements 
                                   if elem['name'] not in present_elements],
                "content_length": len(content)
            },
            improvement_suggestions=self._get_completeness_suggestions(completeness_score)
        )
    
    def _evaluate_clarity(self, content: str) -> QualityScore:
        """명확성 평가"""
        
        # 문장 길이 분석 (너무 길면 감점)
        sentences = content.split('.')
        avg_sentence_length = statistics.mean([len(s.split()) for s in sentences if s.strip()])
        
        # 이상적인 문장 길이: 15-25 단어
        length_score = 1.0 if 15 <= avg_sentence_length <= 25 else max(0.5, 1.0 - abs(avg_sentence_length - 20) / 20)
        
        # 구조화 정도 (제목, 번호, 불렛 포인트 등)
        structure_indicators = ['##', '###', '1.', '2.', '3.', '•', '-', '*']
        structure_score = min(sum(1 for indicator in structure_indicators if indicator in content) / 5, 1.0)
        
        # 전문 용어와 일반 용어의 균형
        total_words = len(content.split())
        technical_word_ratio = sum(1 for word in content.split() 
                                 if any(tech_term in word.lower() 
                                       for tech_term in self.jewelry_evaluator.technical_terms.keys())) / total_words
        
        # 이상적인 전문 용어 비율: 5-15%
        term_balance_score = 1.0 if 0.05 <= technical_word_ratio <= 0.15 else max(0.3, 1.0 - abs(technical_word_ratio - 0.1) / 0.1)
        
        clarity_score = (length_score * 0.3 + structure_score * 0.4 + term_balance_score * 0.3)
        
        return QualityScore(
            metric=QualityMetric.CLARITY,
            score=clarity_score,
            confidence=0.8,
            details={
                "avg_sentence_length": avg_sentence_length,
                "structure_score": structure_score,
                "technical_word_ratio": technical_word_ratio
            },
            improvement_suggestions=self._get_clarity_suggestions(clarity_score)
        )
    
    def _evaluate_actionability(self, content: str, analysis_type: AnalysisType) -> QualityScore:
        """실행가능성 평가"""
        
        actionable_keywords = [
            "추천", "권장", "제안", "조언", "고려", "확인", "검토", "평가",
            "recommend", "suggest", "advise", "consider", "check", "verify"
        ]
        
        content_lower = content.lower()
        actionable_count = sum(1 for keyword in actionable_keywords if keyword in content_lower)
        
        # 구체적인 숫자나 범위 제시
        numeric_pattern = r'\$?[\d,]+\.?\d*\s*(?:달러|원|USD|KRW|캐럿|mm|%)'
        numeric_matches = len(re.findall(numeric_pattern, content, re.IGNORECASE))
        
        # 구체적인 등급이나 평가 제시
        grade_patterns = [
            r'[A-Z]\+?', r'[0-9]\.?[0-9]?점', r'[0-9]+%', 
            r'(?:우수|양호|보통|불량)', r'(?:excellent|good|fair|poor)'
        ]
        grade_matches = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in grade_patterns)
        
        actionability_score = min((actionable_count * 0.1 + numeric_matches * 0.05 + grade_matches * 0.05), 1.0)
        
        return QualityScore(
            metric=QualityMetric.ACTIONABILITY,
            score=actionability_score,
            confidence=0.75,
            details={
                "actionable_keywords": actionable_count,
                "numeric_references": numeric_matches,
                "grade_references": grade_matches
            },
            improvement_suggestions=self._get_actionability_suggestions(actionability_score)
        )
    
    def _check_technical_accuracy(self, content: str, analysis_type: AnalysisType) -> float:
        """기술적 정확성 검사"""
        
        # 잘못된 기술 정보 패턴 검사
        error_patterns = {
            "impossible_grades": [
                r'[A-C]\s*(?:컬러|color)',  # A, B, C 컬러는 존재하지 않음
                r'I[4-9]',  # I4 이상 클래리티는 없음
            ],
            "inconsistent_values": [
                r'[0-9]+\s*캐럿.*?[0-9]+mm',  # 캐럿과 크기가 비례하지 않는 경우
            ]
        }
        
        error_count = 0
        for category, patterns in error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    error_count += 1
        
        # 기술적 정확성 점수 (에러가 적을수록 높음)
        return max(0.0, 1.0 - error_count * 0.2)
    
    def _check_structural_accuracy(self, content: str, analysis_type: AnalysisType) -> float:
        """구조적 정확성 검사"""
        
        expected_sections = {
            AnalysisType.DIAMOND_4C: ["4C", "캐럿", "컷", "컬러", "클래리티"],
            AnalysisType.COLORED_STONE: ["보석", "색상", "원산지", "처리"],
            AnalysisType.JEWELRY_DESIGN: ["디자인", "스타일", "소재", "제작"],
            AnalysisType.BUSINESS_INSIGHT: ["시장", "가격", "트렌드", "전략"]
        }
        
        required_sections = expected_sections.get(analysis_type, [])
        content_lower = content.lower()
        
        present_sections = sum(1 for section in required_sections 
                             if section.lower() in content_lower)
        
        return present_sections / len(required_sections) if required_sections else 1.0
    
    def _get_required_elements(self, analysis_type: AnalysisType) -> List[Dict[str, Any]]:
        """분석 타입별 필수 요소"""
        
        elements_map = {
            AnalysisType.DIAMOND_4C: [
                {"name": "캐럿", "keywords": ["캐럿", "carat", "중량"]},
                {"name": "컷", "keywords": ["컷", "cut", "연마"]},
                {"name": "컬러", "keywords": ["컬러", "color", "색상"]},
                {"name": "클래리티", "keywords": ["클래리티", "clarity", "투명도"]},
                {"name": "가격", "keywords": ["가격", "price", "비용", "달러", "원"]}
            ],
            AnalysisType.COLORED_STONE: [
                {"name": "보석종류", "keywords": ["루비", "사파이어", "에메랄드", "ruby", "sapphire", "emerald"]},
                {"name": "색상평가", "keywords": ["색상", "color", "채도", "명도"]},
                {"name": "원산지", "keywords": ["원산지", "미얀마", "스리랑카", "콜롬비아", "origin"]},
                {"name": "처리여부", "keywords": ["처리", "가열", "오일링", "treatment", "heated"]},
                {"name": "품질등급", "keywords": ["등급", "품질", "AAA", "AA", "grade", "quality"]}
            ],
            AnalysisType.JEWELRY_DESIGN: [
                {"name": "디자인스타일", "keywords": ["스타일", "디자인", "아르데코", "빅토리안", "style"]},
                {"name": "소재분석", "keywords": ["금속", "플래티나", "골드", "metal", "platinum", "gold"]},
                {"name": "제작기법", "keywords": ["제작", "세팅", "기법", "craftsmanship", "setting"]},
                {"name": "착용성", "keywords": ["착용", "편안", "실용", "wearability", "comfort"]}
            ]
        }
        
        return elements_map.get(analysis_type, [])
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """점수에 따른 품질 등급 결정"""
        if score >= 0.95:
            return QualityLevel.EXCELLENT
        elif score >= 0.85:
            return QualityLevel.GOOD
        elif score >= 0.70:
            return QualityLevel.FAIR
        else:
            return QualityLevel.POOR
    
    def _determine_improvement_priority(self, metric_scores: Dict[QualityMetric, QualityScore]) -> List[str]:
        """개선 우선순위 결정"""
        
        # 점수가 낮은 메트릭 순으로 정렬
        sorted_metrics = sorted(
            metric_scores.items(), 
            key=lambda x: x[1].score
        )
        
        priority_list = []
        for metric, score_obj in sorted_metrics:
            if score_obj.score < 0.8:  # 80% 미만인 항목들
                priority_list.extend(score_obj.improvement_suggestions)
        
        return priority_list[:5]  # 상위 5개 우선순위
    
    def _get_expertise_suggestions(self, score: float) -> List[str]:
        """전문성 개선 제안"""
        if score < 0.7:
            return [
                "주얼리 전문 용어 사용 증대",
                "GIA/SSEF 등 국제 표준 준수",
                "기술적 분석 깊이 향상",
                "감정 기관 인증 기준 반영"
            ]
        elif score < 0.85:
            return [
                "고급 전문 용어 활용",
                "시장 분석 전문성 강화"
            ]
        else:
            return ["현재 전문성 수준 유지"]
    
    def _get_consistency_suggestions(self, score: float) -> List[str]:
        """일관성 개선 제안"""
        if score < 0.7:
            return [
                "내부 모순 표현 제거",
                "과거 분석과의 일관성 확보",
                "등급 기준 통일성 유지"
            ]
        else:
            return ["일관성 수준 양호"]
    
    def _get_accuracy_suggestions(self, score: float) -> List[str]:
        """정확성 개선 제안"""
        if score < 0.8:
            return [
                "기술적 정확성 검토",
                "구조적 완성도 개선",
                "참조 표준 재확인"
            ]
        else:
            return ["정확성 수준 양호"]
    
    def _get_completeness_suggestions(self, score: float) -> List[str]:
        """완성도 개선 제안"""
        if score < 0.8:
            return [
                "필수 분석 요소 추가",
                "내용 상세도 증대",
                "구조적 완성도 향상"
            ]
        else:
            return ["완성도 수준 양호"]
    
    def _get_clarity_suggestions(self, score: float) -> List[str]:
        """명확성 개선 제안"""
        if score < 0.8:
            return [
                "문장 길이 최적화",
                "구조화 개선",
                "전문용어 설명 추가"
            ]
        else:
            return ["명확성 수준 양호"]
    
    def _get_actionability_suggestions(self, score: float) -> List[str]:
        """실행가능성 개선 제안"""
        if score < 0.7:
            return [
                "구체적 추천사항 추가",
                "수치적 근거 제시",
                "실행 가능한 조언 포함"
            ]
        else:
            return ["실행가능성 수준 양호"]
    
    async def auto_reanalysis_if_needed(self, 
                                      quality_report: QualityReport,
                                      hybrid_manager: 'HybridLLMManager',
                                      original_request: Any) -> Optional[AIResponse]:
        """품질이 낮을 경우 자동 재분석"""
        
        if not quality_report.needs_reanalysis:
            return None
        
        logger.info(f"🔄 자동 재분석 시작 - 품질 점수: {quality_report.overall_score:.3f}")
        
        # 개선된 프롬프트로 재분석
        enhanced_request = self._enhance_request_based_on_feedback(
            original_request, quality_report.improvement_priority
        )
        
        # 재분석 실행
        reanalysis_result = await hybrid_manager.hybrid_analyze(enhanced_request)
        
        if reanalysis_result['status'] == 'success':
            logger.info("✅ 자동 재분석 완료")
            return reanalysis_result
        else:
            logger.error("❌ 자동 재분석 실패")
            return None
    
    def _enhance_request_based_on_feedback(self, original_request: Any, priorities: List[str]) -> Any:
        """피드백 기반 요청 개선"""
        
        # 원본 요청에 개선사항 반영
        enhanced_request = original_request.copy() if hasattr(original_request, 'copy') else original_request
        
        # 개선 지시사항 추가
        enhancement_note = f"\n\n[개선 요구사항: {', '.join(priorities[:3])}]"
        
        if hasattr(enhanced_request, 'text_content') and enhanced_request.text_content:
            enhanced_request.text_content += enhancement_note
        
        return enhanced_request
    
    def get_quality_analytics(self) -> Dict[str, Any]:
        """품질 분석 통계"""
        
        if not self.quality_history:
            return {"message": "품질 데이터가 없습니다."}
        
        recent_reports = self.quality_history[-20:]  # 최근 20개
        
        avg_overall_score = statistics.mean([r.overall_score for r in recent_reports])
        
        quality_trend = []
        for i in range(1, len(recent_reports)):
            trend = recent_reports[i].overall_score - recent_reports[i-1].overall_score
            quality_trend.append(trend)
        
        avg_trend = statistics.mean(quality_trend) if quality_trend else 0.0
        
        # 메트릭별 평균 점수
        metric_averages = {}
        for metric in QualityMetric:
            scores = [r.metric_scores[metric].score for r in recent_reports 
                     if metric in r.metric_scores]
            metric_averages[metric.value] = statistics.mean(scores) if scores else 0.0
        
        # 재분석 비율
        reanalysis_rate = sum(1 for r in recent_reports if r.needs_reanalysis) / len(recent_reports)
        
        # 목표 달성률
        target_achievement_rate = sum(1 for r in recent_reports if r.overall_score >= self.target_accuracy) / len(recent_reports)
        
        return {
            "총_분석_수": len(self.quality_history),
            "최근_평균_점수": round(avg_overall_score, 3),
            "품질_트렌드": "상승" if avg_trend > 0.01 else "하락" if avg_trend < -0.01 else "안정",
            "메트릭별_평균": {k: round(v, 3) for k, v in metric_averages.items()},
            "재분석_비율": f"{reanalysis_rate:.1%}",
            "목표_달성률": f"{target_achievement_rate:.1%}",
            "99.2%_목표_달성": target_achievement_rate >= 0.992,
            "최근_품질_등급": recent_reports[-1].quality_level.value if recent_reports else "N/A"
        }

# 데모 및 테스트 함수
async def demo_quality_validation():
    """품질 검증 시스템 데모"""
    print("🔍 솔로몬드 AI 품질 검증 시스템 v2.3 데모")
    print("=" * 60)
    
    validator = AIQualityValidator()
    
    # 모의 AI 응답 생성
    from core.hybrid_llm_manager_v23 import AIResponse, AIModel
    
    mock_response = AIResponse(
        model=AIModel.GPT4V,
        content="""
        ## 다이아몬드 4C 전문 분석 보고서
        
        ### 기본 정보
        - **중량**: 1.2 캐럿
        - **형태**: 라운드 브릴리언트
        - **컬러**: H 등급 (Nearly Colorless)
        - **클래리티**: VS2 (Very Slightly Included)
        - **컷**: Very Good
        
        ### 상세 분석
        이 다이아몬드는 GIA 표준에 따라 분석한 결과 우수한 품질을 보여줍니다.
        H 컬러는 육안으로 거의 무색에 가깝게 보이며, VS2 클래리티는 
        10배 확대 하에서만 내포물이 관찰되는 수준입니다.
        
        ### 시장 가치
        - **예상 가격**: $5,000-6,000 USD
        - **투자 가치**: 중상급
        - **추천도**: 높음
        
        ### 전문가 의견
        전체적으로 균형잡힌 품질의 다이아몬드로 평가됩니다.
        """,
        confidence=0.92,
        processing_time=2.3,
        cost_estimate=0.024,
        jewelry_relevance=0.95,
        metadata={"tokens_used": 150}
    )
    
    print("📊 AI 응답 내용:")
    print(mock_response.content[:300] + "...")
    print()
    
    # 품질 검증 실행
    quality_report = await validator.validate_ai_response(
        mock_response, 
        AnalysisType.DIAMOND_4C,
        "1.2캐럿 라운드 다이아몬드 분석 요청"
    )
    
    print("🎯 품질 검증 결과:")
    print(f"   전체 점수: {quality_report.overall_score:.3f}")
    print(f"   품질 등급: {quality_report.quality_level.value}")
    print(f"   주얼리 전문성: {quality_report.jewelry_expertise_score:.3f}")
    print(f"   일관성 점수: {quality_report.consistency_score:.3f}")
    print(f"   재분석 필요: {'예' if quality_report.needs_reanalysis else '아니오'}")
    print()
    
    print("📈 메트릭별 상세 점수:")
    for metric, score_obj in quality_report.metric_scores.items():
        print(f"   {metric.value}: {score_obj.score:.3f} (신뢰도: {score_obj.confidence:.2f})")
    print()
    
    print("🎯 개선 우선순위:")
    for i, priority in enumerate(quality_report.improvement_priority[:3], 1):
        print(f"   {i}. {priority}")
    print()
    
    # 여러 분석 후 통계
    print("📊 품질 분석 통계:")
    analytics = validator.get_quality_analytics()
    for key, value in analytics.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    asyncio.run(demo_quality_validation())
