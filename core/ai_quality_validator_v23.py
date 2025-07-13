"""
AI Quality Validator v2.3 for Solomond Jewelry AI Platform
AI 품질 검증기 v2.3 - 99.2% 정확도 달성을 위한 실시간 품질 관리 시스템

🎯 목표: 99.2% 분석 정확도 달성
📅 개발기간: 2025.07.13 - 2025.08.03 (3주)
👨‍💼 프로젝트 리더: 전근혁 (솔로몬드 대표)

핵심 기능:
- 실시간 AI 응답 품질 검증
- 주얼리 전문성 점수 측정
- 자동 재분석 트리거 시스템
- 다중 검증 레이어 (논리성, 정확성, 전문성)
- 학습 기반 품질 개선
- 국제 감정 기준 준수 검증
"""

import asyncio
import logging
import json
import time
import re
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import statistics
from collections import defaultdict, deque
import hashlib

# 자연어 처리
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# 기계학습
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 기존 솔로몬드 모듈
try:
    from core.hybrid_llm_manager_v23 import HybridLLMManagerV23, HybridResult, ModelResult, AIModelType
    from core.jewelry_specialized_prompts_v23 import JewelryPromptOptimizerV23, JewelryCategory
    SOLOMOND_V23_AVAILABLE = True
except ImportError as e:
    logging.warning(f"솔로몬드 v2.3 모듈 import 실패: {e}")
    SOLOMOND_V23_AVAILABLE = False

class QualityDimension(Enum):
    """품질 평가 차원"""
    ACCURACY = "accuracy"              # 정확성
    COMPLETENESS = "completeness"      # 완성도
    RELEVANCE = "relevance"           # 관련성
    PROFESSIONALISM = "professionalism"  # 전문성
    CONSISTENCY = "consistency"        # 일관성
    CLARITY = "clarity"               # 명확성
    ACTIONABILITY = "actionability"    # 실행가능성

class ValidationLevel(Enum):
    """검증 수준"""
    BASIC = "basic"           # 기본 검증
    STANDARD = "standard"     # 표준 검증
    EXPERT = "expert"         # 전문가 검증
    CERTIFICATION = "certification"  # 감정서 수준 검증

class QualityStatus(Enum):
    """품질 상태"""
    EXCELLENT = "excellent"   # 99.5% 이상
    GOOD = "good"            # 95-99.4%
    ACCEPTABLE = "acceptable" # 90-94.9%
    NEEDS_IMPROVEMENT = "needs_improvement"  # 85-89.9%
    POOR = "poor"            # 85% 미만

@dataclass
class QualityMetrics:
    """품질 메트릭"""
    accuracy_score: float
    completeness_score: float
    relevance_score: float
    professionalism_score: float
    consistency_score: float
    clarity_score: float
    actionability_score: float
    overall_score: float
    confidence_level: float
    validation_timestamp: datetime

@dataclass
class ValidationRule:
    """검증 규칙"""
    rule_id: str
    category: JewelryCategory
    dimension: QualityDimension
    validation_function: str
    weight: float
    threshold: float
    error_message: str
    improvement_suggestion: str

@dataclass
class QualityIssue:
    """품질 이슈"""
    issue_id: str
    dimension: QualityDimension
    severity: str  # critical, major, minor
    description: str
    location: str  # 문제 발생 위치
    suggestion: str
    auto_fixable: bool

@dataclass
class ValidationResult:
    """검증 결과"""
    content_id: str
    overall_quality: QualityStatus
    metrics: QualityMetrics
    issues: List[QualityIssue]
    passed_rules: List[str]
    failed_rules: List[str]
    improvement_recommendations: List[str]
    reanalysis_required: bool
    validation_time: float

class JewelryKnowledgeBase:
    """주얼리 지식 베이스"""
    
    def __init__(self):
        self.diamond_standards = self._load_diamond_standards()
        self.gemstone_standards = self._load_gemstone_standards()
        self.business_terms = self._load_business_terms()
        self.quality_indicators = self._load_quality_indicators()
    
    def _load_diamond_standards(self) -> Dict[str, Any]:
        """다이아몬드 기준 로드"""
        return {
            "gia_color_scale": ["D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],
            "gia_clarity_scale": ["FL", "IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2", "I3"],
            "gia_cut_grades": ["Excellent", "Very Good", "Good", "Fair", "Poor"],
            "ags_scale": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "carat_ranges": {
                "melee": (0.01, 0.17),
                "small": (0.18, 0.49),
                "medium": (0.50, 0.99),
                "large": (1.00, 2.99),
                "very_large": (3.00, float('inf'))
            },
            "price_factors": ["carat", "color", "clarity", "cut", "fluorescence", "polish", "symmetry"],
            "required_terms": ["캐럿", "컬러", "클래리티", "컷", "4C", "GIA", "등급"]
        }
    
    def _load_gemstone_standards(self) -> Dict[str, Any]:
        """유색보석 기준 로드"""
        return {
            "major_gemstones": ["ruby", "sapphire", "emerald", "루비", "사파이어", "에메랄드"],
            "origins": {
                "ruby": ["Myanmar", "Thailand", "Sri Lanka", "Madagascar", "Mozambique", "미얀마", "태국"],
                "sapphire": ["Kashmir", "Sri Lanka", "Myanmar", "Madagascar", "Australia", "카시미르"],
                "emerald": ["Colombia", "Zambia", "Brazil", "Afghanistan", "콜롬비아", "잠비아"]
            },
            "treatments": {
                "heating": ["heated", "unheated", "가열", "무가열"],
                "oiling": ["minor", "moderate", "significant", "오일링"],
                "other": ["fracture_filling", "diffusion", "irradiation", "균열충전", "확산"]
            },
            "certification_labs": ["SSEF", "Gübelin", "GIA", "AGL", "Lotus"],
            "quality_terms": ["pigeon_blood", "cornflower", "royal_blue", "padparadscha", "피죤블러드"],
            "required_terms": ["원산지", "처리", "감정서", "품질", "희소성"]
        }
    
    def _load_business_terms(self) -> Dict[str, Any]:
        """비즈니스 용어 로드"""
        return {
            "financial_terms": ["가격", "시장가치", "투자", "수익률", "ROI", "마진", "비용"],
            "market_terms": ["트렌드", "수요", "공급", "경쟁", "점유율", "성장률", "예측"],
            "strategy_terms": ["전략", "기회", "위험", "분석", "계획", "실행", "KPI"],
            "customer_terms": ["고객", "소비자", "세그먼트", "니즈", "선호도", "행동패턴"],
            "operational_terms": ["운영", "프로세스", "효율성", "품질관리", "공급망", "유통"],
            "required_metrics": ["시장규모", "성장률", "수익성", "경쟁분석", "추천사항"]
        }
    
    def _load_quality_indicators(self) -> Dict[str, Any]:
        """품질 지표 로드"""
        return {
            "high_quality_phrases": [
                "전문가", "정확한", "상세한", "종합적인", "체계적인", "논리적인",
                "근거있는", "신뢰할 수 있는", "업계 표준", "국제 기준"
            ],
            "low_quality_indicators": [
                "추측", "대략", "아마도", "확실하지 않은", "불명확한",
                "일반적인", "기본적인", "단순한"
            ],
            "completeness_indicators": [
                "결론", "요약", "권장사항", "구체적", "세부사항",
                "분석결과", "평가", "의견"
            ],
            "professionalism_indicators": [
                "GIA", "AGS", "SSEF", "Gübelin", "감정서", "인증서",
                "등급", "기준", "표준", "규정"
            ]
        }

class ContentAnalyzer:
    """콘텐츠 분석기"""
    
    def __init__(self):
        self.knowledge_base = JewelryKnowledgeBase()
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        if NLTK_AVAILABLE:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except:
                self.sentiment_analyzer = None
    
    def analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """콘텐츠 구조 분석"""
        
        # 기본 통계
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        # 섹션 분석
        sections = self._identify_sections(content)
        
        # 리스트/구조 분석
        has_bullets = bool(re.search(r'[•\-\*]\s', content))
        has_numbers = bool(re.search(r'\d+\.\s', content))
        has_headers = bool(re.search(r'^#{1,6}\s', content, re.MULTILINE))
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "sections": sections,
            "has_structure": {
                "bullets": has_bullets,
                "numbers": has_numbers,
                "headers": has_headers
            },
            "avg_sentence_length": word_count / max(sentence_count, 1),
            "readability_score": self._calculate_readability(content)
        }
    
    def _identify_sections(self, content: str) -> List[str]:
        """섹션 식별"""
        sections = []
        
        # 헤더 패턴
        header_patterns = [
            r'^#{1,6}\s+(.+)$',
            r'^\*\*(.+)\*\*$',
            r'^【(.+)】$',
            r'^■\s*(.+)$',
            r'^▶\s*(.+)$'
        ]
        
        for line in content.split('\n'):
            line = line.strip()
            for pattern in header_patterns:
                match = re.search(pattern, line, re.MULTILINE)
                if match:
                    sections.append(match.group(1))
                    break
        
        return sections
    
    def _calculate_readability(self, content: str) -> float:
        """가독성 점수 계산 (간단한 버전)"""
        
        words = content.split()
        sentences = [s for s in content.split('.') if s.strip()]
        
        if not sentences:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # 복잡한 단어 비율
        complex_words = [w for w in words if len(w) > 6]
        complex_ratio = len(complex_words) / len(words) if words else 0
        
        # 간단한 가독성 점수 (0-100)
        score = 100 - (avg_sentence_length * 1.5) - (complex_ratio * 100)
        return max(0, min(100, score))
    
    def analyze_jewelry_terminology(self, content: str, category: JewelryCategory) -> Dict[str, Any]:
        """주얼리 전문용어 분석"""
        
        content_lower = content.lower()
        
        if category == JewelryCategory.DIAMOND_4C:
            standards = self.knowledge_base.diamond_standards
            required_terms = standards["required_terms"]
            
            # 4C 용어 확인
            four_c_coverage = {
                "carat": any(term in content_lower for term in ["캐럿", "carat", "중량"]),
                "color": any(term in content_lower for term in ["컬러", "color", "색상"]),
                "clarity": any(term in content_lower for term in ["클래리티", "clarity", "투명도"]),
                "cut": any(term in content_lower for term in ["컷", "cut", "연마"])
            }
            
            # GIA 등급 확인
            gia_terms_found = []
            for grade in standards["gia_color_scale"] + standards["gia_clarity_scale"] + standards["gia_cut_grades"]:
                if grade.lower() in content_lower:
                    gia_terms_found.append(grade)
            
            return {
                "category_relevance": sum(four_c_coverage.values()) / 4,
                "four_c_coverage": four_c_coverage,
                "gia_terms_found": gia_terms_found,
                "required_terms_count": sum(1 for term in required_terms if term in content_lower),
                "professional_depth": len(gia_terms_found) / 10  # 정규화
            }
        
        elif category == JewelryCategory.COLORED_GEMSTONE:
            standards = self.knowledge_base.gemstone_standards
            
            # 주요 보석 확인
            gemstones_mentioned = []
            for gem in standards["major_gemstones"]:
                if gem.lower() in content_lower:
                    gemstones_mentioned.append(gem)
            
            # 원산지 확인
            origins_mentioned = []
            for gem_type, origins in standards["origins"].items():
                for origin in origins:
                    if origin.lower() in content_lower:
                        origins_mentioned.append(origin)
            
            # 처리 확인
            treatments_mentioned = []
            for treatment_type, treatments in standards["treatments"].items():
                for treatment in treatments:
                    if treatment.lower() in content_lower:
                        treatments_mentioned.append(treatment)
            
            return {
                "category_relevance": min(1.0, len(gemstones_mentioned) / 2),
                "gemstones_mentioned": gemstones_mentioned,
                "origins_mentioned": origins_mentioned,
                "treatments_mentioned": treatments_mentioned,
                "certification_mentioned": any(lab in content for lab in standards["certification_labs"]),
                "professional_depth": (len(origins_mentioned) + len(treatments_mentioned)) / 10
            }
        
        elif category == JewelryCategory.BUSINESS_INSIGHT:
            standards = self.knowledge_base.business_terms
            
            term_counts = {}
            for term_type, terms in standards.items():
                count = sum(1 for term in terms if term in content_lower)
                term_counts[term_type] = count
            
            total_business_terms = sum(term_counts.values())
            
            return {
                "category_relevance": min(1.0, total_business_terms / 20),
                "term_distribution": term_counts,
                "strategic_depth": term_counts.get("strategy_terms", 0) / 5,
                "analytical_depth": term_counts.get("market_terms", 0) / 5,
                "actionability": term_counts.get("operational_terms", 0) / 5
            }
        
        else:
            # 기본 분석
            return {
                "category_relevance": 0.5,
                "professional_depth": 0.5,
                "term_coverage": 0.5
            }
    
    def detect_logical_inconsistencies(self, content: str) -> List[str]:
        """논리적 불일치 감지"""
        
        inconsistencies = []
        
        # 모순되는 표현 검사
        contradiction_patterns = [
            (r'최고\s*등급', r'낮은\s*품질'),
            (r'무가열', r'가열\s*처리'),
            (r'FL\s*등급', r'내포물'),
            (r'투자\s*가치\s*높음', r'품질\s*낮음'),
            (r'희귀', r'흔함|일반적')
        ]
        
        for positive_pattern, negative_pattern in contradiction_patterns:
            if re.search(positive_pattern, content, re.IGNORECASE) and re.search(negative_pattern, content, re.IGNORECASE):
                inconsistencies.append(f"모순된 표현 발견: '{positive_pattern}' vs '{negative_pattern}'")
        
        # 숫자 불일치 검사
        price_numbers = re.findall(r'\$[\d,]+', content)
        carat_numbers = re.findall(r'([\d.]+)\s*(?:캐럿|carat)', content, re.IGNORECASE)
        
        if len(price_numbers) > 1:
            # 가격 범위 일관성 검사
            prices = [int(p.replace('$', '').replace(',', '')) for p in price_numbers]
            if max(prices) < min(prices):
                inconsistencies.append("가격 범위가 논리적이지 않음")
        
        return inconsistencies

class QualityRuleEngine:
    """품질 규칙 엔진"""
    
    def __init__(self):
        self.rules = self._load_validation_rules()
        self.analyzer = ContentAnalyzer()
    
    def _load_validation_rules(self) -> List[ValidationRule]:
        """검증 규칙 로드"""
        
        rules = []
        
        # 정확성 규칙
        rules.extend([
            ValidationRule(
                rule_id="ACC_001",
                category=JewelryCategory.DIAMOND_4C,
                dimension=QualityDimension.ACCURACY,
                validation_function="validate_diamond_grades",
                weight=0.25,
                threshold=0.9,
                error_message="다이아몬드 등급이 GIA 표준과 일치하지 않음",
                improvement_suggestion="GIA 공식 등급 체계를 확인하여 정확한 등급을 사용하세요"
            ),
            ValidationRule(
                rule_id="ACC_002",
                category=JewelryCategory.COLORED_GEMSTONE,
                dimension=QualityDimension.ACCURACY,
                validation_function="validate_gemstone_origins",
                weight=0.3,
                threshold=0.85,
                error_message="보석 원산지 정보가 부정확하거나 누락됨",
                improvement_suggestion="SSEF/Gübelin 기준의 정확한 원산지 정보를 포함하세요"
            ),
            ValidationRule(
                rule_id="ACC_003",
                category=JewelryCategory.BUSINESS_INSIGHT,
                dimension=QualityDimension.ACCURACY,
                validation_function="validate_market_data",
                weight=0.2,
                threshold=0.8,
                error_message="시장 데이터가 현실적이지 않거나 근거가 부족함",
                improvement_suggestion="최신 시장 데이터와 신뢰할 수 있는 출처를 인용하세요"
            )
        ])
        
        # 완성도 규칙
        rules.extend([
            ValidationRule(
                rule_id="COMP_001",
                category=JewelryCategory.DIAMOND_4C,
                dimension=QualityDimension.COMPLETENESS,
                validation_function="validate_4c_completeness",
                weight=0.2,
                threshold=0.95,
                error_message="4C 분석이 불완전함 (일부 요소 누락)",
                improvement_suggestion="Carat, Color, Clarity, Cut 모든 요소를 포함하여 분석하세요"
            ),
            ValidationRule(
                rule_id="COMP_002",
                category=JewelryCategory.BUSINESS_INSIGHT,
                dimension=QualityDimension.COMPLETENESS,
                validation_function="validate_business_completeness",
                weight=0.25,
                threshold=0.9,
                error_message="비즈니스 분석이 불완전함 (핵심 요소 누락)",
                improvement_suggestion="시장분석, 경쟁분석, 재무분석, 실행계획을 모두 포함하세요"
            )
        ])
        
        # 전문성 규칙
        rules.extend([
            ValidationRule(
                rule_id="PROF_001",
                category=JewelryCategory.DIAMOND_4C,
                dimension=QualityDimension.PROFESSIONALISM,
                validation_function="validate_professional_terminology",
                weight=0.2,
                threshold=0.9,
                error_message="전문 용어 사용이 부족하거나 부정확함",
                improvement_suggestion="GIA, AGS 등 국제 기준의 정확한 전문 용어를 사용하세요"
            ),
            ValidationRule(
                rule_id="PROF_002",
                category=JewelryCategory.COLORED_GEMSTONE,
                dimension=QualityDimension.PROFESSIONALISM,
                validation_function="validate_gemstone_expertise",
                weight=0.25,
                threshold=0.85,
                error_message="유색보석 전문 지식이 부족함",
                improvement_suggestion="SSEF, Gübelin 등 전문 감정 기관의 기준을 적용하세요"
            )
        ])
        
        return rules
    
    def validate_content(self, content: str, category: JewelryCategory, 
                        validation_level: ValidationLevel = ValidationLevel.STANDARD) -> List[QualityIssue]:
        """콘텐츠 검증"""
        
        issues = []
        
        # 카테고리별 규칙 필터링
        applicable_rules = [r for r in self.rules if r.category == category]
        
        # 검증 수준에 따른 임계값 조정
        threshold_multiplier = {
            ValidationLevel.BASIC: 0.8,
            ValidationLevel.STANDARD: 1.0,
            ValidationLevel.EXPERT: 1.1,
            ValidationLevel.CERTIFICATION: 1.2
        }[validation_level]
        
        for rule in applicable_rules:
            try:
                score = self._execute_validation_function(rule.validation_function, content, category)
                adjusted_threshold = rule.threshold * threshold_multiplier
                
                if score < adjusted_threshold:
                    severity = self._determine_severity(score, adjusted_threshold, rule.weight)
                    
                    issue = QualityIssue(
                        issue_id=f"{rule.rule_id}_{int(time.time())}",
                        dimension=rule.dimension,
                        severity=severity,
                        description=rule.error_message,
                        location=self._find_issue_location(content, rule),
                        suggestion=rule.improvement_suggestion,
                        auto_fixable=self._is_auto_fixable(rule)
                    )
                    issues.append(issue)
            
            except Exception as e:
                logging.error(f"검증 규칙 {rule.rule_id} 실행 오류: {e}")
        
        return issues
    
    def _execute_validation_function(self, function_name: str, content: str, category: JewelryCategory) -> float:
        """검증 함수 실행"""
        
        if function_name == "validate_diamond_grades":
            return self._validate_diamond_grades(content)
        elif function_name == "validate_gemstone_origins":
            return self._validate_gemstone_origins(content)
        elif function_name == "validate_market_data":
            return self._validate_market_data(content)
        elif function_name == "validate_4c_completeness":
            return self._validate_4c_completeness(content)
        elif function_name == "validate_business_completeness":
            return self._validate_business_completeness(content)
        elif function_name == "validate_professional_terminology":
            return self._validate_professional_terminology(content, category)
        elif function_name == "validate_gemstone_expertise":
            return self._validate_gemstone_expertise(content)
        else:
            return 0.5  # 기본값
    
    def _validate_diamond_grades(self, content: str) -> float:
        """다이아몬드 등급 검증"""
        
        standards = self.analyzer.knowledge_base.diamond_standards
        content_upper = content.upper()
        
        # GIA 등급 체계 확인
        valid_grades = 0
        total_grades = 0
        
        # 컬러 등급 검증
        color_mentions = re.findall(r'\b([D-Z])\s*(?:컬러|색상|color)', content_upper)
        for color in color_mentions:
            total_grades += 1
            if color in standards["gia_color_scale"]:
                valid_grades += 1
        
        # 클래리티 등급 검증
        clarity_pattern = r'\b(FL|IF|VVS[12]|VS[12]|SI[12]|I[123])\b'
        clarity_mentions = re.findall(clarity_pattern, content_upper)
        for clarity in clarity_mentions:
            total_grades += 1
            if clarity in standards["gia_clarity_scale"]:
                valid_grades += 1
        
        # 컷 등급 검증
        cut_pattern = r'\b(EXCELLENT|VERY\s*GOOD|GOOD|FAIR|POOR)\b'
        cut_mentions = re.findall(cut_pattern, content_upper)
        for cut in cut_mentions:
            total_grades += 1
            if cut.replace(' ', ' ') in [g.upper() for g in standards["gia_cut_grades"]]:
                valid_grades += 1
        
        if total_grades == 0:
            return 0.5  # 등급 언급이 없으면 중간 점수
        
        return valid_grades / total_grades
    
    def _validate_gemstone_origins(self, content: str) -> float:
        """보석 원산지 검증"""
        
        standards = self.analyzer.knowledge_base.gemstone_standards
        content_lower = content.lower()
        
        score = 0.0
        mentions = 0
        
        # 각 보석별 원산지 확인
        for gemstone, valid_origins in standards["origins"].items():
            if gemstone in content_lower:
                mentions += 1
                # 해당 보석의 유효한 원산지가 언급되었는지 확인
                origin_found = any(origin.lower() in content_lower for origin in valid_origins)
                if origin_found:
                    score += 1
        
        if mentions == 0:
            return 0.8  # 원산지 언급이 없으면 기본 점수
        
        return score / mentions
    
    def _validate_market_data(self, content: str) -> float:
        """시장 데이터 검증"""
        
        score_factors = []
        
        # 가격 정보의 현실성 검사
        price_pattern = r'\$[\d,]+'
        prices = re.findall(price_pattern, content)
        if prices:
            # 가격 범위가 현실적인지 확인
            price_values = [int(p.replace('$', '').replace(',', '')) for p in prices]
            if all(100 <= p <= 10000000 for p in price_values):  # 현실적인 범위
                score_factors.append(1.0)
            else:
                score_factors.append(0.3)
        
        # 성장률 정보의 현실성
        growth_pattern = r'(\d+(?:\.\d+)?)\s*%'
        growth_rates = re.findall(growth_pattern, content)
        if growth_rates:
            rates = [float(r) for r in growth_rates]
            if all(-50 <= r <= 100 for r in rates):  # 현실적인 성장률
                score_factors.append(1.0)
            else:
                score_factors.append(0.4)
        
        # 근거 제시 여부
        evidence_keywords = ["출처", "데이터", "조사", "연구", "보고서", "통계"]
        if any(keyword in content for keyword in evidence_keywords):
            score_factors.append(1.0)
        else:
            score_factors.append(0.6)
        
        return statistics.mean(score_factors) if score_factors else 0.5
    
    def _validate_4c_completeness(self, content: str) -> float:
        """4C 완성도 검증"""
        
        content_lower = content.lower()
        
        # 4C 요소별 체크
        four_c_coverage = {
            "carat": any(term in content_lower for term in ["캐럿", "carat", "중량", "무게"]),
            "color": any(term in content_lower for term in ["컬러", "color", "색상", "색깔"]),
            "clarity": any(term in content_lower for term in ["클래리티", "clarity", "투명도", "내포물"]),
            "cut": any(term in content_lower for term in ["컷", "cut", "연마", "폴리시", "시메트리"])
        }
        
        # 각 C에 대한 상세 설명 체크
        detailed_analysis = 0
        
        if four_c_coverage["carat"]:
            if re.search(r'\d+\.?\d*\s*(?:캐럿|carat)', content_lower):
                detailed_analysis += 0.25
        
        if four_c_coverage["color"]:
            if re.search(r'\b[D-Z]\s*(?:컬러|색상)', content, re.IGNORECASE):
                detailed_analysis += 0.25
        
        if four_c_coverage["clarity"]:
            if re.search(r'\b(?:FL|IF|VVS|VS|SI|I)\d?\b', content, re.IGNORECASE):
                detailed_analysis += 0.25
        
        if four_c_coverage["cut"]:
            if re.search(r'(?:excellent|very good|good|fair|poor)', content, re.IGNORECASE):
                detailed_analysis += 0.25
        
        basic_coverage = sum(four_c_coverage.values()) / 4
        return (basic_coverage * 0.6) + (detailed_analysis * 0.4)
    
    def _validate_business_completeness(self, content: str) -> float:
        """비즈니스 완성도 검증"""
        
        content_lower = content.lower()
        
        # 필수 비즈니스 요소들
        business_elements = {
            "market_analysis": any(term in content_lower for term in ["시장", "분석", "규모", "성장"]),
            "competition": any(term in content_lower for term in ["경쟁", "경쟁사", "경쟁분석", "브랜드"]),
            "financial": any(term in content_lower for term in ["매출", "수익", "가격", "비용", "roi"]),
            "strategy": any(term in content_lower for term in ["전략", "계획", "목표", "실행"]),
            "recommendations": any(term in content_lower for term in ["권장", "추천", "제안", "결론"])
        }
        
        # 정량적 데이터 포함 여부
        has_numbers = bool(re.search(r'\d+\.?\d*\s*%', content))
        has_financial_data = bool(re.search(r'\$[\d,]+', content))
        
        basic_score = sum(business_elements.values()) / len(business_elements)
        data_bonus = 0.1 if has_numbers else 0
        financial_bonus = 0.1 if has_financial_data else 0
        
        return min(1.0, basic_score + data_bonus + financial_bonus)
    
    def _validate_professional_terminology(self, content: str, category: JewelryCategory) -> float:
        """전문 용어 사용 검증"""
        
        if category == JewelryCategory.DIAMOND_4C:
            professional_terms = [
                "GIA", "AGS", "감정서", "인증서", "등급", "폴리시", "시메트리",
                "형광성", "거들", "큘릿", "프로포션"
            ]
        elif category == JewelryCategory.COLORED_GEMSTONE:
            professional_terms = [
                "SSEF", "Gübelin", "원산지", "가열", "무가열", "오일링", "처리",
                "내포물", "피죤블러드", "코른플라워블루", "패드파라차"
            ]
        else:
            professional_terms = [
                "분석", "평가", "전략", "시장", "경쟁", "수익성", "ROI",
                "KPI", "벤치마크", "최적화"
            ]
        
        content_lower = content.lower()
        terms_found = sum(1 for term in professional_terms if term.lower() in content_lower)
        
        return min(1.0, terms_found / len(professional_terms) * 2)
    
    def _validate_gemstone_expertise(self, content: str) -> float:
        """유색보석 전문성 검증"""
        
        content_lower = content.lower()
        
        expertise_indicators = [
            # 처리 관련 전문 지식
            ("처리", ["가열", "무가열", "오일링", "수지", "확산", "조사"]),
            # 원산지 관련 지식  
            ("원산지", ["미얀마", "카시미르", "콜롬비아", "실론", "마다가스카르"]),
            # 품질 용어
            ("품질", ["피죤블러드", "코른플라워", "패드파라차", "로얄블루"]),
            # 감정 기관
            ("감정", ["SSEF", "Gübelin", "GIA", "AGL", "Lotus"])
        ]
        
        scores = []
        for category, terms in expertise_indicators:
            found_terms = sum(1 for term in terms if term in content_lower)
            category_score = min(1.0, found_terms / len(terms) * 2)
            scores.append(category_score)
        
        return statistics.mean(scores) if scores else 0.5
    
    def _determine_severity(self, score: float, threshold: float, weight: float) -> str:
        """심각도 결정"""
        
        gap = threshold - score
        weighted_gap = gap * weight
        
        if weighted_gap > 0.3:
            return "critical"
        elif weighted_gap > 0.15:
            return "major"
        else:
            return "minor"
    
    def _find_issue_location(self, content: str, rule: ValidationRule) -> str:
        """이슈 위치 찾기"""
        
        # 간단한 위치 추정
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ["grade", "등급", "품질", "market", "시장"]):
                return f"Line {i+1}"
        
        return "전체 내용"
    
    def _is_auto_fixable(self, rule: ValidationRule) -> bool:
        """자동 수정 가능 여부"""
        
        # 일부 규칙은 자동 수정 가능
        auto_fixable_rules = ["COMP_001", "PROF_001"]
        return rule.rule_id in auto_fixable_rules

class AIQualityValidatorV23:
    """AI 품질 검증기 v2.3 - 99.2% 정확도 달성을 위한 핵심 시스템"""
    
    def __init__(self, config_path: str = "config/quality_validator_v23.json"):
        self.config_path = config_path
        self.rule_engine = QualityRuleEngine()
        self.content_analyzer = ContentAnalyzer()
        
        # 성능 추적
        self.validation_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        
        # 하이브리드 LLM 매니저 연동
        if SOLOMOND_V23_AVAILABLE:
            try:
                self.llm_manager = HybridLLMManagerV23()
                self.prompt_optimizer = JewelryPromptOptimizerV23()
                self.llm_integration = True
            except Exception as e:
                logging.warning(f"LLM 매니저 연동 실패: {e}")
                self.llm_integration = False
        else:
            self.llm_integration = False
        
        # 학습 기반 개선
        self.quality_patterns = {}
        self.improvement_suggestions = {}
        
        logging.info("🔍 AI 품질 검증기 v2.3 초기화 완료")
    
    async def validate_ai_response(self, 
                                 content: str,
                                 category: JewelryCategory,
                                 expected_accuracy: float = 0.992,
                                 validation_level: ValidationLevel = ValidationLevel.EXPERT) -> ValidationResult:
        """AI 응답 품질 검증 - 메인 진입점"""
        
        start_time = time.time()
        content_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # 1. 기본 콘텐츠 분석
        structure_analysis = self.content_analyzer.analyze_content_structure(content)
        terminology_analysis = self.content_analyzer.analyze_jewelry_terminology(content, category)
        logical_issues = self.content_analyzer.detect_logical_inconsistencies(content)
        
        # 2. 규칙 기반 검증
        quality_issues = self.rule_engine.validate_content(content, category, validation_level)
        
        # 3. 차원별 품질 메트릭 계산
        metrics = self._calculate_quality_metrics(
            content, category, structure_analysis, terminology_analysis, quality_issues
        )
        
        # 4. 전체 품질 상태 결정
        overall_quality = self._determine_overall_quality(metrics.overall_score, expected_accuracy)
        
        # 5. 개선 권장사항 생성
        improvement_recommendations = self._generate_improvement_recommendations(
            quality_issues, metrics, category
        )
        
        # 6. 재분석 필요 여부 판단
        reanalysis_required = self._should_reanalyze(metrics.overall_score, expected_accuracy, quality_issues)
        
        # 7. 검증 결과 생성
        validation_result = ValidationResult(
            content_id=content_id,
            overall_quality=overall_quality,
            metrics=metrics,
            issues=quality_issues,
            passed_rules=[],  # TODO: 통과한 규칙 추적
            failed_rules=[issue.issue_id for issue in quality_issues],
            improvement_recommendations=improvement_recommendations,
            reanalysis_required=reanalysis_required,
            validation_time=time.time() - start_time
        )
        
        # 8. 성능 추적 업데이트
        self._update_performance_tracking(validation_result, category)
        
        # 9. 학습 기반 개선
        await self._update_quality_patterns(content, category, validation_result)
        
        logging.info(f"🔍 품질 검증 완료 - ID: {content_id}, 점수: {metrics.overall_score:.3f}, 상태: {overall_quality.value}")
        
        return validation_result
    
    def _calculate_quality_metrics(self, 
                                 content: str,
                                 category: JewelryCategory,
                                 structure_analysis: Dict[str, Any],
                                 terminology_analysis: Dict[str, Any],
                                 quality_issues: List[QualityIssue]) -> QualityMetrics:
        """품질 메트릭 계산"""
        
        # 기본 점수들
        accuracy_score = self._calculate_accuracy_score(content, category, terminology_analysis)
        completeness_score = self._calculate_completeness_score(content, category, structure_analysis)
        relevance_score = terminology_analysis.get("category_relevance", 0.5)
        professionalism_score = terminology_analysis.get("professional_depth", 0.5)
        consistency_score = self._calculate_consistency_score(content, quality_issues)
        clarity_score = structure_analysis.get("readability_score", 50) / 100
        actionability_score = self._calculate_actionability_score(content, category)
        
        # 이슈 기반 점수 조정
        issue_penalty = self._calculate_issue_penalty(quality_issues)
        
        # 조정된 점수들
        adjusted_scores = {
            "accuracy": max(0, accuracy_score - issue_penalty * 0.3),
            "completeness": max(0, completeness_score - issue_penalty * 0.2),
            "relevance": max(0, relevance_score - issue_penalty * 0.1),
            "professionalism": max(0, professionalism_score - issue_penalty * 0.25),
            "consistency": max(0, consistency_score - issue_penalty * 0.4),
            "clarity": max(0, clarity_score - issue_penalty * 0.15),
            "actionability": max(0, actionability_score - issue_penalty * 0.2)
        }
        
        # 가중 평균으로 전체 점수 계산
        weights = {
            "accuracy": 0.25,
            "completeness": 0.20,
            "relevance": 0.15,
            "professionalism": 0.20,
            "consistency": 0.10,
            "clarity": 0.05,
            "actionability": 0.05
        }
        
        overall_score = sum(score * weights[dim] for dim, score in adjusted_scores.items())
        
        # 신뢰도 계산
        confidence_level = self._calculate_confidence_level(adjusted_scores, quality_issues)
        
        return QualityMetrics(
            accuracy_score=adjusted_scores["accuracy"],
            completeness_score=adjusted_scores["completeness"],
            relevance_score=adjusted_scores["relevance"],
            professionalism_score=adjusted_scores["professionalism"],
            consistency_score=adjusted_scores["consistency"],
            clarity_score=adjusted_scores["clarity"],
            actionability_score=adjusted_scores["actionability"],
            overall_score=overall_score,
            confidence_level=confidence_level,
            validation_timestamp=datetime.now()
        )
    
    def _calculate_accuracy_score(self, content: str, category: JewelryCategory, 
                                terminology_analysis: Dict[str, Any]) -> float:
        """정확성 점수 계산"""
        
        base_score = 0.7  # 기본 점수
        
        # 전문 용어 정확성
        terminology_bonus = terminology_analysis.get("professional_depth", 0) * 0.2
        
        # 카테고리별 특수 검증
        if category == JewelryCategory.DIAMOND_4C:
            # GIA 등급 체계 정확성
            gia_accuracy = self.rule_engine._validate_diamond_grades(content)
            base_score += gia_accuracy * 0.2
        
        elif category == JewelryCategory.COLORED_GEMSTONE:
            # 원산지 정확성
            origin_accuracy = self.rule_engine._validate_gemstone_origins(content)
            base_score += origin_accuracy * 0.2
        
        elif category == JewelryCategory.BUSINESS_INSIGHT:
            # 시장 데이터 현실성
            market_accuracy = self.rule_engine._validate_market_data(content)
            base_score += market_accuracy * 0.2
        
        return min(1.0, base_score + terminology_bonus)
    
    def _calculate_completeness_score(self, content: str, category: JewelryCategory,
                                    structure_analysis: Dict[str, Any]) -> float:
        """완성도 점수 계산"""
        
        # 구조적 완성도
        structure_score = 0.5
        
        if structure_analysis["sections"]:
            structure_score += 0.2
        if structure_analysis["has_structure"]["bullets"] or structure_analysis["has_structure"]["numbers"]:
            structure_score += 0.1
        if structure_analysis["word_count"] > 200:
            structure_score += 0.2
        
        # 카테고리별 완성도
        if category == JewelryCategory.DIAMOND_4C:
            completeness = self.rule_engine._validate_4c_completeness(content)
        elif category == JewelryCategory.BUSINESS_INSIGHT:
            completeness = self.rule_engine._validate_business_completeness(content)
        else:
            completeness = 0.8  # 기본값
        
        return (structure_score * 0.3) + (completeness * 0.7)
    
    def _calculate_consistency_score(self, content: str, quality_issues: List[QualityIssue]) -> float:
        """일관성 점수 계산"""
        
        # 논리적 불일치 확인
        logical_issues = self.content_analyzer.detect_logical_inconsistencies(content)
        
        base_score = 1.0
        
        # 논리적 불일치 페널티
        if logical_issues:
            base_score -= len(logical_issues) * 0.2
        
        # 품질 이슈 중 일관성 관련 페널티
        consistency_issues = [issue for issue in quality_issues if issue.dimension == QualityDimension.CONSISTENCY]
        if consistency_issues:
            base_score -= len(consistency_issues) * 0.15
        
        return max(0, base_score)
    
    def _calculate_actionability_score(self, content: str, category: JewelryCategory) -> float:
        """실행가능성 점수 계산"""
        
        content_lower = content.lower()
        
        # 실행 가능한 권장사항 키워드
        actionable_keywords = [
            "권장", "추천", "제안", "방법", "전략", "계획", "단계", "실행",
            "구체적", "세부", "방안", "해결", "개선", "최적화"
        ]
        
        actionable_count = sum(1 for keyword in actionable_keywords if keyword in content_lower)
        base_score = min(1.0, actionable_count / 10)
        
        # 구체적인 수치나 방법 제시
        has_specific_numbers = bool(re.search(r'\d+\.?\d*\s*%', content))
        has_step_by_step = bool(re.search(r'\d+\.\s', content))
        
        if has_specific_numbers:
            base_score += 0.2
        if has_step_by_step:
            base_score += 0.2
        
        return min(1.0, base_score)
    
    def _calculate_issue_penalty(self, quality_issues: List[QualityIssue]) -> float:
        """이슈 기반 페널티 계산"""
        
        if not quality_issues:
            return 0.0
        
        penalty = 0.0
        
        for issue in quality_issues:
            if issue.severity == "critical":
                penalty += 0.3
            elif issue.severity == "major":
                penalty += 0.15
            elif issue.severity == "minor":
                penalty += 0.05
        
        return min(1.0, penalty)
    
    def _calculate_confidence_level(self, scores: Dict[str, float], quality_issues: List[QualityIssue]) -> float:
        """신뢰도 계산"""
        
        # 점수들의 분산 계산
        score_values = list(scores.values())
        score_variance = statistics.variance(score_values) if len(score_values) > 1 else 0
        
        # 분산이 낮을수록 신뢰도 높음
        variance_factor = max(0, 1 - score_variance * 4)
        
        # 이슈 개수에 따른 신뢰도 조정
        issue_factor = max(0, 1 - len(quality_issues) * 0.1)
        
        # 최소 점수가 너무 낮으면 신뢰도 하락
        min_score = min(score_values)
        min_score_factor = min_score if min_score > 0.5 else min_score * 0.5
        
        confidence = (variance_factor * 0.4) + (issue_factor * 0.3) + (min_score_factor * 0.3)
        
        return max(0.1, min(1.0, confidence))
    
    def _determine_overall_quality(self, overall_score: float, expected_accuracy: float) -> QualityStatus:
        """전체 품질 상태 결정"""
        
        if overall_score >= expected_accuracy:
            return QualityStatus.EXCELLENT
        elif overall_score >= expected_accuracy * 0.97:
            return QualityStatus.GOOD
        elif overall_score >= expected_accuracy * 0.93:
            return QualityStatus.ACCEPTABLE
        elif overall_score >= expected_accuracy * 0.88:
            return QualityStatus.NEEDS_IMPROVEMENT
        else:
            return QualityStatus.POOR
    
    def _should_reanalyze(self, overall_score: float, expected_accuracy: float, 
                         quality_issues: List[QualityIssue]) -> bool:
        """재분석 필요 여부 판단"""
        
        # 목표 정확도에 크게 미달하는 경우
        if overall_score < expected_accuracy * 0.9:
            return True
        
        # 심각한 이슈가 있는 경우
        critical_issues = [issue for issue in quality_issues if issue.severity == "critical"]
        if critical_issues:
            return True
        
        # 주요 이슈가 많은 경우
        major_issues = [issue for issue in quality_issues if issue.severity == "major"]
        if len(major_issues) >= 3:
            return True
        
        return False
    
    def _generate_improvement_recommendations(self, 
                                            quality_issues: List[QualityIssue],
                                            metrics: QualityMetrics,
                                            category: JewelryCategory) -> List[str]:
        """개선 권장사항 생성"""
        
        recommendations = []
        
        # 이슈 기반 권장사항
        for issue in quality_issues:
            if issue.severity in ["critical", "major"]:
                recommendations.append(issue.suggestion)
        
        # 메트릭 기반 권장사항
        if metrics.accuracy_score < 0.9:
            recommendations.append("전문 용어와 업계 표준을 더 정확히 적용하세요")
        
        if metrics.completeness_score < 0.85:
            recommendations.append("분석의 완성도를 높이기 위해 누락된 요소들을 추가하세요")
        
        if metrics.professionalism_score < 0.8:
            recommendations.append("국제 감정 기준(GIA, SSEF 등)을 더 적극적으로 활용하세요")
        
        if metrics.clarity_score < 0.7:
            recommendations.append("내용을 더 명확하고 구조적으로 정리하세요")
        
        # 카테고리별 특화 권장사항
        if category == JewelryCategory.DIAMOND_4C:
            if metrics.completeness_score < 0.9:
                recommendations.append("4C(Carat, Color, Clarity, Cut) 모든 요소에 대한 상세 분석을 포함하세요")
        
        elif category == JewelryCategory.COLORED_GEMSTONE:
            if metrics.professionalism_score < 0.85:
                recommendations.append("원산지, 처리 여부, 감정 기관 정보를 포함한 전문적인 분석을 제공하세요")
        
        elif category == JewelryCategory.BUSINESS_INSIGHT:
            if metrics.actionability_score < 0.7:
                recommendations.append("실행 가능한 구체적인 전략과 단계별 실행 계획을 제시하세요")
        
        # 중복 제거 및 우선순위 정렬
        unique_recommendations = list(dict.fromkeys(recommendations))
        
        return unique_recommendations[:5]  # 최대 5개로 제한
    
    async def trigger_reanalysis(self, original_content: str, 
                               category: JewelryCategory,
                               validation_result: ValidationResult) -> Optional[str]:
        """자동 재분석 트리거"""
        
        if not self.llm_integration:
            logging.warning("LLM 통합이 비활성화되어 재분석 불가")
            return None
        
        # 개선된 프롬프트 생성
        improved_prompt = self._generate_improved_prompt(original_content, category, validation_result)
        
        try:
            # 하이브리드 LLM으로 재분석
            from core.hybrid_llm_manager_v23 import AnalysisRequest
            
            reanalysis_request = AnalysisRequest(
                content_type="text",
                data={"content": original_content},
                analysis_type=category.value,
                quality_threshold=0.995,  # 더 높은 품질 요구
                max_cost=0.08,
                language="ko"
            )
            
            hybrid_result = await self.llm_manager.analyze_with_hybrid_ai(reanalysis_request)
            
            # 재분석 결과 품질 검증
            revalidation_result = await self.validate_ai_response(
                hybrid_result.best_result.content,
                category,
                expected_accuracy=0.995
            )
            
            # 개선되었는지 확인
            if revalidation_result.metrics.overall_score > validation_result.metrics.overall_score:
                logging.info(f"✅ 재분석 성공 - 품질 개선: {validation_result.metrics.overall_score:.3f} → {revalidation_result.metrics.overall_score:.3f}")
                return hybrid_result.best_result.content
            else:
                logging.warning("⚠️ 재분석했으나 품질이 개선되지 않음")
                return None
                
        except Exception as e:
            logging.error(f"재분석 중 오류 발생: {e}")
            return None
    
    def _generate_improved_prompt(self, content: str, category: JewelryCategory,
                                validation_result: ValidationResult) -> str:
        """개선된 프롬프트 생성"""
        
        # 기존 이슈들을 해결하기 위한 추가 지침
        improvement_instructions = []
        
        for issue in validation_result.issues:
            if issue.severity in ["critical", "major"]:
                improvement_instructions.append(f"• {issue.suggestion}")
        
        for recommendation in validation_result.improvement_recommendations:
            improvement_instructions.append(f"• {recommendation}")
        
        improved_instruction = f"""
이전 분석에서 발견된 개선점들을 반영하여 다시 분석해주세요:

{chr(10).join(improvement_instructions)}

목표 정확도: 99.5% 이상
현재 점수: {validation_result.metrics.overall_score:.1%}

더욱 정확하고 전문적인 분석을 제공해주세요.
"""
        
        return improved_instruction
    
    def _update_performance_tracking(self, validation_result: ValidationResult, 
                                   category: JewelryCategory):
        """성능 추적 업데이트"""
        
        # 검증 히스토리에 추가
        self.validation_history.append({
            "timestamp": datetime.now(),
            "category": category.value,
            "overall_score": validation_result.metrics.overall_score,
            "quality_status": validation_result.overall_quality.value,
            "issue_count": len(validation_result.issues),
            "validation_time": validation_result.validation_time
        })
        
        # 카테고리별 성능 메트릭 업데이트
        category_key = category.value
        self.performance_metrics[category_key].append(validation_result.metrics.overall_score)
        
        # 최근 100개 기록만 유지
        if len(self.performance_metrics[category_key]) > 100:
            self.performance_metrics[category_key] = self.performance_metrics[category_key][-100:]
    
    async def _update_quality_patterns(self, content: str, category: JewelryCategory,
                                     validation_result: ValidationResult):
        """품질 패턴 학습 업데이트"""
        
        # 고품질 패턴 학습
        if validation_result.metrics.overall_score >= 0.95:
            pattern_key = f"{category.value}_high_quality"
            if pattern_key not in self.quality_patterns:
                self.quality_patterns[pattern_key] = []
            
            # 고품질 콘텐츠의 특성 추출
            structure = self.content_analyzer.analyze_content_structure(content)
            terminology = self.content_analyzer.analyze_jewelry_terminology(content, category)
            
            pattern_features = {
                "word_count": structure["word_count"],
                "sections": len(structure["sections"]),
                "readability": structure["readability_score"],
                "professional_depth": terminology.get("professional_depth", 0),
                "category_relevance": terminology.get("category_relevance", 0)
            }
            
            self.quality_patterns[pattern_key].append(pattern_features)
        
        # 개선 제안 패턴 학습
        if validation_result.improvement_recommendations:
            improvement_key = f"{category.value}_improvements"
            if improvement_key not in self.improvement_suggestions:
                self.improvement_suggestions[improvement_key] = defaultdict(int)
            
            for recommendation in validation_result.improvement_recommendations:
                self.improvement_suggestions[improvement_key][recommendation] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트 생성"""
        
        if not self.validation_history:
            return {"status": "데이터 없음"}
        
        recent_validations = list(self.validation_history)[-50:]  # 최근 50개
        
        # 전체 통계
        overall_scores = [v["overall_score"] for v in recent_validations]
        avg_score = statistics.mean(overall_scores)
        median_score = statistics.median(overall_scores)
        
        # 품질 상태 분포
        quality_distribution = defaultdict(int)
        for validation in recent_validations:
            quality_distribution[validation["quality_status"]] += 1
        
        # 카테고리별 성능
        category_performance = {}
        for category, scores in self.performance_metrics.items():
            if scores:
                category_performance[category] = {
                    "average_score": statistics.mean(scores),
                    "best_score": max(scores),
                    "recent_score": scores[-1] if scores else 0,
                    "total_validations": len(scores),
                    "trend": "improving" if len(scores) > 5 and scores[-5:] > scores[-10:-5] else "stable"
                }
        
        # 시간별 성능 트렌드
        validation_times = [v["validation_time"] for v in recent_validations]
        avg_validation_time = statistics.mean(validation_times) if validation_times else 0
        
        return {
            "overall_performance": {
                "average_score": avg_score,
                "median_score": median_score,
                "target_achievement_rate": len([s for s in overall_scores if s >= 0.992]) / len(overall_scores) * 100,
                "total_validations": len(self.validation_history)
            },
            "quality_distribution": dict(quality_distribution),
            "category_performance": category_performance,
            "system_performance": {
                "avg_validation_time": avg_validation_time,
                "llm_integration_status": self.llm_integration,
                "quality_patterns_learned": len(self.quality_patterns)
            },
            "improvement_insights": {
                "most_common_issues": self._get_most_common_issues(recent_validations),
                "success_factors": self._identify_success_factors()
            }
        }
    
    def _get_most_common_issues(self, validations: List[Dict[str, Any]]) -> List[str]:
        """가장 흔한 이슈들 추출"""
        
        # 실제 구현에서는 validation 기록에서 이슈들을 추출해야 함
        common_issues = [
            "전문 용어 사용 부족",
            "분석 완성도 미흡", 
            "구체적인 근거 부족",
            "실행 가능한 권장사항 부족"
        ]
        
        return common_issues[:3]
    
    def _identify_success_factors(self) -> List[str]:
        """성공 요인 식별"""
        
        success_factors = [
            "국제 감정 기준 정확한 적용",
            "체계적인 구조와 명확한 섹션 구분",
            "구체적인 수치와 데이터 제시",
            "실무진을 위한 실행 가능한 권장사항"
        ]
        
        return success_factors

# 테스트 및 데모 함수
async def demo_ai_quality_validator_v23():
    """AI 품질 검증기 v2.3 데모"""
    
    print("🔍 솔로몬드 AI 품질 검증기 v2.3 데모 시작")
    print("=" * 60)
    
    # 검증기 초기화
    validator = AIQualityValidatorV23()
    
    # 테스트 케이스 1: 고품질 다이아몬드 분석
    print("\n💎 테스트 1: 고품질 다이아몬드 분석 검증")
    print("-" * 50)
    
    high_quality_content = """
💎 다이아몬드 4C 전문 분석 보고서

📊 기본 정보
- 형태: 라운드 브릴리언트 컷
- 감정 기관: GIA
- 감정서 번호: 2171234567

📏 CARAT (캐럿)
- 중량: 1.52ct
- 등급: Large (1.0ct 이상)
- 시장 희소성: 중상

🎨 COLOR (컬러)
- GIA 등급: F
- 상세 설명: Near Colorless, 프리미엄 등급
- 시장 평가: 프리미엄

🔍 CLARITY (클래리티)
- GIA 등급: VVS1
- 내포물 위치: 크라운 부분에 미세한 크리스탈
- 아이클린: Yes

✨ CUT (컷)
- 컷 등급: Excellent
- 폴리시: Excellent
- 시메트리: Excellent
- 비율 분석: 테이블 57%, 깊이 61.5%

💰 시장 가치 평가
- 도매가: $18,000-20,000 (₩23,400,000-26,000,000)
- 소매가: $25,000-28,000 (₩32,500,000-36,400,000)
- 투자 전망: 안정적 상승

🎯 종합 평가
- 종합 등급: AAA
- 강점: 뛰어난 투명도와 완벽한 컷 품질
- 추천 용도: 프리미엄 약혼반지

📋 전문가 의견
GIA 감정서가 있는 고품질 다이아몬드로, 4C 모든 요소에서 우수한 등급을 보여줍니다.
"""
    
    result1 = await validator.validate_ai_response(
        high_quality_content,
        JewelryCategory.DIAMOND_4C,
        expected_accuracy=0.992
    )
    
    print(f"✅ 전체 품질: {result1.overall_quality.value}")
    print(f"📊 종합 점수: {result1.metrics.overall_score:.3f}")
    print(f"🎯 정확성: {result1.metrics.accuracy_score:.3f}")
    print(f"📋 완성도: {result1.metrics.completeness_score:.3f}")
    print(f"💡 전문성: {result1.metrics.professionalism_score:.3f}")
    print(f"⚠️ 발견된 이슈: {len(result1.issues)}개")
    print(f"🔄 재분석 필요: {'예' if result1.reanalysis_required else '아니오'}")
    
    # 테스트 케이스 2: 낮은 품질 분석
    print("\n\n🔴 테스트 2: 낮은 품질 분석 검증")
    print("-" * 50)
    
    low_quality_content = """
다이아몬드는 좋은 보석입니다. 이 다이아몬드는 크기가 좀 크고 색깔도 괜찮습니다. 
투명도도 나쁘지 않고 컷도 잘 되어 있는 것 같습니다. 
가격은 대략 비쌀 것 같고 투자 가치도 있을 것입니다.
추천합니다.
"""
    
    result2 = await validator.validate_ai_response(
        low_quality_content,
        JewelryCategory.DIAMOND_4C,
        expected_accuracy=0.992
    )
    
    print(f"❌ 전체 품질: {result2.overall_quality.value}")
    print(f"📊 종합 점수: {result2.metrics.overall_score:.3f}")
    print(f"⚠️ 발견된 이슈: {len(result2.issues)}개")
    print(f"🔄 재분석 필요: {'예' if result2.reanalysis_required else '아니오'}")
    
    if result2.improvement_recommendations:
        print("💡 개선 권장사항:")
        for i, rec in enumerate(result2.improvement_recommendations[:3], 1):
            print(f"   {i}. {rec}")
    
    # 테스트 케이스 3: 재분석 테스트
    if result2.reanalysis_required and validator.llm_integration:
        print("\n\n🔄 테스트 3: 자동 재분석")
        print("-" * 50)
        
        reanalyzed_content = await validator.trigger_reanalysis(
            low_quality_content,
            JewelryCategory.DIAMOND_4C,
            result2
        )
        
        if reanalyzed_content:
            print("✅ 재분석 성공")
            print(f"📝 재분석 내용 (일부): {reanalyzed_content[:200]}...")
        else:
            print("⚠️ 재분석 실패 또는 개선되지 않음")
    
    # 성능 리포트
    print("\n\n📈 성능 리포트")
    print("-" * 50)
    performance = validator.get_performance_report()
    
    if "overall_performance" in performance:
        overall = performance["overall_performance"]
        print(f"평균 점수: {overall['average_score']:.3f}")
        print(f"목표 달성률: {overall['target_achievement_rate']:.1f}%")
        print(f"총 검증 수: {overall['total_validations']}")
    
    print("\n🎯 AI 품질 검증기 v2.3 데모 완료!")
    print("🏆 99.2% 정확도 달성을 위한 품질 관리 시스템 구축 완료!")

if __name__ == "__main__":
    asyncio.run(demo_ai_quality_validator_v23())
