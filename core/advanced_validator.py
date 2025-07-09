# Phase 2 Week 3 Day 3: 고급 크로스 검증 시스템
# AI 기반 지능형 검증 + 이상치 감지 + 품질 점수 정교화

import asyncio
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import math
from collections import Counter, defaultdict
import time
from pathlib import Path
import logging

# 텍스트 유사도 분석을 위한 간단한 구현 (실제로는 transformers 사용 권장)
from difflib import SequenceMatcher
import hashlib

class ValidationLevel(Enum):
    """검증 레벨"""
    BASIC = "basic"           # 기본 검증
    ADVANCED = "advanced"     # 고급 검증  
    AI_POWERED = "ai_powered" # AI 기반 검증
    COMPREHENSIVE = "comprehensive"  # 종합 검증

class AnomalyType(Enum):
    """이상치 유형"""
    CONTENT_MISMATCH = "content_mismatch"      # 내용 불일치
    QUALITY_DEGRADATION = "quality_degradation" # 품질 저하
    PROCESSING_ERROR = "processing_error"       # 처리 오류
    STATISTICAL_OUTLIER = "statistical_outlier" # 통계적 이상치
    SEMANTIC_INCONSISTENCY = "semantic_inconsistency" # 의미적 비일관성

@dataclass
class ValidationMetrics:
    """검증 지표"""
    content_similarity: float = 0.0      # 내용 유사도 (0-1)
    semantic_coherence: float = 0.0      # 의미적 일관성 (0-1)
    statistical_consistency: float = 0.0  # 통계적 일관성 (0-1)
    quality_score: float = 0.0           # 품질 점수 (0-1)
    confidence_level: float = 0.0        # 신뢰도 (0-1)
    anomaly_score: float = 0.0           # 이상치 점수 (0-1, 낮을수록 정상)

@dataclass
class CrossValidationItem:
    """크로스 검증 항목"""
    item_id: str
    content: str
    metadata: Dict[str, Any]
    processing_quality: float
    source_reliability: float
    extracted_features: Dict[str, Any] = field(default_factory=dict)
    validation_metrics: ValidationMetrics = field(default_factory=ValidationMetrics)

@dataclass
class AnomalyDetection:
    """이상치 탐지 결과"""
    anomaly_type: AnomalyType
    severity: float  # 0-1
    description: str
    affected_items: List[str]
    suggested_action: str
    confidence: float

@dataclass
class ValidationResult:
    """검증 결과"""
    overall_score: float
    individual_scores: Dict[str, float]
    anomalies: List[AnomalyDetection]
    recommendations: List[str]
    validation_level: ValidationLevel
    processing_time: float

class JewelrySemanticAnalyzer:
    """주얼리 특화 의미 분석기"""
    
    def __init__(self):
        # 주얼리 도메인 특화 용어 가중치
        self.jewelry_terms_weights = {
            # 핵심 용어 (높은 가중치)
            "다이아몬드": 3.0, "diamond": 3.0,
            "루비": 2.5, "ruby": 2.5,
            "사파이어": 2.5, "sapphire": 2.5,
            "에메랄드": 2.5, "emerald": 2.5,
            
            # 품질 용어
            "4C": 2.8, "캐럿": 2.5, "carat": 2.5,
            "컬러": 2.3, "color": 2.3,
            "클래리티": 2.3, "clarity": 2.3,
            "컷": 2.3, "cut": 2.3,
            
            # 인증 용어
            "GIA": 2.8, "AGS": 2.0, "SSEF": 2.0,
            "감정서": 2.5, "certificate": 2.5,
            "인증": 2.0, "certification": 2.0,
            
            # 가격 용어
            "도매가": 2.2, "wholesale": 2.2,
            "소매가": 2.0, "retail": 2.0,
            "시중가": 1.8, "market": 1.8,
            
            # 기술 용어
            "브릴리언트": 1.8, "brilliant": 1.8,
            "프린세스": 1.8, "princess": 1.8,
            "세팅": 1.5, "setting": 1.5
        }
        
        # 의미 그룹 (유사한 의미를 가진 용어들)
        self.semantic_groups = {
            "precious_stones": ["다이아몬드", "루비", "사파이어", "에메랄드", "diamond", "ruby", "sapphire", "emerald"],
            "quality_grades": ["4C", "캐럿", "컬러", "클래리티", "컷", "carat", "color", "clarity", "cut"],
            "certifications": ["GIA", "AGS", "SSEF", "감정서", "인증", "certificate", "certification"],
            "pricing": ["도매가", "소매가", "시중가", "wholesale", "retail", "market"],
            "cuts": ["브릴리언트", "프린세스", "라운드", "brilliant", "princess", "round"]
        }
        
        # 패턴 인식용 정규표현식
        self.price_patterns = [
            r'\$[\d,]+',           # $1,000
            r'[\d,]+달러',          # 1000달러
            r'[\d,]+원',           # 1000원
            r'캐럿당\s*[\d,]+',      # 캐럿당 1000
        ]
        
        self.grade_patterns = [
            r'[A-Z]\s*등급',       # D등급
            r'[DEFGHIJK][12]?',    # D, E, F1, VS1 등
            r'FL|IF|VVS[12]|VS[12]|SI[12]|I[123]',  # 클래리티 등급
        ]
    
    def extract_semantic_features(self, text: str) -> Dict[str, Any]:
        """의미적 특징 추출"""
        features = {
            "jewelry_terms": [],
            "prices": [],
            "grades": [],
            "semantic_density": 0.0,
            "domain_relevance": 0.0,
            "technical_depth": 0.0
        }
        
        text_lower = text.lower()
        words = text.split()
        
        # 주얼리 용어 추출
        total_weight = 0.0
        for term, weight in self.jewelry_terms_weights.items():
            if term.lower() in text_lower:
                features["jewelry_terms"].append({
                    "term": term,
                    "weight": weight,
                    "occurrences": text_lower.count(term.lower())
                })
                total_weight += weight
        
        # 가격 정보 추출
        for pattern in self.price_patterns:
            matches = re.findall(pattern, text)
            features["prices"].extend(matches)
        
        # 등급 정보 추출
        for pattern in self.grade_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            features["grades"].extend(matches)
        
        # 의미 밀도 계산 (주얼리 용어 밀도)
        if len(words) > 0:
            features["semantic_density"] = len(features["jewelry_terms"]) / len(words)
            features["domain_relevance"] = min(1.0, total_weight / 10.0)  # 정규화
        
        # 기술적 깊이 (전문 용어 복잡도)
        tech_terms = ["4C", "GIA", "클래리티", "브릴리언트", "세팅"]
        tech_count = sum(1 for term in tech_terms if term.lower() in text_lower)
        features["technical_depth"] = min(1.0, tech_count / 5.0)
        
        return features
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """의미적 유사도 계산"""
        features1 = self.extract_semantic_features(text1)
        features2 = self.extract_semantic_features(text2)
        
        # 용어 기반 유사도
        terms1 = set(item["term"] for item in features1["jewelry_terms"])
        terms2 = set(item["term"] for item in features2["jewelry_terms"])
        
        if not terms1 and not terms2:
            term_similarity = 1.0
        elif not terms1 or not terms2:
            term_similarity = 0.0
        else:
            intersection = len(terms1.intersection(terms2))
            union = len(terms1.union(terms2))
            term_similarity = intersection / union if union > 0 else 0.0
        
        # 의미 그룹 기반 유사도
        group_similarity = self._calculate_group_similarity(features1, features2)
        
        # 가격/등급 정보 유사도
        price_similarity = self._calculate_info_similarity(
            features1["prices"], features2["prices"]
        )
        grade_similarity = self._calculate_info_similarity(
            features1["grades"], features2["grades"]
        )
        
        # 가중 평균
        weights = [0.4, 0.3, 0.15, 0.15]  # 용어, 그룹, 가격, 등급
        similarities = [term_similarity, group_similarity, price_similarity, grade_similarity]
        
        return sum(w * s for w, s in zip(weights, similarities))
    
    def _calculate_group_similarity(self, features1: Dict, features2: Dict) -> float:
        """의미 그룹 기반 유사도"""
        group_scores = []
        
        for group_name, group_terms in self.semantic_groups.items():
            # 각 그룹에서 발견된 용어 수
            count1 = sum(1 for item in features1["jewelry_terms"] 
                        if item["term"] in group_terms)
            count2 = sum(1 for item in features2["jewelry_terms"] 
                        if item["term"] in group_terms)
            
            # 그룹별 유사도 (둘 다 용어가 있거나 둘 다 없으면 유사)
            if count1 > 0 and count2 > 0:
                group_score = 1.0
            elif count1 == 0 and count2 == 0:
                group_score = 0.5  # 중립
            else:
                group_score = 0.0
            
            group_scores.append(group_score)
        
        return np.mean(group_scores) if group_scores else 0.0
    
    def _calculate_info_similarity(self, info1: List, info2: List) -> float:
        """정보 유사도 계산"""
        if not info1 and not info2:
            return 1.0
        if not info1 or not info2:
            return 0.0
        
        # 집합 기반 유사도
        set1, set2 = set(info1), set(info2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0

class AnomalyDetector:
    """이상치 탐지기"""
    
    def __init__(self, sensitivity: float = 0.7):
        self.sensitivity = sensitivity  # 민감도 (0-1, 높을수록 민감)
        self.statistical_thresholds = {
            "content_length_ratio": 3.0,      # 내용 길이 비율 임계값
            "quality_deviation": 0.3,         # 품질 편차 임계값
            "similarity_threshold": 0.3,      # 유사도 임계값
            "consistency_threshold": 0.5      # 일관성 임계값
        }
    
    def detect_anomalies(self, validation_items: List[CrossValidationItem]) -> List[AnomalyDetection]:
        """이상치 탐지"""
        anomalies = []
        
        if len(validation_items) < 2:
            return anomalies
        
        # 1. 내용 불일치 탐지
        content_anomalies = self._detect_content_mismatches(validation_items)
        anomalies.extend(content_anomalies)
        
        # 2. 품질 저하 탐지
        quality_anomalies = self._detect_quality_degradation(validation_items)
        anomalies.extend(quality_anomalies)
        
        # 3. 통계적 이상치 탐지
        statistical_anomalies = self._detect_statistical_outliers(validation_items)
        anomalies.extend(statistical_anomalies)
        
        # 4. 의미적 비일관성 탐지
        semantic_anomalies = self._detect_semantic_inconsistencies(validation_items)
        anomalies.extend(semantic_anomalies)
        
        # 심각도별 정렬
        anomalies.sort(key=lambda x: x.severity, reverse=True)
        
        return anomalies
    
    def _detect_content_mismatches(self, items: List[CrossValidationItem]) -> List[AnomalyDetection]:
        """내용 불일치 탐지"""
        anomalies = []
        
        # 내용 길이 비교
        lengths = [len(item.content) for item in items]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        for i, item in enumerate(items):
            if std_length > 0:
                z_score = abs(lengths[i] - mean_length) / std_length
                if z_score > self.statistical_thresholds["content_length_ratio"]:
                    severity = min(1.0, z_score / 5.0)
                    
                    anomalies.append(AnomalyDetection(
                        anomaly_type=AnomalyType.CONTENT_MISMATCH,
                        severity=severity,
                        description=f"내용 길이 이상: 평균 대비 {z_score:.1f} 표준편차",
                        affected_items=[item.item_id],
                        suggested_action="내용 완전성 재검토 필요",
                        confidence=0.8
                    ))
        
        return anomalies
    
    def _detect_quality_degradation(self, items: List[CrossValidationItem]) -> List[AnomalyDetection]:
        """품질 저하 탐지"""
        anomalies = []
        
        qualities = [item.processing_quality for item in items]
        mean_quality = np.mean(qualities)
        
        low_quality_items = []
        for item in items:
            if item.processing_quality < mean_quality - self.statistical_thresholds["quality_deviation"]:
                low_quality_items.append(item.item_id)
        
        if low_quality_items:
            severity = len(low_quality_items) / len(items)
            
            anomalies.append(AnomalyDetection(
                anomaly_type=AnomalyType.QUALITY_DEGRADATION,
                severity=severity,
                description=f"{len(low_quality_items)}개 항목에서 품질 저하 감지",
                affected_items=low_quality_items,
                suggested_action="품질이 낮은 항목 재처리 검토",
                confidence=0.9
            ))
        
        return anomalies
    
    def _detect_statistical_outliers(self, items: List[CrossValidationItem]) -> List[AnomalyDetection]:
        """통계적 이상치 탐지"""
        anomalies = []
        
        # 신뢰도 기반 이상치
        reliabilities = [item.source_reliability for item in items]
        
        if len(reliabilities) > 2:
            q1 = np.percentile(reliabilities, 25)
            q3 = np.percentile(reliabilities, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_items = []
            for item in items:
                if item.source_reliability < lower_bound or item.source_reliability > upper_bound:
                    outlier_items.append(item.item_id)
            
            if outlier_items:
                severity = len(outlier_items) / len(items)
                
                anomalies.append(AnomalyDetection(
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=severity,
                    description=f"신뢰도 통계적 이상치: {len(outlier_items)}개 항목",
                    affected_items=outlier_items,
                    suggested_action="이상치 항목의 처리 과정 재검토",
                    confidence=0.7
                ))
        
        return anomalies
    
    def _detect_semantic_inconsistencies(self, items: List[CrossValidationItem]) -> List[AnomalyDetection]:
        """의미적 비일관성 탐지"""
        anomalies = []
        
        semantic_analyzer = JewelrySemanticAnalyzer()
        
        # 항목 간 의미적 유사도 매트릭스 계산
        n_items = len(items)
        similarity_matrix = np.zeros((n_items, n_items))
        
        for i in range(n_items):
            for j in range(i + 1, n_items):
                similarity = semantic_analyzer.calculate_semantic_similarity(
                    items[i].content, items[j].content
                )
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        # 평균 유사도가 낮은 항목 탐지
        avg_similarities = np.mean(similarity_matrix, axis=1)
        threshold = self.statistical_thresholds["similarity_threshold"]
        
        inconsistent_items = []
        for i, avg_sim in enumerate(avg_similarities):
            if avg_sim < threshold:
                inconsistent_items.append(items[i].item_id)
        
        if inconsistent_items:
            severity = len(inconsistent_items) / len(items)
            
            anomalies.append(AnomalyDetection(
                anomaly_type=AnomalyType.SEMANTIC_INCONSISTENCY,
                severity=severity,
                description=f"의미적 비일관성: {len(inconsistent_items)}개 항목",
                affected_items=inconsistent_items,
                suggested_action="의미적으로 일관성이 없는 항목 검토",
                confidence=0.6
            ))
        
        return anomalies

class AdvancedCrossValidator:
    """고급 크로스 검증기"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE):
        self.validation_level = validation_level
        self.semantic_analyzer = JewelrySemanticAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.logger = logging.getLogger(__name__)
        
        # 검증 가중치 (검증 레벨에 따라 조정)
        self.validation_weights = self._get_validation_weights()
    
    def _get_validation_weights(self) -> Dict[str, float]:
        """검증 레벨에 따른 가중치 반환"""
        if self.validation_level == ValidationLevel.BASIC:
            return {
                "content_similarity": 0.6,
                "statistical_consistency": 0.4,
                "semantic_coherence": 0.0,
                "anomaly_penalty": 0.1
            }
        elif self.validation_level == ValidationLevel.ADVANCED:
            return {
                "content_similarity": 0.4,
                "statistical_consistency": 0.3,
                "semantic_coherence": 0.3,
                "anomaly_penalty": 0.2
            }
        else:  # AI_POWERED, COMPREHENSIVE
            return {
                "content_similarity": 0.3,
                "statistical_consistency": 0.25,
                "semantic_coherence": 0.35,
                "anomaly_penalty": 0.3
            }
    
    async def validate_cross_consistency(
        self, 
        items: List[Dict[str, Any]], 
        context: Dict[str, Any] = None
    ) -> ValidationResult:
        """크로스 일관성 검증"""
        
        start_time = time.time()
        
        # 검증 항목 준비
        validation_items = self._prepare_validation_items(items)
        
        # 각 항목별 지표 계산
        await self._calculate_individual_metrics(validation_items)
        
        # 크로스 검증 수행
        cross_validation_score = await self._perform_cross_validation(validation_items)
        
        # 이상치 탐지
        anomalies = self.anomaly_detector.detect_anomalies(validation_items)
        
        # 전체 점수 계산
        overall_score = self._calculate_overall_score(validation_items, anomalies)
        
        # 개별 점수 추출
        individual_scores = {
            item.item_id: item.validation_metrics.quality_score 
            for item in validation_items
        }
        
        # 추천사항 생성
        recommendations = self._generate_recommendations(validation_items, anomalies)
        
        processing_time = time.time() - start_time
        
        return ValidationResult(
            overall_score=overall_score,
            individual_scores=individual_scores,
            anomalies=anomalies,
            recommendations=recommendations,
            validation_level=self.validation_level,
            processing_time=processing_time
        )
    
    def _prepare_validation_items(self, items: List[Dict[str, Any]]) -> List[CrossValidationItem]:
        """검증 항목 준비"""
        validation_items = []
        
        for i, item in enumerate(items):
            validation_item = CrossValidationItem(
                item_id=item.get("id", f"item_{i}"),
                content=item.get("content", ""),
                metadata=item.get("metadata", {}),
                processing_quality=item.get("quality", 0.8),
                source_reliability=item.get("reliability", 0.8)
            )
            
            # 의미적 특징 추출
            validation_item.extracted_features = self.semantic_analyzer.extract_semantic_features(
                validation_item.content
            )
            
            validation_items.append(validation_item)
        
        return validation_items
    
    async def _calculate_individual_metrics(self, items: List[CrossValidationItem]):
        """개별 항목 지표 계산"""
        
        for item in items:
            metrics = ValidationMetrics()
            
            # 도메인 관련성 점수
            domain_relevance = item.extracted_features.get("domain_relevance", 0.0)
            
            # 기술적 깊이 점수
            technical_depth = item.extracted_features.get("technical_depth", 0.0)
            
            # 의미 밀도 점수
            semantic_density = item.extracted_features.get("semantic_density", 0.0)
            
            # 품질 점수 계산 (여러 요소 결합)
            quality_components = [
                item.processing_quality,
                item.source_reliability,
                domain_relevance,
                technical_depth * 0.5,  # 기술 깊이는 낮은 가중치
                semantic_density * 0.3   # 의미 밀도도 낮은 가중치
            ]
            
            weights = [0.4, 0.3, 0.2, 0.05, 0.05]
            metrics.quality_score = sum(w * c for w, c in zip(weights, quality_components))
            
            # 기본 신뢰도 설정
            metrics.confidence_level = item.source_reliability
            
            item.validation_metrics = metrics
    
    async def _perform_cross_validation(self, items: List[CrossValidationItem]) -> float:
        """크로스 검증 수행"""
        
        if len(items) < 2:
            return 1.0
        
        n_items = len(items)
        similarity_scores = []
        consistency_scores = []
        
        # 모든 쌍에 대해 유사도 계산
        for i in range(n_items):
            for j in range(i + 1, n_items):
                # 내용 유사도
                content_sim = self.semantic_analyzer.calculate_semantic_similarity(
                    items[i].content, items[j].content
                )
                
                # 통계적 일관성 (품질, 신뢰도)
                quality_diff = abs(items[i].processing_quality - items[j].processing_quality)
                reliability_diff = abs(items[i].source_reliability - items[j].source_reliability)
                
                stat_consistency = 1.0 - (quality_diff + reliability_diff) / 2.0
                
                similarity_scores.append(content_sim)
                consistency_scores.append(stat_consistency)
                
                # 개별 항목에 지표 업데이트
                items[i].validation_metrics.content_similarity = max(
                    items[i].validation_metrics.content_similarity, content_sim
                )
                items[j].validation_metrics.content_similarity = max(
                    items[j].validation_metrics.content_similarity, content_sim
                )
                
                items[i].validation_metrics.statistical_consistency = max(
                    items[i].validation_metrics.statistical_consistency, stat_consistency
                )
                items[j].validation_metrics.statistical_consistency = max(
                    items[j].validation_metrics.statistical_consistency, stat_consistency
                )
        
        # 전체 크로스 검증 점수
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        cross_score = (avg_similarity + avg_consistency) / 2.0
        
        # 의미적 일관성 업데이트
        for item in items:
            item.validation_metrics.semantic_coherence = avg_similarity
        
        return cross_score
    
    def _calculate_overall_score(
        self, 
        items: List[CrossValidationItem], 
        anomalies: List[AnomalyDetection]
    ) -> float:
        """전체 점수 계산"""
        
        # 개별 점수들의 평균
        individual_avg = np.mean([item.validation_metrics.quality_score for item in items])
        
        # 크로스 검증 점수
        similarity_avg = np.mean([item.validation_metrics.content_similarity for item in items])
        consistency_avg = np.mean([item.validation_metrics.statistical_consistency for item in items])
        coherence_avg = np.mean([item.validation_metrics.semantic_coherence for item in items])
        
        # 이상치 페널티 계산
        anomaly_penalty = 0.0
        if anomalies:
            total_severity = sum(anomaly.severity for anomaly in anomalies)
            anomaly_penalty = min(0.5, total_severity / len(items))  # 최대 50% 페널티
        
        # 가중 점수 계산
        weights = self.validation_weights
        
        weighted_score = (
            weights["content_similarity"] * similarity_avg +
            weights["statistical_consistency"] * consistency_avg +
            weights["semantic_coherence"] * coherence_avg
        )
        
        # 이상치 페널티 적용
        final_score = weighted_score * (1.0 - weights["anomaly_penalty"] * anomaly_penalty)
        
        return max(0.0, min(1.0, final_score))
    
    def _generate_recommendations(
        self, 
        items: List[CrossValidationItem], 
        anomalies: List[AnomalyDetection]
    ) -> List[str]:
        """추천사항 생성"""
        
        recommendations = []
        
        # 품질 기반 추천
        low_quality_count = sum(1 for item in items if item.validation_metrics.quality_score < 0.7)
        if low_quality_count > 0:
            recommendations.append(
                f"{low_quality_count}개 항목의 품질이 낮습니다. 재처리를 검토하세요."
            )
        
        # 유사도 기반 추천
        low_similarity_count = sum(1 for item in items if item.validation_metrics.content_similarity < 0.5)
        if low_similarity_count > 0:
            recommendations.append(
                f"{low_similarity_count}개 항목의 내용 일치도가 낮습니다. 원본 데이터를 확인하세요."
            )
        
        # 이상치 기반 추천
        critical_anomalies = [a for a in anomalies if a.severity > 0.7]
        if critical_anomalies:
            recommendations.append(
                f"{len(critical_anomalies)}개의 심각한 이상치가 발견되었습니다. 즉시 검토가 필요합니다."
            )
        
        # 주얼리 특화 추천
        jewelry_terms_avg = np.mean([
            len(item.extracted_features.get("jewelry_terms", [])) 
            for item in items
        ])
        
        if jewelry_terms_avg < 2:
            recommendations.append(
                "주얼리 전문 용어가 부족합니다. 내용의 관련성을 확인하세요."
            )
        
        # 기본 추천사항
        if not recommendations:
            if len(items) >= 3:
                recommendations.append("모든 항목이 양호한 품질을 보입니다.")
            else:
                recommendations.append("추가 검증을 위해 더 많은 데이터를 확보하는 것을 권장합니다.")
        
        return recommendations

# 사용 예시
async def demo_advanced_validation():
    """고급 크로스 검증 데모"""
    
    print("🔍 고급 크로스 검증 시스템 데모")
    print("=" * 60)
    
    # 테스트 데이터 (주얼리 관련 내용)
    test_items = [
        {
            "id": "item_1",
            "content": "이 다이아몬드는 1.5캐럿 D컬러 FL 등급으로 GIA 감정서가 있습니다. 라운드 브릴리언트 컷으로 완벽한 대칭성을 보여줍니다.",
            "quality": 0.95,
            "reliability": 0.9,
            "metadata": {"source": "main_recording", "duration": 120}
        },
        {
            "id": "item_2", 
            "content": "1.5ct D컬러 Flawless 다이아몬드입니다. GIA 인증을 받았으며 라운드 컷으로 처리되었습니다. 매우 높은 품질의 보석입니다.",
            "quality": 0.92,
            "reliability": 0.88,
            "metadata": {"source": "backup_recording", "duration": 115}
        },
        {
            "id": "item_3",
            "content": "이 루비는 2캐럿 크기로 미얀마산입니다. 비둘기피 색상이 매우 인상적이며 히트 트리트먼트가 적용되었습니다.",
            "quality": 0.85,
            "reliability": 0.85,
            "metadata": {"source": "document", "pages": 2}
        },
        {
            "id": "item_4",
            "content": "날씨가 좋네요. 오늘 점심은 뭘 먹을까요? 회의가 늦게 끝날 것 같습니다.",
            "quality": 0.6,
            "reliability": 0.7,
            "metadata": {"source": "noise_data", "duration": 30}
        }
    ]
    
    # 다양한 검증 레벨 테스트
    validation_levels = [
        ValidationLevel.BASIC,
        ValidationLevel.ADVANCED, 
        ValidationLevel.COMPREHENSIVE
    ]
    
    for level in validation_levels:
        print(f"\n🔍 검증 레벨: {level.value.upper()}")
        print("-" * 40)
        
        validator = AdvancedCrossValidator(validation_level=level)
        
        # 검증 실행
        result = await validator.validate_cross_consistency(test_items)
        
        print(f"📊 전체 점수: {result.overall_score:.3f}")
        print(f"⏱️ 처리 시간: {result.processing_time:.2f}초")
        
        # 개별 점수
        print(f"\n📋 개별 점수:")
        for item_id, score in result.individual_scores.items():
            print(f"   {item_id}: {score:.3f}")
        
        # 이상치 탐지 결과
        if result.anomalies:
            print(f"\n⚠️ 이상치 탐지 ({len(result.anomalies)}개):")
            for anomaly in result.anomalies[:3]:  # 상위 3개만 표시
                print(f"   - {anomaly.description} (심각도: {anomaly.severity:.2f})")
        
        # 추천사항
        print(f"\n💡 추천사항:")
        for i, rec in enumerate(result.recommendations[:3], 1):
            print(f"   {i}. {rec}")
    
    print("\n" + "=" * 60)
    print("✅ 고급 크로스 검증 시스템 데모 완료!")

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 데모 실행
    asyncio.run(demo_advanced_validation())
