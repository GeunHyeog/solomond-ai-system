#!/usr/bin/env python3
"""
Quality Validation System - 실시간 품질 모니터링
분석 결과의 정확성을 실시간으로 검증하고 개선하는 시스템
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
import requests
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """품질 지표 데이터 클래스"""
    accuracy_score: float
    completeness_score: float
    relevance_score: float
    clarity_score: float
    actionability_score: float
    overall_score: float
    timestamp: str

class QualityValidator:
    """분석 품질 검증 시스템"""
    
    def __init__(self):
        self.validation_history = []
        self.quality_thresholds = {
            "accuracy": 0.7,
            "completeness": 0.6,
            "relevance": 0.8,
            "clarity": 0.7,
            "actionability": 0.6,
            "overall": 0.7
        }
        
        # 품질 평가 키워드
        self.quality_keywords = {
            "accuracy": {
                "positive": ["정확한", "구체적인", "근거가", "사실", "확실한", "명확한"],
                "negative": ["추측", "아마도", "~것 같다", "불분명", "모호한"]
            },
            "completeness": {
                "sections": ["요약", "분석", "결론", "제안", "배경"],
                "min_length": 200
            },
            "relevance": {
                "required": ["핵심", "중요", "주요", "결과"],
                "context_match": ["관련", "연관", "맥락"]
            },
            "clarity": {
                "structure": ["1.", "2.", "3.", "가.", "나.", "다.", "●", "•"],
                "connectors": ["따라서", "그러므로", "또한", "반면", "결과적으로"]
            },
            "actionability": {
                "action_words": ["제안", "추천", "개선", "실행", "적용", "구현", "수행"]
            }
        }
    
    def validate_accuracy(self, analysis_result: str, original_content: str) -> float:
        """정확성 점수 계산"""
        if not analysis_result or not original_content:
            return 0.0
        
        score = 0.5  # 기본점수
        
        # 긍정적 정확성 지표
        positive_indicators = self.quality_keywords["accuracy"]["positive"]
        positive_count = sum(1 for word in positive_indicators 
                           if word in analysis_result.lower())
        score += min(positive_count * 0.1, 0.3)
        
        # 부정적 정확성 지표 (감점)
        negative_indicators = self.quality_keywords["accuracy"]["negative"]
        negative_count = sum(1 for word in negative_indicators 
                           if word in analysis_result.lower())
        score -= min(negative_count * 0.15, 0.4)
        
        # 원본 내용과의 키워드 일치도
        original_keywords = set(re.findall(r'\b\w{4,}\b', original_content.lower()))
        analysis_keywords = set(re.findall(r'\b\w{4,}\b', analysis_result.lower()))
        
        if original_keywords:
            keyword_overlap = len(original_keywords & analysis_keywords) / len(original_keywords)
            score += keyword_overlap * 0.2
        
        return max(0.0, min(1.0, score))
    
    def validate_completeness(self, analysis_result: str) -> float:
        """완전성 점수 계산"""
        if not analysis_result:
            return 0.0
        
        score = 0.0
        
        # 최소 길이 체크
        min_length = self.quality_keywords["completeness"]["min_length"]
        if len(analysis_result) >= min_length:
            score += 0.3
        elif len(analysis_result) >= min_length * 0.7:
            score += 0.2
        elif len(analysis_result) >= min_length * 0.5:
            score += 0.1
        
        # 필수 섹션 체크
        required_sections = self.quality_keywords["completeness"]["sections"]
        section_count = sum(1 for section in required_sections 
                          if section in analysis_result.lower())
        score += min(section_count / len(required_sections), 1.0) * 0.4
        
        # 구조화 점수
        has_structure = any(marker in analysis_result 
                          for marker in ["##", "===", "***", "1.", "2.", "3."])
        if has_structure:
            score += 0.3
        
        return min(1.0, score)
    
    def validate_relevance(self, analysis_result: str, context: str = "") -> float:
        """관련성 점수 계산"""
        if not analysis_result:
            return 0.0
        
        score = 0.5  # 기본점수
        
        # 필수 키워드 체크
        required_keywords = self.quality_keywords["relevance"]["required"]
        keyword_count = sum(1 for word in required_keywords 
                          if word in analysis_result.lower())
        score += min(keyword_count * 0.15, 0.3)
        
        # 컨텍스트 매칭 (제공된 경우)
        if context:
            context_keywords = self.quality_keywords["relevance"]["context_match"]
            context_match = sum(1 for word in context_keywords 
                              if word in analysis_result.lower())
            score += min(context_match * 0.1, 0.2)
        
        return min(1.0, score)
    
    def validate_clarity(self, analysis_result: str) -> float:
        """명확성 점수 계산"""
        if not analysis_result:
            return 0.0
        
        score = 0.3  # 기본점수
        
        # 구조화 지표
        structure_indicators = self.quality_keywords["clarity"]["structure"]
        structure_count = sum(1 for marker in structure_indicators 
                            if marker in analysis_result)
        score += min(structure_count * 0.1, 0.3)
        
        # 연결어 사용
        connectors = self.quality_keywords["clarity"]["connectors"]
        connector_count = sum(1 for conn in connectors 
                            if conn in analysis_result)
        score += min(connector_count * 0.08, 0.2)
        
        # 문장 길이 적정성 (너무 길거나 짧지 않은지)
        sentences = re.split(r'[.!?]+', analysis_result)
        if sentences:
            avg_length = np.mean([len(s.strip()) for s in sentences if s.strip()])
            if 20 <= avg_length <= 100:  # 적정 문장 길이
                score += 0.2
        
        return min(1.0, score)
    
    def validate_actionability(self, analysis_result: str) -> float:
        """실행가능성 점수 계산"""
        if not analysis_result:
            return 0.0
        
        score = 0.2  # 기본점수
        
        # 실행 관련 키워드
        action_words = self.quality_keywords["actionability"]["action_words"]
        action_count = sum(1 for word in action_words 
                         if word in analysis_result.lower())
        score += min(action_count * 0.15, 0.5)
        
        # 구체적인 제안 섹션 존재 여부
        suggestion_patterns = [
            r'제안.*:',
            r'추천.*:',
            r'개선.*방안',
            r'액션.*플랜',
            r'다음.*단계'
        ]
        
        has_suggestions = any(re.search(pattern, analysis_result.lower()) 
                            for pattern in suggestion_patterns)
        if has_suggestions:
            score += 0.3
        
        return min(1.0, score)
    
    def calculate_quality_metrics(self, analysis_result: str, original_content: str = "", context: str = "") -> QualityMetrics:
        """종합 품질 지표 계산"""
        
        accuracy = self.validate_accuracy(analysis_result, original_content)
        completeness = self.validate_completeness(analysis_result)
        relevance = self.validate_relevance(analysis_result, context)
        clarity = self.validate_clarity(analysis_result)
        actionability = self.validate_actionability(analysis_result)
        
        # 가중 평균으로 전체 점수 계산
        weights = {
            "accuracy": 0.25,
            "completeness": 0.20,
            "relevance": 0.25,
            "clarity": 0.15,
            "actionability": 0.15
        }
        
        overall_score = (
            accuracy * weights["accuracy"] +
            completeness * weights["completeness"] +
            relevance * weights["relevance"] +
            clarity * weights["clarity"] +
            actionability * weights["actionability"]
        )
        
        return QualityMetrics(
            accuracy_score=accuracy,
            completeness_score=completeness,
            relevance_score=relevance,
            clarity_score=clarity,
            actionability_score=actionability,
            overall_score=overall_score,
            timestamp=datetime.now().isoformat()
        )
    
    def generate_improvement_suggestions(self, metrics: QualityMetrics) -> List[str]:
        """품질 개선 제안 생성"""
        suggestions = []
        
        if metrics.accuracy_score < self.quality_thresholds["accuracy"]:
            suggestions.append("🎯 정확성 개선: 더 구체적인 근거와 사실을 포함하고, 추측성 표현을 줄이세요")
        
        if metrics.completeness_score < self.quality_thresholds["completeness"]:
            suggestions.append("📋 완전성 개선: 분석에 요약, 상세 분석, 결론, 제안 섹션을 모두 포함하세요")
        
        if metrics.relevance_score < self.quality_thresholds["relevance"]:
            suggestions.append("🔍 관련성 개선: 핵심 주제와 더 밀접하게 연관된 내용으로 분석하세요")
        
        if metrics.clarity_score < self.quality_thresholds["clarity"]:
            suggestions.append("✨ 명확성 개선: 구조화된 형식과 논리적 연결어를 사용하여 가독성을 높이세요")
        
        if metrics.actionability_score < self.quality_thresholds["actionability"]:
            suggestions.append("🚀 실행가능성 개선: 구체적인 실행 계획과 추천사항을 포함하세요")
        
        return suggestions
    
    def validate_analysis_quality(self, analysis_result: str, original_content: str = "", context: str = "") -> Dict:
        """분석 품질 종합 검증"""
        
        start_time = time.time()
        
        # 품질 지표 계산
        metrics = self.calculate_quality_metrics(analysis_result, original_content, context)
        
        # 개선 제안 생성
        suggestions = self.generate_improvement_suggestions(metrics)
        
        # 품질 등급 결정
        if metrics.overall_score >= 0.8:
            quality_grade = "EXCELLENT"
            grade_emoji = "🏆"
        elif metrics.overall_score >= 0.7:
            quality_grade = "GOOD"
            grade_emoji = "✅"
        elif metrics.overall_score >= 0.5:
            quality_grade = "FAIR"
            grade_emoji = "⚠️"
        else:
            quality_grade = "POOR"
            grade_emoji = "❌"
        
        validation_time = time.time() - start_time
        
        # 검증 결과 저장
        validation_result = {
            "quality_grade": quality_grade,
            "grade_emoji": grade_emoji,
            "overall_score": round(metrics.overall_score, 3),
            "detailed_scores": {
                "accuracy": round(metrics.accuracy_score, 3),
                "completeness": round(metrics.completeness_score, 3),
                "relevance": round(metrics.relevance_score, 3),
                "clarity": round(metrics.clarity_score, 3),
                "actionability": round(metrics.actionability_score, 3)
            },
            "improvement_suggestions": suggestions,
            "validation_time": round(validation_time, 3),
            "timestamp": metrics.timestamp,
            "thresholds_met": {
                "accuracy": metrics.accuracy_score >= self.quality_thresholds["accuracy"],
                "completeness": metrics.completeness_score >= self.quality_thresholds["completeness"],
                "relevance": metrics.relevance_score >= self.quality_thresholds["relevance"],
                "clarity": metrics.clarity_score >= self.quality_thresholds["clarity"],
                "actionability": metrics.actionability_score >= self.quality_thresholds["actionability"],
                "overall": metrics.overall_score >= self.quality_thresholds["overall"]
            }
        }
        
        # 이력에 추가
        self.validation_history.append(validation_result)
        
        logger.info(f"품질 검증 완료: {quality_grade} ({metrics.overall_score:.3f})")
        
        return validation_result
    
    def get_quality_report(self, days: int = 7) -> Dict:
        """품질 트렌드 보고서 생성"""
        if not self.validation_history:
            return {"message": "검증 이력이 없습니다"}
        
        # 최근 N일 데이터 필터링
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_validations = [
            v for v in self.validation_history
            if datetime.fromisoformat(v["timestamp"]) >= cutoff_date
        ]
        
        if not recent_validations:
            return {"message": f"최근 {days}일간 검증 이력이 없습니다"}
        
        # 통계 계산
        total_validations = len(recent_validations)
        avg_score = sum(v["overall_score"] for v in recent_validations) / total_validations
        
        grade_counts = defaultdict(int)
        for v in recent_validations:
            grade_counts[v["quality_grade"]] += 1
        
        # 개선 트렌드
        scores = [v["overall_score"] for v in recent_validations]
        if len(scores) >= 2:
            trend = "개선" if scores[-1] > scores[0] else "악화" if scores[-1] < scores[0] else "유지"
        else:
            trend = "데이터 부족"
        
        return {
            "period": f"최근 {days}일",
            "총_검증수": total_validations,
            "평균_품질점수": round(avg_score, 3),
            "품질_등급_분포": dict(grade_counts),
            "트렌드": trend,
            "최고_점수": max(scores),
            "최저_점수": min(scores),
            "기준_달성률": {
                criterion: sum(1 for v in recent_validations if v["thresholds_met"][criterion]) / total_validations * 100
                for criterion in self.quality_thresholds.keys()
            }
        }

# 전역 인스턴스
quality_validator = QualityValidator()

if __name__ == "__main__":
    # 테스트 실행
    test_analysis = """
    === 주얼리 시장 분석 결과 ===
    
    ## 핵심 요약
    최근 다이아몬드 반지 시장은 프리미엄 세그먼트에서 15% 성장을 보였습니다.
    
    ## 상세 분석
    1. 시장 동향: 고품질 다이아몬드에 대한 수요 증가
    2. 고객 선호: 클래식한 디자인보다 모던한 스타일 선호
    3. 가격 동향: 원자재 가격 상승으로 인한 제품 가격 10% 인상
    
    ## 제안사항
    - 모던 디자인 라인 확대 개발
    - 프리미엄 마케팅 전략 수립
    - 고객 맞춤 서비스 강화
    """
    
    test_original = "다이아몬드 반지 시장 분석을 위한 회의 내용입니다."
    
    result = quality_validator.validate_analysis_quality(
        analysis_result=test_analysis,
        original_content=test_original,
        context="주얼리 시장 분석"
    )
    
    print("=== 품질 검증 결과 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n=== 품질 보고서 ===")
    report = quality_validator.get_quality_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))