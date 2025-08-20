#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[MULTIMODAL FUSION] 멀티모달 융합 엔진
Advanced Multimodal Fusion Engine for Conference Analysis

핵심 기능:
1. 이미지-텍스트 상관관계 분석 (시간 기반 매칭)
2. 오디오-텍스트 의미적 연결 분석
3. 크로스 모달 정보 추출 및 강화
4. 컨퍼런스 맥락 통합 분석
5. 멀티모달 인사이트 생성
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import time
import logging
from pathlib import Path
import json
import hashlib
import math
from collections import defaultdict, Counter
from datetime import datetime, timedelta

# 텍스트 분석
try:
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    TEXT_ANALYSIS_AVAILABLE = True
except ImportError:
    TEXT_ANALYSIS_AVAILABLE = False

# 이미지 분석
try:
    import cv2
    from PIL import Image
    IMAGE_ANALYSIS_AVAILABLE = True
except ImportError:
    IMAGE_ANALYSIS_AVAILABLE = False

# 로깅 설정
logger = logging.getLogger(__name__)

@dataclass
class ModalData:
    """모달별 데이터 구조"""
    modal_type: str  # 'image', 'audio', 'text'
    content: str
    timestamp: float
    confidence: float
    file_source: str
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CrossModalCorrelation:
    """크로스 모달 상관관계"""
    modal1: ModalData
    modal2: ModalData
    correlation_score: float
    correlation_type: str  # 'temporal', 'semantic', 'contextual'
    explanation: str
    confidence: float

@dataclass
class FusionResult:
    """융합 결과"""
    success: bool
    correlations: List[CrossModalCorrelation]
    unified_narrative: str
    key_insights: List[str]
    modal_summary: Dict[str, Any]
    processing_time: float
    confidence_score: float
    error_message: Optional[str] = None

class TemporalAnalyzer:
    """시간적 상관관계 분석기"""
    
    def __init__(self, time_window: float = 30.0):
        self.time_window = time_window  # 30초 윈도우
    
    def find_temporal_correlations(self, modal_data: List[ModalData]) -> List[CrossModalCorrelation]:
        """시간 기반 상관관계 찾기"""
        correlations = []
        
        # 시간순 정렬
        sorted_data = sorted(modal_data, key=lambda x: x.timestamp)
        
        for i, data1 in enumerate(sorted_data):
            for j, data2 in enumerate(sorted_data[i+1:], i+1):
                time_diff = abs(data2.timestamp - data1.timestamp)
                
                # 시간 윈도우 내에 있는 경우
                if time_diff <= self.time_window:
                    correlation_score = self._calculate_temporal_score(time_diff)
                    
                    if correlation_score > 0.3:  # 임계값
                        correlation = CrossModalCorrelation(
                            modal1=data1,
                            modal2=data2,
                            correlation_score=correlation_score,
                            correlation_type='temporal',
                            explanation=f"{time_diff:.1f}초 간격으로 발생한 {data1.modal_type}-{data2.modal_type} 연관성",
                            confidence=correlation_score
                        )
                        correlations.append(correlation)
        
        return correlations
    
    def _calculate_temporal_score(self, time_diff: float) -> float:
        """시간 차이 기반 상관관계 점수"""
        # 가우시안 함수로 시간 근접도 계산
        sigma = self.time_window / 3  # 3시그마 규칙
        score = math.exp(-(time_diff ** 2) / (2 * sigma ** 2))
        return score

class SemanticAnalyzer:
    """의미적 상관관계 분석기"""
    
    def __init__(self):
        self.vectorizer = None
        if TEXT_ANALYSIS_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
    
    def find_semantic_correlations(self, modal_data: List[ModalData]) -> List[CrossModalCorrelation]:
        """의미적 상관관계 찾기"""
        if not TEXT_ANALYSIS_AVAILABLE or not self.vectorizer:
            return []
        
        correlations = []
        
        # 텍스트 콘텐츠만 추출
        text_data = [data for data in modal_data if data.content.strip()]
        
        if len(text_data) < 2:
            return correlations
        
        try:
            # TF-IDF 벡터화
            texts = [self._preprocess_text(data.content) for data in text_data]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # 코사인 유사도 계산
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            for i, data1 in enumerate(text_data):
                for j, data2 in enumerate(text_data[i+1:], i+1):
                    similarity_score = similarity_matrix[i][j]
                    
                    if similarity_score > 0.2:  # 임계값
                        # 공통 키워드 추출
                        common_keywords = self._extract_common_keywords(data1.content, data2.content)
                        
                        correlation = CrossModalCorrelation(
                            modal1=data1,
                            modal2=data2,
                            correlation_score=similarity_score,
                            correlation_type='semantic',
                            explanation=f"공통 주제: {', '.join(common_keywords[:3])}",
                            confidence=similarity_score
                        )
                        correlations.append(correlation)
            
        except Exception as e:
            logger.warning(f"[WARNING] 의미적 분석 실패: {e}")
        
        return correlations
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        # 한글과 영어만 유지
        text = re.sub(r'[^가-힣a-zA-Z\s]', ' ', text)
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_common_keywords(self, text1: str, text2: str) -> List[str]:
        """공통 키워드 추출"""
        # 간단한 단어 빈도 기반 키워드 추출
        words1 = set(self._preprocess_text(text1).lower().split())
        words2 = set(self._preprocess_text(text2).lower().split())
        
        common_words = words1 & words2
        
        # 길이가 3자 이상인 단어만
        meaningful_words = [w for w in common_words if len(w) >= 3]
        
        return sorted(meaningful_words)[:5]

class ContextualAnalyzer:
    """맥락적 상관관계 분석기"""
    
    def __init__(self):
        # 컨퍼런스 관련 키워드 사전
        self.conference_keywords = {
            'presentation': ['발표', 'presentation', '슬라이드', 'slide', '스크린', 'screen'],
            'discussion': ['토론', 'discussion', '질문', 'question', '답변', 'answer'],
            'networking': ['네트워킹', 'networking', '명함', '소개', 'introduction'],
            'technical': ['기술', 'technology', '개발', 'development', '솔루션', 'solution'],
            'business': ['비즈니스', 'business', '투자', 'investment', '수익', 'profit']
        }
    
    def find_contextual_correlations(self, modal_data: List[ModalData]) -> List[CrossModalCorrelation]:
        """맥락적 상관관계 찾기"""
        correlations = []
        
        # 각 데이터의 컨텍스트 분류
        contextualized_data = []
        for data in modal_data:
            context_scores = self._analyze_context(data.content)
            data.metadata['context_scores'] = context_scores
            contextualized_data.append(data)
        
        # 같은 컨텍스트 내의 데이터 간 상관관계 찾기
        for context_type in self.conference_keywords.keys():
            context_data = [
                data for data in contextualized_data 
                if data.metadata.get('context_scores', {}).get(context_type, 0) > 0.3
            ]
            
            if len(context_data) >= 2:
                for i, data1 in enumerate(context_data):
                    for j, data2 in enumerate(context_data[i+1:], i+1):
                        # 다른 모달리티 간의 연결만
                        if data1.modal_type != data2.modal_type:
                            correlation_score = (
                                data1.metadata['context_scores'][context_type] +
                                data2.metadata['context_scores'][context_type]
                            ) / 2
                            
                            correlation = CrossModalCorrelation(
                                modal1=data1,
                                modal2=data2,
                                correlation_score=correlation_score,
                                correlation_type='contextual',
                                explanation=f"{context_type} 맥락에서의 {data1.modal_type}-{data2.modal_type} 연관성",
                                confidence=correlation_score
                            )
                            correlations.append(correlation)
        
        return correlations
    
    def _analyze_context(self, content: str) -> Dict[str, float]:
        """컨텍스트 분석"""
        content_lower = content.lower()
        context_scores = {}
        
        for context_type, keywords in self.conference_keywords.items():
            score = 0
            total_keywords = len(keywords)
            
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    score += 1
            
            context_scores[context_type] = score / total_keywords if total_keywords > 0 else 0
        
        return context_scores

class NarrativeGenerator:
    """통합 내러티브 생성기"""
    
    def __init__(self):
        self.narrative_templates = {
            'temporal': "{time}에 {modal1}과 {modal2}가 동시에 발생하여 연관성을 보입니다.",
            'semantic': "{modal1}과 {modal2}에서 '{keywords}' 주제로 의미적 연결이 발견되었습니다.",
            'contextual': "{context} 맥락에서 {modal1}과 {modal2}가 상호 보완적인 정보를 제공합니다."
        }
    
    def generate_unified_narrative(self, correlations: List[CrossModalCorrelation], modal_data: List[ModalData]) -> str:
        """통합된 내러티브 생성"""
        if not correlations:
            return self._generate_basic_summary(modal_data)
        
        # 상관관계 유형별 그룹화
        correlation_groups = defaultdict(list)
        for corr in correlations:
            correlation_groups[corr.correlation_type].append(corr)
        
        narrative_parts = []
        
        # 시간적 흐름 기반 내러티브
        if 'temporal' in correlation_groups:
            temporal_narrative = self._generate_temporal_narrative(correlation_groups['temporal'])
            narrative_parts.append(temporal_narrative)
        
        # 의미적 연결 기반 내러티브
        if 'semantic' in correlation_groups:
            semantic_narrative = self._generate_semantic_narrative(correlation_groups['semantic'])
            narrative_parts.append(semantic_narrative)
        
        # 맥락적 연결 기반 내러티브
        if 'contextual' in correlation_groups:
            contextual_narrative = self._generate_contextual_narrative(correlation_groups['contextual'])
            narrative_parts.append(contextual_narrative)
        
        # 전체 요약
        summary = self._generate_overall_summary(modal_data, correlations)
        narrative_parts.append(summary)
        
        return "\n\n".join(narrative_parts)
    
    def _generate_temporal_narrative(self, temporal_correlations: List[CrossModalCorrelation]) -> str:
        """시간적 내러티브 생성"""
        if not temporal_correlations:
            return ""
        
        narrative = "[시간적 연관성 분석]\n"
        
        # 시간순 정렬
        sorted_correlations = sorted(temporal_correlations, key=lambda x: x.modal1.timestamp)
        
        for i, corr in enumerate(sorted_correlations[:3]):  # 상위 3개만
            time_str = f"{corr.modal1.timestamp:.1f}초 ~ {corr.modal2.timestamp:.1f}초"
            narrative += f"• {time_str}: {corr.modal1.modal_type}과 {corr.modal2.modal_type}의 동시 발생 (연관도: {corr.correlation_score:.2f})\n"
        
        return narrative
    
    def _generate_semantic_narrative(self, semantic_correlations: List[CrossModalCorrelation]) -> str:
        """의미적 내러티브 생성"""
        if not semantic_correlations:
            return ""
        
        narrative = "[의미적 연관성 분석]\n"
        
        # 상관관계 점수 기준 정렬
        sorted_correlations = sorted(semantic_correlations, key=lambda x: x.correlation_score, reverse=True)
        
        for i, corr in enumerate(sorted_correlations[:3]):  # 상위 3개만
            narrative += f"• {corr.explanation} (유사도: {corr.correlation_score:.2f})\n"
        
        return narrative
    
    def _generate_contextual_narrative(self, contextual_correlations: List[CrossModalCorrelation]) -> str:
        """맥락적 내러티브 생성"""
        if not contextual_correlations:
            return ""
        
        narrative = "[맥락적 연관성 분석]\n"
        
        # 컨텍스트 유형별 그룹화
        context_groups = defaultdict(list)
        for corr in contextual_correlations:
            # explanation에서 컨텍스트 타입 추출
            for context_type in ['presentation', 'discussion', 'networking', 'technical', 'business']:
                if context_type in corr.explanation:
                    context_groups[context_type].append(corr)
                    break
        
        for context_type, corrs in context_groups.items():
            if corrs:
                best_corr = max(corrs, key=lambda x: x.correlation_score)
                narrative += f"• {context_type.title()} 맥락: {best_corr.explanation}\n"
        
        return narrative
    
    def _generate_overall_summary(self, modal_data: List[ModalData], correlations: List[CrossModalCorrelation]) -> str:
        """전체 요약 생성"""
        # 모달별 통계
        modal_counts = Counter(data.modal_type for data in modal_data)
        total_correlations = len(correlations)
        
        summary = f"[전체 요약]\n"
        summary += f"• 분석 데이터: {dict(modal_counts)}\n"
        summary += f"• 발견된 상관관계: {total_correlations}개\n"
        
        if correlations:
            avg_confidence = np.mean([corr.confidence for corr in correlations])
            summary += f"• 평균 신뢰도: {avg_confidence:.2f}\n"
            
            # 가장 강한 상관관계
            strongest_corr = max(correlations, key=lambda x: x.correlation_score)
            summary += f"• 최강 연관성: {strongest_corr.explanation} (점수: {strongest_corr.correlation_score:.2f})\n"
        
        return summary
    
    def _generate_basic_summary(self, modal_data: List[ModalData]) -> str:
        """기본 요약 (상관관계가 없을 때)"""
        modal_counts = Counter(data.modal_type for data in modal_data)
        
        summary = f"[기본 분석 결과]\n"
        summary += f"• 처리된 데이터: {dict(modal_counts)}\n"
        summary += f"• 명확한 상관관계를 찾지 못했지만, 각 모달리티별로 독립적인 정보를 제공합니다.\n"
        
        return summary

class MultimodalFusionEngine:
    """멀티모달 융합 엔진"""
    
    def __init__(self):
        self.temporal_analyzer = TemporalAnalyzer()
        self.semantic_analyzer = SemanticAnalyzer()
        self.contextual_analyzer = ContextualAnalyzer()
        self.narrative_generator = NarrativeGenerator()
        
        logger.info("[MULTIMODAL FUSION] 멀티모달 융합 엔진 초기화 완료")
    
    def fuse_modalities(self, fragments: List[Dict[str, Any]]) -> FusionResult:
        """멀티모달리티 융합 처리"""
        start_time = time.time()
        
        try:
            # 입력 데이터 검증
            if not fragments:
                return FusionResult(
                    success=False, correlations=[], unified_narrative="",
                    key_insights=[], modal_summary={}, processing_time=0,
                    confidence_score=0, error_message="입력 데이터가 없습니다"
                )
            
            # ModalData 객체로 변환
            modal_data = self._convert_to_modal_data(fragments)
            
            if len(modal_data) < 2:
                return self._single_modal_result(modal_data, time.time() - start_time)
            
            # 각 분석기로 상관관계 찾기
            all_correlations = []
            
            # 1. 시간적 상관관계
            temporal_correlations = self.temporal_analyzer.find_temporal_correlations(modal_data)
            all_correlations.extend(temporal_correlations)
            
            # 2. 의미적 상관관계
            semantic_correlations = self.semantic_analyzer.find_semantic_correlations(modal_data)
            all_correlations.extend(semantic_correlations)
            
            # 3. 맥락적 상관관계
            contextual_correlations = self.contextual_analyzer.find_contextual_correlations(modal_data)
            all_correlations.extend(contextual_correlations)
            
            # 중복 제거 및 정렬
            unique_correlations = self._deduplicate_correlations(all_correlations)
            sorted_correlations = sorted(unique_correlations, key=lambda x: x.correlation_score, reverse=True)
            
            # 통합 내러티브 생성
            unified_narrative = self.narrative_generator.generate_unified_narrative(sorted_correlations, modal_data)
            
            # 핵심 인사이트 추출
            key_insights = self._extract_key_insights(sorted_correlations, modal_data)
            
            # 모달 요약 생성
            modal_summary = self._generate_modal_summary(modal_data)
            
            # 전체 신뢰도 계산
            if sorted_correlations:
                confidence_score = np.mean([corr.confidence for corr in sorted_correlations])
            else:
                confidence_score = 0.3  # 기본 신뢰도
            
            processing_time = time.time() - start_time
            
            return FusionResult(
                success=True,
                correlations=sorted_correlations,
                unified_narrative=unified_narrative,
                key_insights=key_insights,
                modal_summary=modal_summary,
                processing_time=processing_time,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"[ERROR] 멀티모달 융합 실패: {e}")
            return FusionResult(
                success=False, correlations=[], unified_narrative="",
                key_insights=[], modal_summary={}, processing_time=processing_time,
                confidence_score=0, error_message=str(e)
            )
    
    def _convert_to_modal_data(self, fragments: List[Dict[str, Any]]) -> List[ModalData]:
        """프래그먼트를 ModalData로 변환"""
        modal_data = []
        
        for fragment in fragments:
            try:
                # 파일 타입에 따른 모달 타입 결정
                file_type = fragment.get('file_type', 'unknown')
                modal_type = self._determine_modal_type(file_type)
                
                # 타임스탬프 처리
                timestamp = self._extract_timestamp(fragment)
                
                modal = ModalData(
                    modal_type=modal_type,
                    content=fragment.get('content', ''),
                    timestamp=timestamp,
                    confidence=fragment.get('confidence', 0.5),
                    file_source=fragment.get('file_source', ''),
                    features=fragment.get('features', {}),
                    metadata=fragment
                )
                
                modal_data.append(modal)
            except Exception as e:
                logger.warning(f"[WARNING] 프래그먼트 변환 실패: {e}")
                continue
        
        return modal_data
    
    def _determine_modal_type(self, file_type: str) -> str:
        """파일 타입으로부터 모달 타입 결정"""
        if file_type in ['image', 'jpg', 'jpeg', 'png']:
            return 'image'
        elif file_type in ['audio', 'wav', 'mp3', 'm4a']:
            return 'audio'
        else:
            return 'text'
    
    def _extract_timestamp(self, fragment: Dict[str, Any]) -> float:
        """프래그먼트에서 타임스탬프 추출"""
        # 여러 필드에서 타임스탬프 추출 시도
        timestamp_fields = ['timestamp', 'start_time', 'created_at']
        
        for field in timestamp_fields:
            if field in fragment:
                value = fragment[field]
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    try:
                        # ISO 형식 타임스탬프 파싱 시도
                        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                        return dt.timestamp()
                    except:
                        continue
        
        # 기본값: 현재 시간
        return time.time()
    
    def _deduplicate_correlations(self, correlations: List[CrossModalCorrelation]) -> List[CrossModalCorrelation]:
        """중복 상관관계 제거"""
        seen = set()
        unique_correlations = []
        
        for corr in correlations:
            # 고유 키 생성 (순서 무관)
            key1 = f"{corr.modal1.file_source}_{corr.modal1.modal_type}"
            key2 = f"{corr.modal2.file_source}_{corr.modal2.modal_type}"
            unique_key = tuple(sorted([key1, key2]))
            
            if unique_key not in seen:
                seen.add(unique_key)
                unique_correlations.append(corr)
        
        return unique_correlations
    
    def _extract_key_insights(self, correlations: List[CrossModalCorrelation], modal_data: List[ModalData]) -> List[str]:
        """핵심 인사이트 추출"""
        insights = []
        
        if not correlations:
            insights.append("각 모달리티가 독립적인 정보를 제공하고 있습니다.")
            return insights
        
        # 1. 상관관계 통계
        correlation_types = Counter(corr.correlation_type for corr in correlations)
        if correlation_types:
            dominant_type = correlation_types.most_common(1)[0]
            insights.append(f"주요 연관 패턴: {dominant_type[0]} ({dominant_type[1]}개 발견)")
        
        # 2. 모달리티 간 연결성
        modal_pairs = Counter(f"{corr.modal1.modal_type}-{corr.modal2.modal_type}" for corr in correlations)
        if modal_pairs:
            strongest_pair = modal_pairs.most_common(1)[0]
            insights.append(f"가장 강한 연결: {strongest_pair[0]} ({strongest_pair[1]}번 연관)")
        
        # 3. 고신뢰도 상관관계
        high_confidence_corrs = [corr for corr in correlations if corr.confidence > 0.7]
        if high_confidence_corrs:
            insights.append(f"고신뢰도 연관성 {len(high_confidence_corrs)}개 발견 (신뢰도 70% 이상)")
        
        # 4. 시간적 클러스터링
        temporal_corrs = [corr for corr in correlations if corr.correlation_type == 'temporal']
        if temporal_corrs:
            time_clusters = self._find_time_clusters([corr.modal1.timestamp for corr in temporal_corrs])
            if len(time_clusters) > 1:
                insights.append(f"시간적으로 {len(time_clusters)}개 구간에서 집중적인 활동 감지")
        
        return insights[:5]  # 상위 5개만
    
    def _find_time_clusters(self, timestamps: List[float]) -> List[List[float]]:
        """시간 클러스터 찾기"""
        if len(timestamps) < 2:
            return [timestamps]
        
        try:
            # K-means로 시간 클러스터링
            timestamps_array = np.array(timestamps).reshape(-1, 1)
            n_clusters = min(3, len(timestamps))  # 최대 3개 클러스터
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(timestamps_array)
            
            clusters = defaultdict(list)
            for timestamp, label in zip(timestamps, labels):
                clusters[label].append(timestamp)
            
            return list(clusters.values())
        except:
            return [timestamps]
    
    def _generate_modal_summary(self, modal_data: List[ModalData]) -> Dict[str, Any]:
        """모달별 요약 생성"""
        summary = {
            'total_items': len(modal_data),
            'modal_distribution': {},
            'time_range': {},
            'average_confidence': 0,
            'content_length_stats': {}
        }
        
        # 모달별 분포
        modal_counts = Counter(data.modal_type for data in modal_data)
        summary['modal_distribution'] = dict(modal_counts)
        
        # 시간 범위
        if modal_data:
            timestamps = [data.timestamp for data in modal_data]
            summary['time_range'] = {
                'start': min(timestamps),
                'end': max(timestamps),
                'duration': max(timestamps) - min(timestamps)
            }
        
        # 평균 신뢰도
        confidences = [data.confidence for data in modal_data]
        if confidences:
            summary['average_confidence'] = np.mean(confidences)
        
        # 콘텐츠 길이 통계
        content_lengths = [len(data.content) for data in modal_data]
        if content_lengths:
            summary['content_length_stats'] = {
                'min': min(content_lengths),
                'max': max(content_lengths),
                'average': np.mean(content_lengths),
                'total_chars': sum(content_lengths)
            }
        
        return summary
    
    def _single_modal_result(self, modal_data: List[ModalData], processing_time: float) -> FusionResult:
        """단일 모달 결과"""
        if modal_data:
            narrative = f"단일 {modal_data[0].modal_type} 데이터가 분석되었습니다."
            insights = [f"{modal_data[0].modal_type} 모달리티에서 {len(modal_data[0].content)}자의 정보가 추출되었습니다."]
            modal_summary = self._generate_modal_summary(modal_data)
        else:
            narrative = "분석할 데이터가 없습니다."
            insights = ["데이터가 충분하지 않습니다."]
            modal_summary = {}
        
        return FusionResult(
            success=True,
            correlations=[],
            unified_narrative=narrative,
            key_insights=insights,
            modal_summary=modal_summary,
            processing_time=processing_time,
            confidence_score=0.3
        )

# 테스트 및 데모
if __name__ == "__main__":
    # 멀티모달 융합 엔진 테스트
    engine = MultimodalFusionEngine()
    
    print("[SUCCESS] 멀티모달 융합 엔진 초기화 완료")
    print(f"[INFO] 텍스트 분석: {'가능' if TEXT_ANALYSIS_AVAILABLE else '불가능 (sklearn 필요)'}")
    print(f"[INFO] 이미지 분석: {'가능' if IMAGE_ANALYSIS_AVAILABLE else '불가능 (opencv 필요)'}")
    
    # 더미 데이터로 테스트
    test_fragments = [
        {
            'file_type': 'image',
            'content': '발표 슬라이드 기술 혁신 AI 솔루션',
            'timestamp': '2025-08-20T10:30:00',
            'confidence': 0.9,
            'file_source': 'slide1.jpg'
        },
        {
            'file_type': 'audio',
            'content': '오늘 AI 기술에 대해 발표하겠습니다 혁신적인 솔루션을 소개드리겠습니다',
            'timestamp': '2025-08-20T10:30:15',
            'confidence': 0.85,
            'file_source': 'audio1.wav'
        }
    ]
    
    result = engine.fuse_modalities(test_fragments)
    
    if result.success:
        print(f"[SUCCESS] 융합 분석 완료 ({result.processing_time:.2f}초)")
        print(f"[INFO] 발견된 상관관계: {len(result.correlations)}개")
        print(f"[INFO] 신뢰도: {result.confidence_score:.2f}")
    else:
        print(f"[ERROR] 융합 분석 실패: {result.error_message}")
    
    print("[SUCCESS] 멀티모달 융합 엔진 테스트 완료")