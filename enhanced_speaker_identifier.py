#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
향상된 화자 구분 시스템
Sequential Thinking + Perplexity 조사 결과 기반 구현
"""
import re
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import Counter
import math

class EnhancedSpeakerIdentifier:
    """향상된 화자 구분 시스템"""
    
    def __init__(self, expected_speakers=3):
        self.expected_speakers = expected_speakers
        self.speaker_features = {}
        self.speech_patterns = {
            'question_endings': ['습니까', '나요', '까요', '세요', '어요', '아요'],
            'response_starters': ['네', '예', '아니요', '그렇습니다', '맞습니다', '아닙니다'],
            'connectors': ['그런데', '그래서', '따라서', '그리고', '하지만', '그러나'],
            'formal_endings': ['습니다', '였습니다', '하겠습니다', '되었습니다'],
            'informal_endings': ['해요', '어요', '아요', '죠', '네요']
        }
    
    def extract_text_features(self, text: str) -> Dict[str, float]:
        """텍스트에서 화자 특징 추출"""
        features = {}
        
        # 기본 통계
        words = text.split()
        sentences = text.split('.')
        
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['avg_sentence_length'] = np.mean([len(sent.strip()) for sent in sentences if sent.strip()]) if sentences else 0
        features['total_length'] = len(text)
        features['word_count'] = len(words)
        
        # 어휘 다양성
        unique_words = set(words)
        features['vocabulary_diversity'] = len(unique_words) / len(words) if words else 0
        
        # 언어 패턴 분석
        features.update(self._analyze_speech_patterns(text))
        
        # 감정/톤 분석
        features.update(self._analyze_emotional_tone(text))
        
        return features
    
    def _analyze_speech_patterns(self, text: str) -> Dict[str, float]:
        """발화 패턴 분석"""
        patterns = {}
        text_lower = text.lower()
        
        # 질문 패턴
        question_count = sum(1 for ending in self.speech_patterns['question_endings'] 
                           if ending in text_lower)
        patterns['question_ratio'] = question_count / max(1, len(text.split('.')))
        
        # 응답 패턴
        response_count = sum(1 for starter in self.speech_patterns['response_starters']
                           if text_lower.startswith(starter) or f" {starter}" in text_lower)
        patterns['response_ratio'] = response_count / max(1, len(text.split()))
        
        # 연결어 사용
        connector_count = sum(1 for connector in self.speech_patterns['connectors']
                            if connector in text_lower)
        patterns['connector_ratio'] = connector_count / max(1, len(text.split()))
        
        # 격식/비격식 언어
        formal_count = sum(1 for ending in self.speech_patterns['formal_endings']
                         if ending in text_lower)
        informal_count = sum(1 for ending in self.speech_patterns['informal_endings']
                           if ending in text_lower)
        
        total_endings = formal_count + informal_count
        patterns['formality_ratio'] = formal_count / max(1, total_endings)
        
        return patterns
    
    def _analyze_emotional_tone(self, text: str) -> Dict[str, float]:
        """감정 톤 분석"""
        tone = {}
        
        # 감탄사 빈도
        exclamations = ['아', '오', '우와', '와', '어머', '음', '으음']
        exclamation_count = sum(text.lower().count(exc) for exc in exclamations)
        tone['exclamation_ratio'] = exclamation_count / max(1, len(text.split()))
        
        # 강조 표현
        emphasis_patterns = ['정말', '진짜', '너무', '아주', '매우', '굉장히']
        emphasis_count = sum(text.lower().count(emp) for emp in emphasis_patterns)
        tone['emphasis_ratio'] = emphasis_count / max(1, len(text.split()))
        
        # 긍정/부정 키워드
        positive_words = ['좋', '훌륭', '멋진', '완벽', '우수', '최고']
        negative_words = ['안', '못', '없', '나쁜', '어렵', '힘든']
        
        positive_count = sum(text.lower().count(pos) for pos in positive_words)
        negative_count = sum(text.lower().count(neg) for neg in negative_words)
        
        tone['positive_ratio'] = positive_count / max(1, len(text.split()))
        tone['negative_ratio'] = negative_count / max(1, len(text.split()))
        
        return tone
    
    def identify_speakers_from_segments(self, segments: List[Dict]) -> List[Dict]:
        """세그먼트 리스트에서 화자 구분"""
        
        if not segments:
            return []
        
        # 개별 세그먼트별로 화자 구분 (그룹핑 없이)
        segment_features = []
        for segment in segments:
            text = segment.get('text', '')
            if text.strip():
                features = self.extract_text_features(text)
                segment_features.append({
                    'segments': [segment],  # 개별 세그먼트
                    'features': features,
                    'text': text
                })
        
        # 화자 구분
        speaker_assignments = self._cluster_speakers(segment_features)
        
        # 결과 정리
        result_segments = []
        for i, feature_group in enumerate(segment_features):
            speaker_id = speaker_assignments[i]
            for segment in feature_group['segments']:
                segment_copy = segment.copy()
                segment_copy['speaker'] = f"화자_{speaker_id + 1}"
                result_segments.append(segment_copy)
        
        return result_segments
    
    def _group_by_silence_gaps(self, segments: List[Dict], silence_threshold: float = 2.0) -> List[List[Dict]]:
        """침묵 구간을 기준으로 세그먼트 그룹핑"""
        if not segments:
            return []
        
        groups = []
        current_group = [segments[0]]
        
        for i in range(1, len(segments)):
            prev_end = segments[i-1].get('end', 0)
            curr_start = segments[i].get('start', 0)
            
            # 침묵 구간이 임계값보다 크면 새 그룹 시작
            if curr_start - prev_end > silence_threshold:
                groups.append(current_group)
                current_group = [segments[i]]
            else:
                current_group.append(segments[i])
        
        groups.append(current_group)
        return groups
    
    def _cluster_speakers(self, segment_features: List[Dict]) -> List[int]:
        """하이브리드 화자 구분: 규칙 기반 + 클러스터링"""
        
        if not segment_features:
            return []
        
        if len(segment_features) == 1:
            return [0]
        
        # 1단계: 규칙 기반 화자 구분 (우선)
        rule_based_assignments = self._rule_based_speaker_classification(segment_features)
        
        # 2단계: 규칙으로 구분되지 않은 경우 클러스터링 적용
        if len(set(rule_based_assignments)) >= 2:
            print(f"[SPEAKER] 규칙 기반 구분 성공: {len(set(rule_based_assignments))}명")
            return rule_based_assignments
        else:
            print(f"[SPEAKER] 클러스터링 방법 적용")
            return self._clustering_fallback(segment_features)
    
    def _rule_based_speaker_classification(self, segment_features: List[Dict]) -> List[int]:
        """규칙 기반 화자 분류"""
        assignments = []
        speaker_mapping = {}  # 화자 스타일 -> 숫자 매핑
        next_speaker_id = 0
        
        for seg_feat in segment_features:
            features = seg_feat['features']
            
            # 명확한 언어 패턴으로 화자 타입 결정
            speaker_type = self._classify_speaker_type(features)
            
            # 화자 타입을 숫자 ID로 매핑
            if speaker_type not in speaker_mapping:
                speaker_mapping[speaker_type] = next_speaker_id
                next_speaker_id += 1
            
            assignments.append(speaker_mapping[speaker_type])
        
        return assignments
    
    def _classify_speaker_type(self, features: Dict[str, float]) -> str:
        """특징 기반 화자 타입 분류"""
        # 격식도 기반 분류 (가장 명확한 구분자)
        if features.get('formality_ratio', 0) > 0.5:
            return "formal"  # 격식체 화자
        
        # 질문 패턴 기반 분류
        if features.get('question_ratio', 0) > 0.3:
            return "questioner"  # 질문형 화자
        
        # 응답 패턴 기반 분류
        if features.get('response_ratio', 0) > 0.15:
            return "responder"  # 응답형 화자
        
        # 감탄사/강조 표현이 많은 경우
        if features.get('exclamation_ratio', 0) > 0.1 or features.get('emphasis_ratio', 0) > 0.1:
            return "expressive"  # 감정표현형 화자
        
        # 연결어를 많이 사용하는 경우
        if features.get('connector_ratio', 0) > 0.1:
            return "narrative"  # 서술형 화자
        
        # 문장이 긴 경우
        if features.get('avg_sentence_length', 0) > 30:
            return "detailed"  # 상세설명형 화자
        
        # 기본값
        return "general"  # 일반형 화자
    
    def _clustering_fallback(self, segment_features: List[Dict]) -> List[int]:
        """클러스터링 백업 방법"""
        # 특징 벡터 생성
        feature_vectors = []
        feature_names = None
        
        for seg_feat in segment_features:
            features = seg_feat['features']
            if feature_names is None:
                feature_names = list(features.keys())
            
            vector = [features.get(name, 0) for name in feature_names]
            feature_vectors.append(vector)
        
        feature_matrix = np.array(feature_vectors)
        
        # 정규화
        feature_matrix = self._normalize_features(feature_matrix)
        
        # K-means 클러스터링
        n_clusters = min(self.expected_speakers, len(segment_features))
        cluster_assignments = self._simple_kmeans(feature_matrix, n_clusters)
        
        return cluster_assignments
    
    def _normalize_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """특징 정규화"""
        # Min-max 정규화
        for i in range(feature_matrix.shape[1]):
            col = feature_matrix[:, i]
            col_min, col_max = col.min(), col.max()
            if col_max > col_min:
                feature_matrix[:, i] = (col - col_min) / (col_max - col_min)
        
        return feature_matrix
    
    def _simple_kmeans(self, data: np.ndarray, k: int, max_iters: int = 100) -> List[int]:
        """간단한 K-means 클러스터링 구현"""
        
        if len(data) <= k:
            return list(range(len(data)))
        
        # 초기 중심점 설정
        np.random.seed(42)  # 재현 가능한 결과를 위해
        centroids = data[np.random.choice(len(data), k, replace=False)]
        
        for _ in range(max_iters):
            # 각 점을 가장 가까운 중심점에 할당
            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            assignments = np.argmin(distances, axis=0)
            
            # 중심점 업데이트
            new_centroids = np.array([data[assignments == i].mean(axis=0) if np.any(assignments == i) 
                                    else centroids[i] for i in range(k)])
            
            # 수렴 확인
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        return assignments.tolist()
    
    def analyze_speaker_characteristics(self, speaker_segments: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """화자별 특성 분석"""
        
        speaker_analysis = {}
        
        for speaker_id, segments in speaker_segments.items():
            combined_text = " ".join([seg.get('text', '') for seg in segments])
            
            if not combined_text.strip():
                continue
            
            features = self.extract_text_features(combined_text)
            
            # 추가 분석
            analysis = {
                'features': features,
                'total_speaking_time': sum([seg.get('end', 0) - seg.get('start', 0) for seg in segments]),
                'utterance_count': len(segments),
                'avg_utterance_length': np.mean([len(seg.get('text', '')) for seg in segments]),
                'speech_style': self._classify_speech_style(features),
                'dominant_patterns': self._identify_dominant_patterns(combined_text)
            }
            
            speaker_analysis[speaker_id] = analysis
        
        return speaker_analysis
    
    def _classify_speech_style(self, features: Dict[str, float]) -> str:
        """발화 스타일 분류"""
        
        formality = features.get('formality_ratio', 0)
        question_ratio = features.get('question_ratio', 0)
        response_ratio = features.get('response_ratio', 0)
        
        if formality > 0.6:
            return "격식체"
        elif question_ratio > 0.3:
            return "질문형"
        elif response_ratio > 0.2:
            return "응답형"
        else:
            return "대화형"
    
    def _identify_dominant_patterns(self, text: str) -> List[str]:
        """주요 언어 패턴 식별"""
        patterns = []
        text_lower = text.lower()
        
        # 각 패턴 카테고리별 사용 빈도 확인
        for category, keywords in self.speech_patterns.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                patterns.append(f"{category}: {count}회")
        
        return patterns

def test_enhanced_speaker_identifier():
    """향상된 화자 구분 시스템 테스트"""
    
    print("=== 향상된 화자 구분 시스템 테스트 ===")
    
    # 테스트 데이터
    test_segments = [
        {"start": 0.0, "end": 3.0, "text": "안녕하세요. 오늘 회의에 참석해 주셔서 감사합니다."},
        {"start": 4.0, "end": 6.0, "text": "네, 안녕하세요."},
        {"start": 7.0, "end": 12.0, "text": "그럼 이제 시작하겠습니다. 첫 번째 안건은 무엇입니까?"},
        {"start": 13.0, "end": 18.0, "text": "첫 번째 안건은 예산 관련 사항입니다. 자료를 준비했어요."},
        {"start": 20.0, "end": 25.0, "text": "좋습니다. 그럼 설명해 주세요."},
        {"start": 26.0, "end": 35.0, "text": "네, 이번 분기 예산은 다음과 같습니다. 항목별로 설명드리겠습니다."}
    ]
    
    identifier = EnhancedSpeakerIdentifier(expected_speakers=3)
    
    # 화자 구분 실행
    result_segments = identifier.identify_speakers_from_segments(test_segments)
    
    # 결과 출력
    print("화자 구분 결과:")
    for segment in result_segments:
        print(f"[{segment['start']:.1f}s-{segment['end']:.1f}s] {segment['speaker']}: {segment['text']}")
    
    # 화자별 그룹핑
    speaker_segments = {}
    for segment in result_segments:
        speaker = segment['speaker']
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append(segment)
    
    # 화자별 특성 분석
    print("\n화자별 특성 분석:")
    analysis = identifier.analyze_speaker_characteristics(speaker_segments)
    
    for speaker, data in analysis.items():
        print(f"\n{speaker}:")
        print(f"  발화 횟수: {data['utterance_count']}회")
        print(f"  총 발화 시간: {data['total_speaking_time']:.1f}초")
        print(f"  평균 발화 길이: {data['avg_utterance_length']:.0f}자")
        print(f"  발화 스타일: {data['speech_style']}")
        print(f"  주요 패턴: {', '.join(data['dominant_patterns'][:3])}")

if __name__ == "__main__":
    test_enhanced_speaker_identifier()