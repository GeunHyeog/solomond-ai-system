#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-1단계: 클러스터링 알고리즘 개선
"""
import os
import sys
import numpy as np
from pathlib import Path

# 인코딩 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'

def improve_enhanced_speaker_identifier():
    """Enhanced Speaker Identifier 클러스터링 개선"""
    print("=== 2-1단계: 클러스터링 알고리즘 개선 ===\n")
    
    # 기존 클래스 수정을 위한 패치
    from enhanced_speaker_identifier import EnhancedSpeakerIdentifier
    
    class ImprovedSpeakerIdentifier(EnhancedSpeakerIdentifier):
        """개선된 화자 구분 시스템"""
        
        def __init__(self, expected_speakers=3):
            super().__init__(expected_speakers)
            # 중요 특징에 가중치 부여
            self.feature_weights = {
                'formality_ratio': 3.0,      # 격식도는 화자 구분에 매우 중요
                'question_ratio': 2.5,       # 질문 패턴도 중요
                'response_ratio': 2.5,       # 응답 패턴도 중요
                'vocabulary_diversity': 2.0, # 어휘 다양성 중요
                'avg_sentence_length': 1.5,  # 문장 길이 패턴
                'emphasis_ratio': 1.5,       # 강조 표현
                'connector_ratio': 1.0,      # 연결어 사용
                'exclamation_ratio': 1.0,    # 감탄사
                'positive_ratio': 0.8,       # 긍정 표현
                'negative_ratio': 0.8,       # 부정 표현
                'avg_word_length': 0.5,      # 단어 길이는 덜 중요
                'total_length': 0.3,         # 전체 길이는 화자 특성과 관련성 낮음
                'word_count': 0.3            # 단어 수도 화자 특성과 관련성 낮음
            }
        
        def _apply_feature_weights(self, feature_matrix, feature_names):
            """특징에 가중치 적용"""
            weighted_matrix = feature_matrix.copy()
            
            for i, feature_name in enumerate(feature_names):
                weight = self.feature_weights.get(feature_name, 1.0)
                weighted_matrix[:, i] *= weight
                
            return weighted_matrix
        
        def _improved_kmeans(self, data, k, max_iters=100):
            """개선된 K-means 클러스터링"""
            if len(data) <= k:
                return list(range(len(data)))
            
            # 1. 더 나은 초기화: k-means++
            centroids = self._kmeans_plus_plus_init(data, k)
            
            best_assignments = None
            best_inertia = float('inf')
            
            # 여러 번 시도해서 최적 결과 선택
            for attempt in range(3):
                current_centroids = centroids.copy()
                
                for iteration in range(max_iters):
                    # 각 점을 가장 가까운 중심점에 할당
                    distances = np.sqrt(((data - current_centroids[:, np.newaxis])**2).sum(axis=2))
                    assignments = np.argmin(distances, axis=0)
                    
                    # 중심점 업데이트
                    new_centroids = np.array([
                        data[assignments == i].mean(axis=0) if np.any(assignments == i) 
                        else current_centroids[i] for i in range(k)
                    ])
                    
                    # 수렴 확인
                    if np.allclose(current_centroids, new_centroids, rtol=1e-4):
                        break
                    
                    current_centroids = new_centroids
                
                # inertia 계산 (클러스터 내 분산)
                inertia = 0
                for i in range(k):
                    cluster_points = data[assignments == i]
                    if len(cluster_points) > 0:
                        inertia += np.sum((cluster_points - current_centroids[i])**2)
                
                if inertia < best_inertia:
                    best_inertia = inertia
                    best_assignments = assignments.copy()
                
                # 다음 시도를 위해 중심점 재초기화
                centroids = self._kmeans_plus_plus_init(data, k)
            
            return best_assignments.tolist() if best_assignments is not None else list(range(len(data)))
        
        def _kmeans_plus_plus_init(self, data, k):
            """k-means++ 초기화"""
            n_samples, n_features = data.shape
            centroids = np.empty((k, n_features))
            
            # 첫 번째 중심점은 랜덤
            centroids[0] = data[np.random.randint(n_samples)]
            
            # 나머지 중심점들을 확률적으로 선택
            for i in range(1, k):
                distances = np.array([min([np.sum((x - c)**2) for c in centroids[:i]]) for x in data])
                probabilities = distances / distances.sum()
                cumulative_probabilities = probabilities.cumsum()
                r = np.random.rand()
                
                for j, p in enumerate(cumulative_probabilities):
                    if r < p:
                        centroids[i] = data[j]
                        break
            
            return centroids
        
        def _adaptive_clustering(self, feature_matrix, feature_names):
            """적응적 클러스터링"""
            # 가중치 적용
            weighted_matrix = self._apply_feature_weights(feature_matrix, feature_names)
            
            # 정규화
            normalized_matrix = self._normalize_features(weighted_matrix)
            
            # 최적 클러스터 수 결정 (2~expected_speakers 범위에서)
            best_k = self.expected_speakers
            best_silhouette = -1
            
            for k in range(2, min(self.expected_speakers + 2, len(feature_matrix))):
                assignments = self._improved_kmeans(normalized_matrix, k)
                
                # 간단한 실루엣 점수 계산
                silhouette = self._calculate_silhouette_simple(normalized_matrix, assignments, k)
                
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_k = k
            
            # 최적 k로 최종 클러스터링
            final_assignments = self._improved_kmeans(normalized_matrix, best_k)
            
            print(f"[ADAPTIVE] 최적 클러스터 수: {best_k}, 실루엣 점수: {best_silhouette:.3f}")
            
            return final_assignments
        
        def _calculate_silhouette_simple(self, data, assignments, k):
            """간단한 실루엣 점수 계산"""
            if k <= 1 or len(set(assignments)) <= 1:
                return -1
            
            silhouette_scores = []
            
            for i in range(len(data)):
                # 같은 클러스터 내 평균 거리 (a)
                same_cluster = [j for j, label in enumerate(assignments) if label == assignments[i] and j != i]
                if len(same_cluster) == 0:
                    continue
                
                a = np.mean([np.linalg.norm(data[i] - data[j]) for j in same_cluster])
                
                # 다른 클러스터와의 최소 평균 거리 (b)
                b = float('inf')
                for cluster_id in set(assignments):
                    if cluster_id != assignments[i]:
                        other_cluster = [j for j, label in enumerate(assignments) if label == cluster_id]
                        if len(other_cluster) > 0:
                            avg_dist = np.mean([np.linalg.norm(data[i] - data[j]) for j in other_cluster])
                            b = min(b, avg_dist)
                
                if b == float('inf'):
                    continue
                
                # 실루엣 점수
                s = (b - a) / max(a, b) if max(a, b) > 0 else 0
                silhouette_scores.append(s)
            
            return np.mean(silhouette_scores) if silhouette_scores else -1
        
        def _cluster_speakers(self, segment_features):
            """개선된 화자 클러스터링"""
            if not segment_features:
                return []
            
            if len(segment_features) == 1:
                return [0]
            
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
            
            # 적응적 클러스터링 사용
            cluster_assignments = self._adaptive_clustering(feature_matrix, feature_names)
            
            return cluster_assignments
    
    return ImprovedSpeakerIdentifier

def test_improved_system():
    """개선된 시스템 테스트"""
    print("[1] 개선된 시스템 테스트")
    
    ImprovedSpeakerIdentifier = improve_enhanced_speaker_identifier()
    identifier = ImprovedSpeakerIdentifier(expected_speakers=3)
    
    # 극명한 차이의 테스트 데이터
    test_segments = [
        # 격식체 (높은 formality_ratio)
        {"start": 0, "end": 3, "text": "안녕하십니까. 정말로 감사드립니다. 준비해주신 자료를 검토하겠습니다."},
        {"start": 4, "end": 7, "text": "귀중한 의견을 주셔서 감사합니다. 충분히 검토하도록 하겠습니다."},
        
        # 질문형 (높은 question_ratio)  
        {"start": 8, "end": 11, "text": "이거 어떻게 하나요? 언제까지 해야 되나요? 누가 도와주나요?"},
        {"start": 12, "end": 15, "text": "진짜 이게 맞나요? 다른 방법은 없나요? 확실한가요?"},
        
        # 응답형 (높은 response_ratio, connector_ratio)
        {"start": 16, "end": 19, "text": "네, 맞습니다. 그런데 이 부분은 다시 검토해야 할 것 같습니다."},
        {"start": 20, "end": 23, "text": "네, 그렇습니다. 하지만 추가 확인이 필요합니다. 따라서 신중해야겠네요."}
    ]
    
    print("\n각 발언의 주요 특징:")
    for i, segment in enumerate(test_segments):
        features = identifier.extract_text_features(segment['text'])
        print(f"발언 {i+1}: 격식도={features['formality_ratio']:.2f}, "
              f"질문도={features['question_ratio']:.2f}, "
              f"응답도={features['response_ratio']:.2f}")
    
    # 화자 구분 실행
    speaker_segments = identifier.identify_speakers_from_segments(test_segments)
    
    # 결과 분석
    speakers = {}
    for segment in speaker_segments:
        speaker = segment.get('speaker', '미분류')
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(segment)
    
    print(f"\n[RESULT] 개선된 시스템 - 감지된 화자 수: {len(speakers)}명")
    
    for speaker, segments in speakers.items():
        print(f"\n{speaker}:")
        for seg in segments:
            print(f"  \"{seg['text'][:40]}...\"")
    
    return len(speakers) > 1  # 성공 여부: 2명 이상 화자 구분

def main():
    """메인 실행"""
    print("클러스터링 알고리즘 개선을 시작합니다.\n")
    
    success = test_improved_system()
    
    if success:
        print("\n[SUCCESS] 클러스터링 개선 성공!")
        print("- 가중치 기반 특징 선택")
        print("- k-means++ 초기화")
        print("- 적응적 클러스터 수 결정")
        print("- 실루엣 점수 기반 최적화")
        
        # 개선된 코드를 실제 파일에 적용
        print("\n[2] enhanced_speaker_identifier.py 파일 업데이트 중...")
        update_original_file()
        
    else:
        print("\n[INFO] 클러스터링 개선이 필요합니다.")
        print("다음 단계에서 추가 개선 방안을 적용하겠습니다.")
    
    print("\n[NEXT] 3단계: 실제 오디오 파일로 검증")

def update_original_file():
    """원본 파일에 개선사항 적용"""
    print("파일 업데이트 준비 완료. 다음 단계에서 실제 적용하겠습니다.")

if __name__ == "__main__":
    main()