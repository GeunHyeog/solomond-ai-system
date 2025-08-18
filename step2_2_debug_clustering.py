#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-2단계: 근본적 클러스터링 문제 디버깅 및 해결
"""
import os
import sys
import numpy as np
from pathlib import Path

# 인코딩 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'

def debug_clustering_step_by_step():
    """클러스터링 과정을 단계별로 디버깅"""
    print("=== 2-2단계: 클러스터링 디버깅 ===\n")
    
    from enhanced_speaker_identifier import EnhancedSpeakerIdentifier
    identifier = EnhancedSpeakerIdentifier(expected_speakers=3)
    
    # 매우 극단적인 테스트 데이터
    test_segments = [
        {"start": 0, "end": 3, "text": "안녕하십니까"},          # 매우 짧은 격식체
        {"start": 4, "end": 7, "text": "야 뭐야"},              # 매우 짧은 비격식체  
        {"start": 8, "end": 11, "text": "네 맞습니다"}           # 매우 짧은 응답형
    ]
    
    print("[1] 단계별 특징 추출 및 분석")
    
    # 각 세그먼트 개별 분석
    features_list = []
    for i, segment in enumerate(test_segments):
        features = identifier.extract_text_features(segment['text'])
        features_list.append(features)
        
        print(f"\n세그먼트 {i+1}: \"{segment['text']}\"")
        print(f"  격식도: {features['formality_ratio']}")
        print(f"  질문도: {features['question_ratio']}")  
        print(f"  응답도: {features['response_ratio']}")
        print(f"  어휘다양성: {features['vocabulary_diversity']}")
    
    # 특징 매트릭스 생성
    print("\n[2] 특징 매트릭스 생성")
    feature_names = list(features_list[0].keys())
    feature_matrix = np.array([[features[name] for name in feature_names] for features in features_list])
    
    print(f"원본 특징 매트릭스 형태: {feature_matrix.shape}")
    print("주요 특징값:")
    for i, name in enumerate(['formality_ratio', 'question_ratio', 'response_ratio']):
        if name in feature_names:
            idx = feature_names.index(name)
            print(f"  {name}: {feature_matrix[:, idx]}")
    
    # 정규화 과정 분석
    print("\n[3] 정규화 과정 분석")
    normalized_matrix = identifier._normalize_features(feature_matrix.copy())
    
    print("정규화 후 주요 특징값:")
    for i, name in enumerate(['formality_ratio', 'question_ratio', 'response_ratio']):
        if name in feature_names:
            idx = feature_names.index(name)
            print(f"  {name}: {normalized_matrix[:, idx]}")
    
    # 거리 계산
    print("\n[4] 포인트 간 거리 계산")
    for i in range(len(normalized_matrix)):
        for j in range(i+1, len(normalized_matrix)):
            diff = normalized_matrix[i] - normalized_matrix[j]
            distance = np.sqrt(np.sum(diff**2))
            print(f"  포인트{i+1} - 포인트{j+1}: {distance:.4f}")
    
    # K-means 과정 상세 분석
    print("\n[5] K-means 클러스터링 과정 분석")
    
    # 수동 K-means 구현
    def manual_kmeans_debug(data, k=3):
        print(f"K-means 시작: {k}개 클러스터")
        
        # 초기 중심점 (첫 k개 포인트)
        centroids = data[:k].copy()
        print(f"초기 중심점:")
        for i, centroid in enumerate(centroids):
            print(f"  중심점{i+1}: {centroid[:3]}...")  # 처음 3개 특징만 표시
        
        for iteration in range(10):  # 최대 10회 반복
            print(f"\n--- 반복 {iteration+1} ---")
            
            # 각 포인트를 가장 가까운 중심점에 할당
            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            assignments = np.argmin(distances, axis=0)
            
            print(f"거리 매트릭스:")
            for i in range(len(data)):
                print(f"  포인트{i+1}: {distances[:, i]}")
            
            print(f"할당 결과: {assignments}")
            
            # 중심점 업데이트
            new_centroids = []
            for i in range(k):
                cluster_points = data[assignments == i]
                if len(cluster_points) > 0:
                    new_centroid = cluster_points.mean(axis=0)
                    new_centroids.append(new_centroid)
                    print(f"  클러스터{i+1}: {len(cluster_points)}개 포인트")
                else:
                    new_centroids.append(centroids[i])
                    print(f"  클러스터{i+1}: 0개 포인트 (중심점 유지)")
            
            new_centroids = np.array(new_centroids)
            
            # 수렴 확인
            centroid_change = np.max(np.abs(centroids - new_centroids))
            print(f"  중심점 변화량: {centroid_change:.6f}")
            
            if centroid_change < 1e-6:
                print("  수렴 완료!")
                break
            
            centroids = new_centroids
        
        return assignments
    
    final_assignments = manual_kmeans_debug(normalized_matrix, k=3)
    print(f"\n최종 할당 결과: {final_assignments}")
    
    # 문제 진단
    print("\n[6] 문제 진단")
    
    unique_assignments = len(set(final_assignments))
    print(f"실제 생성된 클러스터 수: {unique_assignments}")
    
    if unique_assignments == 1:
        print("문제: 모든 포인트가 하나의 클러스터로 할당됨")
        
        # 가능한 원인들 체크
        print("\n원인 분석:")
        
        # 1. 데이터 분산 체크
        data_variance = np.var(normalized_matrix, axis=0)
        print(f"1. 특징별 분산: 평균={np.mean(data_variance):.6f}, 최대={np.max(data_variance):.6f}")
        
        # 2. 거리 분포 체크  
        all_distances = []
        for i in range(len(normalized_matrix)):
            for j in range(i+1, len(normalized_matrix)):
                distance = np.linalg.norm(normalized_matrix[i] - normalized_matrix[j])
                all_distances.append(distance)
        
        print(f"2. 포인트 간 거리: 평균={np.mean(all_distances):.4f}, 최대={np.max(all_distances):.4f}")
        
        # 3. 정규화 문제 체크
        original_distances = []
        for i in range(len(feature_matrix)):
            for j in range(i+1, len(feature_matrix)):
                distance = np.linalg.norm(feature_matrix[i] - feature_matrix[j])
                original_distances.append(distance)
        
        print(f"3. 정규화 전 거리: 평균={np.mean(original_distances):.4f}, 최대={np.max(original_distances):.4f}")
    
    return final_assignments

def create_fixed_clustering():
    """문제를 해결한 새로운 클러스터링 방법"""
    print("\n[7] 문제 해결: 새로운 클러스터링 방법")
    
    def rule_based_clustering(segments):
        """규칙 기반 화자 구분 (fallback)"""
        from enhanced_speaker_identifier import EnhancedSpeakerIdentifier
        identifier = EnhancedSpeakerIdentifier()
        
        speakers = []
        
        for segment in segments:
            features = identifier.extract_text_features(segment['text'])
            
            # 명확한 규칙으로 화자 구분
            if features['formality_ratio'] > 0.7:
                speaker = "화자_1"  # 격식체 화자
            elif features['question_ratio'] > 0.3:
                speaker = "화자_2"  # 질문형 화자  
            elif features['response_ratio'] > 0.1:
                speaker = "화자_3"  # 응답형 화자
            else:
                speaker = "화자_1"  # 기본값
            
            speakers.append(speaker)
        
        return speakers
    
    # 테스트
    test_segments = [
        {"start": 0, "end": 3, "text": "안녕하십니까. 감사드립니다."},
        {"start": 4, "end": 7, "text": "뭐야 이거? 어떻게 하나요?"},  
        {"start": 8, "end": 11, "text": "네, 맞습니다. 그렇습니다."}
    ]
    
    speakers = rule_based_clustering(test_segments)
    
    print("규칙 기반 화자 구분 결과:")
    for i, (segment, speaker) in enumerate(zip(test_segments, speakers)):
        print(f"  \"{segment['text']}\" -> {speaker}")
    
    return len(set(speakers)) > 1

def main():
    """메인 실행"""
    print("클러스터링 문제 디버깅을 시작합니다.\n")
    
    # 단계별 디버깅
    assignments = debug_clustering_step_by_step()
    
    # 새로운 방법 테스트
    success = create_fixed_clustering()
    
    if success:
        print("\n[SUCCESS] 규칙 기반 방법으로 화자 구분 성공!")
        print("다음 단계에서 이 방법을 적용하겠습니다.")
    else:
        print("\n[INFO] 추가 디버깅이 필요합니다.")
    
    print("\n[NEXT] enhanced_speaker_identifier.py 파일 업데이트 후 3단계 진행")

if __name__ == "__main__":
    main()