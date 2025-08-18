#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2단계: Enhanced Speaker Identifier 심화 테스트 및 개선
"""
import os
import sys
import numpy as np
from pathlib import Path

# 인코딩 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'

def analyze_clustering_issue():
    """클러스터링 문제 분석 및 해결"""
    print("=== 2단계: Enhanced Speaker Identifier 심화 테스트 ===\n")
    
    from enhanced_speaker_identifier import EnhancedSpeakerIdentifier
    
    identifier = EnhancedSpeakerIdentifier(expected_speakers=3)
    
    # 더 극명한 차이를 가진 테스트 데이터
    test_segments = [
        # 격식체 화자 (존댓말, 긴 문장)
        {
            "start": 0.0, "end": 5.0, 
            "text": "안녕하십니까. 오늘 이렇게 귀중한 시간을 내어 중요한 회의에 참석해 주셔서 진심으로 감사드립니다. 준비된 안건을 차례대로 검토하도록 하겠습니다."
        },
        {
            "start": 6.0, "end": 10.0,
            "text": "첫 번째 안건은 예산 관련 사항입니다. 각 부서별 배정 현황과 향후 계획에 대해서 상세히 설명드리겠습니다."
        },
        
        # 비격식체 화자 (반말, 짧은 문장, 질문 많음)
        {
            "start": 12.0, "end": 15.0,
            "text": "네, 안녕하세요! 그런데 이거 진짜 중요한 거 맞나요? 어떻게 해야 되는 거예요?"
        },
        {
            "start": 16.0, "end": 19.0,
            "text": "아, 정말요? 그럼 우리가 뭘 해야 하죠? 언제까지 해야 돼요?"
        },
        
        # 응답형 화자 (짧은 응답, 연결어 많음)
        {
            "start": 21.0, "end": 24.0,
            "text": "네, 맞습니다. 그런데 그 부분에 대해서는 추가 검토가 필요할 것 같아요."
        },
        {
            "start": 25.0, "end": 28.0,
            "text": "그렇습니다. 하지만 이 방법도 고려해볼 만합니다. 따라서 신중하게 결정해야겠네요."
        }
    ]
    
    print("[1] 극명한 차이를 가진 테스트 데이터로 재검증")
    
    # 각 세그먼트의 특징 개별 분석
    print("\n[ANALYSIS] 각 발언의 텍스트 특징:")
    for i, segment in enumerate(test_segments):
        text = segment['text']
        features = identifier.extract_text_features(text)
        
        print(f"\n발언 {i+1}: \"{text[:30]}...\"")
        print(f"  격식도: {features['formality_ratio']:.2f}")
        print(f"  질문도: {features['question_ratio']:.2f}")
        print(f"  응답도: {features['response_ratio']:.2f}")
        print(f"  어휘다양성: {features['vocabulary_diversity']:.2f}")
        print(f"  평균문장길이: {features['avg_sentence_length']:.1f}")
    
    # 화자 구분 실행
    print("\n[2] 화자 구분 실행 및 분석")
    speaker_segments = identifier.identify_speakers_from_segments(test_segments)
    
    # 결과 분석
    speakers = {}
    for segment in speaker_segments:
        speaker = segment.get('speaker', '미분류')
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(segment)
    
    print(f"\n[RESULT] 감지된 화자 수: {len(speakers)}명")
    
    for speaker, segments in speakers.items():
        print(f"\n{speaker}:")
        for seg in segments:
            print(f"  \"{seg['text'][:50]}...\"")
    
    # 클러스터링 과정 상세 분석
    print("\n[3] 클러스터링 과정 상세 분석")
    
    # 특징 벡터 직접 계산
    feature_vectors = []
    for segment in test_segments:
        features = identifier.extract_text_features(segment['text'])
        feature_names = list(features.keys())
        vector = [features[name] for name in feature_names]
        feature_vectors.append(vector)
    
    feature_matrix = np.array(feature_vectors)
    print(f"특징 매트릭스 크기: {feature_matrix.shape}")
    
    # 정규화 전후 비교
    print("\n정규화 전 특징값 범위:")
    for i, name in enumerate(feature_names[:5]):  # 처음 5개만
        col = feature_matrix[:, i]
        print(f"  {name}: {col.min():.3f} ~ {col.max():.3f}")
    
    # 정규화 적용
    normalized_matrix = identifier._normalize_features(feature_matrix.copy())
    
    print("\n정규화 후 특징값 범위:")
    for i, name in enumerate(feature_names[:5]):
        col = normalized_matrix[:, i]
        print(f"  {name}: {col.min():.3f} ~ {col.max():.3f}")
    
    # 거리 계산
    print("\n[4] 발언 간 거리 분석")
    from scipy.spatial.distance import pdist, squareform
    
    try:
        distances = pdist(normalized_matrix, metric='euclidean')
        distance_matrix = squareform(distances)
        
        print("발언 간 유클리드 거리:")
        for i in range(len(test_segments)):
            for j in range(i+1, len(test_segments)):
                print(f"  발언{i+1} - 발언{j+1}: {distance_matrix[i,j]:.3f}")
    
    except ImportError:
        print("scipy가 없어서 거리 계산을 수동으로 수행합니다.")
        
        print("발언 간 수동 거리 계산:")
        for i in range(len(normalized_matrix)):
            for j in range(i+1, len(normalized_matrix)):
                diff = normalized_matrix[i] - normalized_matrix[j]
                distance = np.sqrt(np.sum(diff**2))
                print(f"  발언{i+1} - 발언{j+1}: {distance:.3f}")
    
    return speaker_segments

def test_parameter_tuning():
    """파라미터 튜닝 테스트"""
    print("\n[5] 파라미터 튜닝 테스트")
    
    from enhanced_speaker_identifier import EnhancedSpeakerIdentifier
    
    # 더 극단적인 테스트 데이터
    extreme_segments = [
        {"start": 0, "end": 3, "text": "안녕하십니까? 정말 감사합니다."},  # 격식체
        {"start": 4, "end": 6, "text": "야, 뭐야 이거? 완전 이상해!"},      # 비격식체
        {"start": 7, "end": 9, "text": "네, 그렇습니다. 맞습니다."}         # 응답형
    ]
    
    # 다양한 설정으로 테스트
    configurations = [
        {'expected_speakers': 2, 'name': '2명 화자'},
        {'expected_speakers': 3, 'name': '3명 화자'},
        {'expected_speakers': 4, 'name': '4명 화자'}
    ]
    
    for config in configurations:
        print(f"\n[TEST] {config['name']} 설정:")
        identifier = EnhancedSpeakerIdentifier(expected_speakers=config['expected_speakers'])
        
        speaker_segments = identifier.identify_speakers_from_segments(extreme_segments)
        
        speakers = {}
        for segment in speaker_segments:
            speaker = segment.get('speaker', '미분류')
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append(segment)
        
        print(f"  감지된 화자 수: {len(speakers)}명")
        for speaker, segments in speakers.items():
            texts = [seg['text'][:20] + "..." for seg in segments]
            print(f"    {speaker}: {', '.join(texts)}")

def main():
    """메인 실행"""
    print("Enhanced Speaker Identifier 심화 분석을 시작합니다.\n")
    
    # 클러스터링 문제 분석
    speaker_segments = analyze_clustering_issue()
    
    # 파라미터 튜닝 테스트
    test_parameter_tuning()
    
    print("\n[CONCLUSION] 2단계 분석 결과:")
    print("1. 현재 알고리즘은 미세한 언어 차이를 구분하기 어려움")
    print("2. 더 극명한 특징 차이가 필요함")
    print("3. 클러스터링 파라미터 조정 필요")
    print("4. 특징 가중치 조정이 필요할 수 있음")
    
    print("\n[NEXT] 3단계: 실제 오디오 파일로 테스트 진행")

if __name__ == "__main__":
    main()