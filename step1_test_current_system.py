#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1단계: 현재 화자 구분 시스템 동작 확인 테스트
"""
import os
import sys
from pathlib import Path

# 인코딩 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'

def test_current_system():
    """현재 시스템 동작 확인"""
    print("=== 1단계: 현재 화자 구분 시스템 동작 확인 ===\n")
    
    # 1. Enhanced Speaker Identifier 테스트
    print("[1] Enhanced Speaker Identifier 기본 테스트")
    try:
        from enhanced_speaker_identifier import EnhancedSpeakerIdentifier
        
        identifier = EnhancedSpeakerIdentifier(expected_speakers=3)
        print("   [OK] 객체 생성 성공")
        
        # 간단한 텍스트 특징 추출 테스트
        test_text = "안녕하세요. 오늘 회의를 시작하겠습니다."
        features = identifier.extract_text_features(test_text)
        print(f"   [OK] 텍스트 특징 추출 성공: {len(features)}개 특징")
        print(f"      주요 특징: {list(features.keys())[:5]}")
        
    except Exception as e:
        print(f"   [ERROR] 오류: {e}")
        return False
    
    # 2. Speaker Mindmap Analyzer 테스트  
    print("\n[2] Speaker Mindmap Analyzer 기본 테스트")
    try:
        from speaker_mindmap_analyzer import SpeakerMindmapAnalyzer
        
        analyzer = SpeakerMindmapAnalyzer(expected_speakers=3)
        print("   [OK] 객체 생성 성공")
        print(f"   [OK] Enhanced 시스템 연동: {analyzer.enhanced_speaker_id is not None}")
        
    except Exception as e:
        print(f"   [ERROR] 오류: {e}")
        return False
    
    # 3. 가상 세그먼트 데이터로 화자 구분 테스트
    print("\n[3] 가상 세그먼트 데이터 화자 구분 테스트")
    try:
        # 다양한 스타일의 테스트 데이터
        test_segments = [
            {
                "start": 0.0, 
                "end": 3.0, 
                "text": "안녕하십니까. 오늘 중요한 회의에 참석해 주셔서 진심으로 감사드립니다."
            },
            {
                "start": 4.0, 
                "end": 6.0, 
                "text": "네, 안녕하세요!"
            },
            {
                "start": 7.0, 
                "end": 12.0, 
                "text": "그럼 바로 시작할까요? 첫 번째 안건이 뭔가요?"
            },
            {
                "start": 13.0, 
                "end": 18.0, 
                "text": "첫 번째 안건은 예산 관련 사항입니다. 자세한 내용을 설명드리겠습니다."
            },
            {
                "start": 20.0, 
                "end": 25.0, 
                "text": "아, 그거 정말 중요한 문제네요. 어떻게 해결할 생각이세요?"
            },
            {
                "start": 26.0, 
                "end": 35.0, 
                "text": "저희가 제안하는 방안은 다음과 같습니다. 단계별로 설명해 드리겠습니다."
            }
        ]
        
        # 화자 구분 실행
        speaker_segments = identifier.identify_speakers_from_segments(test_segments)
        
        print(f"   [OK] 화자 구분 완료: {len(speaker_segments)}개 세그먼트")
        
        # 결과 분석
        speakers = {}
        for segment in speaker_segments:
            speaker = segment.get('speaker', '미분류')
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append(segment)
        
        print(f"   [RESULT] 감지된 화자 수: {len(speakers)}명")
        
        for speaker, segments in speakers.items():
            total_time = sum([seg.get('end', 0) - seg.get('start', 0) for seg in segments])
            print(f"      {speaker}: {len(segments)}회 발언, {total_time:.1f}초")
            
            # 첫 번째 발언 예시
            if segments:
                example_text = segments[0].get('text', '')[:50]
                print(f"         예시: \"{example_text}...\"")
        
    except Exception as e:
        print(f"   [ERROR] 오류: {e}")
        return False
    
    # 4. 화자별 특성 분석 테스트
    print("\n[4] 화자별 특성 분석 테스트")
    try:
        # 화자별 그룹핑
        speaker_data = {}
        for segment in speaker_segments:
            speaker = segment.get('speaker', '미분류')
            if speaker not in speaker_data:
                speaker_data[speaker] = []
            speaker_data[speaker].append(segment)
        
        # 각 화자별 특성 분석
        analysis = identifier.analyze_speaker_characteristics(speaker_data)
        
        print(f"   [OK] 특성 분석 완료: {len(analysis)}명")
        
        for speaker, data in analysis.items():
            print(f"      {speaker}:")
            print(f"         발화 스타일: {data.get('speech_style', '분석 중')}")
            print(f"         평균 발언 길이: {data.get('avg_utterance_length', 0):.0f}자")
            
            patterns = data.get('dominant_patterns', [])
            if patterns:
                print(f"         주요 패턴: {', '.join(patterns[:2])}")
        
    except Exception as e:
        print(f"   [ERROR] 오류: {e}")
        return False
    
    print("\n[SUCCESS] 1단계 완료: 현재 시스템이 정상적으로 동작합니다!")
    print("\n[SUMMARY] 확인된 기능:")
    print("   - Enhanced Speaker Identifier 동작 확인")
    print("   - 텍스트 기반 화자 구분 작동")
    print("   - 언어 패턴 분석 기능 확인")
    print("   - 화자별 특성 분석 가능")
    
    return True

if __name__ == "__main__":
    success = test_current_system()
    if success:
        print("\n[NEXT] 다음 단계: Enhanced Speaker Identifier 심화 테스트")
    else:
        print("\n[ERROR] 문제 해결 후 다시 시도해주세요.")