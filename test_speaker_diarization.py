#!/usr/bin/env python3
"""
화자 구분 시스템 테스트
"""

import os
import sys
sys.path.append('.')

from core.speaker_diarization import speaker_diarization, analyze_speakers

def test_speaker_diarization():
    """화자 구분 시스템 테스트"""
    
    print("[INFO] 화자 구분 시스템 테스트 시작")
    print("=" * 50)
    
    # 설치 가이드 확인
    guide = speaker_diarization.get_installation_guide()
    print(f"[INFO] 시스템 사용 가능: {guide['available']}")
    
    if guide['missing_packages']:
        print("[WARNING] 누락된 패키지:")
        for pkg in guide['missing_packages']:
            print(f"  - {pkg['package']}: {pkg['command']} ({pkg['purpose']})")
            if 'note' in pkg:
                print(f"    주의: {pkg['note']}")
        print(f"[INFO] 전체 설치: {guide['install_all']}")
        
        if guide.get('additional_notes'):
            print("[INFO] 추가 참고사항:")
            for note in guide['additional_notes']:
                print(f"  - {note}")
    else:
        print("[SUCCESS] 모든 필요 패키지 설치됨")
    
    print()
    
    # 테스트 오디오 파일 확인
    test_files = [
        "test_files/test_audio.wav",
        "test_files/test_audio.m4a"
    ]
    
    available_files = [f for f in test_files if os.path.exists(f)]
    
    if not available_files:
        print("[INFO] 테스트할 오디오 파일이 없습니다.")
        print("[INFO] test_files/ 폴더에 오디오 파일을 추가하여 테스트할 수 있습니다.")
        return
    
    # 각 파일에 대해 테스트
    for audio_file in available_files:
        print(f"[INFO] 테스트 파일: {audio_file}")
        
        # 기본 화자 분석 테스트
        print("  [TEST] 기본 화자 분석...")
        result_basic = analyze_speakers(audio_file, method="basic")
        
        if result_basic['status'] == 'success':
            print("  [SUCCESS] 기본 분석 성공")
            print(f"    - 화자 수: {result_basic['speaker_count']}")
            print(f"    - 세그먼트 수: {len(result_basic['segments'])}")
            print(f"    - 처리 시간: {result_basic['processing_time']}초")
            
            # 포맷된 결과 생성
            formatted = speaker_diarization.format_diarization_result(result_basic)
            print("    - 세그먼트 정보:")
            for seg in formatted['formatted_segments'][:3]:  # 처음 3개만 표시
                print(f"      {seg['segment_id']}: {seg['speaker']} ({seg['start_time']}-{seg['end_time']})")
            
            # 화자별 통계
            stats = speaker_diarization.get_speaker_statistics(result_basic)
            print("    - 화자별 통계:")
            for speaker, stat in stats.items():
                print(f"      {speaker}: {stat['total_duration']}초, {stat['segment_count']}개 세그먼트")
        
        else:
            print(f"  [ERROR] 기본 분석 실패: {result_basic.get('error', 'Unknown')}")
        
        print()
        
        # 고급 화자 분석 테스트
        print("  [TEST] 고급 화자 분석...")
        result_advanced = analyze_speakers(audio_file, method="advanced")
        
        if result_advanced['status'] == 'success':
            print("  [SUCCESS] 고급 분석 성공")
        elif result_advanced['status'] == 'not_implemented':
            print(f"  [INFO] {result_advanced['message']}")
        else:
            print(f"  [ERROR] 고급 분석 실패: {result_advanced.get('error', 'Unknown')}")
        
        print("-" * 30)

if __name__ == "__main__":
    test_speaker_diarization()