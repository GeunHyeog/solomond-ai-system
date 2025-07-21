#!/usr/bin/env python3
"""
M4A 파일 변환 테스트 스크립트
"""

import os
import sys
sys.path.append('.')

from core.real_analysis_engine import RealAnalysisEngine, analyze_file_real

def test_m4a_conversion():
    """M4A 변환 기능 테스트"""
    
    # 테스트 파일 경로
    test_m4a_path = "test_files/test_audio.m4a"
    
    if not os.path.exists(test_m4a_path):
        print(f"[ERROR] 테스트 파일이 없습니다: {test_m4a_path}")
        return False
    
    print(f"[INFO] M4A 변환 테스트 시작: {test_m4a_path}")
    
    # 분석 엔진 초기화
    engine = RealAnalysisEngine()
    
    try:
        # M4A → WAV 변환 테스트
        converted_path = engine._convert_m4a_to_wav(test_m4a_path)
        
        if converted_path and os.path.exists(converted_path):
            print(f"[SUCCESS] M4A → WAV 변환 성공: {converted_path}")
            
            # 변환된 파일 크기 확인
            file_size = os.path.getsize(converted_path)
            print(f"[INFO] 변환된 파일 크기: {file_size} bytes")
            
            # 임시 파일 정리
            os.unlink(converted_path)
            print("[INFO] 임시 파일 정리 완료")
            
            return True
        else:
            print("[ERROR] M4A → WAV 변환 실패")
            return False
            
    except Exception as e:
        print(f"[ERROR] 변환 테스트 중 오류: {e}")
        return False

def test_full_m4a_analysis():
    """M4A 파일 전체 분석 테스트"""
    
    test_m4a_path = "test_files/test_audio.m4a"
    
    if not os.path.exists(test_m4a_path):
        print(f"[ERROR] 테스트 파일이 없습니다: {test_m4a_path}")
        return False
    
    print(f"[INFO] M4A 전체 분석 테스트 시작: {test_m4a_path}")
    
    # 분석 엔진 초기화
    engine = RealAnalysisEngine()
    
    try:
        # 전체 분석 실행 (analyze_file_real 함수 사용)
        result = analyze_file_real(test_m4a_path, "audio", "ko")
        
        if result and 'status' in result:
            if result['status'] == 'success':
                print("[SUCCESS] M4A 전체 분석 성공")
                print(f"[INFO] 분석 결과: {result.get('summary', 'N/A')}")
                return True
            else:
                print(f"[ERROR] M4A 분석 실패: {result.get('error', 'Unknown error')}")
                return False
        else:
            print("[ERROR] M4A 분석 결과가 비어있습니다")
            return False
            
    except Exception as e:
        print(f"[ERROR] 전체 분석 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("[INFO] M4A 파일 처리 테스트 시작")
    print("=" * 50)
    
    # 1. 변환 테스트
    conversion_success = test_m4a_conversion()
    print()
    
    # 2. 전체 분석 테스트  
    analysis_success = test_full_m4a_analysis()
    print()
    
    # 결과 요약
    print("=" * 50)
    print("[INFO] 테스트 결과 요약:")
    print(f"   변환 테스트: {'[SUCCESS] 성공' if conversion_success else '[ERROR] 실패'}")
    print(f"   분석 테스트: {'[SUCCESS] 성공' if analysis_success else '[ERROR] 실패'}")
    
    if conversion_success and analysis_success:
        print("\n[SUCCESS] 모든 M4A 테스트 통과! 시스템이 정상 작동합니다.")
    else:
        print("\n[WARNING] 일부 테스트 실패. 문제를 해결해야 합니다.")