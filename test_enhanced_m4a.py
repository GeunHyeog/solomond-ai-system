#!/usr/bin/env python3
"""
강화된 M4A 처리 시스템 테스트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.enhanced_m4a_processor import EnhancedM4AProcessor
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_test_m4a_files():
    """테스트용 M4A 파일들 찾기"""
    test_dirs = [
        "test_files",
        "uploads",
        "temp_files",
        "."
    ]
    
    m4a_files = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for file in Path(test_dir).glob("*.m4a"):
                m4a_files.append(str(file))
    
    return m4a_files

def test_m4a_processor():
    """M4A 프로세서 테스트"""
    print("🧪 강화된 M4A 처리 시스템 테스트 시작")
    print("=" * 50)
    
    # 프로세서 초기화
    processor = EnhancedM4AProcessor()
    
    # 테스트 파일 찾기
    m4a_files = find_test_m4a_files()
    
    if not m4a_files:
        print("❌ 테스트할 M4A 파일을 찾을 수 없습니다.")
        print("다음 위치에 M4A 파일을 배치해주세요:")
        print("- test_files/ 폴더")
        print("- uploads/ 폴더")  
        print("- 현재 디렉토리")
        return False
    
    print(f"📁 발견된 M4A 파일: {len(m4a_files)}개")
    
    success_count = 0
    total_count = len(m4a_files)
    
    for i, m4a_file in enumerate(m4a_files, 1):
        print(f"\n🎵 [{i}/{total_count}] 테스트: {os.path.basename(m4a_file)}")
        print("-" * 40)
        
        # 1. 파일 분석
        print("📊 파일 분석 중...")
        analysis = processor.analyze_m4a_file(m4a_file)
        
        print(f"   파일 크기: {analysis['file_size_mb']}MB")
        print(f"   오디오 스트림: {'✅' if analysis['has_audio_stream'] else '❌'}")
        print(f"   지속 시간: {analysis['duration_seconds']:.1f}초")
        print(f"   샘플링 레이트: {analysis['sample_rate']}Hz")
        print(f"   채널: {analysis['channels']}개")
        print(f"   추천 방법: {analysis['recommended_method']}")
        
        if analysis['issues']:
            print("   ⚠️  문제점:")
            for issue in analysis['issues']:
                print(f"      - {issue}")
        
        # 2. 변환 시도
        print("\n🔄 변환 시도 중...")
        converted_path = processor.process_m4a_to_wav(m4a_file)
        
        if converted_path and os.path.exists(converted_path):
            # 변환 결과 확인
            converted_size_mb = os.path.getsize(converted_path) / (1024 * 1024)
            print(f"✅ 변환 성공!")
            print(f"   변환된 파일: {os.path.basename(converted_path)}")
            print(f"   변환된 크기: {converted_size_mb:.2f}MB")
            
            success_count += 1
            
            # 변환된 파일 정리
            processor.cleanup_temp_files()
        else:
            print("❌ 변환 실패")
    
    print("\n" + "=" * 50)
    print("🏁 테스트 완료")
    print(f"✅ 성공: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    print(f"❌ 실패: {total_count-success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 모든 M4A 파일이 성공적으로 처리되었습니다!")
        return True
    elif success_count > total_count * 0.8:
        print("🎊 대부분의 M4A 파일이 성공적으로 처리되었습니다!")
        return True
    else:
        print("⚠️  일부 M4A 파일 처리에 문제가 있습니다.")
        return False

def test_single_m4a_file(file_path: str):
    """단일 M4A 파일 상세 테스트"""
    if not os.path.exists(file_path):
        print(f"❌ 파일이 존재하지 않습니다: {file_path}")
        return
    
    print(f"🎵 단일 파일 테스트: {os.path.basename(file_path)}")
    print("=" * 50)
    
    processor = EnhancedM4AProcessor()
    
    # 상세 분석
    analysis = processor.analyze_m4a_file(file_path)
    
    print("📊 상세 분석 결과:")
    for key, value in analysis.items():
        print(f"   {key}: {value}")
    
    # 변환 시도
    print("\n🔄 변환 시도...")
    converted_path = processor.process_m4a_to_wav(file_path)
    
    if converted_path:
        print(f"✅ 변환 성공: {converted_path}")
        
        # 변환된 파일 분석
        converted_size = os.path.getsize(converted_path)
        print(f"   변환된 파일 크기: {converted_size / (1024*1024):.2f}MB")
        
        # 정리
        processor.cleanup_temp_files()
    else:
        print("❌ 변환 실패")

if __name__ == "__main__":
    # 명령행 인수가 있으면 단일 파일 테스트
    if len(sys.argv) > 1:
        test_single_m4a_file(sys.argv[1])
    else:
        # 전체 테스트
        test_m4a_processor()