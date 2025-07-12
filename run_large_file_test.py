#!/usr/bin/env python3
"""
고용량 파일 실전 테스트 실행기 v2.1
원클릭으로 대용량 파일 테스트 실행

사용법:
  python run_large_file_test.py          # 빠른 테스트 (5분 비디오)
  python run_large_file_test.py --full   # 전체 테스트 (1시간 비디오)
  python run_large_file_test.py --demo   # 데모 모드 (기존 파일 사용)
"""

import os
import sys
import asyncio
import argparse
import time
from pathlib import Path

# 현재 디렉토리를 파이썬 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from large_file_real_test_v21 import LargeFileRealTest, run_quick_test
    print("✅ 테스트 모듈 로드 성공")
except ImportError as e:
    print(f"❌ 모듈 로드 실패: {e}")
    print("필요한 모듈들이 설치되어 있는지 확인하세요:")
    print("pip install opencv-python librosa soundfile psutil aiofiles openai-whisper pytesseract moviepy pillow matplotlib seaborn")
    sys.exit(1)

async def run_demo_test():
    """데모 테스트 - 실제 파일 없이 시뮬레이션"""
    print("🎭 데모 모드 - 시뮬레이션 테스트")
    print("=" * 50)
    
    # 가상 진행률 시뮬레이션
    for i in range(101):
        progress = f"진행률: {i}% | 메모리: {300 + i*2:.1f}MB | CPU: {40 + i*0.3:.1f}% | 속도: {1.2 + i*0.01:.1f}MB/s"
        print(f"\r🔄 {progress}", end="")
        await asyncio.sleep(0.05)
    
    print("\n✅ 데모 테스트 완료!")
    print("📊 시뮬레이션 결과:")
    print("   - 처리 시간: 5.0초")
    print("   - 메모리 피크: 500MB")
    print("   - 평균 속도: 2.5MB/s")
    print("   - 성공률: 100%")

async def main():
    parser = argparse.ArgumentParser(description="고용량 파일 실전 테스트 실행기")
    parser.add_argument("--full", action="store_true", help="전체 테스트 실행 (1시간 비디오)")
    parser.add_argument("--demo", action="store_true", help="데모 모드 (시뮬레이션)")
    parser.add_argument("--output", type=str, help="출력 디렉토리 지정")
    
    args = parser.parse_args()
    
    print("🎯 대용량 파일 실전 테스트 실행기 v2.1")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        if args.demo:
            await run_demo_test()
            
        elif args.full:
            print("🚀 전체 테스트 모드 (1시간 비디오 + 30개 이미지)")
            print("⚠️  이 테스트는 많은 시간과 리소스가 필요합니다.")
            
            # 확인 요청
            confirm = input("계속하시겠습니까? (y/N): ").strip().lower()
            if confirm != 'y':
                print("테스트를 취소했습니다.")
                return
            
            tester = LargeFileRealTest(args.output)
            result = await tester.run_full_test()
            
            if result.get("test_summary", {}).get("overall_success"):
                print("🎉 전체 테스트 성공!")
            else:
                print("⚠️ 테스트 중 일부 문제가 발생했습니다.")
                
        else:
            print("⚡ 빠른 테스트 모드 (5분 비디오 + 5개 이미지)")
            await run_quick_test()
        
        elapsed = time.time() - start_time
        print(f"\n⏱️ 총 실행 시간: {elapsed:.1f}초")
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자가 테스트를 중단했습니다.")
        
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")
        import traceback
        print("상세 오류:")
        traceback.print_exc()

if __name__ == "__main__":
    # 비동기 실행
    asyncio.run(main())
