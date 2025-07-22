#!/usr/bin/env python3
"""
YouTube 분석 기능 테스트
"""

import os
import sys
sys.path.append('.')

from core.real_analysis_engine import analyze_file_real

def test_youtube_analysis():
    """YouTube 분석 기능 테스트"""
    
    print("[INFO] YouTube 분석 기능 테스트 시작")
    print("=" * 50)
    
    # 테스트할 YouTube URL (짧은 영상으로 선택)
    test_urls = [
        "https://www.youtube.com/watch?v=9bZkp7q19f0",  # PSY - GANGNAM STYLE (유명하고 짧음)
        "https://youtu.be/dQw4w9WgXcQ"  # Rick Roll (짧음)
    ]
    
    print("[WARNING] 이 테스트는 실제로 YouTube에서 영상을 다운로드하고 분석합니다.")
    print("네트워크 사용량이 발생할 수 있습니다.")
    print()
    
    for i, url in enumerate(test_urls, 1):
        print(f"[TEST {i}] YouTube 영상 분석: {url}")
        
        try:
            # YouTube 분석 실행 (정보 조회만, 다운로드는 스킵)
            print("[INFO] 영상 정보 조회 중...")
            
            # 일단 정보 조회만 테스트
            from core.youtube_processor import youtube_processor
            
            info_result = youtube_processor.get_video_info(url)
            
            if info_result['status'] == 'success':
                print("[SUCCESS] 영상 정보 조회 성공")
                print(f"  - 제목: {info_result['title']}")
                print(f"  - 지속시간: {info_result['duration_formatted']}")
                print(f"  - 업로더: {info_result['uploader']}")
                print(f"  - 조회수: {info_result.get('view_count', 'N/A'):,}")
                
                # 실제 분석 수행 여부 확인
                print()
                print("실제 오디오 다운로드 및 STT 분석을 수행하시겠습니까?")
                print("주의: 시간이 오래 걸리고 네트워크를 사용합니다. (y/N): ", end="")
                
                # 자동 테스트를 위해 'n' 선택
                user_choice = "n"  # input().strip().lower()
                print("n")
                
                if user_choice == 'y':
                    print("[INFO] 전체 YouTube 분석 시작...")
                    
                    result = analyze_file_real(url, "youtube", "ko")
                    
                    if result['status'] == 'success':
                        print("[SUCCESS] YouTube 분석 완료!")
                        print(f"  - 처리 시간: {result['processing_time']}초")
                        print(f"  - 다운로드 크기: {result['download_info']['file_size_mb']}MB")
                        
                        # 오디오 분석 결과
                        audio_analysis = result.get('audio_analysis', {})
                        if audio_analysis.get('status') == 'success':
                            print(f"  - STT 결과: {audio_analysis.get('transcription', 'N/A')[:100]}...")
                            print(f"  - 신뢰도: {audio_analysis.get('confidence', 'N/A')}")
                        
                        # 통합 분석 결과
                        combined = result.get('combined_analysis', {})
                        if combined:
                            print(f"  - 콘텐츠 타입: {combined.get('content_type', 'N/A')}")
                            print(f"  - 주얼리 키워드: {', '.join(combined.get('jewelry_keywords', []))}")
                            print(f"  - 통합 요약: {combined.get('integrated_summary', 'N/A')[:100]}...")
                    
                    else:
                        print(f"[ERROR] YouTube 분석 실패: {result.get('error', 'Unknown')}")
                
                else:
                    print("[INFO] 전체 분석 건너뜀 - 정보 조회만 완료")
            
            else:
                print(f"[ERROR] 영상 정보 조회 실패: {info_result.get('error', 'Unknown')}")
        
        except Exception as e:
            print(f"[ERROR] 테스트 중 오류: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 30)
        print()
    
    print("[INFO] YouTube 분석 테스트 완료")
    print()
    print("🎯 다음 단계:")
    print("1. Streamlit UI에 YouTube URL 입력 기능 추가")
    print("2. 실시간 다운로드 진행률 표시")
    print("3. YouTube 분석 결과 시각화")

if __name__ == "__main__":
    test_youtube_analysis()