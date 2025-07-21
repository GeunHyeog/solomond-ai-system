#!/usr/bin/env python3
"""
YouTube 처리 모듈 테스트
"""

import os
import sys
sys.path.append('.')

from core.youtube_processor import youtube_processor

def test_youtube_processor():
    """YouTube 처리 모듈 테스트"""
    
    print("[INFO] YouTube 처리 모듈 테스트 시작")
    print("=" * 50)
    
    # 설치 가이드 확인
    guide = youtube_processor.get_installation_guide()
    print(f"[INFO] 시스템 사용 가능: {guide['available']}")
    
    if guide['missing_packages']:
        print("[WARNING] 누락된 패키지:")
        for pkg in guide['missing_packages']:
            print(f"  - {pkg['package']}: {pkg['command']} ({pkg['purpose']})")
        print(f"[INFO] 전체 설치: {guide['install_all']}")
        print(f"[INFO] 추가 요구사항: {guide['ffmpeg_required']}")
    else:
        print("[SUCCESS] 모든 필요 패키지 설치됨")
    
    print()
    
    # URL 패턴 테스트
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ", 
        "invalid_url",
        "https://www.youtube.com/watch?v=9bZkp7q19f0"  # PSY - GANGNAM STYLE
    ]
    
    print("[TEST] YouTube URL 검증 테스트:")
    for url in test_urls:
        is_youtube = youtube_processor.is_youtube_url(url)
        video_id = youtube_processor.extract_video_id(url)
        print(f"  {url}")
        print(f"    - YouTube URL: {is_youtube}")
        print(f"    - Video ID: {video_id}")
        print()
    
    # 실제 YouTube 영상 정보 테스트 (간단한 영상)
    if guide['available']:
        print("[TEST] YouTube 영상 정보 조회 테스트:")
        test_video_url = "https://www.youtube.com/watch?v=9bZkp7q19f0"  # PSY - GANGNAM STYLE (짧고 유명한 영상)
        
        print(f"[INFO] 테스트 영상: {test_video_url}")
        
        # 영상 정보만 조회 (다운로드 X)
        info_result = youtube_processor.get_video_info(test_video_url)
        
        if info_result['status'] == 'success':
            print("[SUCCESS] 영상 정보 조회 성공")
            print(f"  - 제목: {info_result['title']}")
            print(f"  - 업로더: {info_result['uploader']}")
            print(f"  - 지속시간: {info_result['duration_formatted']}")
            print(f"  - 조회수: {info_result.get('view_count', 'N/A'):,}")
            print(f"  - 자막 언어: {', '.join(info_result.get('subtitles', []))}")
            print(f"  - 자동 자막: {', '.join(info_result.get('automatic_captions', []))}")
        else:
            print(f"[ERROR] 영상 정보 조회 실패: {info_result.get('error', 'Unknown')}")
        
        print()
        
        # 사용자 확인 후 다운로드 테스트
        print("[OPTION] 실제 오디오 다운로드 테스트")
        print("주의: 이 테스트는 실제로 YouTube에서 오디오를 다운로드합니다.")
        print("계속하려면 'y'를 입력하세요 (기본값: n):")
        
        # 자동 테스트를 위해 일단 스킵
        user_input = "n"  # input().strip().lower()
        
        if user_input == 'y':
            print("[TEST] 오디오 다운로드 테스트...")
            download_result = youtube_processor.download_audio(test_video_url)
            
            if download_result['status'] == 'success':
                print("[SUCCESS] 오디오 다운로드 성공")
                print(f"  - 파일: {download_result['audio_file']}")
                print(f"  - 크기: {download_result['file_size_mb']}MB")
                print(f"  - 처리 시간: {download_result['processing_time']}초")
                
                # 파일 존재 확인
                if os.path.exists(download_result['audio_file']):
                    print("  - 파일 존재: ✅")
                else:
                    print("  - 파일 존재: ❌")
            else:
                print(f"[ERROR] 오디오 다운로드 실패: {download_result.get('error', 'Unknown')}")
        else:
            print("[INFO] 오디오 다운로드 테스트 스킵됨")
    
    else:
        print("[INFO] 필요 패키지가 설치되지 않아 실제 테스트를 건너뜁니다.")

if __name__ == "__main__":
    test_youtube_processor()