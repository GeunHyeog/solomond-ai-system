#!/usr/bin/env python3
"""
대용량 비디오 처리 모듈 테스트
"""

import os
import sys
sys.path.append('.')

from core.large_video_processor import large_video_processor

def test_large_video_processor():
    """대용량 비디오 처리 모듈 테스트"""
    
    print("[INFO] 대용량 비디오 처리 모듈 테스트 시작")
    print("=" * 50)
    
    # 설치 가이드 확인
    guide = large_video_processor.get_installation_guide()
    print(f"[INFO] 시스템 사용 가능: {guide['available']}")
    print(f"[INFO] FFmpeg 사용 가능: {guide['ffmpeg_available']}")
    print(f"[INFO] 지원 형식: {', '.join(guide['supported_formats'])}")
    print(f"[INFO] 최대 파일 크기: {guide['max_file_size_gb']}GB")
    
    if guide['missing_packages']:
        print("\n[WARNING] 누락된 패키지:")
        for pkg in guide['missing_packages']:
            print(f"  - {pkg['package']}: {pkg['command']} ({pkg['purpose']})")
        print(f"\n[INFO] 전체 설치: {guide['install_all']}")
        
        if 'ffmpeg_install' in guide:
            print("\n[INFO] FFmpeg 설치:")
            print(f"  - Windows: {guide['ffmpeg_install']['windows']}")
            print(f"  - 참고: {guide['ffmpeg_install']['note']}")
    else:
        print("\n[SUCCESS] 모든 필요 패키지 설치됨")
    
    print()
    
    # 테스트 비디오 파일 확인
    test_video_paths = [
        "test_files/test_video.mov",
        "test_files/test_video.mp4",
        "test_files/sample.mov"
    ]
    
    # 실제 존재하는 비디오 파일 찾기
    available_videos = []
    for path in test_video_paths:
        if os.path.exists(path):
            available_videos.append(path)
    
    # 시스템에서 일반적인 비디오 파일 찾기 (예시)
    common_paths = [
        "C:/Users/Public/Videos",
        "C:/Windows/Performance/WinSAT"
    ]
    
    for common_path in common_paths:
        if os.path.exists(common_path):
            try:
                for file in os.listdir(common_path):
                    if any(file.lower().endswith(ext) for ext in ['.mov', '.mp4', '.avi']):
                        full_path = os.path.join(common_path, file)
                        if os.path.getsize(full_path) > 1024:  # 1KB 이상
                            available_videos.append(full_path)
                            break  # 첫 번째 파일만
            except:
                pass
    
    if not available_videos:
        print("[INFO] 테스트할 비디오 파일이 없습니다.")
        print("[INFO] 다음 방법으로 테스트 파일을 준비할 수 있습니다:")
        print("  1. test_files/ 폴더에 .mov, .mp4, .avi 파일 복사")
        print("  2. 샘플 비디오 파일 다운로드")
        print()
        
        # 가상 파일로 정보 조회 테스트
        print("[TEST] 가상 파일 정보 조회 테스트:")
        fake_path = "test_files/nonexistent.mov"
        info_result = large_video_processor.get_video_info(fake_path)
        
        if info_result['status'] == 'error':
            print(f"[EXPECTED] 파일 없음 오류: {info_result['error']}")
        
        return
    
    # 실제 비디오 파일 테스트
    for i, video_path in enumerate(available_videos[:2], 1):  # 최대 2개 파일만 테스트
        print(f"\n[TEST {i}] 비디오 파일 분석: {os.path.basename(video_path)}")
        print(f"경로: {video_path}")
        
        # 1. 비디오 정보 조회
        print("\n[1] 비디오 정보 조회...")
        info_result = large_video_processor.get_video_info(video_path)
        
        if info_result['status'] == 'success':
            print("[SUCCESS] 비디오 정보 조회 성공")
            print(f"  - 파일 크기: {info_result['file_size_mb']}MB")
            print(f"  - 지속시간: {info_result['duration_formatted']}")
            print(f"  - 형식: {info_result['format_name']}")
            
            if info_result.get('has_video'):
                video_info = info_result['video_info']
                print(f"  - 비디오: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.1f}fps")
                print(f"  - 코덱: {video_info['codec']}")
                print(f"  - 품질: {video_info['quality']}")
            
            if info_result.get('has_audio'):
                audio_info = info_result['audio_info']
                print(f"  - 오디오: {audio_info['channels']}ch @ {audio_info['sample_rate']}Hz")
                print(f"  - 오디오 코덱: {audio_info['codec']}")
            
            if 'warning' in info_result:
                print(f"  - 경고: {info_result['warning']}")
            
            # 2. 오디오 추출 테스트 (파일이 너무 크지 않은 경우)
            file_size_mb = info_result['file_size_mb']
            duration = info_result.get('duration', 0)
            
            if file_size_mb <= 100 and duration <= 300:  # 100MB 이하, 5분 이하
                print("\n[2] 오디오 추출 테스트...")
                
                if info_result.get('has_audio'):
                    audio_result = large_video_processor.extract_audio_from_video(video_path)
                    
                    if audio_result['status'] == 'success':
                        print("[SUCCESS] 오디오 추출 성공")
                        print(f"  - 추출 파일: {os.path.basename(audio_result['audio_file'])}")
                        print(f"  - 오디오 크기: {audio_result['audio_size_mb']}MB")
                        print(f"  - 처리 시간: {audio_result['processing_time']}초")
                        print(f"  - 오디오 정보: {audio_result['audio_info']['format']}")
                        
                        # 생성된 오디오 파일 존재 확인
                        if os.path.exists(audio_result['audio_file']):
                            print("  - 파일 생성 확인: ✅")
                            
                            # 임시 파일 정리
                            try:
                                os.unlink(audio_result['audio_file'])
                                print("  - 임시 파일 정리: ✅")
                            except:
                                print("  - 임시 파일 정리: ❌")
                        else:
                            print("  - 파일 생성 확인: ❌")
                    
                    else:
                        print(f"[ERROR] 오디오 추출 실패: {audio_result.get('error', 'Unknown')}")
                else:
                    print("[INFO] 오디오 트랙이 없어 추출 테스트 생략")
            
            else:
                print(f"\n[INFO] 파일이 너무 큽니다 ({file_size_mb}MB, {duration}초). 오디오 추출 테스트 생략")
                print("       실제 사용 시에는 스트리밍 처리를 권장합니다.")
        
        elif info_result['status'] == 'partial_success':
            print("[WARNING] 부분적 정보만 조회 성공")
            print(f"  - 파일 크기: {info_result['file_size_mb']}MB")
            print(f"  - 형식: {info_result['file_format']}")
            print(f"  - 경고: {info_result.get('warning', 'N/A')}")
        
        else:
            print(f"[ERROR] 비디오 정보 조회 실패: {info_result.get('error', 'Unknown')}")
        
        print("-" * 30)
    
    # 처리 통계 출력
    print("\n[INFO] 처리 통계:")
    stats = large_video_processor.get_processing_stats()
    print(f"  - 총 처리 파일: {stats['total_files']}개")
    print(f"  - 성공률: {stats.get('success_rate', 0)}%")
    
    if stats['total_files'] > 0:
        print(f"  - 평균 처리 시간: {stats.get('average_processing_time', 0)}초")
        print(f"  - 최대 파일 크기: {stats.get('largest_file_size_mb', 0)}MB")
    
    print("\n" + "="*50)
    print("[INFO] 대용량 비디오 처리 테스트 완료")
    print()
    print("MOV 파일 지원 상태:")
    if guide['available'] and guide['ffmpeg_available']:
        print("✅ 완전 지원 - 대용량 MOV 파일 처리 가능")
        print("   - 비디오 정보 조회")
        print("   - 오디오 추출")
        print("   - 스트리밍 처리")
        print("   - 메모리 최적화")
    elif guide['ffmpeg_available']:
        print("⚠️ 부분 지원 - FFmpeg 사용 가능하나 일부 패키지 누락")
    else:
        print("❌ 제한적 지원 - FFmpeg 설치 필요")
        print("   설치 후 완전한 MOV 처리 가능")

if __name__ == "__main__":
    test_large_video_processor()