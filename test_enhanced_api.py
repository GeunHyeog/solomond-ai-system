#!/usr/bin/env python3
"""
강화된 API 서버 v2.3 테스트
실제 분석 엔진 및 비디오 처리 기능 테스트
"""

import requests
import json
import os
import time

# API 서버 설정
API_BASE_URL = "http://localhost:8000/api/v23"

def test_system_status():
    """시스템 상태 확인"""
    print("=" * 50)
    print("[TEST] 시스템 상태 확인")
    
    try:
        response = requests.get(f"{API_BASE_URL}/system/status")
        
        if response.status_code == 200:
            data = response.json()
            print("[OK] 시스템 상태 조회 성공")
            print(f"  - 서버 상태: {data.get('server_status')}")
            print(f"  - 실제 분석 엔진: {data.get('real_analysis_available')}")
            print(f"  - 비디오 처리: {data.get('video_processing_available')}")
            print(f"  - MoviePy: {data.get('moviepy_available')}")
            print(f"  - FFmpeg: {data.get('ffmpeg_available')}")
            print(f"  - 지원 형식: {', '.join(data.get('supported_formats', []))}")
            print(f"  - 현재 부하: {data.get('current_load')}개 세션")
            return True
        else:
            print(f"❌ 시스템 상태 조회 실패: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ 연결 오류: {str(e)}")
        return False

def test_file_analysis():
    """파일 분석 API 테스트"""
    print("\n" + "=" * 50)
    print("[TEST] 파일 분석 API")
    
    # 테스트 파일 경로들
    test_files = [
        "test_files/test_image.png",
        "test_files/test_audio.wav"
    ]
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"⚠️ 테스트 파일 없음: {file_path}")
            continue
        
        print(f"\n[분석] {os.path.basename(file_path)}")
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
                data = {'language': 'ko', 'analysis_type': 'comprehensive'}
                
                response = requests.post(
                    f"{API_BASE_URL}/analyze/file",
                    files=files,
                    data=data,
                    timeout=60
                )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 파일 분석 성공")
                print(f"  - 세션 ID: {result.get('session_id')}")
                print(f"  - 파일 타입: {result.get('file_type')}")
                print(f"  - 처리 시간: {result.get('processing_time')}초")
                print(f"  - 신뢰도: {result.get('confidence_score')}")
                print(f"  - 요약: {result.get('summary', '')[:100]}...")
                if result.get('keywords'):
                    print(f"  - 키워드: {', '.join(result['keywords'][:5])}")
            else:
                print(f"❌ 파일 분석 실패: {response.status_code}")
                print(response.text[:200])
                
        except Exception as e:
            print(f"❌ 분석 오류: {str(e)}")

def test_video_info():
    """비디오 정보 조회 API 테스트"""
    print("\n" + "=" * 50)
    print("[TEST] 비디오 정보 조회 API")
    
    # 시스템에서 일반적인 비디오 파일 찾기
    common_video_paths = [
        "C:/Windows/Performance/WinSAT/winsat.wmv",
        "test_files/test_video.mp4"
    ]
    
    for video_path in common_video_paths:
        if not os.path.exists(video_path):
            continue
        
        print(f"\n[정보 조회] {os.path.basename(video_path)}")
        
        try:
            response = requests.get(
                f"{API_BASE_URL}/video/info",
                params={'video_url': video_path},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 비디오 정보 조회 성공")
                
                video_info = result.get('video_info', {})
                if video_info:
                    print(f"  - 파일 크기: {video_info.get('file_size_mb')}MB")
                    print(f"  - 지속시간: {video_info.get('duration_formatted')}")
                    print(f"  - 형식: {video_info.get('format_name')}")
                    
                    if video_info.get('has_video'):
                        vinfo = video_info.get('video_info', {})
                        print(f"  - 해상도: {vinfo.get('width')}x{vinfo.get('height')}")
                        print(f"  - FPS: {vinfo.get('fps')}")
                        print(f"  - 품질: {vinfo.get('quality')}")
                    
                    if video_info.get('has_audio'):
                        ainfo = video_info.get('audio_info', {})
                        print(f"  - 오디오: {ainfo.get('channels')}ch @ {ainfo.get('sample_rate')}Hz")
                
                return True
            else:
                print(f"❌ 비디오 정보 조회 실패: {response.status_code}")
                print(response.text[:200])
                
        except Exception as e:
            print(f"❌ 조회 오류: {str(e)}")
    
    print("⚠️ 테스트용 비디오 파일이 없습니다.")
    return False

def test_api_documentation():
    """API 문서 접근 테스트"""
    print("\n" + "=" * 50)
    print("[TEST] API 문서 접근")
    
    try:
        response = requests.get("http://localhost:8000/api/v23/docs")
        
        if response.status_code == 200:
            print("✅ API 문서 접근 성공")
            print("📖 문서 URL: http://localhost:8000/api/v23/docs")
            return True
        else:
            print(f"❌ API 문서 접근 실패: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 문서 접근 오류: {str(e)}")
        return False

def main():
    """메인 테스트 실행"""
    print("[START] 솔로몬드 AI API v2.3 강화 기능 테스트")
    print("=" * 60)
    
    # 테스트 결과 추적
    results = {
        "system_status": False,
        "file_analysis": False,
        "video_info": False,
        "api_docs": False
    }
    
    # 1. 시스템 상태 확인
    results["system_status"] = test_system_status()
    
    # 2. 파일 분석 테스트
    if results["system_status"]:
        test_file_analysis()
    
    # 3. 비디오 정보 조회
    if results["system_status"]:
        results["video_info"] = test_video_info()
    
    # 4. API 문서 접근
    results["api_docs"] = test_api_documentation()
    
    # 최종 결과
    print("\n" + "=" * 60)
    print("[SUMMARY] 테스트 결과 요약")
    print("=" * 60)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for test_name, success in results.items():
        status = "[OK] 성공" if success else "[FAIL] 실패"
        print(f"  {test_name}: {status}")
    
    print(f"\n[RESULT] 전체 성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("[SUCCESS] 모든 테스트 성공! API 서버가 완전히 작동합니다.")
    elif success_count > 0:
        print("[PARTIAL] 일부 테스트 성공. 시스템이 부분적으로 작동합니다.")
    else:
        print("[FAILED] 모든 테스트 실패. API 서버 연결을 확인하세요.")
    
    print("\n[ENDPOINTS] API 엔드포인트 요약:")
    print("  - POST /api/v23/analyze/file - 파일 분석")
    print("  - POST /api/v23/analyze/video - 비디오 분석")
    print("  - GET /api/v23/system/status - 시스템 상태")
    print("  - GET /api/v23/video/info - 비디오 정보")
    print("  - GET /api/v23/docs - API 문서")

if __name__ == "__main__":
    main()