#!/usr/bin/env python3
"""
실시간 분석 기능 테스트
"""

import os
import sys
import time
sys.path.append('.')

from core.real_time_analyzer import real_time_analyzer

def test_real_time_analysis():
    """실시간 분석 기능 테스트"""
    
    print("[INFO] 실시간 분석 기능 테스트 시작")
    print("=" * 50)
    
    # 설치 가이드 확인
    guide = real_time_analyzer.get_installation_guide()
    print(f"[INFO] 시스템 사용 가능: {guide['available']}")
    
    if guide['missing_packages']:
        print("[WARNING] 누락된 패키지:")
        for pkg in guide['missing_packages']:
            print(f"  - {pkg['package']}: {pkg['command']} ({pkg['purpose']})")
        print(f"[INFO] 전체 설치: {guide['install_all']}")
        
        if guide.get('additional_notes'):
            print("[INFO] 추가 참고사항:")
            for note in guide['additional_notes']:
                print(f"  - {note}")
    else:
        print("[SUCCESS] 모든 필요 패키지 설치됨")
    
    print()
    
    # 오디오 장치 목록 확인
    print("[TEST] 오디오 장치 목록 조회:")
    devices_result = real_time_analyzer.get_audio_devices()
    
    if devices_result['status'] == 'success':
        print("[SUCCESS] 오디오 장치 조회 성공")
        devices = devices_result.get('devices', [])
        
        if devices:
            print(f"[INFO] 발견된 입력 장치: {len(devices)}개")
            for device in devices[:5]:  # 처음 5개만 표시
                print(f"  [{device['index']}] {device['name']} - {device['channels']}ch, {device['sample_rate']}Hz")
            
            if len(devices) > 5:
                print(f"  ... 및 {len(devices) - 5}개 추가 장치")
        else:
            print("[WARNING] 사용 가능한 입력 장치가 없습니다")
    else:
        print(f"[ERROR] 오디오 장치 조회 실패: {devices_result.get('error', 'Unknown')}")
    
    print()
    
    if not guide['available']:
        print("[INFO] 필요 패키지가 설치되지 않아 실제 테스트를 건너뜁니다.")
        print("[INFO] 설치 후 다시 테스트해보세요:")
        print(f"       {guide['install_all']}")
        return
    
    # 실시간 분석 시뮬레이션 테스트
    print("[OPTION] 실제 실시간 분석 테스트")
    print("주의: 이 테스트는 마이크를 사용하여 실제로 음성을 녹음합니다.")
    print("테스트를 진행하시겠습니까? (y/N): ", end="")
    
    # 자동 테스트를 위해 'n' 선택
    user_choice = "n"  # input().strip().lower()
    print("n")
    
    if user_choice == 'y':
        print("[INFO] 실시간 분석 테스트 시작...")
        
        # 분석 결과 콜백 함수 정의
        def analysis_callback(result):
            print(f"[CALLBACK] 분석 결과 수신:")
            if result.get('status') == 'success':
                transcription = result.get('transcription', 'N/A')
                confidence = result.get('confidence', 'N/A')
                timestamp = result.get('real_time_metadata', {}).get('timestamp', 'N/A')
                print(f"  - 시간: {timestamp}")
                print(f"  - 텍스트: {transcription}")
                print(f"  - 신뢰도: {confidence}")
            else:
                print(f"  - 오류: {result.get('error', 'Unknown')}")
            print()
        
        # 콜백 등록
        real_time_analyzer.add_analysis_callback(analysis_callback)
        
        try:
            # 실시간 분석 시작
            start_result = real_time_analyzer.start_real_time_analysis()
            
            if start_result['status'] == 'success':
                print("[SUCCESS] 실시간 분석 시작됨")
                print("[INFO] 5초 동안 테스트합니다. 마이크에 말해보세요...")
                
                # 5초 대기
                time.sleep(5)
                
                # 분석 중지
                stop_result = real_time_analyzer.stop_real_time_analysis()
                
                if stop_result['status'] == 'success':
                    print("[SUCCESS] 실시간 분석 중지됨")
                    
                    # 세션 통계 출력
                    stats = stop_result.get('session_stats', {})
                    print("[INFO] 세션 통계:")
                    print(f"  - 세션 시간: {stats.get('session_duration', 0)}초")
                    print(f"  - 총 청크: {stats.get('total_chunks', 0)}개")
                    print(f"  - 음성 청크: {stats.get('speech_chunks', 0)}개")
                    print(f"  - 음성 비율: {stats.get('speech_ratio', 0)}%")
                    print(f"  - 총 분석 시간: {stats.get('total_analysis_time', 0)}초")
                else:
                    print(f"[ERROR] 분석 중지 실패: {stop_result.get('error', 'Unknown')}")
            
            else:
                print(f"[ERROR] 실시간 분석 시작 실패: {start_result.get('error', 'Unknown')}")
                
        except KeyboardInterrupt:
            print("\n[INFO] 사용자 중단")
            real_time_analyzer.stop_real_time_analysis()
        
        except Exception as e:
            print(f"[ERROR] 테스트 중 오류: {e}")
            real_time_analyzer.stop_real_time_analysis()
        
        finally:
            # 콜백 제거
            real_time_analyzer.remove_analysis_callback(analysis_callback)
    
    else:
        print("[INFO] 실시간 분석 테스트 건너뜀")
    
    print()
    print("[INFO] 실시간 분석 테스트 완료")
    print()
    print("다음 단계:")
    print("1. Streamlit UI에 실시간 분석 인터페이스 추가")
    print("2. 웹소켓 기반 실시간 결과 표시")
    print("3. 음성 활동 감지 시각화")

if __name__ == "__main__":
    test_real_time_analysis()