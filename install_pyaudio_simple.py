#!/usr/bin/env python3
"""
간단한 PyAudio 설치 스크립트 (이모지 제거)
"""

import subprocess
import sys
import platform

def check_pyaudio():
    """PyAudio 설치 상태 확인"""
    
    try:
        import pyaudio
        print("PyAudio 이미 설치됨")
        return True
    except ImportError:
        print("PyAudio 설치되지 않음")
        return False

def install_pyaudio():
    """PyAudio 설치 시도"""
    
    print("\n=== PyAudio 설치 시작 ===")
    
    # 방법 1: 일반 pip 설치
    print("pip를 통한 설치 시도...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "pyaudio"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("pip를 통한 PyAudio 설치 성공")
            return True
        else:
            print(f"pip 설치 실패: {result.stderr}")
    except Exception as e:
        print(f"pip 설치 오류: {str(e)}")
    
    # 방법 2: Microsoft Visual C++ 필요 안내
    print("\n수동 설치가 필요할 수 있습니다:")
    print("1. Microsoft Visual C++ 14.0 설치")
    print("2. 또는 미리 컴파일된 wheel 파일 사용")
    
    return False

def install_audio_dependencies():
    """오디오 관련 의존성 설치"""
    
    print("\n=== 오디오 관련 라이브러리 설치 ===")
    
    dependencies = [
        "numpy",
        "soundfile",
        "librosa"
    ]
    
    for dep in dependencies:
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"{dep} 설치 완료")
            else:
                print(f"{dep} 설치 실패")
                
        except Exception as e:
            print(f"{dep} 설치 오류: {str(e)}")

def main():
    """메인 함수"""
    
    print("PyAudio 설치 도구")
    print("=" * 30)
    
    # 현재 상태 확인
    if check_pyaudio():
        print("\nPyAudio가 이미 설치되어 있습니다.")
        
        # 간단한 테스트
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            print(f"오디오 장치: {device_count}개 감지")
            p.terminate()
            print("PyAudio 정상 작동 확인")
        except Exception as e:
            print(f"PyAudio 테스트 실패: {str(e)}")
            
    else:
        # 설치 시도
        success = install_pyaudio()
        
        if success:
            print("\nPyAudio 설치 완료")
            # 재테스트
            if check_pyaudio():
                print("설치 성공 확인")
            else:
                print("설치 후에도 import 실패")
        else:
            print("\nPyAudio 자동 설치 실패")
            print("수동 설치가 필요합니다.")
    
    # 의존성 설치
    install_audio_dependencies()
    
    print("\n설치 과정 완료")
    print("실시간 스트리밍 테스트: python test_realtime_streaming.py")

if __name__ == "__main__":
    main()