#!/usr/bin/env python3
"""
PyAudio 설치 가이드 및 자동 설치 스크립트
"""

import subprocess
import sys
import platform
import os

def check_pyaudio():
    """PyAudio 설치 상태 확인"""
    
    try:
        import pyaudio
        print("✓ PyAudio 이미 설치됨")
        print(f"  버전: {pyaudio.__version__ if hasattr(pyaudio, '__version__') else '확인불가'}")
        return True
    except ImportError:
        print("✗ PyAudio 설치되지 않음")
        return False

def install_pyaudio_windows():
    """Windows에서 PyAudio 설치"""
    
    print("\n=== Windows PyAudio 설치 ===")
    
    system = platform.architecture()[0]
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    print(f"시스템 아키텍처: {system}")
    print(f"Python 버전: {python_version}")
    
    # 방법 1: pip를 통한 일반 설치 시도
    print("\n방법 1: pip 설치 시도")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "pyaudio"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ pip를 통한 PyAudio 설치 성공")
            return True
        else:
            print(f"✗ pip 설치 실패: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✗ pip 설치 시간 초과")
    except Exception as e:
        print(f"✗ pip 설치 오류: {str(e)}")
    
    # 방법 2: 미리 컴파일된 wheel 사용
    print("\n방법 2: 미리 컴파일된 wheel 설치")
    
    wheel_urls = {
        "3.9": {
            "64bit": "https://files.pythonhosted.org/packages/91/7e/c10522028e67bb78b50bf4be12e3b6bbdfaf09ca7dd64ba58b5c6cb73e0001/PyAudio-0.2.11-cp39-cp39-win_amd64.whl",
            "32bit": "https://files.pythonhosted.org/packages/19/9c/9e3b4ad7cd90e83b32bd37db4d6be4db5ebdcdf01eaef89b2be2a2c95cde/PyAudio-0.2.11-cp39-cp39-win32.whl"
        },
        "3.10": {
            "64bit": "https://files.pythonhosted.org/packages/0e/6a/cca3eb11b0ab0c44dfc7b66c8e4b9a36e6b5aab9bb4b8e5b3a5e67e74e26a/PyAudio-0.2.11-cp310-cp310-win_amd64.whl",
            "32bit": "https://files.pythonhosted.org/packages/7e/31/93c5ce5f5f6c7b3d09e13f13a159568c0cc0d5b7e4b1f1e8e7e8a5a2a7c8/PyAudio-0.2.11-cp310-cp310-win32.whl"
        },
        "3.11": {
            "64bit": "https://files.pythonhosted.org/packages/b3/a9/6f4e3dc48ff6e5af0c49c8c96d7b4e2e5d1ee8fcf2a1b9b7c3b0d8d1c0ae/PyAudio-0.2.11-cp311-cp311-win_amd64.whl"
        },
        "3.12": {
            "64bit": "https://files.pythonhosted.org/packages/c7/68/9da5b9d2e0e0b8b55e2a6e0b5b7e0f5e4e6e5b5d5c5b5e5b5e5b5e5b5/PyAudio-0.2.14-cp312-cp312-win_amd64.whl"
        },
        "3.13": {
            "64bit": "https://files.pythonhosted.org/packages/latest/PyAudio-0.2.14-cp313-cp313-win_amd64.whl"
        }
    }
    
    arch = "64bit" if system == "64bit" else "32bit"
    
    if python_version in wheel_urls and arch in wheel_urls[python_version]:
        wheel_url = wheel_urls[python_version][arch]
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", wheel_url
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("✓ wheel을 통한 PyAudio 설치 성공")
                return True
            else:
                print(f"✗ wheel 설치 실패: {result.stderr}")
        except Exception as e:
            print(f"✗ wheel 설치 오류: {str(e)}")
    
    # 방법 3: 수동 설치 안내
    print("\n방법 3: 수동 설치 안내")
    print("다음 단계를 수동으로 진행하세요:")
    print("1. https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio 방문")
    print(f"2. Python {python_version} {arch} 버전에 맞는 .whl 파일 다운로드")
    print("3. 다운로드 폴더에서 다음 명령어 실행:")
    print("   pip install PyAudio-0.2.11-cp[version]-cp[version]-win_amd64.whl")
    
    return False

def install_dependencies():
    """관련 의존성 설치"""
    
    print("\n=== 관련 의존성 설치 ===")
    
    dependencies = [
        "numpy",
        "scipy", 
        "librosa",  # 오디오 처리
        "soundfile",  # 오디오 파일 읽기/쓰기
        "webrtcvad"  # 음성 활동 감지
    ]
    
    for dep in dependencies:
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"✓ {dep} 설치 완료")
            else:
                print(f"✗ {dep} 설치 실패")
                
        except Exception as e:
            print(f"✗ {dep} 설치 오류: {str(e)}")

def test_audio_functionality():
    """오디오 기능 테스트"""
    
    print("\n=== 오디오 기능 테스트 ===")
    
    try:
        import pyaudio
        
        # PyAudio 인스턴스 생성
        p = pyaudio.PyAudio()
        
        # 오디오 장치 정보 확인
        device_count = p.get_device_count()
        print(f"감지된 오디오 장치: {device_count}개")
        
        # 기본 입력 장치 확인
        try:
            default_input = p.get_default_input_device_info()
            print(f"기본 입력 장치: {default_input['name']}")
            print(f"  채널: {default_input['maxInputChannels']}")
            print(f"  샘플레이트: {default_input['defaultSampleRate']}")
        except Exception as e:
            print(f"기본 입력 장치 없음: {str(e)}")
        
        # PyAudio 종료
        p.terminate()
        
        print("✓ PyAudio 기본 기능 테스트 통과")
        return True
        
    except Exception as e:
        print(f"✗ PyAudio 테스트 실패: {str(e)}")
        return False

def main():
    """메인 설치 프로세스"""
    
    print("PyAudio 설치 및 테스트 도구")
    print("=" * 40)
    
    # 1. 현재 상태 확인
    if check_pyaudio():
        print("\nPyAudio가 이미 설치되어 있습니다.")
        
        # 기능 테스트
        if test_audio_functionality():
            print("\n✓ 모든 기능이 정상 작동합니다!")
            return
        else:
            print("\n⚠️ PyAudio는 설치되어 있지만 제대로 작동하지 않습니다.")
    
    # 2. 운영체제별 설치
    system = platform.system()
    
    if system == "Windows":
        success = install_pyaudio_windows()
    elif system == "Linux":
        print("\nLinux 시스템에서는 다음 명령어를 실행하세요:")
        print("sudo apt-get install portaudio19-dev")
        print("pip install pyaudio")
        success = False
    elif system == "Darwin":  # macOS
        print("\nmacOS 시스템에서는 다음 명령어를 실행하세요:")
        print("brew install portaudio")
        print("pip install pyaudio")
        success = False
    else:
        print(f"\n지원하지 않는 운영체제: {system}")
        success = False
    
    # 3. 설치 후 테스트
    if success:
        print("\n설치가 완료되었습니다. 기능을 테스트합니다...")
        if test_audio_functionality():
            print("\n🎉 PyAudio 설치 및 테스트 완료!")
        else:
            print("\n❌ 설치는 완료되었지만 기능 테스트 실패")
    
    # 4. 의존성 설치
    print("\n관련 오디오 처리 라이브러리를 설치합니다...")
    install_dependencies()
    
    print("\n" + "=" * 40)
    print("설치 과정이 완료되었습니다.")
    print("실시간 스트리밍 테스트를 실행하려면:")
    print("python test_realtime_streaming.py")

if __name__ == "__main__":
    main()