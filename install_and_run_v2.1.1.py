#!/usr/bin/env python3
"""
솔로몬드 AI v2.1.1 - 자동 설치 및 실행 스크립트
필요한 패키지를 자동으로 설치하고 시스템을 실행합니다.

사용법:
python install_and_run_v2.1.1.py

작성자: 전근혁 (솔로몬드 대표)
날짜: 2025.07.11
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def print_header():
    """헤더 출력"""
    print("=" * 80)
    print("💎 솔로몬드 AI v2.1.1 - 멀티모달 통합 분석 시스템")
    print("🎯 자동 설치 및 실행 스크립트")
    print("=" * 80)
    print(f"🐍 Python 버전: {sys.version}")
    print(f"📁 현재 디렉토리: {os.getcwd()}")
    print("=" * 80)

def check_python_version():
    """Python 버전 확인"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        print(f"현재 버전: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python 버전 확인: {sys.version_info.major}.{sys.version_info.minor}")

def install_package(package):
    """개별 패키지 설치"""
    print(f"📦 {package} 설치 중...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        print(f"✅ {package} 설치 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package} 설치 실패: {e}")
        return False

def install_requirements():
    """필수 패키지들 설치"""
    print("\n🔧 솔로몬드 AI v2.1.1 필수 패키지 설치를 시작합니다...")
    
    # 기본 패키지 목록 (반드시 필요)
    essential_packages = [
        "streamlit>=1.28.0",
        "pandas>=1.5.0", 
        "numpy>=1.24.0",
        "plotly>=5.15.0"
    ]
    
    # 멀티모달 처리 패키지 (핵심 기능)
    multimodal_packages = [
        "openai-whisper",
        "torch",
        "torchaudio", 
        "opencv-python",
        "Pillow",
        "pytesseract",
        "moviepy",
        "yt-dlp"
    ]
    
    # 추가 기능 패키지
    optional_packages = [
        "PyPDF2",
        "python-docx", 
        "requests",
        "tqdm",
        "python-dateutil"
    ]
    
    # 1. 기본 패키지 설치
    print("\n📋 1단계: 기본 웹 UI 패키지 설치 중...")
    basic_failed = []
    for package in essential_packages:
        if not install_package(package):
            basic_failed.append(package)
    
    if basic_failed:
        print("❌ 기본 패키지 설치 실패. 시스템을 확인해주세요.")
        return False
    
    # 2. 멀티모달 패키지 설치 (시간이 오래 걸림)
    print("\n🎬 2단계: 멀티모달 처리 패키지 설치 중... (시간이 오래 걸릴 수 있습니다)")
    multimodal_failed = []
    
    for package in multimodal_packages:
        print(f"   현재: {package} (대용량 패키지일 수 있습니다)")
        if not install_package(package):
            multimodal_failed.append(package)
    
    # 3. 추가 기능 패키지 설치
    print("\n📄 3단계: 추가 기능 패키지 설치 중...")
    optional_failed = []
    
    for package in optional_packages:
        if not install_package(package):
            optional_failed.append(package)
    
    # 설치 결과 요약
    print("\n" + "=" * 60)
    print("📋 설치 결과 요약")
    print("=" * 60)
    
    total_success = len(essential_packages + multimodal_packages + optional_packages) - len(basic_failed + multimodal_failed + optional_failed)
    total_packages = len(essential_packages + multimodal_packages + optional_packages)
    
    print(f"✅ 성공: {total_success}/{total_packages}개 패키지")
    
    if basic_failed:
        print("🚨 기본 패키지 설치 실패 (치명적):")
        for pkg in basic_failed:
            print(f"   - {pkg}")
        return False
    
    if multimodal_failed:
        print("⚠️ 멀티모달 패키지 설치 실패:")
        for pkg in multimodal_failed:
            print(f"   - {pkg}")
        print("\n💡 실패한 패키지는 수동으로 설치해주세요:")
        for pkg in multimodal_failed:
            print(f"   pip install {pkg}")
    
    if optional_failed:
        print("ℹ️ 추가 기능 패키지 설치 실패 (선택적):")
        for pkg in optional_failed:
            print(f"   - {pkg}")
    
    success_rate = total_success / total_packages
    if success_rate >= 0.8:
        print("\n🎉 충분한 패키지가 설치되어 시스템 실행이 가능합니다!")
        return True
    else:
        print(f"\n⚠️ 설치 성공률이 낮습니다 ({success_rate:.1%}). 일부 기능이 제한될 수 있습니다.")
        return False

def check_system_requirements():
    """시스템 요구사항 확인"""
    print("\n🔍 시스템 요구사항 확인 중...")
    
    issues = []
    warnings = []
    
    # Tesseract OCR 확인
    try:
        result = subprocess.run(["tesseract", "--version"], capture_output=True, check=True, text=True)
        print("✅ Tesseract OCR 설치됨")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ Tesseract OCR가 설치되지 않았습니다.")
        print("   📥 설치 방법:")
        print("   • Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   • macOS: brew install tesseract")
        print("   • Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-kor")
        warnings.append("tesseract")
    
    # FFmpeg 확인
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, text=True)
        print("✅ FFmpeg 설치됨")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ FFmpeg가 설치되지 않았습니다.")
        print("   📥 설치 방법:")
        print("   • Windows: https://ffmpeg.org/download.html")
        print("   • macOS: brew install ffmpeg")  
        print("   • Ubuntu: sudo apt install ffmpeg")
        warnings.append("ffmpeg")
    
    # 메모리 확인 (간접적)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb >= 8:
            print(f"✅ 시스템 메모리: {memory_gb:.1f}GB (충분)")
        else:
            print(f"⚠️ 시스템 메모리: {memory_gb:.1f}GB (8GB 이상 권장)")
            warnings.append("memory")
    except ImportError:
        print("ℹ️ 메모리 정보를 확인할 수 없습니다 (psutil 없음)")
    
    if warnings:
        print(f"\n⚠️ {len(warnings)}개의 권장사항이 있습니다.")
        print("   완전한 기능을 위해 위의 프로그램들을 설치하는 것을 권장합니다.")
        print("   하지만 기본 기능은 사용 가능합니다.")
        return True
    else:
        print("✅ 모든 시스템 요구사항이 충족되었습니다!")
        return True

def test_imports():
    """핵심 라이브러리 import 테스트"""
    print("\n🧪 핵심 라이브러리 테스트 중...")
    
    tests = [
        ("streamlit", "Streamlit 웹 UI", True),
        ("pandas", "Pandas 데이터 처리", True),
        ("numpy", "NumPy 수치 계산", True),
        ("plotly", "Plotly 차트 생성", True),
        ("whisper", "Whisper STT", False),
        ("cv2", "OpenCV 이미지 처리", False),
        ("PIL", "Pillow 이미지 처리", False),
        ("yt_dlp", "YouTube 다운로더", False),
        ("moviepy.editor", "MoviePy 영상 처리", False)
    ]
    
    essential_failed = []
    optional_failed = []
    
    for module, description, is_essential in tests:
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError:
            print(f"❌ {description} 실패")
            if is_essential:
                essential_failed.append(module)
            else:
                optional_failed.append(module)
    
    print(f"\n📊 테스트 결과:")
    print(f"   필수 라이브러리: {4 - len(essential_failed)}/4 성공")
    print(f"   선택 라이브러리: {5 - len(optional_failed)}/5 성공")
    
    if essential_failed:
        print(f"🚨 필수 라이브러리 실패: {', '.join(essential_failed)}")
        return False
    elif optional_failed:
        print(f"⚠️ 일부 기능 제한: {', '.join(optional_failed)}")
        print("   기본 UI는 작동하지만 해당 기능들은 사용할 수 없습니다.")
        return True
    else:
        print("🎉 모든 라이브러리가 정상적으로 로드됩니다!")
        return True

def run_streamlit_app():
    """Streamlit 앱 실행"""
    script_path = Path(__file__).parent / "jewelry_stt_ui_v2.1.1.py"
    
    if not script_path.exists():
        # 대안 파일명들 시도
        alternatives = [
            "jewelry_stt_ui.py",
            "main.py",
            "app.py"
        ]
        
        for alt in alternatives:
            alt_path = Path(__file__).parent / alt
            if alt_path.exists():
                script_path = alt_path
                break
        else:
            print(f"❌ 실행할 수 있는 파일을 찾을 수 없습니다.")
            print("   다음 파일 중 하나가 필요합니다:")
            print("   - jewelry_stt_ui_v2.1.1.py")
            print("   - jewelry_stt_ui.py")
            print("   - main.py")
            return False
    
    print(f"\n🚀 솔로몬드 AI v2.1.1 실행 중...")
    print(f"📁 실행 파일: {script_path}")
    print("🌐 브라우저에서 자동으로 열립니다...")
    print("   또는 수동으로 http://localhost:8501 을 열어주세요.")
    print("⏹️ 종료하려면 Ctrl+C를 누르세요.")
    print("=" * 80)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(script_path),
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 솔로몬드 AI 시스템이 안전하게 종료되었습니다.")
        print("감사합니다!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 실행 오류: {e}")
        print("다음 방법으로 수동 실행을 시도해보세요:")
        print(f"   streamlit run {script_path}")
        return False
    
    return True

def main():
    """메인 실행 함수"""
    print_header()
    
    # 1. Python 버전 확인
    check_python_version()
    
    # 2. 사용자 확인
    print("\n🎯 솔로몬드 AI v2.1.1 설치를 시작하시겠습니까?")
    print("   📦 필요한 패키지들을 자동으로 설치합니다")
    print("   🔧 시스템 요구사항을 확인합니다") 
    print("   🌐 웹 UI를 실행합니다")
    print("   ⏱️ 예상 시간: 5-15분 (네트워크 속도에 따라)")
    
    response = input("\n계속하시겠습니까? (y/N): ").strip().lower()
    if response not in ['y', 'yes', '예', 'ㅇ']:
        print("설치가 취소되었습니다.")
        return
    
    start_time = time.time()
    
    # 3. 패키지 설치
    print("\n" + "🔄" * 20 + " 설치 시작 " + "🔄" * 20)
    install_success = install_requirements()
    
    # 4. 시스템 요구사항 확인  
    system_ok = check_system_requirements()
    
    # 5. Import 테스트
    import_ok = test_imports()
    
    install_time = time.time() - start_time
    
    # 6. 결과 요약
    print("\n" + "=" * 60)
    print("📋 설치 완료 보고서")
    print("=" * 60)
    print(f"📦 패키지 설치: {'✅ 성공' if install_success else '⚠️ 일부 실패'}")
    print(f"🔧 시스템 구성: {'✅ 완료' if system_ok else '⚠️ 일부 누락'}")
    print(f"🧪 라이브러리 테스트: {'✅ 성공' if import_ok else '❌ 실패'}")
    print(f"⏱️ 총 설치 시간: {install_time:.1f}초")
    
    # 7. 실행 여부 결정
    if import_ok:
        print("\n🎉 솔로몬드 AI v2.1.1 설치가 완료되었습니다!")
        print("✅ pie_chart 오류 해결")
        print("✅ 실제 멀티모달 처리 엔진 구현")
        print("✅ Whisper STT, OCR, 유튜브 다운로드 지원")
        
        auto_run = input("\n지금 솔로몬드 AI를 실행하시겠습니까? (Y/n): ").strip().lower()
        if auto_run in ['', 'y', 'yes', '예', 'ㅇ']:
            run_streamlit_app()
        else:
            print("\n📝 수동 실행 방법:")
            print("   streamlit run jewelry_stt_ui_v2.1.1.py")
            print("   또는")
            print("   python jewelry_stt_ui_v2.1.1.py")
    else:
        print("\n❌ 일부 필수 라이브러리에 문제가 있습니다.")
        print("위의 오류를 해결한 후 다시 시도해주세요.")
        
        print("\n🔧 해결 방법:")
        print("1. Python 버전 확인 (3.8 이상 필요)")
        print("2. 인터넷 연결 확인")
        print("3. 관리자 권한으로 실행")
        print("4. 수동 설치: pip install streamlit pandas numpy plotly")
        
        print("\n📝 수동 실행 방법 (문제가 해결된 경우):")
        print("   streamlit run jewelry_stt_ui_v2.1.1.py")

if __name__ == "__main__":
    main()
