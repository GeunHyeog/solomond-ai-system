#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 주얼리 AI 플랫폼 v2.1 통합 데모 실행기
원클릭 실행 스크립트

사용법:
python run_v21_integrated_demo.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_requirements():
    """필요 패키지 확인 및 설치"""
    print("📦 필요 패키지 확인 중...")
    
    required_packages = [
        'streamlit>=1.28.0',
        'opencv-python>=4.8.0',
        'pillow>=10.0.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'librosa>=0.10.0',
        'scipy>=1.11.0',
        'scikit-learn>=1.3.0',
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'whisper-openai>=20230314',
        'pytesseract>=0.3.10',
        'python-pptx>=0.6.21',
        'PyPDF2>=3.0.1',
        'langdetect>=1.0.9',
        'googletrans==4.0.0rc1',
        'sentence-transformers>=2.2.2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        package_name = package.split('>=')[0].split('==')[0]
        try:
            __import__(package_name.replace('-', '_'))
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - 설치 필요")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📥 {len(missing_packages)}개 패키지 설치 중...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} 설치 완료")
            except subprocess.CalledProcessError:
                print(f"❌ {package} 설치 실패")
    
    print("✅ 패키지 확인 완료!")

def check_system_requirements():
    """시스템 요구사항 확인"""
    print("\n🔧 시스템 요구사항 확인 중...")
    
    # Python 버전 확인
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python 3.8+ 필요 (현재: {python_version.major}.{python_version.minor})")
        return False
    
    # 메모리 확인 (간단한 체크)
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.total >= 4 * 1024**3:  # 4GB
            print(f"✅ 메모리: {memory.total // (1024**3)}GB")
        else:
            print(f"⚠️ 메모리 부족: {memory.total // (1024**3)}GB (권장: 4GB+)")
    except ImportError:
        print("⚠️ 메모리 확인 불가 (psutil 미설치)")
    
    return True

def setup_environment():
    """환경 설정"""
    print("\n🌍 환경 설정 중...")
    
    # 프로젝트 디렉토리 확인
    project_root = Path(__file__).parent
    print(f"📁 프로젝트 경로: {project_root}")
    
    # 필요 디렉토리 생성
    directories = ['data', 'temp', 'outputs', 'logs']
    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"📂 {dir_name}/ 생성 완료")
    
    # 환경 변수 설정
    os.environ['PYTHONPATH'] = str(project_root)
    
    print("✅ 환경 설정 완료!")

def check_demo_files():
    """데모 파일 존재 확인"""
    print("\n📋 데모 파일 확인 중...")
    
    required_files = [
        'jewelry_ai_platform_v21_integrated_demo.py',
        'core/quality_analyzer_v21.py',
        'core/multilingual_processor_v21.py',
        'core/multi_file_integrator_v21.py',
        'core/korean_summary_engine_v21.py',
        'core/mobile_quality_monitor_v21.py',
        'core/smart_content_merger_v21.py',
        'quality/audio_quality_checker.py',
        'quality/ocr_quality_validator.py',
        'quality/image_quality_assessor.py',
        'quality/content_consistency_checker.py'
    ]
    
    missing_files = []
    project_root = Path(__file__).parent
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 파일 누락")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ {len(missing_files)}개 파일이 누락되었습니다.")
        print("GitHub에서 최신 파일을 다운로드하세요.")
        return False
    
    print("✅ 모든 데모 파일 확인 완료!")
    return True

def run_demo():
    """데모 실행"""
    print("\n🚀 주얼리 AI 플랫폼 v2.1 통합 데모 시작!")
    print("=" * 60)
    
    try:
        # Streamlit 앱 실행
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "jewelry_ai_platform_v21_integrated_demo.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ]
        
        print("🌐 브라우저에서 http://localhost:8501 을 열어주세요")
        print("⏹️ 종료하려면 Ctrl+C를 누르세요")
        print("-" * 60)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 데모 종료")
    except Exception as e:
        print(f"\n❌ 실행 중 오류: {str(e)}")
        print("문제가 지속되면 GitHub Issues에 문의하세요.")

def display_banner():
    """시작 배너 표시"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         💎 주얼리 AI 플랫폼 v2.1 통합 데모                    ║
║                                                              ║
║    🔬 품질 혁신 + 🌍 다국어 + 📊 통합분석 + 🇰🇷 한국어      ║
║                                                              ║
║    개발자: 전근혁 (솔로몬드 대표)                            ║
║    날짜: 2025.07.12                                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """메인 실행 함수"""
    display_banner()
    
    try:
        # 1. 시스템 요구사항 확인
        if not check_system_requirements():
            print("❌ 시스템 요구사항을 충족하지 않습니다.")
            return
        
        # 2. 필요 패키지 확인
        check_requirements()
        
        # 3. 환경 설정
        setup_environment()
        
        # 4. 데모 파일 확인
        if not check_demo_files():
            print("❌ 필요한 파일이 누락되었습니다.")
            return
        
        # 5. 데모 실행
        run_demo()
        
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {str(e)}")
        print("자세한 내용은 로그를 확인하거나 GitHub Issues에 문의하세요.")

if __name__ == "__main__":
    main()
