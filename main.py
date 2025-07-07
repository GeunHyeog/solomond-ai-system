#!/usr/bin/env python3
"""
솔로몬드 AI 시스템 v3.0 - 메인 진입점
실제 내용을 읽고 분석하는 차세대 AI 플랫폼

개발자: 전근혁 (솔로몬드 대표, 한국보석협회 사무국장)
"""

import sys
import os
import asyncio
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_banner():
    """시스템 시작 배너 출력"""
    banner = """
🚀 솔로몬드 AI 시스템 v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실제 내용을 읽고 분석하는 차세대 AI 플랫폼

📍 상태: ✅ 모듈화 구조 완성 (Phase 2 진행 중)
🏗️ 아키텍처: config/ + core/ + api/ + ui/ + utils/
🎯 새로운 기능: RESTful API + 배치 처리 + 모델 선택
👤 개발자: 전근혁 (솔로몬드 대표)

💡 사용법: 
   • python main.py              (새로운 모듈화 버전)
   • python minimal_stt_test.py  (기존 안정 버전)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    print(banner)

def check_dependencies():
    """의존성 검사"""
    missing_deps = []
    
    try:
        import fastapi
        print("✅ FastAPI: 설치됨")
    except ImportError:
        missing_deps.append("fastapi")
    
    try:
        import whisper
        print("✅ OpenAI Whisper: 설치됨")
    except ImportError:
        missing_deps.append("openai-whisper")
        
    try:
        import uvicorn
        print("✅ Uvicorn: 설치됨")
    except ImportError:
        missing_deps.append("uvicorn")
    
    if missing_deps:
        print(f"❌ 누락된 의존성: {', '.join(missing_deps)}")
        print("📦 설치 명령: pip install " + " ".join(missing_deps))
        return False
    
    print("✅ 모든 필수 의존성 확인 완료")
    return True

def run_modular_version():
    """새로운 모듈화 버전 실행"""
    try:
        from api.app import run_app
        print("🎯 모듈화된 FastAPI 앱 시작...")
        run_app(host="0.0.0.0", port=8080, debug=True)
    except ImportError as e:
        print(f"❌ 모듈 import 오류: {e}")
        print("🔄 minimal_stt_test.py로 대체 실행...")
        run_legacy_version()
    except Exception as e:
        print(f"❌ 앱 실행 오류: {e}")
        print("🔧 문제 해결을 위해 레거시 버전으로 전환...")
        run_legacy_version()

def run_legacy_version():
    """기존 안정 버전 실행"""
    try:
        print("🔄 레거시 버전 (minimal_stt_test.py) 실행...")
        import subprocess
        subprocess.run([sys.executable, "minimal_stt_test.py"])
    except Exception as e:
        print(f"❌ 레거시 버전 실행 실패: {e}")

def show_usage_guide():
    """사용법 가이드 출력"""
    guide = """
📚 솔로몬드 AI 시스템 사용 가이드

1️⃣ 웹 인터페이스:
   • 브라우저에서 http://localhost:8080 접속
   • 음성 파일 드래그&드롭 또는 파일 선택
   • 처리 결과 실시간 확인

2️⃣ API 사용:
   • POST /api/process_audio: 단일 파일 처리
   • POST /api/analyze_batch: 다중 파일 배치 처리
   • GET /api/test: 시스템 상태 확인
   • GET /docs: 전체 API 문서

3️⃣ 지원 파일 형식:
   • 🎵 MP3: 일반적인 음성 파일
   • 🎶 WAV: 고품질 무압축 오디오
   • 📱 M4A: 모바일 녹음 파일

4️⃣ 새로운 기능:
   • 다중 파일 배치 처리
   • 실시간 진행률 표시
   • 모바일 친화적 인터페이스
   • RESTful API 지원
"""
    print(guide)

def main():
    """메인 함수"""
    print_banner()
    
    # 의존성 확인
    if not check_dependencies():
        print("\n🛠️ 의존성을 설치한 후 다시 실행해주세요.")
        sys.exit(1)
    
    # 모듈화 버전 실행 시도
    print("\n🚀 시스템 시작...")
    
    # 명령행 인수 확인
    if len(sys.argv) > 1:
        if sys.argv[1] == "--legacy":
            print("🔄 레거시 모드로 실행...")
            run_legacy_version()
            return
        elif sys.argv[1] == "--help":
            show_usage_guide()
            return
    
    # 기본적으로 모듈화 버전 실행
    try:
        run_modular_version()
    except KeyboardInterrupt:
        print("\n👋 시스템 종료됨")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        print("📞 문제가 지속되면 개발자에게 문의하세요.")

if __name__ == "__main__":
    main()
