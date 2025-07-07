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

# 모듈 import (아직 구현되지 않은 모듈들은 주석 처리)
try:
    from config.settings import get_settings
    # from core.analyzer import AudioAnalyzer
    # from core.file_processor import FileProcessor  
    # from api.app import create_app
    # from utils.logger import get_logger
except ImportError as e:
    print(f"모듈 import 오류: {e}")
    print("아직 구현되지 않은 모듈이 있습니다. minimal_stt_test.py를 사용해주세요.")

def print_banner():
    """시스템 시작 배너 출력"""
    banner = """
🚀 솔로몬드 AI 시스템 v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실제 내용을 읽고 분석하는 차세대 AI 플랫폼

📍 현재 상태: 모듈화 구조 완성 (Phase 1 완료)
🎯 다음 단계: Phase 2 - 기능 복구 및 확장
👤 개발자: 전근혁 (솔로몬드 대표)

💡 임시 사용법: python minimal_stt_test.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    print(banner)

def check_dependencies():
    """의존성 검사"""
    try:
        import fastapi
        import whisper
        print("✅ 필수 의존성 확인 완료")
        return True
    except ImportError as e:
        print(f"❌ 의존성 오류: {e}")
        print("pip install -r requirements.txt를 실행해주세요.")
        return False

def main():
    """메인 함수"""
    print_banner()
    
    if not check_dependencies():
        sys.exit(1)
    
    print("📋 사용 가능한 실행 방법:")
    print("1. python minimal_stt_test.py  # 현재 작동하는 버전")
    print("2. python main.py              # 모듈화 버전 (개발 중)")
    print()
    print("🔄 Phase 2 개발 진행 중...")
    print("   - 모듈별 기능 구현")  
    print("   - 통합 테스트")
    print("   - UI/UX 개선")

if __name__ == "__main__":
    main()
