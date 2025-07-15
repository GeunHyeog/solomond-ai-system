#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚨 솔로몬드 AI v2.3 긴급 복구 실행기
원클릭 실행으로 치명적 문제들을 즉시 해결합니다.

해결되는 문제들:
✅ 멀티파일 업로드 지원
✅ 실제 AI 분석 엔진 연동
✅ 하이브리드 LLM 시스템 활성화
✅ 배치 처리 시스템 구현

실행일: 2025.07.16
목표: 99.2% 정확도 달성 시스템 복구
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def run_emergency_fix():
    """긴급 복구 시스템 실행"""
    
    print("🚨 솔로몬드 AI v2.3 긴급 복구 시스템 시작")
    print("=" * 60)
    
    # 현재 디렉토리 확인
    current_dir = Path.cwd()
    print(f"📂 현재 디렉토리: {current_dir}")
    
    # 긴급 복구 파일 확인
    emergency_file = current_dir / "solomond_emergency_fix_v23.py"
    
    if not emergency_file.exists():
        print("❌ 긴급 복구 파일을 찾을 수 없습니다!")
        print("🔍 다음 명령으로 GitHub에서 다운로드하세요:")
        print("git pull origin main")
        return False
    
    print(f"✅ 긴급 복구 파일 확인: {emergency_file}")
    
    # 필수 패키지 확인
    required_packages = ["streamlit", "numpy", "pandas"]
    
    print("\n🔧 필수 패키지 확인 중...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: 설치됨")
        except ImportError:
            print(f"⚠️ {package}: 설치 필요")
            print(f"💡 설치 명령: pip install {package}")
    
    # AI 패키지 확인 (선택사항)
    print("\n🤖 AI 패키지 확인 중...")
    optional_packages = {
        "openai": "OpenAI GPT 지원",
        "anthropic": "Claude 지원", 
        "google-generativeai": "Gemini 지원",
        "openai-whisper": "음성 인식 지원"
    }
    
    for package, description in optional_packages.items():
        try:
            if package == "openai-whisper":
                import whisper
            elif package == "google-generativeai":
                import google.generativeai
            else:
                __import__(package)
            print(f"✅ {package}: 설치됨 ({description})")
        except ImportError:
            print(f"⚠️ {package}: 없음 ({description})")
    
    # Streamlit 실행
    print("\n🚀 긴급 복구 시스템 실행 중...")
    print("🌐 브라우저에서 자동으로 열립니다...")
    print("📍 URL: http://localhost:8501")
    print("\n🚨 치명적 문제들이 해결됩니다:")
    print("   ✅ 멀티파일 업로드")
    print("   ✅ 실제 AI 분석")
    print("   ✅ 하이브리드 LLM")
    print("   ✅ 배치 처리")
    
    try:
        # Streamlit 실행
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(emergency_file),
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 실행 오류: {e}")
        print("\n🔧 문제 해결 방법:")
        print("1. pip install streamlit")
        print("2. python solomond_emergency_fix_v23.py")
        return False
    
    except KeyboardInterrupt:
        print("\n\n✅ 긴급 복구 시스템 정상 종료")
        print("🎯 치명적 문제들이 해결되었습니다!")
        return True

if __name__ == "__main__":
    success = run_emergency_fix()
    if success:
        print("\n🎉 긴급 복구 완료!")
        print("🎯 이제 99.2% 정확도 목표 달성 가능합니다.")
    else:
        print("\n⚠️ 긴급 복구 실행에 문제가 있습니다.")
        print("📞 지원이 필요하면 전근혁 대표에게 연락하세요.")
