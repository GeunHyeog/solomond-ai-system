#!/usr/bin/env python3
"""
통합 개발 툴킷 의존성 설치 스크립트
"""

import subprocess
import sys
import os

def install_package(package_name):
    """패키지 설치"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"[OK] {package_name} 설치 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {package_name} 설치 실패: {e}")
        return False

def main():
    """메인 설치 함수"""
    
    print("통합 개발 툴킷 의존성 설치 시작")
    
    # 필수 패키지 목록
    required_packages = [
        "playwright",           # 브라우저 자동화
        "supabase",            # Supabase 클라이언트
        "duckduckgo-search",   # 웹 검색
        "beautifulsoup4",      # HTML 파싱
        "requests",            # HTTP 요청
    ]
    
    # 패키지 설치
    success_count = 0
    for package in required_packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 설치 결과: {success_count}/{len(required_packages)} 완료")
    
    # Playwright 브라우저 설치
    if success_count > 0:
        print("\n🌐 Playwright 브라우저 설치 중...")
        try:
            subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
            print("✅ Playwright Chromium 브라우저 설치 완료")
        except subprocess.CalledProcessError as e:
            print(f"❌ Playwright 브라우저 설치 실패: {e}")
    
    # GitHub CLI 설치 확인
    print("\n🐙 GitHub CLI 설치 확인...")
    try:
        result = subprocess.run(["gh", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GitHub CLI 이미 설치됨")
        else:
            print("[INFO] GitHub CLI 설치 필요")
            print("   설치 방법: https://cli.github.com/")
    except FileNotFoundError:
        print("[INFO] GitHub CLI 설치 필요")
        print("   Windows: winget install --id GitHub.cli")
        print("   또는: https://cli.github.com/")
    
    # 환경 변수 설정 안내
    print("\n🔧 환경 변수 설정 필요:")
    print("   SUPABASE_URL=your_supabase_url")
    print("   SUPABASE_ANON_KEY=your_supabase_anon_key")
    print("   GITHUB_TOKEN=your_github_token")
    
    print("\n✅ 설치 완료! 통합 개발 툴킷을 사용할 수 있습니다.")

if __name__ == "__main__":
    main()