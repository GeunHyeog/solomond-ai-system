#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
솔로몬드 AI v2.1.4 - 시스템 환경 자동 진단 도구
현장 테스트 전 시스템 상태를 자동으로 확인합니다.

작성자: 전근혁 (솔로몬드 대표)
생성일: 2025.07.13
목적: 현장 테스트 환경 자동 검증
"""

import sys
import platform
import psutil
import subprocess
import importlib
import os
from datetime import datetime

def check_system_requirements():
    """시스템 요구사항 자동 확인"""
    print("🔍 솔로몬드 AI v2.1.4 시스템 환경 진단")
    print("=" * 60)
    
    # 기본 시스템 정보
    print(f"📅 진단 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  운영체제: {platform.system()} {platform.release()}")
    print(f"🐍 Python 버전: {sys.version.split()[0]}")
    print(f"💻 아키텍처: {platform.machine()}")
    print()
    
    # CPU 정보
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    print(f"🔥 CPU 정보:")
    print(f"   - 물리 코어: {psutil.cpu_count(logical=False)}개")
    print(f"   - 논리 코어: {cpu_count}개 (병렬 처리 가능)")
    if cpu_freq:
        print(f"   - 현재 속도: {cpu_freq.current:.0f}MHz")
        print(f"   - 최대 속도: {cpu_freq.max:.0f}MHz")
    print()
    
    # 메모리 정보
    memory = psutil.virtual_memory()
    print(f"🧠 메모리 정보:")
    print(f"   - 총 메모리: {memory.total / (1024**3):.1f}GB")
    print(f"   - 사용 가능: {memory.available / (1024**3):.1f}GB")
    print(f"   - 사용률: {memory.percent:.1f}%")
    print(f"   - AI 처리 권장: {memory.total * 0.4 / (1024**3):.1f}GB")
    print()
    
    # 디스크 정보
    disk = psutil.disk_usage('/')
    print(f"💾 저장공간:")
    print(f"   - 총 용량: {disk.total / (1024**3):.1f}GB")
    print(f"   - 사용 가능: {disk.free / (1024**3):.1f}GB")
    print(f"   - 사용률: {(disk.used / disk.total) * 100:.1f}%")
    print()
    
    # 필수 패키지 확인
    essential_packages = [
        ('streamlit', 'Streamlit 웹 프레임워크'),
        ('pandas', '데이터 처리'),
        ('numpy', '수치 연산'),
        ('psutil', '시스템 모니터링'),
        ('pathlib', '파일 경로 처리'),
        ('concurrent.futures', '병렬 처리'),
        ('asyncio', '비동기 처리'),
        ('tempfile', '임시 파일 처리'),
        ('json', 'JSON 데이터 처리'),
        ('base64', '파일 인코딩'),
        ('datetime', '시간 처리'),
        ('logging', '로깅'),
        ('io', '입출력 처리'),
        ('threading', '스레드 처리'),
        ('multiprocessing', '멀티프로세싱'),
        ('functools', '함수형 프로그래밍'),
        ('gc', '가비지 컬렉션')
    ]
    
    print("📦 필수 패키지 확인:")
    missing_essential = []
    for package, description in essential_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package}: 설치됨 ({description})")
        except ImportError:
            print(f"   ❌ {package}: 누락됨 ({description})")
            missing_essential.append(package)
    print()
    
    # AI 선택 패키지 확인
    ai_packages = [
        ('librosa', '음성 분석'),
        ('cv2', '이미지 처리 (opencv-python)'),
        ('moviepy.editor', '비디오 처리'),
        ('PIL', '이미지 처리 (Pillow)'),
        ('requests', 'HTTP 요청'),
        ('youtube_dl', '유튜브 다운로드'),
        ('scipy', '과학 연산'),
        ('matplotlib', '시각화'),
        ('seaborn', '고급 시각화'),
        ('plotly', '인터랙티브 차트')
    ]
    
    print("🤖 AI 패키지 확인 (선택사항):")
    missing_ai = []
    available_ai = []
    for package, description in ai_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package}: 설치됨 ({description})")
            available_ai.append(package)
        except ImportError:
            print(f"   ⚠️  {package}: 누락됨 ({description})")
            missing_ai.append(package)
    print()
    
    # 네트워크 연결 확인
    print("🌐 네트워크 연결 확인:")
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        print("   ✅ 인터넷 연결: 정상")
        
        # GitHub 연결 확인
        try:
            import urllib.request
            urllib.request.urlopen('https://github.com', timeout=5)
            print("   ✅ GitHub 연결: 정상")
        except:
            print("   ⚠️  GitHub 연결: 불안정")
            
    except:
        print("   ❌ 인터넷 연결: 불가능")
    print()
    
    # 전체 평가
    print("📊 전체 시스템 평가:")
    print("=" * 60)
    
    # CPU 평가
    cpu_score = min(100, (cpu_count / 4) * 100)  # 4코어를 100점 기준
    print(f"🔥 CPU 성능: {cpu_score:.0f}점 ({cpu_count}코어)")
    
    # 메모리 평가
    memory_gb = memory.total / (1024**3)
    memory_score = min(100, (memory_gb / 8) * 100)  # 8GB를 100점 기준
    print(f"🧠 메모리 용량: {memory_score:.0f}점 ({memory_gb:.1f}GB)")
    
    # 저장공간 평가
    free_gb = disk.free / (1024**3)
    disk_score = min(100, (free_gb / 10) * 100)  # 10GB를 100점 기준
    print(f"💾 저장공간: {disk_score:.0f}점 ({free_gb:.1f}GB 사용가능)")
    
    # 패키지 평가
    package_score = ((len(essential_packages) - len(missing_essential)) / len(essential_packages)) * 100
    print(f"📦 패키지 완성도: {package_score:.0f}점 ({len(essential_packages) - len(missing_essential)}/{len(essential_packages)})")
    
    # AI 패키지 평가
    ai_score = (len(available_ai) / len(ai_packages)) * 100
    print(f"🤖 AI 기능: {ai_score:.0f}점 ({len(available_ai)}/{len(ai_packages)})")
    
    # 전체 점수
    total_score = (cpu_score + memory_score + disk_score + package_score + ai_score) / 5
    print(f"\n⭐ 전체 준비도: {total_score:.0f}점 (100점 만점)")
    
    # 권장사항
    print("\n💡 권장사항:")
    if cpu_count < 4:
        print("   🔥 CPU: 4코어 이상 권장 (현재 성능으로도 동작 가능)")
    if memory_gb < 8:
        print("   🧠 메모리: 8GB 이상 권장 (대용량 파일 처리 제한)")
    if free_gb < 10:
        print("   💾 저장공간: 10GB 이상 확보 권장")
    if missing_essential:
        print(f"   📦 필수 패키지 설치: pip install {' '.join(missing_essential)}")
    if len(missing_ai) > 5:
        print("   🤖 AI 패키지 설치로 기능 향상 가능")
    
    # 현장 테스트 준비도
    print(f"\n🎯 현장 테스트 준비도:")
    if total_score >= 80:
        print("   ✅ 우수: 모든 고성능 기능 사용 가능")
        test_readiness = "우수"
    elif total_score >= 60:
        print("   ✅ 양호: 기본 기능 정상 사용 가능")
        test_readiness = "양호"
    elif total_score >= 40:
        print("   ⚠️  보통: 일부 기능 제한될 수 있음")
        test_readiness = "보통"
    else:
        print("   ❌ 부족: 시스템 업그레이드 권장")
        test_readiness = "부족"
    
    # 추천 실행 명령어
    print(f"\n🚀 추천 실행 명령어:")
    max_upload_size = min(5120, int(memory.available / (1024**2) * 0.4))  # 가용 메모리의 40%
    print(f"streamlit run jewelry_stt_ui_v214_performance_optimized.py --server.maxUploadSize={max_upload_size}")
    
    return {
        'cpu_cores': cpu_count,
        'memory_gb': memory_gb,
        'disk_free_gb': free_gb,
        'total_score': total_score,
        'test_readiness': test_readiness,
        'missing_essential': missing_essential,
        'missing_ai': missing_ai,
        'max_upload_size': max_upload_size
    }

def generate_install_script(missing_packages):
    """설치 스크립트 생성"""
    if not missing_packages:
        return None
    
    script = f"""#!/bin/bash
# 솔로몬드 AI v2.1.4 자동 설치 스크립트
# 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

echo "🚀 솔로몬드 AI 필수 패키지 설치 시작..."

# Python 업그레이드
python -m pip install --upgrade pip

# 필수 패키지 설치
"""
    
    for package in missing_packages:
        if package == 'cv2':
            script += "pip install opencv-python\n"
        elif package == 'PIL':
            script += "pip install Pillow\n"
        elif package == 'moviepy.editor':
            script += "pip install moviepy\n"
        else:
            script += f"pip install {package}\n"
    
    script += """
echo "✅ 설치 완료!"
echo "🧪 시스템 진단을 다시 실행하여 확인하세요."
"""
    
    return script

if __name__ == "__main__":
    try:
        # 시스템 진단 실행
        result = check_system_requirements()
        
        # 부족한 패키지가 있으면 설치 스크립트 생성
        all_missing = result['missing_essential'] + result['missing_ai']
        if all_missing:
            script = generate_install_script(all_missing)
            if script:
                with open('install_missing_packages.sh', 'w', encoding='utf-8') as f:
                    f.write(script)
                print(f"\n📜 설치 스크립트 생성: install_missing_packages.sh")
                print("실행: bash install_missing_packages.sh")
        
        print(f"\n🎉 시스템 진단 완료! ({result['test_readiness']})")
        
    except Exception as e:
        print(f"❌ 진단 중 오류 발생: {e}")
        print("Python 환경을 확인해주세요.")
