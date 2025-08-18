#!/usr/bin/env python3
"""
🔧 SOLOMOND AI 하이브리드 유틸리티
uv/pip 성능 모니터링 및 관리 도구
"""

import subprocess
import time
import sys
from pathlib import Path

def measure_install_speed(tool: str, package: str) -> float:
    """패키지 매니저별 설치 속도 측정"""
    try:
        start_time = time.time()
        if tool == 'uv':
            # uv는 'uv pip install' 형태로 사용, --system 플래그 추가
            cmd = ['uv', 'pip', 'install', '--system', '--dry-run', package]
        else:
            cmd = [tool, 'install', '--dry-run', package]
            
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=30
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            return elapsed
        else:
            print(f"FAIL {tool} 테스트 실패: {result.stderr[:100]}")
            return -1
    except Exception as e:
        print(f"ERROR {tool} 오류: {e}")
        return -1

def compare_performance():
    """uv vs pip 성능 비교"""
    print("uv vs pip 성능 벤치마크 시작...")
    
    test_packages = ['requests', 'pandas', 'streamlit']
    results = {}
    
    for package in test_packages:
        print(f"\n테스트 중: {package}")
        
        # uv 테스트
        uv_time = measure_install_speed('uv', package)
        if uv_time > 0:
            print(f"   uv: {uv_time:.2f}초")
        
        # pip 테스트
        pip_time = measure_install_speed('pip', package)
        if pip_time > 0:
            print(f"   pip: {pip_time:.2f}초")
        
        # 비교 결과
        if uv_time > 0 and pip_time > 0:
            speedup = pip_time / uv_time
            print(f"   uv가 {speedup:.1f}배 빠름")
            results[package] = {'uv': uv_time, 'pip': pip_time, 'speedup': speedup}
    
    # 종합 결과
    if results:
        avg_speedup = sum(r['speedup'] for r in results.values()) / len(results)
        print(f"\n평균 속도 향상: {avg_speedup:.1f}배")
    
    return results

def check_hybrid_health():
    """하이브리드 시스템 상태 확인"""
    print("SOLOMOND AI 하이브리드 시스템 건강도 체크...")
    
    # 도구 존재 확인
    tools_status = {}
    for tool in ['uv', 'pip', 'python']:
        try:
            result = subprocess.run([tool, '--version'], capture_output=True, timeout=5)
            tools_status[tool] = result.returncode == 0
        except:
            tools_status[tool] = False
    
    print("\n도구 상태:")
    for tool, status in tools_status.items():
        status_icon = "OK" if status else "FAIL"
        print(f"   {status_icon} {tool}")
    
    # 패키지 확인
    critical_packages = ['streamlit', 'torch', 'transformers', 'whisper', 'easyocr']
    package_status = {}
    
    print("\n핵심 패키지 확인:")
    for package in critical_packages:
        try:
            __import__(package.replace('-', '_'))
            package_status[package] = True
            print(f"   OK {package}")
        except ImportError:
            package_status[package] = False
            print(f"   FAIL {package}")
    
    # 건강도 점수
    total_checks = len(tools_status) + len(package_status)
    passed_checks = sum(tools_status.values()) + sum(package_status.values())
    health_score = (passed_checks / total_checks) * 100
    
    print(f"\n시스템 건강도: {health_score:.0f}%")
    
    if health_score >= 80:
        print("OK 시스템 상태 양호")
    elif health_score >= 60:
        print("WARNING 일부 개선 필요")
    else:
        print("ERROR 시스템 점검 필요")
    
    return health_score

def smart_install(package: str):
    """패키지별 스마트 설치"""
    # AI 패키지는 pip, 일반 패키지는 uv
    ai_packages = {'torch', 'transformers', 'whisper', 'easyocr', 'pyannote.audio', 'speechbrain'}
    
    if package in ai_packages:
        tool = 'pip'
        reason = "AI/복잡한 의존성"
    else:
        tool = 'uv'
        reason = "일반/빠른 설치"
    
    print(f"📦 {package} 설치 중 ({tool} 사용 - {reason})")
    
    try:
        if tool == 'uv':
            cmd = ['uv', 'pip', 'install', '--system', package]
        else:
            cmd = [tool, 'install', package]
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {package} 설치 성공")
        else:
            print(f"❌ {package} 설치 실패: {result.stderr[:100]}")
            # 폴백 시도
            fallback_tool = 'pip' if tool == 'uv' else 'uv'
            print(f"🔄 {fallback_tool}로 재시도...")
            if fallback_tool == 'uv':
                subprocess.run(['uv', 'pip', 'install', '--system', package])
            else:
                subprocess.run([fallback_tool, 'install', package])
    except Exception as e:
        print(f"❌ 설치 오류: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "benchmark":
            compare_performance()
        elif command == "health":
            check_hybrid_health()
        elif command == "install" and len(sys.argv) > 2:
            smart_install(sys.argv[2])
        else:
            print("사용법: python hybrid_utils.py [benchmark|health|install <package>]")
    else:
        print("🔧 SOLOMOND AI 하이브리드 유틸리티")
        print("사용 가능한 명령:")
        print("  benchmark - uv vs pip 성능 비교")
        print("  health    - 시스템 상태 확인")  
        print("  install   - 스마트 패키지 설치")