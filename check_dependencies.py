#!/usr/bin/env python3
"""
의존성 및 패키지 상태 검사 스크립트
"""

import importlib.util
import sys
import subprocess
from pathlib import Path

def check_package_installed(package_name):
    """패키지 설치 상태 확인"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def get_package_version(package_name):
    """패키지 버전 확인"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', package_name], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':')[1].strip()
    except:
        pass
    return 'Unknown'

def main():
    print("=" * 50)
    print("솔로몬드 AI 의존성 검사")
    print("=" * 50)
    
    # 핵심 패키지 목록
    critical_packages = [
        ('streamlit', 'UI 프레임워크'),
        ('whisper', '음성 인식'),
        ('easyocr', '이미지 텍스트 추출'),
        ('transformers', 'AI 모델'),
        ('torch', '딥러닝 프레임워크'),
        ('librosa', '오디오 처리'),
        ('fastapi', 'API 서버'),
        ('psutil', '시스템 모니터링'),
        ('pandas', '데이터 처리'),
        ('pillow', '이미지 처리')
    ]
    
    print("\n=== 핵심 패키지 상태 ===")
    installed_count = 0
    
    for package, description in critical_packages:
        is_installed = check_package_installed(package)
        version = get_package_version(package) if is_installed else 'N/A'
        status = "OK" if is_installed else "Missing"
        
        print(f"{status:<7} {package:<15} {version:<10} - {description}")
        
        if is_installed:
            installed_count += 1
    
    print(f"\n총 설치된 패키지: {installed_count}/{len(critical_packages)}")
    
    # 선택적 패키지
    optional_packages = [
        ('anthropic', 'Claude API'),
        ('openai', 'OpenAI API'),
        ('playwright', '브라우저 자동화'),
        ('pydantic', '데이터 검증')
    ]
    
    print("\n=== 선택적 패키지 상태 ===")
    optional_installed = 0
    
    for package, description in optional_packages:
        is_installed = check_package_installed(package)
        version = get_package_version(package) if is_installed else 'N/A'
        status = "OK" if is_installed else "Optional"
        
        print(f"{status:<8} {package:<15} {version:<10} - {description}")
        
        if is_installed:
            optional_installed += 1
    
    print(f"\n선택적 패키지: {optional_installed}/{len(optional_packages)}")
    
    # 환경 설정 확인
    print("\n=== 환경 설정 확인 ===")
    
    # CUDA 확인
    cuda_disabled = os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    print(f"{'OK' if cuda_disabled else 'WARNING':<8} CUDA 비활성화: {'예' if cuda_disabled else '아니오'}")
    
    # MCP 설정 확인
    mcp_config_path = Path.home() / '.config' / 'claude' / 'claude_desktop_config.json'
    mcp_configured = mcp_config_path.exists()
    print(f"{'OK' if mcp_configured else 'WARNING':<8} MCP 설정: {'존재' if mcp_configured else '없음'}")
    
    # 권장사항 출력
    print("\n=== 권장사항 ===")
    
    if installed_count < len(critical_packages):
        missing_critical = len(critical_packages) - installed_count
        print(f"WARNING: {missing_critical}개 핵심 패키지 설치 필요")
        print("   pip install -r requirements_v23_windows.txt")
    
    if not cuda_disabled:
        print("WARNING: CUDA 비활성화 권장 (메모리 절약)")
        print("   set CUDA_VISIBLE_DEVICES=")
    
    if not mcp_configured:
        print("INFO: MCP 설정 파일 필요 (고급 기능용)")
    
    # 종합 평가
    print("\n=== 종합 평가 ===")
    
    total_score = (installed_count / len(critical_packages)) * 70 + (optional_installed / len(optional_packages)) * 30
    
    if total_score >= 90:
        status = "EXCELLENT"
    elif total_score >= 70:
        status = "GOOD"
    else:
        status = "NEEDS_IMPROVEMENT"
    
    print(f"의존성 점수: {total_score:.1f}/100 - {status}")
    
    return total_score >= 70

if __name__ == "__main__":
    import os
    success = main()
    sys.exit(0 if success else 1)