#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최종 시스템 검증 스크립트
전체 시스템 상태를 체크하고 문제점을 진단합니다.
"""

import requests
import subprocess
import json
from pathlib import Path
import time
import sys

def check_system_health():
    """시스템 전체 건강 상태 점검"""
    
    print("[건강검진] SOLOMOND AI 시스템 전체 건강 검진 시작")
    print("=" * 60)
    
    health_score = 0
    max_score = 100
    
    # 1. Streamlit 서버 상태 (20점)
    print("\n1. Streamlit 서버 상태 확인...")
    try:
        response = requests.get("http://localhost:8560", timeout=10)
        if response.status_code == 200:
            print("   ✅ Streamlit 서버 정상 응답 (HTTP 200)")
            health_score += 20
        else:
            print(f"   ⚠️ Streamlit 서버 비정상 응답 (HTTP {response.status_code})")
            health_score += 10
    except Exception as e:
        print(f"   ❌ Streamlit 서버 연결 실패: {e}")
    
    # 2. 필수 파일 존재 확인 (20점)
    print("\n2. 필수 파일 존재 확인...")
    essential_files = [
        "conference_analysis_UNIFIED_COMPLETE.py",
        "core/unicode_safety_system.py",
        "core/enhanced_file_handler.py",
        "core/multimodal_pipeline.py",
        "core/crossmodal_fusion.py",
        "shared/ollama_interface.py"
    ]
    
    missing_files = []
    for file_path in essential_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (누락)")
            missing_files.append(file_path)
    
    if not missing_files:
        health_score += 20
    else:
        health_score += max(0, 20 - len(missing_files) * 3)
    
    # 3. 데이터 파일 상태 (15점)
    print("\n3. 데이터 파일 상태 확인...")
    user_files_dir = Path("user_files")
    if user_files_dir.exists():
        jga_dir = user_files_dir / "JGA2025_D1"
        if jga_dir.exists():
            files = list(jga_dir.glob("*"))
            print(f"   ✅ JGA2025_D1 폴더: {len(files)}개 파일")
            
            # 대용량 파일 확인
            large_files = [f for f in files if f.stat().st_size > 100*1024*1024]
            if large_files:
                print(f"   ✅ 대용량 파일: {len(large_files)}개 (테스트 준비됨)")
                health_score += 15
            else:
                print("   ⚠️ 대용량 파일 없음")
                health_score += 10
        else:
            print("   ❌ JGA2025_D1 폴더 없음")
    else:
        print("   ❌ user_files 디렉토리 없음")
    
    # 4. Python 패키지 상태 (15점)
    print("\n4. Python 패키지 상태 확인...")
    required_packages = [
        'streamlit', 'whisper', 'easyocr', 'torch', 
        'transformers', 'sentence-transformers', 'spacy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (누락)")
            missing_packages.append(package)
    
    if not missing_packages:
        health_score += 15
    else:
        health_score += max(0, 15 - len(missing_packages) * 2)
    
    # 5. Ollama 연결 상태 (10점)
    print("\n5. Ollama AI 서버 상태 확인...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_count = len(models.get('models', []))
            print(f"   ✅ Ollama 서버 연결됨 ({model_count}개 모델)")
            health_score += 10
        else:
            print("   ⚠️ Ollama 서버 응답 이상")
            health_score += 5
    except Exception as e:
        print(f"   ❌ Ollama 서버 연결 실패: {e}")
    
    # 6. SpaCy 모델 상태 (10점)
    print("\n6. SpaCy 한국어 모델 확인...")
    try:
        import spacy
        nlp = spacy.load("ko_core_news_sm")
        print("   ✅ SpaCy 한국어 모델 로드 성공")
        health_score += 10
    except Exception as e:
        print(f"   ❌ SpaCy 한국어 모델 로드 실패: {e}")
    
    # 7. GPU/CUDA 상태 (10점)
    print("\n7. GPU/CUDA 상태 확인...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ CUDA 사용 가능: {gpu_count}개 GPU ({gpu_name})")
            health_score += 10
        else:
            print("   ⚠️ CUDA 사용 불가 (CPU 모드)")
            health_score += 5
    except Exception as e:
        print(f"   ❌ PyTorch/CUDA 확인 실패: {e}")
    
    # 결과 요약
    print("\n" + "=" * 60)
    print(f"[건강도] 시스템 건강도: {health_score}/{max_score}점")
    
    if health_score >= 90:
        status = "[최고] 최고 (Excellent)"
        advice = "시스템이 완벽하게 준비되었습니다!"
    elif health_score >= 75:
        status = "[양호] 양호 (Good)"
        advice = "시스템이 잘 작동하고 있습니다. 몇 가지 개선점이 있습니다."
    elif health_score >= 50:
        status = "[보통] 보통 (Fair)"
        advice = "시스템에 몇 가지 문제가 있습니다. 개선이 필요합니다."
    else:
        status = "[위험] 위험 (Poor)"
        advice = "시스템에 심각한 문제가 있습니다. 즉시 수정이 필요합니다."
    
    print(f"상태: {status}")
    print(f"권장사항: {advice}")
    
    return health_score, missing_files, missing_packages

def generate_health_report():
    """건강 상태 보고서 생성"""
    
    score, missing_files, missing_packages = check_system_health()
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "health_score": score,
        "status": "healthy" if score >= 75 else "needs_attention",
        "missing_files": missing_files,
        "missing_packages": missing_packages,
        "recommendations": []
    }
    
    # 권장사항 생성
    if missing_files:
        report["recommendations"].append(f"누락된 파일 복구: {', '.join(missing_files)}")
    
    if missing_packages:
        report["recommendations"].append(f"누락된 패키지 설치: pip install {' '.join(missing_packages)}")
    
    if score < 90:
        report["recommendations"].append("전체 시스템 최적화 실행 권장")
    
    # 보고서 저장
    with open("system_health_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n[보고서] 상세 보고서가 system_health_report.json에 저장되었습니다.")
    
    return report

def quick_fix_suggestions():
    """빠른 수정 제안"""
    
    print("\n[수정] 빠른 수정 제안:")
    print("1. SpaCy 모델 설치: python -m spacy download ko_core_news_sm")
    print("2. Ollama 서버 시작: ollama serve")
    print("3. 누락 패키지 설치: pip install -r requirements.txt")
    print("4. 시스템 재시작: Ctrl+C 후 재실행")

if __name__ == "__main__":
    try:
        report = generate_health_report()
        quick_fix_suggestions()
        
        # 성공/실패 반환
        sys.exit(0 if report["health_score"] >= 75 else 1)
        
    except KeyboardInterrupt:
        print("\n\n검사가 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[오류] 시스템 검사 중 오류 발생: {e}")
        sys.exit(1)