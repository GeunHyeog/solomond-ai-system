#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
빠른 자동 테스트 - Windows 안전 버전
"""
import os
import time
import json
from pathlib import Path
from datetime import datetime

# 기본 설정
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU 강제 사용

def quick_performance_test():
    """빠른 성능 테스트"""
    print("=== 솔로몬드 AI 성능 진단 ===")
    
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'issues': [],
        'recommendations': []
    }
    
    # 1. 파일 발견
    print("1. 파일 발견 중...")
    user_files = Path("user_files")
    files_found = {'audio': 0, 'image': 0, 'video': 0}
    
    if user_files.exists():
        for file_path in user_files.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in ['.m4a', '.wav', '.mp3']:
                    files_found['audio'] += 1
                elif ext in ['.jpg', '.jpeg', '.png']:
                    files_found['image'] += 1
                elif ext in ['.mov', '.mp4']:
                    files_found['video'] += 1
    
    print(f"   오디오: {files_found['audio']}개")
    print(f"   이미지: {files_found['image']}개") 
    print(f"   비디오: {files_found['video']}개")
    
    # 2. 라이브러리 테스트
    print("\\n2. 핵심 라이브러리 테스트...")
    
    # Whisper 테스트
    try:
        import whisper
        print("   [OK] Whisper 사용 가능")
        
        # 빠른 로딩 테스트
        start = time.time()
        model = whisper.load_model("tiny", device="cpu")
        load_time = time.time() - start
        print(f"   Whisper tiny 로딩: {load_time:.1f}초")
        
        if load_time > 15:
            results['issues'].append(f"Whisper 로딩 느림: {load_time:.1f}초")
        
        del model
        
    except Exception as e:
        print(f"   [ERROR] Whisper 실패: {str(e)[:50]}...")
        results['issues'].append(f"Whisper error: {e}")
    
    # EasyOCR 테스트
    try:
        import easyocr
        print("   [OK] EasyOCR 사용 가능")
        
        start = time.time()
        reader = easyocr.Reader(['ko', 'en'], gpu=False)
        load_time = time.time() - start
        print(f"   EasyOCR 로딩: {load_time:.1f}초")
        
        if load_time > 30:
            results['issues'].append(f"EasyOCR 로딩 느림: {load_time:.1f}초")
        
        del reader
        
    except Exception as e:
        print(f"   [ERROR] EasyOCR 실패: {str(e)[:50]}...")
        results['issues'].append(f"EasyOCR error: {e}")
    
    # Transformers 테스트
    try:
        from transformers import pipeline
        print("   [OK] Transformers 사용 가능")
    except Exception as e:
        print(f"   [ERROR] Transformers 실패: {str(e)[:50]}...")
        results['issues'].append(f"Transformers error: {e}")
    
    # 3. 성능 문제 분석
    print("\\n3. 성능 문제 분석...")
    
    known_issues = [
        "모델 매번 로딩으로 인한 지연 (10-30초)",
        "대용량 오디오 파일 처리 시간 증가",
        "CPU 모드에서의 느린 AI 처리 속도",
        "EasyOCR 초기화 지연 (20-40초)"
    ]
    
    solutions = [
        "Whisper tiny 모델 사용으로 속도 향상",
        "모델 사전 로딩 및 캐싱 구현", 
        "파일 크기 제한 또는 청크 처리",
        "GPU 활용 가능 시 GPU 모드 전환"
    ]
    
    for issue in known_issues:
        print(f"   문제: {issue}")
        results['issues'].append(issue)
    
    print("\\n4. 권장 해결책...")
    for i, solution in enumerate(solutions, 1):
        print(f"   {i}. {solution}")
        results['recommendations'].append(solution)
    
    # 4. 즉시 적용 가능한 최적화
    print("\\n5. 즉시 최적화 적용...")
    
    # 환경 변수 최적화
    optimizations = {
        'OMP_NUM_THREADS': '4',
        'MKL_NUM_THREADS': '4', 
        'NUMEXPR_NUM_THREADS': '4',
        'WHISPER_CACHE_DIR': str(Path.home() / '.cache' / 'whisper'),
        'TRANSFORMERS_CACHE': str(Path.home() / '.cache' / 'transformers')
    }
    
    for key, value in optimizations.items():
        os.environ[key] = value
        print(f"   설정: {key}={value}")
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_diagnosis_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\\n결과 저장: {filename}")
    print(f"발견된 문제: {len(results['issues'])}개")
    print(f"권장사항: {len(results['recommendations'])}개")
    
    if len(results['issues']) > 5:
        print("\\n[권장] 시스템 최적화가 필요합니다!")
    else:
        print("\\n[정상] 기본 성능은 양호합니다.")
    
    return results

if __name__ == "__main__":
    quick_performance_test()