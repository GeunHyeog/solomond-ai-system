#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
성능 최적화 자동 적용
"""
import os
import sys
from pathlib import Path

def apply_whisper_optimization():
    """Whisper 최적화 적용"""
    print("=== Whisper 최적화 적용 ===")
    
    # 1. 환경 변수 최적화
    optimizations = {
        'WHISPER_CACHE_DIR': str(Path.home() / '.cache' / 'whisper'),
        'OMP_NUM_THREADS': '4',
        'MKL_NUM_THREADS': '4'
    }
    
    for key, value in optimizations.items():
        os.environ[key] = value
        print(f"환경변수 설정: {key}={value}")
    
    # 2. real_analysis_engine.py 최적화
    engine_file = Path("core/real_analysis_engine.py")
    if engine_file.exists():
        print("real_analysis_engine.py 최적화 중...")
        
        with open(engine_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Whisper 모델을 tiny로 기본 설정
        if 'whisper.load_model("base"' in content:
            content = content.replace('whisper.load_model("base"', 'whisper.load_model("tiny"')
            print("✓ Whisper 기본 모델을 tiny로 변경")
        
        # 모델 사전 로딩 최적화
        if 'self.whisper_model = None' in content:
            # 이미 최적화되어 있음
            print("✓ Lazy loading 이미 적용됨")
        
        with open(engine_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # 3. 빠른 설정 파일 생성
    fast_config = {
        "whisper_model": "tiny",
        "batch_size": 1,
        "max_file_size_mb": 10,
        "enable_gpu": False,
        "cache_models": True
    }
    
    import json
    with open("fast_config.json", 'w', encoding='utf-8') as f:
        json.dump(fast_config, f, indent=2)
    
    print("✓ 빠른 설정 파일 생성: fast_config.json")

def create_fast_test_script():
    """빠른 테스트 스크립트 생성"""
    print("\\n=== 빠른 테스트 스크립트 생성 ===")
    
    script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최적화된 빠른 테스트
"""
import os
import time
import json
from pathlib import Path

# 최적화 설정 로드
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

def fast_single_file_test(file_path):
    """단일 파일 빠른 테스트"""
    print(f"테스트: {Path(file_path).name}")
    
    try:
        # 파일 크기 체크
        size_mb = Path(file_path).stat().st_size / 1024 / 1024
        if size_mb > 10:  # 10MB 제한
            print(f"  스킵: 파일 너무 큼 ({size_mb:.1f}MB)")
            return False
        
        start = time.time()
        
        # 확장자별 처리
        ext = Path(file_path).suffix.lower()
        
        if ext in ['.m4a', '.wav', '.mp3']:
            # 오디오 처리 (최적화)
            import whisper
            model = whisper.load_model("tiny", device="cpu")
            result = model.transcribe(str(file_path))
            text = result.get('text', '')[:100]
            print(f"  STT 결과: {text}...")
            
        elif ext in ['.jpg', '.jpeg', '.png']:
            # 이미지 처리 (최적화)
            import easyocr
            reader = easyocr.Reader(['ko', 'en'], gpu=False)
            results = reader.readtext(str(file_path))
            text = ' '.join([item[1] for item in results[:3]])
            print(f"  OCR 결과: {text}...")
        
        duration = time.time() - start
        print(f"  처리 시간: {duration:.1f}초")
        
        return True
        
    except Exception as e:
        print(f"  오류: {str(e)[:50]}...")
        return False

def main():
    """메인 실행"""
    print("=== 최적화된 빠른 테스트 ===")
    
    # user_files에서 작은 파일들 찾기
    user_files = Path("user_files")
    test_files = []
    
    for file_path in user_files.rglob("*"):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / 1024 / 1024
            ext = file_path.suffix.lower()
            
            if ext in ['.m4a', '.wav', '.mp3', '.jpg', '.jpeg', '.png'] and size_mb < 10:
                test_files.append((file_path, size_mb))
    
    # 크기순 정렬
    test_files.sort(key=lambda x: x[1])
    
    print(f"테스트 대상: {len(test_files)}개 파일")
    
    # 최대 3개만 테스트
    success_count = 0
    for file_path, size_mb in test_files[:3]:
        if fast_single_file_test(file_path):
            success_count += 1
    
    print(f"\\n완료: {success_count}/{min(3, len(test_files))} 성공")

if __name__ == "__main__":
    main()
'''
    
    with open("optimized_fast_test.py", 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✓ 최적화된 테스트 스크립트 생성: optimized_fast_test.py")

def main():
    """메인 실행"""
    print("=== 솔로몬드 AI 성능 최적화 적용 ===")
    
    # 1. Whisper 최적화
    apply_whisper_optimization()
    
    # 2. 빠른 테스트 스크립트 생성
    create_fast_test_script()
    
    print("\\n=== 최적화 완료 ===")
    print("다음 명령으로 빠른 테스트 실행:")
    print("python optimized_fast_test.py")

if __name__ == "__main__":
    main()