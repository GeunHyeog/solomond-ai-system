#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ThreadPool 이슈 간단 수정 도구
"""

import re
from pathlib import Path

def fix_threadpool_in_file(file_path):
    """파일의 ThreadPool 문제 수정"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # ThreadPoolExecutor 패턴 수정
        # 기본 패턴: executor = ThreadPoolExecutor() 형태를 with문으로 변경
        pattern1 = r'(\s*)(executor\s*=\s*ThreadPoolExecutor\([^)]*\))\s*\n'
        replacement1 = r'\1with ThreadPoolExecutor() as executor:\n\1    # ThreadPool 작업을 여기에 배치하세요\n'
        
        content = re.sub(pattern1, replacement1, content, flags=re.MULTILINE)
        
        # import 확인 및 추가
        if 'ThreadPoolExecutor' in content and 'from concurrent.futures import ThreadPoolExecutor' not in content:
            if 'from concurrent.futures' not in content:
                # import 추가
                import_line = 'from concurrent.futures import ThreadPoolExecutor\n'
                
                # 다른 import 문 뒤에 추가
                import_match = re.search(r'((?:^import .*?\n|^from .*? import .*?\n)+)', content, re.MULTILINE)
                if import_match:
                    content = content[:import_match.end()] + import_line + content[import_match.end():]
                else:
                    content = import_line + content
        
        # 변경사항이 있으면 저장
        if content != original_content:
            # 백업 생성
            backup_path = str(file_path) + '.backup'
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            # 수정된 내용 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"SUCCESS: {file_path.name} ThreadPool 수정 완료")
            print(f"INFO: 백업 파일 생성: {backup_path}")
            return True
        else:
            print(f"INFO: {file_path.name} 수정 사항 없음")
            return False
            
    except Exception as e:
        print(f"ERROR: {file_path.name} 수정 실패: {e}")
        return False

def main():
    """메인 실행"""
    print("ThreadPool 이슈 자동 수정 도구")
    print("=" * 40)
    
    # 수정할 파일들
    files_to_fix = [
        "conference_analysis_COMPLETE_WORKING.py",
        "hybrid_compute_manager.py"
    ]
    
    # 추가로 core 디렉토리의 파일들도 검사
    core_files = list(Path("core").glob("*.py")) if Path("core").exists() else []
    
    success_count = 0
    total_count = 0
    
    # 주요 파일들 수정
    for file_name in files_to_fix:
        file_path = Path(file_name)
        if file_path.exists():
            total_count += 1
            print(f"\n[{file_name}]")
            if fix_threadpool_in_file(file_path):
                success_count += 1
        else:
            print(f"WARNING: {file_name} 파일을 찾을 수 없음")
    
    # core 디렉토리 파일들 중 일부 검사
    important_core_files = [
        "core/multimodal_pipeline.py",
        "core/batch_processing_engine.py",
        "core/parallel_processor.py"
    ]
    
    for file_name in important_core_files:
        file_path = Path(file_name)
        if file_path.exists():
            total_count += 1
            print(f"\n[{file_name}]")
            if fix_threadpool_in_file(file_path):
                success_count += 1
    
    print(f"\n{'='*40}")
    print(f"수정 결과: {success_count}/{total_count} 파일 수정됨")
    
    if success_count > 0:
        print("SUCCESS: ThreadPool 이슈 수정 완료")
        print("INFO: 백업 파일들이 생성되었습니다.")
    else:
        print("INFO: 수정이 필요한 ThreadPool 이슈가 발견되지 않음")

if __name__ == "__main__":
    main()