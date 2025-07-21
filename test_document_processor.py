#!/usr/bin/env python3
"""
문서 처리 모듈 테스트
"""

import os
import sys
sys.path.append('.')

from core.document_processor import document_processor

def test_document_processor():
    """문서 처리 모듈 테스트"""
    
    print("[INFO] 문서 처리 모듈 테스트 시작")
    print("=" * 50)
    
    # 설치 가이드 확인
    guide = document_processor.get_installation_guide()
    print(f"[INFO] 지원 형식: {guide['supported_formats']}")
    
    if guide['missing_packages']:
        print("[WARNING] 누락된 패키지:")
        for pkg in guide['missing_packages']:
            print(f"  - {pkg['package']}: {pkg['command']}")
        print(f"[INFO] 전체 설치: {guide['install_all']}")
    else:
        print("[SUCCESS] 모든 필요 패키지 설치됨")
    
    print()
    
    # 테스트 파일 생성 (간단한 텍스트 파일로 PDF/DOCX 시뮬레이션)
    test_files = []
    
    # 실제 파일이 있다면 테스트
    potential_files = [
        "test_files/sample.pdf",
        "test_files/sample.docx",
        "test_files/sample.doc"
    ]
    
    for file_path in potential_files:
        if os.path.exists(file_path):
            test_files.append(file_path)
    
    if not test_files:
        print("[INFO] 테스트할 문서 파일이 없습니다.")
        print("[INFO] test_files/ 폴더에 PDF, DOCX, DOC 파일을 추가하여 테스트할 수 있습니다.")
        return
    
    # 파일 처리 테스트
    for file_path in test_files:
        print(f"[INFO] 테스트 파일: {file_path}")
        
        result = document_processor.process_document(file_path)
        
        if result['status'] == 'success':
            print(f"[SUCCESS] 처리 성공")
            print(f"  - 파일 형식: {result['file_type']}")
            print(f"  - 텍스트 길이: {result['total_characters']}자")
            if 'page_count' in result:
                print(f"  - 페이지 수: {result['page_count']}")
            if 'paragraph_count' in result:
                print(f"  - 단락 수: {result['paragraph_count']}")
            
            # 텍스트 미리보기
            preview_text = result['extracted_text'][:200]
            print(f"  - 텍스트 미리보기: {preview_text}...")
            
        elif result['status'] == 'partial_success':
            print(f"[WARNING] 부분적 성공")
            print(f"  - 경고: {result.get('warning', 'N/A')}")
            
        else:
            print(f"[ERROR] 처리 실패: {result.get('error', 'Unknown error')}")
            if 'install_command' in result:
                print(f"  - 설치 명령: {result['install_command']}")
        
        print()

if __name__ == "__main__":
    test_document_processor()