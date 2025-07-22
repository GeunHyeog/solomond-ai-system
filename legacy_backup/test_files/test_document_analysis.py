#!/usr/bin/env python3
"""
문서 분석 기능 테스트
"""

import os
import sys
sys.path.append('.')

from core.real_analysis_engine import analyze_file_real

def test_document_analysis():
    """문서 분석 기능 테스트"""
    
    print("[INFO] 문서 분석 기능 테스트 시작")
    print("=" * 50)
    
    # 테스트 파일
    test_file = "test_files/sample.docx"
    
    if not os.path.exists(test_file):
        print(f"[ERROR] 테스트 파일이 없습니다: {test_file}")
        print("[INFO] create_test_documents.py를 먼저 실행하세요.")
        return
    
    print(f"[INFO] 테스트 파일: {test_file}")
    
    try:
        # 문서 분석 실행
        result = analyze_file_real(test_file, "document")
        
        if result['status'] == 'success':
            print("[SUCCESS] 문서 분석 성공!")
            print(f"  - 파일 형식: {result['file_type']}")
            print(f"  - 텍스트 길이: {result['text_length']}자")
            print(f"  - 처리 시간: {result['processing_time']}초")
            print(f"  - 품질 점수: {result['quality_score']}")
            
            if result['jewelry_keywords']:
                print(f"  - 주얼리 키워드: {', '.join(result['jewelry_keywords'])}")
            else:
                print("  - 주얼리 키워드: 없음")
            
            print(f"  - 요약: {result['summary']}")
            
            # 문서 정보
            doc_info = result.get('document_info', {})
            if doc_info.get('paragraph_count'):
                print(f"  - 단락 수: {doc_info['paragraph_count']}")
            if doc_info.get('page_count'):
                print(f"  - 페이지 수: {doc_info['page_count']}")
            
            # 메타데이터
            metadata = result.get('document_metadata', {})
            if metadata.get('title'):
                print(f"  - 문서 제목: {metadata['title']}")
            if metadata.get('author'):
                print(f"  - 작성자: {metadata['author']}")
                
        else:
            print(f"[ERROR] 문서 분석 실패: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"[ERROR] 분석 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_document_analysis()