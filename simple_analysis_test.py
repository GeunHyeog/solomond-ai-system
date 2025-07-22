#!/usr/bin/env python3
"""
간단한 실제 분석 테스트 - 인코딩 문제 없이 직접 테스트
"""

import sys
import os
from pathlib import Path
import tempfile

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# UTF-8 인코딩 강제 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'

def test_message_extraction():
    """메시지 추출 엔진 직접 테스트"""
    
    test_texts = [
        "안녕하세요. 다이아몬드 반지를 찾고 있어요. 1캐럿 정도로 생각하고 있습니다.",
        "오늘 날씨가 정말 좋네요. 커피 한 잔 마시면서 산책하고 싶어요.",
        "파이썬 프로그래밍을 배우고 있어요. 데이터 분석에 관심이 많습니다."
    ]
    
    try:
        from core.comprehensive_message_extractor import extract_comprehensive_messages
        print("SUCCESS: Message extractor loaded")
        
        results = []
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text[:30]}...")
            result = extract_comprehensive_messages(text)
            
            if result.get("status") == "success":
                summary = result.get("main_summary", {}).get("one_line_summary", "No summary")
                print(f"Result: {summary}")
                results.append(summary)
            else:
                print(f"FAILED: {result.get('error', 'Unknown error')}")
                results.append("FAILED")
        
        # 결과 분석
        unique_results = len(set(results))
        print(f"\nAnalysis: {unique_results} unique results out of {len(results)} tests")
        
        if unique_results > 1:
            print("SUCCESS: Real analysis working - different results for different inputs")
            return True
        else:
            print("WARNING: All results are the same - possible fake analysis")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_document_analysis():
    """문서 분석 직접 테스트"""
    
    try:
        from core.real_analysis_engine import global_analysis_engine
        print("SUCCESS: Real analysis engine loaded")
        
        # 임시 텍스트 파일 생성
        test_content = "안녕하세요. 다이아몬드 반지 가격이 궁금합니다. 1캐럿으로 생각하고 있어요."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_file = f.name
        
        try:
            result = global_analysis_engine.analyze_document_file(temp_file)
            if result.get('status') == 'success':
                print("SUCCESS: Document analysis working")
                print(f"Text extracted: {result.get('text', '')[:100]}...")
                return True
            else:
                print(f"FAILED: {result.get('error', 'Unknown error')}")
                return False
        finally:
            os.unlink(temp_file)
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Simple Analysis Test")
    print("=" * 50)
    
    # 테스트 1: 메시지 추출
    print("\n[Test 1] Message Extraction Engine")
    message_test = test_message_extraction()
    
    # 테스트 2: 문서 분석
    print("\n[Test 2] Document Analysis Engine")
    document_test = test_document_analysis()
    
    # 전체 결과
    print("\n" + "=" * 50)
    print("Overall Results:")
    print(f"Message Extraction: {'PASS' if message_test else 'FAIL'}")
    print(f"Document Analysis: {'PASS' if document_test else 'FAIL'}")
    
    if message_test and document_test:
        print("\nCONCLUSION: Real analysis systems are working!")
    else:
        print("\nCONCLUSION: Some analysis systems need fixing.")