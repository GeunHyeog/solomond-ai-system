#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test for comprehensive message extraction
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

sys.path.insert(0, str(Path(__file__).parent))

def test_message_extraction():
    """Test message extraction"""
    
    print("Testing comprehensive message extraction...")
    
    try:
        from core.comprehensive_message_extractor import extract_comprehensive_messages
        print("SUCCESS: Message extractor imported")
    except ImportError as e:
        print(f"FAILED: Import error: {e}")
        return
    
    # Test text
    test_text = """
    안녕하세요. 다이아몬드 반지를 찾고 있어요.
    약혼반지로 쓸 건데 1캐럿 정도로 생각하고 있습니다.
    GIA 인증서 있는 걸로요. 가격이 얼마나 할까요?
    """
    
    try:
        result = extract_comprehensive_messages(test_text)
        
        if result.get('status') == 'success':
            print("SUCCESS: Message extraction completed")
            
            main_summary = result.get('main_summary', {})
            
            if main_summary.get('one_line_summary'):
                print(f"Key message: {main_summary['one_line_summary']}")
            
            if main_summary.get('customer_status'):
                print(f"Customer status: {main_summary['customer_status']}")
                
            print("Test PASSED!")
            
        else:
            print(f"FAILED: Analysis failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"FAILED: Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_message_extraction()