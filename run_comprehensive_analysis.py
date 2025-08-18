#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
종합 분석 전용 실행 스크립트 - 인코딩 안전
"""
import os
import sys

# 인코딩 문제 해결
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '1'

# 안전한 print 함수
def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('utf-8', errors='ignore').decode('utf-8'))
    except:
        print("Output encoding error")

if __name__ == "__main__":
    safe_print("=== 솔로몬드 AI 종합 분석 시작 ===")
    
    try:
        # Streamlit 실행
        import subprocess
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', 
            'jewelry_stt_ui_v23_real_fixed.py', 
            '--server.port', '8503'
        ]
        subprocess.run(cmd)
    except Exception as e:
        safe_print(f"실행 오류: {e}")
        safe_print("수동으로 다음 명령 실행:")
        safe_print("python -m streamlit run jewelry_stt_ui_v23_real_fixed.py --server.port 8503")
