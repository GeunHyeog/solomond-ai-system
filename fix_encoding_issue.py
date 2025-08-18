#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
윈도우 인코딩 문제 해결
"""
import os
import sys
import codecs

def fix_windows_encoding():
    """윈도우 인코딩 문제 해결"""
    print("윈도우 인코딩 문제 해결 중...")
    
    # 1. 환경 변수 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '1'
    
    # 2. stdout/stderr 안전 설정
    if sys.platform == 'win32':
        try:
            # 기존 stdout이 이미 설정되어 있는지 확인
            if not hasattr(sys.stdout, 'buffer'):
                # UTF-8로 재설정
                sys.stdout = codecs.getwriter('utf-8')(
                    sys.stdout.detach() if hasattr(sys.stdout, 'detach') else sys.stdout.buffer
                )
                sys.stderr = codecs.getwriter('utf-8')(
                    sys.stderr.detach() if hasattr(sys.stderr, 'detach') else sys.stderr.buffer
                )
                print("stdout/stderr UTF-8 설정 완료")
        except (AttributeError, OSError) as e:
            print(f"인코딩 설정 실패 (무시 가능): {e}")
    
    # 3. jewelry_stt_ui_v23_real.py 파일 수정
    try:
        with open('jewelry_stt_ui_v23_real.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 문제가 되는 print 문들을 안전한 버전으로 교체
        replacements = [
            ('print("[SUCCESS] 실제 분석 엔진 로드 완료")', 'print("[SUCCESS] Real analysis engine loaded")'),
            ('print("[SUCCESS] 고급 모니터링 시스템 로드 완료")', 'print("[SUCCESS] Advanced monitoring loaded")'),
            ('print("[SUCCESS] YouTube 실시간 처리 시스템 로드 완료")', 'print("[SUCCESS] YouTube processor loaded")'),
            ('print("[SUCCESS] 대용량 파일 핸들러 로드 완료")', 'print("[SUCCESS] Large file handler loaded")'),
            ('print("[SUCCESS] 브라우저 자동화 엔진 로드 완료")', 'print("[SUCCESS] Browser automation loaded")'),
            ('print("[SUCCESS] MCP 브라우저 통합 모듈 로드 완료")', 'print("[SUCCESS] MCP browser integration loaded")'),
        ]
        
        for old, new in replacements:
            content = content.replace(old, new)
        
        # 인코딩 안전 설정 추가
        encoding_fix = '''
# 윈도우 인코딩 문제 해결
import os
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    try:
        import codecs
        if hasattr(sys.stdout, 'detach'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except:
        pass  # 이미 설정되어 있거나 Streamlit 환경

'''
        
        # 파일 시작 부분에 인코딩 수정 코드 추가
        if 'PYTHONIOENCODING' not in content:
            import_pos = content.find('import streamlit as st')
            if import_pos != -1:
                content = content[:import_pos] + encoding_fix + content[import_pos:]
        
        # 수정된 파일 저장
        with open('jewelry_stt_ui_v23_real_fixed.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("수정된 UI 파일 생성: jewelry_stt_ui_v23_real_fixed.py")
        
    except Exception as e:
        print(f"파일 수정 실패: {e}")
    
    # 4. 간단한 시작 스크립트 생성
    start_script = '''@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=1
python -m streamlit run jewelry_stt_ui_v23_real_fixed.py --server.port 8503
pause
'''
    
    with open('start_solomond_safe.bat', 'w', encoding='utf-8') as f:
        f.write(start_script)
    
    print("안전 시작 스크립트 생성: start_solomond_safe.bat")
    
    # 5. 종합 분석 전용 스크립트 생성
    comprehensive_script = '''#!/usr/bin/env python3
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
'''
    
    with open('run_comprehensive_analysis.py', 'w', encoding='utf-8') as f:
        f.write(comprehensive_script)
    
    print("종합 분석 실행 스크립트 생성: run_comprehensive_analysis.py")
    
    print("\\n해결 완료! 다음 중 하나를 선택하세요:")
    print("1. start_solomond_safe.bat (배치 파일)")
    print("2. python run_comprehensive_analysis.py (Python 스크립트)")
    print("3. python -m streamlit run jewelry_stt_ui_v23_real_fixed.py --server.port 8503")

if __name__ == "__main__":
    fix_windows_encoding()