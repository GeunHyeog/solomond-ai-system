#!/usr/bin/env python3
"""
Memory-optimized Streamlit launcher
"""

import os
import subprocess
import sys

def start_optimized_streamlit():
    """메모리 최적화된 Streamlit 시작"""
    
    # CPU 모드 강제 설정 (GPU 메모리 절약)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
    
    # Python 메모리 최적화
    os.environ['PYTHONOPTIMIZE'] = '2'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    
    # Streamlit 메모리 설정
    os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '200'  # 200MB로 제한
    os.environ['STREAMLIT_SERVER_MAX_MESSAGE_SIZE'] = '200'
    
    print("Starting memory-optimized Streamlit...")
    print("CPU-only mode enabled")
    print("Memory limits applied")
    print("-" * 40)
    
    # Streamlit 실행
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 
        'jewelry_stt_ui_v23_real.py',
        '--server.port', '8503',
        '--server.maxUploadSize', '200',
        '--server.maxMessageSize', '200'
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    start_optimized_streamlit()