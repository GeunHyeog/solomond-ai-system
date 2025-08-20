#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 시스템 상태 확인 (Unicode 안전)
"""

import requests
import json
from pathlib import Path
import os
import sys

# 시스템 인코딩 강제 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'

def check_basic_status():
    """기본 시스템 상태 확인"""
    
    print("SOLOMOND AI System Check")
    print("=" * 40)
    
    score = 0
    
    # 1. Streamlit 확인
    print("\n1. Streamlit Server...")
    try:
        response = requests.get("http://localhost:8560", timeout=5)
        if response.status_code == 200:
            print("   OK: Server running")
            score += 20
        else:
            print(f"   WARNING: HTTP {response.status_code}")
            score += 10
    except:
        print("   ERROR: Server not responding")
    
    # 2. 파일 확인
    print("\n2. Essential Files...")
    files = [
        "conference_analysis_UNIFIED_COMPLETE.py",
        "core/unicode_safety_system.py",
        "core/enhanced_file_handler.py"
    ]
    
    for file in files:
        if Path(file).exists():
            print(f"   OK: {file}")
            score += 5
        else:
            print(f"   MISSING: {file}")
    
    # 3. 데이터 파일
    print("\n3. Data Files...")
    user_files = Path("user_files/JGA2025_D1")
    if user_files.exists():
        file_count = len(list(user_files.glob("*")))
        print(f"   OK: {file_count} files in JGA2025_D1")
        score += 15
    else:
        print("   WARNING: No test data")
    
    # 4. Python 패키지
    print("\n4. Python Packages...")
    packages = ['streamlit', 'torch', 'transformers', 'spacy']
    
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"   OK: {pkg}")
            score += 5
        except ImportError:
            print(f"   MISSING: {pkg}")
    
    # 5. Ollama
    print("\n5. Ollama Server...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"   OK: {len(models)} models")
            score += 15
        else:
            print("   WARNING: Ollama not responding")
            score += 5
    except:
        print("   ERROR: Ollama not available")
    
    # 결과
    print("\n" + "=" * 40)
    print(f"System Health: {score}/80 points")
    
    if score >= 70:
        status = "EXCELLENT"
    elif score >= 50:
        status = "GOOD"
    elif score >= 30:
        status = "FAIR"
    else:
        status = "POOR"
    
    print(f"Status: {status}")
    
    return score

def check_upload_function():
    """파일 업로드 기능 확인"""
    
    print("\n" + "=" * 40)
    print("File Upload System Check")
    
    # Enhanced file handler 확인
    try:
        from core.enhanced_file_handler import enhanced_handler
        print("OK: Enhanced file handler loaded")
        
        # 테스트 파일 확인
        test_dir = Path("test_files")
        if test_dir.exists():
            test_files = list(test_dir.glob("*"))
            print(f"OK: {len(test_files)} test files available")
        else:
            print("INFO: Creating test files...")
            enhanced_handler._create_test_files()
            print("OK: Test files created")
        
        # 로컬 파일 확인
        user_files = Path("user_files")
        if user_files.exists():
            folders = [f for f in user_files.iterdir() if f.is_dir()]
            print(f"OK: {len(folders)} folders in user_files")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    try:
        basic_score = check_basic_status()
        upload_ok = check_upload_function()
        
        print("\n" + "=" * 40)
        print("SUMMARY:")
        print(f"- Basic System: {basic_score}/80")
        print(f"- Upload System: {'OK' if upload_ok else 'ERROR'}")
        
        if basic_score >= 50 and upload_ok:
            print("RESULT: System is ready for use!")
            print("Access: http://localhost:8560")
        else:
            print("RESULT: System needs attention")
        
    except Exception as e:
        print(f"Check failed: {e}")
        sys.exit(1)