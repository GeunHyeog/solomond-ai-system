#!/usr/bin/env python3
"""
Quick System Diagnostic - ASCII only
"""

import subprocess
import sys
import os
import glob
from datetime import datetime

def check_memory_usage():
    """메모리 사용량 체크"""
    print("=== MEMORY USAGE CHECK ===")
    try:
        result = subprocess.run([
            'wmic', 'process', 'where', 
            "name='python.exe' and CommandLine like '%streamlit%'", 
            'get', 'ProcessId,PageFileUsage'
        ], capture_output=True, text=True, timeout=10)
        
        if "ProcessId" in result.stdout:
            lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            if len(lines) > 1:
                data_line = lines[1]
                if data_line and not data_line.startswith("No Instance"):
                    parts = data_line.split()
                    if len(parts) >= 2:
                        memory_kb = float(parts[0]) if parts[0].isdigit() else 0
                        memory_mb = memory_kb / 1024
                        
                        print(f"Streamlit Memory Usage: {memory_mb:.1f}MB")
                        
                        if memory_mb > 8000:  # 8GB 이상
                            print("CRITICAL: Memory usage over 8GB - RESTART REQUIRED")
                            return "critical"
                        elif memory_mb > 4000:  # 4GB 이상
                            print("WARNING: High memory usage - monitoring needed")
                            return "high"
                        else:
                            print("OK: Normal memory usage")
                            return "normal"
        
        print("ERROR: Cannot detect Streamlit process")
        return "error"
        
    except Exception as e:
        print(f"ERROR: Memory check failed - {e}")
        return "error"

def check_essential_files():
    """필수 파일 체크"""
    print("\n=== ESSENTIAL FILES CHECK ===")
    
    files = [
        "jewelry_stt_ui_v23_real.py",
        "core/real_analysis_engine.py",
        "core/document_processor.py"
    ]
    
    missing_files = []
    for file_path in files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"OK: {file_path} ({size} bytes)")
        else:
            print(f"MISSING: {file_path}")
            missing_files.append(file_path)
    
    return missing_files

def check_temp_files():
    """임시 파일 체크"""
    print("\n=== TEMP FILES CHECK ===")
    
    patterns = ["tmp*", "temp*", "*.tmp", "analysis_timing_*.json"]
    total_temp = 0
    
    for pattern in patterns:
        files = glob.glob(pattern)
        total_temp += len(files)
        if files:
            print(f"Found {len(files)} files matching {pattern}")
    
    print(f"Total temp files: {total_temp}")
    
    if total_temp > 50:
        print("WARNING: Too many temp files - cleanup recommended")
        return "cleanup_needed"
    else:
        print("OK: Temp files normal")
        return "normal"

def check_import_errors():
    """Import 오류 체크"""
    print("\n=== IMPORT CHECK ===") 
    
    critical_modules = [
        "streamlit",
        "whisper", 
        "easyocr",
        "torch"
    ]
    
    failed_imports = []
    
    for module in critical_modules:
        try:
            __import__(module)
            print(f"OK: {module}")
        except ImportError:
            print(f"FAILED: {module}")
            failed_imports.append(module)
    
    return failed_imports

def main():
    print("SOLOMOND AI QUICK DIAGNOSTIC")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)
    
    issues = []
    
    # 메모리 체크
    memory_status = check_memory_usage()
    if memory_status == "critical":
        issues.append("CRITICAL: Memory over 8GB")
    elif memory_status == "high":
        issues.append("HIGH: Memory usage warning")
    
    # 파일 체크  
    missing_files = check_essential_files()
    if missing_files:
        issues.append(f"CRITICAL: Missing files: {', '.join(missing_files)}")
    
    # 임시 파일 체크
    temp_status = check_temp_files()
    if temp_status == "cleanup_needed":
        issues.append("MEDIUM: Temp file cleanup needed")
    
    # Import 체크
    failed_imports = check_import_errors()
    if failed_imports:
        issues.append(f"CRITICAL: Failed imports: {', '.join(failed_imports)}")
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    if not issues:
        print("STATUS: SYSTEM OK")
    else:
        print(f"STATUS: {len(issues)} ISSUES FOUND")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    
    # 첫 번째 문제 해결 방안 제시
    if issues:
        print("\n=== IMMEDIATE ACTION REQUIRED ===")
        first_issue = issues[0]
        
        if "Memory over 8GB" in first_issue:
            print("SOLUTION 1: Restart Streamlit server")
            print("  Command: wmic process where \"name='python.exe' and CommandLine like '%streamlit%'\" call terminate")
            print("  Then: python -m streamlit run jewelry_stt_ui_v23_real.py --server.port 8503")
            
        elif "Missing files" in first_issue:
            print("SOLUTION 1: Restore missing files from backup")
            print("  Check if files were accidentally deleted")
            
        elif "Failed imports" in first_issue:
            print("SOLUTION 1: Install missing modules")
            print("  Command: pip install streamlit whisper easyocr torch")
    
    return len(issues)

if __name__ == "__main__":
    issue_count = main()
    sys.exit(issue_count)  # 문제 개수를 exit code로 반환