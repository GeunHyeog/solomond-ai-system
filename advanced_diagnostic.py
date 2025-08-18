#!/usr/bin/env python3
"""
Advanced System Diagnostic
Find additional issues beyond memory problems
"""

import json
import os
import sys
import subprocess
from datetime import datetime, timedelta
import glob

def check_import_issues():
    """Import 관련 세부 문제 체크"""
    print("=== IMPORT DETAILED CHECK ===")
    
    issues = []
    
    try:
        # 실제 분석 엔진 import 테스트
        from core.real_analysis_engine import analyze_file_real
        print("OK: Real analysis engine")
    except Exception as e:
        issues.append(f"CRITICAL: Real analysis engine import failed - {e}")
        
    try:
        # MCP 통합 테스트
        from mcp_auto_integration_wrapper import smart_mcp_enhance
        print("OK: MCP integration")
    except Exception as e:
        issues.append(f"HIGH: MCP integration failed - {e}")
        
    try:
        # 고급 모듈들
        from core.comprehensive_message_extractor import extract_speaker_message
        print("OK: Message extractor")
    except Exception as e:
        issues.append(f"MEDIUM: Message extractor failed - {e}")
        
    try:
        from core.large_file_handler import large_file_handler
        print("OK: Large file handler")
    except Exception as e:
        issues.append(f"LOW: Large file handler failed - {e}")
    
    return issues

def check_file_analysis_capabilities():
    """파일 분석 기능 테스트"""
    print("\n=== FILE ANALYSIS CAPABILITIES ===")
    
    issues = []
    
    # 테스트용 임시 파일 생성
    test_files = {
        "test.txt": "Test content for analysis",
        "test.json": '{"test": "data"}'
    }
    
    for filename, content in test_files.items():
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 문서 프로세서 테스트
            from core.document_processor import document_processor
            result = document_processor.process_document(filename)
            
            if result.get('status') == 'success':
                print(f"OK: {filename} processing")
            else:
                issues.append(f"MEDIUM: {filename} processing failed - {result.get('error', 'Unknown')}")
            
            # 파일 정리
            os.remove(filename)
            
        except Exception as e:
            issues.append(f"HIGH: {filename} test failed - {e}")
            try:
                os.remove(filename)
            except:
                pass
    
    return issues

def check_ui_functionality():
    """UI 기능 체크"""
    print("\n=== UI FUNCTIONALITY CHECK ===")
    
    issues = []
    
    try:
        # Streamlit 프로세스 포트 확인
        result = subprocess.run([
            'netstat', '-an'
        ], capture_output=True, text=True, timeout=10)
        
        if ':8503' in result.stdout:
            print("OK: Port 8503 is listening")
        else:
            issues.append("HIGH: Port 8503 not accessible")
            
    except Exception as e:
        issues.append(f"MEDIUM: Port check failed - {e}")
    
    # UI 파일 크기 체크
    ui_file = "jewelry_stt_ui_v23_real.py"
    if os.path.exists(ui_file):
        size = os.path.getsize(ui_file)
        if size < 50000:  # 50KB 미만
            issues.append(f"HIGH: UI file too small ({size} bytes)")
        else:
            print(f"OK: UI file size ({size} bytes)")
    
    return issues

def check_performance_bottlenecks():
    """성능 병목 지점 체크"""
    print("\n=== PERFORMANCE BOTTLENECKS ===")
    
    issues = []
    
    # 최근 성능 로그 분석
    timing_files = glob.glob("analysis_timing_*.json")
    
    if timing_files:
        slow_operations = []
        
        for log_file in timing_files[-5:]:  # 최근 5개
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                
                total_time = data.get('total_seconds', 0)
                if total_time > 60:  # 1분 이상
                    slow_operations.append((log_file, total_time))
                    
            except:
                continue
        
        if slow_operations:
            issues.append(f"MEDIUM: {len(slow_operations)} slow operations detected")
            for log_file, time_taken in slow_operations[:2]:  # 상위 2개만
                print(f"  SLOW: {log_file} took {time_taken:.1f}s")
        else:
            print("OK: No slow operations detected")
    else:
        print("INFO: No performance data available")
    
    return issues

def check_error_patterns():
    """오류 패턴 분석"""
    print("\n=== ERROR PATTERN ANALYSIS ===")
    
    issues = []
    error_keywords = ["error", "exception", "failed", "timeout", "cannot", "unable"]
    
    # Python 파일들에서 TODO, FIXME, XXX 찾기
    python_files = glob.glob("*.py") + glob.glob("core/*.py")
    
    todo_count = 0
    error_prone_files = []
    
    for py_file in python_files[:10]:  # 상위 10개 파일만
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
                
                # TODO/FIXME 카운트
                todo_count += content.count('todo') + content.count('fixme') + content.count('xxx')
                
                # 에러 관련 코드 패턴
                error_patterns = content.count('try:') - content.count('except')
                if error_patterns > 2:  # try 블록이 except보다 많이 초과
                    error_prone_files.append(py_file)
                    
        except:
            continue
    
    if todo_count > 20:
        issues.append(f"LOW: {todo_count} TODO/FIXME items found")
    
    if error_prone_files:
        issues.append(f"MEDIUM: {len(error_prone_files)} files have incomplete error handling")
    
    print(f"INFO: {todo_count} TODO items, {len(error_prone_files)} files need error handling review")
    
    return issues

def run_advanced_diagnostic():
    """고급 진단 실행"""
    print("ADVANCED SYSTEM DIAGNOSTIC")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")  
    print("=" * 50)
    
    all_issues = []
    
    # 각 진단 항목 실행
    all_issues.extend(check_import_issues())
    all_issues.extend(check_file_analysis_capabilities()) 
    all_issues.extend(check_ui_functionality())
    all_issues.extend(check_performance_bottlenecks())
    all_issues.extend(check_error_patterns())
    
    # 결과 정리 및 우선순위 분류
    critical_issues = [i for i in all_issues if i.startswith('CRITICAL')]
    high_issues = [i for i in all_issues if i.startswith('HIGH')]
    medium_issues = [i for i in all_issues if i.startswith('MEDIUM')]
    low_issues = [i for i in all_issues if i.startswith('LOW')]
    
    print("\n" + "=" * 50)
    print("ADVANCED DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    print(f"Total Issues Found: {len(all_issues)}")
    print(f"Critical: {len(critical_issues)}")
    print(f"High: {len(high_issues)}")
    print(f"Medium: {len(medium_issues)}")
    print(f"Low: {len(low_issues)}")
    
    # 문제별 출력
    if critical_issues:
        print("\nCRITICAL ISSUES:")
        for issue in critical_issues:
            print(f"  {issue}")
    
    if high_issues:
        print("\nHIGH PRIORITY ISSUES:")
        for issue in high_issues:
            print(f"  {issue}")
    
    if medium_issues:
        print("\nMEDIUM ISSUES:")
        for issue in medium_issues[:3]:  # 상위 3개만
            print(f"  {issue}")
        if len(medium_issues) > 3:
            print(f"  ... and {len(medium_issues) - 3} more medium issues")
    
    # 다음 단계 추천
    print("\n=== NEXT RECOMMENDED ACTIONS ===")
    
    if critical_issues:
        print("1. Fix critical issues immediately")
        print("2. Test system functionality after each fix")
    elif high_issues:
        print("1. Address high priority issues")
        print("2. Monitor system performance")
    elif medium_issues:
        print("1. Plan medium issue resolution")
        print("2. Continue normal operations")
    else:
        print("1. System is stable - monitor performance")
        print("2. Regular maintenance recommended")
    
    return {
        'total': len(all_issues),
        'critical': len(critical_issues),
        'high': len(high_issues),
        'medium': len(medium_issues),
        'low': len(low_issues),
        'issues': all_issues
    }

if __name__ == "__main__":
    result = run_advanced_diagnostic()
    
    # 진단 결과를 파일로 저장
    with open(f"advanced_diagnostic_{datetime.now().strftime('%H%M%S')}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    sys.exit(result['critical'] + result['high'])  # Critical + High 문제 수를 exit code로