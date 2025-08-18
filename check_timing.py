#!/usr/bin/env python3
"""
Quick timing checker for analysis results
"""

import json
import glob
from datetime import datetime

def check_recent_timings():
    """최근 분석 시간 결과 확인"""
    print("=== Recent Analysis Timings ===")
    
    # JSON 로그 파일들 찾기
    log_files = glob.glob("analysis_timing_*.json") + glob.glob("performance_log_*.json")
    
    if not log_files:
        print("No timing log files found.")
        print("Run analysis first to generate timing data.")
        return
    
    # 최신 파일 순으로 정렬
    log_files.sort(reverse=True)
    
    print(f"Found {len(log_files)} timing log files:")
    print("-" * 40)
    
    for i, log_file in enumerate(log_files[:5]):  # 최근 5개만
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            task_name = data.get('task_name', 'Unknown')
            total_time = data.get('total_seconds', data.get('total_time_seconds', 0))
            checkpoints = data.get('checkpoints', [])
            
            print(f"{i+1}. {log_file}")
            print(f"   Task: {task_name}")
            print(f"   Time: {total_time:.1f}s ({total_time/60:.1f}min)")
            print(f"   Steps: {len(checkpoints)}")
            
            if checkpoints:
                print("   Timeline:")
                for checkpoint in checkpoints[-3:]:  # 마지막 3단계만
                    step = checkpoint.get('step', 'Unknown')
                    elapsed = checkpoint.get('elapsed_seconds', 0)
                    print(f"     - {step}: {elapsed:.1f}s")
            print()
            
        except Exception as e:
            print(f"   Error reading {log_file}: {e}")
    
    print("-" * 40)
    print("Use 'python simple_performance_monitor.py' for new analysis timing")

if __name__ == "__main__":
    check_recent_timings()