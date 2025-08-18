#!/usr/bin/env python3
"""
Real-time Analysis Watcher
Watch your current analysis progress in real-time
"""

import time
import subprocess
import sys
from datetime import datetime

def watch_streamlit_process():
    """실시간으로 Streamlit 프로세스 모니터링"""
    print("=== Real-time Analysis Watcher ===")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("Watching for Streamlit analysis process...")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        while True:
            # 현재 시간
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Streamlit 프로세스 확인
            try:
                result = subprocess.run([
                    'wmic', 'process', 'where', 
                    "name='python.exe' and CommandLine like '%streamlit%'", 
                    'get', 'ProcessId,PageFileUsage'
                ], capture_output=True, text=True, timeout=5)
                
                if "ProcessId" in result.stdout:
                    # 프로세스 찾음
                    lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
                    if len(lines) > 1:  # 헤더 + 데이터
                        data_line = lines[1]
                        if data_line and not data_line.startswith("No Instance"):
                            parts = data_line.split()
                            if len(parts) >= 2:
                                memory_kb = parts[0]
                                pid = parts[1]
                                memory_mb = float(memory_kb) / 1024 if memory_kb.isdigit() else 0
                                
                                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                                      f"Running: {elapsed:.0f}s, "
                                      f"Memory: {memory_mb:.1f}MB, "
                                      f"PID: {pid}", end="", flush=True)
                        else:
                            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                                  f"Elapsed: {elapsed:.0f}s - No active analysis", end="", flush=True)
                    else:
                        print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                              f"Elapsed: {elapsed:.0f}s - Streamlit not found", end="", flush=True)
                else:
                    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Elapsed: {elapsed:.0f}s - Checking...", end="", flush=True)
                          
            except Exception as e:
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Elapsed: {elapsed:.0f}s - Monitor error", end="", flush=True)
            
            time.sleep(2)  # 2초마다 체크
            
    except KeyboardInterrupt:
        total_time = time.time() - start_time
        print(f"\n\n=== Monitoring Stopped ===")
        print(f"Total monitoring time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print("Analysis monitoring completed.")

if __name__ == "__main__":
    watch_streamlit_process()