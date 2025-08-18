#!/usr/bin/env python3
"""
실시간 분석 성능 모니터링 도구
사용자의 실제 파일 분석 테스트 소요시간 측정
"""

import time
import psutil
import json
from datetime import datetime
from pathlib import Path
import sys

class AnalysisPerformanceMonitor:
    """실시간 분석 성능 모니터링"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.performance_log = []
        
    def start_monitoring(self, task_name: str = "Analysis"):
        """모니터링 시작"""
        self.start_time = time.time()
        self.task_name = task_name
        
        print(f"🚀 {task_name} 모니터링 시작")
        print(f"⏰ 시작 시간: {datetime.now().strftime('%H:%M:%S')}")
        print(f"💾 초기 메모리: {self.initial_memory:.1f}MB")
        print("-" * 50)
        
        return self.start_time
    
    def log_checkpoint(self, step_name: str):
        """중간 체크포인트 기록"""
        if not self.start_time:
            print("❌ 모니터링이 시작되지 않았습니다.")
            return
            
        current_time = time.time()
        elapsed = current_time - self.start_time
        current_memory = self.process.memory_info().rss / (1024 * 1024)
        memory_delta = current_memory - self.initial_memory
        
        checkpoint = {
            "step": step_name,
            "elapsed_seconds": elapsed,
            "current_memory_mb": current_memory,
            "memory_delta_mb": memory_delta,
            "timestamp": datetime.now().isoformat()
        }
        
        self.performance_log.append(checkpoint)
        
        print(f"📍 {step_name}")
        print(f"   ⏱️  경과시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
        print(f"   💾 메모리: {current_memory:.1f}MB (+{memory_delta:.1f}MB)")
        
        return checkpoint
    
    def finish_monitoring(self):
        """모니터링 종료 및 최종 결과"""
        if not self.start_time:
            print("❌ 모니터링이 시작되지 않았습니다.")
            return None
            
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        final_memory = self.process.memory_info().rss / (1024 * 1024)
        total_memory_delta = final_memory - self.initial_memory
        
        final_result = {
            "task_name": self.task_name,
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60,
            "initial_memory_mb": self.initial_memory,
            "final_memory_mb": final_memory,
            "total_memory_delta_mb": total_memory_delta,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
            "checkpoints": self.performance_log
        }
        
        print("-" * 50)
        print(f"🏁 {self.task_name} 완료!")
        print(f"⏰ 총 소요시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
        print(f"💾 총 메모리 증가: {total_memory_delta:.1f}MB")
        print(f"📈 평균 메모리 사용률: {(final_memory/psutil.virtual_memory().total*100):.1f}%")
        
        # 성능 로그 저장
        log_file = f"performance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        print(f"📋 성능 로그 저장: {log_file}")
        
        return final_result
    
    def get_current_stats(self):
        """현재 상태 확인"""
        if not self.start_time:
            return None
            
        current_time = time.time()
        elapsed = current_time - self.start_time
        current_memory = self.process.memory_info().rss / (1024 * 1024)
        memory_delta = current_memory - self.initial_memory
        
        return {
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60,
            "current_memory_mb": current_memory,
            "memory_delta_mb": memory_delta,
            "cpu_percent": self.process.cpu_percent()
        }

# 전역 모니터 인스턴스
global_monitor = AnalysisPerformanceMonitor()

def start_analysis_monitoring(task_name: str = "File Analysis"):
    """분석 모니터링 시작"""
    return global_monitor.start_monitoring(task_name)

def log_analysis_step(step_name: str):
    """분석 단계 기록"""
    return global_monitor.log_checkpoint(step_name)

def finish_analysis_monitoring():
    """분석 모니터링 완료"""
    return global_monitor.finish_monitoring()

def get_current_analysis_stats():
    """현재 분석 상태"""
    return global_monitor.get_current_stats()

if __name__ == "__main__":
    # 테스트 실행
    print("Performance Monitoring Test Started")
    
    # 모니터링 시작
    start_analysis_monitoring("Test Analysis")
    
    # 가상의 처리 단계들
    import time
    
    log_analysis_step("File Loading")
    time.sleep(2)
    
    log_analysis_step("Preprocessing")
    time.sleep(3)
    
    log_analysis_step("AI Model Execution")
    time.sleep(5)
    
    log_analysis_step("Post-processing")
    time.sleep(1)
    
    # 모니터링 완료
    result = finish_analysis_monitoring()
    
    print(f"\nTest Complete! Total time: {result['total_time_seconds']:.1f} seconds")