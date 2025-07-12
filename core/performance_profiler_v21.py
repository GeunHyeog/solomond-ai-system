"""
🚀 솔로몬드 AI v2.1.2 - 성능 프로파일러
실시간 성능 모니터링 및 병목지점 분석

주요 기능:
- CPU/메모리/디스크 I/O 실시간 모니터링
- 모듈별 성능 분석
- 대용량 파일 처리 최적화 권장
- 메모리 누수 감지 및 경고
"""

import psutil
import time
import threading
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import tracemalloc
import gc
import logging
from dataclasses import dataclass, asdict
from collections import deque
import functools

@dataclass
class PerformanceMetric:
    """성능 지표 데이터 클래스"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    active_threads: int
    open_files: int

@dataclass
class ModulePerformance:
    """모듈별 성능 데이터"""
    module_name: str
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage: float
    io_operations: int
    error_count: int

class PerformanceProfiler:
    """실시간 성능 프로파일러"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.module_stats = {}
        self.is_monitoring = False
        self.monitor_thread = None
        self.start_time = time.time()
        
        # 메모리 추적 시작
        tracemalloc.start()
        
        # 기준 성능 값들
        self.baseline_metrics = {
            "cpu_threshold": 80.0,
            "memory_threshold": 85.0,
            "disk_io_threshold": 100.0,  # MB/s
            "response_time_threshold": 5.0,  # seconds
        }
        
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self, interval: float = 1.0):
        """실시간 모니터링 시작"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("🔍 성능 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("⏹️ 성능 모니터링 중지")
    
    def _monitor_loop(self, interval: float):
        """모니터링 루프"""
        last_disk_io = psutil.disk_io_counters()
        last_net_io = psutil.net_io_counters()
        
        while self.is_monitoring:
            try:
                # 현재 시스템 상태 수집
                current_time = datetime.now().isoformat()
                
                # CPU 사용률
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # 메모리 사용률
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_mb = memory.used / (1024 * 1024)
                
                # 디스크 I/O
                current_disk_io = psutil.disk_io_counters()
                disk_read_mb = (current_disk_io.read_bytes - last_disk_io.read_bytes) / (1024 * 1024) / interval
                disk_write_mb = (current_disk_io.write_bytes - last_disk_io.write_bytes) / (1024 * 1024) / interval
                last_disk_io = current_disk_io
                
                # 네트워크 I/O
                current_net_io = psutil.net_io_counters()
                net_sent_mb = (current_net_io.bytes_sent - last_net_io.bytes_sent) / (1024 * 1024) / interval
                net_recv_mb = (current_net_io.bytes_recv - last_net_io.bytes_recv) / (1024 * 1024) / interval
                last_net_io = current_net_io
                
                # 프로세스 정보
                process = psutil.Process()
                active_threads = process.num_threads()
                open_files = len(process.open_files())
                
                # 메트릭 생성
                metric = PerformanceMetric(
                    timestamp=current_time,
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    memory_used_mb=memory_used_mb,
                    disk_io_read_mb=disk_read_mb,
                    disk_io_write_mb=disk_write_mb,
                    network_io_sent_mb=net_sent_mb,
                    network_io_recv_mb=net_recv_mb,
                    active_threads=active_threads,
                    open_files=open_files
                )
                
                self.metrics_history.append(metric)
                
                # 임계값 초과 시 경고
                self._check_thresholds(metric)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"모니터링 오류: {e}")
                time.sleep(interval)
    
    def _check_thresholds(self, metric: PerformanceMetric):
        """임계값 확인 및 경고"""
        warnings = []
        
        if metric.cpu_percent > self.baseline_metrics["cpu_threshold"]:
            warnings.append(f"⚠️ 높은 CPU 사용률: {metric.cpu_percent:.1f}%")
        
        if metric.memory_percent > self.baseline_metrics["memory_threshold"]:
            warnings.append(f"⚠️ 높은 메모리 사용률: {metric.memory_percent:.1f}%")
        
        if metric.disk_io_read_mb > self.baseline_metrics["disk_io_threshold"]:
            warnings.append(f"⚠️ 높은 디스크 읽기: {metric.disk_io_read_mb:.1f} MB/s")
        
        if metric.disk_io_write_mb > self.baseline_metrics["disk_io_threshold"]:
            warnings.append(f"⚠️ 높은 디스크 쓰기: {metric.disk_io_write_mb:.1f} MB/s")
        
        if warnings:
            for warning in warnings:
                self.logger.warning(warning)
    
    def profile_function(self, func_name: str = None):
        """함수 성능 프로파일링 데코레이터"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                name = func_name or f"{func.__module__}.{func.__name__}"
                
                # 시작 시점 메트릭
                start_time = time.time()
                start_memory = self._get_memory_usage()
                tracemalloc_start = tracemalloc.take_snapshot()
                
                try:
                    # 함수 실행
                    result = func(*args, **kwargs)
                    
                    # 종료 시점 메트릭
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    tracemalloc_end = tracemalloc.take_snapshot()
                    
                    # 성능 데이터 계산
                    execution_time = end_time - start_time
                    memory_diff = end_memory - start_memory
                    
                    # 피크 메모리 사용량 계산
                    top_stats = tracemalloc_end.compare_to(tracemalloc_start, 'lineno')
                    peak_memory = sum(stat.size_diff for stat in top_stats) / (1024 * 1024)
                    
                    # 모듈 성능 기록
                    module_perf = ModulePerformance(
                        module_name=name,
                        execution_time=execution_time,
                        memory_usage_mb=memory_diff,
                        peak_memory_mb=abs(peak_memory),
                        cpu_usage=psutil.cpu_percent(),
                        io_operations=0,  # 필요시 구현
                        error_count=0
                    )
                    
                    self._record_module_performance(module_perf)
                    
                    # 성능 경고
                    if execution_time > self.baseline_metrics["response_time_threshold"]:
                        self.logger.warning(f"⏱️ 느린 함수 실행: {name} ({execution_time:.2f}초)")
                    
                    return result
                    
                except Exception as e:
                    # 에러 발생 시 기록
                    error_perf = ModulePerformance(
                        module_name=name,
                        execution_time=time.time() - start_time,
                        memory_usage_mb=0,
                        peak_memory_mb=0,
                        cpu_usage=0,
                        io_operations=0,
                        error_count=1
                    )
                    self._record_module_performance(error_perf)
                    raise
            
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """현재 메모리 사용량 (MB)"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def _record_module_performance(self, perf: ModulePerformance):
        """모듈 성능 기록"""
        if perf.module_name not in self.module_stats:
            self.module_stats[perf.module_name] = {
                "call_count": 0,
                "total_time": 0,
                "total_memory": 0,
                "peak_memory": 0,
                "error_count": 0,
                "avg_time": 0,
                "avg_memory": 0
            }
        
        stats = self.module_stats[perf.module_name]
        stats["call_count"] += 1
        stats["total_time"] += perf.execution_time
        stats["total_memory"] += perf.memory_usage_mb
        stats["peak_memory"] = max(stats["peak_memory"], perf.peak_memory_mb)
        stats["error_count"] += perf.error_count
        
        # 평균 계산
        stats["avg_time"] = stats["total_time"] / stats["call_count"]
        stats["avg_memory"] = stats["total_memory"] / stats["call_count"]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보"""
        if not self.metrics_history:
            return {"error": "모니터링 데이터가 없습니다"}
        
        # 최근 메트릭들
        recent_metrics = list(self.metrics_history)[-100:]  # 최근 100개
        
        # 평균 계산
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_disk_read = sum(m.disk_io_read_mb for m in recent_metrics) / len(recent_metrics)
        avg_disk_write = sum(m.disk_io_write_mb for m in recent_metrics) / len(recent_metrics)
        
        # 피크 값
        peak_cpu = max(m.cpu_percent for m in recent_metrics)
        peak_memory = max(m.memory_percent for m in recent_metrics)
        peak_disk_read = max(m.disk_io_read_mb for m in recent_metrics)
        
        return {
            "monitoring_duration": time.time() - self.start_time,
            "total_samples": len(self.metrics_history),
            "averages": {
                "cpu_percent": round(avg_cpu, 2),
                "memory_percent": round(avg_memory, 2),
                "disk_read_mb_s": round(avg_disk_read, 2),
                "disk_write_mb_s": round(avg_disk_write, 2)
            },
            "peaks": {
                "cpu_percent": round(peak_cpu, 2),
                "memory_percent": round(peak_memory, 2),
                "disk_read_mb_s": round(peak_disk_read, 2)
            },
            "current_status": self._get_current_status(),
            "module_performance": self.module_stats,
            "recommendations": self._generate_recommendations()
        }
    
    def _get_current_status(self) -> Dict[str, str]:
        """현재 시스템 상태"""
        if not self.metrics_history:
            return {"status": "unknown"}
        
        latest = self.metrics_history[-1]
        
        status = "정상"
        if latest.cpu_percent > 80:
            status = "CPU 과부하"
        elif latest.memory_percent > 85:
            status = "메모리 부족"
        elif latest.disk_io_read_mb > 50 or latest.disk_io_write_mb > 50:
            status = "디스크 I/O 과부하"
        
        return {
            "status": status,
            "cpu": f"{latest.cpu_percent:.1f}%",
            "memory": f"{latest.memory_percent:.1f}%",
            "threads": str(latest.active_threads),
            "open_files": str(latest.open_files)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """성능 개선 권장사항"""
        recommendations = []
        
        if not self.metrics_history:
            return ["모니터링 데이터가 부족합니다"]
        
        # 최근 메트릭 분석
        recent = list(self.metrics_history)[-50:]  # 최근 50개
        avg_cpu = sum(m.cpu_percent for m in recent) / len(recent)
        avg_memory = sum(m.memory_percent for m in recent) / len(recent)
        avg_disk_io = sum(m.disk_io_read_mb + m.disk_io_write_mb for m in recent) / len(recent)
        
        # CPU 권장사항
        if avg_cpu > 70:
            recommendations.append("💾 CPU 집약적 작업을 배치 처리로 분할하세요")
            recommendations.append("🔄 멀티프로세싱 대신 비동기 처리를 고려하세요")
        
        # 메모리 권장사항
        if avg_memory > 80:
            recommendations.append("🧹 메모리 정리: gc.collect() 호출을 증가시키세요")
            recommendations.append("📝 대용량 데이터는 스트리밍 처리를 사용하세요")
            recommendations.append("🗃️ 캐시 크기를 줄이거나 LRU 캐시를 사용하세요")
        
        # 디스크 I/O 권장사항
        if avg_disk_io > 30:
            recommendations.append("💿 파일 읽기/쓰기를 배치로 처리하세요")
            recommendations.append("⚡ SSD 사용을 권장합니다")
            recommendations.append("📦 파일 압축을 고려하세요")
        
        # 모듈별 권장사항
        slow_modules = [
            name for name, stats in self.module_stats.items()
            if stats["avg_time"] > 2.0
        ]
        
        if slow_modules:
            recommendations.append(f"🐌 느린 모듈 최적화 필요: {', '.join(slow_modules)}")
        
        return recommendations if recommendations else ["✅ 시스템이 최적 상태입니다"]
    
    def export_report(self, filepath: str):
        """성능 리포트 파일 저장"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_performance_summary(),
            "raw_metrics": [asdict(m) for m in list(self.metrics_history)[-500:]],  # 최근 500개
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "platform": os.name
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📊 성능 리포트 저장: {filepath}")
    
    def memory_cleanup(self):
        """메모리 정리"""
        before = self._get_memory_usage()
        
        # 가비지 컬렉션 강제 실행
        collected = gc.collect()
        
        after = self._get_memory_usage()
        freed = before - after
        
        self.logger.info(f"🧹 메모리 정리 완료: {freed:.2f}MB 해제, {collected}개 객체 정리")
        
        return {
            "freed_mb": freed,
            "objects_collected": collected,
            "before_mb": before,
            "after_mb": after
        }

# 전역 프로파일러 인스턴스
global_profiler = PerformanceProfiler()

def profile_performance(func_name: str = None):
    """성능 프로파일링 데코레이터 (간편 사용)"""
    return global_profiler.profile_function(func_name)

def get_system_health() -> Dict[str, Any]:
    """시스템 건강 상태 확인"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    health_score = 100
    issues = []
    
    # CPU 체크
    if cpu_percent > 80:
        health_score -= 30
        issues.append("높은 CPU 사용률")
    elif cpu_percent > 60:
        health_score -= 15
        issues.append("보통 CPU 사용률")
    
    # 메모리 체크
    if memory.percent > 85:
        health_score -= 30
        issues.append("높은 메모리 사용률")
    elif memory.percent > 70:
        health_score -= 15
        issues.append("보통 메모리 사용률")
    
    # 디스크 체크
    if disk.percent > 90:
        health_score -= 20
        issues.append("디스크 공간 부족")
    elif disk.percent > 80:
        health_score -= 10
        issues.append("디스크 공간 주의")
    
    status = "우수"
    if health_score < 70:
        status = "위험"
    elif health_score < 85:
        status = "주의"
    
    return {
        "health_score": max(0, health_score),
        "status": status,
        "issues": issues,
        "metrics": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3)
        },
        "recommendations": _get_health_recommendations(health_score, issues)
    }

def _get_health_recommendations(score: int, issues: List[str]) -> List[str]:
    """건강 상태 기반 권장사항"""
    recommendations = []
    
    if "높은 CPU 사용률" in issues:
        recommendations.append("🔄 백그라운드 프로세스를 확인하고 불필요한 작업을 중지하세요")
    
    if "높은 메모리 사용률" in issues:
        recommendations.append("💾 메모리 정리가 필요합니다. 애플리케이션을 재시작해보세요")
    
    if "디스크 공간" in str(issues):
        recommendations.append("🗂️ 임시 파일과 로그 파일을 정리하세요")
    
    if score > 85:
        recommendations.append("✅ 시스템이 최적 상태입니다")
    
    return recommendations

if __name__ == "__main__":
    # 테스트 실행
    print("🚀 솔로몬드 AI 성능 프로파일러 v2.1.2")
    print("=" * 50)
    
    # 시스템 건강 상태 확인
    health = get_system_health()
    print(f"💊 시스템 건강도: {health['health_score']}/100 ({health['status']})")
    print(f"📊 CPU: {health['metrics']['cpu_percent']:.1f}%")
    print(f"💾 메모리: {health['metrics']['memory_percent']:.1f}%")
    print(f"💿 디스크: {health['metrics']['disk_percent']:.1f}%")
    
    if health['issues']:
        print("\n⚠️ 발견된 문제:")
        for issue in health['issues']:
            print(f"  - {issue}")
    
    print("\n💡 권장사항:")
    for rec in health['recommendations']:
        print(f"  {rec}")
    
    # 프로파일러 테스트
    profiler = PerformanceProfiler()
    
    # 테스트 함수
    @profiler.profile_function("test_function")
    def test_heavy_operation():
        """무거운 작업 시뮬레이션"""
        import random
        data = [random.random() for _ in range(100000)]
        return sum(data)
    
    print("\n🔍 성능 프로파일링 테스트...")
    profiler.start_monitoring(interval=0.5)
    
    # 테스트 실행
    for i in range(3):
        result = test_heavy_operation()
        time.sleep(0.5)
    
    time.sleep(2)  # 모니터링 데이터 수집
    profiler.stop_monitoring()
    
    # 결과 출력
    summary = profiler.get_performance_summary()
    print("\n📈 성능 요약:")
    print(f"  평균 CPU: {summary['averages']['cpu_percent']}%")
    print(f"  평균 메모리: {summary['averages']['memory_percent']}%")
    print(f"  현재 상태: {summary['current_status']['status']}")
    
    print("\n🎯 최적화 권장사항:")
    for rec in summary['recommendations']:
        print(f"  {rec}")
    
    # 메모리 정리 테스트
    cleanup_result = profiler.memory_cleanup()
    print(f"\n🧹 메모리 정리: {cleanup_result['freed_mb']:.2f}MB 해제")
    
    print("\n✅ 성능 프로파일러 테스트 완료!")
