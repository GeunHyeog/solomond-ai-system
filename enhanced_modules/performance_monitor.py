#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[PERFORMANCE] 성능 모니터링 시스템
Real-time Performance Monitoring & Optimization System

핵심 기능:
1. 실시간 시스템 성능 모니터링
2. 메모리/CPU/GPU 사용량 추적
3. 처리 속도 벤치마킹
4. 자동 성능 최적화 제안
5. 성능 히스토리 관리
"""

import os
import sys
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import gc
import tracemalloc
from collections import defaultdict, deque

# 로깅 설정
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """시스템 메트릭 데이터"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    gpu_available: bool
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None

@dataclass
class ProcessingMetrics:
    """처리 성능 메트릭"""
    operation_name: str
    start_time: float
    end_time: float
    processing_time: float
    memory_used_mb: float
    cpu_peak_percent: float
    success: bool
    file_size_mb: Optional[float] = None
    throughput_mbps: Optional[float] = None

class PerformanceMonitor:
    """실시간 성능 모니터링 시스템"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.system_history = deque(maxlen=history_size)
        self.processing_history = deque(maxlen=history_size)
        self.operation_stats = defaultdict(list)
        
        # 모니터링 설정
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 5.0  # 5초마다 시스템 메트릭 수집
        
        # 메모리 추적
        self.memory_tracker = None
        
        # GPU 감지
        self.gpu_available = self._detect_gpu()
        
        logger.info("[PERFORMANCE] 성능 모니터링 시스템 초기화 완료")
    
    def _detect_gpu(self) -> bool:
        """GPU 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                return result.returncode == 0
            except:
                return False
    
    def _get_gpu_info(self) -> Tuple[Optional[float], Optional[float]]:
        """GPU 메모리 사용량 반환 (used_mb, total_mb)"""
        if not self.gpu_available:
            return None, None
        
        try:
            import torch
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1024**2
                total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                return used, total
        except:
            pass
        
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                used, total = lines[0].split(', ')
                return float(used), float(total)
        except:
            pass
        
        return None, None
    
    def start_monitoring(self):
        """백그라운드 시스템 모니터링 시작"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("[PERFORMANCE] 백그라운드 모니터링 시작")
    
    def stop_monitoring(self):
        """백그라운드 시스템 모니터링 중지"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        logger.info("[PERFORMANCE] 백그라운드 모니터링 중지")
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                self.system_history.append(metrics)
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.warning(f"모니터링 오류: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """현재 시스템 메트릭 수집"""
        # CPU 및 메모리 정보
        cpu_percent = psutil.cpu_percent(interval=1.0)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU 정보
        gpu_used, gpu_total = self._get_gpu_info()
        
        return SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_mb=memory.available / 1024**2,
            disk_usage_percent=disk.percent,
            gpu_available=self.gpu_available,
            gpu_memory_used_mb=gpu_used,
            gpu_memory_total_mb=gpu_total
        )
    
    def start_operation_tracking(self, operation_name: str, file_size_mb: Optional[float] = None):
        """작업 성능 추적 시작"""
        # 메모리 추적 시작
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        tracking_data = {
            'operation_name': operation_name,
            'start_time': time.time(),
            'start_memory': self._get_current_memory(),
            'start_cpu': psutil.cpu_percent(),
            'file_size_mb': file_size_mb,
            'peak_cpu': psutil.cpu_percent()
        }
        
        return tracking_data
    
    def end_operation_tracking(self, tracking_data: dict, success: bool = True) -> ProcessingMetrics:
        """작업 성능 추적 종료 및 메트릭 저장"""
        end_time = time.time()
        processing_time = end_time - tracking_data['start_time']
        
        # 메모리 사용량 계산
        current_memory = self._get_current_memory()
        memory_used = max(0, current_memory - tracking_data['start_memory'])
        
        # 처리량 계산
        throughput = None
        if tracking_data.get('file_size_mb') and processing_time > 0:
            throughput = tracking_data['file_size_mb'] / processing_time
        
        metrics = ProcessingMetrics(
            operation_name=tracking_data['operation_name'],
            start_time=tracking_data['start_time'],
            end_time=end_time,
            processing_time=processing_time,
            memory_used_mb=memory_used,
            cpu_peak_percent=tracking_data.get('peak_cpu', 0),
            success=success,
            file_size_mb=tracking_data.get('file_size_mb'),
            throughput_mbps=throughput
        )
        
        # 히스토리에 저장
        self.processing_history.append(metrics)
        self.operation_stats[tracking_data['operation_name']].append(metrics)
        
        return metrics
    
    def _get_current_memory(self) -> float:
        """현재 메모리 사용량 반환 (MB)"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024**2
        except:
            return 0.0
    
    def get_operation_statistics(self, operation_name: str) -> Dict[str, Any]:
        """특정 작업의 통계 정보 반환"""
        if operation_name not in self.operation_stats:
            return {}
        
        metrics_list = self.operation_stats[operation_name]
        if not metrics_list:
            return {}
        
        processing_times = [m.processing_time for m in metrics_list]
        memory_usage = [m.memory_used_mb for m in metrics_list]
        success_rate = sum(1 for m in metrics_list if m.success) / len(metrics_list)
        
        throughputs = [m.throughput_mbps for m in metrics_list if m.throughput_mbps is not None]
        
        stats = {
            'operation_name': operation_name,
            'total_runs': len(metrics_list),
            'success_rate': success_rate,
            'processing_time': {
                'min': min(processing_times),
                'max': max(processing_times),
                'avg': sum(processing_times) / len(processing_times),
                'recent_avg': sum(processing_times[-10:]) / min(10, len(processing_times))
            },
            'memory_usage_mb': {
                'min': min(memory_usage),
                'max': max(memory_usage),
                'avg': sum(memory_usage) / len(memory_usage)
            }
        }
        
        if throughputs:
            stats['throughput_mbps'] = {
                'min': min(throughputs),
                'max': max(throughputs),
                'avg': sum(throughputs) / len(throughputs)
            }
        
        return stats
    
    def get_system_status(self) -> Dict[str, Any]:
        """현재 시스템 상태 반환"""
        current_metrics = self._collect_system_metrics()
        
        status = {
            'current_metrics': asdict(current_metrics),
            'monitoring_active': self.monitoring_active,
            'history_size': len(self.system_history),
            'tracked_operations': list(self.operation_stats.keys()),
            'total_processing_records': len(self.processing_history)
        }
        
        # 최근 시스템 트렌드 (지난 1시간)
        if self.system_history:
            recent_metrics = [m for m in self.system_history 
                            if datetime.fromisoformat(m.timestamp) > datetime.now() - timedelta(hours=1)]
            
            if recent_metrics:
                status['recent_trends'] = {
                    'avg_cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                    'avg_memory_percent': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
                    'samples_count': len(recent_metrics)
                }
        
        return status
    
    def get_optimization_recommendations(self) -> List[str]:
        """성능 최적화 제안 반환"""
        recommendations = []
        
        # 현재 시스템 상태 확인
        current = self._collect_system_metrics()
        
        # CPU 사용률 확인
        if current.cpu_percent > 80:
            recommendations.append("CPU 사용률이 높습니다. 병렬 처리 수준을 줄이거나 백그라운드 프로세스를 확인하세요.")
        
        # 메모리 사용률 확인
        if current.memory_percent > 85:
            recommendations.append("메모리 사용률이 높습니다. 가비지 컬렉션을 수행하거나 배치 크기를 줄이세요.")
        
        # GPU 메모리 확인
        if (current.gpu_available and current.gpu_memory_used_mb and current.gpu_memory_total_mb 
            and (current.gpu_memory_used_mb / current.gpu_memory_total_mb) > 0.9):
            recommendations.append("GPU 메모리 사용률이 높습니다. 모델 크기를 줄이거나 배치 처리를 조정하세요.")
        
        # 작업별 성능 분석
        for operation_name, metrics_list in self.operation_stats.items():
            if len(metrics_list) >= 5:  # 충분한 샘플이 있는 경우
                recent_times = [m.processing_time for m in metrics_list[-5:]]
                overall_avg = sum(m.processing_time for m in metrics_list) / len(metrics_list)
                recent_avg = sum(recent_times) / len(recent_times)
                
                if recent_avg > overall_avg * 1.5:
                    recommendations.append(f"{operation_name} 작업의 최근 성능이 저하되었습니다. 시스템 리소스나 입력 데이터를 확인하세요.")
        
        if not recommendations:
            recommendations.append("시스템 성능이 양호합니다.")
        
        return recommendations
    
    def cleanup_old_data(self, days_to_keep: int = 7):
        """오래된 성능 데이터 정리"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # 시스템 히스토리 정리
        self.system_history = deque([
            m for m in self.system_history 
            if datetime.fromisoformat(m.timestamp) > cutoff_date
        ], maxlen=self.history_size)
        
        # 처리 히스토리 정리
        self.processing_history = deque([
            m for m in self.processing_history 
            if datetime.fromtimestamp(m.start_time) > cutoff_date
        ], maxlen=self.history_size)
        
        # 작업별 통계 정리
        for operation_name in list(self.operation_stats.keys()):
            self.operation_stats[operation_name] = [
                m for m in self.operation_stats[operation_name]
                if datetime.fromtimestamp(m.start_time) > cutoff_date
            ]
            
            # 빈 리스트 제거
            if not self.operation_stats[operation_name]:
                del self.operation_stats[operation_name]
        
        logger.info(f"[PERFORMANCE] {days_to_keep}일 이상 된 데이터 정리 완료")
    
    def save_performance_report(self, output_path: str):
        """성능 보고서 파일 저장"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'system_status': self.get_system_status(),
            'optimization_recommendations': self.get_optimization_recommendations(),
            'operation_statistics': {
                name: self.get_operation_statistics(name) 
                for name in self.operation_stats.keys()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[PERFORMANCE] 성능 보고서 저장: {output_path}")

# 전역 인스턴스
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """성능 모니터 인스턴스 반환 (싱글톤)"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

# 컨텍스트 매니저
class OperationTracker:
    """작업 성능 추적 컨텍스트 매니저"""
    
    def __init__(self, operation_name: str, file_size_mb: Optional[float] = None):
        self.operation_name = operation_name
        self.file_size_mb = file_size_mb
        self.monitor = get_performance_monitor()
        self.tracking_data = None
    
    def __enter__(self):
        self.tracking_data = self.monitor.start_operation_tracking(
            self.operation_name, self.file_size_mb
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        metrics = self.monitor.end_operation_tracking(self.tracking_data, success)
        return False  # 예외를 다시 발생시킴

if __name__ == "__main__":
    # 성능 모니터링 시스템 테스트
    monitor = get_performance_monitor()
    
    print("성능 모니터링 시스템 테스트 시작...")
    
    # 백그라운드 모니터링 시작
    monitor.start_monitoring()
    
    # 테스트 작업 수행
    with OperationTracker("test_operation", file_size_mb=1.5) as tracker:
        time.sleep(2)  # 2초 작업 시뮬레이션
        print("테스트 작업 완료")
    
    # 상태 확인
    status = monitor.get_system_status()
    print(f"현재 시스템 상태: CPU {status['current_metrics']['cpu_percent']:.1f}%, 메모리 {status['current_metrics']['memory_percent']:.1f}%")
    
    # 최적화 제안 확인
    recommendations = monitor.get_optimization_recommendations()
    print("최적화 제안:")
    for rec in recommendations:
        print(f"  - {rec}")
    
    # 모니터링 중지
    monitor.stop_monitoring()
    print("성능 모니터링 테스트 완료")