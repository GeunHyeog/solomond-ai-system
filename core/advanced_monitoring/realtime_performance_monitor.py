#!/usr/bin/env python3
"""
실시간 성능 모니터링 시스템 v2.5
2025 최신 AI 시스템 최적화 전략 반영
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np

@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_io_read: float
    disk_io_write: float
    network_sent: float
    network_recv: float
    ollama_models_active: int
    processing_queue_size: int
    response_time_ms: float
    error_count: int
    user_sessions: int

@dataclass
class SystemAlert:
    """시스템 알림 데이터 클래스"""
    level: str  # 'info', 'warning', 'critical'
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: float
    suggested_action: str

class RealtimePerformanceMonitor:
    """실시간 성능 모니터링 시스템"""
    
    def __init__(self, 
                 alert_callback: Optional[Callable] = None,
                 metrics_retention_hours: int = 24):
        self.alert_callback = alert_callback
        self.metrics_retention_hours = metrics_retention_hours
        
        # 메트릭 저장소 (시간 기반 순환 큐)
        self.metrics_history = deque(maxlen=metrics_retention_hours * 3600)  # 1초당 1개 메트릭
        self.alerts_history = deque(maxlen=1000)
        
        # 성능 임계값 설정
        self.thresholds = {
            'cpu_usage': {'warning': 70.0, 'critical': 85.0},
            'memory_usage': {'warning': 75.0, 'critical': 90.0},
            'response_time_ms': {'warning': 2000.0, 'critical': 5000.0},
            'error_count': {'warning': 5, 'critical': 10},
            'disk_io_read': {'warning': 100.0, 'critical': 200.0},  # MB/s
            'disk_io_write': {'warning': 100.0, 'critical': 200.0},
        }
        
        # 모니터링 상태
        self.is_monitoring = False
        self.monitoring_thread = None
        self.ollama_stats = {'active_models': 0, 'total_requests': 0}
        self.processing_stats = {'queue_size': 0, 'completed_tasks': 0}
        self.user_stats = {'active_sessions': 0, 'total_requests': 0}
        
        # 성능 최적화를 위한 캐시
        self._last_system_stats = None
        self._stats_cache_time = 0
        self._cache_duration = 1.0  # 1초 캐시
        
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.RealtimePerformanceMonitor')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """모니터링 시작"""
        if self.is_monitoring:
            self.logger.warning("⚠️ 모니터링이 이미 실행 중입니다")
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("🚀 실시간 성능 모니터링 시작")
    
    def stop_monitoring(self) -> None:
        """모니터링 중지"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("⏹️ 실시간 성능 모니터링 중지")
    
    def _monitoring_loop(self, interval: float) -> None:
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 알림 검사
                alerts = self._check_alerts(metrics)
                for alert in alerts:
                    self.alerts_history.append(alert)
                    if self.alert_callback:
                        self.alert_callback(alert)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"❌ 모니터링 루프 오류: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """시스템 메트릭 수집"""
        current_time = time.time()
        
        # 캐시된 시스템 통계 사용 (성능 최적화)
        if (current_time - self._stats_cache_time) > self._cache_duration:
            self._last_system_stats = self._get_system_stats()
            self._stats_cache_time = current_time
        
        stats = self._last_system_stats
        
        return PerformanceMetrics(
            timestamp=current_time,
            cpu_usage=stats['cpu_percent'],
            memory_usage=stats['memory_percent'],
            memory_available=stats['memory_available'] / (1024**3),  # GB
            disk_io_read=stats['disk_read_mb'],
            disk_io_write=stats['disk_write_mb'],
            network_sent=stats['network_sent_mb'],
            network_recv=stats['network_recv_mb'],
            ollama_models_active=self.ollama_stats['active_models'],
            processing_queue_size=self.processing_stats['queue_size'],
            response_time_ms=self._calculate_avg_response_time(),
            error_count=self._get_recent_error_count(),
            user_sessions=self.user_stats['active_sessions']
        )
    
    def _get_system_stats(self) -> Dict[str, float]:
        """시스템 통계 수집"""
        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # 메모리 사용률
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available = memory.available
        
        # 디스크 I/O (MB/s)
        disk_io = psutil.disk_io_counters()
        disk_read_mb = getattr(disk_io, 'read_bytes', 0) / (1024**2)
        disk_write_mb = getattr(disk_io, 'write_bytes', 0) / (1024**2)
        
        # 네트워크 I/O (MB/s)
        network_io = psutil.net_io_counters()
        network_sent_mb = getattr(network_io, 'bytes_sent', 0) / (1024**2)
        network_recv_mb = getattr(network_io, 'bytes_recv', 0) / (1024**2)
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_available': memory_available,
            'disk_read_mb': disk_read_mb,
            'disk_write_mb': disk_write_mb,
            'network_sent_mb': network_sent_mb,
            'network_recv_mb': network_recv_mb,
        }
    
    def _calculate_avg_response_time(self) -> float:
        """평균 응답 시간 계산"""
        # 최근 10개 메트릭의 평균 계산
        if len(self.metrics_history) < 2:
            return 0.0
        
        recent_metrics = list(self.metrics_history)[-10:]
        response_times = [m.response_time_ms for m in recent_metrics if m.response_time_ms > 0]
        
        return np.mean(response_times) if response_times else 0.0
    
    def _get_recent_error_count(self) -> int:
        """최근 에러 카운트"""
        # 최근 1분간의 에러 카운트
        cutoff_time = time.time() - 60
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        return sum(m.error_count for m in recent_metrics)
    
    def _check_alerts(self, metrics: PerformanceMetrics) -> List[SystemAlert]:
        """알림 검사"""
        alerts = []
        
        # CPU 사용률 검사
        alerts.extend(self._check_threshold_alerts(
            'cpu_usage', metrics.cpu_usage, '%',
            "CPU 사용률이 높습니다",
            "불필요한 프로세스를 종료하거나 시스템을 재시작하세요"
        ))
        
        # 메모리 사용률 검사
        alerts.extend(self._check_threshold_alerts(
            'memory_usage', metrics.memory_usage, '%',
            "메모리 사용률이 높습니다",
            "메모리를 많이 사용하는 프로세스를 확인하세요"
        ))
        
        # 응답 시간 검사
        alerts.extend(self._check_threshold_alerts(
            'response_time_ms', metrics.response_time_ms, 'ms',
            "응답 시간이 느립니다",
            "시스템 부하를 확인하고 최적화를 고려하세요"
        ))
        
        # 디스크 I/O 검사
        alerts.extend(self._check_threshold_alerts(
            'disk_io_read', metrics.disk_io_read, 'MB/s',
            "디스크 읽기 부하가 높습니다",
            "디스크 사용량을 확인하세요"
        ))
        
        return alerts
    
    def _check_threshold_alerts(self, 
                              metric_name: str, 
                              value: float, 
                              unit: str,
                              message: str,
                              action: str) -> List[SystemAlert]:
        """임계값 기반 알림 검사"""
        alerts = []
        thresholds = self.thresholds.get(metric_name, {})
        
        if value >= thresholds.get('critical', float('inf')):
            alerts.append(SystemAlert(
                level='critical',
                message=f"🚨 CRITICAL: {message}",
                metric=metric_name,
                value=value,
                threshold=thresholds['critical'],
                timestamp=time.time(),
                suggested_action=action
            ))
        elif value >= thresholds.get('warning', float('inf')):
            alerts.append(SystemAlert(
                level='warning',
                message=f"⚠️ WARNING: {message}",
                metric=metric_name,
                value=value,
                threshold=thresholds['warning'],
                timestamp=time.time(),
                suggested_action=action
            ))
        
        return alerts
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """현재 메트릭 반환"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, hours: int = 1) -> List[PerformanceMetrics]:
        """메트릭 히스토리 반환"""
        cutoff_time = time.time() - (hours * 3600)
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보 반환"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.get_metrics_history(1)  # 최근 1시간
        
        if not recent_metrics:
            return {"status": "insufficient_data"}
        
        # 평균 계산
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        avg_response_time = np.mean([m.response_time_ms for m in recent_metrics])
        total_errors = sum([m.error_count for m in recent_metrics])
        
        # 최대값
        max_cpu = max([m.cpu_usage for m in recent_metrics])
        max_memory = max([m.memory_usage for m in recent_metrics])
        max_response_time = max([m.response_time_ms for m in recent_metrics])
        
        # 현재 활성 알림
        active_alerts = [a for a in self.alerts_history 
                        if time.time() - a.timestamp < 300]  # 5분 이내
        
        return {
            "status": "active",
            "monitoring_duration_hours": len(recent_metrics) / 3600,
            "averages": {
                "cpu_usage": round(avg_cpu, 2),
                "memory_usage": round(avg_memory, 2),
                "response_time_ms": round(avg_response_time, 2)
            },
            "peaks": {
                "cpu_usage": round(max_cpu, 2),
                "memory_usage": round(max_memory, 2),
                "response_time_ms": round(max_response_time, 2)
            },
            "total_errors": total_errors,
            "active_alerts": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts if a.level == 'critical']),
            "ollama_models_active": self.ollama_stats['active_models'],
            "user_sessions": self.user_stats['active_sessions']
        }
    
    def update_ollama_stats(self, active_models: int, total_requests: int = None) -> None:
        """Ollama 통계 업데이트"""
        self.ollama_stats['active_models'] = active_models
        if total_requests is not None:
            self.ollama_stats['total_requests'] = total_requests
    
    def update_processing_stats(self, queue_size: int, completed_tasks: int = None) -> None:
        """처리 통계 업데이트"""
        self.processing_stats['queue_size'] = queue_size
        if completed_tasks is not None:
            self.processing_stats['completed_tasks'] = completed_tasks
    
    def update_user_stats(self, active_sessions: int, total_requests: int = None) -> None:
        """사용자 통계 업데이트"""
        self.user_stats['active_sessions'] = active_sessions
        if total_requests is not None:
            self.user_stats['total_requests'] = total_requests
    
    def export_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """메트릭 데이터 내보내기"""
        metrics = self.get_metrics_history(hours)
        alerts = [a for a in self.alerts_history 
                 if time.time() - a.timestamp < (hours * 3600)]
        
        return {
            "export_timestamp": datetime.now().isoformat(),
            "duration_hours": hours,
            "metrics_count": len(metrics),
            "alerts_count": len(alerts),
            "metrics": [asdict(m) for m in metrics],
            "alerts": [asdict(a) for a in alerts],
            "summary": self.get_performance_summary()
        }

# 전역 모니터 인스턴스
_global_monitor = None

def get_global_monitor() -> RealtimePerformanceMonitor:
    """전역 모니터 인스턴스 반환"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = RealtimePerformanceMonitor()
        _global_monitor.start_monitoring()
    return _global_monitor

def setup_monitoring_with_alerts(alert_callback: Callable[[SystemAlert], None]) -> RealtimePerformanceMonitor:
    """알림 콜백과 함께 모니터링 설정"""
    monitor = RealtimePerformanceMonitor(alert_callback=alert_callback)
    monitor.start_monitoring()
    return monitor

# 사용 예시
if __name__ == "__main__":
    def alert_handler(alert: SystemAlert):
        print(f"🚨 ALERT: {alert.message} (값: {alert.value}, 임계값: {alert.threshold})")
        print(f"   권장 조치: {alert.suggested_action}")
    
    # 모니터링 시작
    monitor = setup_monitoring_with_alerts(alert_handler)
    
    try:
        print("🚀 성능 모니터링 시작됨. Ctrl+C로 종료...")
        while True:
            time.sleep(10)
            summary = monitor.get_performance_summary()
            print(f"📊 CPU: {summary.get('averages', {}).get('cpu_usage', 0):.1f}% | "
                  f"메모리: {summary.get('averages', {}).get('memory_usage', 0):.1f}% | "
                  f"알림: {summary.get('active_alerts', 0)}개")
    
    except KeyboardInterrupt:
        print("\n⏹️ 모니터링 중지 중...")
        monitor.stop_monitoring()
        print("✅ 모니터링 완료")