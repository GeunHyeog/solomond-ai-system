"""
성능 모니터링
시스템 리소스 및 처리 성능 추적
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.performance_data = []
        self.start_time = None
        
    def start_monitoring(self):
        """모니터링 시작"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.start_time = time.time()
        self.performance_data = []
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logging.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        logging.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                data_point = self._collect_performance_data()
                self.performance_data.append(data_point)
                
                # 데이터 포인트 수 제한 (메모리 관리)
                if len(self.performance_data) > 1000:
                    self.performance_data = self.performance_data[-500:]
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logging.error(f"Performance monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_performance_data(self) -> Dict[str, Any]:
        """성능 데이터 수집"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time if self.start_time else 0
        
        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        
        # 디스크 I/O (가능한 경우)
        try:
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024)
            disk_write_mb = disk_io.write_bytes / (1024 * 1024)
        except:
            disk_read_mb = 0
            disk_write_mb = 0
        
        # 네트워크 I/O (가능한 경우)
        try:
            net_io = psutil.net_io_counters()
            net_sent_mb = net_io.bytes_sent / (1024 * 1024)
            net_recv_mb = net_io.bytes_recv / (1024 * 1024)
        except:
            net_sent_mb = 0
            net_recv_mb = 0
        
        return {
            'timestamp': current_time,
            'elapsed_time': elapsed_time,
            'cpu_percent': cpu_percent,
            'memory_used_mb': memory.used / (1024 * 1024),
            'memory_available_mb': memory.available / (1024 * 1024),
            'memory_percent': memory.percent,
            'disk_read_mb': disk_read_mb,
            'disk_write_mb': disk_write_mb,
            'network_sent_mb': net_sent_mb,
            'network_recv_mb': net_recv_mb
        }
    
    def get_current_stats(self) -> Dict[str, Any]:
        """현재 성능 통계"""
        if not self.performance_data:
            return {}
        
        latest = self.performance_data[-1]
        
        return {
            'cpu_percent': latest['cpu_percent'],
            'memory_used_gb': round(latest['memory_used_mb'] / 1024, 2),
            'memory_percent': latest['memory_percent'],
            'elapsed_time': round(latest['elapsed_time'], 1),
            'monitoring_duration': len(self.performance_data) * self.monitoring_interval
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 통계"""
        if not self.performance_data:
            return {}
        
        # CPU 통계
        cpu_values = [d['cpu_percent'] for d in self.performance_data]
        cpu_avg = sum(cpu_values) / len(cpu_values)
        cpu_max = max(cpu_values)
        
        # 메모리 통계
        memory_values = [d['memory_percent'] for d in self.performance_data]
        memory_avg = sum(memory_values) / len(memory_values)
        memory_max = max(memory_values)
        
        # 최신 값들
        latest = self.performance_data[-1]
        
        return {
            'monitoring_duration': len(self.performance_data) * self.monitoring_interval,
            'data_points': len(self.performance_data),
            'cpu_stats': {
                'average': round(cpu_avg, 1),
                'maximum': round(cpu_max, 1),
                'current': round(latest['cpu_percent'], 1)
            },
            'memory_stats': {
                'average_percent': round(memory_avg, 1),
                'maximum_percent': round(memory_max, 1),
                'current_percent': round(latest['memory_percent'], 1),
                'current_used_gb': round(latest['memory_used_mb'] / 1024, 2)
            },
            'disk_io': {
                'total_read_mb': round(latest['disk_read_mb'], 1),
                'total_write_mb': round(latest['disk_write_mb'], 1)
            },
            'network_io': {
                'total_sent_mb': round(latest['network_sent_mb'], 1),
                'total_received_mb': round(latest['network_recv_mb'], 1)
            }
        }
    
    def get_performance_alerts(self) -> List[str]:
        """성능 경고 메시지"""
        if not self.performance_data:
            return []
        
        alerts = []
        latest = self.performance_data[-1]
        
        # CPU 경고
        if latest['cpu_percent'] > 80:
            alerts.append(f"⚠️ High CPU usage: {latest['cpu_percent']:.1f}%")
        
        # 메모리 경고
        if latest['memory_percent'] > 85:
            alerts.append(f"⚠️ High memory usage: {latest['memory_percent']:.1f}%")
        
        # 지속적인 높은 사용률 확인
        if len(self.performance_data) >= 10:
            recent_cpu = [d['cpu_percent'] for d in self.performance_data[-10:]]
            if all(cpu > 70 for cpu in recent_cpu):
                alerts.append("⚠️ Sustained high CPU usage detected")
            
            recent_memory = [d['memory_percent'] for d in self.performance_data[-10:]]
            if all(mem > 75 for mem in recent_memory):
                alerts.append("⚠️ Sustained high memory usage detected")
        
        return alerts
    
    def export_performance_data(self, filename: str = None) -> str:
        """성능 데이터 내보내기"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_data_{timestamp}.json"
        
        try:
            import json
            
            export_data = {
                'monitoring_info': {
                    'start_time': self.start_time,
                    'monitoring_interval': self.monitoring_interval,
                    'data_points': len(self.performance_data)
                },
                'summary': self.get_performance_summary(),
                'data': self.performance_data
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            logging.info(f"Performance data exported to: {filename}")
            return filename
            
        except Exception as e:
            logging.error(f"Failed to export performance data: {e}")
            return ""
    
    def reset_monitoring(self):
        """모니터링 데이터 리셋"""
        self.performance_data = []
        self.start_time = time.time()
        logging.info("Performance monitoring data reset")
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.stop_monitoring()