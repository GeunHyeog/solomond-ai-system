#!/usr/bin/env python3
"""
🧠 메모리 사용량 모니터링 및 최적화 시스템
실시간 메모리 모니터링, 자동 정리, 최적화 알고리즘 구현
"""

import psutil
import gc
import os
import time
import json
import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import subprocess

@dataclass
class MemorySnapshot:
    """메모리 스냅샷 클래스"""
    timestamp: datetime
    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    memory_percent: float
    swap_total_gb: float
    swap_used_gb: float
    swap_percent: float
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_memory_gb': self.total_memory_gb,
            'available_memory_gb': self.available_memory_gb,
            'used_memory_gb': self.used_memory_gb,
            'memory_percent': self.memory_percent,
            'swap_total_gb': self.swap_total_gb,
            'swap_used_gb': self.swap_used_gb,
            'swap_percent': self.swap_percent
        }

@dataclass
class ProcessMemoryInfo:
    """프로세스 메모리 정보 클래스"""
    pid: int
    name: str
    cmdline: str
    memory_mb: float
    memory_percent: float
    cpu_percent: float
    port: Optional[int] = None
    is_streamlit: bool = False
    
class MemoryOptimizer:
    """메모리 최적화 및 모니터링 클래스"""
    
    def __init__(self, monitoring_interval: int = 30):
        self.monitoring_interval = monitoring_interval
        self.history_file = Path(__file__).parent / "memory_history.json"
        self.config_file = Path(__file__).parent / "memory_config.json"
        self.log_file = Path(__file__).parent / "memory_optimizer.log"
        
        self.memory_history: List[Dict] = []
        self.is_monitoring = False
        self.monitor_thread = None
        
        # 임계값 설정
        self.thresholds = {
            'memory_warning': 80.0,    # 80% 이상시 경고
            'memory_critical': 90.0,   # 90% 이상시 위험
            'process_high_usage': 500.0,  # 프로세스가 500MB 이상 사용시
            'cleanup_trigger': 85.0,   # 85% 이상시 자동 정리
        }
        
        self.load_config()
        self.load_history()
    
    def load_config(self):
        """설정 로드"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.thresholds.update(config.get('thresholds', {}))
            except Exception:
                pass
    
    def save_config(self):
        """설정 저장"""
        try:
            config = {
                'thresholds': self.thresholds,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        except Exception:
            pass
    
    def load_history(self):
        """메모리 히스토리 로드"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.memory_history = json.load(f)
                    # 24시간 이전 데이터 정리
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    self.memory_history = [
                        entry for entry in self.memory_history
                        if datetime.fromisoformat(entry['timestamp']) > cutoff_time
                    ]
            except Exception:
                self.memory_history = []
    
    def save_history(self):
        """메모리 히스토리 저장"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory_history[-1000:], f, indent=2)  # 최근 1000개만 저장
        except Exception:
            pass
    
    def log(self, message: str, level: str = "INFO"):
        """로그 기록"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] [{level}] {message}\n")
        except Exception:
            pass
    
    def get_memory_snapshot(self) -> MemorySnapshot:
        """현재 메모리 상태 스냅샷"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return MemorySnapshot(
            timestamp=datetime.now(),
            total_memory_gb=round(memory.total / 1024**3, 2),
            available_memory_gb=round(memory.available / 1024**3, 2),
            used_memory_gb=round(memory.used / 1024**3, 2),
            memory_percent=round(memory.percent, 1),
            swap_total_gb=round(swap.total / 1024**3, 2),
            swap_used_gb=round(swap.used / 1024**3, 2),
            swap_percent=round(swap.percent, 1)
        )
    
    def get_streamlit_processes(self) -> List[ProcessMemoryInfo]:
        """Streamlit 프로세스 메모리 정보"""
        streamlit_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
            try:
                cmdline = " ".join(proc.info['cmdline']) if proc.info['cmdline'] else ""
                
                if 'streamlit' in cmdline.lower() or 'streamlit' in proc.info['name'].lower():
                    memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                    
                    # 포트 추출 시도
                    port = None
                    try:
                        for conn in proc.connections():
                            if (conn.status == psutil.CONN_LISTEN and 
                                conn.laddr and 
                                8500 <= conn.laddr.port <= 8600):
                                port = conn.laddr.port
                                break
                    except Exception:
                        pass
                    
                    process_info = ProcessMemoryInfo(
                        pid=proc.info['pid'],
                        name=proc.info['name'],
                        cmdline=cmdline,
                        memory_mb=round(memory_mb, 1),
                        memory_percent=round(memory_mb / (psutil.virtual_memory().total / 1024 / 1024) * 100, 2),
                        cpu_percent=proc.info['cpu_percent'] or 0.0,
                        port=port,
                        is_streamlit=True
                    )
                    
                    streamlit_processes.append(process_info)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return sorted(streamlit_processes, key=lambda x: x.memory_mb, reverse=True)
    
    def analyze_memory_usage(self) -> Dict:
        """메모리 사용량 분석"""
        snapshot = self.get_memory_snapshot()
        processes = self.get_streamlit_processes()
        
        total_streamlit_memory = sum(p.memory_mb for p in processes)
        high_usage_processes = [p for p in processes if p.memory_mb > self.thresholds['process_high_usage']]
        
        # 메모리 상태 판정
        status = "정상"
        if snapshot.memory_percent >= self.thresholds['memory_critical']:
            status = "위험"
        elif snapshot.memory_percent >= self.thresholds['memory_warning']:
            status = "경고"
        
        # 최적화 권장사항
        recommendations = []
        
        if snapshot.memory_percent > self.thresholds['memory_warning']:
            recommendations.append("시스템 메모리 사용량이 높습니다")
        
        if high_usage_processes:
            recommendations.append(f"{len(high_usage_processes)}개 프로세스가 과도한 메모리를 사용 중")
        
        if total_streamlit_memory > 1000:  # 1GB 이상
            recommendations.append("Streamlit 앱들의 총 메모리 사용량이 높습니다")
        
        if snapshot.swap_percent > 50:
            recommendations.append("스왑 메모리 사용량이 높습니다")
        
        return {
            'snapshot': snapshot.to_dict(),
            'processes': [asdict(p) for p in processes],
            'total_streamlit_memory_mb': round(total_streamlit_memory, 1),
            'high_usage_processes': len(high_usage_processes),
            'status': status,
            'recommendations': recommendations,
            'analysis_time': datetime.now().isoformat()
        }
    
    def perform_memory_cleanup(self) -> Dict[str, any]:
        """메모리 정리 수행"""
        self.log("메모리 정리 시작")
        cleanup_results = {}
        
        # 1. Python 가비지 컬렉션
        before_gc = len(gc.get_objects())
        collected = gc.collect()
        after_gc = len(gc.get_objects())
        cleanup_results['garbage_collection'] = {
            'objects_before': before_gc,
            'objects_after': after_gc,
            'collected': collected
        }
        
        # 2. 시스템 캐시 정리 시도 (Windows)
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(['sfc', '/scannow'], capture_output=True, timeout=30)
                cleanup_results['system_cache'] = "시도됨"
        except Exception:
            cleanup_results['system_cache'] = "실패"
        
        # 3. 높은 메모리 사용 프로세스 확인
        high_memory_processes = []
        for proc in self.get_streamlit_processes():
            if proc.memory_mb > self.thresholds['process_high_usage']:
                high_memory_processes.append({
                    'pid': proc.pid,
                    'port': proc.port,
                    'memory_mb': proc.memory_mb,
                    'name': proc.name
                })
        
        cleanup_results['high_memory_processes'] = high_memory_processes
        
        # 정리 후 메모리 상태
        after_snapshot = self.get_memory_snapshot()
        cleanup_results['after_cleanup'] = after_snapshot.to_dict()
        
        self.log(f"메모리 정리 완료: {len(high_memory_processes)}개 고사용 프로세스 발견")
        return cleanup_results
    
    def start_monitoring(self):
        """메모리 모니터링 시작"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.log("메모리 모니터링 시작")
    
    def stop_monitoring(self):
        """메모리 모니터링 중지"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.log("메모리 모니터링 중지")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                analysis = self.analyze_memory_usage()
                self.memory_history.append(analysis)
                
                # 자동 정리 트리거
                memory_percent = analysis['snapshot']['memory_percent']
                if memory_percent >= self.thresholds['cleanup_trigger']:
                    self.log(f"메모리 사용량 {memory_percent}% - 자동 정리 실행")
                    self.perform_memory_cleanup()
                
                # 히스토리 저장 (5분마다)
                if len(self.memory_history) % 10 == 0:
                    self.save_history()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.log(f"모니터링 오류: {e}", "ERROR")
                time.sleep(60)  # 오류시 1분 대기
    
    def get_memory_trends(self, hours: int = 4) -> Dict:
        """메모리 사용량 트렌드 분석"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_history = [
            entry for entry in self.memory_history
            if datetime.fromisoformat(entry['analysis_time']) > cutoff_time
        ]
        
        if not recent_history:
            return {'error': '히스토리 데이터 없음'}
        
        memory_values = [entry['snapshot']['memory_percent'] for entry in recent_history]
        streamlit_memory = [entry['total_streamlit_memory_mb'] for entry in recent_history]
        
        return {
            'period_hours': hours,
            'data_points': len(recent_history),
            'memory_percent': {
                'min': min(memory_values),
                'max': max(memory_values),
                'avg': round(sum(memory_values) / len(memory_values), 1),
                'current': memory_values[-1] if memory_values else 0
            },
            'streamlit_memory_mb': {
                'min': min(streamlit_memory),
                'max': max(streamlit_memory),
                'avg': round(sum(streamlit_memory) / len(streamlit_memory), 1),
                'current': streamlit_memory[-1] if streamlit_memory else 0
            },
            'trend_direction': self._calculate_trend(memory_values[-10:]) if len(memory_values) >= 10 else 'stable'
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """트렌드 방향 계산"""
        if len(values) < 2:
            return 'stable'
        
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        diff = second_half - first_half
        if diff > 2:
            return 'increasing'
        elif diff < -2:
            return 'decreasing'
        else:
            return 'stable'
    
    def force_restart_high_memory_process(self, memory_threshold_mb: float = 1000) -> Dict:
        """높은 메모리 사용 프로세스 강제 재시작"""
        results = {}
        high_memory_processes = []
        
        for proc in self.get_streamlit_processes():
            if proc.memory_mb > memory_threshold_mb and proc.port:
                high_memory_processes.append(proc)
        
        if not high_memory_processes:
            return {'message': f'{memory_threshold_mb}MB 이상 사용하는 프로세스 없음'}
        
        from .port_manager import PortManager
        port_manager = PortManager()
        
        for proc in high_memory_processes:
            try:
                # 프로세스 종료
                psutil.Process(proc.pid).terminate()
                time.sleep(2)
                
                # 포트 매니저를 통해 재시작 시도
                module_key = f"port_{proc.port}"
                restarted = port_manager.restart_module_on_port(module_key, proc.port)
                
                results[f"port_{proc.port}"] = {
                    'old_memory_mb': proc.memory_mb,
                    'restarted': restarted,
                    'action': 'terminated_and_restarted' if restarted else 'terminated_only'
                }
                
                self.log(f"높은 메모리 프로세스 재시작: 포트 {proc.port}, {proc.memory_mb}MB")
                
            except Exception as e:
                results[f"port_{proc.port}"] = {
                    'error': str(e),
                    'action': 'failed'
                }
                self.log(f"프로세스 재시작 실패 포트 {proc.port}: {e}", "ERROR")
        
        return results

def main():
    """테스트 실행"""
    optimizer = MemoryOptimizer()
    
    print("Memory Optimizer Test")
    print("=" * 50)
    
    # 현재 메모리 분석
    analysis = optimizer.analyze_memory_usage()
    
    print(f"Memory Status: {analysis['status']}")
    print(f"Memory Usage: {analysis['snapshot']['memory_percent']}%")
    print(f"Total Streamlit Memory: {analysis['total_streamlit_memory_mb']} MB")
    print(f"High Usage Processes: {analysis['high_usage_processes']}")
    
    print(f"\nStreamlit Processes ({len(analysis['processes'])}):")
    for proc in analysis['processes']:
        port_info = f":{proc['port']}" if proc['port'] else ""
        print(f"  PID {proc['pid']}{port_info} - {proc['memory_mb']} MB")
    
    if analysis['recommendations']:
        print(f"\nRecommendations:")
        for rec in analysis['recommendations']:
            print(f"  - {rec}")
    
    # 정리 수행
    print(f"\nPerforming cleanup...")
    cleanup_results = optimizer.perform_memory_cleanup()
    print(f"Cleanup completed")

if __name__ == "__main__":
    main()