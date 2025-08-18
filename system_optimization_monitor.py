#!/usr/bin/env python3
"""
시스템 모니터링 및 최적화 개선 시스템
- 실시간 시스템 성능 모니터링
- 자동 최적화 권장사항 생성
- 리소스 사용량 최적화
- 성능 병목 지점 자동 감지
"""

import os
import sys
import json
import time
import psutil
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import gc
import tracemalloc

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class SystemOptimizationMonitor:
    """시스템 최적화 모니터링 시스템"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self.session_data = {
            'session_id': f"monitor_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'monitoring_interval': monitoring_interval,
            'performance_data': [],
            'optimization_events': [],
            'system_health': {},
            'recommendations': []
        }
        
        # 성능 임계값 설정
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_io_mb_per_sec': 100.0,
            'network_io_mb_per_sec': 50.0,
            'file_handles': 1000,
            'thread_count': 100
        }
        
        # 최적화 규칙
        self.optimization_rules = {
            'memory_cleanup': 'garbage_collection',
            'cpu_throttling': 'process_priority',
            'disk_optimization': 'temp_file_cleanup',
            'network_optimization': 'connection_pooling'
        }
        
        print("시스템 최적화 모니터 초기화 완료")
        self._initialize_monitoring()
    
    def _initialize_monitoring(self):
        """모니터링 시스템 초기화"""
        print("=== 모니터링 시스템 초기화 ===")
        
        # 메모리 추적 시작
        try:
            tracemalloc.start()
            print("[OK] Memory tracing: 활성화")
        except:
            print("[WARNING] Memory tracing: 활성화 실패")
        
        # 시스템 정보 수집
        self.system_info = self._collect_system_info()
        print(f"[OK] System info: CPU {self.system_info['cpu_count']}코어, RAM {self.system_info['total_memory_gb']:.1f}GB")
        
        # 기준선 성능 측정
        baseline_metrics = self._measure_performance()
        self.session_data['baseline_performance'] = baseline_metrics
        print(f"[OK] Baseline: CPU {baseline_metrics['cpu_percent']:.1f}%, Memory {baseline_metrics['memory_percent']:.1f}%")
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """시스템 정보 수집"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
                'total_memory_gb': memory.total / (1024**3),
                'total_disk_gb': disk.total / (1024**3),
                'python_version': sys.version.split()[0],
                'platform': sys.platform,
                'process_id': os.getpid()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _measure_performance(self) -> Dict[str, Any]:
        """현재 성능 지표 측정"""
        try:
            # CPU 및 메모리
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # 디스크 I/O
            disk_io = psutil.disk_io_counters()
            
            # 네트워크 I/O
            net_io = psutil.net_io_counters()
            
            # 프로세스별 정보
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Python 메모리 추적
            python_memory = 0
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                python_memory = current / (1024**2)  # MB
            
            performance_data = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'process_memory_mb': process_memory.rss / (1024**2),
                'python_memory_mb': python_memory,
                'disk_read_mb': disk_io.read_bytes / (1024**2) if disk_io else 0,
                'disk_write_mb': disk_io.write_bytes / (1024**2) if disk_io else 0,
                'network_sent_mb': net_io.bytes_sent / (1024**2) if net_io else 0,
                'network_recv_mb': net_io.bytes_recv / (1024**2) if net_io else 0,
                'thread_count': process.num_threads(),
                'file_handles': process.num_fds() if hasattr(process, 'num_fds') else 0
            }
            
            return performance_data
            
        except Exception as e:
            return {
                'timestamp': time.time(),
                'error': str(e),
                'cpu_percent': 0,
                'memory_percent': 0
            }
    
    def start_monitoring(self):
        """모니터링 시작"""
        if self.monitoring_active:
            print("모니터링이 이미 실행 중입니다.")
            return
        
        print(f"[START] 시스템 모니터링 시작 (간격: {self.monitoring_interval}초)")
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """모니터링 중지"""
        if not self.monitoring_active:
            print("모니터링이 실행 중이 아닙니다.")
            return
        
        print("[STOP] 시스템 모니터링 중지")
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        last_disk_io = None
        last_net_io = None
        last_timestamp = None
        
        while self.monitoring_active:
            try:
                current_metrics = self._measure_performance()
                current_time = current_metrics['timestamp']
                
                # I/O 속도 계산 (델타)
                if last_disk_io and last_timestamp:
                    time_delta = current_time - last_timestamp
                    disk_read_delta = current_metrics['disk_read_mb'] - last_disk_io['read']
                    disk_write_delta = current_metrics['disk_write_mb'] - last_disk_io['write']
                    net_sent_delta = current_metrics['network_sent_mb'] - last_net_io['sent']
                    net_recv_delta = current_metrics['network_recv_mb'] - last_net_io['recv']
                    
                    current_metrics['disk_read_speed'] = disk_read_delta / time_delta
                    current_metrics['disk_write_speed'] = disk_write_delta / time_delta
                    current_metrics['network_send_speed'] = net_sent_delta / time_delta
                    current_metrics['network_recv_speed'] = net_recv_delta / time_delta
                
                # 데이터 저장
                self.session_data['performance_data'].append(current_metrics)
                
                # 임계값 확인 및 최적화 실행
                self._check_thresholds(current_metrics)
                
                # 이전 값 저장
                last_disk_io = {
                    'read': current_metrics['disk_read_mb'],
                    'write': current_metrics['disk_write_mb']
                }
                last_net_io = {
                    'sent': current_metrics['network_sent_mb'],
                    'recv': current_metrics['network_recv_mb']
                }
                last_timestamp = current_time
                
                # 데이터 크기 제한 (최근 100개 항목만 유지)
                if len(self.session_data['performance_data']) > 100:
                    self.session_data['performance_data'] = self.session_data['performance_data'][-100:]
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"[ERROR] 모니터링 오류: {e}")
                time.sleep(self.monitoring_interval)
    
    def _check_thresholds(self, metrics: Dict[str, Any]):
        """임계값 확인 및 자동 최적화"""
        timestamp = metrics.get('timestamp', time.time())
        
        # CPU 사용률 확인
        if metrics.get('cpu_percent', 0) > self.thresholds['cpu_percent']:
            self._trigger_optimization('cpu_high', metrics, timestamp)
        
        # 메모리 사용률 확인
        if metrics.get('memory_percent', 0) > self.thresholds['memory_percent']:
            self._trigger_optimization('memory_high', metrics, timestamp)
        
        # 디스크 I/O 확인
        disk_io_speed = metrics.get('disk_read_speed', 0) + metrics.get('disk_write_speed', 0)
        if disk_io_speed > self.thresholds['disk_io_mb_per_sec']:
            self._trigger_optimization('disk_io_high', metrics, timestamp)
        
        # 네트워크 I/O 확인
        net_io_speed = metrics.get('network_send_speed', 0) + metrics.get('network_recv_speed', 0)
        if net_io_speed > self.thresholds['network_io_mb_per_sec']:
            self._trigger_optimization('network_io_high', metrics, timestamp)
        
        # 프로세스 리소스 확인
        if metrics.get('thread_count', 0) > self.thresholds['thread_count']:
            self._trigger_optimization('thread_count_high', metrics, timestamp)
    
    def _trigger_optimization(self, issue_type: str, metrics: Dict[str, Any], timestamp: float):
        """최적화 트리거"""
        optimization_event = {
            'timestamp': timestamp,
            'issue_type': issue_type,
            'metrics_snapshot': metrics.copy(),
            'optimization_applied': [],
            'performance_impact': {}
        }
        
        print(f"[OPTIMIZE] {issue_type} 감지 - 자동 최적화 실행")
        
        # 최적화 수행
        if issue_type == 'memory_high':
            impact = self._optimize_memory()
            optimization_event['optimization_applied'].append('memory_cleanup')
            optimization_event['performance_impact']['memory'] = impact
        
        elif issue_type == 'cpu_high':
            impact = self._optimize_cpu()
            optimization_event['optimization_applied'].append('cpu_optimization')
            optimization_event['performance_impact']['cpu'] = impact
        
        elif issue_type in ['disk_io_high', 'network_io_high']:
            impact = self._optimize_io()
            optimization_event['optimization_applied'].append('io_optimization')
            optimization_event['performance_impact']['io'] = impact
        
        elif issue_type == 'thread_count_high':
            impact = self._optimize_threads()
            optimization_event['optimization_applied'].append('thread_optimization')
            optimization_event['performance_impact']['threads'] = impact
        
        self.session_data['optimization_events'].append(optimization_event)
    
    def _optimize_memory(self) -> Dict[str, Any]:
        """메모리 최적화"""
        before_memory = psutil.virtual_memory().percent
        before_process = psutil.Process().memory_info().rss / (1024**2)
        
        # 가비지 컬렉션 강제 실행
        gc.collect()
        
        # 메모리 추적 정보 초기화
        if tracemalloc.is_tracing():
            tracemalloc.clear_traces()
        
        time.sleep(0.1)  # 최적화 효과 반영 대기
        
        after_memory = psutil.virtual_memory().percent
        after_process = psutil.Process().memory_info().rss / (1024**2)
        
        impact = {
            'system_memory_reduction': before_memory - after_memory,
            'process_memory_reduction_mb': before_process - after_process,
            'optimization_time': 0.1
        }
        
        print(f"  메모리 최적화: {impact['system_memory_reduction']:.1f}% 시스템, {impact['process_memory_reduction_mb']:.1f}MB 프로세스")
        return impact
    
    def _optimize_cpu(self) -> Dict[str, Any]:
        """CPU 최적화"""
        before_cpu = psutil.cpu_percent(interval=0.1)
        
        # 프로세스 우선순위 조정 (가능한 경우)
        try:
            process = psutil.Process()
            current_priority = process.nice()
            
            # 우선순위를 낮춤 (더 낮은 CPU 사용률)
            if current_priority < 5:
                process.nice(current_priority + 1)
                priority_adjusted = True
            else:
                priority_adjusted = False
        except:
            priority_adjusted = False
        
        time.sleep(0.2)  # 최적화 효과 반영 대기
        after_cpu = psutil.cpu_percent(interval=0.1)
        
        impact = {
            'cpu_reduction': before_cpu - after_cpu,
            'priority_adjusted': priority_adjusted,
            'optimization_time': 0.2
        }
        
        print(f"  CPU 최적화: {impact['cpu_reduction']:.1f}% 감소, 우선순위 조정: {priority_adjusted}")
        return impact
    
    def _optimize_io(self) -> Dict[str, Any]:
        """I/O 최적화"""
        # 임시 파일 정리
        temp_dirs = [
            Path(os.environ.get('TEMP', '/tmp')),
            Path('temp'),
            Path('tmp'),
            project_root / 'temp'
        ]
        
        cleaned_files = 0
        cleaned_size = 0
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                try:
                    for file_path in temp_dir.glob('**/*'):
                        if file_path.is_file():
                            # 1시간 이상 된 임시 파일 삭제
                            if time.time() - file_path.stat().st_mtime > 3600:
                                file_size = file_path.stat().st_size
                                file_path.unlink()
                                cleaned_files += 1
                                cleaned_size += file_size
                except:
                    pass
        
        impact = {
            'cleaned_files': cleaned_files,
            'cleaned_size_mb': cleaned_size / (1024**2),
            'optimization_time': 0.1
        }
        
        print(f"  I/O 최적화: {cleaned_files}개 파일, {impact['cleaned_size_mb']:.1f}MB 정리")
        return impact
    
    def _optimize_threads(self) -> Dict[str, Any]:
        """스레드 최적화"""
        before_threads = psutil.Process().num_threads()
        
        # 스레드 정리는 복잡하므로 시뮬레이션만
        # 실제로는 불필요한 스레드 종료 로직 필요
        
        after_threads = psutil.Process().num_threads()
        
        impact = {
            'thread_reduction': before_threads - after_threads,
            'optimization_time': 0.05
        }
        
        print(f"  스레드 최적화: {impact['thread_reduction']}개 스레드 정리")
        return impact
    
    def generate_health_report(self) -> Dict[str, Any]:
        """시스템 건강 상태 보고서 생성"""
        if not self.session_data['performance_data']:
            return {'error': '성능 데이터가 없습니다.'}
        
        recent_data = self.session_data['performance_data'][-10:]  # 최근 10개 샘플
        
        # 평균 성능 계산
        avg_metrics = {}
        for key in ['cpu_percent', 'memory_percent', 'process_memory_mb', 'thread_count']:
            values = [d.get(key, 0) for d in recent_data if key in d]
            avg_metrics[f'avg_{key}'] = sum(values) / len(values) if values else 0
        
        # 최대값 계산
        max_metrics = {}
        for key in ['cpu_percent', 'memory_percent', 'process_memory_mb']:
            values = [d.get(key, 0) for d in recent_data if key in d]
            max_metrics[f'max_{key}'] = max(values) if values else 0
        
        # 건강 점수 계산 (0-100)
        health_score = 100
        
        # CPU 건강도
        if avg_metrics.get('avg_cpu_percent', 0) > 70:
            health_score -= 20
        elif avg_metrics.get('avg_cpu_percent', 0) > 50:
            health_score -= 10
        
        # 메모리 건강도
        if avg_metrics.get('avg_memory_percent', 0) > 80:
            health_score -= 25
        elif avg_metrics.get('avg_memory_percent', 0) > 60:
            health_score -= 15
        
        # 프로세스 메모리 건강도
        if avg_metrics.get('avg_process_memory_mb', 0) > 500:
            health_score -= 15
        elif avg_metrics.get('avg_process_memory_mb', 0) > 300:
            health_score -= 10
        
        health_report = {
            'overall_health_score': max(0, health_score),
            'health_status': self._get_health_status(health_score),
            'average_metrics': avg_metrics,
            'peak_metrics': max_metrics,
            'optimization_events_count': len(self.session_data['optimization_events']),
            'monitoring_duration_minutes': len(self.session_data['performance_data']) * self.monitoring_interval / 60,
            'recommendations': self._generate_health_recommendations(avg_metrics, max_metrics)
        }
        
        self.session_data['system_health'] = health_report
        return health_report
    
    def _get_health_status(self, score: float) -> str:
        """건강 점수를 상태로 변환"""
        if score >= 90:
            return '최상'
        elif score >= 70:
            return '양호'
        elif score >= 50:
            return '보통'
        elif score >= 30:
            return '주의'
        else:
            return '위험'
    
    def _generate_health_recommendations(self, avg_metrics: Dict[str, Any], 
                                       max_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """건강 상태 기반 권장사항 생성"""
        recommendations = []
        
        # CPU 권장사항
        if avg_metrics.get('avg_cpu_percent', 0) > 60:
            recommendations.append({
                'category': 'CPU',
                'priority': 'HIGH' if avg_metrics.get('avg_cpu_percent', 0) > 80 else 'MEDIUM',
                'issue': f"높은 CPU 사용률 ({avg_metrics.get('avg_cpu_percent', 0):.1f}%)",
                'recommendation': '프로세스 최적화 또는 작업 분산 고려',
                'actions': [
                    '백그라운드 작업 최소화',
                    '멀티프로세싱 구조 검토',
                    '알고리즘 최적화'
                ]
            })
        
        # 메모리 권장사항
        if avg_metrics.get('avg_memory_percent', 0) > 70:
            recommendations.append({
                'category': 'MEMORY',
                'priority': 'HIGH' if avg_metrics.get('avg_memory_percent', 0) > 85 else 'MEDIUM',
                'issue': f"높은 메모리 사용률 ({avg_metrics.get('avg_memory_percent', 0):.1f}%)",
                'recommendation': '메모리 사용량 최적화 필요',
                'actions': [
                    '정기적인 가비지 컬렉션',
                    '메모리 누수 점검',
                    '데이터 구조 최적화'
                ]
            })
        
        # 프로세스 메모리 권장사항
        if avg_metrics.get('avg_process_memory_mb', 0) > 400:
            recommendations.append({
                'category': 'PROCESS_MEMORY',
                'priority': 'MEDIUM',
                'issue': f"높은 프로세스 메모리 사용량 ({avg_metrics.get('avg_process_memory_mb', 0):.1f}MB)",
                'recommendation': '프로세스별 메모리 관리 강화',
                'actions': [
                    '임시 데이터 정리',
                    '캐시 크기 조정',
                    '메모리 사용 패턴 분석'
                ]
            })
        
        return recommendations
    
    def save_monitoring_results(self):
        """모니터링 결과 저장"""
        # 최종 건강 보고서 생성
        health_report = self.generate_health_report()
        
        # 요약 통계 생성
        summary = self._generate_monitoring_summary()
        self.session_data['summary'] = summary
        
        report_path = project_root / f"system_monitoring_report_{self.session_data['session_id']}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.session_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVE] 시스템 모니터링 보고서 저장: {report_path}")
        return report_path
    
    def _generate_monitoring_summary(self) -> Dict[str, Any]:
        """모니터링 요약 생성"""
        if not self.session_data['performance_data']:
            return {}
        
        data = self.session_data['performance_data']
        
        # 기본 통계
        cpu_values = [d.get('cpu_percent', 0) for d in data]
        memory_values = [d.get('memory_percent', 0) for d in data]
        process_memory_values = [d.get('process_memory_mb', 0) for d in data]
        
        summary = {
            'monitoring_duration_seconds': len(data) * self.monitoring_interval,
            'samples_collected': len(data),
            'optimization_events': len(self.session_data['optimization_events']),
            'performance_stats': {
                'cpu': {
                    'average': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                    'peak': max(cpu_values) if cpu_values else 0,
                    'minimum': min(cpu_values) if cpu_values else 0
                },
                'memory': {
                    'average': sum(memory_values) / len(memory_values) if memory_values else 0,
                    'peak': max(memory_values) if memory_values else 0,
                    'minimum': min(memory_values) if memory_values else 0
                },
                'process_memory': {
                    'average': sum(process_memory_values) / len(process_memory_values) if process_memory_values else 0,
                    'peak': max(process_memory_values) if process_memory_values else 0,
                    'minimum': min(process_memory_values) if process_memory_values else 0
                }
            }
        }
        
        return summary

def main():
    """메인 실행 함수"""
    print("시스템 최적화 모니터 데모 시작")
    print("=" * 50)
    
    # 모니터 생성 (1초 간격)
    monitor = SystemOptimizationMonitor(monitoring_interval=1.0)
    
    # 모니터링 시작
    monitor.start_monitoring()
    
    try:
        # 10초간 모니터링 실행
        print(f"10초간 시스템 모니터링 실행...")
        
        for i in range(10):
            time.sleep(1)
            print(f"  모니터링 중... {i+1}/10초")
            
            # 중간에 인위적 부하 생성 (테스트용)
            if i == 5:
                print("  [TEST] 인위적 부하 생성...")
                # CPU 부하
                for _ in range(100000):
                    sum(range(100))
                
                # 메모리 부하
                test_data = [i for i in range(100000)]
                del test_data
        
        print("모니터링 완료!")
        
    finally:
        # 모니터링 중지
        monitor.stop_monitoring()
    
    # 건강 보고서 생성
    health_report = monitor.generate_health_report()
    
    print(f"\n{'='*50}")
    print("시스템 건강 상태 보고서")
    print(f"{'='*50}")
    print(f"전체 건강 점수: {health_report.get('overall_health_score', 0):.1f}/100")
    print(f"건강 상태: {health_report.get('health_status', 'Unknown')}")
    print(f"최적화 이벤트: {health_report.get('optimization_events_count', 0)}회")
    print(f"모니터링 시간: {health_report.get('monitoring_duration_minutes', 0):.1f}분")
    
    # 권장사항 출력
    recommendations = health_report.get('recommendations', [])
    if recommendations:
        print(f"\n권장사항: {len(recommendations)}개")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. [{rec['category']}] {rec['issue']} ({rec['priority']})")
    
    # 결과 저장
    report_path = monitor.save_monitoring_results()
    print(f"\n상세 보고서: {report_path}")
    
    return monitor.session_data

if __name__ == "__main__":
    main()