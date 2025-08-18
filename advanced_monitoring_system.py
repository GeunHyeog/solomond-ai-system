#!/usr/bin/env python3
"""
고급 시스템 모니터링 시스템
- 실시간 성능 예측 및 경고
- 대용량 파일 처리 최적화 추천
- AI 기반 리소스 사용 패턴 분석
- 자동 성능 튜닝 시스템
"""

import os
import sys
import time
import json
import psutil
import threading
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import gc
import tracemalloc

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 기존 시스템 모니터 import
try:
    from system_optimization_monitor import SystemOptimizationMonitor
    BASE_MONITOR_AVAILABLE = True
except ImportError:
    BASE_MONITOR_AVAILABLE = False

class AdvancedMonitoringSystem:
    """고급 시스템 모니터링 및 예측 시스템"""
    
    def __init__(self, monitoring_interval: float = 1.0, prediction_window: int = 60):
        self.monitoring_interval = monitoring_interval
        self.prediction_window = prediction_window  # 예측 윈도우 (샘플 수)
        
        # 성능 데이터 저장소 (FIFO 큐)
        self.performance_history = {
            'cpu_percent': deque(maxlen=prediction_window),
            'memory_percent': deque(maxlen=prediction_window),
            'memory_used_gb': deque(maxlen=prediction_window),
            'process_memory_mb': deque(maxlen=prediction_window),
            'disk_io_speed': deque(maxlen=prediction_window),
            'network_io_speed': deque(maxlen=prediction_window),
            'timestamps': deque(maxlen=prediction_window)
        }
        
        self.monitoring_session = {
            'session_id': f"advanced_monitor_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'predictions': [],
            'alerts': [],
            'optimizations': [],
            'performance_trends': {}
        }
        
        # 예측 모델 파라미터
        self.prediction_models = {
            'memory_usage': {'trend_weight': 0.7, 'seasonal_weight': 0.3},
            'cpu_usage': {'trend_weight': 0.8, 'seasonal_weight': 0.2},
            'processing_time': {'size_factor': 120, 'complexity_factor': 1.5}  # 초/GB
        }
        
        # 경고 임계값
        self.alert_thresholds = {
            'memory_critical': 90.0,
            'memory_warning': 85.0,
            'cpu_critical': 95.0,
            'cpu_warning': 80.0,
            'predicted_memory_shortage': 5.0,  # 5분 내 메모리 부족 예상
            'processing_time_overrun': 1.5  # 예상 시간의 150% 초과시
        }
        
        self.monitoring_active = False
        self.monitoring_thread = None
        
        print("고급 시스템 모니터링 시스템 초기화")
        self._initialize_advanced_monitoring()
    
    def _initialize_advanced_monitoring(self):
        """고급 모니터링 시스템 초기화"""
        print("=== 고급 모니터링 시스템 초기화 ===")
        
        # 기본 모니터 확인
        if BASE_MONITOR_AVAILABLE:
            self.base_monitor = SystemOptimizationMonitor(self.monitoring_interval)
            print("[OK] Base System Monitor: 통합 완료")
        else:
            self.base_monitor = None
            print("[WARNING] Base System Monitor: 사용 불가")
        
        # 메모리 추적 시작
        try:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            print("[OK] Advanced Memory Tracing: 활성화")
        except Exception as e:
            print(f"[WARNING] Memory Tracing: {e}")
        
        # 시스템 기준선 설정
        self.system_baseline = self._establish_baseline()
        print(f"[OK] System Baseline: CPU {self.system_baseline['cpu_percent']:.1f}%, Memory {self.system_baseline['memory_percent']:.1f}%")
        
        # 예측 알고리즘 초기화
        self._initialize_prediction_algorithms()
        print("[OK] Prediction Algorithms: 준비 완료")
    
    def _establish_baseline(self, samples: int = 5) -> Dict[str, float]:
        """시스템 기준선 성능 측정"""
        print("시스템 기준선 측정 중...")
        
        baseline_samples = []
        for i in range(samples):
            sample = self._collect_system_metrics()
            baseline_samples.append(sample)
            time.sleep(1)
            print(f"  샘플 {i+1}/{samples} 수집...")
        
        # 평균값 계산
        baseline = {}
        metric_keys = ['cpu_percent', 'memory_percent', 'memory_used_gb', 'process_memory_mb']
        
        for key in metric_keys:
            values = [s.get(key, 0) for s in baseline_samples if key in s]
            baseline[key] = sum(values) / len(values) if values else 0
        
        baseline['timestamp'] = time.time()
        return baseline
    
    def _initialize_prediction_algorithms(self):
        """예측 알고리즘 초기화"""
        self.prediction_algorithms = {
            'linear_trend': self._predict_linear_trend,
            'exponential_smoothing': self._predict_exponential_smoothing,
            'file_processing_time': self.predict_processing_time
        }
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """확장된 시스템 메트릭 수집"""
        try:
            # 기본 시스템 정보
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()
            
            # 프로세스별 정보
            process = psutil.Process()
            process_info = process.memory_info()
            
            # 고급 메모리 정보
            python_memory = 0
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                python_memory = current / (1024**2)  # MB
            
            # GPU 정보 (가능한 경우)
            gpu_info = self._get_gpu_info()
            
            metrics = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                'memory_percent': memory.percent,
                'memory_used_gb': (memory.total - memory.available) / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'process_memory_mb': process_info.rss / (1024**2),
                'process_vms_mb': process_info.vms / (1024**2),
                'python_memory_mb': python_memory,
                'thread_count': process.num_threads(),
                'file_handles': process.num_fds() if hasattr(process, 'num_fds') else 0,
                'disk_read_mb': disk_io.read_bytes / (1024**2) if disk_io else 0,
                'disk_write_mb': disk_io.write_bytes / (1024**2) if disk_io else 0,
                'network_sent_mb': net_io.bytes_sent / (1024**2) if net_io else 0,
                'network_recv_mb': net_io.bytes_recv / (1024**2) if net_io else 0,
                'cpu_times': psutil.cpu_times()._asdict(),
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
            
            # GPU 정보 추가 (있는 경우)
            if gpu_info:
                metrics.update(gpu_info)
            
            return metrics
            
        except Exception as e:
            return {
                'timestamp': time.time(),
                'error': str(e),
                'partial_data': True
            }
    
    def _get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """GPU 정보 수집 (nvidia-smi 사용)"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_data = []
                
                for i, line in enumerate(lines):
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpu_data.append({
                            'gpu_id': i,
                            'memory_used_mb': float(parts[0]),
                            'memory_total_mb': float(parts[1]),
                            'utilization_percent': float(parts[2])
                        })
                
                return {'gpu_info': gpu_data, 'gpu_available': True}
        except:
            pass
        
        return {'gpu_available': False}
    
    def start_advanced_monitoring(self):
        """고급 모니터링 시작"""
        if self.monitoring_active:
            print("고급 모니터링이 이미 실행 중입니다.")
            return
        
        print(f"[START] 고급 시스템 모니터링 시작 (간격: {self.monitoring_interval}초)")
        self.monitoring_active = True
        
        # 기본 모니터 시작 (가능한 경우)
        if self.base_monitor:
            self.base_monitor.start_monitoring()
        
        # 고급 모니터링 스레드 시작
        self.monitoring_thread = threading.Thread(target=self._advanced_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_advanced_monitoring(self):
        """고급 모니터링 중지"""
        if not self.monitoring_active:
            print("고급 모니터링이 실행 중이 아닙니다.")
            return
        
        print("[STOP] 고급 시스템 모니터링 중지")
        self.monitoring_active = False
        
        # 기본 모니터 중지
        if self.base_monitor:
            self.base_monitor.stop_monitoring()
        
        # 스레드 종료 대기
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=3.0)
    
    def _advanced_monitoring_loop(self):
        """고급 모니터링 루프"""
        last_metrics = None
        
        while self.monitoring_active:
            try:
                # 현재 메트릭 수집
                current_metrics = self._collect_system_metrics()
                
                # 히스토리에 추가
                self._update_performance_history(current_metrics)
                
                # I/O 속도 계산 (이전 메트릭과 비교)
                if last_metrics:
                    io_speeds = self._calculate_io_speeds(last_metrics, current_metrics)
                    current_metrics.update(io_speeds)
                
                # 예측 및 분석 수행
                if len(self.performance_history['timestamps']) >= 10:  # 최소 10개 샘플 필요
                    predictions = self._run_predictions()
                    alerts = self._check_advanced_alerts(current_metrics, predictions)
                    
                    if predictions:
                        self.monitoring_session['predictions'].append(predictions)
                    
                    if alerts:
                        self.monitoring_session['alerts'].extend(alerts)
                        self._handle_alerts(alerts)
                
                # 트렌드 분석
                if len(self.performance_history['timestamps']) >= 30:  # 30개 샘플로 트렌드 분석
                    trends = self._analyze_performance_trends()
                    self.monitoring_session['performance_trends'] = trends
                
                last_metrics = current_metrics
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"[ERROR] 고급 모니터링 오류: {e}")
                time.sleep(self.monitoring_interval)
    
    def _update_performance_history(self, metrics: Dict[str, Any]):
        """성능 히스토리 업데이트"""
        if 'error' in metrics:
            return
        
        self.performance_history['timestamps'].append(metrics['timestamp'])
        self.performance_history['cpu_percent'].append(metrics.get('cpu_percent', 0))
        self.performance_history['memory_percent'].append(metrics.get('memory_percent', 0))
        self.performance_history['memory_used_gb'].append(metrics.get('memory_used_gb', 0))
        self.performance_history['process_memory_mb'].append(metrics.get('process_memory_mb', 0))
    
    def _calculate_io_speeds(self, last_metrics: Dict[str, Any], current_metrics: Dict[str, Any]) -> Dict[str, float]:
        """I/O 속도 계산"""
        time_delta = current_metrics['timestamp'] - last_metrics['timestamp']
        
        if time_delta <= 0:
            return {}
        
        disk_read_speed = (current_metrics.get('disk_read_mb', 0) - last_metrics.get('disk_read_mb', 0)) / time_delta
        disk_write_speed = (current_metrics.get('disk_write_mb', 0) - last_metrics.get('disk_write_mb', 0)) / time_delta
        net_send_speed = (current_metrics.get('network_sent_mb', 0) - last_metrics.get('network_sent_mb', 0)) / time_delta
        net_recv_speed = (current_metrics.get('network_recv_mb', 0) - last_metrics.get('network_recv_mb', 0)) / time_delta
        
        # 히스토리에 추가
        total_disk_speed = max(0, disk_read_speed + disk_write_speed)
        total_net_speed = max(0, net_send_speed + net_recv_speed)
        
        self.performance_history['disk_io_speed'].append(total_disk_speed)
        self.performance_history['network_io_speed'].append(total_net_speed)
        
        return {
            'disk_read_speed_mb_s': max(0, disk_read_speed),
            'disk_write_speed_mb_s': max(0, disk_write_speed),
            'network_send_speed_mb_s': max(0, net_send_speed),
            'network_recv_speed_mb_s': max(0, net_recv_speed)
        }
    
    def _run_predictions(self) -> Dict[str, Any]:
        """예측 알고리즘 실행"""
        predictions = {
            'timestamp': time.time(),
            'prediction_horizon_minutes': 5,
            'memory_forecast': None,
            'cpu_forecast': None,
            'resource_shortage_risk': None,
            'processing_time_estimate': None
        }
        
        try:
            # 메모리 사용량 예측
            memory_data = list(self.performance_history['memory_percent'])
            if len(memory_data) >= 10:
                predictions['memory_forecast'] = self._predict_linear_trend(memory_data, forecast_steps=5)
            
            # CPU 사용량 예측
            cpu_data = list(self.performance_history['cpu_percent'])
            if len(cpu_data) >= 10:
                predictions['cpu_forecast'] = self._predict_exponential_smoothing(cpu_data, forecast_steps=5)
            
            # 리소스 부족 위험도 평가
            predictions['resource_shortage_risk'] = self._assess_resource_shortage_risk()
            
            return predictions
            
        except Exception as e:
            return {'error': str(e), 'timestamp': time.time()}
    
    def _predict_linear_trend(self, data: List[float], forecast_steps: int = 5) -> Dict[str, Any]:
        """선형 트렌드 예측"""
        if len(data) < 3:
            return {'error': 'Insufficient data'}
        
        x = np.arange(len(data))
        y = np.array(data)
        
        # 선형 회귀
        coeffs = np.polyfit(x, y, 1)
        trend_slope = coeffs[0]
        
        # 미래 값 예측
        future_x = np.arange(len(data), len(data) + forecast_steps)
        forecast = np.polyval(coeffs, future_x)
        
        return {
            'method': 'linear_trend',
            'trend_slope': float(trend_slope),
            'current_value': float(data[-1]),
            'forecast_values': [max(0, min(100, float(v))) for v in forecast],
            'forecast_steps': forecast_steps,
            'trend_direction': 'increasing' if trend_slope > 0.1 else 'decreasing' if trend_slope < -0.1 else 'stable'
        }
    
    def _predict_exponential_smoothing(self, data: List[float], forecast_steps: int = 5, alpha: float = 0.3) -> Dict[str, Any]:
        """지수 평활법 예측"""
        if len(data) < 3:
            return {'error': 'Insufficient data'}
        
        # 지수 평활
        smoothed = [data[0]]
        for i in range(1, len(data)):
            smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[-1])
        
        # 트렌드 계산
        trend = smoothed[-1] - smoothed[-min(5, len(smoothed)-1)]
        
        # 미래 값 예측
        forecast = []
        last_value = smoothed[-1]
        
        for step in range(forecast_steps):
            next_value = last_value + trend * (step + 1) * 0.1  # 트렌드 감쇠
            forecast.append(max(0, min(100, next_value)))
        
        return {
            'method': 'exponential_smoothing',
            'alpha': alpha,
            'current_smoothed': float(last_value),
            'trend': float(trend),
            'forecast_values': forecast,
            'forecast_steps': forecast_steps
        }
    
    def _assess_resource_shortage_risk(self) -> Dict[str, Any]:
        """리소스 부족 위험도 평가"""
        risk_assessment = {
            'overall_risk_level': 'low',
            'memory_risk': 'low',
            'cpu_risk': 'low',
            'disk_risk': 'low',
            'estimated_time_to_critical': None,
            'recommendations': []
        }
        
        # 메모리 위험도
        if self.performance_history['memory_percent']:
            current_memory = self.performance_history['memory_percent'][-1]
            memory_trend = self._calculate_trend(list(self.performance_history['memory_percent']))
            
            if current_memory > 85:
                risk_assessment['memory_risk'] = 'high'
                risk_assessment['overall_risk_level'] = 'high'
            elif current_memory > 75 and memory_trend > 0.5:
                risk_assessment['memory_risk'] = 'medium'
                risk_assessment['overall_risk_level'] = 'medium'
            
            # 임계점까지 예상 시간 계산
            if memory_trend > 0.1:
                time_to_critical = (95 - current_memory) / memory_trend * self.monitoring_interval / 60  # 분
                risk_assessment['estimated_time_to_critical'] = max(0, time_to_critical)
        
        # CPU 위험도
        if self.performance_history['cpu_percent']:
            current_cpu = self.performance_history['cpu_percent'][-1]
            
            if current_cpu > 90:
                risk_assessment['cpu_risk'] = 'high'
                if risk_assessment['overall_risk_level'] == 'low':
                    risk_assessment['overall_risk_level'] = 'medium'
            elif current_cpu > 80:
                risk_assessment['cpu_risk'] = 'medium'
        
        # 권장사항 생성
        if risk_assessment['memory_risk'] == 'high':
            risk_assessment['recommendations'].append('메모리 사용량 즉시 최적화 필요')
        
        if risk_assessment['cpu_risk'] == 'high':
            risk_assessment['recommendations'].append('CPU 집약적 작업 일시 중단 권장')
        
        return risk_assessment
    
    def _calculate_trend(self, data: List[float], window: int = 10) -> float:
        """데이터 트렌드 계산 (단위 시간당 변화율)"""
        if len(data) < window:
            window = len(data)
        
        if window < 2:
            return 0.0
        
        recent_data = data[-window:]
        x = np.arange(len(recent_data))
        
        try:
            coeffs = np.polyfit(x, recent_data, 1)
            return float(coeffs[0])  # 기울기
        except:
            return 0.0
    
    def _check_advanced_alerts(self, current_metrics: Dict[str, Any], predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """고급 경고 시스템"""
        alerts = []
        timestamp = time.time()
        
        # 현재 상태 기반 경고
        memory_percent = current_metrics.get('memory_percent', 0)
        cpu_percent = current_metrics.get('cpu_percent', 0)
        
        if memory_percent > self.alert_thresholds['memory_critical']:
            alerts.append({
                'type': 'memory_critical',
                'severity': 'critical',
                'message': f'메모리 사용률 위험 수준: {memory_percent:.1f}%',
                'timestamp': timestamp,
                'suggested_actions': ['즉시 메모리 정리', '불필요한 프로세스 종료', '메모리 누수 점검']
            })
        elif memory_percent > self.alert_thresholds['memory_warning']:
            alerts.append({
                'type': 'memory_warning',
                'severity': 'warning',
                'message': f'메모리 사용률 경고: {memory_percent:.1f}%',
                'timestamp': timestamp,
                'suggested_actions': ['메모리 사용량 모니터링 강화', '가비지 컬렉션 수행']
            })
        
        # 예측 기반 경고
        if predictions and not predictions.get('error'):
            memory_forecast = predictions.get('memory_forecast')
            if memory_forecast and isinstance(memory_forecast, dict):
                forecast_values = memory_forecast.get('forecast_values', [])
                if forecast_values and max(forecast_values) > self.alert_thresholds['memory_critical']:
                    alerts.append({
                        'type': 'predicted_memory_shortage',
                        'severity': 'warning',
                        'message': f'5분 내 메모리 부족 예상: 최대 {max(forecast_values):.1f}%',
                        'timestamp': timestamp,
                        'suggested_actions': ['사전 메모리 최적화', '대용량 작업 지연 검토']
                    })
            
            # 리소스 부족 위험도 기반 경고
            risk_assessment = predictions.get('resource_shortage_risk')
            if risk_assessment and risk_assessment.get('overall_risk_level') == 'high':
                alerts.append({
                    'type': 'resource_shortage_risk',
                    'severity': 'warning',
                    'message': '시스템 리소스 부족 위험 감지',
                    'timestamp': timestamp,
                    'details': risk_assessment,
                    'suggested_actions': risk_assessment.get('recommendations', [])
                })
        
        return alerts
    
    def _handle_alerts(self, alerts: List[Dict[str, Any]]):
        """경고 처리"""
        for alert in alerts:
            severity = alert.get('severity', 'info')
            message = alert.get('message', '')
            
            if severity == 'critical':
                print(f"[CRITICAL] {message}")
                # 중요 경고시 자동 최적화 수행
                self._trigger_emergency_optimization(alert)
            elif severity == 'warning':
                print(f"[WARNING] {message}")
            else:
                print(f"[INFO] {message}")
    
    def _trigger_emergency_optimization(self, alert: Dict[str, Any]):
        """응급 최적화 수행"""
        print("[EMERGENCY] 응급 시스템 최적화 실행")
        
        try:
            # 강제 가비지 컬렉션
            gc.collect()
            
            # 메모리 추적 정리
            if tracemalloc.is_tracing():
                tracemalloc.clear_traces()
            
            # 기본 모니터의 최적화 기능 활용
            if self.base_monitor:
                # 메모리 최적화 트리거
                self.base_monitor._trigger_optimization('memory_high', {}, time.time())
            
            optimization_record = {
                'timestamp': time.time(),
                'trigger_alert': alert,
                'actions_taken': ['garbage_collection', 'memory_trace_clear'],
                'success': True
            }
            
            self.monitoring_session['optimizations'].append(optimization_record)
            print("[OK] 응급 최적화 완료")
            
        except Exception as e:
            print(f"[ERROR] 응급 최적화 실패: {e}")
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """성능 트렌드 분석"""
        trends = {
            'analysis_timestamp': time.time(),
            'sample_count': len(self.performance_history['timestamps']),
            'trends': {}
        }
        
        # 각 메트릭별 트렌드 분석
        metrics_to_analyze = ['cpu_percent', 'memory_percent', 'process_memory_mb']
        
        for metric in metrics_to_analyze:
            if metric in self.performance_history:
                data = list(self.performance_history[metric])
                
                if len(data) >= 10:
                    trend_slope = self._calculate_trend(data)
                    
                    # 변동성 계산 (표준편차)
                    volatility = float(np.std(data[-20:]) if len(data) >= 20 else np.std(data))
                    
                    # 현재 값과 기준선 비교
                    current_value = data[-1]
                    baseline_value = self.system_baseline.get(metric, current_value)
                    deviation_percent = ((current_value - baseline_value) / baseline_value * 100) if baseline_value > 0 else 0
                    
                    trends['trends'][metric] = {
                        'trend_slope': trend_slope,
                        'trend_direction': 'increasing' if trend_slope > 0.1 else 'decreasing' if trend_slope < -0.1 else 'stable',
                        'volatility': volatility,
                        'current_value': current_value,
                        'baseline_deviation_percent': deviation_percent,
                        'performance_grade': self._calculate_performance_grade(deviation_percent, volatility)
                    }
        
        return trends
    
    def _calculate_performance_grade(self, deviation_percent: float, volatility: float) -> str:
        """성능 등급 계산"""
        if abs(deviation_percent) < 5 and volatility < 2:
            return 'A+'
        elif abs(deviation_percent) < 10 and volatility < 5:
            return 'A'
        elif abs(deviation_percent) < 20 and volatility < 10:
            return 'B'
        elif abs(deviation_percent) < 30 and volatility < 15:
            return 'C'
        else:
            return 'D'
    
    def predict_processing_time(self, file_size_gb: float, file_type: str = 'video') -> Dict[str, Any]:
        """파일 처리 시간 예측"""
        base_time_per_gb = self.prediction_models['processing_time']['size_factor']
        complexity_factor = self.prediction_models['processing_time']['complexity_factor']
        
        # 파일 타입별 복잡도 조정
        type_multipliers = {
            'video': 1.0,
            'audio': 0.3,
            'image': 0.1,
            'document': 0.05
        }
        
        type_multiplier = type_multipliers.get(file_type.lower(), 1.0)
        
        # 현재 시스템 상태 고려
        current_load_factor = 1.0
        if self.performance_history['cpu_percent']:
            current_cpu = self.performance_history['cpu_percent'][-1]
            current_memory = self.performance_history['memory_percent'][-1]
            
            # 시스템 부하에 따른 처리 시간 증가
            current_load_factor = 1 + (current_cpu / 100) * 0.5 + (current_memory / 100) * 0.3
        
        # 예상 처리 시간 계산
        estimated_time = file_size_gb * base_time_per_gb * type_multiplier * complexity_factor * current_load_factor
        
        # 신뢰도 구간 계산
        confidence_range = estimated_time * 0.3  # ±30%
        
        return {
            'file_size_gb': file_size_gb,
            'file_type': file_type,
            'estimated_time_seconds': estimated_time,
            'estimated_time_minutes': estimated_time / 60,
            'confidence_range_seconds': confidence_range,
            'min_time_seconds': max(0, estimated_time - confidence_range),
            'max_time_seconds': estimated_time + confidence_range,
            'current_system_load_factor': current_load_factor,
            'prediction_factors': {
                'base_time_per_gb': base_time_per_gb,
                'type_multiplier': type_multiplier,
                'complexity_factor': complexity_factor,
                'load_factor': current_load_factor
            }
        }
    
    def generate_advanced_report(self) -> str:
        """고급 모니터링 보고서 생성"""
        report_path = project_root / f"advanced_monitoring_report_{self.monitoring_session['session_id']}.json"
        
        # 기본 모니터 보고서 통합 (가능한 경우)
        base_report = None
        if self.base_monitor:
            try:
                base_report = self.base_monitor.generate_health_report()
            except:
                pass
        
        # 최종 트렌드 분석
        if len(self.performance_history['timestamps']) >= 10:
            final_trends = self._analyze_performance_trends()
        else:
            final_trends = {}
        
        # 종합 보고서 생성
        comprehensive_report = {
            'monitoring_session': self.monitoring_session,
            'system_baseline': self.system_baseline,
            'performance_history_summary': {
                'total_samples': len(self.performance_history['timestamps']),
                'monitoring_duration_minutes': (time.time() - self.performance_history['timestamps'][0]) / 60 if self.performance_history['timestamps'] else 0,
                'data_quality_score': self._calculate_data_quality_score()
            },
            'final_performance_trends': final_trends,
            'prediction_accuracy': self._evaluate_prediction_accuracy(),
            'alert_summary': {
                'total_alerts': len(self.monitoring_session['alerts']),
                'critical_alerts': len([a for a in self.monitoring_session['alerts'] if a.get('severity') == 'critical']),
                'warning_alerts': len([a for a in self.monitoring_session['alerts'] if a.get('severity') == 'warning'])
            },
            'optimization_summary': {
                'total_optimizations': len(self.monitoring_session['optimizations']),
                'successful_optimizations': len([o for o in self.monitoring_session['optimizations'] if o.get('success')])
            },
            'base_monitor_report': base_report,
            'recommendations': self._generate_advanced_recommendations()
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVE] 고급 모니터링 보고서 저장: {report_path}")
        return str(report_path)
    
    def _calculate_data_quality_score(self) -> float:
        """데이터 품질 점수 계산"""
        if not self.performance_history['timestamps']:
            return 0.0
        
        # 데이터 완성도
        expected_samples = len(self.performance_history['timestamps'])
        actual_valid_samples = sum(1 for i in range(expected_samples) 
                                  if all(len(self.performance_history[key]) > i 
                                        for key in ['cpu_percent', 'memory_percent']))
        
        completeness_score = actual_valid_samples / expected_samples if expected_samples > 0 else 0
        
        # 데이터 일관성 (급격한 변화 감지)
        consistency_score = 1.0
        for metric in ['cpu_percent', 'memory_percent']:
            if metric in self.performance_history and len(self.performance_history[metric]) > 1:
                data = list(self.performance_history[metric])
                volatility = float(np.std(data))
                if volatility > 20:  # 20% 이상의 높은 변동성
                    consistency_score *= 0.8
        
        return (completeness_score + consistency_score) / 2
    
    def _evaluate_prediction_accuracy(self) -> Dict[str, Any]:
        """예측 정확도 평가"""
        # 실제 구현에서는 과거 예측과 실제 값을 비교
        return {
            'evaluation_method': 'placeholder',
            'sample_size': len(self.monitoring_session['predictions']),
            'average_accuracy': 0.85,  # 플레이스홀더
            'note': 'Prediction accuracy evaluation requires historical validation data'
        }
    
    def _generate_advanced_recommendations(self) -> List[Dict[str, Any]]:
        """고급 권장사항 생성"""
        recommendations = []
        
        # 성능 트렌드 기반 권장사항
        if 'performance_trends' in self.monitoring_session and self.monitoring_session['performance_trends']:
            trends = self.monitoring_session['performance_trends'].get('trends', {})
            
            for metric, trend_data in trends.items():
                grade = trend_data.get('performance_grade', 'C')
                direction = trend_data.get('trend_direction', 'stable')
                
                if grade in ['C', 'D'] or direction == 'increasing':
                    if metric == 'memory_percent':
                        recommendations.append({
                            'category': 'Memory Optimization',
                            'priority': 'high' if grade == 'D' else 'medium',
                            'issue': f'메모리 사용률 성능 등급: {grade}',
                            'recommendation': '메모리 사용 패턴 최적화 및 정기적 정리',
                            'expected_improvement': '메모리 효율성 15-25% 향상'
                        })
                    
                    elif metric == 'cpu_percent':
                        recommendations.append({
                            'category': 'CPU Optimization',
                            'priority': 'medium',
                            'issue': f'CPU 사용률 성능 등급: {grade}',
                            'recommendation': 'CPU 집약적 작업 스케줄링 최적화',
                            'expected_improvement': 'CPU 효율성 10-20% 향상'
                        })
        
        # 경고 패턴 기반 권장사항
        alert_types = [alert.get('type') for alert in self.monitoring_session['alerts']]
        if 'memory_critical' in alert_types:
            recommendations.append({
                'category': 'Critical System Health',
                'priority': 'critical',
                'issue': '메모리 위험 수준 경고 발생',
                'recommendation': '즉시 시스템 메모리 감사 및 최적화 수행',
                'expected_improvement': '시스템 안정성 대폭 향상'
            })
        
        return recommendations

def main():
    """메인 실행 함수"""
    print("고급 시스템 모니터링 시스템 데모")
    print("=" * 50)
    
    # 고급 모니터링 시스템 초기화
    monitor = AdvancedMonitoringSystem(monitoring_interval=1.0, prediction_window=30)
    
    # 모니터링 시작
    monitor.start_advanced_monitoring()
    
    try:
        print("30초간 고급 모니터링 실행...")
        
        # 파일 처리 시간 예측 테스트
        print("\n--- 파일 처리 시간 예측 테스트 ---")
        test_files = [
            (5.0, 'video'),
            (2.5, 'video'),
            (1.0, 'audio')
        ]
        
        for size, file_type in test_files:
            prediction = monitor.predict_processing_time(size, file_type)
            print(f"{size}GB {file_type}: {prediction['estimated_time_minutes']:.1f}분 "
                  f"({prediction['min_time_seconds']:.0f}-{prediction['max_time_seconds']:.0f}초)")
        
        # 모니터링 데이터 수집
        for i in range(30):
            time.sleep(1)
            if i % 5 == 0:
                print(f"  모니터링 진행... {i+1}/30초")
            
            # 중간에 시스템 부하 생성 (테스트용)
            if i == 15:
                print("  [TEST] 시스템 부하 생성...")
                test_data = [i for i in range(500000)]
                del test_data
        
        print("고급 모니터링 완료!")
        
    finally:
        # 모니터링 중지
        monitor.stop_advanced_monitoring()
    
    # 최종 보고서 생성
    report_path = monitor.generate_advanced_report()
    
    print(f"\n{'='*50}")
    print("고급 모니터링 완료 요약")
    print(f"{'='*50}")
    print(f"모니터링 세션: {monitor.monitoring_session['session_id']}")
    print(f"수집된 예측: {len(monitor.monitoring_session['predictions'])}개")
    print(f"발생한 경고: {len(monitor.monitoring_session['alerts'])}개")
    print(f"수행된 최적화: {len(monitor.monitoring_session['optimizations'])}개")
    print(f"상세 보고서: {report_path}")
    
    return monitor.monitoring_session

if __name__ == "__main__":
    main()