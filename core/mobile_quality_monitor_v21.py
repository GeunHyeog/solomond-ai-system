"""
📱 Mobile Quality Monitor v2.1
모바일 현장 품질 실시간 모니터링 모듈

주요 기능:
- 실시간 음성/이미지 품질 모니터링
- 모바일 UI 최적화된 품질 대시보드
- 현장 즉시 피드백 시스템
- 주얼리쇼/회의 현장 특화 모니터링
- 배터리/성능 최적화 모니터링
"""

import numpy as np
import time
import threading
import queue
from typing import Dict, List, Tuple, Optional, Callable, Any
from datetime import datetime, timedelta
import logging
import json
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings("ignore")

@dataclass
class QualityMetric:
    """품질 지표 데이터 클래스"""
    timestamp: float
    source_type: str  # 'audio', 'image', 'ocr', 'consistency'
    metric_name: str
    value: float
    level: str  # 'excellent', 'good', 'fair', 'poor'
    status: str
    color: str
    details: Dict = None

@dataclass
class MonitoringSession:
    """모니터링 세션 정보"""
    session_id: str
    start_time: float
    session_type: str  # 'jewelry_show', 'meeting', 'presentation', 'general'
    location: str
    device_info: Dict
    quality_targets: Dict

class MobileQualityMonitor:
    """모바일 현장 품질 실시간 모니터"""
    
    def __init__(self, session_config: Dict = None):
        self.logger = logging.getLogger(__name__)
        
        # 모니터링 상태
        self.is_monitoring = False
        self.current_session = None
        self.quality_history = []
        self.alert_callbacks = []
        
        # 실시간 데이터 큐
        self.audio_queue = queue.Queue(maxsize=100)
        self.image_queue = queue.Queue(maxsize=50)
        self.ocr_queue = queue.Queue(maxsize=30)
        
        # 모니터링 스레드
        self.monitor_thread = None
        self.alert_thread = None
        
        # 설정값
        self.config = self._init_default_config()
        if session_config:
            self.config.update(session_config)
        
        # 품질 임계값 (모바일 최적화)
        self.mobile_thresholds = {
            'audio_snr_min': 15.0,           # 모바일 환경 최소 SNR
            'audio_confidence_min': 70.0,    # 최소 음성 신뢰도
            'image_sharpness_min': 80.0,     # 최소 선명도
            'ocr_confidence_min': 70.0,      # 최소 OCR 신뢰도
            'consistency_score_min': 0.6,    # 최소 일관성 점수
        }

    def start_monitoring_session(self, 
                                session_type: str = 'general',
                                location: str = 'unknown',
                                custom_targets: Dict = None) -> str:
        """모니터링 세션 시작"""
        try:
            if self.is_monitoring:
                self.stop_monitoring_session()
            
            session_id = f"session_{int(time.time())}"
            quality_targets = self._get_quality_targets(session_type)
            if custom_targets:
                quality_targets.update(custom_targets)
            
            device_info = self._collect_device_info()
            
            self.current_session = MonitoringSession(
                session_id=session_id,
                start_time=time.time(),
                session_type=session_type,
                location=location,
                device_info=device_info,
                quality_targets=quality_targets
            )
            
            self.is_monitoring = True
            self.quality_history.clear()
            self._start_monitoring_threads()
            
            self.logger.info(f"모니터링 세션 시작: {session_id} ({session_type})")
            return session_id
            
        except Exception as e:
            self.logger.error(f"모니터링 세션 시작 오류: {str(e)}")
            raise

    def stop_monitoring_session(self) -> Dict:
        """모니터링 세션 종료 및 리포트 생성"""
        try:
            if not self.is_monitoring:
                return {'status': 'no_active_session'}
            
            self.is_monitoring = False
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2.0)
            
            if self.alert_thread and self.alert_thread.is_alive():
                self.alert_thread.join(timeout=1.0)
            
            session_report = self._generate_session_report()
            self.logger.info(f"모니터링 세션 종료: {self.current_session.session_id}")
            self.current_session = None
            
            return session_report
            
        except Exception as e:
            self.logger.error(f"모니터링 세션 종료 오류: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def add_audio_quality_data(self, quality_data: Dict) -> None:
        """음성 품질 데이터 추가"""
        try:
            if not self.is_monitoring:
                return
            
            metrics = self._extract_audio_metrics(quality_data)
            for metric in metrics:
                try:
                    self.audio_queue.put_nowait(metric)
                except queue.Full:
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.put_nowait(metric)
                    except queue.Empty:
                        pass
            
        except Exception as e:
            self.logger.error(f"음성 품질 데이터 추가 오류: {str(e)}")

    def add_image_quality_data(self, quality_data: Dict) -> None:
        """이미지 품질 데이터 추가"""
        try:
            if not self.is_monitoring:
                return
            
            metrics = self._extract_image_metrics(quality_data)
            for metric in metrics:
                try:
                    self.image_queue.put_nowait(metric)
                except queue.Full:
                    try:
                        self.image_queue.get_nowait()
                        self.image_queue.put_nowait(metric)
                    except queue.Empty:
                        pass
            
        except Exception as e:
            self.logger.error(f"이미지 품질 데이터 추가 오류: {str(e)}")

    def get_realtime_dashboard_data(self) -> Dict:
        """실시간 대시보드 데이터 반환"""
        try:
            if not self.is_monitoring:
                return {'status': 'not_monitoring'}
            
            current_status = self._get_current_quality_status()
            recent_history = self._get_recent_history_for_ui()
            system_status = self._get_system_status()
            active_alerts = self._get_active_alerts()
            
            session_info = {
                'session_id': self.current_session.session_id,
                'session_type': self.current_session.session_type,
                'location': self.current_session.location,
                'duration': time.time() - self.current_session.start_time,
                'start_time': self.current_session.start_time
            } if self.current_session else {}
            
            return {
                'status': 'monitoring',
                'timestamp': time.time(),
                'session_info': session_info,
                'current_status': current_status,
                'recent_history': recent_history,
                'system_status': system_status,
                'active_alerts': active_alerts
            }
            
        except Exception as e:
            self.logger.error(f"대시보드 데이터 조회 오류: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def add_alert_callback(self, callback: Callable[[Dict], None]) -> None:
        """알림 콜백 함수 추가"""
        self.alert_callbacks.append(callback)

    # === 내부 메서드들 ===
    
    def _init_default_config(self) -> Dict:
        """기본 설정 초기화"""
        return {
            'max_history_size': 1000,
            'quality_check_interval': 0.5,
            'alert_cooldown': 10.0,
        }
    
    def _get_quality_targets(self, session_type: str) -> Dict:
        """세션 유형별 품질 목표 설정"""
        base_targets = {
            'audio_snr': 20.0,
            'audio_confidence': 80.0,
            'image_sharpness': 100.0,
            'ocr_confidence': 80.0,
            'overall_score': 0.8
        }
        
        if session_type == 'jewelry_show':
            base_targets.update({
                'image_sharpness': 150.0,
                'image_contrast': 80.0,
                'ocr_confidence': 85.0,
            })
        elif session_type == 'meeting':
            base_targets.update({
                'audio_snr': 25.0,
                'audio_confidence': 85.0,
            })
        
        return base_targets
    
    def _collect_device_info(self) -> Dict:
        """디바이스 정보 수집"""
        device_info = {
            'timestamp': time.time(),
            'platform': 'mobile',
            'available_memory': 'unknown',
            'battery_level': 'unknown',
        }
        
        try:
            import psutil
            device_info.update({
                'available_memory': psutil.virtual_memory().percent,
            })
        except ImportError:
            pass
        
        return device_info
    
    def _start_monitoring_threads(self) -> None:
        """모니터링 스레드들 시작"""
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        self.alert_thread = threading.Thread(
            target=self._alert_processing_loop,
            daemon=True
        )
        self.alert_thread.start()
    
    def _monitoring_loop(self) -> None:
        """메인 모니터링 루프"""
        while self.is_monitoring:
            try:
                self._process_queued_metrics()
                time.sleep(self.config.get('quality_check_interval', 0.5))
            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {str(e)}")
                time.sleep(1.0)
    
    def _process_queued_metrics(self) -> None:
        """큐에 있는 품질 지표들 처리"""
        # 오디오 메트릭 처리
        while not self.audio_queue.empty():
            try:
                metric = self.audio_queue.get_nowait()
                self._process_metric(metric)
            except queue.Empty:
                break
        
        # 이미지 메트릭 처리
        while not self.image_queue.empty():
            try:
                metric = self.image_queue.get_nowait()
                self._process_metric(metric)
            except queue.Empty:
                break
    
    def _process_metric(self, metric: QualityMetric) -> None:
        """개별 품질 지표 처리"""
        self.quality_history.append(metric)
        
        max_size = self.config.get('max_history_size', 1000)
        if len(self.quality_history) > max_size:
            self.quality_history = self.quality_history[-max_size:]
        
        self._check_metric_alerts(metric)
    
    def _check_metric_alerts(self, metric: QualityMetric) -> None:
        """메트릭 기반 알림 체크"""
        alert_triggered = False
        alert_data = {
            'timestamp': metric.timestamp,
            'source_type': metric.source_type,
            'metric_name': metric.metric_name,
            'value': metric.value,
            'level': metric.level,
            'alert_type': 'quality_warning'
        }
        
        if metric.source_type == 'audio':
            if metric.metric_name == 'snr' and metric.value < self.mobile_thresholds['audio_snr_min']:
                alert_data.update({
                    'title': 'SNR 낮음',
                    'message': f'신호 대 잡음비가 {metric.value:.1f}dB로 낮습니다',
                    'suggestion': '마이크를 가까이 하거나 조용한 곳으로 이동하세요'
                })
                alert_triggered = True
        
        if alert_triggered:
            self._trigger_alert(alert_data)
    
    def _trigger_alert(self, alert_data: Dict) -> None:
        """알림 트리거"""
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                self.logger.error(f"알림 콜백 오류: {str(e)}")
    
    def _alert_processing_loop(self) -> None:
        """알림 처리 루프"""
        while self.is_monitoring:
            try:
                time.sleep(2.0)
            except Exception as e:
                self.logger.error(f"알림 처리 루프 오류: {str(e)}")
                time.sleep(5.0)
    
    def _extract_audio_metrics(self, quality_data: Dict) -> List[QualityMetric]:
        """음성 품질 데이터에서 메트릭 추출"""
        metrics = []
        timestamp = time.time()
        
        if 'snr_db' in quality_data:
            snr_value = quality_data['snr_db']
            metrics.append(QualityMetric(
                timestamp=timestamp,
                source_type='audio',
                metric_name='snr',
                value=snr_value,
                level=self._classify_audio_snr(snr_value),
                status=f'{snr_value:.1f}dB',
                color=self._get_color_for_audio_snr(snr_value),
                details={'threshold': self.mobile_thresholds['audio_snr_min']}
            ))
        
        if 'confidence' in quality_data:
            conf_value = quality_data['confidence']
            metrics.append(QualityMetric(
                timestamp=timestamp,
                source_type='audio',
                metric_name='confidence',
                value=conf_value,
                level=self._classify_audio_confidence(conf_value),
                status=f'{conf_value:.1f}%',
                color=self._get_color_for_confidence(conf_value),
                details={'threshold': self.mobile_thresholds['audio_confidence_min']}
            ))
        
        return metrics
    
    def _extract_image_metrics(self, quality_data: Dict) -> List[QualityMetric]:
        """이미지 품질 데이터에서 메트릭 추출"""
        metrics = []
        timestamp = time.time()
        
        sharpness_data = quality_data.get('sharpness', {})
        if 'score' in sharpness_data:
            sharpness_value = sharpness_data['score']
            metrics.append(QualityMetric(
                timestamp=timestamp,
                source_type='image',
                metric_name='sharpness',
                value=sharpness_value,
                level=sharpness_data.get('level', 'unknown'),
                status=f'{sharpness_value:.1f}',
                color=self._get_color_for_sharpness(sharpness_value),
                details={'threshold': self.mobile_thresholds['image_sharpness_min']}
            ))
        
        return metrics
    
    def _get_current_quality_status(self) -> Dict:
        """현재 품질 상태 반환"""
        current_time = time.time()
        recent_window = 30.0
        
        status = {
            'audio': {'available': False},
            'image': {'available': False}, 
            'overall': {'score': 0.0, 'level': 'unknown'}
        }
        
        recent_metrics = [
            m for m in self.quality_history
            if current_time - m.timestamp <= recent_window
        ]
        
        for source_type in ['audio', 'image']:
            source_metrics = [m for m in recent_metrics if m.source_type == source_type]
            if source_metrics:
                latest_metrics = {}
                for metric in source_metrics:
                    if metric.metric_name not in latest_metrics or metric.timestamp > latest_metrics[metric.metric_name].timestamp:
                        latest_metrics[metric.metric_name] = metric
                
                if latest_metrics:
                    status[source_type] = {
                        'available': True,
                        'metrics': {name: asdict(metric) for name, metric in latest_metrics.items()},
                        'last_update': max(m.timestamp for m in latest_metrics.values())
                    }
        
        overall_score = self._calculate_overall_current_score(status)
        status['overall'] = {
            'score': overall_score,
            'level': self._classify_overall_score(overall_score),
            'color': self._get_color_for_overall_score(overall_score)
        }
        
        return status
    
    def _calculate_overall_current_score(self, status: Dict) -> float:
        """현재 전체 점수 계산"""
        scores = []
        
        for source_type in ['audio', 'image']:
            if status[source_type]['available']:
                source_score = self._calculate_source_score(status[source_type]['metrics'])
                scores.append(source_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_source_score(self, metrics: Dict) -> float:
        """소스별 점수 계산"""
        if not metrics:
            return 0.0
        
        scores = []
        for metric_name, metric_data in metrics.items():
            value = metric_data['value']
            
            if metric_name in ['snr']:
                score = min(1.0, max(0.0, value / 30.0))
            elif metric_name in ['confidence']:
                score = min(1.0, max(0.0, value / 100.0))
            elif metric_name in ['sharpness']:
                score = min(1.0, max(0.0, value / 300.0))
            else:
                score = 0.5
            
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def _get_recent_history_for_ui(self) -> List[Dict]:
        """UI 표시용 최근 이력 반환"""
        max_items = 20
        recent_history = self.quality_history[-max_items:] if self.quality_history else []
        return [asdict(metric) for metric in recent_history]
    
    def _get_system_status(self) -> Dict:
        """시스템 상태 반환"""
        device_info = self._collect_device_info()
        
        return {
            'timestamp': time.time(),
            'memory_usage': device_info.get('available_memory', 'unknown'),
            'battery_level': device_info.get('battery_level', 'unknown'),
        }
    
    def _get_active_alerts(self) -> List[Dict]:
        """활성 알림 목록 반환"""
        return []
    
    def _generate_session_report(self) -> Dict:
        """세션 리포트 생성"""
        if not self.current_session:
            return {'status': 'no_session'}
        
        session_duration = time.time() - self.current_session.start_time
        statistics = self._calculate_session_statistics()
        quality_summary = self._calculate_quality_summary()
        
        return {
            'session_id': self.current_session.session_id,
            'session_type': self.current_session.session_type,
            'location': self.current_session.location,
            'start_time': self.current_session.start_time,
            'end_time': time.time(),
            'duration': session_duration,
            'total_metrics': len(self.quality_history),
            'statistics': statistics,
            'quality_summary': quality_summary,
            'device_info': self.current_session.device_info
        }
    
    def _calculate_session_statistics(self) -> Dict:
        """세션 통계 계산"""
        if not self.quality_history:
            return {}
        
        stats = {
            'audio': {'count': 0, 'avg_score': 0},
            'image': {'count': 0, 'avg_score': 0}
        }
        
        for source_type in stats.keys():
            source_metrics = [m for m in self.quality_history if m.source_type == source_type]
            if source_metrics:
                stats[source_type]['count'] = len(source_metrics)
                stats[source_type]['avg_score'] = np.mean([m.value for m in source_metrics])
        
        return stats
    
    def _calculate_quality_summary(self) -> Dict:
        """품질 요약 계산"""
        if not self.quality_history:
            return {'overall_grade': 'unknown'}
        
        level_counts = {}
        for metric in self.quality_history:
            level = metric.level
            level_counts[level] = level_counts.get(level, 0) + 1
        
        if level_counts:
            dominant_level = max(level_counts, key=level_counts.get)
        else:
            dominant_level = 'unknown'
        
        return {
            'overall_grade': dominant_level,
            'level_distribution': level_counts,
            'recommendations': self._generate_session_recommendations()
        }
    
    def _generate_session_recommendations(self) -> List[str]:
        """세션 기반 권장사항 생성"""
        recommendations = []
        
        if self.quality_history:
            poor_metrics = [m for m in self.quality_history if m.level == 'poor']
            
            if len(poor_metrics) > len(self.quality_history) * 0.3:
                recommendations.append("전체적인 품질 개선이 필요합니다")
            
            audio_poor = [m for m in poor_metrics if m.source_type == 'audio']
            if audio_poor:
                recommendations.append("음성 품질 개선을 위해 조용한 환경에서 녹음하세요")
            
            image_poor = [m for m in poor_metrics if m.source_type == 'image']
            if image_poor:
                recommendations.append("이미지 품질 개선을 위해 안정적인 촬영과 적절한 조명을 사용하세요")
        
        return recommendations
    
    # === 분류 및 색상 헬퍼 함수들 ===
    
    def _classify_audio_snr(self, value: float) -> str:
        """음성 SNR 등급 분류"""
        if value >= 25: return 'excellent'
        elif value >= 20: return 'good'
        elif value >= 15: return 'fair'
        else: return 'poor'
    
    def _classify_audio_confidence(self, value: float) -> str:
        """음성 신뢰도 등급 분류"""
        if value >= 90: return 'excellent'
        elif value >= 80: return 'good'
        elif value >= 70: return 'fair'
        else: return 'poor'
    
    def _classify_overall_score(self, score: float) -> str:
        """전체 점수 등급 분류"""
        if score >= 0.9: return 'excellent'
        elif score >= 0.8: return 'good'
        elif score >= 0.7: return 'fair'
        else: return 'poor'
    
    def _get_color_for_audio_snr(self, value: float) -> str:
        """음성 SNR 색상"""
        if value >= 20: return '🟢'
        elif value >= 15: return '🟡'
        else: return '🔴'
    
    def _get_color_for_confidence(self, value: float) -> str:
        """신뢰도 색상"""
        if value >= 80: return '🟢'
        elif value >= 70: return '🟡'
        else: return '🔴'
    
    def _get_color_for_sharpness(self, value: float) -> str:
        """선명도 색상"""
        if value >= 100: return '🟢'
        elif value >= 80: return '🟡'
        else: return '🔴'
    
    def _get_color_for_overall_score(self, score: float) -> str:
        """전체 점수 색상"""
        if score >= 0.8: return '🟢'
        elif score >= 0.7: return '🟡'
        else: return '🔴'


# 사용 예제
if __name__ == "__main__":
    monitor = MobileQualityMonitor()
    
    print("📱 Mobile Quality Monitor v2.1 - 테스트 시작")
    print("=" * 50)
    
    def alert_callback(alert_data):
        print(f"🚨 알림: {alert_data.get('title', 'Unknown')} - {alert_data.get('message', '')}")
    
    monitor.add_alert_callback(alert_callback)
    
    session_id = monitor.start_monitoring_session(
        session_type='jewelry_show',
        location='홍콩 주얼리 전시회'
    )
    
    print(f"모니터링 세션 시작: {session_id}")
    print("모듈 로드 완료 ✅")
