#!/usr/bin/env python3
"""
스마트 알림 필터링 시스템 v2.6
288개 알림을 지능적으로 필터링하고 우선순위화
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import threading
from enum import Enum

class AlertLevel(Enum):
    """알림 레벨"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlertCategory(Enum):
    """알림 카테고리"""
    SYSTEM = "system"
    PERFORMANCE = "performance"
    SECURITY = "security"
    PROCESS = "process"
    USER = "user"
    NETWORK = "network"

@dataclass
class SmartAlert:
    """스마트 알림 데이터 클래스"""
    id: str
    timestamp: float
    level: AlertLevel
    category: AlertCategory
    message: str
    source: str
    metric_name: str
    metric_value: float
    threshold: float
    suggested_action: str
    raw_data: Dict[str, Any]
    
    # 스마트 필터링 필드
    priority_score: float = 0.0
    is_duplicate: bool = False
    duplicate_count: int = 1
    is_suppressed: bool = False
    suppression_reason: str = ""
    correlation_group: Optional[str] = None
    pattern_matched: bool = False

@dataclass
class AlertPattern:
    """알림 패턴 데이터 클래스"""
    pattern_id: str
    name: str
    conditions: List[Dict[str, Any]]
    action: str  # 'suppress', 'escalate', 'group', 'modify'
    priority_adjustment: float
    description: str
    match_count: int = 0
    last_matched: Optional[str] = None

@dataclass
class FilteringStats:
    """필터링 통계 데이터 클래스"""
    total_alerts: int
    filtered_alerts: int
    suppressed_alerts: int
    grouped_alerts: int
    escalated_alerts: int
    filtering_efficiency: float
    processing_time_ms: float
    timestamp: str

class SmartAlertFilter:
    """스마트 알림 필터링 시스템"""
    
    def __init__(self, max_alerts_per_minute: int = 50):
        self.max_alerts_per_minute = max_alerts_per_minute
        self.logger = self._setup_logging()
        
        # 알림 저장소
        self.active_alerts = deque(maxlen=1000)
        self.suppressed_alerts = deque(maxlen=5000)
        self.alert_history = deque(maxlen=10000)
        
        # 중복 감지를 위한 해시 저장소
        self.duplicate_detection_window = 300  # 5분
        self.recent_alert_hashes = {}
        
        # 패턴 매칭 시스템
        self.alert_patterns = self._initialize_alert_patterns()
        
        # 상관관계 분석
        self.correlation_groups = defaultdict(list)
        self.correlation_window = 600  # 10분
        
        # 우선순위 규칙
        self.priority_rules = self._initialize_priority_rules()
        
        # 적응형 임계값
        self.adaptive_thresholds = {}
        self.baseline_metrics = {}
        
        # 스레드 안전성
        self.lock = threading.Lock()
        
        # 통계 추적
        self.filtering_stats = FilteringStats(
            total_alerts=0,
            filtered_alerts=0,
            suppressed_alerts=0,
            grouped_alerts=0,
            escalated_alerts=0,
            filtering_efficiency=0.0,
            processing_time_ms=0.0,
            timestamp=datetime.now().isoformat()
        )
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.SmartAlertFilter')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_alert_patterns(self) -> List[AlertPattern]:
        """알림 패턴 초기화"""
        return [
            AlertPattern(
                pattern_id="cpu_spike_pattern",
                name="CPU 사용률 급증 패턴",
                conditions=[
                    {"metric_name": "cpu_usage", "operator": ">", "value": 90},
                    {"category": "performance"},
                    {"duration": ">", "value": 60}  # 1분 이상 지속
                ],
                action="group",
                priority_adjustment=0.2,
                description="CPU 사용률이 90% 이상으로 1분 이상 지속되는 경우"
            ),
            AlertPattern(
                pattern_id="memory_leak_pattern",
                name="메모리 누수 패턴",
                conditions=[
                    {"metric_name": "memory_usage", "operator": ">", "value": 85},
                    {"trend": "increasing", "duration": 300}  # 5분간 증가 추세
                ],
                action="escalate",
                priority_adjustment=0.5,
                description="메모리 사용률이 5분간 지속적으로 증가하는 패턴"
            ),
            AlertPattern(
                pattern_id="disk_io_storm",
                name="디스크 I/O 폭증",
                conditions=[
                    {"metric_name": "disk_io_read", "operator": ">", "value": 100},
                    {"metric_name": "disk_io_write", "operator": ">", "value": 100},
                    {"frequency": ">", "value": 10}  # 10개 이상 동시 발생
                ],
                action="group",
                priority_adjustment=0.3,
                description="디스크 I/O가 동시에 폭증하는 패턴"
            ),
            AlertPattern(
                pattern_id="noise_pattern",
                name="노이즈 알림 패턴",
                conditions=[
                    {"level": "info"},
                    {"frequency": ">", "value": 50},  # 분당 50개 이상
                    {"metric_variance": "<", "value": 0.1}  # 변화량이 작음
                ],
                action="suppress",
                priority_adjustment=-0.8,
                description="정보성 알림이 과도하게 발생하는 노이즈 패턴"
            ),
            AlertPattern(
                pattern_id="ollama_model_switching",
                name="Ollama 모델 전환 패턴",
                conditions=[
                    {"source": "ollama"},
                    {"message_contains": ["model", "switching", "loading"]},
                    {"level": "info"}
                ],
                action="suppress",
                priority_adjustment=-0.5,
                description="Ollama 모델 전환 시 발생하는 일반적인 알림"
            ),
            AlertPattern(
                pattern_id="file_processing_batch",
                name="파일 처리 배치 패턴",
                conditions=[
                    {"category": "process"},
                    {"message_contains": ["file", "processing", "batch"]},
                    {"frequency": ">", "value": 20}
                ],
                action="group",
                priority_adjustment=0.1,
                description="배치 파일 처리 중 발생하는 알림들을 그룹화"
            )
        ]
    
    def _initialize_priority_rules(self) -> List[Dict[str, Any]]:
        """우선순위 규칙 초기화"""
        return [
            {
                "name": "critical_system_failure",
                "conditions": [
                    {"level": AlertLevel.CRITICAL},
                    {"category": AlertCategory.SYSTEM}
                ],
                "priority_boost": 1.0,
                "max_suppression_time": 0  # 절대 억제하지 않음
            },
            {
                "name": "security_breach",
                "conditions": [
                    {"category": AlertCategory.SECURITY},
                    {"level": [AlertLevel.CRITICAL, AlertLevel.HIGH]}
                ],
                "priority_boost": 0.8,
                "max_suppression_time": 60
            },
            {
                "name": "performance_degradation",
                "conditions": [
                    {"category": AlertCategory.PERFORMANCE},
                    {"metric_value": ">", "threshold_ratio": 2.0}  # 임계값의 2배 이상
                ],
                "priority_boost": 0.6,
                "max_suppression_time": 300
            },
            {
                "name": "user_impact",
                "conditions": [
                    {"category": AlertCategory.USER},
                    {"business_hours": True}
                ],
                "priority_boost": 0.4,
                "max_suppression_time": 180
            },
            {
                "name": "informational_noise",
                "conditions": [
                    {"level": AlertLevel.INFO},
                    {"frequency": ">", "value": 30}
                ],
                "priority_boost": -0.7,
                "max_suppression_time": 3600  # 1시간까지 억제 가능
            }
        ]
    
    def process_alert(self, alert_data: Dict[str, Any]) -> Optional[SmartAlert]:
        """알림 처리 및 필터링"""
        start_time = time.time()
        
        with self.lock:
            # SmartAlert 객체 생성
            alert = self._create_smart_alert(alert_data)
            
            # 통계 업데이트
            self.filtering_stats.total_alerts += 1
            
            # 1단계: 중복 감지
            if self._is_duplicate(alert):
                alert.is_duplicate = True
                self._update_duplicate_count(alert)
                self.logger.debug(f"중복 알림 감지: {alert.id}")
                return None
            
            # 2단계: 패턴 매칭
            matched_pattern = self._match_patterns(alert)
            if matched_pattern:
                alert.pattern_matched = True
                alert = self._apply_pattern_action(alert, matched_pattern)
            
            # 3단계: 우선순위 계산
            alert.priority_score = self._calculate_priority_score(alert)
            
            # 4단계: 상관관계 분석
            correlation_group = self._analyze_correlations(alert)
            if correlation_group:
                alert.correlation_group = correlation_group
            
            # 5단계: 적응형 억제 결정
            if self._should_suppress(alert):
                alert.is_suppressed = True
                self.suppressed_alerts.append(alert)
                self.filtering_stats.suppressed_alerts += 1
                self.logger.debug(f"알림 억제: {alert.id} - {alert.suppression_reason}")
                return None
            
            # 6단계: 활성 알림에 추가
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            self.filtering_stats.filtered_alerts += 1
            
            # 7단계: 적응형 임계값 업데이트
            self._update_adaptive_thresholds(alert)
            
            processing_time = (time.time() - start_time) * 1000
            self.filtering_stats.processing_time_ms = processing_time
            
            self.logger.info(f"✅ 알림 처리 완료: {alert.level.value} - {alert.message[:50]}...")
            
            return alert
    
    def _create_smart_alert(self, alert_data: Dict[str, Any]) -> SmartAlert:
        """SmartAlert 객체 생성"""
        alert_id = f"{alert_data.get('source', 'unknown')}_{int(time.time() * 1000)}"
        
        # 레벨 및 카테고리 파싱
        level = AlertLevel(alert_data.get('level', 'info'))
        category = AlertCategory(alert_data.get('category', 'system'))
        
        return SmartAlert(
            id=alert_id,
            timestamp=time.time(),
            level=level,
            category=category,
            message=alert_data.get('message', ''),
            source=alert_data.get('source', 'unknown'),
            metric_name=alert_data.get('metric', ''),
            metric_value=float(alert_data.get('value', 0)),
            threshold=float(alert_data.get('threshold', 0)),
            suggested_action=alert_data.get('action', ''),
            raw_data=alert_data
        )
    
    def _is_duplicate(self, alert: SmartAlert) -> bool:
        """중복 알림 감지"""
        # 알림 해시 생성 (메시지, 소스, 메트릭 기반)
        alert_hash = hash(f"{alert.source}_{alert.metric_name}_{alert.message[:100]}")
        
        current_time = time.time()
        
        # 최근 윈도우 내 동일한 해시 확인
        if alert_hash in self.recent_alert_hashes:
            last_time = self.recent_alert_hashes[alert_hash]['timestamp']
            if current_time - last_time < self.duplicate_detection_window:
                self.recent_alert_hashes[alert_hash]['count'] += 1
                return True
        
        # 새로운 알림으로 등록
        self.recent_alert_hashes[alert_hash] = {
            'timestamp': current_time,
            'count': 1
        }
        
        # 오래된 해시 정리
        self._cleanup_old_hashes(current_time)
        
        return False
    
    def _update_duplicate_count(self, alert: SmartAlert) -> None:
        """중복 카운트 업데이트"""
        alert_hash = hash(f"{alert.source}_{alert.metric_name}_{alert.message[:100]}")
        if alert_hash in self.recent_alert_hashes:
            alert.duplicate_count = self.recent_alert_hashes[alert_hash]['count']
    
    def _cleanup_old_hashes(self, current_time: float) -> None:
        """오래된 해시 정리"""
        cutoff_time = current_time - self.duplicate_detection_window
        old_hashes = [
            h for h, data in self.recent_alert_hashes.items() 
            if data['timestamp'] < cutoff_time
        ]
        for old_hash in old_hashes:
            del self.recent_alert_hashes[old_hash]
    
    def _match_patterns(self, alert: SmartAlert) -> Optional[AlertPattern]:
        """패턴 매칭"""
        for pattern in self.alert_patterns:
            if self._matches_pattern_conditions(alert, pattern.conditions):
                pattern.match_count += 1
                pattern.last_matched = datetime.now().isoformat()
                return pattern
        return None
    
    def _matches_pattern_conditions(self, alert: SmartAlert, conditions: List[Dict[str, Any]]) -> bool:
        """패턴 조건 매칭"""
        for condition in conditions:
            if not self._matches_single_condition(alert, condition):
                return False
        return True
    
    def _matches_single_condition(self, alert: SmartAlert, condition: Dict[str, Any]) -> bool:
        """단일 조건 매칭"""
        if 'metric_name' in condition:
            if alert.metric_name != condition['metric_name']:
                return False
        
        if 'category' in condition:
            if alert.category.value != condition['category']:
                return False
        
        if 'level' in condition:
            if isinstance(condition['level'], list):
                if alert.level not in condition['level']:
                    return False
            else:
                if alert.level.value != condition['level']:
                    return False
        
        if 'source' in condition:
            if alert.source != condition['source']:
                return False
        
        if 'message_contains' in condition:
            message_lower = alert.message.lower()
            for keyword in condition['message_contains']:
                if keyword.lower() not in message_lower:
                    return False
        
        if 'operator' in condition and 'value' in condition:
            operator = condition['operator']
            value = condition['value']
            
            if operator == '>' and alert.metric_value <= value:
                return False
            elif operator == '<' and alert.metric_value >= value:
                return False
            elif operator == '==' and alert.metric_value != value:
                return False
        
        return True
    
    def _apply_pattern_action(self, alert: SmartAlert, pattern: AlertPattern) -> SmartAlert:
        """패턴 액션 적용"""
        if pattern.action == 'suppress':
            alert.is_suppressed = True
            alert.suppression_reason = f"패턴 매칭: {pattern.name}"
        
        elif pattern.action == 'escalate':
            # 레벨 상승
            if alert.level == AlertLevel.LOW:
                alert.level = AlertLevel.MEDIUM
            elif alert.level == AlertLevel.MEDIUM:
                alert.level = AlertLevel.HIGH
            elif alert.level == AlertLevel.HIGH:
                alert.level = AlertLevel.CRITICAL
        
        elif pattern.action == 'group':
            alert.correlation_group = f"pattern_{pattern.pattern_id}"
        
        elif pattern.action == 'modify':
            alert.message = f"[패턴 감지: {pattern.name}] {alert.message}"
        
        # 우선순위 조정
        alert.priority_score += pattern.priority_adjustment
        
        return alert
    
    def _calculate_priority_score(self, alert: SmartAlert) -> float:
        """우선순위 점수 계산"""
        base_score = 0.5
        
        # 레벨 기반 점수
        level_scores = {
            AlertLevel.CRITICAL: 1.0,
            AlertLevel.HIGH: 0.8,
            AlertLevel.MEDIUM: 0.6,
            AlertLevel.LOW: 0.4,
            AlertLevel.INFO: 0.2
        }
        base_score += level_scores.get(alert.level, 0.5)
        
        # 카테고리 기반 점수
        category_scores = {
            AlertCategory.SECURITY: 0.3,
            AlertCategory.SYSTEM: 0.25,
            AlertCategory.PERFORMANCE: 0.2,
            AlertCategory.PROCESS: 0.15,
            AlertCategory.USER: 0.1,
            AlertCategory.NETWORK: 0.1
        }
        base_score += category_scores.get(alert.category, 0.1)
        
        # 임계값 초과 정도
        if alert.threshold > 0:
            threshold_ratio = alert.metric_value / alert.threshold
            if threshold_ratio > 1:
                base_score += min(0.3, (threshold_ratio - 1) * 0.1)
        
        # 우선순위 규칙 적용
        for rule in self.priority_rules:
            if self._matches_priority_rule(alert, rule):
                base_score += rule['priority_boost']
        
        return max(0.0, min(2.0, base_score))  # 0-2 범위로 제한
    
    def _matches_priority_rule(self, alert: SmartAlert, rule: Dict[str, Any]) -> bool:
        """우선순위 규칙 매칭"""
        conditions = rule['conditions']
        
        for condition in conditions:
            if 'level' in condition:
                if isinstance(condition['level'], list):
                    if alert.level not in condition['level']:
                        return False
                else:
                    if alert.level != condition['level']:
                        return False
            
            if 'category' in condition:
                if alert.category != condition['category']:
                    return False
            
            if 'threshold_ratio' in condition:
                if alert.threshold > 0:
                    ratio = alert.metric_value / alert.threshold
                    operator = condition.get('operator', '>')
                    threshold = condition['threshold_ratio']
                    
                    if operator == '>' and ratio <= threshold:
                        return False
                    elif operator == '<' and ratio >= threshold:
                        return False
        
        return True
    
    def _analyze_correlations(self, alert: SmartAlert) -> Optional[str]:
        """상관관계 분석"""
        current_time = time.time()
        cutoff_time = current_time - self.correlation_window
        
        # 최근 알림들과의 상관관계 분석
        recent_alerts = [
            a for a in self.alert_history 
            if a.timestamp > cutoff_time and a.category == alert.category
        ]
        
        if len(recent_alerts) >= 3:  # 최소 3개 이상의 관련 알림
            group_id = f"corr_{alert.category.value}_{int(current_time)}"
            
            # 기존 그룹에 추가
            for existing_alert in recent_alerts:
                if existing_alert.correlation_group:
                    return existing_alert.correlation_group
            
            return group_id
        
        return None
    
    def _should_suppress(self, alert: SmartAlert) -> bool:
        """알림 억제 여부 결정"""
        # 이미 억제 표시된 경우
        if alert.is_suppressed:
            return True
        
        # 크리티컬 시스템 알림은 절대 억제하지 않음
        if alert.level == AlertLevel.CRITICAL and alert.category == AlertCategory.SYSTEM:
            return False
        
        # 우선순위가 너무 낮은 경우
        if alert.priority_score < 0.3:
            alert.suppression_reason = "낮은 우선순위"
            return True
        
        # 분당 알림 제한 확인
        current_time = time.time()
        recent_active_alerts = [
            a for a in self.active_alerts 
            if current_time - a.timestamp < 60
        ]
        
        if len(recent_active_alerts) >= self.max_alerts_per_minute:
            # 우선순위가 낮은 알림부터 억제
            if alert.priority_score < np.percentile([a.priority_score for a in recent_active_alerts], 75):
                alert.suppression_reason = "분당 알림 제한 초과"
                return True
        
        return False
    
    def _update_adaptive_thresholds(self, alert: SmartAlert) -> None:
        """적응형 임계값 업데이트"""
        metric_key = f"{alert.source}_{alert.metric_name}"
        
        if metric_key not in self.adaptive_thresholds:
            self.adaptive_thresholds[metric_key] = {
                'values': deque(maxlen=100),
                'threshold': alert.threshold,
                'last_updated': time.time()
            }
        
        threshold_data = self.adaptive_thresholds[metric_key]
        threshold_data['values'].append(alert.metric_value)
        
        # 충분한 샘플이 있으면 임계값 조정
        if len(threshold_data['values']) >= 20:
            values = list(threshold_data['values'])
            # 95 퍼센타일을 새로운 임계값으로 사용
            new_threshold = np.percentile(values, 95)
            
            # 점진적 조정 (급격한 변화 방지)
            alpha = 0.1
            threshold_data['threshold'] = (
                (1 - alpha) * threshold_data['threshold'] + 
                alpha * new_threshold
            )
            threshold_data['last_updated'] = time.time()
    
    def get_active_alerts(self, max_count: int = 20) -> List[SmartAlert]:
        """활성 알림 목록 반환 (우선순위 순)"""
        with self.lock:
            sorted_alerts = sorted(
                self.active_alerts, 
                key=lambda x: x.priority_score, 
                reverse=True
            )
            return list(sorted_alerts)[:max_count]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """알림 요약 정보"""
        with self.lock:
            current_time = time.time()
            
            # 최근 1시간 알림 통계
            recent_alerts = [
                a for a in self.alert_history 
                if current_time - a.timestamp < 3600
            ]
            
            level_counts = defaultdict(int)
            category_counts = defaultdict(int)
            
            for alert in recent_alerts:
                level_counts[alert.level.value] += 1
                category_counts[alert.category.value] += 1
            
            # 필터링 효율성 계산
            if self.filtering_stats.total_alerts > 0:
                self.filtering_stats.filtering_efficiency = (
                    self.filtering_stats.suppressed_alerts / 
                    self.filtering_stats.total_alerts * 100
                )
            
            return {
                'active_alerts_count': len(self.active_alerts),
                'suppressed_alerts_count': len(self.suppressed_alerts),
                'total_processed': self.filtering_stats.total_alerts,
                'filtering_efficiency': self.filtering_stats.filtering_efficiency,
                'recent_hour_stats': {
                    'total': len(recent_alerts),
                    'by_level': dict(level_counts),
                    'by_category': dict(category_counts)
                },
                'top_patterns': [
                    {
                        'name': p.name,
                        'matches': p.match_count,
                        'last_matched': p.last_matched
                    }
                    for p in sorted(self.alert_patterns, key=lambda x: x.match_count, reverse=True)[:5]
                ]
            }
    
    def export_filtering_report(self, output_path: str) -> None:
        """필터링 보고서 내보내기"""
        with self.lock:
            report_data = {
                'export_timestamp': datetime.now().isoformat(),
                'filtering_stats': asdict(self.filtering_stats),
                'alert_patterns': [asdict(p) for p in self.alert_patterns],
                'priority_rules': self.priority_rules,
                'adaptive_thresholds': {
                    k: {**v, 'values': list(v['values'])} 
                    for k, v in self.adaptive_thresholds.items()
                },
                'summary': self.get_alert_summary()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"📊 필터링 보고서 저장됨: {output_path}")

# 전역 스마트 필터 인스턴스
_global_smart_filter = None

def get_global_smart_filter() -> SmartAlertFilter:
    """전역 스마트 필터 인스턴스 반환"""
    global _global_smart_filter
    if _global_smart_filter is None:
        _global_smart_filter = SmartAlertFilter()
    return _global_smart_filter

# 사용 예시
if __name__ == "__main__":
    filter_system = SmartAlertFilter()
    
    # 테스트 알림들
    test_alerts = [
        {
            'level': 'critical',
            'category': 'system',
            'message': 'CPU usage extremely high',
            'source': 'system_monitor',
            'metric': 'cpu_usage',
            'value': 95.0,
            'threshold': 85.0,
            'action': 'Check running processes'
        },
        {
            'level': 'info',
            'category': 'process',
            'message': 'File processing started',
            'source': 'file_processor',
            'metric': 'files_processing',
            'value': 1.0,
            'threshold': 10.0,
            'action': 'Monitor progress'
        }
    ]
    
    print("🔍 스마트 알림 필터링 테스트:")
    
    for alert_data in test_alerts:
        result = filter_system.process_alert(alert_data)
        if result:
            print(f"✅ 처리됨: {result.level.value} - {result.message}")
        else:
            print(f"🚫 필터링됨: {alert_data['message']}")
    
    # 요약 정보
    summary = filter_system.get_alert_summary()
    print(f"\n📊 필터링 요약:")
    print(f"활성 알림: {summary['active_alerts_count']}개")
    print(f"억제된 알림: {summary['suppressed_alerts_count']}개")
    print(f"필터링 효율성: {summary['filtering_efficiency']:.1f}%")