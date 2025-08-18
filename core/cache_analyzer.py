#!/usr/bin/env python3
"""
캐시 분석 및 최적화 시스템 v2.6
캐시 성능 분석, 패턴 감지, 자동 최적화 제안
"""

import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import threading
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CacheAccessPattern:
    """캐시 접근 패턴"""
    key_pattern: str
    access_frequency: float
    temporal_pattern: str  # 'peak', 'off-peak', 'steady'
    size_category: str  # 'small', 'medium', 'large'
    ttl_effectiveness: float
    hit_rate: float
    avg_response_time_ms: float
    category: str  # 'audio', 'image', 'model', 'config'

@dataclass
class CacheOptimizationSuggestion:
    """캐시 최적화 제안"""
    suggestion_type: str
    priority: str  # 'high', 'medium', 'low'
    description: str
    expected_improvement: float
    implementation_effort: str  # 'easy', 'medium', 'hard'
    affected_keys: List[str]
    parameters: Dict[str, Any]

@dataclass
class CachePerformanceReport:
    """캐시 성능 보고서"""
    analysis_timestamp: str
    overall_hit_rate: float
    avg_response_time_ms: float
    memory_efficiency: float
    cache_pollution_score: float
    optimization_opportunities: int
    patterns_detected: List[CacheAccessPattern]
    suggestions: List[CacheOptimizationSuggestion]
    performance_trends: Dict[str, List[float]]

class CacheAnalyzer:
    """캐시 분석기"""
    
    def __init__(self, cache_manager, analysis_window_hours: int = 24):
        self.cache_manager = cache_manager
        self.analysis_window_hours = analysis_window_hours
        self.logger = self._setup_logging()
        
        # 분석 데이터 저장소
        self.access_logs = deque(maxlen=10000)
        self.performance_metrics = deque(maxlen=1000)
        self.pattern_history = {}
        
        # 분석 설정
        self.key_clustering_threshold = 0.8
        self.pattern_detection_min_samples = 10
        self.optimization_confidence_threshold = 0.7
        
        # 성능 기준선
        self.baseline_metrics = {
            'target_hit_rate': 0.8,
            'target_response_time_ms': 50.0,
            'target_memory_efficiency': 0.7,
            'acceptable_pollution_score': 0.3
        }
        
        # 패턴 분류기
        self.pattern_classifier = None
        self.is_trained = False
        
        # 스레드 안전성
        self.lock = threading.RLock()
        
        # 분석 캐시 (분석 결과 캐싱)
        self.analysis_cache = {}
        self.last_analysis_time = 0
        self.analysis_cache_ttl = 300  # 5분
        
        self.logger.info("🔍 캐시 분석기 초기화 완료")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.CacheAnalyzer')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def log_access(self, key: str, hit: bool, response_time_ms: float, 
                   cache_level: str, operation: str, data_size: int = 0) -> None:
        """접근 로그 기록"""
        with self.lock:
            access_log = {
                'timestamp': time.time(),
                'key': key,
                'hit': hit,
                'response_time_ms': response_time_ms,
                'cache_level': cache_level,
                'operation': operation,  # 'get', 'set', 'delete'
                'data_size': data_size,
                'hour_of_day': datetime.now().hour,
                'day_of_week': datetime.now().weekday()
            }
            
            self.access_logs.append(access_log)
            
            # 실시간 메트릭 업데이트
            self._update_realtime_metrics(access_log)
    
    def _update_realtime_metrics(self, access_log: Dict[str, Any]) -> None:
        """실시간 메트릭 업데이트"""
        current_time = time.time()
        
        # 최근 1분간 메트릭 계산
        recent_logs = [
            log for log in self.access_logs 
            if current_time - log['timestamp'] <= 60
        ]
        
        if recent_logs:
            hit_rate = sum(1 for log in recent_logs if log['hit']) / len(recent_logs)
            avg_response_time = np.mean([log['response_time_ms'] for log in recent_logs])
            
            metric = {
                'timestamp': current_time,
                'hit_rate': hit_rate,
                'avg_response_time_ms': avg_response_time,
                'total_requests': len(recent_logs),
                'cache_size_mb': self.cache_manager.get_size()
            }
            
            self.performance_metrics.append(metric)
    
    def analyze_cache_performance(self, force_refresh: bool = False) -> CachePerformanceReport:
        """캐시 성능 분석"""
        current_time = time.time()
        
        # 캐시된 분석 결과 확인
        if not force_refresh and \
           current_time - self.last_analysis_time < self.analysis_cache_ttl and \
           'performance_report' in self.analysis_cache:
            return self.analysis_cache['performance_report']
        
        with self.lock:
            try:
                self.logger.info("🔍 캐시 성능 분석 시작")
                
                # 분석 데이터 준비
                analysis_data = self._prepare_analysis_data()
                
                if not analysis_data:
                    return self._create_empty_report()
                
                # 1. 전체 성능 지표 계산
                overall_metrics = self._calculate_overall_metrics(analysis_data)
                
                # 2. 접근 패턴 분석
                patterns = self._analyze_access_patterns(analysis_data)
                
                # 3. 성능 추세 분석
                trends = self._analyze_performance_trends()
                
                # 4. 최적화 기회 식별
                suggestions = self._generate_optimization_suggestions(
                    overall_metrics, patterns, trends
                )
                
                # 5. 보고서 생성
                report = CachePerformanceReport(
                    analysis_timestamp=datetime.now().isoformat(),
                    overall_hit_rate=overall_metrics['hit_rate'],
                    avg_response_time_ms=overall_metrics['avg_response_time_ms'],
                    memory_efficiency=overall_metrics['memory_efficiency'],
                    cache_pollution_score=overall_metrics['pollution_score'],
                    optimization_opportunities=len(suggestions),
                    patterns_detected=patterns,
                    suggestions=suggestions,
                    performance_trends=trends
                )
                
                # 결과 캐싱
                self.analysis_cache['performance_report'] = report
                self.last_analysis_time = current_time
                
                self.logger.info(f"✅ 캐시 성능 분석 완료: 히트율 {report.overall_hit_rate:.1%}")
                return report
                
            except Exception as e:
                self.logger.error(f"❌ 캐시 성능 분석 실패: {e}")
                return self._create_empty_report()
    
    def _prepare_analysis_data(self) -> List[Dict[str, Any]]:
        """분석 데이터 준비"""
        cutoff_time = time.time() - (self.analysis_window_hours * 3600)
        
        # 분석 윈도우 내 데이터 필터링
        analysis_data = [
            log for log in self.access_logs 
            if log['timestamp'] > cutoff_time
        ]
        
        return analysis_data
    
    def _calculate_overall_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """전체 성능 지표 계산"""
        if not data:
            return {
                'hit_rate': 0.0,
                'avg_response_time_ms': 0.0,
                'memory_efficiency': 0.0,
                'pollution_score': 0.0
            }
        
        # 히트율
        hit_rate = sum(1 for log in data if log['hit']) / len(data)
        
        # 평균 응답 시간
        avg_response_time = np.mean([log['response_time_ms'] for log in data])
        
        # 메모리 효율성 (히트율 대비 메모리 사용량)
        cache_size_mb = self.cache_manager.get_size()
        memory_efficiency = hit_rate / max(cache_size_mb, 1.0)
        
        # 캐시 오염 점수 (미사용 데이터 비율)
        pollution_score = self._calculate_cache_pollution(data)
        
        return {
            'hit_rate': hit_rate,
            'avg_response_time_ms': avg_response_time,
            'memory_efficiency': memory_efficiency,
            'pollution_score': pollution_score
        }
    
    def _calculate_cache_pollution(self, data: List[Dict[str, Any]]) -> float:
        """캐시 오염 점수 계산"""
        try:
            # 키별 마지막 접근 시간 계산
            key_last_access = {}
            for log in data:
                key = log['key']
                if key not in key_last_access or log['timestamp'] > key_last_access[key]:
                    key_last_access[key] = log['timestamp']
            
            # 현재 캐시의 모든 키
            cached_keys = set(self.cache_manager.keys())
            
            # 최근 접근되지 않은 키의 비율
            current_time = time.time()
            stale_threshold = 3600  # 1시간
            
            stale_keys = 0
            for key in cached_keys:
                if key not in key_last_access or \
                   (current_time - key_last_access[key]) > stale_threshold:
                    stale_keys += 1
            
            pollution_score = stale_keys / max(len(cached_keys), 1)
            return min(1.0, pollution_score)
            
        except Exception as e:
            self.logger.error(f"❌ 캐시 오염 계산 실패: {e}")
            return 0.0
    
    def _analyze_access_patterns(self, data: List[Dict[str, Any]]) -> List[CacheAccessPattern]:
        """접근 패턴 분석"""
        patterns = []
        
        try:
            # 키별 통계 계산
            key_stats = self._calculate_key_statistics(data)
            
            # 패턴 감지
            for key_pattern, stats in key_stats.items():
                if stats['access_count'] >= self.pattern_detection_min_samples:
                    
                    # 시간적 패턴 분석
                    temporal_pattern = self._analyze_temporal_pattern(
                        key_pattern, data
                    )
                    
                    # 크기 카테고리 결정
                    size_category = self._categorize_size(stats['avg_size'])
                    
                    # 카테고리 분류
                    category = self._classify_key_category(key_pattern)
                    
                    pattern = CacheAccessPattern(
                        key_pattern=key_pattern,
                        access_frequency=stats['access_frequency'],
                        temporal_pattern=temporal_pattern,
                        size_category=size_category,
                        ttl_effectiveness=stats['ttl_effectiveness'],
                        hit_rate=stats['hit_rate'],
                        avg_response_time_ms=stats['avg_response_time'],
                        category=category
                    )
                    
                    patterns.append(pattern)
            
            self.logger.info(f"🔍 감지된 패턴: {len(patterns)}개")
            return patterns
            
        except Exception as e:
            self.logger.error(f"❌ 접근 패턴 분석 실패: {e}")
            return []
    
    def _calculate_key_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """키별 통계 계산"""
        key_stats = defaultdict(lambda: {
            'access_count': 0,
            'hits': 0,
            'total_response_time': 0.0,
            'total_size': 0,
            'access_times': [],
            'ttl_hits': 0,
            'ttl_total': 0
        })
        
        # 키 패턴 그룹핑 (유사한 키들을 하나의 패턴으로)
        key_patterns = self._group_similar_keys(data)
        
        for log in data:
            key = log['key']
            pattern = self._find_key_pattern(key, key_patterns)
            
            stats = key_stats[pattern]
            stats['access_count'] += 1
            stats['total_response_time'] += log['response_time_ms']
            stats['total_size'] += log['data_size']
            stats['access_times'].append(log['timestamp'])
            
            if log['hit']:
                stats['hits'] += 1
                
                # TTL 효과성 (캐시에서 찾은 경우)
                if log.get('ttl_remaining', 0) > 0:
                    stats['ttl_hits'] += 1
                stats['ttl_total'] += 1
        
        # 통계 계산
        processed_stats = {}
        for pattern, stats in key_stats.items():
            if stats['access_count'] > 0:
                processed_stats[pattern] = {
                    'access_count': stats['access_count'],
                    'access_frequency': stats['access_count'] / self.analysis_window_hours,
                    'hit_rate': stats['hits'] / stats['access_count'],
                    'avg_response_time': stats['total_response_time'] / stats['access_count'],
                    'avg_size': stats['total_size'] / stats['access_count'] if stats['total_size'] > 0 else 0,
                    'ttl_effectiveness': stats['ttl_hits'] / max(stats['ttl_total'], 1),
                    'access_times': stats['access_times']
                }
        
        return processed_stats
    
    def _group_similar_keys(self, data: List[Dict[str, Any]]) -> Dict[str, str]:
        """유사한 키들을 패턴으로 그룹핑"""
        keys = list(set(log['key'] for log in data))
        key_patterns = {}
        
        # 간단한 패턴 매칭 (정규식 기반)
        patterns = {
            'audio_analysis': r'audio_.*_\d+',
            'image_ocr': r'image_.*_ocr',
            'model_inference': r'model_.*_inference',
            'user_config': r'user_.*_config',
            'temp_file': r'temp_.*',
            'analysis_result': r'.*_analysis_.*'
        }
        
        import re
        for key in keys:
            matched_pattern = 'other'
            for pattern_name, pattern_regex in patterns.items():
                if re.match(pattern_regex, key):
                    matched_pattern = pattern_name
                    break
            key_patterns[key] = matched_pattern
        
        return key_patterns
    
    def _find_key_pattern(self, key: str, key_patterns: Dict[str, str]) -> str:
        """키의 패턴 찾기"""
        return key_patterns.get(key, 'other')
    
    def _analyze_temporal_pattern(self, key_pattern: str, data: List[Dict[str, Any]]) -> str:
        """시간적 패턴 분석"""
        try:
            # 해당 패턴의 접근 시간들
            pattern_logs = [
                log for log in data 
                if self._find_key_pattern(log['key'], self._group_similar_keys(data)) == key_pattern
            ]
            
            if len(pattern_logs) < 10:
                return 'insufficient_data'
            
            # 시간대별 접근 빈도
            hourly_access = defaultdict(int)
            for log in pattern_logs:
                hour = datetime.fromtimestamp(log['timestamp']).hour
                hourly_access[hour] += 1
            
            # 피크 시간 식별
            max_access = max(hourly_access.values())
            avg_access = np.mean(list(hourly_access.values()))
            
            # 패턴 분류
            if max_access > avg_access * 2:
                return 'peak'
            elif max_access < avg_access * 1.2:
                return 'steady'
            else:
                return 'variable'
                
        except Exception as e:
            self.logger.error(f"❌ 시간적 패턴 분석 실패: {e}")
            return 'unknown'
    
    def _categorize_size(self, avg_size: float) -> str:
        """크기 카테고리 분류"""
        if avg_size < 1024:  # 1KB 미만
            return 'small'
        elif avg_size < 1024 * 1024:  # 1MB 미만
            return 'medium'
        else:
            return 'large'
    
    def _classify_key_category(self, key_pattern: str) -> str:
        """키 카테고리 분류"""
        category_map = {
            'audio_analysis': 'audio',
            'image_ocr': 'image',
            'model_inference': 'model',
            'user_config': 'config',
            'temp_file': 'temp',
            'analysis_result': 'analysis'
        }
        return category_map.get(key_pattern, 'other')
    
    def _analyze_performance_trends(self) -> Dict[str, List[float]]:
        """성능 추세 분석"""
        try:
            trends = {}
            
            if len(self.performance_metrics) < 10:
                return trends
            
            # 시간 순으로 정렬된 메트릭
            sorted_metrics = sorted(self.performance_metrics, key=lambda x: x['timestamp'])
            
            # 추세 데이터 추출
            trends['hit_rate'] = [m['hit_rate'] for m in sorted_metrics]
            trends['response_time'] = [m['avg_response_time_ms'] for m in sorted_metrics]
            trends['cache_size'] = [m['cache_size_mb'] for m in sorted_metrics]
            trends['request_rate'] = [m['total_requests'] for m in sorted_metrics]
            
            # 이동 평균 계산 (스무딩)
            window_size = min(10, len(sorted_metrics) // 2)
            if window_size > 1:
                for metric_name, values in trends.items():
                    smoothed = []
                    for i in range(len(values)):
                        start_idx = max(0, i - window_size // 2)
                        end_idx = min(len(values), i + window_size // 2 + 1)
                        smoothed.append(np.mean(values[start_idx:end_idx]))
                    trends[f'{metric_name}_smoothed'] = smoothed
            
            return trends
            
        except Exception as e:
            self.logger.error(f"❌ 성능 추세 분석 실패: {e}")
            return {}
    
    def _generate_optimization_suggestions(self, 
                                         overall_metrics: Dict[str, float],
                                         patterns: List[CacheAccessPattern],
                                         trends: Dict[str, List[float]]) -> List[CacheOptimizationSuggestion]:
        """최적화 제안 생성"""
        suggestions = []
        
        try:
            # 1. 히트율 개선 제안
            if overall_metrics['hit_rate'] < self.baseline_metrics['target_hit_rate']:
                suggestions.extend(self._suggest_hit_rate_improvements(overall_metrics, patterns))
            
            # 2. 응답 시간 개선 제안
            if overall_metrics['avg_response_time_ms'] > self.baseline_metrics['target_response_time_ms']:
                suggestions.extend(self._suggest_response_time_improvements(overall_metrics, patterns))
            
            # 3. 메모리 효율성 개선 제안
            if overall_metrics['memory_efficiency'] < self.baseline_metrics['target_memory_efficiency']:
                suggestions.extend(self._suggest_memory_efficiency_improvements(overall_metrics, patterns))
            
            # 4. 캐시 오염 해결 제안
            if overall_metrics['pollution_score'] > self.baseline_metrics['acceptable_pollution_score']:
                suggestions.extend(self._suggest_pollution_cleanup(overall_metrics, patterns))
            
            # 5. 패턴 기반 최적화 제안
            suggestions.extend(self._suggest_pattern_optimizations(patterns))
            
            # 6. 추세 기반 예측 제안
            suggestions.extend(self._suggest_trend_based_optimizations(trends))
            
            # 우선순위 정렬
            suggestions.sort(key=lambda x: {
                'high': 3, 'medium': 2, 'low': 1
            }.get(x.priority, 0), reverse=True)
            
            self.logger.info(f"💡 생성된 최적화 제안: {len(suggestions)}개")
            return suggestions
            
        except Exception as e:
            self.logger.error(f"❌ 최적화 제안 생성 실패: {e}")
            return []
    
    def _suggest_hit_rate_improvements(self, metrics: Dict[str, float], patterns: List[CacheAccessPattern]) -> List[CacheOptimizationSuggestion]:
        """히트율 개선 제안"""
        suggestions = []
        
        # 낮은 히트율 패턴 식별
        low_hit_patterns = [p for p in patterns if p.hit_rate < 0.5]
        
        if low_hit_patterns:
            suggestions.append(CacheOptimizationSuggestion(
                suggestion_type='increase_cache_size',
                priority='high',
                description=f'히트율이 낮은 {len(low_hit_patterns)}개 패턴의 캐시 크기 증가',
                expected_improvement=0.15,
                implementation_effort='easy',
                affected_keys=[p.key_pattern for p in low_hit_patterns],
                parameters={'size_increase_factor': 1.5}
            ))
        
        # TTL 최적화
        poor_ttl_patterns = [p for p in patterns if p.ttl_effectiveness < 0.3]
        if poor_ttl_patterns:
            suggestions.append(CacheOptimizationSuggestion(
                suggestion_type='optimize_ttl',
                priority='medium',
                description=f'TTL 효율성이 낮은 {len(poor_ttl_patterns)}개 패턴의 TTL 조정',
                expected_improvement=0.1,
                implementation_effort='easy',
                affected_keys=[p.key_pattern for p in poor_ttl_patterns],
                parameters={'ttl_multiplier': 2.0}
            ))
        
        return suggestions
    
    def _suggest_response_time_improvements(self, metrics: Dict[str, float], patterns: List[CacheAccessPattern]) -> List[CacheOptimizationSuggestion]:
        """응답 시간 개선 제안"""
        suggestions = []
        
        # 느린 패턴 식별
        slow_patterns = [p for p in patterns if p.avg_response_time_ms > 100]
        
        if slow_patterns:
            suggestions.append(CacheOptimizationSuggestion(
                suggestion_type='enable_compression',
                priority='medium',
                description=f'응답 시간이 느린 {len(slow_patterns)}개 패턴에 압축 적용',
                expected_improvement=0.3,
                implementation_effort='medium',
                affected_keys=[p.key_pattern for p in slow_patterns],
                parameters={'compression_type': 'lz4'}
            ))
        
        # 큰 데이터 패턴
        large_data_patterns = [p for p in patterns if p.size_category == 'large']
        if large_data_patterns:
            suggestions.append(CacheOptimizationSuggestion(
                suggestion_type='implement_streaming',
                priority='high',
                description=f'대용량 데이터 {len(large_data_patterns)}개 패턴에 스트리밍 적용',
                expected_improvement=0.5,
                implementation_effort='hard',
                affected_keys=[p.key_pattern for p in large_data_patterns],
                parameters={'chunk_size_mb': 1.0}
            ))
        
        return suggestions
    
    def _suggest_memory_efficiency_improvements(self, metrics: Dict[str, float], patterns: List[CacheAccessPattern]) -> List[CacheOptimizationSuggestion]:
        """메모리 효율성 개선 제안"""
        suggestions = []
        
        # 메모리 효율성이 낮은 경우
        if metrics['memory_efficiency'] < 0.5:
            suggestions.append(CacheOptimizationSuggestion(
                suggestion_type='adjust_cache_levels',
                priority='high',
                description='L1/L2 캐시 레벨 간 데이터 재분배',
                expected_improvement=0.2,
                implementation_effort='medium',
                affected_keys=['all'],
                parameters={'l1_ratio': 0.6, 'l2_ratio': 0.4}
            ))
        
        return suggestions
    
    def _suggest_pollution_cleanup(self, metrics: Dict[str, float], patterns: List[CacheAccessPattern]) -> List[CacheOptimizationSuggestion]:
        """캐시 오염 정리 제안"""
        suggestions = []
        
        if metrics['pollution_score'] > 0.3:
            suggestions.append(CacheOptimizationSuggestion(
                suggestion_type='aggressive_cleanup',
                priority='medium',
                description=f'캐시 오염도 {metrics["pollution_score"]:.1%} - 적극적 정리 필요',
                expected_improvement=0.15,
                implementation_effort='easy',
                affected_keys=['stale_keys'],
                parameters={'cleanup_threshold_hours': 2}
            ))
        
        return suggestions
    
    def _suggest_pattern_optimizations(self, patterns: List[CacheAccessPattern]) -> List[CacheOptimizationSuggestion]:
        """패턴 기반 최적화 제안"""
        suggestions = []
        
        # 피크 시간 패턴
        peak_patterns = [p for p in patterns if p.temporal_pattern == 'peak']
        if peak_patterns:
            suggestions.append(CacheOptimizationSuggestion(
                suggestion_type='implement_preloading',
                priority='medium',
                description=f'피크 시간 패턴 {len(peak_patterns)}개에 사전 로딩 적용',
                expected_improvement=0.2,
                implementation_effort='medium',
                affected_keys=[p.key_pattern for p in peak_patterns],
                parameters={'preload_window_minutes': 30}
            ))
        
        # 카테고리별 최적화
        audio_patterns = [p for p in patterns if p.category == 'audio']
        if len(audio_patterns) > 3:
            suggestions.append(CacheOptimizationSuggestion(
                suggestion_type='category_specific_strategy',
                priority='low',
                description='오디오 분석 전용 캐시 전략 적용',
                expected_improvement=0.1,
                implementation_effort='medium',
                affected_keys=[p.key_pattern for p in audio_patterns],
                parameters={'strategy': 'audio_optimized'}
            ))
        
        return suggestions
    
    def _suggest_trend_based_optimizations(self, trends: Dict[str, List[float]]) -> List[CacheOptimizationSuggestion]:
        """추세 기반 최적화 제안"""
        suggestions = []
        
        try:
            # 히트율 하락 추세
            if 'hit_rate' in trends and len(trends['hit_rate']) > 5:
                recent_hit_rate = np.mean(trends['hit_rate'][-3:])
                older_hit_rate = np.mean(trends['hit_rate'][:3])
                
                if recent_hit_rate < older_hit_rate * 0.9:  # 10% 이상 하락
                    suggestions.append(CacheOptimizationSuggestion(
                        suggestion_type='investigate_hit_rate_decline',
                        priority='high',
                        description=f'히트율 하락 추세 감지 ({older_hit_rate:.1%} → {recent_hit_rate:.1%})',
                        expected_improvement=0.15,
                        implementation_effort='medium',
                        affected_keys=['all'],
                        parameters={'analysis_required': True}
                    ))
            
            # 응답 시간 증가 추세
            if 'response_time' in trends and len(trends['response_time']) > 5:
                recent_rt = np.mean(trends['response_time'][-3:])
                older_rt = np.mean(trends['response_time'][:3])
                
                if recent_rt > older_rt * 1.2:  # 20% 이상 증가
                    suggestions.append(CacheOptimizationSuggestion(
                        suggestion_type='address_performance_degradation',
                        priority='high',
                        description=f'응답 시간 증가 추세 감지 ({older_rt:.1f}ms → {recent_rt:.1f}ms)',
                        expected_improvement=0.25,
                        implementation_effort='medium',
                        affected_keys=['all'],
                        parameters={'performance_analysis_required': True}
                    ))
            
        except Exception as e:
            self.logger.error(f"❌ 추세 기반 제안 생성 실패: {e}")
        
        return suggestions
    
    def _create_empty_report(self) -> CachePerformanceReport:
        """빈 보고서 생성"""
        return CachePerformanceReport(
            analysis_timestamp=datetime.now().isoformat(),
            overall_hit_rate=0.0,
            avg_response_time_ms=0.0,
            memory_efficiency=0.0,
            cache_pollution_score=0.0,
            optimization_opportunities=0,
            patterns_detected=[],
            suggestions=[],
            performance_trends={}
        )
    
    def generate_visual_report(self, report: CachePerformanceReport, output_path: str) -> None:
        """시각적 보고서 생성"""
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('캐시 성능 분석 보고서', fontsize=16, fontweight='bold')
            
            # 1. 전체 성능 지표
            ax1 = axes[0, 0]
            metrics = ['히트율', '응답시간', '메모리효율', '오염점수']
            values = [
                report.overall_hit_rate * 100,
                min(100, report.avg_response_time_ms),  # 스케일 조정
                report.memory_efficiency * 100,
                report.cache_pollution_score * 100
            ]
            targets = [80, 50, 70, 30]  # 목표값
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, values, width, label='현재값', alpha=0.8)
            ax1.bar(x + width/2, targets, width, label='목표값', alpha=0.6)
            ax1.set_ylabel('점수')
            ax1.set_title('성능 지표 비교')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics)
            ax1.legend()
            
            # 2. 패턴별 히트율
            ax2 = axes[0, 1]
            if report.patterns_detected:
                pattern_names = [p.key_pattern[:10] + '...' if len(p.key_pattern) > 10 else p.key_pattern 
                               for p in report.patterns_detected[:8]]  # 상위 8개
                hit_rates = [p.hit_rate * 100 for p in report.patterns_detected[:8]]
                
                bars = ax2.bar(pattern_names, hit_rates)
                ax2.set_ylabel('히트율 (%)')
                ax2.set_title('패턴별 히트율')
                ax2.tick_params(axis='x', rotation=45)
                
                # 색상 구분 (히트율에 따라)
                for bar, hit_rate in zip(bars, hit_rates):
                    if hit_rate >= 80:
                        bar.set_color('green')
                    elif hit_rate >= 60:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')
            else:
                ax2.text(0.5, 0.5, '패턴 데이터 없음', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('패턴별 히트율')
            
            # 3. 성능 추세
            ax3 = axes[1, 0]
            if 'hit_rate' in report.performance_trends:
                hit_trend = report.performance_trends['hit_rate']
                if len(hit_trend) > 1:
                    ax3.plot(hit_trend, label='히트율', marker='o')
                    ax3.set_ylabel('히트율')
                    ax3.set_xlabel('시간 순서')
                    ax3.set_title('히트율 추세')
                    ax3.legend()
                else:
                    ax3.text(0.5, 0.5, '추세 데이터 부족', ha='center', va='center', transform=ax3.transAxes)
            else:
                ax3.text(0.5, 0.5, '추세 데이터 없음', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('히트율 추세')
            
            # 4. 최적화 제안 요약
            ax4 = axes[1, 1]
            if report.suggestions:
                priority_counts = defaultdict(int)
                for suggestion in report.suggestions:
                    priority_counts[suggestion.priority] += 1
                
                priorities = list(priority_counts.keys())
                counts = list(priority_counts.values())
                colors = {'high': 'red', 'medium': 'orange', 'low': 'green'}
                
                ax4.pie(counts, labels=priorities, autopct='%1.1f%%', 
                       colors=[colors.get(p, 'gray') for p in priorities])
                ax4.set_title('최적화 제안 우선순위')
            else:
                ax4.text(0.5, 0.5, '최적화 제안 없음', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('최적화 제안 우선순위')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 시각적 보고서 생성: {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ 시각적 보고서 생성 실패: {e}")
    
    def export_analysis_data(self, output_path: str) -> None:
        """분석 데이터 내보내기"""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'analysis_window_hours': self.analysis_window_hours,
                'access_logs_count': len(self.access_logs),
                'performance_metrics_count': len(self.performance_metrics),
                'baseline_metrics': self.baseline_metrics,
                'recent_access_logs': list(self.access_logs)[-1000:],  # 최근 1000개
                'recent_performance_metrics': list(self.performance_metrics)[-100:],  # 최근 100개
                'pattern_history': dict(self.pattern_history)
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"📊 분석 데이터 내보내기: {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ 분석 데이터 내보내기 실패: {e}")

# 사용 예시
if __name__ == "__main__":
    # 가상의 캐시 매니저 (테스트용)
    class MockCacheManager:
        def __init__(self):
            self.size = 100.0
            
        def get_size(self):
            return self.size
            
        def keys(self):
            return ['audio_001', 'image_002', 'model_003', 'config_004']
    
    # 캐시 분석기 테스트
    cache_manager = MockCacheManager()
    analyzer = CacheAnalyzer(cache_manager)
    
    print("🔍 캐시 분석기 테스트:")
    
    # 가상의 접근 로그 생성
    import random
    for i in range(100):
        analyzer.log_access(
            key=f"test_key_{i % 10}",
            hit=random.choice([True, False]),
            response_time_ms=random.uniform(10, 200),
            cache_level='l1_memory',
            operation='get',
            data_size=random.randint(1024, 1024*1024)
        )
    
    # 성능 분석 실행
    report = analyzer.analyze_cache_performance()
    
    print(f"✅ 분석 완료:")
    print(f"   히트율: {report.overall_hit_rate:.1%}")
    print(f"   평균 응답시간: {report.avg_response_time_ms:.1f}ms")
    print(f"   감지된 패턴: {len(report.patterns_detected)}개")
    print(f"   최적화 제안: {len(report.suggestions)}개")
    
    # 분석 데이터 내보내기
    analyzer.export_analysis_data("cache_analysis_data.json")
    
    print("✅ 캐시 분석기 테스트 완료")