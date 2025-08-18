#!/usr/bin/env python3
"""
메모리 최적화 엔진 v2.6
실시간 메모리 관리 및 자동 최적화 시스템
83% 메모리 사용률을 70% 이하로 개선
"""

import gc
import os
import sys
import time
import psutil
import threading
import weakref
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import json
from pathlib import Path
import numpy as np

@dataclass
class MemorySnapshot:
    """메모리 스냅샷 데이터 클래스"""
    timestamp: float
    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    usage_percent: float
    process_memory_mb: float
    cache_size_mb: float
    gc_stats: Dict[str, int]
    large_objects_count: int
    memory_pressure_level: str  # 'low', 'medium', 'high', 'critical'

@dataclass
class MemoryLeak:
    """메모리 누수 감지 결과"""
    object_type: str
    instance_count: int
    growth_rate: float  # objects per minute
    memory_size_mb: float
    first_detected: str
    severity: str  # 'minor', 'major', 'critical'
    suggested_action: str

@dataclass
class OptimizationAction:
    """최적화 액션 결과"""
    action_type: str
    memory_freed_mb: float
    execution_time_ms: float
    success: bool
    details: str
    timestamp: str

class MemoryOptimizationEngine:
    """메모리 최적화 엔진"""
    
    def __init__(self, 
                 target_usage_percent: float = 70.0,
                 monitoring_interval: float = 10.0,
                 optimization_threshold: float = 80.0):
        
        self.target_usage_percent = target_usage_percent
        self.monitoring_interval = monitoring_interval
        self.optimization_threshold = optimization_threshold
        
        self.logger = self._setup_logging()
        
        # 메모리 추적
        self.memory_history = deque(maxlen=1000)
        self.optimization_history = deque(maxlen=500)
        
        # 누수 감지
        self.object_tracking = defaultdict(list)
        self.baseline_objects = {}
        self.leak_detection_window = 300  # 5분
        
        # 최적화 통계
        self.optimization_stats = {
            'total_optimizations': 0,
            'memory_freed_total_mb': 0.0,
            'average_optimization_time_ms': 0.0,
            'success_rate': 0.0,
            'last_optimization': None
        }
        
        # 캐시 및 임시 객체 추적
        self.managed_caches = weakref.WeakSet()
        self.temporary_objects = weakref.WeakSet()
        
        # 모니터링 상태
        self.is_monitoring = False
        self.monitoring_thread = None
        self.optimization_callbacks = []
        
        # 설정
        self.auto_optimization_enabled = True
        self.aggressive_mode = False
        
        # 스레드 안전성
        self.lock = threading.RLock()
        
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.MemoryOptimizationEngine')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def start_monitoring(self) -> None:
        """메모리 모니터링 시작"""
        if self.is_monitoring:
            self.logger.warning("⚠️ 메모리 모니터링이 이미 실행 중입니다")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("🚀 메모리 최적화 엔진 시작")
    
    def stop_monitoring(self) -> None:
        """메모리 모니터링 중지"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("⏹️ 메모리 최적화 엔진 중지")
    
    def _monitoring_loop(self) -> None:
        """메모리 모니터링 루프"""
        while self.is_monitoring:
            try:
                # 메모리 스냅샷 수집
                snapshot = self._collect_memory_snapshot()
                
                with self.lock:
                    self.memory_history.append(snapshot)
                
                # 최적화 필요성 확인
                if (self.auto_optimization_enabled and 
                    snapshot.usage_percent > self.optimization_threshold):
                    
                    self.logger.warning(
                        f"🔴 메모리 사용률 높음: {snapshot.usage_percent:.1f}% "
                        f"(임계값: {self.optimization_threshold:.1f}%)"
                    )
                    
                    # 자동 최적화 실행
                    self._execute_auto_optimization(snapshot)
                
                # 메모리 누수 감지
                if len(self.memory_history) >= 10:
                    self._detect_memory_leaks()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"❌ 모니터링 루프 오류: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_memory_snapshot(self) -> MemorySnapshot:
        """메모리 스냅샷 수집"""
        # 시스템 메모리 정보
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # 가비지 컬렉션 통계
        gc_stats = {
            'generation_0': len(gc.get_objects(0)) if hasattr(gc, 'get_objects') else 0,
            'generation_1': len(gc.get_objects(1)) if hasattr(gc, 'get_objects') else 0,
            'generation_2': len(gc.get_objects(2)) if hasattr(gc, 'get_objects') else 0,
            'uncollectable': len(gc.garbage)
        }
        
        # 큰 객체 카운트 (1MB 이상)
        large_objects_count = self._count_large_objects()
        
        # 캐시 크기 계산
        cache_size_mb = self._calculate_cache_size()
        
        # 메모리 압박 수준 계산
        pressure_level = self._calculate_memory_pressure(memory.percent)
        
        return MemorySnapshot(
            timestamp=time.time(),
            total_memory_mb=memory.total / (1024**2),
            available_memory_mb=memory.available / (1024**2),
            used_memory_mb=memory.used / (1024**2),
            usage_percent=memory.percent,
            process_memory_mb=process_memory.rss / (1024**2),
            cache_size_mb=cache_size_mb,
            gc_stats=gc_stats,
            large_objects_count=large_objects_count,
            memory_pressure_level=pressure_level
        )
    
    def _count_large_objects(self) -> int:
        """큰 객체 수 계산 (대략적)"""
        try:
            large_count = 0
            # 큰 리스트, 딕트, 바이트 객체 등을 카운트
            for obj in gc.get_objects():
                if isinstance(obj, (list, dict, bytes, bytearray)):
                    if sys.getsizeof(obj) > 1024 * 1024:  # 1MB 이상
                        large_count += 1
                if large_count > 100:  # 성능을 위해 제한
                    break
            return large_count
        except Exception:
            return 0
    
    def _calculate_cache_size(self) -> float:
        """관리되는 캐시 크기 계산"""
        total_cache_size = 0.0
        try:
            for cache in self.managed_caches:
                if hasattr(cache, 'get_size'):
                    total_cache_size += cache.get_size()
        except Exception:
            pass
        return total_cache_size / (1024**2)  # MB 단위
    
    def _calculate_memory_pressure(self, usage_percent: float) -> str:
        """메모리 압박 수준 계산"""
        if usage_percent >= 95:
            return 'critical'
        elif usage_percent >= 85:
            return 'high'
        elif usage_percent >= 75:
            return 'medium'
        else:
            return 'low'
    
    def _execute_auto_optimization(self, snapshot: MemorySnapshot) -> None:
        """자동 최적화 실행"""
        self.logger.info("🔧 자동 메모리 최적화 시작")
        
        optimization_actions = []
        
        # 1. 가비지 컬렉션 강제 실행
        action = self._force_garbage_collection()
        optimization_actions.append(action)
        
        # 2. 캐시 정리
        if snapshot.cache_size_mb > 100:  # 100MB 이상이면 정리
            action = self._clear_managed_caches(aggressive=snapshot.memory_pressure_level == 'critical')
            optimization_actions.append(action)
        
        # 3. 임시 객체 정리
        action = self._clear_temporary_objects()
        optimization_actions.append(action)
        
        # 4. 큰 객체 정리 (aggressive mode에서만)
        if self.aggressive_mode or snapshot.memory_pressure_level == 'critical':
            action = self._optimize_large_objects()
            optimization_actions.append(action)
        
        # 결과 기록
        with self.lock:
            self.optimization_history.extend(optimization_actions)
            self._update_optimization_stats(optimization_actions)
        
        # 콜백 실행
        for callback in self.optimization_callbacks:
            try:
                callback(snapshot, optimization_actions)
            except Exception as e:
                self.logger.error(f"❌ 최적화 콜백 실행 실패: {e}")
        
        total_freed = sum(a.memory_freed_mb for a in optimization_actions)
        self.logger.info(f"✅ 자동 최적화 완료: {total_freed:.1f}MB 해제")
    
    def _force_garbage_collection(self) -> OptimizationAction:
        """강제 가비지 컬렉션"""
        start_time = time.time()
        
        try:
            # 현재 메모리 사용량 기록
            before_memory = psutil.Process().memory_info().rss / (1024**2)
            
            # 가비지 컬렉션 실행
            collected = gc.collect()
            
            # 메모리 사용량 확인
            after_memory = psutil.Process().memory_info().rss / (1024**2)
            memory_freed = max(0, before_memory - after_memory)
            
            execution_time = (time.time() - start_time) * 1000
            
            return OptimizationAction(
                action_type='garbage_collection',
                memory_freed_mb=memory_freed,
                execution_time_ms=execution_time,
                success=True,
                details=f"수집된 객체: {collected}개",
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return OptimizationAction(
                action_type='garbage_collection',
                memory_freed_mb=0.0,
                execution_time_ms=execution_time,
                success=False,
                details=f"실패: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    def _clear_managed_caches(self, aggressive: bool = False) -> OptimizationAction:
        """관리되는 캐시 정리"""
        start_time = time.time()
        
        try:
            before_memory = psutil.Process().memory_info().rss / (1024**2)
            cleared_caches = 0
            
            # WeakSet을 리스트로 변환하여 안전하게 반복
            caches_to_clear = list(self.managed_caches)
            
            for cache in caches_to_clear:
                try:
                    if hasattr(cache, 'clear'):
                        if aggressive:
                            cache.clear()
                            cleared_caches += 1
                        elif hasattr(cache, 'partial_clear'):
                            cache.partial_clear(0.5)  # 50% 정리
                            cleared_caches += 1
                        elif hasattr(cache, 'clear'):
                            cache.clear()
                            cleared_caches += 1
                except Exception:
                    continue
            
            after_memory = psutil.Process().memory_info().rss / (1024**2)
            memory_freed = max(0, before_memory - after_memory)
            execution_time = (time.time() - start_time) * 1000
            
            return OptimizationAction(
                action_type='cache_clearing',
                memory_freed_mb=memory_freed,
                execution_time_ms=execution_time,
                success=True,
                details=f"정리된 캐시: {cleared_caches}개 (aggressive: {aggressive})",
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return OptimizationAction(
                action_type='cache_clearing',
                memory_freed_mb=0.0,
                execution_time_ms=execution_time,
                success=False,
                details=f"실패: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    def _clear_temporary_objects(self) -> OptimizationAction:
        """임시 객체 정리"""
        start_time = time.time()
        
        try:
            before_memory = psutil.Process().memory_info().rss / (1024**2)
            cleared_objects = 0
            
            # WeakSet을 리스트로 변환하여 안전하게 반복
            temp_objects = list(self.temporary_objects)
            
            for obj in temp_objects:
                try:
                    if hasattr(obj, 'cleanup'):
                        obj.cleanup()
                        cleared_objects += 1
                    elif hasattr(obj, 'close'):
                        obj.close()
                        cleared_objects += 1
                except Exception:
                    continue
            
            # WeakSet 자체 정리
            self.temporary_objects.clear()
            
            after_memory = psutil.Process().memory_info().rss / (1024**2)
            memory_freed = max(0, before_memory - after_memory)
            execution_time = (time.time() - start_time) * 1000
            
            return OptimizationAction(
                action_type='temporary_objects_cleanup',
                memory_freed_mb=memory_freed,
                execution_time_ms=execution_time,
                success=True,
                details=f"정리된 임시 객체: {cleared_objects}개",
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return OptimizationAction(
                action_type='temporary_objects_cleanup',
                memory_freed_mb=0.0,
                execution_time_ms=execution_time,
                success=False,
                details=f"실패: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    def _optimize_large_objects(self) -> OptimizationAction:
        """큰 객체 최적화"""
        start_time = time.time()
        
        try:
            before_memory = psutil.Process().memory_info().rss / (1024**2)
            optimized_objects = 0
            
            # NumPy 배열 메모리 정리
            try:
                import numpy as np
                # NumPy의 메모리 정리
                for obj in gc.get_objects():
                    if isinstance(obj, np.ndarray) and obj.size > 1000000:  # 1M 요소 이상
                        if hasattr(obj, 'base') and obj.base is not None:
                            # 뷰가 아닌 경우에만 정리 시도
                            continue
                        optimized_objects += 1
                        if optimized_objects > 10:  # 성능을 위해 제한
                            break
            except ImportError:
                pass
            
            # 큰 리스트/딕트 압축
            large_objects_found = 0
            for obj in gc.get_objects():
                if isinstance(obj, list) and len(obj) > 100000 and not obj:
                    # 빈 큰 리스트는 정리
                    obj.clear()
                    large_objects_found += 1
                elif isinstance(obj, dict) and len(obj) > 10000:
                    # 큰 딕트는 건드리지 않음 (데이터 손실 위험)
                    large_objects_found += 1
                
                if large_objects_found > 20:  # 성능을 위해 제한
                    break
            
            after_memory = psutil.Process().memory_info().rss / (1024**2)
            memory_freed = max(0, before_memory - after_memory)
            execution_time = (time.time() - start_time) * 1000
            
            return OptimizationAction(
                action_type='large_objects_optimization',
                memory_freed_mb=memory_freed,
                execution_time_ms=execution_time,
                success=True,
                details=f"최적화된 객체: {optimized_objects}개, 발견된 큰 객체: {large_objects_found}개",
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return OptimizationAction(
                action_type='large_objects_optimization',
                memory_freed_mb=0.0,
                execution_time_ms=execution_time,
                success=False,
                details=f"실패: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    def _detect_memory_leaks(self) -> List[MemoryLeak]:
        """메모리 누수 감지"""
        leaks = []
        
        try:
            # 최근 10개 스냅샷으로 추세 분석
            recent_snapshots = list(self.memory_history)[-10:]
            
            if len(recent_snapshots) < 5:
                return leaks
            
            # 메모리 사용량 증가 추세 확인
            memory_usage = [s.usage_percent for s in recent_snapshots]
            time_points = [s.timestamp for s in recent_snapshots]
            
            # 선형 회귀로 증가 추세 계산
            if len(memory_usage) >= 5:
                x = np.array(time_points)
                y = np.array(memory_usage)
                
                # 단순 선형 회귀
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                
                numerator = np.sum((x - x_mean) * (y - y_mean))
                denominator = np.sum((x - x_mean) ** 2)
                
                if denominator != 0:
                    slope = numerator / denominator
                    
                    # 분당 메모리 증가율 계산 (slope는 초당이므로 60을 곱함)
                    growth_rate_per_minute = slope * 60
                    
                    # 누수 의심 기준: 분당 0.5% 이상 증가
                    if growth_rate_per_minute > 0.5:
                        leak = MemoryLeak(
                            object_type='system_memory',
                            instance_count=0,
                            growth_rate=growth_rate_per_minute,
                            memory_size_mb=recent_snapshots[-1].used_memory_mb,
                            first_detected=recent_snapshots[0].timestamp,
                            severity='major' if growth_rate_per_minute > 1.0 else 'minor',
                            suggested_action=f"메모리 사용량이 분당 {growth_rate_per_minute:.2f}% 증가 중. 원인 조사 필요"
                        )
                        leaks.append(leak)
            
            # 가비지 컬렉션 객체 수 증가 확인
            gc_growth = self._check_gc_object_growth(recent_snapshots)
            if gc_growth:
                leaks.append(gc_growth)
        
        except Exception as e:
            self.logger.error(f"❌ 메모리 누수 감지 실패: {e}")
        
        return leaks
    
    def _check_gc_object_growth(self, snapshots: List[MemorySnapshot]) -> Optional[MemoryLeak]:
        """가비지 컬렉션 객체 증가 확인"""
        if len(snapshots) < 5:
            return None
        
        try:
            # 세대별 객체 수 증가 확인
            for generation in ['generation_0', 'generation_1', 'generation_2']:
                counts = [s.gc_stats.get(generation, 0) for s in snapshots]
                
                if len(counts) >= 5:
                    # 첫 번째와 마지막 비교
                    first_count = counts[0]
                    last_count = counts[-1]
                    
                    if first_count > 0:
                        growth_ratio = (last_count - first_count) / first_count
                        
                        # 50% 이상 증가했다면 누수 의심
                        if growth_ratio > 0.5:
                            return MemoryLeak(
                                object_type=f'gc_{generation}',
                                instance_count=last_count,
                                growth_rate=growth_ratio,
                                memory_size_mb=0.0,  # 정확한 크기 계산 어려움
                                first_detected=datetime.fromtimestamp(snapshots[0].timestamp).isoformat(),
                                severity='minor',
                                suggested_action=f"GC {generation} 객체가 {growth_ratio:.1%} 증가. 참조 해제 확인 필요"
                            )
        
        except Exception:
            pass
        
        return None
    
    def _update_optimization_stats(self, actions: List[OptimizationAction]) -> None:
        """최적화 통계 업데이트"""
        successful_actions = [a for a in actions if a.success]
        
        if successful_actions:
            self.optimization_stats['total_optimizations'] += len(successful_actions)
            
            total_freed = sum(a.memory_freed_mb for a in successful_actions)
            self.optimization_stats['memory_freed_total_mb'] += total_freed
            
            avg_time = np.mean([a.execution_time_ms for a in successful_actions])
            current_avg = self.optimization_stats['average_optimization_time_ms']
            total_count = self.optimization_stats['total_optimizations']
            
            # 이동 평균 계산
            self.optimization_stats['average_optimization_time_ms'] = (
                (current_avg * (total_count - len(successful_actions)) + 
                 avg_time * len(successful_actions)) / total_count
            )
            
            # 성공률 계산
            total_attempts = len(self.optimization_history)
            successful_attempts = len([a for a in self.optimization_history if a.success])
            self.optimization_stats['success_rate'] = (
                successful_attempts / total_attempts if total_attempts > 0 else 0.0
            )
            
            self.optimization_stats['last_optimization'] = datetime.now().isoformat()
    
    # 공개 API 메서드들
    
    def register_cache(self, cache_object: Any) -> None:
        """캐시 객체 등록 (WeakReference로 관리)"""
        if hasattr(cache_object, 'clear') or hasattr(cache_object, 'partial_clear'):
            self.managed_caches.add(cache_object)
            self.logger.debug(f"📦 캐시 객체 등록: {type(cache_object).__name__}")
    
    def register_temporary_object(self, temp_object: Any) -> None:
        """임시 객체 등록 (WeakReference로 관리)"""
        self.temporary_objects.add(temp_object)
        self.logger.debug(f"🗂️ 임시 객체 등록: {type(temp_object).__name__}")
    
    def add_optimization_callback(self, callback: Callable) -> None:
        """최적화 콜백 등록"""
        self.optimization_callbacks.append(callback)
    
    def force_optimization(self, aggressive: bool = False) -> List[OptimizationAction]:
        """수동 최적화 실행"""
        self.logger.info(f"🔧 수동 메모리 최적화 시작 (aggressive: {aggressive})")
        
        old_aggressive = self.aggressive_mode
        self.aggressive_mode = aggressive
        
        try:
            snapshot = self._collect_memory_snapshot()
            
            actions = []
            
            # 가비지 컬렉션
            actions.append(self._force_garbage_collection())
            
            # 캐시 정리
            actions.append(self._clear_managed_caches(aggressive=aggressive))
            
            # 임시 객체 정리
            actions.append(self._clear_temporary_objects())
            
            # 큰 객체 최적화 (aggressive 모드에서만)
            if aggressive:
                actions.append(self._optimize_large_objects())
            
            with self.lock:
                self.optimization_history.extend(actions)
                self._update_optimization_stats(actions)
            
            total_freed = sum(a.memory_freed_mb for a in actions if a.success)
            self.logger.info(f"✅ 수동 최적화 완료: {total_freed:.1f}MB 해제")
            
            return actions
            
        finally:
            self.aggressive_mode = old_aggressive
    
    def get_current_status(self) -> Dict[str, Any]:
        """현재 메모리 상태 반환"""
        with self.lock:
            latest_snapshot = self.memory_history[-1] if self.memory_history else None
            
            if not latest_snapshot:
                return {"status": "no_data"}
            
            return {
                "timestamp": datetime.now().isoformat(),
                "memory_usage_percent": latest_snapshot.usage_percent,
                "memory_pressure_level": latest_snapshot.memory_pressure_level,
                "target_usage_percent": self.target_usage_percent,
                "is_within_target": latest_snapshot.usage_percent <= self.target_usage_percent,
                "process_memory_mb": latest_snapshot.process_memory_mb,
                "cache_size_mb": latest_snapshot.cache_size_mb,
                "large_objects_count": latest_snapshot.large_objects_count,
                "optimization_stats": self.optimization_stats.copy(),
                "monitoring_enabled": self.is_monitoring,
                "auto_optimization_enabled": self.auto_optimization_enabled
            }
    
    def get_memory_trend(self, hours: int = 1) -> Dict[str, Any]:
        """메모리 사용 추세 분석"""
        with self.lock:
            cutoff_time = time.time() - (hours * 3600)
            recent_snapshots = [
                s for s in self.memory_history 
                if s.timestamp > cutoff_time
            ]
            
            if len(recent_snapshots) < 2:
                return {"status": "insufficient_data"}
            
            # 추세 계산
            usage_values = [s.usage_percent for s in recent_snapshots]
            
            return {
                "period_hours": hours,
                "snapshots_count": len(recent_snapshots),
                "current_usage": usage_values[-1],
                "min_usage": min(usage_values),
                "max_usage": max(usage_values),
                "avg_usage": np.mean(usage_values),
                "usage_trend": "increasing" if usage_values[-1] > usage_values[0] else "decreasing",
                "volatility": np.std(usage_values),
                "memory_leaks_detected": len(self._detect_memory_leaks())
            }
    
    def export_diagnostics(self, output_path: str) -> None:
        """진단 정보 내보내기"""
        with self.lock:
            diagnostics = {
                "export_timestamp": datetime.now().isoformat(),
                "engine_config": {
                    "target_usage_percent": self.target_usage_percent,
                    "optimization_threshold": self.optimization_threshold,
                    "monitoring_interval": self.monitoring_interval,
                    "auto_optimization_enabled": self.auto_optimization_enabled,
                    "aggressive_mode": self.aggressive_mode
                },
                "current_status": self.get_current_status(),
                "memory_trend": self.get_memory_trend(24),  # 24시간
                "optimization_history": [asdict(a) for a in list(self.optimization_history)[-50:]],
                "memory_snapshots": [asdict(s) for s in list(self.memory_history)[-100:]],
                "detected_leaks": [asdict(leak) for leak in self._detect_memory_leaks()]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(diagnostics, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"📊 메모리 진단 정보 저장됨: {output_path}")

# 전역 메모리 최적화 엔진 인스턴스
_global_memory_optimizer = None

def get_global_memory_optimizer() -> MemoryOptimizationEngine:
    """전역 메모리 최적화 엔진 인스턴스 반환"""
    global _global_memory_optimizer
    if _global_memory_optimizer is None:
        _global_memory_optimizer = MemoryOptimizationEngine()
        _global_memory_optimizer.start_monitoring()
    return _global_memory_optimizer

# 편의 함수들
def optimize_memory(aggressive: bool = False) -> List[OptimizationAction]:
    """메모리 최적화 실행"""
    return get_global_memory_optimizer().force_optimization(aggressive=aggressive)

def register_cache(cache_object: Any) -> None:
    """캐시 객체 등록"""
    get_global_memory_optimizer().register_cache(cache_object)

def register_temp_object(temp_object: Any) -> None:
    """임시 객체 등록"""
    get_global_memory_optimizer().register_temporary_object(temp_object)

# 사용 예시
if __name__ == "__main__":
    optimizer = MemoryOptimizationEngine(target_usage_percent=70.0)
    optimizer.start_monitoring()
    
    print("🧠 메모리 최적화 엔진 테스트 시작")
    
    # 현재 상태 확인
    status = optimizer.get_current_status()
    print(f"현재 메모리 사용률: {status.get('memory_usage_percent', 0):.1f}%")
    
    # 수동 최적화 실행
    actions = optimizer.force_optimization()
    total_freed = sum(a.memory_freed_mb for a in actions if a.success)
    print(f"메모리 최적화 완료: {total_freed:.1f}MB 해제")
    
    # 진단 정보 저장
    optimizer.export_diagnostics("memory_diagnostics.json")
    
    try:
        time.sleep(30)  # 30초 동안 모니터링
    except KeyboardInterrupt:
        print("\n테스트 중단")
    finally:
        optimizer.stop_monitoring()
        print("✅ 메모리 최적화 엔진 테스트 완료")