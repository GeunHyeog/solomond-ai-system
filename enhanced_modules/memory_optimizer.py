#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[MEMORY] 메모리 최적화 시스템
Advanced Memory Management & Optimization System

핵심 기능:
1. 실시간 메모리 사용량 모니터링
2. 자동 가비지 컬렉션 최적화
3. 메모리 누수 탐지 및 방지
4. 캐시 관리 및 최적화
5. 대용량 파일 처리 메모리 효율화
"""

import os
import sys
import gc
import psutil
import weakref
import threading
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass
from datetime import datetime
import time
import logging
from pathlib import Path
import tracemalloc
from collections import defaultdict
import numpy as np

# 로깅 설정
logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """메모리 스냅샷 데이터"""
    timestamp: str
    total_mb: float
    available_mb: float
    used_mb: float
    used_percent: float
    process_memory_mb: float
    gc_counts: Dict[int, int]

class MemoryOptimizer:
    """메모리 최적화 시스템"""
    
    def __init__(self):
        self.memory_threshold = 0.85  # 85% 메모리 사용 시 최적화 트리거
        self.gc_threshold_scale = 1.5  # 기본 GC 임계값 스케일 조정
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 10.0  # 10초마다 체크
        
        # 메모리 추적
        self.memory_history = []
        self.max_history_size = 100
        
        # 객체 참조 추적
        self.tracked_objects = weakref.WeakSet()
        self.large_objects = weakref.WeakValueDictionary()
        
        # 캐시 관리
        self.cache_registry = {}
        self.cache_stats = defaultdict(dict)
        
        # GC 최적화 설정
        self._optimize_gc_settings()
        
        # 메모리 추적 시작
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        logger.info("[MEMORY] 메모리 최적화 시스템 초기화 완료")
    
    def _optimize_gc_settings(self):
        """가비지 컬렉션 설정 최적화"""
        # 현재 GC 임계값 가져오기
        current_thresholds = gc.get_threshold()
        
        # 최적화된 임계값 계산 (더 적극적인 GC)
        optimized_thresholds = (
            int(current_thresholds[0] / self.gc_threshold_scale),  # generation 0
            int(current_thresholds[1] / self.gc_threshold_scale),  # generation 1
            int(current_thresholds[2] / self.gc_threshold_scale)   # generation 2
        )
        
        # 새 임계값 설정
        gc.set_threshold(*optimized_thresholds)
        
        logger.info(f"[MEMORY] GC 임계값 최적화: {current_thresholds} -> {optimized_thresholds}")
    
    def start_monitoring(self):
        """메모리 모니터링 시작"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("[MEMORY] 메모리 모니터링 시작")
    
    def stop_monitoring(self):
        """메모리 모니터링 중지"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        logger.info("[MEMORY] 메모리 모니터링 중지")
    
    def _monitor_loop(self):
        """메모리 모니터링 루프"""
        while self.monitoring_active:
            try:
                snapshot = self._take_memory_snapshot()
                self._update_history(snapshot)
                
                # 메모리 사용률이 임계값을 초과하면 최적화 수행
                if snapshot.used_percent > self.memory_threshold:
                    self.optimize_memory()
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.warning(f"[MEMORY] 모니터링 오류: {e}")
                time.sleep(self.monitor_interval)
    
    def _take_memory_snapshot(self) -> MemorySnapshot:
        """현재 메모리 상태 스냅샷"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024**2
        
        return MemorySnapshot(
            timestamp=datetime.now().isoformat(),
            total_mb=memory.total / 1024**2,
            available_mb=memory.available / 1024**2,
            used_mb=memory.used / 1024**2,
            used_percent=memory.percent,
            process_memory_mb=process_memory,
            gc_counts={i: gc.get_count()[i] for i in range(3)}
        )
    
    def _update_history(self, snapshot: MemorySnapshot):
        """메모리 히스토리 업데이트"""
        self.memory_history.append(snapshot)
        
        # 히스토리 크기 제한
        if len(self.memory_history) > self.max_history_size:
            self.memory_history = self.memory_history[-self.max_history_size:]
    
    def optimize_memory(self, force: bool = False):
        """메모리 최적화 수행"""
        start_snapshot = self._take_memory_snapshot()
        logger.info(f"[MEMORY] 메모리 최적화 시작 - 사용률: {start_snapshot.used_percent:.1f}%")
        
        optimization_steps = []
        
        # 1. 명시적 가비지 컬렉션
        collected_objects = self._force_garbage_collection()
        optimization_steps.append(f"GC: {collected_objects}개 객체 정리")
        
        # 2. 캐시 정리
        cache_cleared = self._clear_old_caches()
        optimization_steps.append(f"캐시: {cache_cleared}개 항목 정리")
        
        # 3. 대용량 객체 정리
        large_objects_cleared = self._cleanup_large_objects()
        optimization_steps.append(f"대용량 객체: {large_objects_cleared}개 정리")
        
        # 4. NumPy/PyTorch 메모리 정리
        ml_memory_cleared = self._cleanup_ml_memory()
        if ml_memory_cleared:
            optimization_steps.append("ML 프레임워크 메모리 정리")
        
        # 최적화 후 상태 확인
        end_snapshot = self._take_memory_snapshot()
        
        memory_saved = start_snapshot.used_mb - end_snapshot.used_mb
        
        logger.info(f"[MEMORY] 메모리 최적화 완료 - {memory_saved:.1f}MB 절약")
        for step in optimization_steps:
            logger.info(f"[MEMORY]   - {step}")
        
        return {
            'start_memory_mb': start_snapshot.used_mb,
            'end_memory_mb': end_snapshot.used_mb,
            'memory_saved_mb': memory_saved,
            'optimization_steps': optimization_steps
        }
    
    def _force_garbage_collection(self) -> int:
        """강제 가비지 컬렉션 수행"""
        # 각 세대별로 가비지 컬렉션 수행
        total_collected = 0
        
        for generation in range(3):
            collected = gc.collect(generation)
            total_collected += collected
        
        # 전체 가비지 컬렉션
        collected = gc.collect()
        total_collected += collected
        
        return total_collected
    
    def _clear_old_caches(self) -> int:
        """오래된 캐시 정리"""
        cleared_count = 0
        current_time = time.time()
        
        for cache_name, cache_info in list(self.cache_registry.items()):
            cache = cache_info.get('cache')
            max_age = cache_info.get('max_age', 3600)  # 기본 1시간
            
            if cache and hasattr(cache, 'clear'):
                if current_time - cache_info.get('last_access', 0) > max_age:
                    cache.clear()
                    cleared_count += 1
                    logger.debug(f"[MEMORY] 캐시 정리: {cache_name}")
        
        return cleared_count
    
    def _cleanup_large_objects(self) -> int:
        """대용량 객체 정리"""
        cleared_count = 0
        
        # WeakValueDictionary에서 참조가 끊어진 객체들은 자동으로 제거됨
        # 여기서는 명시적으로 정리할 대상을 찾음
        for obj_id, obj in list(self.large_objects.items()):
            try:
                # 객체가 여전히 유효하고 대용량인지 확인
                if hasattr(obj, '__sizeof__'):
                    size = obj.__sizeof__()
                    if size > 100 * 1024 * 1024:  # 100MB 이상
                        # 대용량 객체에 대한 추가 정리 로직
                        if hasattr(obj, 'clear') and callable(getattr(obj, 'clear')):
                            obj.clear()
                            cleared_count += 1
            except:
                # 객체가 이미 정리되었거나 접근 불가능
                cleared_count += 1
        
        return cleared_count
    
    def _cleanup_ml_memory(self) -> bool:
        """ML 프레임워크 메모리 정리"""
        cleaned = False
        
        # PyTorch GPU 메모리 정리
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                cleaned = True
        except ImportError:
            pass
        
        # TensorFlow GPU 메모리 정리
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            cleaned = True
        except ImportError:
            pass
        
        return cleaned
    
    def register_cache(self, name: str, cache_object: Any, max_age: int = 3600):
        """캐시 객체 등록"""
        self.cache_registry[name] = {
            'cache': cache_object,
            'max_age': max_age,
            'last_access': time.time()
        }
        logger.debug(f"[MEMORY] 캐시 등록: {name}")
    
    def access_cache(self, name: str):
        """캐시 접근 시간 업데이트"""
        if name in self.cache_registry:
            self.cache_registry[name]['last_access'] = time.time()
    
    def track_large_object(self, obj: Any, identifier: str = None):
        """대용량 객체 추적"""
        if identifier is None:
            identifier = f"obj_{id(obj)}"
        
        try:
            size = obj.__sizeof__()
            if size > 10 * 1024 * 1024:  # 10MB 이상만 추적
                self.large_objects[identifier] = obj
                logger.debug(f"[MEMORY] 대용량 객체 추적 시작: {identifier} ({size/1024/1024:.1f}MB)")
        except:
            pass
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계 정보 반환"""
        current_snapshot = self._take_memory_snapshot()
        
        stats = {
            'current': {
                'total_mb': current_snapshot.total_mb,
                'used_mb': current_snapshot.used_mb,
                'available_mb': current_snapshot.available_mb,
                'used_percent': current_snapshot.used_percent,
                'process_memory_mb': current_snapshot.process_memory_mb
            },
            'gc_info': {
                'thresholds': gc.get_threshold(),
                'counts': gc.get_count(),
                'stats': gc.get_stats()
            },
            'tracking': {
                'tracked_objects': len(self.tracked_objects),
                'large_objects': len(self.large_objects),
                'registered_caches': len(self.cache_registry)
            }
        }
        
        # 메모리 트렌드 (최근 10개 스냅샷)
        if len(self.memory_history) >= 2:
            recent_snapshots = self.memory_history[-10:]
            memory_trend = [s.used_percent for s in recent_snapshots]
            
            stats['trend'] = {
                'recent_usage': memory_trend,
                'avg_usage': sum(memory_trend) / len(memory_trend),
                'trend_direction': 'increasing' if memory_trend[-1] > memory_trend[0] else 'decreasing'
            }
        
        return stats
    
    def get_optimization_suggestions(self) -> List[str]:
        """메모리 최적화 제안 반환"""
        stats = self.get_memory_stats()
        suggestions = []
        
        current_usage = stats['current']['used_percent']
        
        if current_usage > 90:
            suggestions.append("메모리 사용률이 매우 높습니다. 즉시 메모리 최적화를 수행하세요.")
        elif current_usage > 80:
            suggestions.append("메모리 사용률이 높습니다. 불필요한 캐시나 객체를 정리하세요.")
        
        # GC 통계 기반 제안
        gc_counts = stats['gc_info']['counts']
        if gc_counts[2] > gc_counts[1] * 2:
            suggestions.append("Generation 2 GC가 자주 발생합니다. 장기 참조 객체를 확인하세요.")
        
        # 대용량 객체 제안
        if stats['tracking']['large_objects'] > 10:
            suggestions.append("추적 중인 대용량 객체가 많습니다. 메모리 사용 패턴을 검토하세요.")
        
        # 트렌드 기반 제안
        if 'trend' in stats and stats['trend']['trend_direction'] == 'increasing':
            suggestions.append("메모리 사용량이 지속적으로 증가 중입니다. 메모리 누수 가능성을 확인하세요.")
        
        if not suggestions:
            suggestions.append("메모리 상태가 양호합니다.")
        
        return suggestions

class MemoryContext:
    """메모리 최적화 컨텍스트 매니저"""
    
    def __init__(self, optimizer: MemoryOptimizer, auto_optimize: bool = True):
        self.optimizer = optimizer
        self.auto_optimize = auto_optimize
        self.start_snapshot = None
        self.tracked_objects = []
    
    def __enter__(self):
        self.start_snapshot = self.optimizer._take_memory_snapshot()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_snapshot = self.optimizer._take_memory_snapshot()
        
        # 메모리 사용량이 크게 증가했거나 사용률이 높으면 최적화
        memory_increase = end_snapshot.used_mb - self.start_snapshot.used_mb
        
        if self.auto_optimize and (memory_increase > 100 or end_snapshot.used_percent > 85):
            self.optimizer.optimize_memory()
        
        return False
    
    def track_object(self, obj: Any, name: str = None):
        """컨텍스트 내에서 객체 추적"""
        self.optimizer.track_large_object(obj, name)
        self.tracked_objects.append(obj)

# 전역 인스턴스
_memory_optimizer = None

def get_memory_optimizer() -> MemoryOptimizer:
    """메모리 최적화 인스턴스 반환 (싱글톤)"""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer

def optimize_memory():
    """편의 함수: 메모리 최적화 수행"""
    optimizer = get_memory_optimizer()
    return optimizer.optimize_memory()

def memory_context(auto_optimize: bool = True):
    """편의 함수: 메모리 컨텍스트 생성"""
    optimizer = get_memory_optimizer()
    return MemoryContext(optimizer, auto_optimize)

if __name__ == "__main__":
    # 메모리 최적화 시스템 테스트
    optimizer = get_memory_optimizer()
    
    print("메모리 최적화 시스템 테스트 시작...")
    
    # 현재 메모리 상태
    stats = optimizer.get_memory_stats()
    print(f"현재 메모리 사용률: {stats['current']['used_percent']:.1f}%")
    
    # 최적화 제안
    suggestions = optimizer.get_optimization_suggestions()
    print("최적화 제안:")
    for suggestion in suggestions:
        print(f"  - {suggestion}")
    
    # 메모리 컨텍스트 테스트
    with memory_context() as ctx:
        # 대용량 배열 생성 (테스트용)
        test_array = np.random.rand(1000, 1000) * 100
        ctx.track_object(test_array, "test_large_array")
        print("대용량 테스트 객체 생성 및 추적")
    
    print("메모리 최적화 테스트 완료")