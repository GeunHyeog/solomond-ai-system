"""
🧠 솔로몬드 AI v2.1.2 - 메모리 최적화 엔진
스마트 메모리 관리 및 대용량 파일 처리 최적화

주요 기능:
- 적응형 메모리 관리
- 대용량 파일 스트리밍 처리
- 지능형 캐시 시스템
- 메모리 누수 방지 및 정리
- 실시간 메모리 모니터링
"""

import gc
import os
import sys
import mmap
import tempfile
import weakref
import threading
import time
import psutil
from typing import Dict, List, Any, Optional, Iterator, Union, Callable
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
import json
import hashlib
from io import BytesIO
import functools

@dataclass
class MemoryStats:
    """메모리 통계 정보"""
    total_mb: float
    used_mb: float
    available_mb: float
    percent: float
    cached_mb: float
    gc_collections: int
    objects_tracked: int

@dataclass
class CacheEntry:
    """캐시 엔트리"""
    key: str
    value: Any
    size_mb: float
    access_count: int
    last_accessed: float
    created_at: float

class LRUCache:
    """메모리 효율적인 LRU 캐시"""
    
    def __init__(self, max_size_mb: float = 100.0, max_items: int = 1000):
        self.max_size_mb = max_size_mb
        self.max_items = max_items
        self.cache = OrderedDict()
        self.current_size_mb = 0.0
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 가져오기"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.access_count += 1
                entry.last_accessed = time.time()
                # LRU: 최근 접근한 항목을 맨 뒤로 이동
                self.cache.move_to_end(key)
                self.hits += 1
                return entry.value
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any, size_mb: float = None) -> bool:
        """캐시에 값 저장"""
        if size_mb is None:
            size_mb = self._estimate_size(value)
        
        # 너무 큰 객체는 캐시하지 않음
        if size_mb > self.max_size_mb * 0.5:
            return False
        
        with self.lock:
            current_time = time.time()
            
            # 기존 키가 있으면 제거
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_size_mb -= old_entry.size_mb
                del self.cache[key]
            
            # 공간 확보
            while (len(self.cache) >= self.max_items or 
                   self.current_size_mb + size_mb > self.max_size_mb):
                if not self.cache:
                    break
                self._evict_lru()
            
            # 새 엔트리 추가
            entry = CacheEntry(
                key=key,
                value=value,
                size_mb=size_mb,
                access_count=1,
                last_accessed=current_time,
                created_at=current_time
            )
            
            self.cache[key] = entry
            self.current_size_mb += size_mb
            return True
    
    def _evict_lru(self):
        """가장 오래된 항목 제거"""
        if self.cache:
            key, entry = self.cache.popitem(last=False)
            self.current_size_mb -= entry.size_mb
    
    def _estimate_size(self, obj: Any) -> float:
        """객체 크기 추정 (MB)"""
        try:
            if hasattr(obj, '__len__'):
                # 리스트, 딕셔너리 등
                return sys.getsizeof(obj) / (1024 * 1024)
            elif isinstance(obj, (str, bytes)):
                return len(obj) / (1024 * 1024)
            else:
                return sys.getsizeof(obj) / (1024 * 1024)
        except:
            return 1.0  # 기본값
    
    def clear(self):
        """캐시 전체 삭제"""
        with self.lock:
            self.cache.clear()
            self.current_size_mb = 0.0
    
    def stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "items": len(self.cache),
            "size_mb": round(self.current_size_mb, 2),
            "max_size_mb": self.max_size_mb,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "utilization": round(self.current_size_mb / self.max_size_mb * 100, 2)
        }

class StreamingFileProcessor:
    """대용량 파일 스트리밍 처리기"""
    
    def __init__(self, chunk_size_mb: float = 10.0):
        self.chunk_size = int(chunk_size_mb * 1024 * 1024)
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def open_large_file(self, filepath: str, mode: str = 'rb'):
        """대용량 파일 안전 열기"""
        file_size = os.path.getsize(filepath)
        
        if file_size > 100 * 1024 * 1024:  # 100MB 이상
            self.logger.info(f"📁 대용량 파일 감지: {file_size / (1024*1024):.1f}MB")
            
            # 메모리 매핑 사용
            with open(filepath, mode) as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    yield mm
        else:
            # 일반 파일 열기
            with open(filepath, mode) as f:
                yield f
    
    def process_file_chunks(self, filepath: str, processor_func: Callable) -> Iterator[Any]:
        """파일을 청크 단위로 처리"""
        with self.open_large_file(filepath, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                
                yield processor_func(chunk)
                
                # 중간에 메모리 정리
                if gc.get_count()[0] > 1000:
                    gc.collect()
    
    def stream_text_file(self, filepath: str, encoding: str = 'utf-8') -> Iterator[str]:
        """텍스트 파일 라인별 스트리밍"""
        try:
            with open(filepath, 'r', encoding=encoding, buffering=8192) as f:
                for line in f:
                    yield line.strip()
        except UnicodeDecodeError:
            # 인코딩 오류 시 바이트로 처리
            with open(filepath, 'rb') as f:
                for line in f:
                    try:
                        yield line.decode('utf-8', errors='ignore').strip()
                    except:
                        continue
    
    def batch_process_lines(self, lines: Iterator[str], batch_size: int = 1000) -> Iterator[List[str]]:
        """라인들을 배치 단위로 그룹화"""
        batch = []
        for line in lines:
            batch.append(line)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:  # 마지막 배치
            yield batch

class MemoryManager:
    """통합 메모리 관리자"""
    
    def __init__(self, 
                 cache_size_mb: float = 200.0,
                 gc_threshold: float = 80.0,
                 emergency_threshold: float = 95.0):
        self.cache = LRUCache(max_size_mb=cache_size_mb)
        self.streaming_processor = StreamingFileProcessor()
        self.gc_threshold = gc_threshold
        self.emergency_threshold = emergency_threshold
        
        # 메모리 추적
        self.memory_snapshots = []
        self.object_refs = weakref.WeakSet()
        self.temp_files = []
        
        # 통계
        self.total_cleanups = 0
        self.bytes_freed = 0
        
        self.logger = logging.getLogger(__name__)
        
        # 백그라운드 정리 스레드
        self.cleanup_thread = None
        self.should_stop = False
        self.start_background_cleanup()
    
    def start_background_cleanup(self):
        """백그라운드 메모리 정리 시작"""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            return
        
        self.should_stop = False
        self.cleanup_thread = threading.Thread(
            target=self._background_cleanup_loop,
            daemon=True
        )
        self.cleanup_thread.start()
        self.logger.info("🧹 백그라운드 메모리 정리 시작")
    
    def stop_background_cleanup(self):
        """백그라운드 정리 중지"""
        self.should_stop = True
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
    
    def _background_cleanup_loop(self):
        """백그라운드 정리 루프"""
        while not self.should_stop:
            try:
                memory_percent = self.get_memory_usage().percent
                
                if memory_percent > self.emergency_threshold:
                    self.emergency_cleanup()
                elif memory_percent > self.gc_threshold:
                    self.routine_cleanup()
                
                time.sleep(30)  # 30초마다 확인
                
            except Exception as e:
                self.logger.error(f"백그라운드 정리 오류: {e}")
                time.sleep(60)
    
    def get_memory_usage(self) -> MemoryStats:
        """현재 메모리 사용량"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return MemoryStats(
            total_mb=memory.total / (1024 * 1024),
            used_mb=memory.used / (1024 * 1024),
            available_mb=memory.available / (1024 * 1024),
            percent=memory.percent,
            cached_mb=self.cache.current_size_mb,
            gc_collections=sum(gc.get_stats()[i]['collections'] for i in range(3)),
            objects_tracked=len(gc.get_objects())
        )
    
    def routine_cleanup(self) -> Dict[str, Any]:
        """일반 메모리 정리"""
        before_stats = self.get_memory_usage()
        
        # 1. 가비지 컬렉션
        collected_objects = gc.collect()
        
        # 2. 캐시 정리 (50% 비우기)
        cache_items_before = len(self.cache.cache)
        items_to_remove = cache_items_before // 2
        for _ in range(items_to_remove):
            if self.cache.cache:
                self.cache._evict_lru()
        
        # 3. 임시 파일 정리
        self._cleanup_temp_files()
        
        after_stats = self.get_memory_usage()
        freed_mb = before_stats.used_mb - after_stats.used_mb
        
        self.total_cleanups += 1
        self.bytes_freed += max(0, freed_mb)
        
        result = {
            "type": "routine",
            "freed_mb": round(freed_mb, 2),
            "objects_collected": collected_objects,
            "cache_items_removed": cache_items_before - len(self.cache.cache),
            "memory_before": round(before_stats.percent, 1),
            "memory_after": round(after_stats.percent, 1)
        }
        
        self.logger.info(f"🧽 일반 정리 완료: {freed_mb:.1f}MB 해제")
        return result
    
    def emergency_cleanup(self) -> Dict[str, Any]:
        """긴급 메모리 정리"""
        self.logger.warning("🚨 긴급 메모리 정리 시작")
        before_stats = self.get_memory_usage()
        
        # 1. 캐시 전체 삭제
        cache_size_before = self.cache.current_size_mb
        self.cache.clear()
        
        # 2. 강제 가비지 컬렉션 (여러 번 실행)
        total_collected = 0
        for _ in range(3):
            total_collected += gc.collect()
            time.sleep(0.1)
        
        # 3. 임시 파일 모두 삭제
        temp_files_removed = self._cleanup_temp_files(force=True)
        
        # 4. 약한 참조 정리
        weak_refs_before = len(self.object_refs)
        self.object_refs.clear()
        
        after_stats = self.get_memory_usage()
        freed_mb = before_stats.used_mb - after_stats.used_mb
        
        result = {
            "type": "emergency",
            "freed_mb": round(freed_mb, 2),
            "objects_collected": total_collected,
            "cache_cleared_mb": round(cache_size_before, 2),
            "temp_files_removed": temp_files_removed,
            "weak_refs_cleared": weak_refs_before,
            "memory_before": round(before_stats.percent, 1),
            "memory_after": round(after_stats.percent, 1)
        }
        
        self.logger.warning(f"🚨 긴급 정리 완료: {freed_mb:.1f}MB 해제")
        return result
    
    def _cleanup_temp_files(self, force: bool = False) -> int:
        """임시 파일 정리"""
        removed_count = 0
        files_to_remove = []
        
        for temp_file in self.temp_files[:]:
            try:
                if os.path.exists(temp_file):
                    # 파일이 너무 오래되었거나 force 모드
                    if force or (time.time() - os.path.getctime(temp_file)) > 3600:  # 1시간
                        os.remove(temp_file)
                        files_to_remove.append(temp_file)
                        removed_count += 1
            except Exception as e:
                self.logger.debug(f"임시 파일 삭제 실패: {temp_file} - {e}")
        
        # 리스트에서 제거
        for temp_file in files_to_remove:
            self.temp_files.remove(temp_file)
        
        return removed_count
    
    @contextmanager
    def temporary_file(self, suffix: str = '.tmp', prefix: str = 'solomond_'):
        """임시 파일 컨텍스트 매니저"""
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=suffix, 
                prefix=prefix
            ) as f:
                temp_file = f.name
                self.temp_files.append(temp_file)
                yield temp_file
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    if temp_file in self.temp_files:
                        self.temp_files.remove(temp_file)
                except:
                    pass
    
    def track_object(self, obj: Any) -> Any:
        """객체 추적 (메모리 누수 방지)"""
        self.object_refs.add(obj)
        return obj
    
    def smart_cache(self, key: str, factory_func: Callable, 
                   size_hint_mb: float = None, ttl_seconds: float = 3600) -> Any:
        """스마트 캐싱"""
        # 캐시에서 확인
        cached_value = self.cache.get(key)
        if cached_value is not None:
            return cached_value
        
        # 메모리 상태 확인
        memory_stats = self.get_memory_usage()
        if memory_stats.percent > self.gc_threshold:
            # 메모리 부족 시 캐싱 제한
            self.logger.warning("메모리 부족으로 캐싱 제한")
            return factory_func()
        
        # 새 값 생성 및 캐싱
        value = factory_func()
        
        if size_hint_mb is None:
            size_hint_mb = self._estimate_object_size(value)
        
        # TTL 고려하여 캐싱
        success = self.cache.put(key, value, size_hint_mb)
        if not success:
            self.logger.debug(f"캐싱 실패: {key}")
        
        return value
    
    def _estimate_object_size(self, obj: Any) -> float:
        """객체 크기 추정"""
        try:
            # 직렬화하여 정확한 크기 측정
            pickled = pickle.dumps(obj)
            return len(pickled) / (1024 * 1024)
        except:
            # 직렬화 실패 시 sys.getsizeof 사용
            return sys.getsizeof(obj) / (1024 * 1024)
    
    def optimize_for_large_file(self, filepath: str) -> Dict[str, Any]:
        """대용량 파일 처리 최적화"""
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        memory_stats = self.get_memory_usage()
        
        recommendations = []
        
        # 파일 크기에 따른 권장사항
        if file_size_mb > memory_stats.available_mb * 0.5:
            recommendations.append("스트리밍 처리 필수")
            chunk_size_mb = max(1.0, memory_stats.available_mb * 0.1)
        elif file_size_mb > 100:
            recommendations.append("청크 기반 처리 권장")
            chunk_size_mb = 10.0
        else:
            recommendations.append("일반 처리 가능")
            chunk_size_mb = file_size_mb
        
        # 메모리 정리 필요성
        if memory_stats.percent > 70:
            recommendations.append("처리 전 메모리 정리 필요")
            self.routine_cleanup()
        
        return {
            "file_size_mb": round(file_size_mb, 2),
            "available_memory_mb": round(memory_stats.available_mb, 2),
            "recommended_chunk_size_mb": chunk_size_mb,
            "recommendations": recommendations,
            "use_streaming": file_size_mb > memory_stats.available_mb * 0.3
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """최적화 리포트"""
        memory_stats = self.get_memory_usage()
        cache_stats = self.cache.stats()
        
        return {
            "memory": {
                "current_usage_percent": round(memory_stats.percent, 1),
                "available_mb": round(memory_stats.available_mb, 1),
                "cached_mb": round(memory_stats.cached_mb, 1)
            },
            "cache": cache_stats,
            "cleanup_stats": {
                "total_cleanups": self.total_cleanups,
                "bytes_freed_mb": round(self.bytes_freed, 1),
                "temp_files_active": len(self.temp_files),
                "objects_tracked": len(self.object_refs)
            },
            "performance_tips": self._generate_performance_tips(memory_stats, cache_stats)
        }
    
    def _generate_performance_tips(self, memory_stats: MemoryStats, cache_stats: Dict) -> List[str]:
        """성능 팁 생성"""
        tips = []
        
        if memory_stats.percent > 85:
            tips.append("🚨 메모리 사용률 높음 - 긴급 정리 권장")
        elif memory_stats.percent > 70:
            tips.append("⚠️ 메모리 사용률 주의 - 정리 권장")
        
        if cache_stats["hit_rate"] < 50:
            tips.append("📊 캐시 효율성 낮음 - 캐시 크기 조정 고려")
        
        if cache_stats["utilization"] > 90:
            tips.append("💾 캐시 거의 가득참 - 캐시 크기 증가 고려")
        
        if memory_stats.objects_tracked > 10000:
            tips.append("🧹 추적 객체 많음 - 가비지 컬렉션 권장")
        
        if not tips:
            tips.append("✅ 메모리 상태 양호")
        
        return tips

# 전역 메모리 매니저
global_memory_manager = MemoryManager()

def memory_optimized(cache_key: str = None, ttl: float = 3600):
    """메모리 최적화 데코레이터"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 캐시 키 생성
            if cache_key:
                key = cache_key
            else:
                key = f"{func.__module__}.{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            return global_memory_manager.smart_cache(
                key=key,
                factory_func=lambda: func(*args, **kwargs),
                ttl_seconds=ttl
            )
        return wrapper
    return decorator

if __name__ == "__main__":
    # 테스트 실행
    print("🧠 솔로몬드 AI 메모리 최적화 엔진 v2.1.2")
    print("=" * 50)
    
    manager = MemoryManager(cache_size_mb=50.0)
    
    # 현재 메모리 상태
    stats = manager.get_memory_usage()
    print(f"💾 현재 메모리 사용률: {stats.percent:.1f}%")
    print(f"📊 사용 가능 메모리: {stats.available_mb:.1f}MB")
    
    # 캐시 테스트
    @memory_optimized(cache_key="test_function")
    def expensive_computation(n):
        return sum(i**2 for i in range(n))
    
    print("\n🔄 캐시 테스트...")
    start_time = time.time()
    result1 = expensive_computation(100000)
    first_time = time.time() - start_time
    
    start_time = time.time()
    result2 = expensive_computation(100000)  # 캐시된 결과
    second_time = time.time() - start_time
    
    print(f"첫 번째 실행: {first_time:.4f}초")
    print(f"두 번째 실행 (캐시): {second_time:.4f}초")
    print(f"속도 향상: {first_time/second_time:.1f}배")
    
    # 대용량 파일 시뮬레이션
    print("\n📁 대용량 파일 처리 시뮬레이션...")
    with manager.temporary_file(suffix='.txt') as temp_file:
        # 10MB 테스트 파일 생성
        with open(temp_file, 'w', encoding='utf-8') as f:
            for i in range(100000):
                f.write(f"이것은 테스트 라인 {i:06d} 입니다. " * 10 + "\n")
        
        # 최적화 분석
        optimization = manager.optimize_for_large_file(temp_file)
        print(f"파일 크기: {optimization['file_size_mb']:.1f}MB")
        print(f"권장 청크 크기: {optimization['recommended_chunk_size_mb']:.1f}MB")
        print("권장사항:")
        for rec in optimization['recommendations']:
            print(f"  - {rec}")
        
        # 스트리밍 처리 테스트
        if optimization['use_streaming']:
            print("\n🌊 스트리밍 처리 테스트...")
            line_count = 0
            for batch in manager.streaming_processor.batch_process_lines(
                manager.streaming_processor.stream_text_file(temp_file),
                batch_size=1000
            ):
                line_count += len(batch)
            print(f"총 {line_count:,}개 라인 처리 완료")
    
    # 메모리 정리 테스트
    print("\n🧹 메모리 정리 테스트...")
    cleanup_result = manager.routine_cleanup()
    print(f"해제된 메모리: {cleanup_result['freed_mb']:.2f}MB")
    print(f"수집된 객체: {cleanup_result['objects_collected']}개")
    
    # 최적화 리포트
    print("\n📊 최적화 리포트:")
    report = manager.get_optimization_report()
    print(f"메모리 사용률: {report['memory']['current_usage_percent']}%")
    print(f"캐시 적중률: {report['cache']['hit_rate']}%")
    print("성능 팁:")
    for tip in report['performance_tips']:
        print(f"  {tip}")
    
    # 정리
    manager.stop_background_cleanup()
    print("\n✅ 메모리 최적화 엔진 테스트 완료!")
