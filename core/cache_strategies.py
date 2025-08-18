#!/usr/bin/env python3
"""
캐시 전략 및 알고리즘 v2.6
LRU, LFU, TTL, Adaptive 등 다양한 캐시 전략 구현
"""

import time
import heapq
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from collections import OrderedDict, defaultdict
import logging

@dataclass
class CacheItem:
    """캐시 아이템"""
    key: str
    data: Any
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float]
    priority: float = 1.0
    metadata: Dict[str, Any] = None

class CacheStrategy(ABC):
    """캐시 전략 추상 클래스"""
    
    def __init__(self, max_size_mb: float = 512.0):
        self.max_size_mb = max_size_mb
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.current_size_bytes = 0
        self.lock = threading.RLock()
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @abstractmethod
    def put(self, item: CacheItem) -> bool:
        """아이템 추가"""
        pass
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheItem]:
        """아이템 조회"""
        pass
    
    @abstractmethod
    def remove(self, key: str) -> bool:
        """아이템 제거"""
        pass
    
    @abstractmethod
    def evict(self, needed_space: int = 0) -> List[str]:
        """아이템 축출"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """전체 정리"""
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        """모든 키 반환"""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """현재 크기 반환"""
        pass
    
    def is_expired(self, item: CacheItem) -> bool:
        """만료 여부 확인"""
        if item.ttl is None:
            return False
        return (time.time() - item.created_at) > item.ttl
    
    def has_space(self, size_bytes: int) -> bool:
        """공간 여유 확인"""
        return (self.current_size_bytes + size_bytes) <= self.max_size_bytes

class LRUCacheStrategy(CacheStrategy):
    """LRU (Least Recently Used) 캐시 전략"""
    
    def __init__(self, max_size_mb: float = 512.0):
        super().__init__(max_size_mb)
        self.cache = OrderedDict()  # 순서 보장
        self.items = {}  # key -> CacheItem 매핑
    
    def put(self, item: CacheItem) -> bool:
        """아이템 추가"""
        with self.lock:
            try:
                # 기존 아이템이 있으면 제거
                if item.key in self.items:
                    old_item = self.items[item.key]
                    self.current_size_bytes -= old_item.size_bytes
                    del self.cache[item.key]
                
                # 공간 확보
                if not self.has_space(item.size_bytes):
                    needed_space = item.size_bytes - (self.max_size_bytes - self.current_size_bytes)
                    evicted = self.evict(needed_space)
                    if not self.has_space(item.size_bytes):
                        self.logger.warning(f"⚠️ LRU 캐시 공간 부족: {item.key}")
                        return False
                
                # 아이템 추가
                self.cache[item.key] = True  # 순서 추적용
                self.items[item.key] = item
                self.current_size_bytes += item.size_bytes
                
                self.logger.debug(f"✅ LRU 추가: {item.key} ({item.size_bytes} bytes)")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ LRU 추가 실패: {e}")
                return False
    
    def get(self, key: str) -> Optional[CacheItem]:
        """아이템 조회"""
        with self.lock:
            if key not in self.items:
                return None
            
            item = self.items[key]
            
            # 만료 확인
            if self.is_expired(item):
                self.remove(key)
                return None
            
            # 접근 정보 업데이트
            item.last_accessed = time.time()
            item.access_count += 1
            
            # LRU 순서 업데이트 (맨 뒤로 이동)
            self.cache.move_to_end(key)
            
            self.logger.debug(f"🎯 LRU 히트: {key}")
            return item
    
    def remove(self, key: str) -> bool:
        """아이템 제거"""
        with self.lock:
            if key not in self.items:
                return False
            
            item = self.items[key]
            self.current_size_bytes -= item.size_bytes
            
            del self.cache[key]
            del self.items[key]
            
            self.logger.debug(f"🗑️ LRU 제거: {key}")
            return True
    
    def evict(self, needed_space: int = 0) -> List[str]:
        """아이템 축출 (가장 오래된 것부터)"""
        evicted_keys = []
        freed_space = 0
        
        with self.lock:
            while (needed_space == 0 and len(self.cache) > 0) or \
                  (needed_space > 0 and freed_space < needed_space and len(self.cache) > 0):
                
                # 가장 오래된 키 가져오기
                key = next(iter(self.cache))
                item = self.items[key]
                
                freed_space += item.size_bytes
                evicted_keys.append(key)
                
                self.remove(key)
                
                if needed_space == 0:  # 단일 축출
                    break
            
            self.logger.info(f"🔄 LRU 축출: {len(evicted_keys)}개 ({freed_space} bytes)")
            return evicted_keys
    
    def clear(self) -> None:
        """전체 정리"""
        with self.lock:
            self.cache.clear()
            self.items.clear()
            self.current_size_bytes = 0
            self.logger.info("🧹 LRU 전체 정리 완료")
    
    def keys(self) -> List[str]:
        """모든 키 반환"""
        with self.lock:
            return list(self.items.keys())
    
    def size(self) -> int:
        """현재 크기 반환"""
        return self.current_size_bytes

class LFUCacheStrategy(CacheStrategy):
    """LFU (Least Frequently Used) 캐시 전략"""
    
    def __init__(self, max_size_mb: float = 512.0):
        super().__init__(max_size_mb)
        self.items = {}  # key -> CacheItem
        self.frequencies = defaultdict(int)  # key -> 접근 빈도
        self.freq_to_keys = defaultdict(set)  # 빈도 -> 키 집합
        self.min_frequency = 0
    
    def put(self, item: CacheItem) -> bool:
        """아이템 추가"""
        with self.lock:
            try:
                # 기존 아이템이 있으면 업데이트
                if item.key in self.items:
                    old_item = self.items[item.key]
                    self.current_size_bytes -= old_item.size_bytes
                    
                    # 빈도 정보 업데이트
                    old_freq = self.frequencies[item.key]
                    self.freq_to_keys[old_freq].discard(item.key)
                    
                    item.access_count = old_item.access_count + 1
                    new_freq = item.access_count
                    self.frequencies[item.key] = new_freq
                    self.freq_to_keys[new_freq].add(item.key)
                else:
                    # 새 아이템
                    if not self.has_space(item.size_bytes):
                        needed_space = item.size_bytes - (self.max_size_bytes - self.current_size_bytes)
                        evicted = self.evict(needed_space)
                        if not self.has_space(item.size_bytes):
                            self.logger.warning(f"⚠️ LFU 캐시 공간 부족: {item.key}")
                            return False
                    
                    # 빈도 1로 시작
                    self.frequencies[item.key] = 1
                    self.freq_to_keys[1].add(item.key)
                    self.min_frequency = 1
                
                self.items[item.key] = item
                self.current_size_bytes += item.size_bytes
                
                self.logger.debug(f"✅ LFU 추가: {item.key} (빈도: {self.frequencies[item.key]})")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ LFU 추가 실패: {e}")
                return False
    
    def get(self, key: str) -> Optional[CacheItem]:
        """아이템 조회"""
        with self.lock:
            if key not in self.items:
                return None
            
            item = self.items[key]
            
            # 만료 확인
            if self.is_expired(item):
                self.remove(key)
                return None
            
            # 접근 정보 업데이트
            item.last_accessed = time.time()
            item.access_count += 1
            
            # 빈도 업데이트
            old_freq = self.frequencies[key]
            new_freq = old_freq + 1
            
            self.freq_to_keys[old_freq].discard(key)
            self.freq_to_keys[new_freq].add(key)
            self.frequencies[key] = new_freq
            
            # 최소 빈도 업데이트
            if old_freq == self.min_frequency and len(self.freq_to_keys[old_freq]) == 0:
                self.min_frequency += 1
            
            self.logger.debug(f"🎯 LFU 히트: {key} (빈도: {new_freq})")
            return item
    
    def remove(self, key: str) -> bool:
        """아이템 제거"""
        with self.lock:
            if key not in self.items:
                return False
            
            item = self.items[key]
            freq = self.frequencies[key]
            
            self.current_size_bytes -= item.size_bytes
            
            # 빈도 정보 제거
            self.freq_to_keys[freq].discard(key)
            del self.frequencies[key]
            del self.items[key]
            
            # 최소 빈도 업데이트
            if freq == self.min_frequency and len(self.freq_to_keys[freq]) == 0:
                self.min_frequency += 1
            
            self.logger.debug(f"🗑️ LFU 제거: {key}")
            return True
    
    def evict(self, needed_space: int = 0) -> List[str]:
        """아이템 축출 (가장 적게 사용된 것부터)"""
        evicted_keys = []
        freed_space = 0
        
        with self.lock:
            while (needed_space == 0 and len(self.items) > 0) or \
                  (needed_space > 0 and freed_space < needed_space and len(self.items) > 0):
                
                # 최소 빈도의 키 중 하나 선택 (임의)
                if self.min_frequency in self.freq_to_keys and self.freq_to_keys[self.min_frequency]:
                    key = next(iter(self.freq_to_keys[self.min_frequency]))
                else:
                    # 빈도 정보 재계산
                    self._recalculate_min_frequency()
                    if self.min_frequency in self.freq_to_keys and self.freq_to_keys[self.min_frequency]:
                        key = next(iter(self.freq_to_keys[self.min_frequency]))
                    else:
                        break
                
                item = self.items[key]
                freed_space += item.size_bytes
                evicted_keys.append(key)
                
                self.remove(key)
                
                if needed_space == 0:  # 단일 축출
                    break
            
            self.logger.info(f"🔄 LFU 축출: {len(evicted_keys)}개 ({freed_space} bytes)")
            return evicted_keys
    
    def _recalculate_min_frequency(self) -> None:
        """최소 빈도 재계산"""
        if not self.frequencies:
            self.min_frequency = 0
            return
        
        self.min_frequency = min(self.frequencies.values())
    
    def clear(self) -> None:
        """전체 정리"""
        with self.lock:
            self.items.clear()
            self.frequencies.clear()
            self.freq_to_keys.clear()
            self.current_size_bytes = 0
            self.min_frequency = 0
            self.logger.info("🧹 LFU 전체 정리 완료")
    
    def keys(self) -> List[str]:
        """모든 키 반환"""
        with self.lock:
            return list(self.items.keys())
    
    def size(self) -> int:
        """현재 크기 반환"""
        return self.current_size_bytes

class TTLCacheStrategy(CacheStrategy):
    """TTL (Time To Live) 캐시 전략"""
    
    def __init__(self, max_size_mb: float = 512.0, default_ttl: float = 3600.0):
        super().__init__(max_size_mb)
        self.default_ttl = default_ttl
        self.items = {}  # key -> CacheItem
        self.expiry_heap = []  # (expiry_time, key) 힙
        self.last_cleanup = time.time()
        self.cleanup_interval = 60.0  # 1분마다 정리
    
    def put(self, item: CacheItem) -> bool:
        """아이템 추가"""
        with self.lock:
            try:
                # TTL 설정
                if item.ttl is None:
                    item.ttl = self.default_ttl
                
                expiry_time = item.created_at + item.ttl
                
                # 기존 아이템이 있으면 제거
                if item.key in self.items:
                    old_item = self.items[item.key]
                    self.current_size_bytes -= old_item.size_bytes
                
                # 공간 확보
                if not self.has_space(item.size_bytes):
                    self._cleanup_expired()
                    if not self.has_space(item.size_bytes):
                        needed_space = item.size_bytes - (self.max_size_bytes - self.current_size_bytes)
                        evicted = self.evict(needed_space)
                        if not self.has_space(item.size_bytes):
                            self.logger.warning(f"⚠️ TTL 캐시 공간 부족: {item.key}")
                            return False
                
                # 아이템 추가
                self.items[item.key] = item
                heapq.heappush(self.expiry_heap, (expiry_time, item.key))
                self.current_size_bytes += item.size_bytes
                
                self.logger.debug(f"✅ TTL 추가: {item.key} (만료: {datetime.fromtimestamp(expiry_time)})")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ TTL 추가 실패: {e}")
                return False
    
    def get(self, key: str) -> Optional[CacheItem]:
        """아이템 조회"""
        with self.lock:
            # 주기적 정리
            current_time = time.time()
            if current_time - self.last_cleanup > self.cleanup_interval:
                self._cleanup_expired()
                self.last_cleanup = current_time
            
            if key not in self.items:
                return None
            
            item = self.items[key]
            
            # 만료 확인
            if self.is_expired(item):
                self.remove(key)
                return None
            
            # 접근 정보 업데이트
            item.last_accessed = time.time()
            item.access_count += 1
            
            self.logger.debug(f"🎯 TTL 히트: {key}")
            return item
    
    def remove(self, key: str) -> bool:
        """아이템 제거"""
        with self.lock:
            if key not in self.items:
                return False
            
            item = self.items[key]
            self.current_size_bytes -= item.size_bytes
            del self.items[key]
            
            self.logger.debug(f"🗑️ TTL 제거: {key}")
            return True
    
    def evict(self, needed_space: int = 0) -> List[str]:
        """아이템 축출 (만료 시간 순)"""
        evicted_keys = []
        freed_space = 0
        
        with self.lock:
            # 먼저 만료된 항목들 정리
            self._cleanup_expired()
            
            # 추가 공간이 필요하면 만료 시간이 가까운 순으로 제거
            current_time = time.time()
            items_by_expiry = sorted(
                self.items.items(),
                key=lambda x: x[1].created_at + x[1].ttl
            )
            
            for key, item in items_by_expiry:
                if (needed_space == 0 and len(evicted_keys) > 0) or \
                   (needed_space > 0 and freed_space >= needed_space):
                    break
                
                freed_space += item.size_bytes
                evicted_keys.append(key)
                self.remove(key)
                
                if needed_space == 0:  # 단일 축출
                    break
            
            self.logger.info(f"🔄 TTL 축출: {len(evicted_keys)}개 ({freed_space} bytes)")
            return evicted_keys
    
    def _cleanup_expired(self) -> int:
        """만료된 아이템 정리"""
        current_time = time.time()
        removed_count = 0
        
        # 힙에서 만료된 아이템들 제거
        while self.expiry_heap:
            expiry_time, key = self.expiry_heap[0]
            
            if expiry_time > current_time:
                break  # 아직 만료되지 않음
            
            heapq.heappop(self.expiry_heap)
            
            if key in self.items:
                self.remove(key)
                removed_count += 1
        
        if removed_count > 0:
            self.logger.debug(f"🧹 TTL 만료 정리: {removed_count}개")
        
        return removed_count
    
    def clear(self) -> None:
        """전체 정리"""
        with self.lock:
            self.items.clear()
            self.expiry_heap.clear()
            self.current_size_bytes = 0
            self.logger.info("🧹 TTL 전체 정리 완료")
    
    def keys(self) -> List[str]:
        """모든 키 반환"""
        with self.lock:
            return list(self.items.keys())
    
    def size(self) -> int:
        """현재 크기 반환"""
        return self.current_size_bytes

class AdaptiveCacheStrategy(CacheStrategy):
    """적응형 캐시 전략 (LRU + LFU + TTL 조합)"""
    
    def __init__(self, max_size_mb: float = 512.0):
        super().__init__(max_size_mb)
        self.items = {}  # key -> CacheItem
        self.access_order = OrderedDict()  # LRU용
        self.frequencies = defaultdict(int)  # LFU용
        self.access_times = {}  # key -> 최근 접근 시간
        
        # 적응형 가중치 (동적 조정)
        self.lru_weight = 0.4
        self.lfu_weight = 0.4
        self.ttl_weight = 0.2
        
        # 성능 통계
        self.strategy_stats = {
            'lru_evictions': 0,
            'lfu_evictions': 0,
            'ttl_evictions': 0,
            'total_hits': 0,
            'total_misses': 0
        }
    
    def put(self, item: CacheItem) -> bool:
        """아이템 추가"""
        with self.lock:
            try:
                # 기존 아이템이 있으면 제거
                if item.key in self.items:
                    old_item = self.items[item.key]
                    self.current_size_bytes -= old_item.size_bytes
                
                # 공간 확보
                if not self.has_space(item.size_bytes):
                    needed_space = item.size_bytes - (self.max_size_bytes - self.current_size_bytes)
                    evicted = self.evict(needed_space)
                    if not self.has_space(item.size_bytes):
                        self.logger.warning(f"⚠️ 적응형 캐시 공간 부족: {item.key}")
                        return False
                
                # 아이템 추가
                self.items[item.key] = item
                self.access_order[item.key] = time.time()
                self.frequencies[item.key] = 1
                self.access_times[item.key] = time.time()
                self.current_size_bytes += item.size_bytes
                
                self.logger.debug(f"✅ 적응형 추가: {item.key}")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ 적응형 추가 실패: {e}")
                return False
    
    def get(self, key: str) -> Optional[CacheItem]:
        """아이템 조회"""
        with self.lock:
            if key not in self.items:
                self.strategy_stats['total_misses'] += 1
                return None
            
            item = self.items[key]
            
            # 만료 확인
            if self.is_expired(item):
                self.remove(key)
                self.strategy_stats['total_misses'] += 1
                return None
            
            # 접근 정보 업데이트
            current_time = time.time()
            item.last_accessed = current_time
            item.access_count += 1
            
            # 적응형 정보 업데이트
            self.access_order.move_to_end(key)  # LRU
            self.frequencies[key] += 1  # LFU
            self.access_times[key] = current_time
            
            self.strategy_stats['total_hits'] += 1
            
            # 성능에 따른 가중치 조정
            self._adjust_weights()
            
            self.logger.debug(f"🎯 적응형 히트: {key}")
            return item
    
    def remove(self, key: str) -> bool:
        """아이템 제거"""
        with self.lock:
            if key not in self.items:
                return False
            
            item = self.items[key]
            self.current_size_bytes -= item.size_bytes
            
            del self.items[key]
            if key in self.access_order:
                del self.access_order[key]
            if key in self.frequencies:
                del self.frequencies[key]
            if key in self.access_times:
                del self.access_times[key]
            
            self.logger.debug(f"🗑️ 적응형 제거: {key}")
            return True
    
    def evict(self, needed_space: int = 0) -> List[str]:
        """적응형 축출"""
        evicted_keys = []
        freed_space = 0
        
        with self.lock:
            # 먼저 만료된 항목들 정리
            expired_keys = self._get_expired_keys()
            for key in expired_keys:
                item = self.items[key]
                freed_space += item.size_bytes
                evicted_keys.append(key)
                self.remove(key)
                self.strategy_stats['ttl_evictions'] += 1
                
                if needed_space > 0 and freed_space >= needed_space:
                    break
            
            # 추가 공간이 필요하면 적응형 점수 기반 축출
            if (needed_space == 0 and len(evicted_keys) == 0) or \
               (needed_space > 0 and freed_space < needed_space):
                
                candidates = self._calculate_eviction_scores()
                
                for score, key in candidates:
                    if (needed_space == 0 and len(evicted_keys) > 0) or \
                       (needed_space > 0 and freed_space >= needed_space):
                        break
                    
                    item = self.items[key]
                    freed_space += item.size_bytes
                    evicted_keys.append(key)
                    
                    # 어떤 전략이 선택되었는지 기록
                    if score < 0.3:
                        self.strategy_stats['lru_evictions'] += 1
                    elif score < 0.6:
                        self.strategy_stats['lfu_evictions'] += 1
                    else:
                        self.strategy_stats['ttl_evictions'] += 1
                    
                    self.remove(key)
                    
                    if needed_space == 0:  # 단일 축출
                        break
            
            self.logger.info(f"🔄 적응형 축출: {len(evicted_keys)}개 ({freed_space} bytes)")
            return evicted_keys
    
    def _get_expired_keys(self) -> List[str]:
        """만료된 키 목록"""
        current_time = time.time()
        expired_keys = []
        
        for key, item in self.items.items():
            if self.is_expired(item):
                expired_keys.append(key)
        
        return expired_keys
    
    def _calculate_eviction_scores(self) -> List[Tuple[float, str]]:
        """축출 점수 계산"""
        current_time = time.time()
        scores = []
        
        for key, item in self.items.items():
            # LRU 점수 (최근 접근 시간, 0-1)
            time_since_access = current_time - item.last_accessed
            max_time = max(current_time - t for t in self.access_times.values()) or 1
            lru_score = time_since_access / max_time
            
            # LFU 점수 (접근 빈도, 0-1)
            max_freq = max(self.frequencies.values()) or 1
            lfu_score = 1.0 - (self.frequencies[key] / max_freq)
            
            # TTL 점수 (만료까지 남은 시간, 0-1)
            if item.ttl:
                time_to_expiry = (item.created_at + item.ttl) - current_time
                max_ttl = max(
                    (i.created_at + i.ttl) - current_time 
                    for i in self.items.values() 
                    if i.ttl
                ) or 1
                ttl_score = 1.0 - max(0, time_to_expiry) / max_ttl
            else:
                ttl_score = 0.0
            
            # 가중 평균 점수
            total_score = (
                self.lru_weight * lru_score +
                self.lfu_weight * lfu_score +
                self.ttl_weight * ttl_score
            )
            
            scores.append((total_score, key))
        
        # 점수가 높은 순으로 정렬 (높을수록 축출 우선)
        scores.sort(reverse=True)
        return scores
    
    def _adjust_weights(self) -> None:
        """성능에 따른 가중치 조정"""
        total_evictions = (
            self.strategy_stats['lru_evictions'] +
            self.strategy_stats['lfu_evictions'] +
            self.strategy_stats['ttl_evictions']
        )
        
        if total_evictions < 10:  # 충분한 데이터가 없으면 조정하지 않음
            return
        
        # 히트율 계산
        total_requests = self.strategy_stats['total_hits'] + self.strategy_stats['total_misses']
        if total_requests == 0:
            return
        
        hit_rate = self.strategy_stats['total_hits'] / total_requests
        
        # 히트율에 따른 가중치 조정
        if hit_rate < 0.5:  # 낮은 히트율 - TTL 중심으로
            self.ttl_weight = min(0.5, self.ttl_weight + 0.05)
            self.lru_weight = max(0.2, self.lru_weight - 0.025)
            self.lfu_weight = max(0.2, self.lfu_weight - 0.025)
        elif hit_rate > 0.8:  # 높은 히트율 - LRU/LFU 중심으로
            self.ttl_weight = max(0.1, self.ttl_weight - 0.05)
            self.lru_weight = min(0.5, self.lru_weight + 0.025)
            self.lfu_weight = min(0.5, self.lfu_weight + 0.025)
        
        # 가중치 정규화
        total_weight = self.lru_weight + self.lfu_weight + self.ttl_weight
        self.lru_weight /= total_weight
        self.lfu_weight /= total_weight
        self.ttl_weight /= total_weight
        
        self.logger.debug(f"🎯 가중치 조정: LRU={self.lru_weight:.2f}, LFU={self.lfu_weight:.2f}, TTL={self.ttl_weight:.2f}")
    
    def clear(self) -> None:
        """전체 정리"""
        with self.lock:
            self.items.clear()
            self.access_order.clear()
            self.frequencies.clear()
            self.access_times.clear()
            self.current_size_bytes = 0
            self.logger.info("🧹 적응형 전체 정리 완료")
    
    def keys(self) -> List[str]:
        """모든 키 반환"""
        with self.lock:
            return list(self.items.keys())
    
    def size(self) -> int:
        """현재 크기 반환"""
        return self.current_size_bytes
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """전략 통계 반환"""
        with self.lock:
            return {
                **self.strategy_stats.copy(),
                'weights': {
                    'lru_weight': self.lru_weight,
                    'lfu_weight': self.lfu_weight,
                    'ttl_weight': self.ttl_weight
                },
                'current_items': len(self.items),
                'current_size_mb': self.current_size_bytes / (1024**2)
            }

# 캐시 전략 팩토리
class CacheStrategyFactory:
    """캐시 전략 팩토리"""
    
    @staticmethod
    def create_strategy(strategy_type: str, max_size_mb: float = 512.0, **kwargs) -> CacheStrategy:
        """캐시 전략 생성"""
        strategy_type = strategy_type.lower()
        
        if strategy_type == 'lru':
            return LRUCacheStrategy(max_size_mb)
        elif strategy_type == 'lfu':
            return LFUCacheStrategy(max_size_mb)
        elif strategy_type == 'ttl':
            default_ttl = kwargs.get('default_ttl', 3600.0)
            return TTLCacheStrategy(max_size_mb, default_ttl)
        elif strategy_type == 'adaptive':
            return AdaptiveCacheStrategy(max_size_mb)
        else:
            raise ValueError(f"지원하지 않는 캐시 전략: {strategy_type}")

# 사용 예시
if __name__ == "__main__":
    # 테스트 데이터
    test_items = []
    for i in range(10):
        item = CacheItem(
            key=f"test_key_{i}",
            data=f"test_data_{i}" * 100,  # 약 1KB
            size_bytes=1024,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            ttl=60.0 if i % 2 == 0 else None  # 절반만 TTL 설정
        )
        test_items.append(item)
    
    # 각 전략 테스트
    strategies = ['lru', 'lfu', 'ttl', 'adaptive']
    
    for strategy_name in strategies:
        print(f"\n🧪 {strategy_name.upper()} 전략 테스트:")
        
        strategy = CacheStrategyFactory.create_strategy(strategy_name, max_size_mb=0.01)  # 10KB 제한
        
        # 아이템 추가
        for item in test_items:
            success = strategy.put(item)
            print(f"  추가 {item.key}: {'✅' if success else '❌'}")
        
        # 통계 출력
        print(f"  최종 크기: {strategy.size()} bytes")
        print(f"  저장된 키: {len(strategy.keys())}개")
        
        if hasattr(strategy, 'get_strategy_stats'):
            stats = strategy.get_strategy_stats()
            print(f"  전략 통계: {stats}")
    
    print("\n✅ 캐시 전략 테스트 완료")