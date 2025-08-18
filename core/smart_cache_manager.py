#!/usr/bin/env python3
"""
스마트 캐시 관리 시스템 v2.6
다중 레벨 캐시 및 지능적 메모리 연동 관리
"""

import os
import time
import json
import pickle
import hashlib
import threading
import weakref
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import OrderedDict, deque
import logging
import psutil
import gzip
import lz4.frame
from enum import Enum
import asyncio

class CacheLevel(Enum):
    """캐시 레벨"""
    L1_MEMORY = "l1_memory"
    L2_DISK = "l2_disk"
    L3_NETWORK = "l3_network"

class CacheStrategy(Enum):
    """캐시 전략"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # 적응형

class CompressionType(Enum):
    """압축 타입"""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    PICKLE = "pickle"

@dataclass
class CacheEntry:
    """캐시 엔트리"""
    key: str
    data: Any
    level: CacheLevel
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float]
    size_bytes: int
    compressed: bool
    compression_type: CompressionType
    metadata: Dict[str, Any]

@dataclass
class CacheStats:
    """캐시 통계"""
    total_entries: int
    total_size_mb: float
    hit_count: int
    miss_count: int
    hit_rate: float
    l1_entries: int
    l2_entries: int
    l3_entries: int
    eviction_count: int
    compression_ratio: float
    last_cleanup: str

@dataclass
class CacheConfiguration:
    """캐시 설정"""
    l1_max_size_mb: float = 512.0  # L1 최대 크기
    l2_max_size_mb: float = 2048.0  # L2 최대 크기
    l3_max_size_mb: float = 8192.0  # L3 최대 크기
    default_ttl_seconds: float = 3600.0  # 기본 TTL (1시간)
    cleanup_interval_seconds: float = 300.0  # 정리 간격 (5분)
    compression_threshold_kb: float = 100.0  # 압축 임계값 (100KB)
    memory_pressure_threshold: float = 80.0  # 메모리 압박 임계값 (80%)
    enable_preloading: bool = True
    enable_analytics: bool = True

class SmartCacheManager:
    """스마트 캐시 관리자"""
    
    def __init__(self, 
                 config: Optional[CacheConfiguration] = None,
                 cache_dir: str = "./cache"):
        
        self.config = config or CacheConfiguration()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.logger = self._setup_logging()
        
        # 캐시 저장소 (레벨별)
        self.l1_cache = OrderedDict()  # 메모리 캐시
        self.l2_index = {}  # 디스크 캐시 인덱스
        self.l3_index = {}  # 네트워크 캐시 인덱스
        
        # 캐시 통계
        self.stats = CacheStats(
            total_entries=0,
            total_size_mb=0.0,
            hit_count=0,
            miss_count=0,
            hit_rate=0.0,
            l1_entries=0,
            l2_entries=0,
            l3_entries=0,
            eviction_count=0,
            compression_ratio=1.0,
            last_cleanup=datetime.now().isoformat()
        )
        
        # 접근 빈도 추적 (LFU용)
        self.access_frequency = {}
        
        # 캐시 키 패턴 분석
        self.key_patterns = {}
        self.prediction_model = {}
        
        # 스레드 안전성
        self.lock = threading.RLock()
        
        # 자동 정리 스레드
        self.cleanup_thread = None
        self.is_running = False
        
        # 메모리 최적화 엔진 연동
        self.memory_optimizer = None
        self.memory_callbacks = []
        
        # 성능 분석
        self.performance_history = deque(maxlen=1000)
        
        # 압축 엔진
        self.compression_engines = {
            CompressionType.GZIP: self._gzip_compress,
            CompressionType.LZ4: self._lz4_compress,
            CompressionType.PICKLE: self._pickle_compress
        }
        
        self.decompression_engines = {
            CompressionType.GZIP: self._gzip_decompress,
            CompressionType.LZ4: self._lz4_decompress,
            CompressionType.PICKLE: self._pickle_decompress
        }
        
        # 디스크 캐시 초기화
        self._initialize_disk_cache()
        
        self.logger.info("🚀 스마트 캐시 관리자 초기화 완료")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.SmartCacheManager')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_disk_cache(self) -> None:
        """디스크 캐시 초기화"""
        try:
            # L2 디스크 캐시 디렉토리
            self.l2_dir = self.cache_dir / "l2_disk"
            self.l2_dir.mkdir(exist_ok=True)
            
            # L3 네트워크 캐시 디렉토리
            self.l3_dir = self.cache_dir / "l3_network"
            self.l3_dir.mkdir(exist_ok=True)
            
            # 기존 캐시 인덱스 로드
            self._load_cache_indexes()
            
        except Exception as e:
            self.logger.error(f"❌ 디스크 캐시 초기화 실패: {e}")
    
    def _load_cache_indexes(self) -> None:
        """캐시 인덱스 로드"""
        try:
            # L2 인덱스 로드
            l2_index_file = self.cache_dir / "l2_index.json"
            if l2_index_file.exists():
                with open(l2_index_file, 'r', encoding='utf-8') as f:
                    self.l2_index = json.load(f)
            
            # L3 인덱스 로드
            l3_index_file = self.cache_dir / "l3_index.json"
            if l3_index_file.exists():
                with open(l3_index_file, 'r', encoding='utf-8') as f:
                    self.l3_index = json.load(f)
            
            self.logger.info(f"📋 캐시 인덱스 로드: L2={len(self.l2_index)}, L3={len(self.l3_index)}")
            
        except Exception as e:
            self.logger.error(f"❌ 캐시 인덱스 로드 실패: {e}")
    
    def _save_cache_indexes(self) -> None:
        """캐시 인덱스 저장"""
        try:
            # L2 인덱스 저장
            l2_index_file = self.cache_dir / "l2_index.json"
            with open(l2_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.l2_index, f, ensure_ascii=False, indent=2)
            
            # L3 인덱스 저장
            l3_index_file = self.cache_dir / "l3_index.json"
            with open(l3_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.l3_index, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            self.logger.error(f"❌ 캐시 인덱스 저장 실패: {e}")
    
    def start_auto_cleanup(self) -> None:
        """자동 정리 시작"""
        if self.is_running:
            return
        
        self.is_running = True
        self.cleanup_thread = threading.Thread(
            target=self._auto_cleanup_loop,
            daemon=True
        )
        self.cleanup_thread.start()
        self.logger.info("🔄 자동 캐시 정리 시작")
    
    def stop_auto_cleanup(self) -> None:
        """자동 정리 중지"""
        self.is_running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        self.logger.info("⏹️ 자동 캐시 정리 중지")
    
    def _auto_cleanup_loop(self) -> None:
        """자동 정리 루프"""
        while self.is_running:
            try:
                time.sleep(self.config.cleanup_interval_seconds)
                if self.is_running:
                    self.cleanup_expired_entries()
                    self._check_memory_pressure()
                    self._update_statistics()
                    
            except Exception as e:
                self.logger.error(f"❌ 자동 정리 루프 오류: {e}")
    
    def set(self, 
            key: str, 
            data: Any, 
            level: CacheLevel = CacheLevel.L1_MEMORY,
            ttl: Optional[float] = None,
            strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """캐시 항목 설정"""
        
        start_time = time.time()
        
        try:
            with self.lock:
                # TTL 설정
                if ttl is None:
                    ttl = self.config.default_ttl_seconds
                
                # 데이터 압축 결정
                raw_size = self._calculate_size(data)
                should_compress = raw_size > (self.config.compression_threshold_kb * 1024)
                
                # 데이터 처리
                processed_data, compression_type, compressed_size = self._process_data_for_storage(
                    data, should_compress
                )
                
                # 캐시 엔트리 생성
                entry = CacheEntry(
                    key=key,
                    data=processed_data,
                    level=level,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    access_count=1,
                    ttl=ttl,
                    size_bytes=compressed_size,
                    compressed=should_compress,
                    compression_type=compression_type,
                    metadata=metadata or {}
                )
                
                # 레벨별 저장
                success = self._store_by_level(entry, strategy)
                
                if success:
                    # 통계 업데이트
                    self.stats.total_entries += 1
                    self.stats.total_size_mb += compressed_size / (1024**2)
                    
                    # 성능 기록
                    self.performance_history.append({
                        'operation': 'set',
                        'key': key,
                        'level': level.value,
                        'size_bytes': compressed_size,
                        'duration_ms': (time.time() - start_time) * 1000,
                        'timestamp': time.time()
                    })
                    
                    self.logger.debug(f"✅ 캐시 저장: {key} ({level.value}, {compressed_size}bytes)")
                    return True
                else:
                    self.logger.warning(f"⚠️ 캐시 저장 실패: {key}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ 캐시 저장 오류: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """캐시 항목 조회"""
        
        start_time = time.time()
        
        try:
            with self.lock:
                # L1 메모리 캐시에서 조회
                if key in self.l1_cache:
                    entry = self.l1_cache[key]
                    
                    # TTL 확인
                    if self._is_expired(entry):
                        self._remove_entry(key, CacheLevel.L1_MEMORY)
                        self.stats.miss_count += 1
                        return default
                    
                    # 접근 정보 업데이트
                    entry.last_accessed = time.time()
                    entry.access_count += 1
                    self.access_frequency[key] = self.access_frequency.get(key, 0) + 1
                    
                    # LRU 순서 업데이트
                    self.l1_cache.move_to_end(key)
                    
                    # 데이터 복원
                    data = self._restore_data_from_storage(entry)
                    
                    # 통계 업데이트
                    self.stats.hit_count += 1
                    self._update_hit_rate()
                    
                    # 성능 기록
                    self.performance_history.append({
                        'operation': 'get',
                        'key': key,
                        'level': 'l1_memory',
                        'hit': True,
                        'duration_ms': (time.time() - start_time) * 1000,
                        'timestamp': time.time()
                    })
                    
                    self.logger.debug(f"🎯 L1 캐시 히트: {key}")
                    return data
                
                # L2 디스크 캐시에서 조회
                if key in self.l2_index:
                    entry_info = self.l2_index[key]
                    
                    # TTL 확인
                    if time.time() - entry_info['created_at'] > entry_info['ttl']:
                        self._remove_entry(key, CacheLevel.L2_DISK)
                        self.stats.miss_count += 1
                        return default
                    
                    # 디스크에서 로드
                    data = self._load_from_disk(key, entry_info)
                    
                    if data is not None:
                        # L1으로 승격
                        self._promote_to_l1(key, data, entry_info)
                        
                        # 통계 업데이트
                        self.stats.hit_count += 1
                        self._update_hit_rate()
                        
                        # 성능 기록
                        self.performance_history.append({
                            'operation': 'get',
                            'key': key,
                            'level': 'l2_disk',
                            'hit': True,
                            'duration_ms': (time.time() - start_time) * 1000,
                            'timestamp': time.time()
                        })
                        
                        self.logger.debug(f"💾 L2 캐시 히트: {key}")
                        return data
                
                # L3 네트워크 캐시에서 조회 (향후 구현)
                # if key in self.l3_index:
                #     ...
                
                # 캐시 미스
                self.stats.miss_count += 1
                self._update_hit_rate()
                
                # 성능 기록
                self.performance_history.append({
                    'operation': 'get',
                    'key': key,
                    'level': 'miss',
                    'hit': False,
                    'duration_ms': (time.time() - start_time) * 1000,
                    'timestamp': time.time()
                })
                
                self.logger.debug(f"❌ 캐시 미스: {key}")
                return default
                
        except Exception as e:
            self.logger.error(f"❌ 캐시 조회 오류: {e}")
            return default
    
    def delete(self, key: str) -> bool:
        """캐시 항목 삭제"""
        try:
            with self.lock:
                removed = False
                
                # L1에서 삭제
                if key in self.l1_cache:
                    del self.l1_cache[key]
                    removed = True
                
                # L2에서 삭제
                if key in self.l2_index:
                    self._remove_from_disk(key)
                    del self.l2_index[key]
                    removed = True
                
                # L3에서 삭제 (향후 구현)
                if key in self.l3_index:
                    del self.l3_index[key]
                    removed = True
                
                # 접근 빈도에서 삭제
                if key in self.access_frequency:
                    del self.access_frequency[key]
                
                if removed:
                    self.stats.total_entries -= 1
                    self.logger.debug(f"🗑️ 캐시 삭제: {key}")
                
                return removed
                
        except Exception as e:
            self.logger.error(f"❌ 캐시 삭제 오류: {e}")
            return False
    
    def clear(self, level: Optional[CacheLevel] = None) -> None:
        """캐시 정리"""
        try:
            with self.lock:
                if level is None or level == CacheLevel.L1_MEMORY:
                    self.l1_cache.clear()
                    self.stats.l1_entries = 0
                
                if level is None or level == CacheLevel.L2_DISK:
                    self.l2_index.clear()
                    self.stats.l2_entries = 0
                    # 디스크 파일들 삭제
                    for file_path in self.l2_dir.glob("*.cache"):
                        file_path.unlink()
                
                if level is None or level == CacheLevel.L3_NETWORK:
                    self.l3_index.clear()
                    self.stats.l3_entries = 0
                
                if level is None:
                    self.access_frequency.clear()
                    self.stats.total_entries = 0
                    self.stats.total_size_mb = 0.0
                
                self.logger.info(f"🧹 캐시 정리 완료: {level.value if level else 'ALL'}")
                
        except Exception as e:
            self.logger.error(f"❌ 캐시 정리 오류: {e}")
    
    def cleanup_expired_entries(self) -> int:
        """만료된 항목 정리"""
        try:
            with self.lock:
                removed_count = 0
                current_time = time.time()
                
                # L1 만료 항목 정리
                expired_l1_keys = []
                for key, entry in self.l1_cache.items():
                    if self._is_expired(entry):
                        expired_l1_keys.append(key)
                
                for key in expired_l1_keys:
                    del self.l1_cache[key]
                    removed_count += 1
                
                # L2 만료 항목 정리
                expired_l2_keys = []
                for key, entry_info in self.l2_index.items():
                    if current_time - entry_info['created_at'] > entry_info['ttl']:
                        expired_l2_keys.append(key)
                
                for key in expired_l2_keys:
                    self._remove_from_disk(key)
                    del self.l2_index[key]
                    removed_count += 1
                
                if removed_count > 0:
                    self.stats.total_entries -= removed_count
                    self.stats.last_cleanup = datetime.now().isoformat()
                    self.logger.info(f"🧹 만료 항목 정리: {removed_count}개")
                
                return removed_count
                
        except Exception as e:
            self.logger.error(f"❌ 만료 항목 정리 오류: {e}")
            return 0
    
    def _process_data_for_storage(self, data: Any, should_compress: bool) -> Tuple[Any, CompressionType, int]:
        """저장용 데이터 처리"""
        if not should_compress:
            return data, CompressionType.NONE, self._calculate_size(data)
        
        # 최적 압축 방식 선택
        compression_type = self._select_compression_type(data)
        
        # 압축 실행
        compressed_data = self.compression_engines[compression_type](data)
        compressed_size = self._calculate_size(compressed_data)
        
        return compressed_data, compression_type, compressed_size
    
    def _restore_data_from_storage(self, entry: CacheEntry) -> Any:
        """저장된 데이터 복원"""
        if not entry.compressed:
            return entry.data
        
        return self.decompression_engines[entry.compression_type](entry.data)
    
    def _select_compression_type(self, data: Any) -> CompressionType:
        """최적 압축 타입 선택"""
        # 데이터 타입에 따른 압축 방식 선택
        if isinstance(data, (dict, list)):
            return CompressionType.LZ4  # JSON 데이터에 LZ4가 효율적
        elif isinstance(data, bytes):
            return CompressionType.GZIP  # 바이너리 데이터에 GZIP 효율적
        else:
            return CompressionType.PICKLE  # 기타 객체는 pickle + gzip
    
    def _gzip_compress(self, data: Any) -> bytes:
        """GZIP 압축"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        elif not isinstance(data, bytes):
            data = pickle.dumps(data)
        return gzip.compress(data)
    
    def _gzip_decompress(self, data: bytes) -> Any:
        """GZIP 압축 해제"""
        decompressed = gzip.decompress(data)
        try:
            return pickle.loads(decompressed)
        except:
            return decompressed.decode('utf-8')
    
    def _lz4_compress(self, data: Any) -> bytes:
        """LZ4 압축"""
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        return lz4.frame.compress(data)
    
    def _lz4_decompress(self, data: bytes) -> Any:
        """LZ4 압축 해제"""
        decompressed = lz4.frame.decompress(data)
        try:
            return json.loads(decompressed.decode('utf-8'))
        except:
            return decompressed
    
    def _pickle_compress(self, data: Any) -> bytes:
        """Pickle + GZIP 압축"""
        pickled = pickle.dumps(data)
        return gzip.compress(pickled)
    
    def _pickle_decompress(self, data: bytes) -> Any:
        """Pickle + GZIP 압축 해제"""
        decompressed = gzip.decompress(data)
        return pickle.loads(decompressed)
    
    def _calculate_size(self, data: Any) -> int:
        """데이터 크기 계산"""
        if isinstance(data, bytes):
            return len(data)
        elif isinstance(data, str):
            return len(data.encode('utf-8'))
        else:
            try:
                return len(pickle.dumps(data))
            except:
                return 1024  # 추정값
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """만료 여부 확인"""
        if entry.ttl is None:
            return False
        return (time.time() - entry.created_at) > entry.ttl
    
    def _store_by_level(self, entry: CacheEntry, strategy: CacheStrategy) -> bool:
        """레벨별 저장"""
        if entry.level == CacheLevel.L1_MEMORY:
            return self._store_to_l1(entry, strategy)
        elif entry.level == CacheLevel.L2_DISK:
            return self._store_to_l2(entry)
        elif entry.level == CacheLevel.L3_NETWORK:
            return self._store_to_l3(entry)
        return False
    
    def _store_to_l1(self, entry: CacheEntry, strategy: CacheStrategy) -> bool:
        """L1 메모리에 저장"""
        try:
            # 메모리 한계 확인
            current_size = sum(e.size_bytes for e in self.l1_cache.values())
            max_size = self.config.l1_max_size_mb * 1024 * 1024
            
            # 공간 확보 필요 시 정리
            if current_size + entry.size_bytes > max_size:
                self._evict_l1_entries(strategy, entry.size_bytes)
            
            # 저장
            self.l1_cache[entry.key] = entry
            self.stats.l1_entries = len(self.l1_cache)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ L1 저장 실패: {e}")
            return False
    
    def _store_to_l2(self, entry: CacheEntry) -> bool:
        """L2 디스크에 저장"""
        try:
            # 파일 경로
            file_path = self.l2_dir / f"{entry.key}.cache"
            
            # 데이터 저장
            with open(file_path, 'wb') as f:
                pickle.dump(entry, f)
            
            # 인덱스 업데이트
            self.l2_index[entry.key] = {
                'file_path': str(file_path),
                'created_at': entry.created_at,
                'ttl': entry.ttl,
                'size_bytes': entry.size_bytes,
                'compressed': entry.compressed,
                'compression_type': entry.compression_type.value
            }
            
            self.stats.l2_entries = len(self.l2_index)
            return True
            
        except Exception as e:
            self.logger.error(f"❌ L2 저장 실패: {e}")
            return False
    
    def _store_to_l3(self, entry: CacheEntry) -> bool:
        """L3 네트워크에 저장 (향후 구현)"""
        # 향후 클라우드 저장소 연동 구현
        return True
    
    def _evict_l1_entries(self, strategy: CacheStrategy, needed_space: int) -> None:
        """L1 엔트리 축출"""
        if strategy == CacheStrategy.LRU:
            self._evict_lru()
        elif strategy == CacheStrategy.LFU:
            self._evict_lfu()
        elif strategy == CacheStrategy.TTL:
            self._evict_expired()
        else:  # ADAPTIVE
            self._evict_adaptive(needed_space)
    
    def _evict_lru(self) -> None:
        """LRU 축출"""
        if self.l1_cache:
            key, _ = self.l1_cache.popitem(last=False)  # 가장 오래된 항목
            self.stats.eviction_count += 1
            self.logger.debug(f"🔄 LRU 축출: {key}")
    
    def _evict_lfu(self) -> None:
        """LFU 축출"""
        if self.l1_cache:
            # 가장 적게 접근된 항목 찾기
            min_access = min(entry.access_count for entry in self.l1_cache.values())
            for key, entry in self.l1_cache.items():
                if entry.access_count == min_access:
                    del self.l1_cache[key]
                    self.stats.eviction_count += 1
                    self.logger.debug(f"🔄 LFU 축출: {key}")
                    break
    
    def _evict_expired(self) -> None:
        """만료된 항목 축출"""
        expired_keys = []
        for key, entry in self.l1_cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.l1_cache[key]
            self.stats.eviction_count += 1
            self.logger.debug(f"🔄 TTL 축출: {key}")
    
    def _evict_adaptive(self, needed_space: int) -> None:
        """적응형 축출"""
        # 1. 만료된 항목 우선 제거
        self._evict_expired()
        
        # 2. 필요한 공간이 확보되었는지 확인
        current_size = sum(e.size_bytes for e in self.l1_cache.values())
        max_size = self.config.l1_max_size_mb * 1024 * 1024
        
        if current_size + needed_space <= max_size:
            return
        
        # 3. LFU와 LRU 조합
        entries_by_score = []
        for key, entry in self.l1_cache.items():
            # 점수 = 접근 빈도 * 최근 접근 시간 가중치
            time_weight = 1.0 / (1.0 + (time.time() - entry.last_accessed) / 3600)
            score = entry.access_count * time_weight
            entries_by_score.append((score, key))
        
        # 낮은 점수부터 정렬
        entries_by_score.sort()
        
        # 필요한 공간까지 제거
        freed_space = 0
        for score, key in entries_by_score:
            if freed_space >= needed_space:
                break
            
            entry = self.l1_cache[key]
            freed_space += entry.size_bytes
            del self.l1_cache[key]
            self.stats.eviction_count += 1
            self.logger.debug(f"🔄 적응형 축출: {key} (점수: {score:.2f})")
    
    def _load_from_disk(self, key: str, entry_info: Dict) -> Any:
        """디스크에서 로드"""
        try:
            file_path = Path(entry_info['file_path'])
            if not file_path.exists():
                return None
            
            with open(file_path, 'rb') as f:
                entry = pickle.load(f)
            
            return self._restore_data_from_storage(entry)
            
        except Exception as e:
            self.logger.error(f"❌ 디스크 로드 실패: {e}")
            return None
    
    def _remove_from_disk(self, key: str) -> None:
        """디스크에서 제거"""
        try:
            if key in self.l2_index:
                file_path = Path(self.l2_index[key]['file_path'])
                if file_path.exists():
                    file_path.unlink()
        except Exception as e:
            self.logger.error(f"❌ 디스크 제거 실패: {e}")
    
    def _promote_to_l1(self, key: str, data: Any, entry_info: Dict) -> None:
        """L1으로 승격"""
        try:
            # L1 엔트리 생성
            entry = CacheEntry(
                key=key,
                data=data,
                level=CacheLevel.L1_MEMORY,
                created_at=entry_info['created_at'],
                last_accessed=time.time(),
                access_count=1,
                ttl=entry_info['ttl'],
                size_bytes=entry_info['size_bytes'],
                compressed=False,  # L1에서는 압축 해제 상태로 보관
                compression_type=CompressionType.NONE,
                metadata={}
            )
            
            # L1에 저장
            self._store_to_l1(entry, CacheStrategy.ADAPTIVE)
            
            self.logger.debug(f"⬆️ L1 승격: {key}")
            
        except Exception as e:
            self.logger.error(f"❌ L1 승격 실패: {e}")
    
    def _remove_entry(self, key: str, level: CacheLevel) -> None:
        """엔트리 제거"""
        if level == CacheLevel.L1_MEMORY and key in self.l1_cache:
            del self.l1_cache[key]
        elif level == CacheLevel.L2_DISK and key in self.l2_index:
            self._remove_from_disk(key)
            del self.l2_index[key]
        elif level == CacheLevel.L3_NETWORK and key in self.l3_index:
            del self.l3_index[key]
    
    def _check_memory_pressure(self) -> None:
        """메모리 압박 확인 및 대응"""
        try:
            # 시스템 메모리 확인
            memory = psutil.virtual_memory()
            
            if memory.percent > self.config.memory_pressure_threshold:
                # 메모리 압박 상황 - 캐시 정리
                self._handle_memory_pressure(memory.percent)
                
        except Exception as e:
            self.logger.error(f"❌ 메모리 압박 확인 실패: {e}")
    
    def _handle_memory_pressure(self, memory_percent: float) -> None:
        """메모리 압박 처리"""
        if memory_percent > 95:
            # 심각: L1 캐시 50% 정리
            self._reduce_l1_cache(0.5)
            self.logger.warning(f"🚨 심각한 메모리 압박 ({memory_percent:.1f}%) - L1 캐시 50% 정리")
        elif memory_percent > 90:
            # 높음: L1 캐시 25% 정리
            self._reduce_l1_cache(0.25)
            self.logger.warning(f"⚠️ 높은 메모리 압박 ({memory_percent:.1f}%) - L1 캐시 25% 정리")
        elif memory_percent > 85:
            # 보통: 만료된 항목만 정리
            self._evict_expired()
            self.logger.info(f"ℹ️ 메모리 압박 ({memory_percent:.1f}%) - 만료 항목 정리")
    
    def _reduce_l1_cache(self, ratio: float) -> None:
        """L1 캐시 크기 감소"""
        target_count = int(len(self.l1_cache) * (1 - ratio))
        current_count = len(self.l1_cache)
        
        while len(self.l1_cache) > target_count:
            self._evict_adaptive(0)
        
        removed = current_count - len(self.l1_cache)
        self.logger.info(f"🧹 L1 캐시 축소: {removed}개 항목 제거")
    
    def _update_statistics(self) -> None:
        """통계 업데이트"""
        try:
            with self.lock:
                # 엔트리 수 업데이트
                self.stats.l1_entries = len(self.l1_cache)
                self.stats.l2_entries = len(self.l2_index)
                self.stats.l3_entries = len(self.l3_index)
                self.stats.total_entries = self.stats.l1_entries + self.stats.l2_entries + self.stats.l3_entries
                
                # 총 크기 계산
                l1_size = sum(entry.size_bytes for entry in self.l1_cache.values())
                l2_size = sum(info['size_bytes'] for info in self.l2_index.values())
                self.stats.total_size_mb = (l1_size + l2_size) / (1024**2)
                
                # 압축률 계산
                self._calculate_compression_ratio()
                
        except Exception as e:
            self.logger.error(f"❌ 통계 업데이트 실패: {e}")
    
    def _calculate_compression_ratio(self) -> None:
        """압축률 계산"""
        try:
            total_original = 0
            total_compressed = 0
            
            for entry in self.l1_cache.values():
                if entry.compressed:
                    # 압축된 크기와 원본 크기 비교 (추정)
                    total_compressed += entry.size_bytes
                    total_original += entry.size_bytes * 2  # 추정값
                else:
                    total_original += entry.size_bytes
                    total_compressed += entry.size_bytes
            
            if total_original > 0:
                self.stats.compression_ratio = total_compressed / total_original
            else:
                self.stats.compression_ratio = 1.0
                
        except Exception as e:
            self.logger.error(f"❌ 압축률 계산 실패: {e}")
            self.stats.compression_ratio = 1.0
    
    def _update_hit_rate(self) -> None:
        """히트율 업데이트"""
        total_requests = self.stats.hit_count + self.stats.miss_count
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hit_count / total_requests
        else:
            self.stats.hit_rate = 0.0
    
    def get_statistics(self) -> CacheStats:
        """캐시 통계 반환"""
        with self.lock:
            self._update_statistics()
            return self.stats
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 보고서 생성"""
        try:
            with self.lock:
                recent_ops = list(self.performance_history)[-100:]  # 최근 100개 작업
                
                if not recent_ops:
                    return {"status": "no_data"}
                
                # 작업별 통계
                set_ops = [op for op in recent_ops if op['operation'] == 'set']
                get_ops = [op for op in recent_ops if op['operation'] == 'get']
                hit_ops = [op for op in get_ops if op.get('hit', False)]
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "total_operations": len(recent_ops),
                    "set_operations": len(set_ops),
                    "get_operations": len(get_ops),
                    "cache_hits": len(hit_ops),
                    "hit_rate": len(hit_ops) / len(get_ops) if get_ops else 0.0,
                    "avg_set_time_ms": sum(op['duration_ms'] for op in set_ops) / len(set_ops) if set_ops else 0.0,
                    "avg_get_time_ms": sum(op['duration_ms'] for op in get_ops) / len(get_ops) if get_ops else 0.0,
                    "level_distribution": {
                        "l1_memory": len([op for op in hit_ops if op.get('level') == 'l1_memory']),
                        "l2_disk": len([op for op in hit_ops if op.get('level') == 'l2_disk']),
                        "l3_network": len([op for op in hit_ops if op.get('level') == 'l3_network'])
                    }
                }
                
        except Exception as e:
            self.logger.error(f"❌ 성능 보고서 생성 실패: {e}")
            return {"status": "error", "message": str(e)}
    
    def register_memory_optimizer(self, memory_optimizer) -> None:
        """메모리 최적화 엔진 등록"""
        self.memory_optimizer = memory_optimizer
        if memory_optimizer:
            memory_optimizer.register_cache(self)
            self.logger.info("🔗 메모리 최적화 엔진 연동 완료")
    
    def partial_clear(self, ratio: float = 0.5) -> None:
        """부분 캐시 정리 (메모리 최적화 엔진용)"""
        with self.lock:
            target_count = int(len(self.l1_cache) * (1 - ratio))
            
            while len(self.l1_cache) > target_count:
                self._evict_adaptive(0)
            
            self.logger.info(f"🧹 부분 캐시 정리: {ratio:.1%} 정리 완료")
    
    def get_size(self) -> float:
        """캐시 크기 반환 (MB 단위)"""
        return self.stats.total_size_mb
    
    def export_cache_report(self, output_path: str) -> None:
        """캐시 보고서 내보내기"""
        try:
            report_data = {
                "export_timestamp": datetime.now().isoformat(),
                "cache_statistics": asdict(self.get_statistics()),
                "performance_report": self.get_performance_report(),
                "configuration": asdict(self.config),
                "cache_keys": {
                    "l1_keys": list(self.l1_cache.keys())[:50],  # 최대 50개
                    "l2_keys": list(self.l2_index.keys())[:50],
                    "l3_keys": list(self.l3_index.keys())[:50]
                },
                "access_frequency": dict(sorted(
                    self.access_frequency.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:20])  # 상위 20개
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"📊 캐시 보고서 생성: {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ 캐시 보고서 생성 실패: {e}")
    
    def shutdown(self) -> None:
        """캐시 매니저 종료"""
        try:
            # 자동 정리 중지
            self.stop_auto_cleanup()
            
            # 인덱스 저장
            self._save_cache_indexes()
            
            # 통계 업데이트
            self._update_statistics()
            
            self.logger.info("✅ 스마트 캐시 관리자 종료 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 캐시 매니저 종료 실패: {e}")

# 전역 캐시 매니저 인스턴스
_global_cache_manager = None

def get_global_cache_manager() -> SmartCacheManager:
    """전역 캐시 매니저 인스턴스 반환"""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = SmartCacheManager()
        _global_cache_manager.start_auto_cleanup()
    return _global_cache_manager

# 편의 함수들
def cache_set(key: str, data: Any, ttl: Optional[float] = None, level: CacheLevel = CacheLevel.L1_MEMORY) -> bool:
    """캐시 설정"""
    return get_global_cache_manager().set(key, data, level=level, ttl=ttl)

def cache_get(key: str, default: Any = None) -> Any:
    """캐시 조회"""
    return get_global_cache_manager().get(key, default)

def cache_delete(key: str) -> bool:
    """캐시 삭제"""
    return get_global_cache_manager().delete(key)

def cache_clear(level: Optional[CacheLevel] = None) -> None:
    """캐시 정리"""
    get_global_cache_manager().clear(level)

# 사용 예시
if __name__ == "__main__":
    cache_manager = SmartCacheManager()
    cache_manager.start_auto_cleanup()
    
    print("🚀 스마트 캐시 관리자 테스트:")
    
    # 테스트 데이터
    test_data = {
        "analysis_result": {
            "transcription": "안녕하세요, 테스트 음성입니다.",
            "sentiment": "positive",
            "entities": ["테스트", "음성"],
            "confidence": 0.95
        }
    }
    
    # 캐시 설정
    cache_manager.set("audio_analysis_001", test_data, ttl=3600)
    
    # 캐시 조회
    result = cache_manager.get("audio_analysis_001")
    print(f"✅ 캐시 조회 결과: {result is not None}")
    
    # 통계 확인
    stats = cache_manager.get_statistics()
    print(f"📊 캐시 통계:")
    print(f"   총 엔트리: {stats.total_entries}개")
    print(f"   히트율: {stats.hit_rate:.1%}")
    print(f"   총 크기: {stats.total_size_mb:.2f}MB")
    
    try:
        time.sleep(2)  # 테스트 대기
    except KeyboardInterrupt:
        print("\n테스트 중단")
    finally:
        cache_manager.shutdown()
        print("✅ 스마트 캐시 관리자 테스트 완료")