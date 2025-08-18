#!/usr/bin/env python3
"""
대용량 파일 스트리밍 최적화 시스템 v2.6
메모리 효율적 파일 처리 및 스트리밍 구현
"""

import os
import io
import gc
import time
import hashlib
import threading
import mmap
from typing import Iterator, BinaryIO, Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import logging
from contextlib import contextmanager
import asyncio
import aiofiles
import psutil

@dataclass
class StreamingConfig:
    """스트리밍 설정"""
    chunk_size_mb: float = 8.0  # 8MB 청크
    max_memory_usage_mb: float = 256.0  # 최대 메모리 사용량
    enable_memory_mapping: bool = True  # 메모리 매핑 활성화
    enable_compression: bool = False  # 압축 활성화 (CPU vs 메모리 트레이드오프)
    temp_dir: Optional[str] = None  # 임시 디렉토리
    buffer_count: int = 3  # 버퍼 개수 (트리플 버퍼링)
    enable_async: bool = True  # 비동기 처리
    progress_callback: Optional[callable] = None  # 진행률 콜백

@dataclass
class StreamingStats:
    """스트리밍 통계"""
    total_size_mb: float = 0.0
    processed_size_mb: float = 0.0
    chunks_processed: int = 0
    processing_time_seconds: float = 0.0
    peak_memory_usage_mb: float = 0.0
    average_chunk_time_ms: float = 0.0
    throughput_mbps: float = 0.0
    memory_efficiency: float = 1.0  # 메모리 효율성 (낮을수록 좋음)
    compression_ratio: float = 1.0  # 압축률 (낮을수록 좋음)

@dataclass
class FileChunk:
    """파일 청크"""
    chunk_id: int
    data: bytes
    size_bytes: int
    offset: int
    is_compressed: bool = False
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class LargeFileStreamingOptimizer:
    """대용량 파일 스트리밍 최적화 시스템"""
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()
        self.stats = StreamingStats()
        self.is_streaming = False
        self.cancel_requested = False
        self.lock = threading.RLock()
        self.logger = self._setup_logging()
        
        # 청크 크기를 바이트로 변환
        self.chunk_size_bytes = int(self.config.chunk_size_mb * 1024 * 1024)
        self.max_memory_bytes = int(self.config.max_memory_usage_mb * 1024 * 1024)
        
        # 임시 디렉토리 설정
        if self.config.temp_dir:
            self.temp_dir = Path(self.config.temp_dir)
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / "solomond_streaming"
        
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        # 버퍼 풀
        self.buffer_pool = []
        self._init_buffer_pool()
        
        # 압축 모듈 (옵션)
        self.compressor = None
        if self.config.enable_compression:
            try:
                import lz4.frame
                self.compressor = lz4.frame
                self.logger.info("✅ LZ4 압축 활성화")
            except ImportError:
                try:
                    import gzip
                    self.compressor = gzip
                    self.logger.info("✅ GZIP 압축 활성화")
                except ImportError:
                    self.logger.warning("⚠️ 압축 라이브러리를 찾을 수 없음")
    
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
    
    def _init_buffer_pool(self) -> None:
        """버퍼 풀 초기화"""
        try:
            for i in range(self.config.buffer_count):
                buffer = bytearray(self.chunk_size_bytes)
                self.buffer_pool.append(buffer)
            
            self.logger.debug(f"✅ 버퍼 풀 초기화: {self.config.buffer_count}개 × {self.config.chunk_size_mb}MB")
        except Exception as e:
            self.logger.error(f"❌ 버퍼 풀 초기화 실패: {e}")
    
    def _get_buffer(self) -> bytearray:
        """버퍼 가져오기"""
        with self.lock:
            if self.buffer_pool:
                return self.buffer_pool.pop()
            else:
                # 풀이 비어있으면 새로 생성
                return bytearray(self.chunk_size_bytes)
    
    def _return_buffer(self, buffer: bytearray) -> None:
        """버퍼 반환"""
        with self.lock:
            if len(self.buffer_pool) < self.config.buffer_count:
                # 버퍼 초기화
                buffer[:] = b'\x00' * len(buffer)
                self.buffer_pool.append(buffer)
    
    def _calculate_checksum(self, data: bytes) -> str:
        """체크섬 계산"""
        return hashlib.md5(data).hexdigest()
    
    def _compress_data(self, data: bytes) -> Tuple[bytes, float]:
        """데이터 압축"""
        if not self.compressor:
            return data, 1.0
        
        try:
            if hasattr(self.compressor, 'compress'):
                # LZ4 또는 gzip
                if self.compressor.__name__ == 'lz4.frame':
                    compressed = self.compressor.compress(data)
                else:
                    compressed = self.compressor.compress(data)
            else:
                compressed = data
            
            compression_ratio = len(compressed) / len(data) if len(data) > 0 else 1.0
            return compressed, compression_ratio
        
        except Exception as e:
            self.logger.warning(f"⚠️ 압축 실패: {e}")
            return data, 1.0
    
    def _monitor_memory_usage(self) -> float:
        """메모리 사용량 모니터링"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # 피크 메모리 업데이트
            if memory_mb > self.stats.peak_memory_usage_mb:
                self.stats.peak_memory_usage_mb = memory_mb
            
            return memory_mb
        except Exception:
            return 0.0
    
    @contextmanager
    def _memory_mapped_file(self, file_path: Union[str, Path]):
        """메모리 매핑된 파일 컨텍스트 관리자"""
        file_path = Path(file_path)
        
        if not self.config.enable_memory_mapping or file_path.stat().st_size > self.max_memory_bytes:
            # 메모리 매핑 비활성화 또는 파일이 너무 큰 경우 일반 파일 열기
            with open(file_path, 'rb') as f:
                yield f
        else:
            # 메모리 매핑 사용
            with open(file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    yield mmapped_file
    
    def stream_file_chunks(self, file_path: Union[str, Path]) -> Iterator[FileChunk]:
        """파일을 청크 단위로 스트리밍"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        # 통계 초기화
        self.stats = StreamingStats()
        self.stats.total_size_mb = file_path.stat().st_size / (1024 * 1024)
        self.is_streaming = True
        self.cancel_requested = False
        
        start_time = time.time()
        chunk_id = 0
        
        try:
            with self._memory_mapped_file(file_path) as file_obj:
                self.logger.info(f"🚀 파일 스트리밍 시작: {file_path.name} ({self.stats.total_size_mb:.1f}MB)")
                
                while not self.cancel_requested:
                    # 메모리 사용량 체크
                    current_memory = self._monitor_memory_usage()
                    if current_memory > self.config.max_memory_usage_mb:
                        self.logger.warning(f"⚠️ 메모리 사용량 초과: {current_memory:.1f}MB")
                        # 가비지 컬렉션 강제 실행
                        gc.collect()
                    
                    # 버퍼 가져오기
                    buffer = self._get_buffer()
                    
                    try:
                        # 청크 읽기
                        chunk_start_time = time.time()
                        data = file_obj.read(self.chunk_size_bytes)
                        
                        if not data:
                            break  # 파일 끝
                        
                        # 실제 데이터 크기에 맞게 버퍼 조정
                        if len(data) < len(buffer):
                            buffer = buffer[:len(data)]
                        
                        buffer[:len(data)] = data
                        actual_data = bytes(buffer[:len(data)])
                        
                        # 압축 (옵션)
                        compressed_data, compression_ratio = actual_data, 1.0
                        if self.config.enable_compression and len(actual_data) > 1024:  # 1KB 이상만 압축
                            compressed_data, compression_ratio = self._compress_data(actual_data)
                        
                        # 체크섬 계산
                        checksum = self._calculate_checksum(actual_data)
                        
                        # 청크 객체 생성
                        chunk = FileChunk(
                            chunk_id=chunk_id,
                            data=compressed_data,
                            size_bytes=len(actual_data),
                            offset=chunk_id * self.chunk_size_bytes,
                            is_compressed=compression_ratio < 1.0,
                            checksum=checksum,
                            metadata={
                                'compression_ratio': compression_ratio,
                                'processing_time_ms': (time.time() - chunk_start_time) * 1000
                            }
                        )
                        
                        # 통계 업데이트
                        self.stats.chunks_processed += 1
                        self.stats.processed_size_mb += len(actual_data) / (1024 * 1024)
                        
                        chunk_time_ms = (time.time() - chunk_start_time) * 1000
                        self.stats.average_chunk_time_ms = (
                            (self.stats.average_chunk_time_ms * (chunk_id) + chunk_time_ms) / (chunk_id + 1)
                        )
                        
                        # 진행률 콜백
                        if self.config.progress_callback:
                            progress = (self.stats.processed_size_mb / self.stats.total_size_mb) * 100
                            self.config.progress_callback(progress, chunk)
                        
                        yield chunk
                        
                        chunk_id += 1
                    
                    finally:
                        # 버퍼 반환
                        self._return_buffer(buffer)
        
        finally:
            self.is_streaming = False
            
            # 최종 통계 계산
            self.stats.processing_time_seconds = time.time() - start_time
            if self.stats.processing_time_seconds > 0:
                self.stats.throughput_mbps = self.stats.processed_size_mb / self.stats.processing_time_seconds
            
            self.stats.memory_efficiency = self.stats.peak_memory_usage_mb / self.stats.total_size_mb if self.stats.total_size_mb > 0 else 1.0
            
            self.logger.info(f"✅ 파일 스트리밍 완료: {self.stats.chunks_processed}개 청크, {self.stats.throughput_mbps:.1f}MB/s")
    
    async def async_stream_file_chunks(self, file_path: Union[str, Path]) -> Iterator[FileChunk]:
        """비동기 파일 스트리밍"""
        if not self.config.enable_async:
            # 동기 버전으로 폴백
            for chunk in self.stream_file_chunks(file_path):
                yield chunk
            return
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        # 통계 초기화
        self.stats = StreamingStats()
        self.stats.total_size_mb = file_path.stat().st_size / (1024 * 1024)
        self.is_streaming = True
        self.cancel_requested = False
        
        start_time = time.time()
        chunk_id = 0
        
        try:
            async with aiofiles.open(file_path, 'rb') as file_obj:
                self.logger.info(f"🚀 비동기 파일 스트리밍 시작: {file_path.name} ({self.stats.total_size_mb:.1f}MB)")
                
                while not self.cancel_requested:
                    # 메모리 사용량 체크
                    current_memory = self._monitor_memory_usage()
                    if current_memory > self.config.max_memory_usage_mb:
                        self.logger.warning(f"⚠️ 메모리 사용량 초과: {current_memory:.1f}MB")
                        gc.collect()
                        await asyncio.sleep(0.01)  # 짧은 대기
                    
                    # 청크 읽기
                    chunk_start_time = time.time()
                    data = await file_obj.read(self.chunk_size_bytes)
                    
                    if not data:
                        break  # 파일 끝
                    
                    # 압축 (옵션)
                    compressed_data, compression_ratio = data, 1.0
                    if self.config.enable_compression and len(data) > 1024:
                        compressed_data, compression_ratio = self._compress_data(data)
                    
                    # 체크섬 계산
                    checksum = self._calculate_checksum(data)
                    
                    # 청크 객체 생성
                    chunk = FileChunk(
                        chunk_id=chunk_id,
                        data=compressed_data,
                        size_bytes=len(data),
                        offset=chunk_id * self.chunk_size_bytes,
                        is_compressed=compression_ratio < 1.0,
                        checksum=checksum,
                        metadata={
                            'compression_ratio': compression_ratio,
                            'processing_time_ms': (time.time() - chunk_start_time) * 1000
                        }
                    )
                    
                    # 통계 업데이트
                    self.stats.chunks_processed += 1
                    self.stats.processed_size_mb += len(data) / (1024 * 1024)
                    
                    # 진행률 콜백
                    if self.config.progress_callback:
                        progress = (self.stats.processed_size_mb / self.stats.total_size_mb) * 100
                        self.config.progress_callback(progress, chunk)
                    
                    yield chunk
                    
                    chunk_id += 1
                    
                    # 다른 태스크에게 제어권 양보
                    await asyncio.sleep(0)
        
        finally:
            self.is_streaming = False
            
            # 최종 통계 계산
            self.stats.processing_time_seconds = time.time() - start_time
            if self.stats.processing_time_seconds > 0:
                self.stats.throughput_mbps = self.stats.processed_size_mb / self.stats.processing_time_seconds
            
            self.stats.memory_efficiency = self.stats.peak_memory_usage_mb / self.stats.total_size_mb if self.stats.total_size_mb > 0 else 1.0
            
            self.logger.info(f"✅ 비동기 파일 스트리밍 완료: {self.stats.chunks_processed}개 청크, {self.stats.throughput_mbps:.1f}MB/s")
    
    def process_file_streaming(self, file_path: Union[str, Path], processor_func: callable) -> Any:
        """스트리밍 방식으로 파일 처리"""
        results = []
        
        try:
            for chunk in self.stream_file_chunks(file_path):
                if self.cancel_requested:
                    break
                
                # 청크 처리
                try:
                    result = processor_func(chunk)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"❌ 청크 처리 오류: {e}")
                    continue
                
                # 메모리 정리 (주기적)
                if chunk.chunk_id % 10 == 0:
                    gc.collect()
            
            return results
        
        except Exception as e:
            self.logger.error(f"❌ 스트리밍 처리 실패: {e}")
            raise
    
    def cancel_streaming(self) -> None:
        """스트리밍 취소"""
        self.cancel_requested = True
        self.logger.info("🛑 스트리밍 취소 요청")
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """스트리밍 통계 반환"""
        return {
            'is_streaming': self.is_streaming,
            'total_size_mb': self.stats.total_size_mb,
            'processed_size_mb': self.stats.processed_size_mb,
            'progress_percent': (self.stats.processed_size_mb / self.stats.total_size_mb * 100) if self.stats.total_size_mb > 0 else 0,
            'chunks_processed': self.stats.chunks_processed,
            'processing_time_seconds': self.stats.processing_time_seconds,
            'peak_memory_usage_mb': self.stats.peak_memory_usage_mb,
            'average_chunk_time_ms': self.stats.average_chunk_time_ms,
            'throughput_mbps': self.stats.throughput_mbps,
            'memory_efficiency': self.stats.memory_efficiency,
            'compression_ratio': self.stats.compression_ratio,
            'estimated_remaining_time_seconds': self._estimate_remaining_time(),
            'memory_usage_trend': self._get_memory_trend()
        }
    
    def _estimate_remaining_time(self) -> float:
        """남은 시간 예측"""
        if not self.is_streaming or self.stats.throughput_mbps <= 0:
            return 0.0
        
        remaining_mb = self.stats.total_size_mb - self.stats.processed_size_mb
        return remaining_mb / self.stats.throughput_mbps if self.stats.throughput_mbps > 0 else 0.0
    
    def _get_memory_trend(self) -> str:
        """메모리 사용 추세"""
        current_memory = self._monitor_memory_usage()
        if current_memory > self.config.max_memory_usage_mb * 0.9:
            return "critical"
        elif current_memory > self.config.max_memory_usage_mb * 0.7:
            return "high"
        elif current_memory > self.config.max_memory_usage_mb * 0.5:
            return "medium"
        else:
            return "low"
    
    def optimize_for_file_type(self, file_path: Union[str, Path]) -> None:
        """파일 타입에 따른 최적화"""
        file_path = Path(file_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        file_ext = file_path.suffix.lower()
        
        # 파일 타입별 최적화
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # 비디오 파일: 큰 청크, 압축 비활성화
            self.config.chunk_size_mb = min(32.0, file_size_mb / 10)
            self.config.enable_compression = False
            self.config.buffer_count = 4
            
        elif file_ext in ['.mp3', '.wav', '.m4a', '.flac']:
            # 오디오 파일: 중간 청크, 압축 선택적
            self.config.chunk_size_mb = min(16.0, file_size_mb / 20)
            self.config.enable_compression = file_size_mb > 100  # 100MB 이상만 압축
            self.config.buffer_count = 3
            
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            # 이미지 파일: 작은 청크, 압축 비활성화 (이미 압축됨)
            self.config.chunk_size_mb = min(4.0, file_size_mb / 5)
            self.config.enable_compression = False
            self.config.buffer_count = 2
            
        elif file_ext in ['.txt', '.log', '.csv', '.json']:
            # 텍스트 파일: 작은 청크, 압축 활성화
            self.config.chunk_size_mb = min(2.0, file_size_mb / 50)
            self.config.enable_compression = True
            self.config.buffer_count = 2
        
        else:
            # 기본 설정
            self.config.chunk_size_mb = min(8.0, file_size_mb / 20)
            self.config.enable_compression = file_size_mb > 50
            self.config.buffer_count = 3
        
        # 청크 크기 재계산
        self.chunk_size_bytes = int(self.config.chunk_size_mb * 1024 * 1024)
        
        # 버퍼 풀 재초기화
        self.buffer_pool.clear()
        self._init_buffer_pool()
        
        self.logger.info(f"📊 파일 타입 최적화 완료: {file_ext} -> 청크 {self.config.chunk_size_mb}MB, 압축 {'ON' if self.config.enable_compression else 'OFF'}")
    
    def cleanup(self) -> None:
        """리소스 정리"""
        self.cancel_streaming()
        
        # 버퍼 풀 정리
        with self.lock:
            self.buffer_pool.clear()
        
        # 임시 파일 정리
        try:
            if self.temp_dir.exists():
                for temp_file in self.temp_dir.glob("*"):
                    temp_file.unlink()
                
                # 빈 디렉토리면 제거
                try:
                    self.temp_dir.rmdir()
                except OSError:
                    pass  # 디렉토리가 비어있지 않음
        except Exception as e:
            self.logger.warning(f"⚠️ 임시 파일 정리 실패: {e}")
        
        # 가비지 컬렉션
        gc.collect()
        
        self.logger.info("🧹 스트리밍 최적화 시스템 정리 완료")

# 전역 스트리밍 최적화 시스템
_global_streaming_optimizer = None
_global_lock = threading.Lock()

def get_global_streaming_optimizer(config: Optional[StreamingConfig] = None) -> LargeFileStreamingOptimizer:
    """전역 스트리밍 최적화 시스템 가져오기"""
    global _global_streaming_optimizer
    
    with _global_lock:
        if _global_streaming_optimizer is None:
            _global_streaming_optimizer = LargeFileStreamingOptimizer(config)
        return _global_streaming_optimizer

# 편의 함수들
def stream_large_file(file_path: Union[str, Path], 
                     chunk_size_mb: float = 8.0,
                     max_memory_mb: float = 256.0,
                     enable_compression: bool = False,
                     progress_callback: Optional[callable] = None) -> Iterator[FileChunk]:
    """대용량 파일 스트리밍 (편의 함수)"""
    config = StreamingConfig(
        chunk_size_mb=chunk_size_mb,
        max_memory_usage_mb=max_memory_mb,
        enable_compression=enable_compression,
        progress_callback=progress_callback
    )
    
    optimizer = LargeFileStreamingOptimizer(config)
    optimizer.optimize_for_file_type(file_path)
    
    for chunk in optimizer.stream_file_chunks(file_path):
        yield chunk

# 사용 예시
if __name__ == "__main__":
    # 테스트용 큰 파일 생성
    test_file = Path("test_large_file.bin")
    test_size_mb = 100  # 100MB 테스트 파일
    
    if not test_file.exists():
        print(f"🔧 테스트 파일 생성 중: {test_size_mb}MB")
        with open(test_file, 'wb') as f:
            chunk_size = 1024 * 1024  # 1MB 청크
            for i in range(test_size_mb):
                data = os.urandom(chunk_size)
                f.write(data)
        print("✅ 테스트 파일 생성 완료")
    
    # 진행률 콜백 함수
    def progress_callback(progress: float, chunk: FileChunk):
        print(f"📊 진행률: {progress:.1f}% - 청크 {chunk.chunk_id} ({chunk.size_bytes/1024/1024:.1f}MB)")
    
    # 스트리밍 테스트
    print("\n🚀 스트리밍 테스트 시작")
    
    config = StreamingConfig(
        chunk_size_mb=8.0,
        max_memory_usage_mb=64.0,
        enable_compression=True,
        progress_callback=progress_callback
    )
    
    optimizer = LargeFileStreamingOptimizer(config)
    optimizer.optimize_for_file_type(test_file)
    
    total_chunks = 0
    total_size = 0
    
    start_time = time.time()
    
    for chunk in optimizer.stream_file_chunks(test_file):
        total_chunks += 1
        total_size += chunk.size_bytes
        
        # 처리 시뮬레이션
        time.sleep(0.01)  # 10ms 처리 시간
    
    end_time = time.time()
    
    # 결과 출력
    stats = optimizer.get_streaming_stats()
    print(f"\n📊 스트리밍 완료:")
    print(f"  총 청크: {total_chunks}개")
    print(f"  총 크기: {total_size/1024/1024:.1f}MB")
    print(f"  처리 시간: {end_time - start_time:.2f}초")
    print(f"  처리량: {stats['throughput_mbps']:.1f}MB/s")
    print(f"  피크 메모리: {stats['peak_memory_usage_mb']:.1f}MB")
    print(f"  메모리 효율성: {stats['memory_efficiency']:.2f}")
    
    # 정리
    optimizer.cleanup()
    
    # 테스트 파일 삭제
    if test_file.exists():
        test_file.unlink()
        print("🧹 테스트 파일 삭제 완료")
    
    print("✅ 스트리밍 테스트 완료!")