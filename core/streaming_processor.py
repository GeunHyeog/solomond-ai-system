# Phase 2 Week 3: 스트리밍 처리 엔진
# 메모리 최적화 - 대용량 파일 청킹 및 순차 처리

import asyncio
import os
import psutil
import gc
import tempfile
import gzip
import pickle
from typing import AsyncGenerator, Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
import logging
from enum import Enum
import hashlib
import json

# 메모리 모니터링
@dataclass
class MemoryStats:
    """메모리 사용 통계"""
    used_mb: float
    available_mb: float
    percent: float
    peak_mb: float
    timestamp: float

class ChunkStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"

@dataclass
class FileChunk:
    """파일 청크 정보"""
    chunk_id: str
    file_id: str
    start_byte: int
    end_byte: int
    size_mb: float
    status: ChunkStatus = ChunkStatus.PENDING
    processing_time: float = 0.0
    content_hash: str = ""
    compressed_path: Optional[str] = None
    result_data: Optional[Dict] = None

@dataclass
class StreamingConfig:
    """스트리밍 처리 설정"""
    chunk_size_mb: float = 50.0  # 청크 크기 (MB)
    max_memory_mb: float = 512.0  # 최대 메모리 사용량 (MB)
    compression_enabled: bool = True  # 압축 사용 여부
    cache_intermediate: bool = True  # 중간 결과 캐싱
    max_concurrent_chunks: int = 3  # 동시 처리 청크 수
    memory_threshold: float = 0.8  # 메모리 사용률 임계값
    cleanup_interval: int = 30  # 정리 주기 (초)

class MemoryMonitor:
    """실시간 메모리 모니터링"""
    
    def __init__(self):
        self.stats_history: List[MemoryStats] = []
        self.peak_memory = 0.0
        self.alerts: List[str] = []
        self.monitoring = False
    
    def start_monitoring(self):
        """모니터링 시작"""
        self.monitoring = True
        asyncio.create_task(self._monitor_loop())
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
    
    async def _monitor_loop(self):
        """모니터링 루프"""
        while self.monitoring:
            stats = self.get_current_stats()
            self.stats_history.append(stats)
            
            # 피크 메모리 업데이트
            if stats.used_mb > self.peak_memory:
                self.peak_memory = stats.used_mb
            
            # 메모리 부족 경고
            if stats.percent > 85:
                self.alerts.append(f"높은 메모리 사용률: {stats.percent:.1f}%")
            
            # 히스토리 크기 제한 (최근 100개)
            if len(self.stats_history) > 100:
                self.stats_history = self.stats_history[-100:]
            
            await asyncio.sleep(5)  # 5초마다 체크
    
    def get_current_stats(self) -> MemoryStats:
        """현재 메모리 통계 반환"""
        memory = psutil.virtual_memory()
        return MemoryStats(
            used_mb=memory.used / 1024 / 1024,
            available_mb=memory.available / 1024 / 1024,
            percent=memory.percent,
            peak_mb=self.peak_memory,
            timestamp=time.time()
        )
    
    def get_memory_report(self) -> Dict[str, Any]:
        """메모리 사용 리포트"""
        if not self.stats_history:
            return {"error": "모니터링 데이터가 없습니다"}
        
        current = self.stats_history[-1]
        avg_usage = sum(s.used_mb for s in self.stats_history[-10:]) / min(10, len(self.stats_history))
        
        return {
            "current_mb": current.used_mb,
            "peak_mb": self.peak_memory,
            "average_mb": avg_usage,
            "current_percent": current.percent,
            "available_mb": current.available_mb,
            "alerts": self.alerts[-5:],  # 최근 5개 알림
            "trend": "increasing" if len(self.stats_history) > 5 and 
                    self.stats_history[-1].used_mb > self.stats_history[-5].used_mb else "stable"
        }

class FileChunker:
    """파일 청킹 관리자"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.temp_dir = Path(tempfile.mkdtemp(prefix="solomond_chunks_"))
        self.temp_dir.mkdir(exist_ok=True)
    
    def create_chunks(self, file_path: str, file_id: str) -> List[FileChunk]:
        """파일을 청크로 분할"""
        file_size = os.path.getsize(file_path)
        chunk_size_bytes = int(self.config.chunk_size_mb * 1024 * 1024)
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < file_size:
            end = min(start + chunk_size_bytes, file_size)
            chunk_size_mb = (end - start) / 1024 / 1024
            
            chunk = FileChunk(
                chunk_id=f"{file_id}_chunk_{chunk_index}",
                file_id=file_id,
                start_byte=start,
                end_byte=end,
                size_mb=chunk_size_mb
            )
            
            chunks.append(chunk)
            start = end
            chunk_index += 1
        
        return chunks
    
    async def read_chunk(self, file_path: str, chunk: FileChunk) -> bytes:
        """청크 데이터 읽기"""
        try:
            with open(file_path, 'rb') as f:
                f.seek(chunk.start_byte)
                data = f.read(chunk.end_byte - chunk.start_byte)
                
                # 해시 생성
                chunk.content_hash = hashlib.md5(data).hexdigest()
                
                return data
        except Exception as e:
            raise Exception(f"청크 읽기 실패: {e}")
    
    async def compress_chunk(self, data: bytes, chunk: FileChunk) -> str:
        """청크 압축 저장"""
        if not self.config.compression_enabled:
            return None
        
        compressed_path = self.temp_dir / f"{chunk.chunk_id}.gz"
        
        try:
            with gzip.open(compressed_path, 'wb') as f:
                f.write(data)
            
            # 압축률 확인
            original_size = len(data)
            compressed_size = os.path.getsize(compressed_path)
            compression_ratio = compressed_size / original_size
            
            chunk.compressed_path = str(compressed_path)
            
            return str(compressed_path)
        except Exception as e:
            raise Exception(f"청크 압축 실패: {e}")
    
    async def decompress_chunk(self, chunk: FileChunk) -> bytes:
        """청크 압축 해제"""
        if not chunk.compressed_path:
            raise Exception("압축된 파일이 없습니다")
        
        try:
            with gzip.open(chunk.compressed_path, 'rb') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"압축 해제 실패: {e}")
    
    def cleanup_chunks(self, chunks: List[FileChunk]):
        """청크 파일 정리"""
        for chunk in chunks:
            if chunk.compressed_path and os.path.exists(chunk.compressed_path):
                try:
                    os.remove(chunk.compressed_path)
                except:
                    pass
    
    def __del__(self):
        """임시 디렉토리 정리"""
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass

class StreamingProcessor:
    """스트리밍 기반 파일 처리기"""
    
    def __init__(self, config: StreamingConfig = None):
        self.config = config or StreamingConfig()
        self.memory_monitor = MemoryMonitor()
        self.chunker = FileChunker(self.config)
        self.processing_queue = asyncio.Queue()
        self.active_chunks: Dict[str, FileChunk] = {}
        self.completed_chunks: Dict[str, FileChunk] = {}
        self.logger = logging.getLogger(__name__)
        
        # 통계
        self.total_processed_mb = 0.0
        self.total_processing_time = 0.0
        self.error_count = 0
        
    async def start_monitoring(self):
        """모니터링 시작"""
        self.memory_monitor.start_monitoring()
        asyncio.create_task(self._cleanup_loop())
    
    async def stop_monitoring(self):
        """모니터링 중지"""
        self.memory_monitor.stop_monitoring()
    
    async def _cleanup_loop(self):
        """주기적 메모리 정리"""
        while True:
            await asyncio.sleep(self.config.cleanup_interval)
            await self._force_cleanup()
    
    async def _force_cleanup(self):
        """강제 메모리 정리"""
        # 완료된 청크 메모리 해제
        cleanup_count = 0
        for chunk_id in list(self.completed_chunks.keys()):
            chunk = self.completed_chunks[chunk_id]
            if chunk.result_data:
                # 압축 저장 후 메모리 해제
                if self.config.cache_intermediate:
                    await self._cache_chunk_result(chunk)
                chunk.result_data = None
                cleanup_count += 1
        
        # 가비지 컬렉션 강제 실행
        gc.collect()
        
        if cleanup_count > 0:
            self.logger.info(f"메모리 정리 완료: {cleanup_count}개 청크")
    
    async def _cache_chunk_result(self, chunk: FileChunk):
        """청크 결과 캐싱"""
        if not chunk.result_data:
            return
        
        cache_path = self.chunker.temp_dir / f"{chunk.chunk_id}_result.pkl.gz"
        
        try:
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(chunk.result_data, f)
            
            # 메모리에서 제거
            chunk.result_data = None
            chunk.status = ChunkStatus.CACHED
            
        except Exception as e:
            self.logger.warning(f"캐싱 실패: {e}")
    
    async def _load_cached_result(self, chunk: FileChunk) -> Optional[Dict]:
        """캐시된 결과 로드"""
        cache_path = self.chunker.temp_dir / f"{chunk.chunk_id}_result.pkl.gz"
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with gzip.open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"캐시 로드 실패: {e}")
            return None
    
    async def process_file_streaming(
        self, 
        file_path: str, 
        file_id: str, 
        processor_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """스트리밍 방식으로 파일 처리"""
        
        start_time = time.time()
        
        try:
            # 파일 청킹
            chunks = self.chunker.create_chunks(file_path, file_id)
            self.logger.info(f"파일 청킹 완료: {len(chunks)}개 청크")
            
            # 청크별 처리
            processed_chunks = []
            total_chunks = len(chunks)
            
            # 세마포어를 사용한 동시 처리 제한
            semaphore = asyncio.Semaphore(self.config.max_concurrent_chunks)
            
            async def process_chunk_with_semaphore(chunk):
                async with semaphore:
                    return await self._process_single_chunk(
                        file_path, chunk, processor_func
                    )
            
            # 청크들을 배치로 처리
            tasks = [process_chunk_with_semaphore(chunk) for chunk in chunks]
            
            for i, task in enumerate(asyncio.as_completed(tasks)):
                try:
                    result_chunk = await task
                    processed_chunks.append(result_chunk)
                    
                    # 진행률 콜백
                    if progress_callback:
                        progress = (i + 1) / total_chunks * 100
                        await progress_callback(progress, f"청크 {i+1}/{total_chunks} 완료")
                    
                    # 메모리 확인 및 정리
                    await self._check_memory_and_cleanup()
                    
                except Exception as e:
                    self.error_count += 1
                    self.logger.error(f"청크 처리 오류: {e}")
            
            # 결과 통합
            final_result = await self._merge_chunk_results(processed_chunks)
            
            # 통계 업데이트
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.total_processed_mb += sum(c.size_mb for c in chunks)
            
            # 정리
            self.chunker.cleanup_chunks(chunks)
            
            return {
                "success": True,
                "file_id": file_id,
                "chunks_processed": len(processed_chunks),
                "total_chunks": total_chunks,
                "processing_time": processing_time,
                "total_size_mb": sum(c.size_mb for c in chunks),
                "average_chunk_time": processing_time / total_chunks,
                "memory_peak_mb": self.memory_monitor.peak_memory,
                "result": final_result
            }
            
        except Exception as e:
            self.logger.error(f"스트리밍 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_id": file_id
            }
    
    async def _process_single_chunk(
        self, 
        file_path: str, 
        chunk: FileChunk, 
        processor_func: Callable
    ) -> FileChunk:
        """단일 청크 처리"""
        
        start_time = time.time()
        chunk.status = ChunkStatus.PROCESSING
        self.active_chunks[chunk.chunk_id] = chunk
        
        try:
            # 청크 데이터 읽기
            chunk_data = await self.chunker.read_chunk(file_path, chunk)
            
            # 메모리 사용량 체크
            current_memory = self.memory_monitor.get_current_stats()
            if current_memory.percent > self.config.memory_threshold * 100:
                # 메모리 부족시 압축 저장
                await self.chunker.compress_chunk(chunk_data, chunk)
                chunk_data = None  # 메모리 해제
                gc.collect()
            
            # 처리 함수 실행
            if chunk_data:
                result = await processor_func(chunk_data, chunk)
            else:
                # 압축된 데이터로부터 처리
                chunk_data = await self.chunker.decompress_chunk(chunk)
                result = await processor_func(chunk_data, chunk)
                chunk_data = None  # 즉시 메모리 해제
            
            # 결과 저장
            chunk.result_data = result
            chunk.status = ChunkStatus.COMPLETED
            chunk.processing_time = time.time() - start_time
            
            # 활성 청크에서 완료 청크로 이동
            del self.active_chunks[chunk.chunk_id]
            self.completed_chunks[chunk.chunk_id] = chunk
            
            return chunk
            
        except Exception as e:
            chunk.status = ChunkStatus.FAILED
            chunk.processing_time = time.time() - start_time
            self.logger.error(f"청크 {chunk.chunk_id} 처리 실패: {e}")
            
            if chunk.chunk_id in self.active_chunks:
                del self.active_chunks[chunk.chunk_id]
            
            raise e
    
    async def _check_memory_and_cleanup(self):
        """메모리 확인 및 필요시 정리"""
        current_stats = self.memory_monitor.get_current_stats()
        
        if current_stats.percent > self.config.memory_threshold * 100:
            self.logger.warning(f"메모리 사용률 높음: {current_stats.percent:.1f}%")
            await self._force_cleanup()
    
    async def _merge_chunk_results(self, chunks: List[FileChunk]) -> Dict[str, Any]:
        """청크 결과 통합"""
        
        merged_content = ""
        total_confidence = 0.0
        chunk_count = 0
        all_keywords = []
        processing_times = []
        
        for chunk in chunks:
            if chunk.status != ChunkStatus.COMPLETED:
                continue
            
            # 캐시된 결과 로드
            result_data = chunk.result_data
            if not result_data and chunk.status == ChunkStatus.CACHED:
                result_data = await self._load_cached_result(chunk)
            
            if result_data:
                merged_content += result_data.get("content", "")
                total_confidence += result_data.get("confidence", 0.0)
                all_keywords.extend(result_data.get("keywords", []))
                processing_times.append(chunk.processing_time)
                chunk_count += 1
        
        # 통합 지표 계산
        avg_confidence = total_confidence / max(chunk_count, 1)
        unique_keywords = list(set(all_keywords))
        total_processing_time = sum(processing_times)
        
        return {
            "merged_content": merged_content,
            "average_confidence": avg_confidence,
            "unique_keywords": unique_keywords,
            "total_keywords": len(all_keywords),
            "chunks_merged": chunk_count,
            "total_processing_time": total_processing_time,
            "average_chunk_time": total_processing_time / max(chunk_count, 1)
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        memory_report = self.memory_monitor.get_memory_report()
        
        return {
            "memory": memory_report,
            "processing": {
                "total_processed_mb": self.total_processed_mb,
                "total_processing_time": self.total_processing_time,
                "average_speed_mbps": self.total_processed_mb / max(self.total_processing_time, 1),
                "error_count": self.error_count,
                "active_chunks": len(self.active_chunks),
                "completed_chunks": len(self.completed_chunks)
            },
            "configuration": {
                "chunk_size_mb": self.config.chunk_size_mb,
                "max_memory_mb": self.config.max_memory_mb,
                "compression_enabled": self.config.compression_enabled,
                "max_concurrent_chunks": self.config.max_concurrent_chunks
            }
        }

# 사용 예시: 주얼리 STT 처리를 위한 스트리밍 프로세서
async def jewelry_stt_chunk_processor(chunk_data: bytes, chunk: FileChunk) -> Dict[str, Any]:
    """주얼리 STT용 청크 처리 함수"""
    
    # 임시 파일 생성
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_file.write(chunk_data)
    temp_file.close()
    
    try:
        # 실제로는 Whisper STT 호출
        # 여기서는 모의 처리
        await asyncio.sleep(0.1)  # STT 처리 시뮬레이션
        
        # 모의 STT 결과
        content = f"다이아몬드 4C 등급에 대해 설명드리겠습니다. (청크 {chunk.chunk_id})"
        
        # 주얼리 키워드 추출
        jewelry_keywords = ["다이아몬드", "4C", "등급", "캐럿", "컬러"]
        
        return {
            "content": content,
            "confidence": 0.95,
            "keywords": jewelry_keywords,
            "chunk_info": {
                "chunk_id": chunk.chunk_id,
                "size_mb": chunk.size_mb,
                "hash": chunk.content_hash
            }
        }
        
    finally:
        # 임시 파일 정리
        os.unlink(temp_file.name)

# 데모 함수
async def demo_streaming_processing():
    """스트리밍 처리 데모"""
    
    # 설정
    config = StreamingConfig(
        chunk_size_mb=10.0,  # 작은 청크로 테스트
        max_memory_mb=256.0,
        compression_enabled=True,
        max_concurrent_chunks=2
    )
    
    # 스트리밍 프로세서 생성
    processor = StreamingProcessor(config)
    await processor.start_monitoring()
    
    print("🚀 스트리밍 처리 엔진 데모 시작")
    print("=" * 60)
    
    # 진행률 콜백
    async def progress_callback(progress: float, message: str):
        print(f"📊 진행률: {progress:.1f}% - {message}")
    
    try:
        # 대형 파일 처리 시뮬레이션
        # 실제로는 사용자 업로드 파일 경로
        test_file = "test_large_audio.wav"
        
        # 모의 파일 생성 (실제 환경에서는 실제 파일 사용)
        if not os.path.exists(test_file):
            print("📝 테스트 파일 생성 중...")
            with open(test_file, 'wb') as f:
                # 100MB 모의 오디오 데이터
                f.write(b'0' * (100 * 1024 * 1024))
        
        # 스트리밍 처리 실행
        result = await processor.process_file_streaming(
            file_path=test_file,
            file_id="test_file_001",
            processor_func=jewelry_stt_chunk_processor,
            progress_callback=progress_callback
        )
        
        # 결과 출력
        print("\n" + "=" * 60)
        print("🎉 처리 완료!")
        print("=" * 60)
        
        if result["success"]:
            print(f"✅ 파일 ID: {result['file_id']}")
            print(f"📊 처리된 청크: {result['chunks_processed']}/{result['total_chunks']}")
            print(f"⏱️ 처리 시간: {result['processing_time']:.2f}초")
            print(f"📦 파일 크기: {result['total_size_mb']:.1f}MB")
            print(f"⚡ 평균 속도: {result['total_size_mb']/result['processing_time']:.1f}MB/s")
            print(f"🧠 메모리 피크: {result['memory_peak_mb']:.1f}MB")
            
            # 결과 상세
            final_result = result['result']
            print(f"\n📝 통합 결과:")
            print(f"   신뢰도: {final_result['average_confidence']:.1%}")
            print(f"   키워드 수: {final_result['total_keywords']}개")
            print(f"   고유 키워드: {len(final_result['unique_keywords'])}개")
            
        else:
            print(f"❌ 처리 실패: {result['error']}")
        
        # 처리 통계
        stats = processor.get_processing_stats()
        print(f"\n📈 성능 통계:")
        print(f"   메모리 사용률: {stats['memory']['current_percent']:.1f}%")
        print(f"   메모리 피크: {stats['memory']['peak_mb']:.1f}MB")
        print(f"   평균 처리 속도: {stats['processing']['average_speed_mbps']:.1f}MB/s")
        print(f"   오류 횟수: {stats['processing']['error_count']}")
        
        # 메모리 트렌드
        if stats['memory']['trend'] == 'increasing':
            print("⚠️ 메모리 사용량 증가 추세")
        else:
            print("✅ 메모리 사용량 안정")
    
    finally:
        await processor.stop_monitoring()
        
        # 테스트 파일 정리
        if os.path.exists(test_file):
            os.remove(test_file)
        
        print("\n🧹 정리 완료")

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 데모 실행
    asyncio.run(demo_streaming_processing())
