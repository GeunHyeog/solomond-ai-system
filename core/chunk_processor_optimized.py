#!/usr/bin/env python3
"""
대용량 파일 청크 처리 최적화 시스템
메모리 효율적인 스트리밍 처리 및 병렬 분석
"""

import os
import asyncio
import threading
import time
import math
import logging
from typing import Dict, List, Any, Optional, Callable, Generator, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from dataclasses import dataclass
from enum import Enum

# 메모리 관리 시스템 import
from .memory_cleanup_manager import get_global_memory_manager, create_safe_temp_file

class ChunkType(Enum):
    """청크 타입"""
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"

@dataclass
class ChunkInfo:
    """청크 정보"""
    index: int
    start_time: Optional[float]
    end_time: Optional[float]
    file_path: str
    chunk_type: ChunkType
    size_bytes: int
    metadata: Dict[str, Any]

@dataclass
class ProcessingResult:
    """처리 결과"""
    chunk_info: ChunkInfo
    result: Any
    processing_time: float
    success: bool
    error_message: Optional[str] = None

class ChunkProcessorOptimized:
    """최적화된 청크 프로세서"""
    
    def __init__(self, max_workers: int = 2, max_memory_mb: int = 500):
        self.logger = logging.getLogger(__name__)
        self.memory_manager = get_global_memory_manager()
        
        # 설정
        self.max_workers = max_workers
        self.max_memory_mb = max_memory_mb
        self.chunk_size_mb = 50  # 50MB 청크
        
        # 스레드풀
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ChunkProcessor")
        
        # 진행 상황 추적
        self.progress_callbacks: List[Callable] = []
        
        self.logger.info(f"청크 프로세서 초기화: {max_workers}개 워커, {max_memory_mb}MB 메모리 제한")
    
    def add_progress_callback(self, callback: Callable) -> None:
        """진행 상황 콜백 추가"""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self, current: int, total: int, message: str = "") -> None:
        """진행 상황 알림"""
        for callback in self.progress_callbacks:
            try:
                callback(current, total, message)
            except Exception as e:
                self.logger.warning(f"진행 상황 콜백 오류: {e}")
    
    def calculate_optimal_chunks(self, file_path: str, chunk_type: ChunkType) -> List[ChunkInfo]:
        """최적 청크 계산"""
        file_size = os.path.getsize(file_path)
        chunk_size_bytes = self.chunk_size_mb * 1024 * 1024
        
        if file_size <= chunk_size_bytes:
            # 작은 파일은 단일 청크
            return [ChunkInfo(
                index=0,
                start_time=None,
                end_time=None,
                file_path=file_path,
                chunk_type=chunk_type,
                size_bytes=file_size,
                metadata={}
            )]
        
        # 대용량 파일 청크 분할
        num_chunks = math.ceil(file_size / chunk_size_bytes)
        chunks = []
        
        for i in range(num_chunks):
            start_byte = i * chunk_size_bytes
            end_byte = min((i + 1) * chunk_size_bytes, file_size)
            chunk_size = end_byte - start_byte
            
            # 오디오/비디오 파일의 경우 시간 기반 청크 계산
            start_time = None
            end_time = None
            if chunk_type in [ChunkType.AUDIO, ChunkType.VIDEO]:
                try:
                    duration = self._get_media_duration(file_path)
                    if duration:
                        chunk_duration = duration / num_chunks
                        start_time = i * chunk_duration
                        end_time = min((i + 1) * chunk_duration, duration)
                except Exception as e:
                    self.logger.warning(f"미디어 길이 계산 실패: {e}")
            
            chunks.append(ChunkInfo(
                index=i,
                start_time=start_time,
                end_time=end_time,
                file_path=file_path,
                chunk_type=chunk_type,
                size_bytes=chunk_size,
                metadata={
                    'start_byte': start_byte,
                    'end_byte': end_byte,
                    'total_chunks': num_chunks
                }
            ))
        
        self.logger.info(f"청크 계산 완료: {file_path} -> {num_chunks}개 청크")
        return chunks
    
    def _get_media_duration(self, file_path: str) -> Optional[float]:
        """미디어 파일 길이 조회"""
        try:
            import librosa
            duration = librosa.get_duration(path=file_path)
            return duration
        except:
            try:
                import subprocess
                result = subprocess.run([
                    'ffprobe', '-v', 'error', '-show_entries', 
                    'format=duration', '-of', 
                    'default=noprint_wrappers=1:nokey=1', 
                    file_path
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    return float(result.stdout.strip())
            except Exception as e:
                self.logger.debug(f"ffprobe 실패: {e}")
        
        return None
    
    def _create_chunk_file(self, chunk_info: ChunkInfo) -> str:
        """청크 파일 생성"""
        if chunk_info.chunk_type == ChunkType.AUDIO or chunk_info.chunk_type == ChunkType.VIDEO:
            return self._create_media_chunk(chunk_info)
        else:
            return self._create_binary_chunk(chunk_info)
    
    def _create_media_chunk(self, chunk_info: ChunkInfo) -> str:
        """미디어 청크 생성 (시간 기반)"""
        if chunk_info.start_time is None or chunk_info.end_time is None:
            return chunk_info.file_path  # 단일 청크
        
        # FFmpeg로 청크 추출
        try:
            import subprocess
            
            chunk_file = create_safe_temp_file(
                suffix=f"_chunk_{chunk_info.index}.{Path(chunk_info.file_path).suffix}",
                prefix="media_chunk_"
            )
            
            duration = chunk_info.end_time - chunk_info.start_time
            
            cmd = [
                'ffmpeg', '-y',
                '-i', chunk_info.file_path,
                '-ss', str(chunk_info.start_time),
                '-t', str(duration),
                '-c', 'copy',  # 재인코딩 없이 복사
                chunk_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            
            if result.returncode == 0:
                self.logger.debug(f"미디어 청크 생성 완료: {chunk_file}")
                return chunk_file
            else:
                self.logger.error(f"FFmpeg 실패: {result.stderr.decode()}")
                return chunk_info.file_path
                
        except Exception as e:
            self.logger.error(f"미디어 청크 생성 실패: {e}")
            return chunk_info.file_path
    
    def _create_binary_chunk(self, chunk_info: ChunkInfo) -> str:
        """바이너리 청크 생성 (바이트 기반)"""
        if chunk_info.metadata.get('total_chunks', 1) == 1:
            return chunk_info.file_path  # 단일 청크
        
        try:
            chunk_file = create_safe_temp_file(
                suffix=f"_chunk_{chunk_info.index}.bin",
                prefix="binary_chunk_"
            )
            
            start_byte = chunk_info.metadata['start_byte']
            end_byte = chunk_info.metadata['end_byte']
            
            with open(chunk_info.file_path, 'rb') as src:
                src.seek(start_byte)
                with open(chunk_file, 'wb') as dst:
                    remaining = end_byte - start_byte
                    while remaining > 0:
                        chunk_size = min(1024 * 1024, remaining)  # 1MB씩 읽기
                        data = src.read(chunk_size)
                        if not data:
                            break
                        dst.write(data)
                        remaining -= len(data)
            
            self.logger.debug(f"바이너리 청크 생성 완료: {chunk_file}")
            return chunk_file
            
        except Exception as e:
            self.logger.error(f"바이너리 청크 생성 실패: {e}")
            return chunk_info.file_path
    
    def _process_single_chunk(self, chunk_info: ChunkInfo, 
                            processor_func: Callable) -> ProcessingResult:
        """단일 청크 처리"""
        start_time = time.time()
        
        try:
            # 메모리 체크
            memory_usage = self.memory_manager.get_memory_usage()
            if memory_usage['rss_mb'] > self.max_memory_mb:
                self.memory_manager.emergency_cleanup()
            
            # 청크 파일 생성
            chunk_file = self._create_chunk_file(chunk_info)
            
            # 실제 처리
            result = processor_func(chunk_file, chunk_info)
            
            processing_time = time.time() - start_time
            
            self.logger.debug(
                f"청크 처리 완료: {chunk_info.index} "
                f"({processing_time:.2f}s)"
            )
            
            return ProcessingResult(
                chunk_info=chunk_info,
                result=result,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            self.logger.error(f"청크 처리 실패 {chunk_info.index}: {error_msg}")
            
            return ProcessingResult(
                chunk_info=chunk_info,
                result=None,
                processing_time=processing_time,
                success=False,
                error_message=error_msg
            )
    
    def process_file_chunked(self, file_path: str, chunk_type: ChunkType,
                           processor_func: Callable,
                           merger_func: Optional[Callable] = None) -> Dict[str, Any]:
        """파일 청크 단위 처리"""
        self.logger.info(f"청크 처리 시작: {file_path}")
        
        # 청크 계산
        chunks = self.calculate_optimal_chunks(file_path, chunk_type)
        total_chunks = len(chunks)
        
        # 진행 상황 초기화
        self._notify_progress(0, total_chunks, "청크 처리 시작")
        
        # 병렬 처리
        results = []
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 청크 처리 작업 제출
            future_to_chunk = {
                executor.submit(self._process_single_chunk, chunk, processor_func): chunk
                for chunk in chunks
            }
            
            # 완료된 작업 수집
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                result = future.result()
                results.append(result)
                completed_count += 1
                
                # 진행 상황 알림
                self._notify_progress(
                    completed_count, 
                    total_chunks,
                    f"청크 {completed_count}/{total_chunks} 완료"
                )
        
        # 결과 정렬 (청크 순서대로)
        results.sort(key=lambda r: r.chunk_info.index)
        
        # 성공/실패 분리
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        # 결과 병합
        final_result = None
        if successful_results and merger_func:
            try:
                final_result = merger_func([r.result for r in successful_results])
            except Exception as e:
                self.logger.error(f"결과 병합 실패: {e}")
        elif successful_results:
            final_result = [r.result for r in successful_results]
        
        # 통계 계산
        total_time = sum(r.processing_time for r in results)
        success_rate = len(successful_results) / total_chunks if total_chunks > 0 else 0
        
        summary = {
            'success': len(failed_results) == 0,
            'total_chunks': total_chunks,
            'successful_chunks': len(successful_results),
            'failed_chunks': len(failed_results),
            'success_rate': success_rate,
            'total_processing_time': total_time,
            'average_chunk_time': total_time / total_chunks if total_chunks > 0 else 0,
            'result': final_result,
            'errors': [r.error_message for r in failed_results if r.error_message]
        }
        
        self.logger.info(
            f"청크 처리 완료: {file_path} "
            f"({successful_results}/{total_chunks} 성공, "
            f"{total_time:.2f}s)"
        )
        
        return summary
    
    def shutdown(self) -> None:
        """프로세서 종료"""
        self.executor.shutdown(wait=True)
        self.logger.info("청크 프로세서 종료됨")

# 전역 인스턴스
global_chunk_processor = ChunkProcessorOptimized()

def get_global_chunk_processor() -> ChunkProcessorOptimized:
    """전역 청크 프로세서 반환"""
    return global_chunk_processor

# 편의 함수들
def process_audio_chunked(file_path: str, processor_func: Callable,
                         merger_func: Optional[Callable] = None) -> Dict[str, Any]:
    """오디오 파일 청크 처리 (편의 함수)"""
    return global_chunk_processor.process_file_chunked(
        file_path, ChunkType.AUDIO, processor_func, merger_func
    )

def process_image_chunked(file_path: str, processor_func: Callable,
                         merger_func: Optional[Callable] = None) -> Dict[str, Any]:
    """이미지 파일 청크 처리 (편의 함수)"""
    return global_chunk_processor.process_file_chunked(
        file_path, ChunkType.IMAGE, processor_func, merger_func
    )

def process_video_chunked(file_path: str, processor_func: Callable,
                         merger_func: Optional[Callable] = None) -> Dict[str, Any]:
    """비디오 파일 청크 처리 (편의 함수)"""
    return global_chunk_processor.process_file_chunked(
        file_path, ChunkType.VIDEO, processor_func, merger_func
    )