#!/usr/bin/env python3
"""
메모리 인식 파일 프로세서 v2.6
대용량 파일을 메모리 효율적으로 처리
"""

import gc
import time
import threading
import psutil
from typing import Dict, List, Any, Optional, Union, Callable, Iterator, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime
from contextlib import contextmanager

from .large_file_streaming_optimizer import (
    LargeFileStreamingOptimizer, 
    StreamingConfig, 
    FileChunk,
    get_global_streaming_optimizer
)

@dataclass
class ProcessingConfig:
    """파일 처리 설정"""
    max_memory_usage_mb: float = 512.0  # 최대 메모리 사용량
    memory_warning_threshold: float = 0.8  # 메모리 경고 임계값 (80%)
    memory_critical_threshold: float = 0.9  # 메모리 위험 임계값 (90%)
    auto_gc_enabled: bool = True  # 자동 가비지 컬렉션
    gc_frequency: int = 10  # GC 주기 (처리된 청크 수)
    batch_size: int = 5  # 배치 크기
    enable_progress_tracking: bool = True  # 진행률 추적
    memory_optimization_level: str = "balanced"  # conservative, balanced, aggressive

@dataclass
class ProcessingStats:
    """처리 통계"""
    files_processed: int = 0
    total_size_processed_mb: float = 0.0
    total_processing_time_seconds: float = 0.0
    peak_memory_usage_mb: float = 0.0
    average_processing_speed_mbps: float = 0.0
    gc_collections: int = 0
    memory_warnings: int = 0
    memory_pressure_events: int = 0
    successful_files: int = 0
    failed_files: int = 0
    optimization_actions: List[str] = field(default_factory=list)

class MemoryAwareFileProcessor:
    """메모리 인식 파일 프로세서"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.stats = ProcessingStats()
        self.is_processing = False
        self.cancel_requested = False
        self.lock = threading.RLock()
        self.logger = self._setup_logging()
        
        # 스트리밍 최적화 시스템
        streaming_config = StreamingConfig(
            max_memory_usage_mb=self.config.max_memory_usage_mb * 0.7,  # 70%를 스트리밍에 할당
            enable_memory_mapping=True,
            enable_async=True
        )
        self.streaming_optimizer = LargeFileStreamingOptimizer(streaming_config)
        
        # 메모리 모니터링
        self.memory_monitor = MemoryMonitor(self.config)
        
        # 처리 결과 임시 저장소
        self.temp_results = []
        self.temp_results_size_mb = 0.0
    
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
    
    @contextmanager
    def _memory_management_context(self):
        """메모리 관리 컨텍스트"""
        # 시작 시 메모리 상태 체크
        initial_memory = self.memory_monitor.get_current_usage_mb()
        
        try:
            yield
        finally:
            # 종료 시 메모리 정리
            if self.config.auto_gc_enabled:
                gc.collect()
                self.stats.gc_collections += 1
            
            # 메모리 사용량 체크
            final_memory = self.memory_monitor.get_current_usage_mb()
            memory_increase = final_memory - initial_memory
            
            if memory_increase > 50:  # 50MB 이상 증가
                self.logger.warning(f"⚠️ 메모리 사용량 증가: {memory_increase:.1f}MB")
                self.stats.memory_warnings += 1
    
    def _should_trigger_gc(self, processed_chunks: int) -> bool:
        """가비지 컬렉션 트리거 조건 확인"""
        if not self.config.auto_gc_enabled:
            return False
        
        # 주기적 GC
        if processed_chunks % self.config.gc_frequency == 0:
            return True
        
        # 메모리 압박 시 강제 GC
        current_usage = self.memory_monitor.get_current_usage_mb()
        usage_ratio = current_usage / self.config.max_memory_usage_mb
        
        if usage_ratio > self.config.memory_critical_threshold:
            self.stats.memory_pressure_events += 1
            return True
        
        return False
    
    def _optimize_memory_settings(self, file_size_mb: float) -> None:
        """파일 크기에 따른 메모리 설정 최적화"""
        optimization_level = self.config.memory_optimization_level
        
        if optimization_level == "conservative":
            # 보수적: 안전하게 작은 청크 사용
            chunk_size_mb = min(4.0, file_size_mb / 50)
            max_memory = self.config.max_memory_usage_mb * 0.5
            
        elif optimization_level == "aggressive":
            # 공격적: 큰 청크로 빠른 처리
            chunk_size_mb = min(32.0, file_size_mb / 5)
            max_memory = self.config.max_memory_usage_mb * 0.9
            
        else:  # balanced
            # 균형: 적절한 청크 크기
            chunk_size_mb = min(16.0, file_size_mb / 20)
            max_memory = self.config.max_memory_usage_mb * 0.7
        
        # 스트리밍 설정 업데이트
        self.streaming_optimizer.config.chunk_size_mb = chunk_size_mb
        self.streaming_optimizer.config.max_memory_usage_mb = max_memory
        self.streaming_optimizer.chunk_size_bytes = int(chunk_size_mb * 1024 * 1024)
        
        self.stats.optimization_actions.append(
            f"메모리 설정 최적화: 청크 {chunk_size_mb}MB, 최대 메모리 {max_memory}MB"
        )
        
        self.logger.info(f"🎯 메모리 설정 최적화 ({optimization_level}): 청크 {chunk_size_mb}MB")
    
    def process_file(self, 
                    file_path: Union[str, Path],
                    processor_func: Callable[[FileChunk], Any],
                    progress_callback: Optional[Callable[[float, Dict], None]] = None) -> List[Any]:
        """단일 파일 처리"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"🚀 파일 처리 시작: {file_path.name} ({file_size_mb:.1f}MB)")
        
        # 메모리 설정 최적화
        self._optimize_memory_settings(file_size_mb)
        
        # 스트리밍 최적화
        self.streaming_optimizer.optimize_for_file_type(file_path)
        
        results = []
        start_time = time.time()
        processed_chunks = 0
        
        try:
            with self._memory_management_context():
                # 진행률 콜백 설정
                def streaming_progress_callback(progress: float, chunk: FileChunk):
                    if progress_callback and self.config.enable_progress_tracking:
                        progress_info = {
                            'file_name': file_path.name,
                            'progress_percent': progress,
                            'chunk_id': chunk.chunk_id,
                            'memory_usage_mb': self.memory_monitor.get_current_usage_mb(),
                            'processing_speed_mbps': self._calculate_current_speed()
                        }
                        progress_callback(progress, progress_info)
                
                self.streaming_optimizer.config.progress_callback = streaming_progress_callback
                
                # 청크별 처리
                for chunk in self.streaming_optimizer.stream_file_chunks(file_path):
                    if self.cancel_requested:
                        break
                    
                    try:
                        # 메모리 압박 체크
                        if self.memory_monitor.is_memory_pressure():
                            self.logger.warning("⚠️ 메모리 압박 감지, 처리 일시 중단")
                            time.sleep(0.1)  # 짧은 대기
                            
                            # 강제 가비지 컬렉션
                            gc.collect()
                            self.stats.gc_collections += 1
                        
                        # 청크 처리
                        chunk_start_time = time.time()
                        result = processor_func(chunk)
                        
                        if result is not None:
                            results.append(result)
                            
                            # 결과 크기 추적 (메모리 관리용)
                            try:
                                import sys
                                result_size = sys.getsizeof(result) / (1024 * 1024)  # MB
                                self.temp_results_size_mb += result_size
                            except:
                                pass
                        
                        processed_chunks += 1
                        
                        # 주기적 가비지 컬렉션
                        if self._should_trigger_gc(processed_chunks):
                            gc.collect()
                            self.stats.gc_collections += 1
                        
                        # 배치 단위로 중간 결과 정리
                        if processed_chunks % self.config.batch_size == 0:
                            self._cleanup_intermediate_results()
                    
                    except Exception as e:
                        self.logger.error(f"❌ 청크 {chunk.chunk_id} 처리 실패: {e}")
                        continue
                
                # 최종 통계 업데이트
                processing_time = time.time() - start_time
                self.stats.files_processed += 1
                self.stats.total_size_processed_mb += file_size_mb
                self.stats.total_processing_time_seconds += processing_time
                
                if processing_time > 0:
                    speed_mbps = file_size_mb / processing_time
                    self.stats.average_processing_speed_mbps = (
                        (self.stats.average_processing_speed_mbps * (self.stats.files_processed - 1) + speed_mbps) / 
                        self.stats.files_processed
                    )
                
                # 메모리 피크 업데이트
                current_memory = self.memory_monitor.get_current_usage_mb()
                if current_memory > self.stats.peak_memory_usage_mb:
                    self.stats.peak_memory_usage_mb = current_memory
                
                self.stats.successful_files += 1
                self.logger.info(f"✅ 파일 처리 완료: {len(results)}개 결과, {speed_mbps:.1f}MB/s")
        
        except Exception as e:
            self.stats.failed_files += 1
            self.logger.error(f"❌ 파일 처리 실패: {e}")
            raise
        
        return results
    
    def process_files_batch(self,
                           file_paths: List[Union[str, Path]],
                           processor_func: Callable[[FileChunk], Any],
                           progress_callback: Optional[Callable[[float, Dict], None]] = None) -> Dict[str, List[Any]]:
        """여러 파일 배치 처리"""
        self.is_processing = True
        self.cancel_requested = False
        
        total_files = len(file_paths)
        results = {}
        
        try:
            for i, file_path in enumerate(file_paths):
                if self.cancel_requested:
                    break
                
                file_path = Path(file_path)
                
                # 전체 진행률 계산
                overall_progress = (i / total_files) * 100
                
                def batch_progress_callback(file_progress: float, progress_info: Dict):
                    # 파일별 진행률을 전체 진행률에 반영
                    adjusted_progress = overall_progress + (file_progress / total_files)
                    progress_info['overall_progress'] = adjusted_progress
                    progress_info['file_index'] = i + 1
                    progress_info['total_files'] = total_files
                    
                    if progress_callback:
                        progress_callback(adjusted_progress, progress_info)
                
                try:
                    self.logger.info(f"📁 파일 {i+1}/{total_files} 처리 중: {file_path.name}")
                    file_results = self.process_file(file_path, processor_func, batch_progress_callback)
                    results[str(file_path)] = file_results
                    
                except Exception as e:
                    self.logger.error(f"❌ 파일 {file_path.name} 처리 실패: {e}")
                    results[str(file_path)] = []
                    continue
                
                # 배치 간 메모리 정리
                if (i + 1) % 3 == 0:  # 3개 파일마다
                    self._cleanup_batch_memory()
        
        finally:
            self.is_processing = False
            
            # 최종 정리
            self._cleanup_intermediate_results()
            gc.collect()
            self.stats.gc_collections += 1
        
        self.logger.info(f"🎉 배치 처리 완료: {len(results)}개 파일")
        return results
    
    def _cleanup_intermediate_results(self) -> None:
        """중간 결과 정리"""
        if hasattr(self, 'temp_results'):
            self.temp_results.clear()
            self.temp_results_size_mb = 0.0
    
    def _cleanup_batch_memory(self) -> None:
        """배치 메모리 정리"""
        self._cleanup_intermediate_results()
        gc.collect()
        self.stats.gc_collections += 1
        
        self.logger.debug("🧹 배치 메모리 정리 완료")
    
    def _calculate_current_speed(self) -> float:
        """현재 처리 속도 계산"""
        if self.stats.total_processing_time_seconds > 0:
            return self.stats.total_size_processed_mb / self.stats.total_processing_time_seconds
        return 0.0
    
    def cancel_processing(self) -> None:
        """처리 취소"""
        self.cancel_requested = True
        self.streaming_optimizer.cancel_streaming()
        self.logger.info("🛑 파일 처리 취소 요청")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        return {
            'is_processing': self.is_processing,
            'files_processed': self.stats.files_processed,
            'successful_files': self.stats.successful_files,
            'failed_files': self.stats.failed_files,
            'total_size_processed_mb': self.stats.total_size_processed_mb,
            'total_processing_time_seconds': self.stats.total_processing_time_seconds,
            'average_processing_speed_mbps': self.stats.average_processing_speed_mbps,
            'peak_memory_usage_mb': self.stats.peak_memory_usage_mb,
            'current_memory_usage_mb': self.memory_monitor.get_current_usage_mb(),
            'memory_usage_percent': self.memory_monitor.get_usage_percent(),
            'gc_collections': self.stats.gc_collections,
            'memory_warnings': self.stats.memory_warnings,
            'memory_pressure_events': self.stats.memory_pressure_events,
            'optimization_actions': self.stats.optimization_actions,
            'memory_trend': self.memory_monitor.get_memory_trend(),
            'estimated_memory_efficiency': self._calculate_memory_efficiency()
        }
    
    def _calculate_memory_efficiency(self) -> float:
        """메모리 효율성 계산"""
        if self.stats.total_size_processed_mb > 0 and self.stats.peak_memory_usage_mb > 0:
            return self.stats.peak_memory_usage_mb / self.stats.total_size_processed_mb
        return 1.0
    
    def cleanup(self) -> None:
        """리소스 정리"""
        self.cancel_processing()
        self._cleanup_intermediate_results()
        self.streaming_optimizer.cleanup()
        
        gc.collect()
        self.logger.info("🧹 메모리 인식 파일 프로세서 정리 완료")

class MemoryMonitor:
    """메모리 모니터"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(f'{__name__}.MemoryMonitor')
        self.memory_history = []
        self.lock = threading.Lock()
    
    def get_current_usage_mb(self) -> float:
        """현재 메모리 사용량 (MB)"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # 히스토리 업데이트
            with self.lock:
                self.memory_history.append((time.time(), memory_mb))
                # 최근 100개만 유지
                if len(self.memory_history) > 100:
                    self.memory_history = self.memory_history[-100:]
            
            return memory_mb
        except Exception:
            return 0.0
    
    def get_usage_percent(self) -> float:
        """메모리 사용률 (%)"""
        current_usage = self.get_current_usage_mb()
        return (current_usage / self.config.max_memory_usage_mb) * 100
    
    def is_memory_pressure(self) -> bool:
        """메모리 압박 상태 확인"""
        usage_ratio = self.get_usage_percent() / 100
        return usage_ratio > self.config.memory_critical_threshold
    
    def get_memory_trend(self) -> str:
        """메모리 사용 추세"""
        with self.lock:
            if len(self.memory_history) < 10:
                return "insufficient_data"
            
            recent_values = [mem for _, mem in self.memory_history[-10:]]
            trend_sum = sum(recent_values[i] - recent_values[i-1] for i in range(1, len(recent_values)))
            
            if trend_sum > 5:  # 5MB 이상 증가
                return "increasing"
            elif trend_sum < -5:  # 5MB 이상 감소
                return "decreasing"
            else:
                return "stable"

# 전역 파일 프로세서
_global_file_processor = None
_global_processor_lock = threading.Lock()

def get_global_file_processor(config: Optional[ProcessingConfig] = None) -> MemoryAwareFileProcessor:
    """전역 파일 프로세서 가져오기"""
    global _global_file_processor
    
    with _global_processor_lock:
        if _global_file_processor is None:
            _global_file_processor = MemoryAwareFileProcessor(config)
        return _global_file_processor

# 편의 함수들
def process_large_file_memory_safe(file_path: Union[str, Path],
                                  processor_func: Callable[[FileChunk], Any],
                                  max_memory_mb: float = 512.0,
                                  optimization_level: str = "balanced",
                                  progress_callback: Optional[Callable] = None) -> List[Any]:
    """메모리 안전 대용량 파일 처리 (편의 함수)"""
    config = ProcessingConfig(
        max_memory_usage_mb=max_memory_mb,
        memory_optimization_level=optimization_level,
        auto_gc_enabled=True
    )
    
    processor = MemoryAwareFileProcessor(config)
    return processor.process_file(file_path, processor_func, progress_callback)

# 사용 예시
if __name__ == "__main__":
    # 테스트용 프로세서 함수
    def test_processor(chunk: FileChunk) -> Dict[str, Any]:
        """테스트용 청크 프로세서"""
        # 청크 분석 시뮬레이션
        time.sleep(0.01)  # 10ms 처리 시간
        
        return {
            'chunk_id': chunk.chunk_id,
            'size_mb': chunk.size_bytes / (1024 * 1024),
            'checksum': chunk.checksum,
            'is_compressed': chunk.is_compressed,
            'processing_time': datetime.now().isoformat()
        }
    
    # 진행률 콜백
    def progress_callback(progress: float, info: Dict):
        print(f"📊 진행률: {progress:.1f}% - {info.get('file_name', 'Unknown')} - 메모리: {info.get('memory_usage_mb', 0):.1f}MB")
    
    # 테스트 파일 경로 (실제 파일이 있다고 가정)
    test_file = Path("test_large_file.bin")
    
    if test_file.exists():
        print(f"🧪 메모리 인식 파일 프로세서 테스트: {test_file.name}")
        
        config = ProcessingConfig(
            max_memory_usage_mb=256.0,
            memory_optimization_level="balanced",
            auto_gc_enabled=True,
            enable_progress_tracking=True
        )
        
        processor = MemoryAwareFileProcessor(config)
        
        try:
            results = processor.process_file(test_file, test_processor, progress_callback)
            
            # 통계 출력
            stats = processor.get_processing_stats()
            print(f"\n📊 처리 결과:")
            print(f"  처리된 결과: {len(results)}개")
            print(f"  처리 속도: {stats['average_processing_speed_mbps']:.1f}MB/s")
            print(f"  피크 메모리: {stats['peak_memory_usage_mb']:.1f}MB")
            print(f"  메모리 효율성: {stats['estimated_memory_efficiency']:.2f}")
            print(f"  GC 횟수: {stats['gc_collections']}회")
            
        finally:
            processor.cleanup()
    
    else:
        print("⚠️ 테스트 파일이 없습니다. 실제 사용 시에는 대용량 파일을 지정하세요.")
    
    print("✅ 메모리 인식 파일 프로세서 테스트 완료!")