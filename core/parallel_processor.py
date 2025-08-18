#!/usr/bin/env python3
"""
병렬 처리 및 비동기 처리 시스템
다중 파일 동시 분석 및 리소스 최적화
"""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import queue
import multiprocessing as mp
from pathlib import Path

# 최적화 시스템들 import
from .memory_cleanup_manager import get_global_memory_manager
from .optimized_model_loader import get_optimized_model_loader

class ProcessingMode(Enum):
    """처리 모드"""
    SEQUENTIAL = "sequential"      # 순차 처리
    PARALLEL_THREAD = "parallel_thread"    # 스레드 병렬
    PARALLEL_PROCESS = "parallel_process"  # 프로세스 병렬
    HYBRID = "hybrid"             # 하이브리드

@dataclass
class ProcessingTask:
    """처리 작업"""
    task_id: str
    file_path: str
    file_type: str
    priority: int = 0
    metadata: Dict[str, Any] = None

@dataclass
class ProcessingResult:
    """처리 결과"""
    task_id: str
    success: bool
    result: Any
    processing_time: float
    memory_usage: Dict[str, float]
    error_message: Optional[str] = None

class ParallelProcessor:
    """병렬 처리 시스템"""
    
    def __init__(self, max_workers: int = None, processing_mode: ProcessingMode = ProcessingMode.HYBRID):
        self.logger = logging.getLogger(__name__)
        self.memory_manager = get_global_memory_manager()
        
        # 최적 워커 수 계산
        if max_workers is None:
            cpu_count = mp.cpu_count()
            # 메모리 기반 제한 (각 프로세스당 약 500MB 가정)
            memory_limit = 4  # 4개 프로세스까지
            max_workers = min(cpu_count, memory_limit, 3)  # 최대 3개로 제한
        
        self.max_workers = max_workers
        self.processing_mode = processing_mode
        
        # 실행자들
        self.thread_with ThreadPoolExecutor() as executor:
    # ThreadPool 작업을 여기에 배치하세요
        self.process_executor = ProcessPoolExecutor(max_workers=min(max_workers, 2))  # 프로세스는 더 적게
        
        # 작업 큐
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        
        # 진행 상황 추적
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: Dict[str, ProcessingResult] = {}
        self.progress_callbacks: List[Callable] = []
        
        # 통계
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_processing_time": 0,
            "parallel_efficiency": 0
        }
        
        self.logger.info(f"병렬 처리 시스템 초기화: {max_workers}개 워커, {processing_mode.value} 모드")
    
    def add_progress_callback(self, callback: Callable) -> None:
        """진행 상황 콜백 추가"""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self, task_id: str, status: str, progress: float = 0) -> None:
        """진행 상황 알림"""
        for callback in self.progress_callbacks:
            try:
                callback(task_id, status, progress)
            except Exception as e:
                self.logger.warning(f"진행 상황 콜백 오류: {e}")
    
    def submit_task(self, task: ProcessingTask) -> None:
        """작업 제출"""
        self.task_queue.put((task.priority, task))
        self.stats["total_tasks"] += 1
        self.logger.debug(f"작업 제출: {task.task_id} ({task.file_type})")
    
    def submit_multiple_tasks(self, tasks: List[ProcessingTask]) -> None:
        """다중 작업 제출"""
        for task in tasks:
            self.submit_task(task)
        self.logger.info(f"다중 작업 제출 완료: {len(tasks)}개")
    
    def _process_single_file(self, task: ProcessingTask) -> ProcessingResult:
        """단일 파일 처리 (워커 함수)"""
        start_time = time.time()
        task_id = task.task_id
        
        try:
            self._notify_progress(task_id, "시작", 0)
            
            # 메모리 상태 체크
            before_memory = self.memory_manager.get_memory_usage()
            
            # 메모리 압박 시 정리
            if before_memory['rss_mb'] > 1000:  # 1GB 초과시
                self.memory_manager.emergency_cleanup()
            
            # 파일 타입별 처리
            if task.file_type == "audio":
                result = self._process_audio_file(task.file_path)
            elif task.file_type == "image":
                result = self._process_image_file(task.file_path)
            else:
                raise ValueError(f"지원하지 않는 파일 타입: {task.file_type}")
            
            after_memory = self.memory_manager.get_memory_usage()
            processing_time = time.time() - start_time
            
            self._notify_progress(task_id, "완료", 100)
            
            return ProcessingResult(
                task_id=task_id,
                success=True,
                result=result,
                processing_time=processing_time,
                memory_usage={
                    "before_mb": before_memory["rss_mb"],
                    "after_mb": after_memory["rss_mb"],
                    "peak_mb": max(before_memory["rss_mb"], after_memory["rss_mb"])
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            self._notify_progress(task_id, "실패", 0)
            self.logger.error(f"파일 처리 실패 {task_id}: {error_msg}")
            
            return ProcessingResult(
                task_id=task_id,
                success=False,
                result=None,
                processing_time=processing_time,
                memory_usage={"before_mb": 0, "after_mb": 0, "peak_mb": 0},
                error_message=error_msg
            )
    
    def _process_audio_file(self, file_path: str) -> Dict[str, Any]:
        """오디오 파일 처리"""
        from .real_analysis_engine_optimized import analyze_file_optimized
        return analyze_file_optimized(file_path, "audio")
    
    def _process_image_file(self, file_path: str) -> Dict[str, Any]:
        """이미지 파일 처리"""
        from .real_analysis_engine_optimized import analyze_file_optimized
        return analyze_file_optimized(file_path, "image")
    
    def process_batch_sequential(self, tasks: List[ProcessingTask]) -> List[ProcessingResult]:
        """순차 배치 처리"""
        self.logger.info(f"순차 배치 처리 시작: {len(tasks)}개 작업")
        
        results = []
        for i, task in enumerate(tasks):
            self.active_tasks[task.task_id] = task
            result = self._process_single_file(task)
            results.append(result)
            
            # 통계 업데이트
            if result.success:
                self.stats["completed_tasks"] += 1
            else:
                self.stats["failed_tasks"] += 1
            
            self.stats["total_processing_time"] += result.processing_time
            self.completed_tasks[task.task_id] = result
            del self.active_tasks[task.task_id]
        
        return results
    
    def process_batch_parallel_thread(self, tasks: List[ProcessingTask]) -> List[ProcessingResult]:
        """스레드 병렬 배치 처리"""
        self.logger.info(f"스레드 병렬 처리 시작: {len(tasks)}개 작업, {self.max_workers}개 워커")
        
        results = []
        
        # 활성 작업 등록
        for task in tasks:
            self.active_tasks[task.task_id] = task
        
        # 병렬 실행
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._process_single_file, task): task
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                result = future.result()
                results.append(result)
                
                # 통계 업데이트
                if result.success:
                    self.stats["completed_tasks"] += 1
                else:
                    self.stats["failed_tasks"] += 1
                
                self.stats["total_processing_time"] += result.processing_time
                self.completed_tasks[task.task_id] = result
                del self.active_tasks[task.task_id]
        
        return results
    
    def process_batch_parallel_process(self, tasks: List[ProcessingTask]) -> List[ProcessingResult]:
        """프로세스 병렬 배치 처리"""
        self.logger.info(f"프로세스 병렬 처리 시작: {len(tasks)}개 작업")
        
        # 프로세스 병렬은 메모리 사용량이 많으므로 제한적으로 사용
        limited_workers = min(self.max_workers, 2, len(tasks))
        
        results = []
        
        # 활성 작업 등록
        for task in tasks:
            self.active_tasks[task.task_id] = task
        
        # 프로세스 풀로 실행
        with ProcessPoolExecutor(max_workers=limited_workers) as executor:
            future_to_task = {
                executor.submit(_process_file_worker, task.file_path, task.file_type, task.task_id): task
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result_data = future.result()
                    result = ProcessingResult(
                        task_id=task.task_id,
                        success=result_data["success"],
                        result=result_data["result"],
                        processing_time=result_data["processing_time"],
                        memory_usage=result_data.get("memory_usage", {}),
                        error_message=result_data.get("error_message")
                    )
                except Exception as e:
                    result = ProcessingResult(
                        task_id=task.task_id,
                        success=False,
                        result=None,
                        processing_time=0,
                        memory_usage={},
                        error_message=str(e)
                    )
                
                results.append(result)
                
                # 통계 업데이트
                if result.success:
                    self.stats["completed_tasks"] += 1
                else:
                    self.stats["failed_tasks"] += 1
                
                self.stats["total_processing_time"] += result.processing_time
                self.completed_tasks[task.task_id] = result
                del self.active_tasks[task.task_id]
        
        return results
    
    def process_batch_hybrid(self, tasks: List[ProcessingTask]) -> List[ProcessingResult]:
        """하이브리드 배치 처리 (파일 타입별 최적화)"""
        self.logger.info(f"하이브리드 처리 시작: {len(tasks)}개 작업")
        
        # 파일 타입별 분류
        audio_tasks = [t for t in tasks if t.file_type == "audio"]
        image_tasks = [t for t in tasks if t.file_type == "image"]
        
        results = []
        
        # 오디오 파일은 스레드 병렬 (I/O 바운드)
        if audio_tasks:
            self.logger.info(f"오디오 파일 스레드 병렬 처리: {len(audio_tasks)}개")
            audio_results = self.process_batch_parallel_thread(audio_tasks)
            results.extend(audio_results)
        
        # 이미지 파일은 프로세스 병렬 (CPU 바운드)
        if image_tasks:
            self.logger.info(f"이미지 파일 프로세스 병렬 처리: {len(image_tasks)}개")
            
            # 이미지가 많으면 청크 단위로 처리
            if len(image_tasks) > 4:
                chunk_size = 2
                for i in range(0, len(image_tasks), chunk_size):
                    chunk = image_tasks[i:i+chunk_size]
                    chunk_results = self.process_batch_parallel_process(chunk)
                    results.extend(chunk_results)
                    
                    # 청크 간 메모리 정리
                    self.memory_manager.emergency_cleanup()
                    time.sleep(0.5)  # 잠시 대기
            else:
                image_results = self.process_batch_parallel_process(image_tasks)
                results.extend(image_results)
        
        return results
    
    def process_batch(self, tasks: List[ProcessingTask], 
                     mode: Optional[ProcessingMode] = None) -> List[ProcessingResult]:
        """배치 처리 (모드별)"""
        if not tasks:
            return []
        
        start_time = time.time()
        processing_mode = mode or self.processing_mode
        
        # 우선순위순 정렬
        tasks.sort(key=lambda t: t.priority, reverse=True)
        
        # 모드별 처리
        if processing_mode == ProcessingMode.SEQUENTIAL:
            results = self.process_batch_sequential(tasks)
        elif processing_mode == ProcessingMode.PARALLEL_THREAD:
            results = self.process_batch_parallel_thread(tasks)
        elif processing_mode == ProcessingMode.PARALLEL_PROCESS:
            results = self.process_batch_parallel_process(tasks)
        elif processing_mode == ProcessingMode.HYBRID:
            results = self.process_batch_hybrid(tasks)
        else:
            raise ValueError(f"지원하지 않는 처리 모드: {processing_mode}")
        
        total_time = time.time() - start_time
        
        # 병렬 효율성 계산
        sequential_time = sum(r.processing_time for r in results)
        if sequential_time > 0:
            self.stats["parallel_efficiency"] = sequential_time / total_time
        
        self.logger.info(
            f"배치 처리 완료: {len(tasks)}개 작업 "
            f"({total_time:.2f}s, 효율성: {self.stats['parallel_efficiency']:.2f}x)"
        )
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """처리 상태 조회"""
        return {
            "processing_mode": self.processing_mode.value,
            "max_workers": self.max_workers,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "stats": self.stats,
            "memory_status": self.memory_manager.get_status()
        }
    
    def shutdown(self) -> None:
        """처리 시스템 종료"""
        self.logger.info("병렬 처리 시스템 종료 중...")
        
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        self.logger.info("병렬 처리 시스템 종료 완료")

def _process_file_worker(file_path: str, file_type: str, task_id: str) -> Dict[str, Any]:
    """프로세스 워커 함수 (전역 함수로 분리 - pickle 지원)"""
    import time
    
    start_time = time.time()
    
    try:
        # 별도 프로세스에서 최적화된 분석 엔진 import
        from .real_analysis_engine_optimized import analyze_file_optimized
        
        result = analyze_file_optimized(file_path, file_type)
        processing_time = time.time() - start_time
        
        return {
            "success": result.get("success", False),
            "result": result,
            "processing_time": processing_time,
            "memory_usage": result.get("memory_usage", {}),
            "error_message": result.get("error") if not result.get("success", False) else None
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "success": False,
            "result": None,
            "processing_time": processing_time,
            "memory_usage": {},
            "error_message": str(e)
        }

# 전역 인스턴스
global_parallel_processor = ParallelProcessor()

def get_global_parallel_processor() -> ParallelProcessor:
    """전역 병렬 처리기 반환"""
    return global_parallel_processor

# 편의 함수들
def process_files_parallel(file_paths: List[str], 
                          mode: ProcessingMode = ProcessingMode.HYBRID) -> List[ProcessingResult]:
    """파일들을 병렬로 처리 (편의 함수)"""
    processor = get_global_parallel_processor()
    
    # 작업 생성
    tasks = []
    for i, file_path in enumerate(file_paths):
        # 파일 타입 자동 감지
        ext = Path(file_path).suffix.lower()
        if ext in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
            file_type = "audio"
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            file_type = "image"
        else:
            continue  # 지원하지 않는 파일 건너뛰기
        
        task = ProcessingTask(
            task_id=f"task_{i}_{Path(file_path).name}",
            file_path=file_path,
            file_type=file_type,
            priority=0
        )
        tasks.append(task)
    
    return processor.process_batch(tasks, mode)