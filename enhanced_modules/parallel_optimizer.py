#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[PARALLEL] 병렬 처리 최적화 시스템
Advanced Parallel Processing Optimization System

핵심 기능:
1. 동적 스레드 풀 관리
2. CPU 코어별 작업 분산
3. I/O 바운드 vs CPU 바운드 작업 구분
4. 배치 처리 최적화
5. 비동기 처리 지원
"""

import os
import sys
import threading
import multiprocessing
import concurrent.futures
import asyncio
import time
import psutil
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import queue
import logging
from pathlib import Path
import functools

# 로깅 설정
logger = logging.getLogger(__name__)

@dataclass
class TaskProfile:
    """작업 프로필"""
    task_type: str  # 'cpu_bound', 'io_bound', 'mixed'
    estimated_duration: float
    memory_requirement: float
    priority: int  # 1-10, 높을수록 우선순위 높음
    batch_size: Optional[int] = None

@dataclass
class ProcessingResult:
    """처리 결과"""
    task_id: str
    success: bool
    result: Any
    processing_time: float
    worker_id: str
    error_message: Optional[str] = None

class AdaptiveThreadPool:
    """적응형 스레드 풀"""
    
    def __init__(self, min_workers: int = 2, max_workers: int = None):
        self.min_workers = min_workers
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        
        # 현재 활성 스레드 풀
        self._thread_pools = {}
        self._lock = threading.Lock()
        
        # 성능 메트릭
        self.performance_history = []
        self.worker_stats = {}
        
        # CPU 정보
        self.cpu_count = multiprocessing.cpu_count()
        self.cpu_usage_threshold = 80.0  # CPU 사용률 임계값
        
        logger.info(f"[PARALLEL] 적응형 스레드 풀 초기화 - CPU: {self.cpu_count}개")
    
    def _get_optimal_workers(self, task_type: str, current_load: float) -> int:
        """최적 워커 수 계산"""
        if task_type == 'cpu_bound':
            # CPU 집약적 작업: CPU 코어 수 기반
            base_workers = self.cpu_count
            if current_load > self.cpu_usage_threshold:
                base_workers = max(1, int(base_workers * 0.7))  # 30% 감소
        elif task_type == 'io_bound':
            # I/O 집약적 작업: 더 많은 스레드 허용
            base_workers = min(self.max_workers, self.cpu_count * 2)
            if current_load < 50:
                base_workers = min(self.max_workers, int(base_workers * 1.5))  # 50% 증가
        else:
            # 혼합 작업: 중간값
            base_workers = max(2, int(self.cpu_count * 1.2))
        
        return max(self.min_workers, min(self.max_workers, base_workers))
    
    def get_thread_pool(self, task_type: str) -> concurrent.futures.ThreadPoolExecutor:
        """작업 유형별 스레드 풀 반환"""
        with self._lock:
            current_cpu_usage = psutil.cpu_percent(interval=0.1)
            optimal_workers = self._get_optimal_workers(task_type, current_cpu_usage)
            
            # 기존 풀이 있고 최적 워커 수가 같으면 재사용
            if task_type in self._thread_pools:
                pool_info = self._thread_pools[task_type]
                if pool_info['workers'] == optimal_workers:
                    return pool_info['pool']
                else:
                    # 워커 수가 변경되면 기존 풀 종료 후 재생성
                    pool_info['pool'].shutdown(wait=False)
                    del self._thread_pools[task_type]
            
            # 새 스레드 풀 생성
            new_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=optimal_workers,
                thread_name_prefix=f"parallel_{task_type}"
            )
            
            self._thread_pools[task_type] = {
                'pool': new_pool,
                'workers': optimal_workers,
                'created_at': time.time()
            }
            
            logger.info(f"[PARALLEL] {task_type} 스레드 풀 생성: {optimal_workers}개 워커")
            return new_pool
    
    def cleanup_pools(self):
        """모든 스레드 풀 정리"""
        with self._lock:
            for task_type, pool_info in self._thread_pools.items():
                pool_info['pool'].shutdown(wait=True)
                logger.info(f"[PARALLEL] {task_type} 스레드 풀 종료")
            self._thread_pools.clear()

class ParallelOptimizer:
    """병렬 처리 최적화 시스템"""
    
    def __init__(self):
        self.thread_pool_manager = AdaptiveThreadPool()
        self.task_queue = queue.PriorityQueue()
        self.results_cache = {}
        self.batch_processors = {}
        
        # 성능 통계
        self.processing_stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0
        }
        
        logger.info("[PARALLEL] 병렬 처리 최적화 시스템 초기화 완료")
    
    def process_batch(self, 
                     tasks: List[Tuple[Callable, tuple, dict]], 
                     task_profile: TaskProfile,
                     timeout: Optional[float] = None) -> List[ProcessingResult]:
        """배치 작업 병렬 처리"""
        start_time = time.time()
        results = []
        
        # 작업 유형에 맞는 스레드 풀 선택
        executor = self.thread_pool_manager.get_thread_pool(task_profile.task_type)
        
        # 배치 크기 최적화
        if task_profile.batch_size:
            # 지정된 배치 크기로 분할
            task_batches = [tasks[i:i + task_profile.batch_size] 
                          for i in range(0, len(tasks), task_profile.batch_size)]
        else:
            # 자동 배치 크기 계산
            optimal_batch_size = self._calculate_optimal_batch_size(len(tasks), task_profile)
            task_batches = [tasks[i:i + optimal_batch_size] 
                          for i in range(0, len(tasks), optimal_batch_size)]
        
        logger.info(f"[PARALLEL] 배치 처리 시작: {len(tasks)}개 작업, {len(task_batches)}개 배치")
        
        # 병렬 실행
        futures = []
        for batch_idx, batch in enumerate(task_batches):
            for task_idx, (func, args, kwargs) in enumerate(batch):
                task_id = f"batch_{batch_idx}_task_{task_idx}"
                future = executor.submit(self._execute_task_with_monitoring, 
                                       task_id, func, args, kwargs)
                futures.append((task_id, future))
        
        # 결과 수집
        for task_id, future in futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
                
                if result.success:
                    self.processing_stats['successful_tasks'] += 1
                else:
                    self.processing_stats['failed_tasks'] += 1
                    
            except concurrent.futures.TimeoutError:
                results.append(ProcessingResult(
                    task_id=task_id,
                    success=False,
                    result=None,
                    processing_time=timeout or 0,
                    worker_id="timeout",
                    error_message="작업 시간 초과"
                ))
                self.processing_stats['failed_tasks'] += 1
            except Exception as e:
                results.append(ProcessingResult(
                    task_id=task_id,
                    success=False,
                    result=None,
                    processing_time=0,
                    worker_id="error",
                    error_message=str(e)
                ))
                self.processing_stats['failed_tasks'] += 1
        
        # 통계 업데이트
        total_time = time.time() - start_time
        self.processing_stats['total_tasks'] += len(tasks)
        self.processing_stats['total_processing_time'] += total_time
        self.processing_stats['avg_processing_time'] = (
            self.processing_stats['total_processing_time'] / 
            max(1, self.processing_stats['total_tasks'])
        )
        
        logger.info(f"[PARALLEL] 배치 처리 완료: {total_time:.2f}초, 성공률: {len([r for r in results if r.success])}/{len(results)}")
        
        return results
    
    def _execute_task_with_monitoring(self, 
                                    task_id: str, 
                                    func: Callable, 
                                    args: tuple, 
                                    kwargs: dict) -> ProcessingResult:
        """모니터링을 포함한 작업 실행"""
        start_time = time.time()
        worker_id = threading.current_thread().name
        
        try:
            result = func(*args, **kwargs)
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                task_id=task_id,
                success=True,
                result=result,
                processing_time=processing_time,
                worker_id=worker_id
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.warning(f"[PARALLEL] 작업 실패 {task_id}: {e}")
            
            return ProcessingResult(
                task_id=task_id,
                success=False,
                result=None,
                processing_time=processing_time,
                worker_id=worker_id,
                error_message=str(e)
            )
    
    def _calculate_optimal_batch_size(self, total_tasks: int, task_profile: TaskProfile) -> int:
        """최적 배치 크기 계산"""
        # 기본 배치 크기
        if task_profile.task_type == 'cpu_bound':
            # CPU 집약적: 코어 수 기반
            base_batch_size = max(1, total_tasks // (multiprocessing.cpu_count() * 2))
        elif task_profile.task_type == 'io_bound':
            # I/O 집약적: 더 작은 배치로 반응성 향상
            base_batch_size = max(1, total_tasks // (multiprocessing.cpu_count() * 4))
        else:
            # 혼합: 중간값
            base_batch_size = max(1, total_tasks // (multiprocessing.cpu_count() * 3))
        
        # 메모리 요구사항 고려
        if task_profile.memory_requirement > 100:  # 100MB 이상
            base_batch_size = max(1, base_batch_size // 2)
        
        # 최소/최대 제한
        return max(1, min(base_batch_size, 50))
    
    async def process_async(self, 
                          tasks: List[Tuple[Callable, tuple, dict]], 
                          task_profile: TaskProfile,
                          max_concurrent: Optional[int] = None) -> List[ProcessingResult]:
        """비동기 병렬 처리"""
        if max_concurrent is None:
            max_concurrent = min(len(tasks), multiprocessing.cpu_count() * 2)
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_async_task(task_id: str, func: Callable, args: tuple, kwargs: dict):
            async with semaphore:
                start_time = time.time()
                try:
                    # CPU 집약적 작업은 스레드 풀에서 실행
                    if task_profile.task_type == 'cpu_bound':
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, func, *args)
                    else:
                        # I/O 작업 또는 코루틴
                        if asyncio.iscoroutinefunction(func):
                            result = await func(*args, **kwargs)
                        else:
                            result = func(*args, **kwargs)
                    
                    processing_time = time.time() - start_time
                    return ProcessingResult(
                        task_id=task_id,
                        success=True,
                        result=result,
                        processing_time=processing_time,
                        worker_id="async"
                    )
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    return ProcessingResult(
                        task_id=task_id,
                        success=False,
                        result=None,
                        processing_time=processing_time,
                        worker_id="async",
                        error_message=str(e)
                    )
        
        # 비동기 작업 생성
        async_tasks = [
            execute_async_task(f"async_task_{i}", func, args, kwargs)
            for i, (func, args, kwargs) in enumerate(tasks)
        ]
        
        # 모든 작업 완료 대기
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        # 예외 처리된 결과 변환
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    task_id=f"async_task_{i}",
                    success=False,
                    result=None,
                    processing_time=0,
                    worker_id="async",
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def optimize_for_files(self, 
                          file_paths: List[str], 
                          processing_function: Callable,
                          task_profile: TaskProfile) -> List[ProcessingResult]:
        """파일 처리 최적화"""
        # 파일 크기별로 정렬 (큰 파일부터 처리)
        files_with_sizes = []
        for file_path in file_paths:
            try:
                size = os.path.getsize(file_path)
                files_with_sizes.append((file_path, size))
            except OSError:
                files_with_sizes.append((file_path, 0))
        
        # 크기 순으로 정렬
        files_with_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # 작업 리스트 생성
        tasks = []
        for file_path, size in files_with_sizes:
            # 파일 크기를 task_profile에 추가
            modified_profile = TaskProfile(
                task_type=task_profile.task_type,
                estimated_duration=task_profile.estimated_duration * (size / (10 * 1024 * 1024)),  # 10MB 기준
                memory_requirement=task_profile.memory_requirement,
                priority=task_profile.priority,
                batch_size=task_profile.batch_size
            )
            
            tasks.append((processing_function, (file_path,), {'file_size': size}))
        
        return self.process_batch(tasks, modified_profile)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = self.processing_stats.copy()
        
        # 성공률 계산
        if stats['total_tasks'] > 0:
            stats['success_rate'] = stats['successful_tasks'] / stats['total_tasks']
            stats['failure_rate'] = stats['failed_tasks'] / stats['total_tasks']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        # 시스템 정보 추가
        stats['system_info'] = {
            'cpu_count': multiprocessing.cpu_count(),
            'current_cpu_usage': psutil.cpu_percent(interval=0.1),
            'available_memory_gb': psutil.virtual_memory().available / 1024**3,
            'active_thread_pools': len(self.thread_pool_manager._thread_pools)
        }
        
        return stats
    
    def cleanup(self):
        """리소스 정리"""
        self.thread_pool_manager.cleanup_pools()
        logger.info("[PARALLEL] 병렬 처리 시스템 정리 완료")

# 편의 함수들
def create_task_profile(task_type: str, 
                       estimated_duration: float = 1.0,
                       memory_requirement: float = 50.0,
                       priority: int = 5,
                       batch_size: Optional[int] = None) -> TaskProfile:
    """작업 프로필 생성 편의 함수"""
    return TaskProfile(
        task_type=task_type,
        estimated_duration=estimated_duration,
        memory_requirement=memory_requirement,
        priority=priority,
        batch_size=batch_size
    )

def parallel_map(func: Callable, 
                items: List[Any], 
                task_type: str = 'mixed',
                timeout: Optional[float] = None) -> List[Any]:
    """병렬 매핑 편의 함수"""
    optimizer = ParallelOptimizer()
    
    tasks = [(func, (item,), {}) for item in items]
    profile = create_task_profile(task_type)
    
    results = optimizer.process_batch(tasks, profile, timeout)
    optimizer.cleanup()
    
    return [r.result for r in results if r.success]

if __name__ == "__main__":
    # 병렬 처리 최적화 시스템 테스트
    def test_cpu_task(n):
        """CPU 집약적 테스트 작업"""
        result = sum(i * i for i in range(n))
        time.sleep(0.1)  # 약간의 처리 시간
        return result
    
    def test_io_task(delay):
        """I/O 시뮬레이션 테스트 작업"""
        time.sleep(delay)
        return f"IO task completed after {delay}s"
    
    print("병렬 처리 최적화 시스템 테스트 시작...")
    
    optimizer = ParallelOptimizer()
    
    # CPU 집약적 작업 테스트
    cpu_tasks = [(test_cpu_task, (1000 + i * 100,), {}) for i in range(10)]
    cpu_profile = create_task_profile('cpu_bound', estimated_duration=0.2)
    
    start_time = time.time()
    cpu_results = optimizer.process_batch(cpu_tasks, cpu_profile)
    cpu_time = time.time() - start_time
    
    print(f"CPU 작업 완료: {len(cpu_results)}개 작업, {cpu_time:.2f}초")
    print(f"성공한 작업: {len([r for r in cpu_results if r.success])}개")
    
    # I/O 집약적 작업 테스트
    io_tasks = [(test_io_task, (0.1,), {}) for _ in range(20)]
    io_profile = create_task_profile('io_bound', estimated_duration=0.1)
    
    start_time = time.time()
    io_results = optimizer.process_batch(io_tasks, io_profile)
    io_time = time.time() - start_time
    
    print(f"I/O 작업 완료: {len(io_results)}개 작업, {io_time:.2f}초")
    print(f"성공한 작업: {len([r for r in io_results if r.success])}개")
    
    # 성능 통계 출력
    stats = optimizer.get_performance_stats()
    print(f"전체 성공률: {stats['success_rate']:.2%}")
    print(f"평균 처리 시간: {stats['avg_processing_time']:.3f}초")
    
    # 정리
    optimizer.cleanup()
    print("병렬 처리 최적화 테스트 완료")