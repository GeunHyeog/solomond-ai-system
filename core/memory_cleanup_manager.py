#!/usr/bin/env python3
"""
메모리 정리 및 임시 파일 관리 시스템
메모리 누수 방지 및 성능 최적화
"""

import os
import gc
import psutil
import tempfile
import shutil
import threading
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import weakref

class MemoryCleanupManager:
    """메모리 및 파일 정리 통합 관리자"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.temp_files: Set[str] = set()
        self.temp_dirs: Set[str] = set()
        self.cleanup_thread: Optional[threading.Thread] = None
        self.cleanup_running = False
        self.max_memory_mb = 1500  # 1.5GB 제한
        self.cleanup_interval = 60  # 60초마다 정리
        
        # 약한 참조로 모델 추적
        self.active_models: weakref.WeakSet = weakref.WeakSet()
        
        # 자동 정리 시작
        self.start_auto_cleanup()
    
    def register_temp_file(self, file_path: str) -> None:
        """임시 파일 등록"""
        if os.path.exists(file_path):
            self.temp_files.add(os.path.abspath(file_path))
            self.logger.debug(f"임시 파일 등록: {file_path}")
    
    def register_temp_dir(self, dir_path: str) -> None:
        """임시 디렉토리 등록"""
        if os.path.exists(dir_path):
            self.temp_dirs.add(os.path.abspath(dir_path))
            self.logger.debug(f"임시 디렉토리 등록: {dir_path}")
    
    def register_model(self, model) -> None:
        """AI 모델 등록 (약한 참조)"""
        self.active_models.add(model)
        self.logger.debug(f"AI 모델 등록: {type(model).__name__}")
    
    def cleanup_temp_files(self) -> int:
        """임시 파일 정리"""
        cleaned_count = 0
        
        # 파일 정리
        for file_path in list(self.temp_files):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    cleaned_count += 1
                self.temp_files.discard(file_path)
            except (OSError, PermissionError) as e:
                self.logger.warning(f"임시 파일 삭제 실패 {file_path}: {e}")
        
        # 디렉토리 정리
        for dir_path in list(self.temp_dirs):
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path, ignore_errors=True)
                    cleaned_count += 1
                self.temp_dirs.discard(dir_path)
            except (OSError, PermissionError) as e:
                self.logger.warning(f"임시 디렉토리 삭제 실패 {dir_path}: {e}")
        
        return cleaned_count
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """강제 가비지 컬렉션"""
        before_objects = len(gc.get_objects())
        
        # 3번 연속 가비지 컬렉션 수행
        collected = 0
        for i in range(3):
            collected += gc.collect()
        
        after_objects = len(gc.get_objects())
        
        stats = {
            'collected_objects': collected,
            'before_objects': before_objects,
            'after_objects': after_objects,
            'freed_objects': before_objects - after_objects
        }
        
        self.logger.info(f"가비지 컬렉션 완료: {stats}")
        return stats
    
    def get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량 조회"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def check_memory_pressure(self) -> bool:
        """메모리 압박 상태 확인"""
        memory_usage = self.get_memory_usage()
        return memory_usage['rss_mb'] > self.max_memory_mb
    
    def emergency_cleanup(self) -> Dict[str, any]:
        """응급 메모리 정리"""
        self.logger.warning("응급 메모리 정리 시작")
        
        before_memory = self.get_memory_usage()
        
        # 1. 임시 파일 정리
        cleaned_files = self.cleanup_temp_files()
        
        # 2. 강제 가비지 컬렉션
        gc_stats = self.force_garbage_collection()
        
        # 3. 캐시 정리
        self.clear_caches()
        
        after_memory = self.get_memory_usage()
        
        result = {
            'cleaned_files': cleaned_files,
            'gc_stats': gc_stats,
            'before_memory_mb': before_memory['rss_mb'],
            'after_memory_mb': after_memory['rss_mb'],
            'freed_memory_mb': before_memory['rss_mb'] - after_memory['rss_mb']
        }
        
        self.logger.info(f"응급 정리 완료: {result}")
        return result
    
    def clear_caches(self) -> None:
        """캐시 정리"""
        try:
            # Python 내부 캐시 정리
            if hasattr(gc, 'clear_cache'):
                gc.clear_cache()
            
            # import 캐시 정리
            import sys
            cache_items = list(sys.modules.keys())
            for module_name in cache_items:
                if any(pattern in module_name for pattern in ['temp', 'cache', 'buffer']):
                    try:
                        del sys.modules[module_name]
                    except KeyError:
                        pass
                        
        except Exception as e:
            self.logger.warning(f"캐시 정리 중 오류: {e}")
    
    def auto_cleanup_worker(self) -> None:
        """자동 정리 워커 스레드"""
        while self.cleanup_running:
            try:
                # 메모리 압박 시 응급 정리
                if self.check_memory_pressure():
                    self.emergency_cleanup()
                else:
                    # 일반 정리
                    self.cleanup_temp_files()
                    self.force_garbage_collection()
                
                # 대기
                time.sleep(self.cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"자동 정리 중 오류: {e}")
                time.sleep(5)
    
    def start_auto_cleanup(self) -> None:
        """자동 정리 시작"""
        if not self.cleanup_running:
            self.cleanup_running = True
            self.cleanup_thread = threading.Thread(
                target=self.auto_cleanup_worker,
                daemon=True,
                name="MemoryCleanupWorker"
            )
            self.cleanup_thread.start()
            self.logger.info("자동 메모리 정리 시작됨")
    
    def stop_auto_cleanup(self) -> None:
        """자동 정리 중지"""
        self.cleanup_running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        self.logger.info("자동 메모리 정리 중지됨")
    
    def create_temp_file(self, suffix: str = "", prefix: str = "solomond_", 
                        dir: Optional[str] = None, delete: bool = False) -> str:
        """안전한 임시 파일 생성"""
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix, 
            prefix=prefix, 
            dir=dir, 
            delete=delete
        )
        
        if not delete:
            # 자동 정리를 위해 등록
            self.register_temp_file(temp_file.name)
        
        return temp_file.name
    
    def create_temp_dir(self, suffix: str = "", prefix: str = "solomond_", 
                       dir: Optional[str] = None) -> str:
        """안전한 임시 디렉토리 생성"""
        temp_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        self.register_temp_dir(temp_dir)
        return temp_dir
    
    def get_status(self) -> Dict[str, any]:
        """현재 상태 조회"""
        return {
            'memory_usage': self.get_memory_usage(),
            'temp_files_count': len(self.temp_files),
            'temp_dirs_count': len(self.temp_dirs),
            'active_models_count': len(self.active_models),
            'cleanup_running': self.cleanup_running,
            'memory_pressure': self.check_memory_pressure()
        }
    
    def __del__(self):
        """소멸자 - 정리 작업"""
        try:
            self.stop_auto_cleanup()
            self.cleanup_temp_files()
        except:
            pass

# 전역 인스턴스
global_memory_manager = MemoryCleanupManager()

def get_global_memory_manager() -> MemoryCleanupManager:
    """전역 메모리 관리자 반환"""
    return global_memory_manager

# 편의 함수들
def register_temp_file(file_path: str) -> None:
    """임시 파일 등록 (편의 함수)"""
    global_memory_manager.register_temp_file(file_path)

def register_temp_dir(dir_path: str) -> None:
    """임시 디렉토리 등록 (편의 함수)"""
    global_memory_manager.register_temp_dir(dir_path)

def register_model(model) -> None:
    """AI 모델 등록 (편의 함수)"""
    global_memory_manager.register_model(model)

def emergency_cleanup() -> Dict[str, any]:
    """응급 정리 (편의 함수)"""
    return global_memory_manager.emergency_cleanup()

def get_memory_status() -> Dict[str, any]:
    """메모리 상태 조회 (편의 함수)"""
    return global_memory_manager.get_status()

def create_safe_temp_file(suffix: str = "", prefix: str = "solomond_") -> str:
    """안전한 임시 파일 생성 (편의 함수)"""
    return global_memory_manager.create_temp_file(suffix=suffix, prefix=prefix)

def create_safe_temp_dir(suffix: str = "", prefix: str = "solomond_") -> str:
    """안전한 임시 디렉토리 생성 (편의 함수)"""
    return global_memory_manager.create_temp_dir(suffix=suffix, prefix=prefix)