#!/usr/bin/env python3
"""
최적화된 AI 모델 로더
싱글톤 패턴으로 모델 중복 로딩 방지, 지능적 캐싱 및 메모리 관리
"""

import os
import gc
import threading
import time
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import weakref
from pathlib import Path

# 메모리 관리 시스템 import
from .memory_cleanup_manager import get_global_memory_manager, register_model

class OptimizedModelLoader:
    """최적화된 AI 모델 로더"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """초기화 (한 번만 실행됨)"""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.logger = logging.getLogger(__name__)
        self.memory_manager = get_global_memory_manager()
        
        # 모델 캐시 (약한 참조 사용)
        self._model_cache: Dict[str, Any] = {}
        self._model_timestamps: Dict[str, datetime] = {}
        self._model_locks: Dict[str, threading.Lock] = {}
        
        # 설정
        self.max_cache_time = timedelta(minutes=30)  # 30분 후 캐시 만료
        self.max_cached_models = 3  # 최대 3개 모델만 캐싱
        
        self.logger.info("최적화된 모델 로더 초기화 완료")
    
    def _get_model_key(self, model_type: str, **kwargs) -> str:
        """모델 캐시 키 생성"""
        key_parts = [model_type]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return "|".join(key_parts)
    
    def _cleanup_expired_models(self) -> None:
        """만료된 모델 정리"""
        now = datetime.now()
        expired_keys = []
        
        for key, timestamp in self._model_timestamps.items():
            if now - timestamp > self.max_cache_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_model_from_cache(key)
            self.logger.info(f"만료된 모델 제거: {key}")
    
    def _remove_model_from_cache(self, key: str) -> None:
        """캐시에서 모델 제거"""
        if key in self._model_cache:
            model = self._model_cache.pop(key, None)
            self._model_timestamps.pop(key, None)
            self._model_locks.pop(key, None)
            
            # 메모리 정리
            if model is not None:
                del model
            gc.collect()
    
    def _ensure_cache_size_limit(self) -> None:
        """캐시 크기 제한 확인"""
        if len(self._model_cache) >= self.max_cached_models:
            # 가장 오래된 모델 제거
            oldest_key = min(self._model_timestamps.keys(), 
                           key=lambda k: self._model_timestamps[k])
            self._remove_model_from_cache(oldest_key)
            self.logger.info(f"캐시 크기 제한으로 모델 제거: {oldest_key}")
    
    def load_whisper_model(self, model_name: str = "base", device: str = "cpu") -> Any:
        """Whisper 모델 로딩 (최적화)"""
        import whisper
        
        key = self._get_model_key("whisper", model=model_name, device=device)
        
        # 캐시에서 확인
        if key in self._model_cache:
            self._model_timestamps[key] = datetime.now()  # 타임스탬프 갱신
            self.logger.info(f"Whisper 모델 캐시 히트: {key}")
            return self._model_cache[key]
        
        # 캐시 정리
        self._cleanup_expired_models()
        self._ensure_cache_size_limit()
        
        # 모델별 락 생성
        if key not in self._model_locks:
            self._model_locks[key] = threading.Lock()
        
        # 동시 로딩 방지
        with self._model_locks[key]:
            # 다시 한 번 캐시 확인 (다른 스레드에서 로딩했을 수 있음)
            if key in self._model_cache:
                return self._model_cache[key]
            
            self.logger.info(f"Whisper 모델 로딩 시작: {model_name} on {device}")
            start_time = time.time()
            
            try:
                # 메모리 사용량 모니터링
                before_memory = self.memory_manager.get_memory_usage()
                
                # 모델 로딩
                model = whisper.load_model(model_name, device=device)
                
                # 캐시 저장
                self._model_cache[key] = model
                self._model_timestamps[key] = datetime.now()
                
                # 메모리 관리자에 등록
                register_model(model)
                
                after_memory = self.memory_manager.get_memory_usage()
                load_time = time.time() - start_time
                
                self.logger.info(
                    f"Whisper 모델 로딩 완료: {key} "
                    f"({load_time:.2f}s, "
                    f"메모리: {before_memory['rss_mb']:.1f}→{after_memory['rss_mb']:.1f}MB)"
                )
                
                return model
                
            except Exception as e:
                self.logger.error(f"Whisper 모델 로딩 실패 {key}: {e}")
                # 실패한 키 정리
                self._model_locks.pop(key, None)
                raise
    
    def load_easyocr_model(self, lang_list: list = ['ko', 'en'], 
                          gpu: bool = False, verbose: bool = False) -> Any:
        """EasyOCR 모델 로딩 (최적화)"""
        import easyocr
        
        key = self._get_model_key("easyocr", 
                                 lang="|".join(sorted(lang_list)), 
                                 gpu=gpu)
        
        # 캐시에서 확인
        if key in self._model_cache:
            self._model_timestamps[key] = datetime.now()
            self.logger.info(f"EasyOCR 모델 캐시 히트: {key}")
            return self._model_cache[key]
        
        # 캐시 정리
        self._cleanup_expired_models()
        self._ensure_cache_size_limit()
        
        # 모델별 락 생성
        if key not in self._model_locks:
            self._model_locks[key] = threading.Lock()
        
        # 동시 로딩 방지
        with self._model_locks[key]:
            # 다시 한 번 캐시 확인
            if key in self._model_cache:
                return self._model_cache[key]
            
            self.logger.info(f"EasyOCR 모델 로딩 시작: {lang_list}, GPU={gpu}")
            start_time = time.time()
            
            try:
                # 메모리 사용량 모니터링
                before_memory = self.memory_manager.get_memory_usage()
                
                # 모델 로딩
                reader = easyocr.Reader(lang_list, gpu=gpu, verbose=verbose)
                
                # 캐시 저장
                self._model_cache[key] = reader
                self._model_timestamps[key] = datetime.now()
                
                # 메모리 관리자에 등록
                register_model(reader)
                
                after_memory = self.memory_manager.get_memory_usage()
                load_time = time.time() - start_time
                
                self.logger.info(
                    f"EasyOCR 모델 로딩 완료: {key} "
                    f"({load_time:.2f}s, "
                    f"메모리: {before_memory['rss_mb']:.1f}→{after_memory['rss_mb']:.1f}MB)"
                )
                
                return reader
                
            except Exception as e:
                self.logger.error(f"EasyOCR 모델 로딩 실패 {key}: {e}")
                self._model_locks.pop(key, None)
                raise
    
    def load_transformers_model(self, model_name: str, task: str = "summarization") -> Any:
        """Transformers 모델 로딩 (최적화)"""
        from transformers import pipeline
        
        key = self._get_model_key("transformers", model=model_name, task=task)
        
        # 캐시에서 확인
        if key in self._model_cache:
            self._model_timestamps[key] = datetime.now()
            self.logger.info(f"Transformers 모델 캐시 히트: {key}")
            return self._model_cache[key]
        
        # 캐시 정리
        self._cleanup_expired_models()
        self._ensure_cache_size_limit()
        
        # 모델별 락 생성
        if key not in self._model_locks:
            self._model_locks[key] = threading.Lock()
        
        # 동시 로딩 방지
        with self._model_locks[key]:
            # 다시 한 번 캐시 확인
            if key in self._model_cache:
                return self._model_cache[key]
            
            self.logger.info(f"Transformers 모델 로딩 시작: {model_name} for {task}")
            start_time = time.time()
            
            try:
                # 메모리 사용량 모니터링
                before_memory = self.memory_manager.get_memory_usage()
                
                # 모델 로딩
                model = pipeline(task, model=model_name, device=-1)  # CPU 사용
                
                # 캐시 저장
                self._model_cache[key] = model
                self._model_timestamps[key] = datetime.now()
                
                # 메모리 관리자에 등록
                register_model(model)
                
                after_memory = self.memory_manager.get_memory_usage()
                load_time = time.time() - start_time
                
                self.logger.info(
                    f"Transformers 모델 로딩 완료: {key} "
                    f"({load_time:.2f}s, "
                    f"메모리: {before_memory['rss_mb']:.1f}→{after_memory['rss_mb']:.1f}MB)"
                )
                
                return model
                
            except Exception as e:
                self.logger.error(f"Transformers 모델 로딩 실패 {key}: {e}")
                self._model_locks.pop(key, None)
                raise
    
    def clear_cache(self, model_type: Optional[str] = None) -> int:
        """캐시 정리"""
        if model_type is None:
            # 전체 캐시 정리
            removed_count = len(self._model_cache)
            for key in list(self._model_cache.keys()):
                self._remove_model_from_cache(key)
        else:
            # 특정 모델 타입만 정리
            removed_count = 0
            for key in list(self._model_cache.keys()):
                if key.startswith(model_type + "|"):
                    self._remove_model_from_cache(key)
                    removed_count += 1
        
        # 강제 가비지 컬렉션
        gc.collect()
        
        self.logger.info(f"모델 캐시 정리 완료: {removed_count}개 제거")
        return removed_count
    
    def get_cache_info(self) -> Dict[str, Any]:
        """캐시 정보 조회"""
        return {
            'cached_models': list(self._model_cache.keys()),
            'cache_count': len(self._model_cache),
            'max_cached_models': self.max_cached_models,
            'timestamps': {k: v.isoformat() for k, v in self._model_timestamps.items()},
            'memory_usage': self.memory_manager.get_memory_usage()
        }
    
    def preload_models(self, models_config: Dict[str, Dict]) -> None:
        """모델 사전 로딩"""
        self.logger.info("모델 사전 로딩 시작")
        
        for model_name, config in models_config.items():
            try:
                if config['type'] == 'whisper':
                    self.load_whisper_model(
                        model_name=config.get('model', 'base'),
                        device=config.get('device', 'cpu')
                    )
                elif config['type'] == 'easyocr':
                    self.load_easyocr_model(
                        lang_list=config.get('lang_list', ['ko', 'en']),
                        gpu=config.get('gpu', False)
                    )
                elif config['type'] == 'transformers':
                    self.load_transformers_model(
                        model_name=config.get('model_name'),
                        task=config.get('task', 'summarization')
                    )
                
            except Exception as e:
                self.logger.error(f"모델 사전 로딩 실패 {model_name}: {e}")
        
        self.logger.info("모델 사전 로딩 완료")

# 전역 인스턴스
global_model_loader = OptimizedModelLoader()

def get_optimized_model_loader() -> OptimizedModelLoader:
    """최적화된 모델 로더 반환"""
    return global_model_loader

# 편의 함수들
def load_whisper(model_name: str = "base", device: str = "cpu") -> Any:
    """Whisper 모델 로딩 (편의 함수)"""
    return global_model_loader.load_whisper_model(model_name, device)

def load_easyocr(lang_list: list = ['ko', 'en'], gpu: bool = False) -> Any:
    """EasyOCR 모델 로딩 (편의 함수)"""
    return global_model_loader.load_easyocr_model(lang_list, gpu)

def load_transformers(model_name: str, task: str = "summarization") -> Any:
    """Transformers 모델 로딩 (편의 함수)"""
    return global_model_loader.load_transformers_model(model_name, task)

def clear_model_cache(model_type: Optional[str] = None) -> int:
    """모델 캐시 정리 (편의 함수)"""
    return global_model_loader.clear_cache(model_type)

def get_model_cache_info() -> Dict[str, Any]:
    """모델 캐시 정보 조회 (편의 함수)"""
    return global_model_loader.get_cache_info()