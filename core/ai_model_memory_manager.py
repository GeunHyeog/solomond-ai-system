#!/usr/bin/env python3
"""
AI 모델 메모리 관리 시스템 v2.6
Whisper, EasyOCR, Transformers 등 AI 모델의 지능적 메모리 관리
"""

import gc
import time
import threading
import psutil
import torch
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import weakref
from enum import Enum
import json

class ModelType(Enum):
    """AI 모델 타입"""
    WHISPER = "whisper"
    EASYOCR = "easyocr"
    TRANSFORMERS = "transformers"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    CUSTOM = "custom"

class ModelPriority(Enum):
    """모델 우선순위"""
    CRITICAL = 1  # 항상 메모리에 유지
    HIGH = 2      # 가능한 한 유지
    MEDIUM = 3    # 필요시 언로드
    LOW = 4       # 적극적으로 언로드

@dataclass
class ModelInfo:
    """AI 모델 정보"""
    model_id: str
    model_type: ModelType
    priority: ModelPriority
    memory_usage_mb: float = 0.0
    last_accessed: datetime = field(default_factory=datetime.now)
    load_count: int = 0
    reference: Optional[Any] = None
    device: str = "cpu"
    model_size_mb: float = 0.0
    initialization_time_seconds: float = 0.0
    is_loaded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryProfile:
    """메모리 프로파일"""
    total_memory_mb: float
    available_memory_mb: float
    ai_models_memory_mb: float
    system_memory_mb: float
    gpu_memory_mb: float = 0.0
    gpu_available_mb: float = 0.0
    memory_pressure_level: str = "low"  # low, medium, high, critical

class AIModelMemoryManager:
    """AI 모델 메모리 관리 시스템"""
    
    def __init__(self, max_memory_mb: float = 2048.0, enable_gpu_management: bool = True):
        self.max_memory_mb = max_memory_mb
        self.enable_gpu_management = enable_gpu_management
        self.lock = threading.RLock()
        self.logger = self._setup_logging()
        
        # 모델 레지스트리
        self.models: Dict[str, ModelInfo] = {}
        self.weak_references: Dict[str, weakref.ref] = {}
        
        # 메모리 모니터링
        self.memory_monitor_thread = None
        self.monitoring_enabled = True
        self.monitoring_interval = 5.0  # 5초 간격
        
        # 통계
        self.stats = {
            'models_loaded': 0,
            'models_unloaded': 0,
            'memory_optimizations': 0,
            'total_memory_saved_mb': 0.0,
            'last_optimization': None
        }
        
        # 설정
        self.auto_unload_timeout = 300.0  # 5분 후 자동 언로드
        self.memory_warning_threshold = 0.8  # 80% 사용 시 경고
        self.memory_critical_threshold = 0.9  # 90% 사용 시 위험
        
        # GPU 관리
        self.gpu_available = False
        if enable_gpu_management:
            self._initialize_gpu_management()
        
        # 모니터링 시작
        self._start_memory_monitoring()
        
        self.logger.info("🧠 AI 모델 메모리 관리 시스템 초기화 완료")
    
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
    
    def _initialize_gpu_management(self) -> None:
        """GPU 관리 초기화"""
        try:
            if torch.cuda.is_available():
                self.gpu_available = True
                gpu_count = torch.cuda.device_count()
                self.logger.info(f"🎮 GPU 관리 활성화: {gpu_count}개 GPU 감지")
                
                # GPU 메모리 정보 출력
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    total_memory = props.total_memory / (1024**3)  # GB
                    self.logger.info(f"  GPU {i}: {props.name} ({total_memory:.1f}GB)")
            else:
                self.logger.info("💻 CPU 전용 모드로 실행")
        except Exception as e:
            self.logger.warning(f"⚠️ GPU 초기화 실패: {e}")
    
    def register_model(self, 
                      model_id: str,
                      model_type: ModelType,
                      model_obj: Any,
                      priority: ModelPriority = ModelPriority.MEDIUM,
                      device: str = "cpu",
                      metadata: Optional[Dict] = None) -> bool:
        """AI 모델 등록"""
        with self.lock:
            try:
                # 메모리 사용량 계산
                memory_usage = self._calculate_model_memory(model_obj)
                model_size = self._estimate_model_size(model_obj, model_type)
                
                # 모델 정보 생성
                model_info = ModelInfo(
                    model_id=model_id,
                    model_type=model_type,
                    priority=priority,
                    memory_usage_mb=memory_usage,
                    reference=model_obj,
                    device=device,
                    model_size_mb=model_size,
                    is_loaded=True,
                    metadata=metadata or {}
                )
                
                # 약한 참조 생성
                def cleanup_callback(ref):
                    self._handle_model_cleanup(model_id)
                
                weak_ref = weakref.ref(model_obj, cleanup_callback)
                
                # 등록
                self.models[model_id] = model_info
                self.weak_references[model_id] = weak_ref
                
                self.stats['models_loaded'] += 1
                
                self.logger.info(f"✅ 모델 등록: {model_id} ({memory_usage:.1f}MB, {priority.name})")
                
                # 메모리 압박 체크
                self._check_memory_pressure()
                
                return True
                
            except Exception as e:
                self.logger.error(f"❌ 모델 등록 실패 {model_id}: {e}")
                return False
    
    def unregister_model(self, model_id: str) -> bool:
        """AI 모델 등록 해제"""
        with self.lock:
            if model_id in self.models:
                model_info = self.models[model_id]
                
                # 메모리 정리
                if model_info.reference:
                    del model_info.reference
                
                # GPU 메모리 정리
                if model_info.device.startswith('cuda'):
                    self._cleanup_gpu_memory()
                
                # 등록 해제
                del self.models[model_id]
                if model_id in self.weak_references:
                    del self.weak_references[model_id]
                
                self.stats['models_unloaded'] += 1
                self.stats['total_memory_saved_mb'] += model_info.memory_usage_mb
                
                self.logger.info(f"🗑️ 모델 등록 해제: {model_id} ({model_info.memory_usage_mb:.1f}MB 해제)")
                
                return True
            return False
    
    def access_model(self, model_id: str) -> Optional[Any]:
        """모델 접근 (마지막 접근 시간 업데이트)"""
        with self.lock:
            if model_id in self.models:
                model_info = self.models[model_id]
                model_info.last_accessed = datetime.now()
                model_info.load_count += 1
                
                # 약한 참조에서 모델 가져오기
                if model_id in self.weak_references:
                    weak_ref = self.weak_references[model_id]
                    model_obj = weak_ref()
                    
                    if model_obj is not None:
                        self.logger.debug(f"🎯 모델 접근: {model_id}")
                        return model_obj
                    else:
                        # 모델이 GC되었음
                        self._handle_model_cleanup(model_id)
                
            return None
    
    def optimize_memory(self, target_memory_mb: Optional[float] = None) -> Dict[str, Any]:
        """메모리 최적화 실행"""
        with self.lock:
            optimization_start = time.time()
            initial_memory = self.get_memory_profile()
            
            if target_memory_mb is None:
                target_memory_mb = self.max_memory_mb * 0.8  # 80% 목표
            
            # 최적화 전략 결정
            current_ai_memory = initial_memory.ai_models_memory_mb
            memory_to_free = max(0, current_ai_memory - target_memory_mb)
            
            if memory_to_free <= 0:
                return {
                    'status': 'no_action_needed',
                    'initial_memory_mb': current_ai_memory,
                    'target_memory_mb': target_memory_mb,
                    'optimization_time_ms': 0
                }
            
            # 언로드 후보 모델 선정
            candidates = self._select_unload_candidates(memory_to_free)
            
            freed_memory = 0.0
            unloaded_models = []
            
            # 모델 순차적 언로드
            for model_id, model_info in candidates:
                if freed_memory >= memory_to_free:
                    break
                
                success = self._unload_model(model_id)
                if success:
                    freed_memory += model_info.memory_usage_mb
                    unloaded_models.append(model_id)
            
            # GPU 메모리 정리
            if self.gpu_available:
                self._cleanup_gpu_memory()
            
            # 가비지 컬렉션
            gc.collect()
            
            # 최종 메모리 상태
            final_memory = self.get_memory_profile()
            optimization_time = (time.time() - optimization_start) * 1000
            
            # 통계 업데이트
            self.stats['memory_optimizations'] += 1
            self.stats['last_optimization'] = datetime.now().isoformat()
            
            result = {
                'status': 'success',
                'initial_memory_mb': current_ai_memory,
                'final_memory_mb': final_memory.ai_models_memory_mb,
                'target_memory_mb': target_memory_mb,
                'freed_memory_mb': freed_memory,
                'unloaded_models': unloaded_models,
                'optimization_time_ms': optimization_time
            }
            
            self.logger.info(f"🚀 메모리 최적화 완료: {freed_memory:.1f}MB 해제, {len(unloaded_models)}개 모델 언로드")
            
            return result
    
    def _select_unload_candidates(self, target_memory_mb: float) -> List[tuple]:
        """언로드 후보 모델 선정"""
        candidates = []
        
        for model_id, model_info in self.models.items():
            # 중요도가 CRITICAL인 모델은 제외
            if model_info.priority == ModelPriority.CRITICAL:
                continue
            
            # 언로드 점수 계산
            time_since_access = (datetime.now() - model_info.last_accessed).total_seconds()
            
            # 점수 = 우선순위 가중치 + 시간 가중치 - 사용 빈도 가중치
            priority_weight = model_info.priority.value * 100
            time_weight = min(time_since_access / 60, 300)  # 최대 5분
            usage_weight = -model_info.load_count * 10
            
            score = priority_weight + time_weight + usage_weight
            
            candidates.append((model_id, model_info, score))
        
        # 점수 순으로 정렬 (높은 점수 = 언로드 우선)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # 목표 메모리까지 선택
        selected = []
        cumulative_memory = 0.0
        
        for model_id, model_info, score in candidates:
            if cumulative_memory >= target_memory_mb:
                break
            
            selected.append((model_id, model_info))
            cumulative_memory += model_info.memory_usage_mb
        
        return selected
    
    def _unload_model(self, model_id: str) -> bool:
        """특정 모델 언로드"""
        try:
            if model_id in self.models:
                model_info = self.models[model_id]
                
                # 참조 제거
                if model_info.reference:
                    del model_info.reference
                    model_info.reference = None
                
                model_info.is_loaded = False
                
                # GPU 메모리 정리
                if model_info.device.startswith('cuda'):
                    torch.cuda.empty_cache()
                
                self.logger.debug(f"📤 모델 언로드: {model_id}")
                return True
        except Exception as e:
            self.logger.error(f"❌ 모델 언로드 실패 {model_id}: {e}")
        
        return False
    
    def _calculate_model_memory(self, model_obj: Any) -> float:
        """모델 메모리 사용량 계산"""
        try:
            import sys
            
            # PyTorch 모델
            if hasattr(model_obj, 'parameters'):
                total_params = sum(p.numel() for p in model_obj.parameters())
                # 대략적인 메모리 계산 (파라미터 수 × 4바이트)
                memory_bytes = total_params * 4
                return memory_bytes / (1024 * 1024)  # MB
            
            # 기본 객체 크기
            return sys.getsizeof(model_obj) / (1024 * 1024)
            
        except Exception:
            return 0.0
    
    def _estimate_model_size(self, model_obj: Any, model_type: ModelType) -> float:
        """모델 크기 추정"""
        try:
            # 모델 타입별 추정
            if model_type == ModelType.WHISPER:
                # Whisper 모델 크기 추정
                if hasattr(model_obj, 'dims'):
                    if model_obj.dims.n_audio_state >= 1280:  # large
                        return 1550.0  # MB
                    elif model_obj.dims.n_audio_state >= 1024:  # medium
                        return 769.0
                    elif model_obj.dims.n_audio_state >= 768:  # small
                        return 244.0
                    else:  # base/tiny
                        return 142.0
                return 500.0  # 기본값
            
            elif model_type == ModelType.EASYOCR:
                return 100.0  # EasyOCR 기본 크기
            
            elif model_type == ModelType.TRANSFORMERS:
                # Transformers 모델 크기 추정
                if hasattr(model_obj, 'config'):
                    config = model_obj.config
                    if hasattr(config, 'hidden_size'):
                        # 대략적인 추정 공식
                        hidden_size = config.hidden_size
                        num_layers = getattr(config, 'num_hidden_layers', 12)
                        vocab_size = getattr(config, 'vocab_size', 50000)
                        
                        # 파라미터 수 추정
                        params = hidden_size * hidden_size * num_layers * 8 + vocab_size * hidden_size
                        return (params * 4) / (1024 * 1024)  # MB
                
                return 400.0  # 기본값
            
            else:
                return self._calculate_model_memory(model_obj)
                
        except Exception:
            return 100.0  # 기본값
    
    def _cleanup_gpu_memory(self) -> None:
        """GPU 메모리 정리"""
        if self.gpu_available:
            try:
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.logger.debug("🎮 GPU 메모리 정리 완료")
            except Exception as e:
                self.logger.warning(f"⚠️ GPU 메모리 정리 실패: {e}")
    
    def _handle_model_cleanup(self, model_id: str) -> None:
        """모델 정리 콜백"""
        with self.lock:
            if model_id in self.models:
                self.logger.debug(f"🧹 자동 모델 정리: {model_id}")
                self.models[model_id].is_loaded = False
                self.models[model_id].reference = None
    
    def _start_memory_monitoring(self) -> None:
        """메모리 모니터링 시작"""
        if self.memory_monitor_thread is None or not self.memory_monitor_thread.is_alive():
            self.memory_monitor_thread = threading.Thread(
                target=self._memory_monitoring_loop,
                daemon=True
            )
            self.memory_monitor_thread.start()
            self.logger.debug("📊 메모리 모니터링 시작")
    
    def _memory_monitoring_loop(self) -> None:
        """메모리 모니터링 루프"""
        while self.monitoring_enabled:
            try:
                profile = self.get_memory_profile()
                
                # 메모리 압박 체크
                if profile.memory_pressure_level in ['high', 'critical']:
                    self.logger.warning(f"⚠️ 메모리 압박 감지: {profile.memory_pressure_level}")
                    
                    if profile.memory_pressure_level == 'critical':
                        # 긴급 최적화 실행
                        self.optimize_memory(self.max_memory_mb * 0.6)
                
                # 자동 언로드 체크
                self._check_auto_unload()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"❌ 메모리 모니터링 오류: {e}")
                time.sleep(10)  # 오류 시 더 긴 대기
    
    def _check_memory_pressure(self) -> None:
        """메모리 압박 상태 체크"""
        profile = self.get_memory_profile()
        
        if profile.memory_pressure_level == 'critical':
            self.logger.warning("🚨 심각한 메모리 압박! 긴급 최적화 실행")
            self.optimize_memory(self.max_memory_mb * 0.5)
        elif profile.memory_pressure_level == 'high':
            self.logger.warning("⚠️ 높은 메모리 압박 감지")
    
    def _check_auto_unload(self) -> None:
        """자동 언로드 체크"""
        current_time = datetime.now()
        candidates = []
        
        with self.lock:
            for model_id, model_info in self.models.items():
                if model_info.priority == ModelPriority.CRITICAL:
                    continue
                
                time_since_access = (current_time - model_info.last_accessed).total_seconds()
                
                if time_since_access > self.auto_unload_timeout:
                    candidates.append(model_id)
            
            # 자동 언로드 실행
            for model_id in candidates:
                self._unload_model(model_id)
                self.logger.debug(f"⏰ 자동 언로드: {model_id} (비활성 시간: {time_since_access/60:.1f}분)")
    
    def get_memory_profile(self) -> MemoryProfile:
        """현재 메모리 프로파일 반환"""
        try:
            # 시스템 메모리
            memory = psutil.virtual_memory()
            total_mb = memory.total / (1024**2)
            available_mb = memory.available / (1024**2)
            used_mb = memory.used / (1024**2)
            
            # AI 모델 메모리 계산
            ai_memory_mb = sum(
                model_info.memory_usage_mb 
                for model_info in self.models.values() 
                if model_info.is_loaded
            )
            
            # GPU 메모리
            gpu_memory_mb = 0.0
            gpu_available_mb = 0.0
            
            if self.gpu_available:
                try:
                    gpu_memory_mb = torch.cuda.memory_allocated() / (1024**2)
                    gpu_available_mb = torch.cuda.memory_reserved() / (1024**2) - gpu_memory_mb
                except:
                    pass
            
            # 메모리 압박 수준 계산
            usage_ratio = used_mb / total_mb
            if usage_ratio > self.memory_critical_threshold:
                pressure_level = "critical"
            elif usage_ratio > self.memory_warning_threshold:
                pressure_level = "high"
            elif usage_ratio > 0.6:
                pressure_level = "medium"
            else:
                pressure_level = "low"
            
            return MemoryProfile(
                total_memory_mb=total_mb,
                available_memory_mb=available_mb,
                ai_models_memory_mb=ai_memory_mb,
                system_memory_mb=used_mb - ai_memory_mb,
                gpu_memory_mb=gpu_memory_mb,
                gpu_available_mb=gpu_available_mb,
                memory_pressure_level=pressure_level
            )
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 프로파일 생성 실패: {e}")
            return MemoryProfile(0, 0, 0, 0)
    
    def get_model_stats(self) -> Dict[str, Any]:
        """모델 통계 반환"""
        with self.lock:
            loaded_models = sum(1 for m in self.models.values() if m.is_loaded)
            total_memory = sum(m.memory_usage_mb for m in self.models.values() if m.is_loaded)
            
            model_types = {}
            for model_info in self.models.values():
                model_type = model_info.model_type.value
                if model_type not in model_types:
                    model_types[model_type] = 0
                model_types[model_type] += 1
            
            return {
                'total_models': len(self.models),
                'loaded_models': loaded_models,
                'total_memory_mb': total_memory,
                'model_types': model_types,
                'stats': self.stats.copy(),
                'gpu_available': self.gpu_available
            }
    
    def get_model_details(self) -> List[Dict[str, Any]]:
        """모델 상세 정보 반환"""
        with self.lock:
            details = []
            
            for model_id, model_info in self.models.items():
                detail = {
                    'model_id': model_id,
                    'model_type': model_info.model_type.value,
                    'priority': model_info.priority.name,
                    'memory_usage_mb': model_info.memory_usage_mb,
                    'model_size_mb': model_info.model_size_mb,
                    'is_loaded': model_info.is_loaded,
                    'device': model_info.device,
                    'load_count': model_info.load_count,
                    'last_accessed': model_info.last_accessed.isoformat(),
                    'metadata': model_info.metadata
                }
                details.append(detail)
            
            # 메모리 사용량 순으로 정렬
            details.sort(key=lambda x: x['memory_usage_mb'], reverse=True)
            
            return details
    
    def set_auto_unload_timeout(self, timeout_seconds: float) -> None:
        """자동 언로드 타임아웃 설정"""
        self.auto_unload_timeout = timeout_seconds
        self.logger.info(f"⏰ 자동 언로드 타임아웃: {timeout_seconds/60:.1f}분")
    
    def set_memory_thresholds(self, warning: float, critical: float) -> None:
        """메모리 임계값 설정"""
        self.memory_warning_threshold = warning
        self.memory_critical_threshold = critical
        self.logger.info(f"🎯 메모리 임계값: 경고 {warning*100:.0f}%, 위험 {critical*100:.0f}%")
    
    def cleanup(self) -> None:
        """리소스 정리"""
        self.monitoring_enabled = False
        
        if self.memory_monitor_thread and self.memory_monitor_thread.is_alive():
            self.memory_monitor_thread.join(timeout=2.0)
        
        with self.lock:
            # 모든 모델 언로드
            model_ids = list(self.models.keys())
            for model_id in model_ids:
                self.unregister_model(model_id)
        
        # GPU 메모리 정리
        if self.gpu_available:
            self._cleanup_gpu_memory()
        
        # 가비지 컬렉션
        gc.collect()
        
        self.logger.info("🧹 AI 모델 메모리 관리 시스템 정리 완료")

# 전역 AI 모델 메모리 관리자
_global_ai_memory_manager = None
_global_manager_lock = threading.Lock()

def get_global_ai_memory_manager(max_memory_mb: float = 2048.0, enable_gpu: bool = True) -> AIModelMemoryManager:
    """전역 AI 모델 메모리 관리자 가져오기"""
    global _global_ai_memory_manager
    
    with _global_manager_lock:
        if _global_ai_memory_manager is None:
            _global_ai_memory_manager = AIModelMemoryManager(max_memory_mb, enable_gpu)
        return _global_ai_memory_manager

# 편의 함수들
def register_ai_model(model_id: str, model_type: str, model_obj: Any, 
                     priority: str = "medium", device: str = "cpu") -> bool:
    """AI 모델 등록 (편의 함수)"""
    manager = get_global_ai_memory_manager()
    
    model_type_enum = ModelType(model_type.lower())
    priority_enum = ModelPriority[priority.upper()]
    
    return manager.register_model(model_id, model_type_enum, model_obj, priority_enum, device)

def access_ai_model(model_id: str) -> Optional[Any]:
    """AI 모델 접근 (편의 함수)"""
    manager = get_global_ai_memory_manager()
    return manager.access_model(model_id)

def optimize_ai_memory(target_mb: Optional[float] = None) -> Dict[str, Any]:
    """AI 모델 메모리 최적화 (편의 함수)"""
    manager = get_global_ai_memory_manager()
    return manager.optimize_memory(target_mb)

# 사용 예시
if __name__ == "__main__":
    # AI 모델 메모리 관리자 테스트
    manager = AIModelMemoryManager(max_memory_mb=1024.0)
    
    # 가상의 모델 등록
    class DummyModel:
        def __init__(self, size_mb: float):
            self.data = bytearray(int(size_mb * 1024 * 1024))
    
    # 모델들 등록
    models = [
        ("whisper_large", ModelType.WHISPER, DummyModel(500), ModelPriority.HIGH),
        ("easyocr_en", ModelType.EASYOCR, DummyModel(100), ModelPriority.MEDIUM),
        ("bert_base", ModelType.TRANSFORMERS, DummyModel(300), ModelPriority.LOW),
    ]
    
    for model_id, model_type, model_obj, priority in models:
        success = manager.register_model(model_id, model_type, model_obj, priority)
        print(f"모델 등록 {model_id}: {'성공' if success else '실패'}")
    
    # 메모리 프로파일 확인
    profile = manager.get_memory_profile()
    print(f"\n메모리 프로파일:")
    print(f"  AI 모델 메모리: {profile.ai_models_memory_mb:.1f}MB")
    print(f"  메모리 압박 수준: {profile.memory_pressure_level}")
    
    # 모델 접근 테스트
    model = manager.access_model("whisper_large")
    if model:
        print(f"모델 접근 성공: whisper_large")
    
    # 메모리 최적화 테스트
    result = manager.optimize_memory(600.0)
    print(f"\n메모리 최적화 결과:")
    print(f"  해제된 메모리: {result.get('freed_memory_mb', 0):.1f}MB")
    print(f"  언로드된 모델: {result.get('unloaded_models', [])}")
    
    # 통계 확인
    stats = manager.get_model_stats()
    print(f"\n모델 통계:")
    print(f"  총 모델: {stats['total_models']}개")
    print(f"  로드된 모델: {stats['loaded_models']}개")
    print(f"  총 메모리: {stats['total_memory_mb']:.1f}MB")
    
    # 정리
    manager.cleanup()
    print("\n✅ AI 모델 메모리 관리 테스트 완료!")