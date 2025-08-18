#!/usr/bin/env python3
"""
스마트 AI 모델 로더 v2.6
메모리 관리 기능이 통합된 AI 모델 로더
"""

import os
import time
import threading
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from contextlib import contextmanager

from .ai_model_memory_manager import (
    get_global_ai_memory_manager, 
    ModelType, 
    ModelPriority,
    AIModelMemoryManager
)

@dataclass
class ModelLoadResult:
    """모델 로드 결과"""
    success: bool
    model: Optional[Any] = None
    model_id: str = ""
    load_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    error_message: Optional[str] = None

class SmartAIModelLoader:
    """스마트 AI 모델 로더"""
    
    def __init__(self, memory_manager: Optional[AIModelMemoryManager] = None):
        self.memory_manager = memory_manager or get_global_ai_memory_manager()
        self.logger = self._setup_logging()
        
        # 로드된 모델 캐시
        self.model_cache: Dict[str, Any] = {}
        self.load_lock = threading.RLock()
        
        # 모델별 기본 설정
        self.model_configs = {
            ModelType.WHISPER: {
                'default_model': 'base',
                'device_preference': 'cpu',  # GPU 메모리 절약
                'priority': ModelPriority.HIGH
            },
            ModelType.EASYOCR: {
                'default_languages': ['ko', 'en'],
                'device_preference': 'cpu',
                'priority': ModelPriority.MEDIUM
            },
            ModelType.TRANSFORMERS: {
                'default_model': 'bert-base-multilingual-cased',
                'device_preference': 'cpu',
                'priority': ModelPriority.MEDIUM
            }
        }
        
        self.logger.info("🤖 스마트 AI 모델 로더 초기화 완료")
    
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
    
    def load_whisper_model(self, model_size: str = "base", device: str = "auto") -> ModelLoadResult:
        """Whisper 모델 로드"""
        model_id = f"whisper_{model_size}"
        
        with self.load_lock:
            # 이미 로드된 모델 확인
            cached_model = self.memory_manager.access_model(model_id)
            if cached_model is not None:
                return ModelLoadResult(
                    success=True,
                    model=cached_model,
                    model_id=model_id,
                    load_time_seconds=0.0
                )
            
            # 메모리 압박 상황 체크
            profile = self.memory_manager.get_memory_profile()
            if profile.memory_pressure_level in ['high', 'critical']:
                self.logger.warning("⚠️ 메모리 압박으로 인한 사전 최적화 실행")
                self.memory_manager.optimize_memory()
            
            start_time = time.time()
            
            try:
                # 장치 결정
                if device == "auto":
                    device = "cpu"  # GPU 메모리 절약 우선
                
                # Whisper 모델 로드
                import whisper
                
                self.logger.info(f"🎤 Whisper 모델 로드 중: {model_size} (장치: {device})")
                
                # CPU 모드 강제 설정
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                
                model = whisper.load_model(model_size, device=device)
                
                load_time = time.time() - start_time
                
                # 메모리 관리자에 등록
                success = self.memory_manager.register_model(
                    model_id=model_id,
                    model_type=ModelType.WHISPER,
                    model_obj=model,
                    priority=ModelPriority.HIGH,
                    device=device,
                    metadata={
                        'model_size': model_size,
                        'load_time': load_time,
                        'framework': 'whisper'
                    }
                )
                
                if success:
                    self.logger.info(f"✅ Whisper 모델 로드 완료: {model_size} ({load_time:.2f}초)")
                    
                    return ModelLoadResult(
                        success=True,
                        model=model,
                        model_id=model_id,
                        load_time_seconds=load_time,
                        memory_usage_mb=self._get_model_memory(model_id)
                    )
                else:
                    return ModelLoadResult(
                        success=False,
                        error_message="메모리 관리자 등록 실패"
                    )
                
            except Exception as e:
                error_msg = f"Whisper 모델 로드 실패: {e}"
                self.logger.error(f"❌ {error_msg}")
                
                return ModelLoadResult(
                    success=False,
                    error_message=error_msg
                )
    
    def load_easyocr_model(self, languages: List[str] = None, device: str = "auto") -> ModelLoadResult:
        """EasyOCR 모델 로드"""
        if languages is None:
            languages = ['ko', 'en']
        
        model_id = f"easyocr_{'_'.join(languages)}"
        
        with self.load_lock:
            # 이미 로드된 모델 확인
            cached_model = self.memory_manager.access_model(model_id)
            if cached_model is not None:
                return ModelLoadResult(
                    success=True,
                    model=cached_model,
                    model_id=model_id,
                    load_time_seconds=0.0
                )
            
            start_time = time.time()
            
            try:
                # 장치 결정
                if device == "auto":
                    device = "cpu"  # GPU 메모리 절약
                
                # EasyOCR 모델 로드
                import easyocr
                
                self.logger.info(f"🖼️ EasyOCR 모델 로드 중: {languages} (장치: {device})")
                
                # GPU 사용 여부 결정
                gpu = False if device == "cpu" else True
                
                reader = easyocr.Reader(languages, gpu=gpu)
                
                load_time = time.time() - start_time
                
                # 메모리 관리자에 등록
                success = self.memory_manager.register_model(
                    model_id=model_id,
                    model_type=ModelType.EASYOCR,
                    model_obj=reader,
                    priority=ModelPriority.MEDIUM,
                    device=device,
                    metadata={
                        'languages': languages,
                        'load_time': load_time,
                        'framework': 'easyocr'
                    }
                )
                
                if success:
                    self.logger.info(f"✅ EasyOCR 모델 로드 완료: {languages} ({load_time:.2f}초)")
                    
                    return ModelLoadResult(
                        success=True,
                        model=reader,
                        model_id=model_id,
                        load_time_seconds=load_time,
                        memory_usage_mb=self._get_model_memory(model_id)
                    )
                else:
                    return ModelLoadResult(
                        success=False,
                        error_message="메모리 관리자 등록 실패"
                    )
                
            except Exception as e:
                error_msg = f"EasyOCR 모델 로드 실패: {e}"
                self.logger.error(f"❌ {error_msg}")
                
                return ModelLoadResult(
                    success=False,
                    error_message=error_msg
                )
    
    def load_transformers_model(self, model_name: str, task: str = "summarization", device: str = "auto") -> ModelLoadResult:
        """Transformers 모델 로드"""
        model_id = f"transformers_{model_name.replace('/', '_')}_{task}"
        
        with self.load_lock:
            # 이미 로드된 모델 확인
            cached_model = self.memory_manager.access_model(model_id)
            if cached_model is not None:
                return ModelLoadResult(
                    success=True,
                    model=cached_model,
                    model_id=model_id,
                    load_time_seconds=0.0
                )
            
            start_time = time.time()
            
            try:
                # 장치 결정
                if device == "auto":
                    device = "cpu"  # GPU 메모리 절약
                
                # Transformers 모델 로드
                from transformers import pipeline
                
                self.logger.info(f"🤖 Transformers 모델 로드 중: {model_name} ({task}, 장치: {device})")
                
                # CPU 모드 강제 설정
                import torch
                device_id = -1 if device == "cpu" else 0
                
                model = pipeline(
                    task=task,
                    model=model_name,
                    device=device_id,
                    torch_dtype=torch.float32  # 메모리 절약을 위해 float32 사용
                )
                
                load_time = time.time() - start_time
                
                # 메모리 관리자에 등록
                success = self.memory_manager.register_model(
                    model_id=model_id,
                    model_type=ModelType.TRANSFORMERS,
                    model_obj=model,
                    priority=ModelPriority.MEDIUM,
                    device=device,
                    metadata={
                        'model_name': model_name,
                        'task': task,
                        'load_time': load_time,
                        'framework': 'transformers'
                    }
                )
                
                if success:
                    self.logger.info(f"✅ Transformers 모델 로드 완료: {model_name} ({load_time:.2f}초)")
                    
                    return ModelLoadResult(
                        success=True,
                        model=model,
                        model_id=model_id,
                        load_time_seconds=load_time,
                        memory_usage_mb=self._get_model_memory(model_id)
                    )
                else:
                    return ModelLoadResult(
                        success=False,
                        error_message="메모리 관리자 등록 실패"
                    )
                
            except Exception as e:
                error_msg = f"Transformers 모델 로드 실패: {e}"
                self.logger.error(f"❌ {error_msg}")
                
                return ModelLoadResult(
                    success=False,
                    error_message=error_msg
                )
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """모델 가져오기"""
        return self.memory_manager.access_model(model_id)
    
    def unload_model(self, model_id: str) -> bool:
        """모델 언로드"""
        success = self.memory_manager.unregister_model(model_id)
        if success:
            self.logger.info(f"📤 모델 언로드: {model_id}")
        return success
    
    def optimize_memory(self, target_mb: Optional[float] = None) -> Dict[str, Any]:
        """메모리 최적화"""
        return self.memory_manager.optimize_memory(target_mb)
    
    def get_memory_status(self) -> Dict[str, Any]:
        """메모리 상태 반환"""
        profile = self.memory_manager.get_memory_profile()
        stats = self.memory_manager.get_model_stats()
        
        return {
            'memory_profile': {
                'total_memory_mb': profile.total_memory_mb,
                'available_memory_mb': profile.available_memory_mb,
                'ai_models_memory_mb': profile.ai_models_memory_mb,
                'memory_pressure_level': profile.memory_pressure_level
            },
            'model_stats': stats,
            'loaded_models': self.memory_manager.get_model_details()
        }
    
    def _get_model_memory(self, model_id: str) -> float:
        """모델 메모리 사용량 가져오기"""
        model_details = self.memory_manager.get_model_details()
        for detail in model_details:
            if detail['model_id'] == model_id:
                return detail['memory_usage_mb']
        return 0.0
    
    @contextmanager
    def smart_model_context(self, model_type: str, **kwargs):
        """스마트 모델 컨텍스트 매니저"""
        model_result = None
        
        try:
            # 모델 타입별 로드
            if model_type.lower() == "whisper":
                model_result = self.load_whisper_model(**kwargs)
            elif model_type.lower() == "easyocr":
                model_result = self.load_easyocr_model(**kwargs)
            elif model_type.lower() == "transformers":
                model_result = self.load_transformers_model(**kwargs)
            else:
                raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
            
            if not model_result.success:
                raise RuntimeError(f"모델 로드 실패: {model_result.error_message}")
            
            yield model_result.model
            
        finally:
            # 컨텍스트 종료 시 메모리 최적화 (선택적)
            if model_result and model_result.success:
                # 낮은 우선순위 모델들은 자동 언로드될 수 있음
                pass
    
    def preload_models(self, model_configs: List[Dict[str, Any]]) -> Dict[str, ModelLoadResult]:
        """모델들 사전 로드"""
        results = {}
        
        self.logger.info(f"🚀 모델 사전 로드 시작: {len(model_configs)}개 모델")
        
        for config in model_configs:
            model_type = config.get('type')
            model_id = config.get('id', f"{model_type}_{len(results)}")
            
            try:
                if model_type == "whisper":
                    result = self.load_whisper_model(
                        model_size=config.get('model_size', 'base'),
                        device=config.get('device', 'auto')
                    )
                elif model_type == "easyocr":
                    result = self.load_easyocr_model(
                        languages=config.get('languages', ['ko', 'en']),
                        device=config.get('device', 'auto')
                    )
                elif model_type == "transformers":
                    result = self.load_transformers_model(
                        model_name=config.get('model_name', 'bert-base-multilingual-cased'),
                        task=config.get('task', 'summarization'),
                        device=config.get('device', 'auto')
                    )
                else:
                    result = ModelLoadResult(
                        success=False,
                        error_message=f"지원하지 않는 모델 타입: {model_type}"
                    )
                
                results[model_id] = result
                
                if result.success:
                    self.logger.info(f"✅ 사전 로드 성공: {model_id}")
                else:
                    self.logger.error(f"❌ 사전 로드 실패: {model_id} - {result.error_message}")
                
            except Exception as e:
                results[model_id] = ModelLoadResult(
                    success=False,
                    error_message=str(e)
                )
                self.logger.error(f"❌ 사전 로드 오류: {model_id} - {e}")
        
        success_count = sum(1 for r in results.values() if r.success)
        self.logger.info(f"🎉 모델 사전 로드 완료: {success_count}/{len(model_configs)} 성공")
        
        return results
    
    def get_model_recommendations(self) -> Dict[str, Any]:
        """모델 사용 권장사항"""
        profile = self.memory_manager.get_memory_profile()
        
        recommendations = {
            'memory_status': profile.memory_pressure_level,
            'recommendations': []
        }
        
        if profile.memory_pressure_level == 'critical':
            recommendations['recommendations'].extend([
                "메모리 사용량이 심각합니다. 즉시 일부 모델을 언로드하세요.",
                "Whisper 모델을 'tiny' 또는 'base'로 변경하세요.",
                "불필요한 Transformers 모델을 제거하세요."
            ])
        elif profile.memory_pressure_level == 'high':
            recommendations['recommendations'].extend([
                "메모리 사용량이 높습니다. 모델 최적화를 권장합니다.",
                "사용하지 않는 모델들을 확인하여 언로드하세요.",
                "더 작은 모델 사용을 고려하세요."
            ])
        elif profile.memory_pressure_level == 'medium':
            recommendations['recommendations'].extend([
                "메모리 사용량이 보통 수준입니다.",
                "필요에 따라 더 큰 모델 사용이 가능합니다.",
                "주기적인 메모리 최적화를 권장합니다."
            ])
        else:
            recommendations['recommendations'].extend([
                "메모리 상태가 양호합니다.",
                "고성능 모델 사용이 가능합니다.",
                "추가 모델 로드가 안전합니다."
            ])
        
        return recommendations
    
    def cleanup(self) -> None:
        """리소스 정리"""
        self.logger.info("🧹 스마트 AI 모델 로더 정리")
        # 메모리 관리자는 전역이므로 여기서 정리하지 않음

# 전역 스마트 모델 로더
_global_smart_loader = None
_global_loader_lock = threading.Lock()

def get_global_smart_loader() -> SmartAIModelLoader:
    """전역 스마트 모델 로더 가져오기"""
    global _global_smart_loader
    
    with _global_loader_lock:
        if _global_smart_loader is None:
            _global_smart_loader = SmartAIModelLoader()
        return _global_smart_loader

# 편의 함수들
def smart_load_whisper(model_size: str = "base", device: str = "auto") -> ModelLoadResult:
    """Whisper 모델 스마트 로드"""
    loader = get_global_smart_loader()
    return loader.load_whisper_model(model_size, device)

def smart_load_easyocr(languages: List[str] = None, device: str = "auto") -> ModelLoadResult:
    """EasyOCR 모델 스마트 로드"""
    loader = get_global_smart_loader()
    return loader.load_easyocr_model(languages, device)

def smart_load_transformers(model_name: str, task: str = "summarization", device: str = "auto") -> ModelLoadResult:
    """Transformers 모델 스마트 로드"""
    loader = get_global_smart_loader()
    return loader.load_transformers_model(model_name, task, device)

def get_ai_memory_status() -> Dict[str, Any]:
    """AI 모델 메모리 상태 조회"""
    loader = get_global_smart_loader()
    return loader.get_memory_status()

def optimize_ai_models_memory(target_mb: Optional[float] = None) -> Dict[str, Any]:
    """AI 모델 메모리 최적화"""
    loader = get_global_smart_loader()
    return loader.optimize_memory(target_mb)

# 사용 예시
if __name__ == "__main__":
    # 스마트 AI 모델 로더 테스트
    loader = SmartAIModelLoader()
    
    # Whisper 모델 로드 테스트
    print("🎤 Whisper 모델 로드 테스트")
    whisper_result = loader.load_whisper_model("base")
    
    if whisper_result.success:
        print(f"✅ Whisper 로드 성공: {whisper_result.load_time_seconds:.2f}초, {whisper_result.memory_usage_mb:.1f}MB")
    else:
        print(f"❌ Whisper 로드 실패: {whisper_result.error_message}")
    
    # EasyOCR 모델 로드 테스트
    print("\n🖼️ EasyOCR 모델 로드 테스트")
    easyocr_result = loader.load_easyocr_model(['ko', 'en'])
    
    if easyocr_result.success:
        print(f"✅ EasyOCR 로드 성공: {easyocr_result.load_time_seconds:.2f}초, {easyocr_result.memory_usage_mb:.1f}MB")
    else:
        print(f"❌ EasyOCR 로드 실패: {easyocr_result.error_message}")
    
    # 메모리 상태 확인
    print("\n📊 메모리 상태:")
    status = loader.get_memory_status()
    memory_profile = status['memory_profile']
    print(f"  AI 모델 메모리: {memory_profile['ai_models_memory_mb']:.1f}MB")
    print(f"  메모리 압박 수준: {memory_profile['memory_pressure_level']}")
    
    # 모델 권장사항
    print("\n💡 권장사항:")
    recommendations = loader.get_model_recommendations()
    for rec in recommendations['recommendations']:
        print(f"  - {rec}")
    
    # 정리
    loader.cleanup()
    print("\n✅ 스마트 AI 모델 로더 테스트 완료!")