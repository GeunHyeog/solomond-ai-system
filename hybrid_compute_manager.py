#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 SOLOMOND AI 하이브리드 컴퓨팅 매니저
Hybrid Compute Manager - CPU/GPU 동적 전환 시스템

핵심 기능:
1. 실시간 GPU/CPU 상태 모니터링
2. 작업 유형별 최적 리소스 자동 선택
3. 메모리 부족 시 자동 폴백
4. 성능 기반 동적 스케줄링
"""

import os
import gc
import time
import psutil
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
from pathlib import Path

# GPU 관련 임포트 (선택적)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
    NVIDIA_ML_AVAILABLE = True
except (ImportError, Exception):
    NVIDIA_ML_AVAILABLE = False

class ComputeMode(Enum):
    """컴퓨팅 모드"""
    AUTO = "auto"           # 자동 선택 (권장)
    GPU_PREFERRED = "gpu_preferred"  # GPU 우선, CPU 폴백
    CPU_PREFERRED = "cpu_preferred"  # CPU 우선, GPU 보조
    GPU_ONLY = "gpu_only"   # GPU 전용
    CPU_ONLY = "cpu_only"   # CPU 전용

class TaskType(Enum):
    """작업 유형"""
    STT_SMALL = "stt_small"         # 작은 음성 파일 (<5분)
    STT_LARGE = "stt_large"         # 큰 음성 파일 (>5분)
    OCR_BATCH = "ocr_batch"         # 배치 이미지 OCR
    OCR_REALTIME = "ocr_realtime"   # 실시간 OCR
    LLM_INFERENCE = "llm_inference" # LLM 추론
    VIDEO_PROCESSING = "video_processing" # 비디오 처리

@dataclass
class ResourceStatus:
    """리소스 상태"""
    device_type: str  # "cpu" or "gpu"
    total_memory_gb: float
    available_memory_gb: float
    usage_percent: float
    temperature: Optional[float] = None
    power_usage: Optional[float] = None

@dataclass
class TaskSchedule:
    """작업 스케줄"""
    task_type: TaskType
    estimated_time: float
    memory_requirement: float
    preferred_device: str
    fallback_device: str

class HybridComputeManager:
    """하이브리드 컴퓨팅 관리자"""
    
    def __init__(self, default_mode: ComputeMode = ComputeMode.AUTO):
        """초기화"""
        self.mode = default_mode
        self.gpu_available = self._check_gpu_availability()
        self.cpu_cores = psutil.cpu_count()
        self.total_memory = psutil.virtual_memory().total / (1024**3)
        
        # 성능 히스토리 저장
        self.performance_history = {}
        self.current_tasks = {}
        
        # 동적 임계값
        self.gpu_memory_threshold = 0.8  # 80% 이상 사용시 CPU 폴백
        self.cpu_usage_threshold = 80    # 80% 이상 사용시 GPU 활용
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"HybridComputeManager 초기화 완료")
        self.logger.info(f"GPU 사용 가능: {self.gpu_available}")
        self.logger.info(f"CPU 코어: {self.cpu_cores}개")
        self.logger.info(f"총 메모리: {self.total_memory:.1f}GB")
    
    def _check_gpu_availability(self) -> bool:
        """GPU 사용 가능 여부 확인"""
        if not TORCH_AVAILABLE:
            return False
            
        try:
            # CUDA 사용 가능성 확인
            if torch.cuda.is_available():
                # GPU 메모리 테스트
                device = torch.device('cuda')
                test_tensor = torch.ones(100, device=device)
                del test_tensor
                torch.cuda.empty_cache()
                return True
        except Exception as e:
            self.logger.warning(f"GPU 테스트 실패: {e}")
        
        return False
    
    def get_resource_status(self) -> Dict[str, ResourceStatus]:
        """현재 리소스 상태 반환"""
        status = {}
        
        # CPU 상태
        cpu_memory = psutil.virtual_memory()
        status['cpu'] = ResourceStatus(
            device_type="cpu",
            total_memory_gb=cpu_memory.total / (1024**3),
            available_memory_gb=cpu_memory.available / (1024**3),
            usage_percent=cpu_memory.percent
        )
        
        # GPU 상태 (사용 가능한 경우)
        if self.gpu_available and TORCH_AVAILABLE:
            try:
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_memory_available = gpu_memory_total - gpu_memory_allocated
                usage_percent = (gpu_memory_allocated / gpu_memory_total) * 100
                
                # NVIDIA ML을 통한 온도 정보 (선택적)
                temperature = None
                if NVIDIA_ML_AVAILABLE:
                    try:
                        handle = nvml.nvmlDeviceGetHandleByIndex(0)
                        temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    except:
                        pass
                
                status['gpu'] = ResourceStatus(
                    device_type="gpu",
                    total_memory_gb=gpu_memory_total,
                    available_memory_gb=gpu_memory_available,
                    usage_percent=usage_percent,
                    temperature=temperature
                )
            except Exception as e:
                self.logger.warning(f"GPU 상태 확인 실패: {e}")
        
        return status
    
    def select_optimal_device(self, task_type: TaskType, memory_requirement: float = 1.0) -> str:
        """작업에 최적인 디바이스 선택"""
        
        if self.mode == ComputeMode.CPU_ONLY:
            return "cpu"
        elif self.mode == ComputeMode.GPU_ONLY:
            if self.gpu_available:
                return "gpu"
            else:
                self.logger.warning("GPU 전용 모드이지만 GPU 사용 불가, CPU로 폴백")
                return "cpu"
        
        # 현재 리소스 상태 확인
        resource_status = self.get_resource_status()
        
        # 작업 유형별 최적화 전략
        task_preferences = {
            TaskType.STT_SMALL: ("cpu", "gpu"),      # 작은 음성은 CPU가 효율적
            TaskType.STT_LARGE: ("gpu", "cpu"),      # 큰 음성은 GPU가 효율적
            TaskType.OCR_BATCH: ("gpu", "cpu"),      # 배치 OCR은 GPU 최적
            TaskType.OCR_REALTIME: ("cpu", "gpu"),   # 실시간 OCR은 CPU가 빠름
            TaskType.LLM_INFERENCE: ("gpu", "cpu"),  # LLM은 GPU 우선
            TaskType.VIDEO_PROCESSING: ("gpu", "cpu") # 비디오는 GPU 필수
        }
        
        preferred, fallback = task_preferences.get(task_type, ("cpu", "gpu"))
        
        # AUTO 모드 로직
        if self.mode == ComputeMode.AUTO:
            # GPU 사용 가능하고 메모리 충분한 경우
            if (self.gpu_available and 
                preferred == "gpu" and 
                "gpu" in resource_status and
                resource_status["gpu"].available_memory_gb >= memory_requirement):
                
                # GPU 메모리 사용률 체크
                if resource_status["gpu"].usage_percent < self.gpu_memory_threshold * 100:
                    return "gpu"
            
            # CPU 사용률 체크
            if resource_status["cpu"].usage_percent < self.cpu_usage_threshold:
                return "cpu"
            
            # 둘 다 바쁜 경우, 메모리 여유가 많은 쪽 선택
            if "gpu" in resource_status:
                gpu_memory_ratio = resource_status["gpu"].available_memory_gb / resource_status["gpu"].total_memory_gb
                cpu_memory_ratio = resource_status["cpu"].available_memory_gb / resource_status["cpu"].total_memory_gb
                
                if gpu_memory_ratio > cpu_memory_ratio and self.gpu_available:
                    return "gpu"
            
            return "cpu"
        
        # GPU_PREFERRED 모드
        elif self.mode == ComputeMode.GPU_PREFERRED:
            if (self.gpu_available and 
                "gpu" in resource_status and
                resource_status["gpu"].available_memory_gb >= memory_requirement and
                resource_status["gpu"].usage_percent < 90):
                return "gpu"
            return "cpu"
        
        # CPU_PREFERRED 모드
        elif self.mode == ComputeMode.CPU_PREFERRED:
            if (resource_status["cpu"].usage_percent < 70 and
                resource_status["cpu"].available_memory_gb >= memory_requirement):
                return "cpu"
            elif self.gpu_available and "gpu" in resource_status:
                return "gpu"
            return "cpu"
        
        return "cpu"  # 기본값
    
    def optimize_for_task(self, task_type: TaskType, **kwargs) -> Dict[str, Any]:
        """작업별 최적화 설정 반환"""
        
        device = self.select_optimal_device(task_type, kwargs.get('memory_requirement', 1.0))
        
        optimization_configs = {
            TaskType.STT_SMALL: {
                "device": device,
                "batch_size": 1,
                "fp16": device == "gpu",
                "num_workers": 2 if device == "cpu" else 1
            },
            TaskType.STT_LARGE: {
                "device": device,
                "batch_size": 4 if device == "gpu" else 1,
                "fp16": device == "gpu",
                "chunk_size": 30 if device == "gpu" else 15
            },
            TaskType.OCR_BATCH: {
                "device": device,
                "gpu": device == "gpu",
                "batch_size": 8 if device == "gpu" else 4,
                "parallel": device == "cpu"
            },
            TaskType.OCR_REALTIME: {
                "device": device,
                "gpu": False,  # 실시간은 CPU가 더 안정적
                "parallel": True
            },
            TaskType.LLM_INFERENCE: {
                "device": device,
                "fp16": device == "gpu",
                "max_length": 2048 if device == "gpu" else 1024
            },
            TaskType.VIDEO_PROCESSING: {
                "device": device,
                "gpu": device == "gpu",
                "batch_size": 2 if device == "gpu" else 1
            }
        }
        
        config = optimization_configs.get(task_type, {"device": device})
        
        # 환경변수 설정
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] == "":
                del os.environ["CUDA_VISIBLE_DEVICES"]
        
        self.logger.info(f"작업 '{task_type.value}'에 대해 '{device}' 디바이스 선택됨")
        
        return config
    
    def cleanup_memory(self, device: str = "both"):
        """메모리 정리"""
        if device in ["both", "cpu"]:
            gc.collect()
        
        if device in ["both", "gpu"] and self.gpu_available and TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as e:
                self.logger.warning(f"GPU 메모리 정리 실패: {e}")
    
    def get_performance_recommendation(self) -> Dict[str, str]:
        """성능 최적화 권장사항"""
        status = self.get_resource_status()
        recommendations = []
        
        # CPU 기반 권장사항
        if status["cpu"].usage_percent > 80:
            recommendations.append("CPU 사용률이 높습니다. GPU 활용을 고려하세요.")
        
        if status["cpu"].available_memory_gb < 2:
            recommendations.append("시스템 메모리가 부족합니다. 배치 크기를 줄이세요.")
        
        # GPU 기반 권장사항 (사용 가능한 경우)
        if "gpu" in status:
            if status["gpu"].usage_percent > 85:
                recommendations.append("GPU 메모리가 부족합니다. CPU 폴백을 권장합니다.")
            elif status["gpu"].usage_percent < 30 and self.gpu_available:
                recommendations.append("GPU 리소스가 충분합니다. GPU 활용을 늘려보세요.")
        
        # 모드 권장사항
        mode_recommendations = {
            "light_tasks": "실시간 처리에는 CPU_PREFERRED 모드를 권장합니다.",
            "heavy_tasks": "대용량 처리에는 GPU_PREFERRED 모드를 권장합니다.",
            "mixed_tasks": "다양한 작업에는 AUTO 모드를 권장합니다."
        }
        
        return {
            "general": recommendations,
            "mode": mode_recommendations
        }

# 전역 인스턴스
_global_manager = None

def get_compute_manager() -> HybridComputeManager:
    """전역 하이브리드 컴퓨팅 매니저 반환"""
    global _global_manager
    if _global_manager is None:
        _global_manager = HybridComputeManager()
    return _global_manager

def auto_optimize_for_whisper(audio_duration: float = 60) -> Dict[str, Any]:
    """Whisper STT 자동 최적화"""
    manager = get_compute_manager()
    task_type = TaskType.STT_LARGE if audio_duration > 300 else TaskType.STT_SMALL
    return manager.optimize_for_task(task_type, memory_requirement=audio_duration/60)

def auto_optimize_for_ocr(image_count: int = 1, realtime: bool = False) -> Dict[str, Any]:
    """OCR 자동 최적화"""
    manager = get_compute_manager()
    task_type = TaskType.OCR_REALTIME if realtime else TaskType.OCR_BATCH
    return manager.optimize_for_task(task_type, memory_requirement=image_count * 0.1)

def auto_optimize_for_llm(context_length: int = 1024) -> Dict[str, Any]:
    """LLM 추론 자동 최적화"""
    manager = get_compute_manager()
    return manager.optimize_for_task(TaskType.LLM_INFERENCE, memory_requirement=context_length/1024)

if __name__ == "__main__":
    # 테스트 코드
    manager = HybridComputeManager()
    
    print("=== 하이브리드 컴퓨팅 매니저 테스트 ===")
    
    # 리소스 상태 출력
    status = manager.get_resource_status()
    for device, info in status.items():
        print(f"{device.upper()}: {info.available_memory_gb:.1f}GB 사용 가능 ({info.usage_percent:.1f}% 사용 중)")
    
    # 작업별 최적화 테스트
    test_tasks = [
        (TaskType.STT_SMALL, "짧은 음성 처리"),
        (TaskType.OCR_BATCH, "배치 이미지 OCR"),
        (TaskType.LLM_INFERENCE, "LLM 추론")
    ]
    
    print("\n=== 작업별 최적화 결과 ===")
    for task_type, description in test_tasks:
        config = manager.optimize_for_task(task_type)
        print(f"{description}: {config['device']} 디바이스 선택됨")
    
    # 성능 권장사항
    recommendations = manager.get_performance_recommendation()
    print("\n=== 성능 권장사항 ===")
    for rec in recommendations["general"]:
        print(f"- {rec}")