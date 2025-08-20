#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ Dynamic Resource Manager
GPU/CPU 동적 전환 및 최적화 시스템

기능:
1. GPU 가용성 자동 감지
2. 작업 부하에 따른 동적 전환  
3. 메모리 사용량 모니터링
4. 성능 기반 자동 최적화
"""

import os
import sys
import psutil
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import threading

logger = logging.getLogger(__name__)

@dataclass
class ResourceStatus:
    """리소스 상태 정보"""
    gpu_available: bool
    gpu_memory_free: float  # GB
    gpu_memory_total: float  # GB
    cpu_percent: float
    ram_available: float  # GB
    recommendation: str  # 'gpu', 'cpu', 'hybrid'

class DynamicResourceManager:
    """동적 리소스 관리자"""
    
    def __init__(self):
        self.gpu_available = self._check_gpu_availability()
        self.performance_history = []
        self.current_mode = 'auto'  # 'gpu', 'cpu', 'auto'
        
        logger.info(f"리소스 관리자 초기화 - GPU 사용 가능: {self.gpu_available}")
    
    def _check_gpu_availability(self) -> bool:
        """GPU 사용 가능성 확인"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"CUDA GPU {gpu_count}개 감지")
                return True
        except ImportError:
            pass
        
        try:
            # NVIDIA-SMI로 확인
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("NVIDIA GPU 감지 (nvidia-smi)")
                return True
        except:
            pass
        
        logger.info("GPU 사용 불가 - CPU 모드 사용")
        return False
    
    def get_current_status(self) -> ResourceStatus:
        """현재 리소스 상태 반환"""
        gpu_memory_free = 0.0
        gpu_memory_total = 0.0
        
        # GPU 메모리 정보
        if self.gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_free = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    # 실제 사용 가능한 메모리
                    torch.cuda.empty_cache()
                    gpu_memory_free = (gpu_memory_total - 
                                     torch.cuda.memory_allocated(0) / (1024**3))
            except:
                pass
        
        # CPU 및 RAM 정보
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        ram_available = memory.available / (1024**3)  # GB
        
        # 추천 모드 결정
        recommendation = self._get_recommendation(
            self.gpu_available, gpu_memory_free, cpu_percent, ram_available
        )
        
        return ResourceStatus(
            gpu_available=self.gpu_available,
            gpu_memory_free=gpu_memory_free,
            gpu_memory_total=gpu_memory_total,
            cpu_percent=cpu_percent,
            ram_available=ram_available,
            recommendation=recommendation
        )
    
    def _get_recommendation(self, gpu_available: bool, gpu_memory: float, 
                          cpu_percent: float, ram_available: float) -> str:
        """최적 리소스 추천"""
        if not gpu_available:
            return 'cpu'
        
        # GPU 메모리 부족
        if gpu_memory < 2.0:  # 2GB 미만
            return 'cpu'
        
        # CPU 과부하
        if cpu_percent > 80:
            return 'gpu'
        
        # RAM 부족
        if ram_available < 4.0:  # 4GB 미만
            return 'gpu'
        
        # 균형 상태 - GPU 선호 (일반적으로 더 빠름)
        if gpu_memory > 4.0:
            return 'gpu'
        else:
            return 'hybrid'  # 작은 작업은 CPU, 큰 작업은 GPU
    
    def configure_for_task(self, task_type: str, file_size_mb: float = 0) -> Dict[str, Any]:
        """작업에 최적화된 설정 반환"""
        status = self.get_current_status()
        
        config = {
            'use_gpu': False,
            'batch_size': 1,
            'num_workers': 1,
            'memory_limit': None,
            'reason': ''
        }
        
        # 작업 유형별 최적화
        if task_type == 'ocr':
            if status.recommendation == 'gpu' and file_size_mb < 50:  # 50MB 미만
                config.update({
                    'use_gpu': True,
                    'batch_size': min(4, int(status.gpu_memory_free)),
                    'reason': 'GPU 최적화 - 빠른 OCR 처리'
                })
            else:
                config.update({
                    'use_gpu': False,
                    'num_workers': min(4, psutil.cpu_count()),
                    'reason': 'CPU 최적화 - 안정적 OCR 처리'
                })
        
        elif task_type == 'whisper':
            # Whisper는 GPU에서 매우 빠름
            if status.recommendation in ['gpu', 'hybrid'] and status.gpu_memory_free > 3.0:
                config.update({
                    'use_gpu': True,
                    'memory_limit': int(status.gpu_memory_free * 0.8),  # 80% 사용
                    'reason': 'GPU 최적화 - 10배 빠른 음성 인식'
                })
            else:
                config.update({
                    'use_gpu': False,
                    'num_workers': 2,  # Whisper는 멀티프로세싱 제한적
                    'reason': 'CPU 모드 - GPU 메모리 부족 또는 안정성 우선'
                })
        
        elif task_type == 'multimodal':
            # 멀티모달은 많은 메모리 필요
            if status.recommendation == 'gpu' and status.gpu_memory_free > 6.0:
                config.update({
                    'use_gpu': True,
                    'batch_size': 2,
                    'reason': 'GPU 최적화 - 대용량 멀티모달 처리'
                })
            else:
                config.update({
                    'use_gpu': False,
                    'num_workers': psutil.cpu_count() // 2,
                    'reason': 'CPU 분산 처리 - 메모리 효율 우선'
                })
        
        logger.info(f"{task_type} 작업 설정: {config['reason']}")
        return config
    
    def monitor_performance(self, task_name: str, processing_time: float, 
                          success: bool, resource_used: str):
        """성능 모니터링 및 학습"""
        performance_record = {
            'timestamp': time.time(),
            'task': task_name,
            'processing_time': processing_time,
            'success': success,
            'resource': resource_used,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }
        
        self.performance_history.append(performance_record)
        
        # 최근 10개 기록만 유지
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]
        
        # 성능 분석 및 추천 업데이트
        self._analyze_performance()
    
    def _analyze_performance(self):
        """성능 분석 및 최적화 제안"""
        if len(self.performance_history) < 3:
            return
        
        recent_records = self.performance_history[-5:]  # 최근 5개
        
        gpu_times = [r['processing_time'] for r in recent_records 
                    if r['resource'] == 'gpu' and r['success']]
        cpu_times = [r['processing_time'] for r in recent_records 
                    if r['resource'] == 'cpu' and r['success']]
        
        if gpu_times and cpu_times:
            avg_gpu = sum(gpu_times) / len(gpu_times)
            avg_cpu = sum(cpu_times) / len(cpu_times)
            
            if avg_cpu < avg_gpu * 0.8:  # CPU가 20% 이상 빠르면
                logger.info(f"성능 분석: CPU 모드가 더 효율적 (CPU: {avg_cpu:.1f}s vs GPU: {avg_gpu:.1f}s)")
            elif avg_gpu < avg_cpu * 0.5:  # GPU가 50% 이상 빠르면
                logger.info(f"성능 분석: GPU 모드가 더 효율적 (GPU: {avg_gpu:.1f}s vs CPU: {avg_cpu:.1f}s)")
    
    def set_environment_for_config(self, config: Dict[str, Any]):
        """설정에 따른 환경 변수 설정"""
        if config['use_gpu']:
            # GPU 활성화
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                del os.environ['CUDA_VISIBLE_DEVICES']
            
            # GPU 메모리 제한 설정 (PyTorch)
            if config.get('memory_limit'):
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f"max_split_size_mb:{config['memory_limit']*1024}"
        else:
            # GPU 비활성화
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    def get_optimal_settings_for_easyocr(self) -> Dict[str, Any]:
        """EasyOCR 최적 설정"""
        status = self.get_current_status()
        
        if status.recommendation == 'gpu' and status.gpu_memory_free > 2.0:
            return {
                'gpu': True,
                'model_storage_directory': None,
                'download_enabled': True,
                'detector': True,
                'recognizer': True,
                'verbose': False,
                'quantize': False  # GPU에서는 양자화 불필요
            }
        else:
            return {
                'gpu': False,
                'model_storage_directory': None,
                'download_enabled': True,
                'detector': True,
                'recognizer': True,
                'verbose': False,
                'quantize': True  # CPU에서는 양자화로 속도 향상
            }
    
    def get_optimal_settings_for_whisper(self) -> Dict[str, Any]:
        """Whisper 최적 설정"""
        status = self.get_current_status()
        
        if status.recommendation == 'gpu' and status.gpu_memory_free > 3.0:
            # GPU 모드: 더 큰 모델 사용 가능
            model_size = 'large' if status.gpu_memory_free > 6.0 else 'medium'
            return {
                'model_size': model_size,
                'device': 'cuda',
                'fp16': True,  # GPU에서 FP16으로 메모리 절약
                'condition_on_previous_text': True,
                'temperature': 0.0  # 재현 가능한 결과
            }
        else:
            # CPU 모드: 작은 모델로 안정성 확보
            return {
                'model_size': 'base',  # CPU에서 적절한 크기
                'device': 'cpu',
                'fp16': False,
                'condition_on_previous_text': True,
                'temperature': 0.0
            }

# 전역 인스턴스
_resource_manager = None

def get_resource_manager() -> DynamicResourceManager:
    """리소스 관리자 인스턴스 반환"""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = DynamicResourceManager()
    return _resource_manager

# 편의 함수들
def get_optimal_ocr_settings(file_size_mb: float = 0) -> Dict[str, Any]:
    """OCR 최적 설정 반환"""
    manager = get_resource_manager()
    base_config = manager.configure_for_task('ocr', file_size_mb)
    easyocr_config = manager.get_optimal_settings_for_easyocr()
    
    return {**base_config, **easyocr_config}

def get_optimal_whisper_settings(file_size_mb: float = 0) -> Dict[str, Any]:
    """Whisper 최적 설정 반환"""
    manager = get_resource_manager()
    base_config = manager.configure_for_task('whisper', file_size_mb)
    whisper_config = manager.get_optimal_settings_for_whisper()
    
    return {**base_config, **whisper_config}

def log_performance(task_name: str, processing_time: float, 
                   success: bool, resource_used: str):
    """성능 로깅"""
    manager = get_resource_manager()
    manager.monitor_performance(task_name, processing_time, success, resource_used)

if __name__ == "__main__":
    # 동적 리소스 관리자 테스트
    manager = get_resource_manager()
    
    # 현재 상태 확인
    status = manager.get_current_status()
    print(f"GPU 사용 가능: {status.gpu_available}")
    print(f"GPU 메모리: {status.gpu_memory_free:.1f}/{status.gpu_memory_total:.1f} GB")
    print(f"CPU 사용률: {status.cpu_percent:.1f}%")
    print(f"RAM 사용 가능: {status.ram_available:.1f} GB")
    print(f"추천 모드: {status.recommendation}")
    
    print("\n--- 작업별 최적 설정 ---")
    
    # OCR 설정
    ocr_settings = get_optimal_ocr_settings(file_size_mb=10)
    print(f"OCR 설정: GPU={ocr_settings.get('use_gpu', False)}, {ocr_settings.get('reason', '')}")
    
    # Whisper 설정  
    whisper_settings = get_optimal_whisper_settings(file_size_mb=50)
    print(f"Whisper 설정: GPU={whisper_settings.get('use_gpu', False)}, {whisper_settings.get('reason', '')}")