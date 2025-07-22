#!/usr/bin/env python3
"""
GPU 메모리 관리 시스템
동적 할당, CPU 폴백, 메모리 모니터링 및 최적화
"""

import os
import gc
import logging
import psutil
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import threading
import warnings

# GPU 관련 임포트 (선택적)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False

class ComputeMode(Enum):
    """연산 모드"""
    AUTO = "auto"          # 자동 선택
    GPU_ONLY = "gpu_only"  # GPU 전용
    CPU_ONLY = "cpu_only"  # CPU 전용
    HYBRID = "hybrid"      # 하이브리드

@dataclass
class MemoryInfo:
    """메모리 정보 데이터클래스"""
    total_mb: float
    available_mb: float
    used_mb: float
    usage_percent: float
    device_name: str = "Unknown"

@dataclass
class GPUInfo:
    """GPU 정보 데이터클래스"""
    device_id: int
    name: str
    memory_total_mb: float
    memory_free_mb: float
    memory_used_mb: float
    utilization_percent: float
    temperature_c: Optional[float] = None
    power_draw_w: Optional[float] = None

class GPUMemoryManager:
    """GPU 메모리 관리 클래스"""
    
    def __init__(self, 
                 memory_threshold_mb: float = 1000,  # 최소 필요 메모리 (MB)
                 cpu_fallback_enabled: bool = True,
                 monitor_interval: float = 30.0):    # 모니터링 주기 (초)
        
        self.logger = self._setup_logging()
        self.memory_threshold_mb = memory_threshold_mb
        self.cpu_fallback_enabled = cpu_fallback_enabled
        self.monitor_interval = monitor_interval
        
        # 상태 관리
        self.current_mode = ComputeMode.AUTO
        self.gpu_available = False
        self.gpu_info_list: List[GPUInfo] = []
        self.selected_gpu_id = 0
        
        # 모니터링 관련
        self.monitoring_active = False
        self.monitor_thread = None
        self.memory_history = []
        
        # 초기화
        self._initialize_gpu_detection()
        self._set_initial_mode()
        
        # 환경 변수 설정
        self._configure_environment()
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.GPUMemoryManager')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - 🖥️  GPU - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_gpu_detection(self):
        """GPU 감지 및 초기화"""
        self.logger.info("🔍 GPU 감지 시작...")
        
        # NVIDIA GPU 감지
        if NVIDIA_ML_AVAILABLE:
            try:
                nvml.nvmlInit()
                device_count = nvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_info = self._get_gpu_info(handle, i)
                    self.gpu_info_list.append(gpu_info)
                
                if self.gpu_info_list:
                    self.gpu_available = True
                    self.selected_gpu_id = self._select_best_gpu()
                    
                    self.logger.info(f"✅ GPU 감지 완료: {len(self.gpu_info_list)}개")
                    for i, gpu in enumerate(self.gpu_info_list):
                        self.logger.info(f"   GPU {i}: {gpu.name} ({gpu.memory_total_mb:.0f}MB)")
                    self.logger.info(f"🎯 선택된 GPU: {self.selected_gpu_id}")
                else:
                    self.logger.info("❌ 사용 가능한 GPU 없음")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ NVIDIA GPU 감지 실패: {e}")
        
        # PyTorch GPU 감지 (보조)
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**2)  # MB
                    
                    gpu_info = GPUInfo(
                        device_id=i,
                        name=gpu_name,
                        memory_total_mb=gpu_memory,
                        memory_free_mb=gpu_memory * 0.9,  # 추정치
                        memory_used_mb=gpu_memory * 0.1,
                        utilization_percent=0.0
                    )
                    self.gpu_info_list.append(gpu_info)
                
                if self.gpu_info_list:
                    self.gpu_available = True
                    self.selected_gpu_id = 0
                    self.logger.info(f"✅ PyTorch GPU 감지: {gpu_count}개")
            
            except Exception as e:
                self.logger.warning(f"⚠️ PyTorch GPU 감지 실패: {e}")
        
        else:
            self.logger.info("ℹ️ GPU 감지 라이브러리 없음 (nvidia-ml-py3, torch)")
    
    def _get_gpu_info(self, handle, device_id: int) -> GPUInfo:
        """GPU 정보 수집"""
        try:
            # 기본 정보
            name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
            
            # 선택적 정보
            try:
                temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = None
            
            try:
                power_draw = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
            except:
                power_draw = None
            
            return GPUInfo(
                device_id=device_id,
                name=name,
                memory_total_mb=memory_info.total / (1024**2),
                memory_free_mb=memory_info.free / (1024**2),
                memory_used_mb=memory_info.used / (1024**2),
                utilization_percent=utilization.gpu,
                temperature_c=temperature,
                power_draw_w=power_draw
            )
            
        except Exception as e:
            self.logger.warning(f"GPU {device_id} 정보 수집 실패: {e}")
            return GPUInfo(
                device_id=device_id,
                name="Unknown GPU",
                memory_total_mb=0,
                memory_free_mb=0,
                memory_used_mb=0,
                utilization_percent=0
            )
    
    def _select_best_gpu(self) -> int:
        """최적 GPU 선택"""
        if not self.gpu_info_list:
            return 0
        
        # 여유 메모리가 가장 많은 GPU 선택
        best_gpu_id = 0
        max_free_memory = 0
        
        for gpu in self.gpu_info_list:
            if gpu.memory_free_mb > max_free_memory:
                max_free_memory = gpu.memory_free_mb
                best_gpu_id = gpu.device_id
        
        return best_gpu_id
    
    def _set_initial_mode(self):
        """초기 연산 모드 설정"""
        if self.gpu_available:
            # GPU 메모리 충분성 확인
            selected_gpu = self.gpu_info_list[self.selected_gpu_id]
            if selected_gpu.memory_free_mb >= self.memory_threshold_mb:
                self.current_mode = ComputeMode.GPU_ONLY
                self.logger.info(f"🚀 초기 모드: GPU 사용 (여유 메모리: {selected_gpu.memory_free_mb:.0f}MB)")
            else:
                self.current_mode = ComputeMode.CPU_ONLY
                self.logger.info(f"🔄 초기 모드: CPU 사용 (GPU 메모리 부족: {selected_gpu.memory_free_mb:.0f}MB)")
        else:
            self.current_mode = ComputeMode.CPU_ONLY
            self.logger.info("💻 초기 모드: CPU 사용 (GPU 없음)")
    
    def _configure_environment(self):
        """환경 변수 설정"""
        if self.current_mode == ComputeMode.CPU_ONLY:
            # CPU 모드 강제
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            self.logger.info("⚙️ CPU 모드로 환경 설정 완료")
            
        elif self.current_mode == ComputeMode.GPU_ONLY:
            # 선택된 GPU만 사용
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.selected_gpu_id)
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            self.logger.info(f"⚙️ GPU {self.selected_gpu_id} 모드로 환경 설정 완료")
    
    def get_recommended_mode(self, required_memory_mb: float = None) -> ComputeMode:
        """권장 연산 모드 반환"""
        if required_memory_mb is None:
            required_memory_mb = self.memory_threshold_mb
        
        if not self.gpu_available:
            return ComputeMode.CPU_ONLY
        
        # 현재 GPU 상태 업데이트
        self._update_gpu_info()
        
        # 선택된 GPU 메모리 확인
        selected_gpu = self.gpu_info_list[self.selected_gpu_id]
        
        if selected_gpu.memory_free_mb >= required_memory_mb:
            return ComputeMode.GPU_ONLY
        elif self.cpu_fallback_enabled:
            return ComputeMode.CPU_ONLY
        else:
            return ComputeMode.GPU_ONLY  # 강제 GPU 사용
    
    def switch_mode(self, target_mode: ComputeMode, required_memory_mb: float = None) -> bool:
        """연산 모드 전환"""
        if target_mode == self.current_mode:
            return True
        
        self.logger.info(f"🔄 모드 전환: {self.current_mode.value} → {target_mode.value}")
        
        # 메모리 정리
        self._cleanup_memory()
        
        # 모드별 설정
        if target_mode == ComputeMode.CPU_ONLY:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
        elif target_mode == ComputeMode.GPU_ONLY:
            if not self.gpu_available:
                self.logger.warning("⚠️ GPU 없음, CPU 모드 유지")
                return False
            
            # GPU 메모리 확인
            recommended_mode = self.get_recommended_mode(required_memory_mb)
            if recommended_mode != ComputeMode.GPU_ONLY:
                self.logger.warning("⚠️ GPU 메모리 부족, CPU 모드로 폴백")
                target_mode = ComputeMode.CPU_ONLY
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.selected_gpu_id)
        
        self.current_mode = target_mode
        
        # PyTorch 캐시 정리 (사용 가능한 경우)
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        self.logger.info(f"✅ 모드 전환 완료: {self.current_mode.value}")
        return True
    
    def _cleanup_memory(self):
        """메모리 정리"""
        # Python 가비지 컬렉션
        gc.collect()
        
        # PyTorch 캐시 정리
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("🧹 메모리 정리 완료")
    
    def _update_gpu_info(self):
        """GPU 정보 업데이트"""
        if not NVIDIA_ML_AVAILABLE or not self.gpu_available:
            return
        
        try:
            for i, gpu_info in enumerate(self.gpu_info_list):
                handle = nvml.nvmlDeviceGetHandleByIndex(gpu_info.device_id)
                updated_info = self._get_gpu_info(handle, gpu_info.device_id)
                self.gpu_info_list[i] = updated_info
                
        except Exception as e:
            self.logger.warning(f"GPU 정보 업데이트 실패: {e}")
    
    def get_memory_info(self) -> Dict[str, MemoryInfo]:
        """시스템 및 GPU 메모리 정보 반환"""
        info = {}
        
        # 시스템 RAM 정보
        ram = psutil.virtual_memory()
        info['system_ram'] = MemoryInfo(
            total_mb=ram.total / (1024**2),
            available_mb=ram.available / (1024**2),
            used_mb=ram.used / (1024**2),
            usage_percent=ram.percent,
            device_name="System RAM"
        )
        
        # GPU 메모리 정보
        if self.gpu_available:
            self._update_gpu_info()
            for gpu in self.gpu_info_list:
                info[f'gpu_{gpu.device_id}'] = MemoryInfo(
                    total_mb=gpu.memory_total_mb,
                    available_mb=gpu.memory_free_mb,
                    used_mb=gpu.memory_used_mb,
                    usage_percent=(gpu.memory_used_mb / gpu.memory_total_mb) * 100,
                    device_name=gpu.name
                )
        
        return info
    
    def start_monitoring(self) -> bool:
        """메모리 모니터링 시작"""
        if self.monitoring_active:
            return True
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"📊 메모리 모니터링 시작 (주기: {self.monitor_interval}초)")
        return True
    
    def stop_monitoring(self):
        """메모리 모니터링 중지"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("📊 메모리 모니터링 중지")
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                memory_info = self.get_memory_info()
                
                # 히스토리에 추가 (최근 100개만 유지)
                self.memory_history.append({
                    'timestamp': time.time(),
                    'memory_info': memory_info
                })
                
                if len(self.memory_history) > 100:
                    self.memory_history = self.memory_history[-100:]
                
                # 메모리 부족 경고
                for device, info in memory_info.items():
                    if info.usage_percent > 90:
                        self.logger.warning(f"⚠️ {device} 메모리 부족: {info.usage_percent:.1f}%")
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.warning(f"모니터링 오류: {e}")
                time.sleep(self.monitor_interval)
    
    def get_status_report(self) -> Dict[str, Any]:
        """상태 보고서 생성"""
        memory_info = self.get_memory_info()
        
        return {
            "current_mode": self.current_mode.value,
            "gpu_available": self.gpu_available,
            "gpu_count": len(self.gpu_info_list),
            "selected_gpu_id": self.selected_gpu_id if self.gpu_available else None,
            "memory_threshold_mb": self.memory_threshold_mb,
            "cpu_fallback_enabled": self.cpu_fallback_enabled,
            "monitoring_active": self.monitoring_active,
            "memory_info": {k: {
                "total_mb": round(v.total_mb, 1),
                "available_mb": round(v.available_mb, 1),
                "used_mb": round(v.used_mb, 1),
                "usage_percent": round(v.usage_percent, 1),
                "device_name": v.device_name
            } for k, v in memory_info.items()},
            "gpu_details": [
                {
                    "device_id": gpu.device_id,
                    "name": gpu.name,
                    "memory_total_mb": round(gpu.memory_total_mb, 1),
                    "memory_free_mb": round(gpu.memory_free_mb, 1),
                    "utilization_percent": round(gpu.utilization_percent, 1),
                    "temperature_c": gpu.temperature_c,
                    "power_draw_w": gpu.power_draw_w
                } for gpu in self.gpu_info_list
            ]
        }
    
    def __del__(self):
        """소멸자"""
        if hasattr(self, 'monitoring_active'):
            self.stop_monitoring()

# 전역 GPU 메모리 매니저 인스턴스
global_gpu_manager = GPUMemoryManager()

def get_recommended_compute_mode(required_memory_mb: float = 1000) -> ComputeMode:
    """권장 연산 모드 반환"""
    return global_gpu_manager.get_recommended_mode(required_memory_mb)

def switch_compute_mode(target_mode: ComputeMode, required_memory_mb: float = None) -> bool:
    """연산 모드 전환"""
    return global_gpu_manager.switch_mode(target_mode, required_memory_mb)

def get_memory_status() -> Dict[str, Any]:
    """메모리 상태 정보 반환"""
    return global_gpu_manager.get_status_report()

def cleanup_memory():
    """메모리 정리"""
    global_gpu_manager._cleanup_memory()

if __name__ == "__main__":
    # 테스트 코드
    manager = GPUMemoryManager()
    
    print("=== GPU 메모리 매니저 테스트 ===")
    
    # 상태 보고
    status = manager.get_status_report()
    print(f"현재 모드: {status['current_mode']}")
    print(f"GPU 사용 가능: {status['gpu_available']}")
    
    # 메모리 정보
    for device, info in status['memory_info'].items():
        print(f"{device}: {info['used_mb']:.1f}MB / {info['total_mb']:.1f}MB ({info['usage_percent']:.1f}%)")
    
    # 모니터링 테스트
    if input("모니터링을 시작하시겠습니까? (y/n): ").lower() == 'y':
        manager.start_monitoring()
        time.sleep(10)
        manager.stop_monitoring()