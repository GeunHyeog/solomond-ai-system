"""
🔧 GPU/CPU 컴퓨팅 설정 통합 모듈
솔로몬드 AI 시스템 - 중복 코드 제거 (2/3단계)

목적: 17개 파일에서 반복되는 GPU/CPU 설정을 중앙화
효과: 환경별 설정 관리 개선 및 하드코딩 제거
"""

import os
import platform
import subprocess
from typing import Optional, Dict, List
from enum import Enum
import logging

class ComputeMode(Enum):
    """컴퓨팅 모드 열거형"""
    AUTO = "auto"        # 자동 감지
    CPU_ONLY = "cpu"     # CPU만 사용
    GPU_CUDA = "cuda"    # NVIDIA CUDA GPU
    GPU_MPS = "mps"      # Apple Metal Performance Shaders
    GPU_OPENCL = "opencl" # OpenCL GPU

class ComputeConfig:
    """솔로몬드 AI 시스템 컴퓨팅 설정 관리자"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = logging.getLogger(__name__)
        self.current_mode = ComputeMode.AUTO
        self.available_modes = []
        self.gpu_info = {}
        
        # 시스템 정보 수집
        self._detect_system_capabilities()
        self._set_optimal_mode()
        
        self._initialized = True
    
    def _detect_system_capabilities(self):
        """시스템 GPU/CPU 기능 감지"""
        
        self.available_modes = [ComputeMode.CPU_ONLY]  # CPU는 항상 사용 가능
        
        # NVIDIA CUDA 지원 확인
        if self._check_cuda_support():
            self.available_modes.append(ComputeMode.GPU_CUDA)
            self.gpu_info['cuda'] = self._get_cuda_info()
        
        # Apple Metal 지원 확인 (macOS)
        if platform.system() == "Darwin" and self._check_mps_support():
            self.available_modes.append(ComputeMode.GPU_MPS)
            self.gpu_info['mps'] = True
        
        # OpenCL 지원 확인
        if self._check_opencl_support():
            self.available_modes.append(ComputeMode.GPU_OPENCL)
            self.gpu_info['opencl'] = True
        
        self.logger.info(f"사용 가능한 컴퓨팅 모드: {[mode.value for mode in self.available_modes]}")
    
    def _check_cuda_support(self) -> bool:
        """CUDA 지원 여부 확인"""
        try:
            # nvidia-smi 명령으로 GPU 확인
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _get_cuda_info(self) -> Dict:
        """CUDA GPU 정보 수집"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpus = []
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 3:
                            gpus.append({
                                'name': parts[0].strip(),
                                'memory_mb': int(parts[1].strip()),
                                'driver_version': parts[2].strip()
                            })
                return {'gpus': gpus, 'count': len(gpus)}
        except:
            pass
        
        return {'gpus': [], 'count': 0}
    
    def _check_mps_support(self) -> bool:
        """Apple Metal Performance Shaders 지원 확인"""
        try:
            import torch
            return torch.backends.mps.is_available()
        except:
            return False
    
    def _check_opencl_support(self) -> bool:
        """OpenCL 지원 확인"""
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            return len(platforms) > 0
        except:
            return False
    
    def _set_optimal_mode(self):
        """최적 컴퓨팅 모드 설정"""
        
        # 환경 변수 확인
        env_mode = os.getenv('SOLOMOND_COMPUTE_MODE', '').lower()
        if env_mode:
            for mode in ComputeMode:
                if mode.value == env_mode and mode in self.available_modes:
                    self.current_mode = mode
                    self.logger.info(f"환경 변수로 설정된 컴퓨팅 모드: {mode.value}")
                    return
        
        # 자동 모드 선택 (우선순위: CUDA > MPS > CPU)
        if ComputeMode.GPU_CUDA in self.available_modes:
            # CUDA GPU가 있지만 메모리가 부족한 경우 CPU 모드 선택
            cuda_info = self.gpu_info.get('cuda', {})
            if cuda_info.get('count', 0) > 0:
                min_memory = min(gpu['memory_mb'] for gpu in cuda_info['gpus'])
                if min_memory < 4000:  # 4GB 미만이면 CPU 모드
                    self.current_mode = ComputeMode.CPU_ONLY
                    self.logger.warning(f"GPU 메모리 부족 ({min_memory}MB < 4GB), CPU 모드로 전환")
                else:
                    self.current_mode = ComputeMode.GPU_CUDA
            else:
                self.current_mode = ComputeMode.CPU_ONLY
        elif ComputeMode.GPU_MPS in self.available_modes:
            self.current_mode = ComputeMode.GPU_MPS
        else:
            self.current_mode = ComputeMode.CPU_ONLY
        
        self.logger.info(f"자동 선택된 컴퓨팅 모드: {self.current_mode.value}")
    
    def set_mode(self, mode: ComputeMode) -> bool:
        """컴퓨팅 모드 설정
        
        Args:
            mode: 설정할 컴퓨팅 모드
            
        Returns:
            bool: 설정 성공 여부
        """
        
        if mode not in self.available_modes:
            self.logger.error(f"지원되지 않는 컴퓨팅 모드: {mode.value}")
            return False
        
        self.current_mode = mode
        self._apply_environment_settings()
        self.logger.info(f"컴퓨팅 모드 변경: {mode.value}")
        return True
    
    def _apply_environment_settings(self):
        """현재 모드에 따른 환경 변수 설정"""
        
        if self.current_mode == ComputeMode.CPU_ONLY:
            # CPU 전용 모드
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
            
        elif self.current_mode == ComputeMode.GPU_CUDA:
            # CUDA GPU 모드
            if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '':
                del os.environ['CUDA_VISIBLE_DEVICES']  # 기본값으로 복원
        
        # 메모리 관련 설정
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Hugging Face 경고 방지
    
    def get_device_string(self) -> str:
        """PyTorch/TensorFlow에서 사용할 device 문자열 반환
        
        Returns:
            str: device 문자열 ('cpu', 'cuda', 'mps' 등)
        """
        
        if self.current_mode == ComputeMode.CPU_ONLY:
            return 'cpu'
        elif self.current_mode == ComputeMode.GPU_CUDA:
            return 'cuda'
        elif self.current_mode == ComputeMode.GPU_MPS:
            return 'mps'
        else:
            return 'cpu'
    
    def get_whisper_device(self) -> str:
        """Whisper 모델용 device 설정 반환"""
        if self.current_mode == ComputeMode.CPU_ONLY:
            return 'cpu'
        elif self.current_mode == ComputeMode.GPU_CUDA:
            return 'cuda'
        else:
            return 'cpu'  # Whisper는 CUDA와 CPU만 지원
    
    def get_transformers_device(self) -> str:
        """Transformers 모델용 device 설정 반환"""
        return self.get_device_string()
    
    def is_gpu_available(self) -> bool:
        """GPU 사용 가능 여부"""
        return self.current_mode != ComputeMode.CPU_ONLY
    
    def get_memory_info(self) -> Dict:
        """메모리 사용량 정보 반환"""
        info = {'mode': self.current_mode.value}
        
        if self.current_mode == ComputeMode.GPU_CUDA and self.gpu_info.get('cuda'):
            try:
                import torch
                if torch.cuda.is_available():
                    info['gpu_memory_allocated'] = torch.cuda.memory_allocated()
                    info['gpu_memory_reserved'] = torch.cuda.memory_reserved()
                    info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory
            except:
                pass
        
        return info
    
    def optimize_for_inference(self):
        """추론 최적화 설정"""
        if self.current_mode == ComputeMode.GPU_CUDA:
            try:
                import torch
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            except:
                pass
    
    def get_recommended_batch_size(self, model_type: str = "default") -> int:
        """모델 타입에 따른 권장 배치 크기 반환"""
        
        batch_sizes = {
            'whisper': {
                ComputeMode.CPU_ONLY: 1,
                ComputeMode.GPU_CUDA: 4,
                ComputeMode.GPU_MPS: 2
            },
            'transformers': {
                ComputeMode.CPU_ONLY: 1,
                ComputeMode.GPU_CUDA: 8,
                ComputeMode.GPU_MPS: 4
            },
            'default': {
                ComputeMode.CPU_ONLY: 1,
                ComputeMode.GPU_CUDA: 2,
                ComputeMode.GPU_MPS: 2
            }
        }
        
        return batch_sizes.get(model_type, batch_sizes['default']).get(
            self.current_mode, 1
        )

# 전역 인스턴스
_compute_config = None

def get_compute_config() -> ComputeConfig:
    """전역 ComputeConfig 인스턴스 반환"""
    global _compute_config
    if _compute_config is None:
        _compute_config = ComputeConfig()
    return _compute_config

def setup_compute_environment(mode: str = None) -> ComputeConfig:
    """컴퓨팅 환경 설정 - 메인 진입점
    
    기존 코드에서 이렇게 사용:
    ```python
    from config.compute_config import setup_compute_environment
    config = setup_compute_environment()  # 또는 setup_compute_environment('cpu')
    ```
    
    Args:
        mode: 강제 설정할 모드 ('cpu', 'cuda', 'auto' 등)
        
    Returns:
        ComputeConfig: 설정된 컴퓨팅 설정 객체
    """
    
    config = get_compute_config()
    
    if mode:
        try:
            compute_mode = ComputeMode(mode.lower())
            config.set_mode(compute_mode)
        except ValueError:
            config.logger.warning(f"알 수 없는 컴퓨팅 모드: {mode}, 자동 모드 사용")
    
    return config

def force_cpu_mode():
    """CPU 전용 모드 강제 설정 (기존 코드 호환용)
    
    기존 코드에서 이렇게 사용:
    ```python
    from config.compute_config import force_cpu_mode
    force_cpu_mode()
    # 이후 os.environ['CUDA_VISIBLE_DEVICES'] = '' 와 동일 효과
    ```
    """
    config = get_compute_config()
    config.set_mode(ComputeMode.CPU_ONLY)

def get_device_string() -> str:
    """현재 설정된 device 문자열 반환 (기존 코드 호환용)"""
    return get_compute_config().get_device_string()

# 환경 변수 설정 자동 적용
_config = get_compute_config()  # 모듈 import 시 자동 초기화