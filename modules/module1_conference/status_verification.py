"""
SOLOMOND AI 시스템 무결성 - 상태 검증 표준 함수들
허위 상태 표시 문제 해결을 위한 검증 시스템

작성일: 2025-08-11
목적: 54% 시스템 무결성 정확도 → 95% 향상
"""

import os
import sys
import importlib.util
import logging
from pathlib import Path
from typing import Callable, Dict, Any, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_status(feature_name: str, check_function: Callable[[], bool]) -> str:
    """
    상태 검증 표준 함수
    
    Args:
        feature_name: 기능명 (예: "멀티모달 시스템")
        check_function: 실제 검증을 수행할 함수
        
    Returns:
        실제 검증된 상태 메시지
    """
    try:
        result = check_function()
        if result:
            logger.info(f"✅ {feature_name}: 검증됨")
            return f"✅ {feature_name}: 검증됨"
        else:
            logger.warning(f"❌ {feature_name}: 사용 불가")
            return f"❌ {feature_name}: 사용 불가"
    except Exception as e:
        logger.error(f"⚠️ {feature_name}: 검증 실패 ({e})")
        return f"⚠️ {feature_name}: 검증 실패 ({e})"


def verify_completion(task_name: str, validation_function: Callable[[], bool]) -> bool:
    """
    완료 상태 검증 표준 함수
    
    Args:
        task_name: 작업명
        validation_function: 검증 함수
        
    Returns:
        실제 완료 여부
    """
    try:
        if validation_function():
            logger.info(f"✅ {task_name} 완료 (검증됨)")
            return True
        else:
            logger.warning(f"⚠️ {task_name} 완료되지 않음 (검증 실패)")
            return False
    except Exception as e:
        logger.error(f"❌ {task_name} 검증 오류: {e}")
        return False


def verify_activation(system_name: str, activation_check: Callable[[], bool]) -> str:
    """
    활성화 상태 검증 표준 함수
    
    Args:
        system_name: 시스템명 (예: "멀티모달 시스템")
        activation_check: 활성화 확인 함수
        
    Returns:
        실제 활성화 상태 메시지
    """
    try:
        is_active = activation_check()
        if is_active:
            logger.info(f"🟢 {system_name}: 활성화됨 (검증됨)")
            return f"🟢 {system_name}: 활성화됨"
        else:
            logger.warning(f"🔴 {system_name}: 비활성화됨")
            return f"🔴 {system_name}: 비활성화됨"
    except Exception as e:
        logger.error(f"🟡 {system_name}: 상태 불명 ({e})")
        return f"🟡 {system_name}: 상태 불명 ({e})"


class MultimodalSystemVerifier:
    """멀티모달 시스템 검증 클래스"""
    
    def __init__(self):
        self.current_dir = Path(__file__).parent.parent.parent
        
    def check_multimodal_availability(self) -> bool:
        """멀티모달 시스템 사용 가능 여부 실제 확인"""
        try:
            # 1. 필수 모듈 존재 확인
            multimodal_file = self.current_dir / "multimodal_speaker_diarization.py"
            if not multimodal_file.exists():
                return False
                
            # 2. 필수 의존성 확인
            required_modules = ['librosa', 'cv2', 'numpy', 'scikit-learn']
            for module_name in required_modules:
                try:
                    __import__(module_name)
                except ImportError:
                    return False
            
            # 3. 실제 클래스 import 가능 여부 확인
            spec = importlib.util.spec_from_file_location(
                "multimodal_speaker_diarization", 
                multimodal_file
            )
            if spec is None:
                return False
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 4. 핵심 클래스 존재 확인
            if not hasattr(module, 'MultimodalSpeakerDiarization'):
                return False
                
            return True
            
        except Exception:
            return False
    
    def check_multimodal_activation(self) -> bool:
        """멀티모달 시스템이 실제로 활성화 가능한지 확인"""
        try:
            if not self.check_multimodal_availability():
                return False
                
            # 실제 인스턴스 생성 테스트
            multimodal_file = self.current_dir / "multimodal_speaker_diarization.py"
            spec = importlib.util.spec_from_file_location(
                "multimodal_speaker_diarization", 
                multimodal_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 인스턴스 생성 테스트 (초기화만)
            instance = module.MultimodalSpeakerDiarization()
            return hasattr(instance, 'analyze_multimodal')
            
        except Exception:
            return False


class OllamaVerifier:
    """Ollama AI 시스템 검증 클래스"""
    
    def check_ollama_availability(self) -> bool:
        """Ollama 사용 가능 여부 실제 확인"""
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False
    
    def check_models_availability(self) -> Dict[str, bool]:
        """Ollama 모델들 사용 가능 여부 확인"""
        models = ['qwen2.5:7b', 'gemma3:27b', 'gemma:4b', 'qwen3:8b', 'qwen:8b']
        results = {}
        
        if not self.check_ollama_availability():
            return {model: False for model in models}
            
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                available_models = result.stdout.lower()
                for model in models:
                    results[model] = model.lower() in available_models
            else:
                results = {model: False for model in models}
        except Exception:
            results = {model: False for model in models}
            
        return results


def get_system_verifiers():
    """시스템 검증자들 인스턴스 반환"""
    return {
        'multimodal': MultimodalSystemVerifier(),
        'ollama': OllamaVerifier()
    }


# 실제 사용 예시
def example_usage():
    """사용 예시"""
    verifiers = get_system_verifiers()
    
    # 멀티모달 시스템 상태 검증
    multimodal_status = verify_activation(
        "멀티모달 시스템",
        verifiers['multimodal'].check_multimodal_activation
    )
    print(multimodal_status)
    
    # Ollama 시스템 상태 검증
    ollama_status = verify_status(
        "Ollama AI 시스템",
        verifiers['ollama'].check_ollama_availability
    )
    print(ollama_status)


if __name__ == "__main__":
    example_usage()