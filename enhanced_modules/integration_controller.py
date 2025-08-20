#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 통합 제어기 (Integration Controller)
Enhanced Modules Integration & Control System

기능:
1. 개선 모듈들의 안전한 통합 관리
2. 설정 기반 기능 활성화/비활성화
3. 자동 폴백 시스템
4. 성능 비교 및 모니터링
5. 사용자 선택 인터페이스
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import threading
import traceback

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 시스템 보호 import
try:
    from system_protection import get_system_protection
    PROTECTION_AVAILABLE = True
except ImportError:
    PROTECTION_AVAILABLE = False
    print("⚠️ 시스템 보호 모듈을 찾을 수 없습니다")

# 로깅 설정
logger = logging.getLogger(__name__)

@dataclass
class ModuleResult:
    """모듈 처리 결과"""
    module_name: str
    success: bool
    result: Any
    processing_time: float
    error_message: Optional[str] = None
    fallback_used: bool = False

@dataclass
class ComparisonResult:
    """기존 vs 개선 비교 결과"""
    original_result: ModuleResult
    enhanced_result: ModuleResult
    improvement_score: Optional[float] = None
    recommendation: str = "original"  # 'original', 'enhanced', 'hybrid'

class IntegrationController:
    """통합 제어기 - 개선 모듈들의 안전한 통합 관리"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "enhanced_modules_config.json"
        self.config = self.load_config()
        self.module_registry = {}
        self.performance_history = []
        self.lock = threading.Lock()
        
        # 시스템 보호 연동
        if PROTECTION_AVAILABLE:
            self.protector = get_system_protection()
        else:
            self.protector = None
        
        logger.info("🔧 통합 제어기 초기화 완료")
    
    def load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        config_path = PROJECT_ROOT / self.config_file
        
        # 기본 설정
        default_config = {
            'enhancements': {
                'use_enhanced_ocr': False,
                'use_noise_reduction': False,
                'use_improved_fusion': False,
                'use_precise_speaker': False,
                'use_performance_optimizer': False,
                'use_quality_enhancer': False
            },
            'safety': {
                'fallback_on_error': True,
                'compare_results': True,
                'log_performance': True,
                'max_processing_time': 300,  # 5분
                'auto_disable_on_failure': True
            },
            'performance': {
                'max_response_time_factor': 2.0,  # 기존 대비 2배까지 허용
                'min_accuracy_improvement': 0.05,  # 최소 5% 개선
                'enable_parallel_processing': True
            }
        }
        
        # 기존 설정 파일 로드 (있다면)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # 기본 설정에 로드된 설정 병합
                    self._deep_update(default_config, loaded_config)
            except Exception as e:
                logger.warning(f"설정 파일 로드 실패, 기본 설정 사용: {e}")
        
        return default_config
    
    def save_config(self):
        """설정 파일 저장"""
        config_path = PROJECT_ROOT / self.config_file
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"설정 파일 저장 실패: {e}")
    
    def register_module(self, module_name: str, original_function: Callable, 
                       enhanced_function: Optional[Callable] = None):
        """모듈 등록"""
        self.module_registry[module_name] = {
            'original': original_function,
            'enhanced': enhanced_function,
            'enabled': self.config['enhancements'].get(f'use_{module_name}', False),
            'performance_history': []
        }
        logger.info(f"📦 모듈 등록: {module_name}")
    
    def process_with_enhancement(self, module_name: str, *args, **kwargs) -> ModuleResult:
        """개선 모듈을 사용한 처리 (폴백 포함)"""
        if module_name not in self.module_registry:
            raise ValueError(f"등록되지 않은 모듈: {module_name}")
        
        module_info = self.module_registry[module_name]
        
        # 개선 모듈이 활성화되고 사용 가능한 경우
        if (module_info['enabled'] and 
            module_info['enhanced'] is not None and
            self.config['enhancements'].get(f'use_{module_name}', False)):
            
            return self._process_with_fallback(module_name, True, *args, **kwargs)
        else:
            # 기존 모듈 사용
            return self._process_with_fallback(module_name, False, *args, **kwargs)
    
    def compare_modules(self, module_name: str, *args, **kwargs) -> ComparisonResult:
        """기존 vs 개선 모듈 비교"""
        if module_name not in self.module_registry:
            raise ValueError(f"등록되지 않은 모듈: {module_name}")
        
        module_info = self.module_registry[module_name]
        
        if not module_info['enhanced']:
            raise ValueError(f"개선 모듈이 없음: {module_name}")
        
        # 병렬 처리 (가능한 경우)
        if self.config['performance']['enable_parallel_processing']:
            return self._compare_parallel(module_name, *args, **kwargs)
        else:
            return self._compare_sequential(module_name, *args, **kwargs)
    
    def _compare_parallel(self, module_name: str, *args, **kwargs) -> ComparisonResult:
        """병렬 비교 처리"""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # 기존 모듈 실행
            future_original = executor.submit(
                self._process_with_fallback, module_name, False, *args, **kwargs
            )
            
            # 개선 모듈 실행
            future_enhanced = executor.submit(
                self._process_with_fallback, module_name, True, *args, **kwargs
            )
            
            # 결과 수집
            original_result = future_original.result(timeout=self.config['safety']['max_processing_time'])
            enhanced_result = future_enhanced.result(timeout=self.config['safety']['max_processing_time'])
        
        return self._create_comparison_result(original_result, enhanced_result)
    
    def _compare_sequential(self, module_name: str, *args, **kwargs) -> ComparisonResult:
        """순차 비교 처리"""
        # 기존 모듈 실행
        original_result = self._process_with_fallback(module_name, False, *args, **kwargs)
        
        # 개선 모듈 실행
        enhanced_result = self._process_with_fallback(module_name, True, *args, **kwargs)
        
        return self._create_comparison_result(original_result, enhanced_result)
    
    def _create_comparison_result(self, original: ModuleResult, enhanced: ModuleResult) -> ComparisonResult:
        """비교 결과 생성"""
        # 개선 점수 계산
        improvement_score = None
        recommendation = "original"
        
        if original.success and enhanced.success:
            # 성능 비교 (처리 시간 기준)
            time_factor = enhanced.processing_time / max(original.processing_time, 0.001)
            
            if time_factor <= self.config['performance']['max_response_time_factor']:
                if enhanced.error_message is None:
                    recommendation = "enhanced"
                    improvement_score = 1.0 / time_factor  # 빠를수록 높은 점수
        
        elif enhanced.success and not original.success:
            recommendation = "enhanced"
            improvement_score = 1.0
        
        elif original.success and not enhanced.success:
            recommendation = "original"
            improvement_score = 0.0
        
        return ComparisonResult(
            original_result=original,
            enhanced_result=enhanced,
            improvement_score=improvement_score,
            recommendation=recommendation
        )
    
    def _process_with_fallback(self, module_name: str, use_enhanced: bool, 
                              *args, **kwargs) -> ModuleResult:
        """폴백을 포함한 안전한 처리"""
        module_info = self.module_registry[module_name]
        
        target_function = module_info['enhanced'] if use_enhanced else module_info['original']
        fallback_function = module_info['original']
        
        if not target_function:
            # 대상 함수가 없으면 기존 함수 사용
            target_function = fallback_function
            use_enhanced = False
        
        start_time = time.time()
        
        try:
            # 주 함수 실행
            result = target_function(*args, **kwargs)
            processing_time = time.time() - start_time
            
            return ModuleResult(
                module_name=module_name,
                success=True,
                result=result,
                processing_time=processing_time,
                fallback_used=False
            )
            
        except Exception as e:
            logger.warning(f"⚠️ {module_name} {'개선' if use_enhanced else '기존'} 모듈 실패: {e}")
            
            # 폴백 사용 (개선 모듈 실패시에만)
            if use_enhanced and self.config['safety']['fallback_on_error']:
                try:
                    logger.info(f"🔄 {module_name} 폴백 실행")
                    fallback_result = fallback_function(*args, **kwargs)
                    processing_time = time.time() - start_time
                    
                    return ModuleResult(
                        module_name=module_name,
                        success=True,
                        result=fallback_result,
                        processing_time=processing_time,
                        error_message=f"개선 모듈 실패, 폴백 사용: {str(e)}",
                        fallback_used=True
                    )
                    
                except Exception as fallback_error:
                    logger.error(f"❌ {module_name} 폴백도 실패: {fallback_error}")
            
            # 완전 실패
            processing_time = time.time() - start_time
            return ModuleResult(
                module_name=module_name,
                success=False,
                result=None,
                processing_time=processing_time,
                error_message=str(e),
                fallback_used=False
            )
    
    def update_module_setting(self, module_name: str, enabled: bool):
        """모듈 설정 업데이트"""
        if module_name in self.module_registry:
            self.module_registry[module_name]['enabled'] = enabled
            self.config['enhancements'][f'use_{module_name}'] = enabled
            self.save_config()
            logger.info(f"🔧 {module_name} 모듈 {'활성화' if enabled else '비활성화'}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """통합 제어기 상태 반환"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'registered_modules': len(self.module_registry),
            'active_enhancements': sum(1 for info in self.module_registry.values() if info['enabled']),
            'modules': {},
            'config': self.config
        }
        
        for module_name, module_info in self.module_registry.items():
            status['modules'][module_name] = {
                'enabled': module_info['enabled'],
                'has_enhanced': module_info['enhanced'] is not None,
                'performance_samples': len(module_info['performance_history'])
            }
        
        return status
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """딕셔너리 깊은 업데이트"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

# 전역 인스턴스
_integration_controller = None

def get_integration_controller() -> IntegrationController:
    """통합 제어기 인스턴스 반환 (싱글톤)"""
    global _integration_controller
    if _integration_controller is None:
        _integration_controller = IntegrationController()
    return _integration_controller

if __name__ == "__main__":
    # 통합 제어기 테스트
    controller = get_integration_controller()
    
    # 테스트용 함수들
    def original_ocr(text):
        return f"Original OCR: {text}"
    
    def enhanced_ocr(text):
        return f"Enhanced OCR: {text.upper()}"
    
    # 모듈 등록
    controller.register_module('ocr', original_ocr, enhanced_ocr)
    
    # 기본 처리
    result = controller.process_with_enhancement('ocr', "test image")
    print(f"처리 결과: {result}")
    
    # 비교 처리
    comparison = controller.compare_modules('ocr', "test image")
    print(f"비교 결과: {comparison.recommendation}")
    
    # 상태 확인
    status = controller.get_system_status()
    print(f"시스템 상태: {status['active_enhancements']}개 활성화")