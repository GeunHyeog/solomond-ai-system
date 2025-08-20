#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 SOLOMOND AI Enhanced Modules Package
v7.1 시스템 개선 모듈들

이 패키지는 기존 시스템의 기능을 완전히 보존하면서
새로운 고급 기능들을 안전하게 추가하는 모듈들을 포함합니다.

모든 모듈은 기존 시스템과 완전히 격리되어 개발되며,
사용자가 선택적으로 활성화할 수 있습니다.
"""

__version__ = "7.1.0"
__author__ = "SOLOMOND AI Team"

# 개선 모듈 목록
ENHANCEMENT_MODULES = {
    'enhanced_ocr_engine': {
        'name': 'OCR 강화 엔진',
        'description': 'PPT 이미지 특화 다중 OCR 시스템',
        'status': 'development',
        'critical': True
    },
    'advanced_noise_processor': {
        'name': '고급 노이즈 처리기',
        'description': '오디오/이미지 품질 자동 향상',
        'status': 'planned',
        'critical': False
    },
    'multimodal_fusion_v2': {
        'name': '멀티모달 융합 v2',
        'description': '향상된 크로스모달 상관관계 분석',
        'status': 'planned',
        'critical': False
    },
    'precise_speaker_detector': {
        'name': '정밀 화자 탐지기',
        'description': '고급 화자 구분 및 추적 시스템',
        'status': 'planned',
        'critical': False
    },
    'performance_optimizer': {
        'name': '성능 최적화기',
        'description': 'GPU 가속 및 메모리 최적화',
        'status': 'planned',
        'critical': False
    },
    'insight_quality_enhancer': {
        'name': '인사이트 품질 향상기',
        'description': 'AI 기반 분석 품질 개선',
        'status': 'planned',
        'critical': False
    }
}

# 설정
DEFAULT_CONFIG = {
    'use_enhanced_ocr': False,
    'use_noise_reduction': False,
    'use_improved_fusion': False,
    'use_precise_speaker': False,
    'use_performance_optimizer': False,
    'use_quality_enhancer': False,
    'fallback_on_error': True,
    'compare_results': True,
    'log_performance': True
}

def get_module_info():
    """개선 모듈 정보 반환"""
    return ENHANCEMENT_MODULES

def get_default_config():
    """기본 설정 반환"""
    return DEFAULT_CONFIG.copy()