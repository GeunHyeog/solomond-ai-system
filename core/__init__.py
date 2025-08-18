#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
솔로몬드 AI v4.1 - 핵심 모듈 패키지

주요 모듈:
- quality_analyzer_v21: 음성/이미지 품질 검증 엔진
- multilingual_processor_v21: 다국어 처리 및 한국어 통합 분석
- multimodal_encoder: 멀티모달 인코딩 엔진
- crossmodal_fusion: 크로스 모달 퓨전 시스템
- ollama_decoder: Ollama AI 디코더
- crossmodal_visualization: 크로스 모달 시각화

작성자: 전근혁 (솔로몬드 대표)
생성일: 2025.07.11
업데이트: 2025.08.12 (멀티모달 시스템 통합)
"""

__version__ = "4.1"
__author__ = "전근혁 (솔로몬드)"
__email__ = "solomond.jgh@gmail.com"

# 핵심 클래스들을 패키지 레벨에서 import 가능하도록 설정
try:
    # 기존 v2.1.1 모듈들
    from .quality_analyzer_v21 import QualityManager, AudioQualityAnalyzer, OCRQualityAnalyzer
    from .multilingual_processor_v21 import MultilingualProcessor, LanguageDetector, JewelryTermTranslator
    
    # v4.1 멀티모달 시스템 모듈들
    from . import multimodal_encoder
    from . import crossmodal_fusion  
    from . import ollama_decoder
    from . import crossmodal_visualization
    
    __all__ = [
        # 기존 모듈들
        'QualityManager',
        'AudioQualityAnalyzer', 
        'OCRQualityAnalyzer',
        'MultilingualProcessor',
        'LanguageDetector',
        'JewelryTermTranslator',
        # 새로운 멀티모달 모듈들
        'multimodal_encoder',
        'crossmodal_fusion',
        'ollama_decoder',
        'crossmodal_visualization'
    ]
    
except ImportError as e:
    # 의존성 모듈이 없어도 패키지는 import 가능하도록
    print(f"[WARNING] 일부 의존성 모듈 누락: {e}")
    print("[INFO] 필요 패키지: numpy, cv2, librosa, torch, transformers")
    print("[INFO] 설치 명령: pip install numpy opencv-python librosa torch transformers")
    
    __all__ = []

print(f"솔로몬드 AI v{__version__} 핵심 모듈 로드 완료")