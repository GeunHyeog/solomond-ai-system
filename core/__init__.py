#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
솔로몬드 AI v2.1.1 - 핵심 모듈 패키지

주요 모듈:
- quality_analyzer_v21: 음성/이미지 품질 검증 엔진
- multilingual_processor_v21: 다국어 처리 및 한국어 통합 분석

작성자: 전근혁 (솔로몬드 대표)
생성일: 2025.07.11
"""

__version__ = "2.1.1"
__author__ = "전근혁 (솔로몬드)"
__email__ = "solomond.jgh@gmail.com"

# 핵심 클래스들을 패키지 레벨에서 import 가능하도록 설정
try:
    from .quality_analyzer_v21 import QualityManager, AudioQualityAnalyzer, OCRQualityAnalyzer
    from .multilingual_processor_v21 import MultilingualProcessor, LanguageDetector, JewelryTermTranslator
    
    __all__ = [
        'QualityManager',
        'AudioQualityAnalyzer', 
        'OCRQualityAnalyzer',
        'MultilingualProcessor',
        'LanguageDetector',
        'JewelryTermTranslator'
    ]
    
except ImportError as e:
    # 의존성 모듈이 없어도 패키지는 import 가능하도록
    print(f"[WARNING] 일부 의존성 모듈 누락: {e}")
    print("[INFO] 필요 패키지: numpy, cv2, librosa")
    print("[INFO] 설치 명령: pip install numpy opencv-python librosa")
    
    __all__ = []

print(f"솔로몬드 AI v{__version__} 핵심 모듈 로드 완료")