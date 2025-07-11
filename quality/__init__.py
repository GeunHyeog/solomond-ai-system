"""
🔬 Quality Analysis Module v2.1
품질 검증 시스템 - 주얼리 AI 플랫폼 현장 최적화

품질 모듈:
- audio_quality_checker: 음성 품질 분석 (SNR, 노이즈, 명료도)
- ocr_quality_validator: OCR 품질 검증 (정확도, 신뢰도)
- image_quality_assessor: 이미지 품질 평가 (해상도, 블러, 조명)
- content_consistency_checker: 내용 일관성 검증 (음성-이미지-문서 매칭)
"""

from .audio_quality_checker import AudioQualityChecker
from .ocr_quality_validator import OCRQualityValidator
from .image_quality_assessor import ImageQualityAssessor
from .content_consistency_checker import ContentConsistencyChecker

__version__ = "2.1.0"
__author__ = "솔로몬드 AI팀"
