#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 Enhanced OCR Engine - PPT 이미지 특화 OCR 시스템
Advanced Multi-Engine OCR System for Presentation Images

핵심 기능:
1. PPT 이미지 특화 전처리 파이프라인
2. 다중 OCR 엔진 통합 (EasyOCR + Tesseract + PaddleOCR)
3. 결과 교차 검증 및 최적 선택
4. 프로젝션 색상 왜곡 자동 보정
5. 복잡한 배경에서 텍스트 분리
6. 적응형 신뢰도 임계값 조정
"""

import os
import sys
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
import logging
from pathlib import Path
import json
import hashlib
from PIL import Image, ImageEnhance, ImageFilter
import math

# 로깅 설정
logger = logging.getLogger(__name__)

# OCR 엔진 import (사용 가능한 것만)
OCR_ENGINES = {}

try:
    import easyocr
    OCR_ENGINES['easyocr'] = True
    logger.info("✅ EasyOCR 사용 가능")
except ImportError:
    OCR_ENGINES['easyocr'] = False
    logger.warning("⚠️ EasyOCR 사용 불가")

try:
    import pytesseract
    from PIL import Image
    OCR_ENGINES['tesseract'] = True
    logger.info("✅ Tesseract 사용 가능")
except ImportError:
    OCR_ENGINES['tesseract'] = False
    logger.warning("⚠️ Tesseract 사용 불가")

try:
    # PaddleOCR은 선택적으로 사용
    import paddleocr
    OCR_ENGINES['paddleocr'] = True
    logger.info("✅ PaddleOCR 사용 가능")
except ImportError:
    OCR_ENGINES['paddleocr'] = False
    logger.info("ℹ️ PaddleOCR 사용 불가 (선택사항)")

@dataclass
class OCRResult:
    """OCR 결과 구조"""
    engine: str
    text: str
    confidence: float
    processing_time: float
    bbox_count: int
    metadata: Dict[str, Any]

@dataclass 
class EnhancedOCRResult:
    """강화된 OCR 최종 결과"""
    final_text: str
    confidence: float
    processing_time: float
    engines_used: List[str]
    best_engine: str
    individual_results: List[OCRResult]
    preprocessing_applied: List[str]
    image_quality_score: float

class PPTImagePreprocessor:
    """PPT 이미지 전처리기"""
    
    def __init__(self):
        self.preprocess_history = []
    
    def enhance_ppt_image(self, image_array: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """PPT 이미지 특화 전처리"""
        applied_operations = []
        processed_image = image_array.copy()
        
        try:
            # 1. 기본 색상 공간 변환 및 정규화
            if len(processed_image.shape) == 3:
                # 밝기 및 대비 자동 조정
                processed_image = self._auto_adjust_brightness_contrast(processed_image)
                applied_operations.append("brightness_contrast_adjustment")
                
                # 색상 정규화 (프로젝터 색왜곡 보정)
                processed_image = self._normalize_projection_colors(processed_image)
                applied_operations.append("projection_color_normalization")
            
            # 2. 노이즈 제거
            processed_image = cv2.bilateralFilter(processed_image, 9, 75, 75)
            applied_operations.append("bilateral_noise_reduction")
            
            # 3. 선명도 향상
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            processed_image = cv2.filter2D(processed_image, -1, kernel)
            applied_operations.append("sharpening")
            
            # 4. 각도 보정 (기울어진 사진 보정)
            processed_image, angle = self._correct_skew(processed_image)
            if abs(angle) > 0.5:  # 0.5도 이상 보정된 경우
                applied_operations.append(f"skew_correction_{angle:.1f}deg")
            
            # 5. 텍스트 영역 강조
            processed_image = self._enhance_text_regions(processed_image)
            applied_operations.append("text_region_enhancement")
            
            return processed_image, applied_operations
            
        except Exception as e:
            logger.warning(f"⚠️ 전처리 실패, 원본 사용: {e}")
            return image_array, ["preprocessing_failed"]
    
    def _auto_adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """자동 밝기/대비 조정"""
        # 히스토그램 기반 자동 조정
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 히스토그램 분석
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # 누적 분포 함수 계산
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        
        # 1%와 99% 지점 찾기
        low_percentile = np.searchsorted(cdf_normalized, 1)
        high_percentile = np.searchsorted(cdf_normalized, 99)
        
        if high_percentile > low_percentile:
            # 대비 스트레칭 적용
            alpha = 255.0 / (high_percentile - low_percentile)
            beta = -low_percentile * alpha
            
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            return adjusted
        
        return image
    
    def _normalize_projection_colors(self, image: np.ndarray) -> np.ndarray:
        """프로젝터 색왜곡 보정"""
        # LAB 색공간에서 색상 정규화
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # L 채널 (밝기) 히스토그램 평활화
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
        
        # A, B 채널 정규화
        a = cv2.normalize(a, None, alpha=127-30, beta=127+30, norm_type=cv2.NORM_MINMAX)
        b = cv2.normalize(b, None, alpha=127-30, beta=127+30, norm_type=cv2.NORM_MINMAX)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _correct_skew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """기울어진 이미지 자동 보정"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 가장자리 검출
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 허프 변환으로 직선 검출
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None and len(lines) > 0:
            # 각도 계산
            angles = []
            for line in lines[:20]:  # 상위 20개 직선만 사용
                rho, theta = line[0]
                angle = (theta - np.pi/2) * 180 / np.pi
                if -45 < angle < 45:  # 유효한 각도 범위
                    angles.append(angle)
            
            if angles:
                # 중앙값 사용 (노이즈에 강함)
                median_angle = np.median(angles)
                
                # 기울기 보정
                if abs(median_angle) > 0.5:
                    center = (image.shape[1]//2, image.shape[0]//2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    corrected = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),
                                             borderMode=cv2.BORDER_REPLICATE)
                    return corrected, median_angle
        
        return image, 0.0
    
    def _enhance_text_regions(self, image: np.ndarray) -> np.ndarray:
        """텍스트 영역 강조"""
        # 모폴로지 연산으로 텍스트 영역 강조
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 텍스트와 배경 분리를 위한 적응형 이진화
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 모폴로지 연산으로 글자 연결
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 원본 이미지와 결합
        if len(image.shape) == 3:
            enhanced = image.copy()
            # 텍스트 영역만 선명하게 
            mask = processed < 128
            enhanced[mask] = cv2.addWeighted(enhanced[mask], 0.7, 
                                           cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)[mask], 0.3, 0)
            return enhanced
        
        return processed

class EnhancedOCREngine:
    """강화된 다중 OCR 엔진"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.preprocessor = PPTImagePreprocessor()
        self.ocr_instances = {}
        self.cache = {}
        
        # 사용 가능한 OCR 엔진 초기화
        self._initialize_engines()
        logger.info(f"🔍 Enhanced OCR 엔진 초기화 완료 ({len(self.ocr_instances)}개 엔진)")
    
    def _get_default_config(self) -> Dict:
        """기본 설정"""
        return {
            'engines': {
                'easyocr': {'enabled': True, 'languages': ['ko', 'en'], 'gpu': False},
                'tesseract': {'enabled': True, 'config': '--psm 6'},
                'paddleocr': {'enabled': True, 'use_angle_cls': True, 'lang': 'korean'}
            },
            'preprocessing': {
                'enabled': True,
                'ppt_optimization': True,
                'max_image_size': 2048
            },
            'confidence': {
                'min_threshold': 0.7,  # 기존보다 높음
                'adaptive_threshold': True,
                'cross_validation': True
            },
            'performance': {
                'parallel_processing': True,
                'cache_results': True,
                'max_processing_time': 60
            }
        }
    
    def _initialize_engines(self):
        """OCR 엔진들 초기화"""
        # EasyOCR
        if OCR_ENGINES['easyocr'] and self.config['engines']['easyocr']['enabled']:
            try:
                self.ocr_instances['easyocr'] = easyocr.Reader(
                    self.config['engines']['easyocr']['languages'],
                    gpu=self.config['engines']['easyocr']['gpu']
                )
                logger.info("✅ EasyOCR 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ EasyOCR 초기화 실패: {e}")
        
        # Tesseract
        if OCR_ENGINES['tesseract'] and self.config['engines']['tesseract']['enabled']:
            try:
                # Tesseract 설치 확인
                pytesseract.get_tesseract_version()
                self.ocr_instances['tesseract'] = True
                logger.info("✅ Tesseract 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ Tesseract 초기화 실패: {e}")
        
        # PaddleOCR
        if OCR_ENGINES['paddleocr'] and self.config['engines']['paddleocr']['enabled']:
            try:
                self.ocr_instances['paddleocr'] = paddleocr.PaddleOCR(
                    use_angle_cls=self.config['engines']['paddleocr']['use_angle_cls'],
                    lang=self.config['engines']['paddleocr']['lang'],
                    show_log=False
                )
                logger.info("✅ PaddleOCR 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ PaddleOCR 초기화 실패: {e}")
        
        if not self.ocr_instances:
            raise RuntimeError("❌ 사용 가능한 OCR 엔진이 없습니다")
    
    def process_image(self, image_path: Union[str, Path], 
                     compare_engines: bool = False) -> EnhancedOCRResult:
        """이미지 OCR 처리 (다중 엔진)"""
        start_time = time.time()
        image_path = Path(image_path)
        
        # 캐시 확인
        cache_key = self._get_cache_key(image_path)
        if cache_key in self.cache and self.config['performance']['cache_results']:
            logger.info(f"📋 캐시에서 결과 로드: {image_path.name}")
            return self.cache[cache_key]
        
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지 로드 실패: {image_path}")
        
        # 전처리
        preprocessing_applied = []
        if self.config['preprocessing']['enabled']:
            image, preprocessing_applied = self.preprocessor.enhance_ppt_image(image)
        
        # 이미지 품질 점수 계산
        quality_score = self._calculate_image_quality(image)
        
        # 각 OCR 엔진으로 처리
        individual_results = []
        
        for engine_name in self.ocr_instances.keys():
            try:
                engine_start = time.time()
                result = self._process_with_engine(engine_name, image)
                engine_time = time.time() - engine_start
                
                individual_results.append(OCRResult(
                    engine=engine_name,
                    text=result['text'],
                    confidence=result['confidence'],
                    processing_time=engine_time,
                    bbox_count=result.get('bbox_count', 0),
                    metadata=result.get('metadata', {})
                ))
                
            except Exception as e:
                logger.warning(f"⚠️ {engine_name} OCR 실패: {e}")
                individual_results.append(OCRResult(
                    engine=engine_name,
                    text="",
                    confidence=0.0,
                    processing_time=0.0,
                    bbox_count=0,
                    metadata={'error': str(e)}
                ))
        
        # 최적 결과 선택
        best_result = self._select_best_result(individual_results)
        
        total_time = time.time() - start_time
        
        # 최종 결과 생성
        enhanced_result = EnhancedOCRResult(
            final_text=best_result.text,
            confidence=best_result.confidence,
            processing_time=total_time,
            engines_used=[r.engine for r in individual_results if r.confidence > 0],
            best_engine=best_result.engine,
            individual_results=individual_results,
            preprocessing_applied=preprocessing_applied,
            image_quality_score=quality_score
        )
        
        # 캐시 저장
        if self.config['performance']['cache_results']:
            self.cache[cache_key] = enhanced_result
        
        return enhanced_result
    
    def _process_with_engine(self, engine_name: str, image: np.ndarray) -> Dict:
        """특정 OCR 엔진으로 처리"""
        if engine_name == 'easyocr':
            return self._process_easyocr(image)
        elif engine_name == 'tesseract':
            return self._process_tesseract(image)
        elif engine_name == 'paddleocr':
            return self._process_paddleocr(image)
        else:
            raise ValueError(f"지원하지 않는 엔진: {engine_name}")
    
    def _process_easyocr(self, image: np.ndarray) -> Dict:
        """EasyOCR 처리"""
        results = self.ocr_instances['easyocr'].readtext(image)
        
        text_blocks = []
        confidences = []
        
        for detection in results:
            bbox, text, confidence = detection
            if confidence >= self.config['confidence']['min_threshold']:
                text_blocks.append(text.strip())
                confidences.append(confidence)
        
        full_text = ' '.join(text_blocks)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'text': full_text,
            'confidence': avg_confidence,
            'bbox_count': len(text_blocks),
            'metadata': {'engine': 'easyocr', 'raw_results': len(results)}
        }
    
    def _process_tesseract(self, image: np.ndarray) -> Dict:
        """Tesseract 처리"""
        # PIL Image로 변환
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # OCR 실행
        config = self.config['engines']['tesseract']['config']
        text = pytesseract.image_to_string(pil_image, config=config, lang='kor+eng')
        
        # 신뢰도 정보 가져오기
        data = pytesseract.image_to_data(pil_image, config=config, lang='kor+eng', output_type=pytesseract.Output.DICT)
        
        # 유효한 텍스트 블록의 신뢰도 계산
        confidences = []
        valid_text_count = 0
        
        for i, conf in enumerate(data['conf']):
            if int(conf) > 0 and data['text'][i].strip():
                confidences.append(int(conf) / 100.0)  # 0-1 범위로 정규화
                valid_text_count += 1
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'text': text.strip(),
            'confidence': avg_confidence,
            'bbox_count': valid_text_count,
            'metadata': {'engine': 'tesseract', 'total_boxes': len(data['text'])}
        }
    
    def _process_paddleocr(self, image: np.ndarray) -> Dict:
        """PaddleOCR 처리"""
        results = self.ocr_instances['paddleocr'].ocr(image, cls=True)
        
        if not results or not results[0]:
            return {'text': '', 'confidence': 0.0, 'bbox_count': 0, 'metadata': {'engine': 'paddleocr'}}
        
        text_blocks = []
        confidences = []
        
        for detection in results[0]:
            if len(detection) >= 2:
                bbox, (text, confidence) = detection[0], detection[1]
                if confidence >= self.config['confidence']['min_threshold']:
                    text_blocks.append(text.strip())
                    confidences.append(confidence)
        
        full_text = ' '.join(text_blocks)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'text': full_text,
            'confidence': avg_confidence,
            'bbox_count': len(text_blocks),
            'metadata': {'engine': 'paddleocr', 'total_detections': len(results[0])}
        }
    
    def _select_best_result(self, results: List[OCRResult]) -> OCRResult:
        """최적 OCR 결과 선택"""
        if not results:
            raise ValueError("OCR 결과가 없습니다")
        
        # 성공한 결과들만 필터링
        valid_results = [r for r in results if r.confidence > 0 and r.text.strip()]
        
        if not valid_results:
            # 모든 엔진이 실패한 경우, 가장 처리 시간이 짧은 것 반환
            return min(results, key=lambda x: x.processing_time)
        
        # 종합 점수 계산 (신뢰도 + 텍스트 길이 + 엔진 우선순위)
        engine_priority = {'paddleocr': 3, 'easyocr': 2, 'tesseract': 1}
        
        best_result = None
        best_score = -1
        
        for result in valid_results:
            # 점수 계산
            confidence_score = result.confidence
            length_score = min(len(result.text) / 100.0, 1.0)  # 텍스트 길이 (최대 100자 기준)
            priority_score = engine_priority.get(result.engine, 1) / 3.0
            
            total_score = (confidence_score * 0.6 + 
                          length_score * 0.3 + 
                          priority_score * 0.1)
            
            if total_score > best_score:
                best_score = total_score
                best_result = result
        
        return best_result or results[0]
    
    def _calculate_image_quality(self, image: np.ndarray) -> float:
        """이미지 품질 점수 계산 (0-1)"""
        try:
            # 그레이스케일 변환
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 선명도 측정 (라플라시안 분산)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(sharpness / 1000.0, 1.0)
            
            # 대비 측정
            contrast = gray.std()
            contrast_score = min(contrast / 128.0, 1.0)
            
            # 밝기 분포 측정
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            brightness_distribution = np.std(hist)
            brightness_score = min(brightness_distribution / 1000.0, 1.0)
            
            # 종합 점수
            quality_score = (sharpness_score * 0.5 + 
                           contrast_score * 0.3 + 
                           brightness_score * 0.2)
            
            return quality_score
            
        except Exception:
            return 0.5  # 기본값
    
    def _get_cache_key(self, image_path: Path) -> str:
        """캐시 키 생성"""
        # 파일 내용의 해시값 사용
        try:
            with open(image_path, 'rb') as f:
                file_content = f.read()
            return hashlib.md5(file_content).hexdigest()
        except:
            return str(image_path)

# 기존 시스템과 호환되는 인터페이스
def process_image_enhanced(image_path: Union[str, Path], 
                          fallback_function: Optional[callable] = None) -> Dict[str, Any]:
    """
    기존 시스템과 호환되는 강화된 OCR 처리 함수
    
    Args:
        image_path: 이미지 파일 경로
        fallback_function: 실패시 사용할 기존 OCR 함수
    
    Returns:
        기존 형식과 호환되는 결과 딕셔너리
    """
    try:
        # Enhanced OCR 엔진 사용
        ocr_engine = EnhancedOCREngine()
        result = ocr_engine.process_image(image_path)
        
        # 기존 형식으로 변환
        return {
            'text': result.final_text,
            'text_blocks': [{'text': result.final_text, 'confidence': result.confidence}],
            'total_blocks': 1,
            'average_confidence': result.confidence,
            'processing_time': result.processing_time,
            'enhanced': True,
            'engines_used': result.engines_used,
            'best_engine': result.best_engine,
            'image_quality': result.image_quality_score,
            'preprocessing': result.preprocessing_applied
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced OCR 실패: {e}")
        
        # 폴백 함수 사용
        if fallback_function:
            logger.info("🔄 기존 OCR 폴백 사용")
            try:
                fallback_result = fallback_function(image_path)
                if isinstance(fallback_result, dict):
                    fallback_result['enhanced'] = False
                    fallback_result['fallback_used'] = True
                    return fallback_result
            except Exception as fallback_error:
                logger.error(f"❌ 폴백 OCR도 실패: {fallback_error}")
        
        # 최종 실패
        return {
            'text': '',
            'text_blocks': [],
            'total_blocks': 0,
            'average_confidence': 0.0,
            'processing_time': 0.0,
            'enhanced': False,
            'error': str(e)
        }

if __name__ == "__main__":
    # Enhanced OCR 테스트
    try:
        ocr_engine = EnhancedOCREngine()
        print(f"✅ Enhanced OCR 엔진 초기화 성공 ({len(ocr_engine.ocr_instances)}개 엔진)")
        
        # 사용 가능한 엔진 출력
        for engine in ocr_engine.ocr_instances.keys():
            print(f"  🔍 {engine} 사용 가능")
        
        # 테스트 이미지가 있다면 처리
        test_image_path = Path("test_files/test.png")
        if test_image_path.exists():
            print(f"\n🧪 테스트 이미지 처리: {test_image_path}")
            result = ocr_engine.process_image(test_image_path)
            print(f"📝 추출된 텍스트: {result.final_text[:100]}...")
            print(f"🎯 신뢰도: {result.confidence:.2f}")
            print(f"⏱️ 처리 시간: {result.processing_time:.2f}초")
            print(f"🏆 최적 엔진: {result.best_engine}")
        
    except Exception as e:
        print(f"❌ Enhanced OCR 테스트 실패: {e}")
        import traceback
        traceback.print_exc()