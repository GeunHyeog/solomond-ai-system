"""
주얼리 AI 플랫폼 v2.1 - OCR 품질 검증 시스템
============================================

이미지/문서 OCR 품질 사전 평가, 실패 영역 감지, 개선 가이드 제공
PPT 슬라이드, GIA 인증서, 주얼리 카탈로그 특화 최적화

Author: 전근혁 (solomond.jgh@gmail.com)
Created: 2025.07.10
Version: 2.1.0
"""

import cv2
import numpy as np
import PIL
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import easyocr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings
import base64
import io
import json
from pathlib import Path
import tempfile
import math

warnings.filterwarnings('ignore')

# 이미지 처리 라이브러리
try:
    from skimage import measure, filters, morphology
    from skimage.feature import canny
    from skimage.transform import hough_line, hough_line_peaks
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import pdf2image
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRQualityValidator:
    """
    OCR 품질 검증 및 최적화 시스템
    
    주요 기능:
    - 이미지 품질 사전 평가 (해상도/명도/대비/기울기)
    - OCR 신뢰도 점수 예측
    - 텍스트 인식 실패 영역 감지
    - 이미지 전처리 자동 최적화
    - 재촬영 가이드 제공
    - PPT/인증서 특화 처리
    """
    
    def __init__(self, 
                 tesseract_config: str = '--oem 3 --psm 6',
                 easyocr_languages: List[str] = ['ko', 'en', 'ch_sim', 'ja']):
        """
        OCRQualityValidator 초기화
        
        Args:
            tesseract_config: Tesseract OCR 설정
            easyocr_languages: EasyOCR 지원 언어
        """
        self.tesseract_config = tesseract_config
        self.easyocr_languages = easyocr_languages
        
        # EasyOCR 리더 초기화
        try:
            self.easyocr_reader = easyocr.Reader(easyocr_languages, gpu=False)
            self.easyocr_available = True
        except Exception as e:
            logger.warning(f"EasyOCR 초기화 실패: {e}")
            self.easyocr_available = False
        
        # 품질 기준 설정
        self.quality_thresholds = {
            'resolution': {
                'excellent': (1920, 1080),  # Full HD 이상
                'good': (1280, 720),        # HD 이상  
                'fair': (1024, 768),        # XGA 이상
                'poor': (640, 480)          # VGA 이상
            },
            'sharpness': {
                'excellent': 100,
                'good': 50,
                'fair': 25,
                'poor': 10
            },
            'brightness': {
                'optimal_range': (120, 200),  # 0-255 스케일
                'acceptable_range': (80, 240)
            },
            'contrast': {
                'excellent': 80,
                'good': 50,
                'fair': 30,
                'poor': 15
            },
            'skew': {
                'excellent': 0.5,   # ±0.5도
                'good': 1.0,        # ±1.0도
                'fair': 2.0,        # ±2.0도
                'poor': 5.0         # ±5.0도
            }
        }
        
        # 문서 타입별 특화 설정
        self.document_profiles = {
            'ppt_slide': {
                'expected_elements': ['title', 'bullet_points', 'images'],
                'text_area_ratio': (0.3, 0.8),  # 텍스트 영역 비율
                'background_type': 'uniform',
                'font_size_range': (14, 48)
            },
            'certificate': {
                'expected_elements': ['header', 'serial_number', 'signatures', 'stamps'],
                'text_area_ratio': (0.6, 0.9),
                'background_type': 'formal',
                'font_size_range': (8, 24)
            },
            'catalog': {
                'expected_elements': ['product_images', 'descriptions', 'prices'],
                'text_area_ratio': (0.2, 0.6),
                'background_type': 'complex',
                'font_size_range': (6, 18)
            },
            'document': {
                'expected_elements': ['paragraphs', 'headers'],
                'text_area_ratio': (0.7, 0.95),
                'background_type': 'simple',
                'font_size_range': (10, 16)
            }
        }
        
        logger.info("OCRQualityValidator 초기화 완료")
    
    def validate_image_quality(self, image_path: str, document_type: str = 'auto') -> Dict:
        """
        이미지 품질 종합 검증
        
        Args:
            image_path: 검증할 이미지 파일 경로
            document_type: 문서 타입 ('ppt_slide', 'certificate', 'catalog', 'document', 'auto')
            
        Returns:
            Dict: 종합 품질 검증 결과
        """
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                return self._create_error_result(f"이미지 로드 실패: {image_path}")
            
            logger.info(f"이미지 품질 검증 시작: {image_path}")
            
            # 문서 타입 자동 감지
            if document_type == 'auto':
                document_type = self._detect_document_type(image)
            
            # 종합 품질 분석
            quality_result = self._comprehensive_quality_analysis(image, document_type)
            
            # OCR 신뢰도 예측
            ocr_prediction = self._predict_ocr_accuracy(image, quality_result)
            
            # 실패 영역 감지
            failed_regions = self._detect_failed_regions(image, quality_result)
            
            # 개선 제안 생성
            improvement_suggestions = self._generate_improvement_suggestions(
                quality_result, failed_regions, document_type
            )
            
            # 최종 결과 구성
            final_result = {
                'image_path': image_path,
                'document_type': document_type,
                'image_info': self._get_image_info(image),
                'quality_analysis': quality_result,
                'ocr_prediction': ocr_prediction,
                'failed_regions': failed_regions,
                'improvement_suggestions': improvement_suggestions,
                'overall_assessment': self._calculate_overall_assessment(quality_result, ocr_prediction),
                'processing_recommendations': self._get_processing_recommendations(quality_result)
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"이미지 품질 검증 오류: {str(e)}")
            return self._create_error_result(str(e))
    
    def _comprehensive_quality_analysis(self, image: np.ndarray, document_type: str) -> Dict:
        """이미지 품질 종합 분석"""
        analysis = {}
        
        # 1. 기본 이미지 속성
        analysis['basic_properties'] = self._analyze_basic_properties(image)
        
        # 2. 해상도 분석
        analysis['resolution_analysis'] = self._analyze_resolution(image)
        
        # 3. 선명도 분석
        analysis['sharpness_analysis'] = self._analyze_sharpness(image)
        
        # 4. 밝기/대비 분석
        analysis['brightness_contrast'] = self._analyze_brightness_contrast(image)
        
        # 5. 기울기 감지
        analysis['skew_analysis'] = self._analyze_skew(image)
        
        # 6. 텍스트 영역 감지
        analysis['text_regions'] = self._detect_text_regions(image)
        
        # 7. 배경 분석
        analysis['background_analysis'] = self._analyze_background(image)
        
        # 8. 문서 타입별 특화 분석
        analysis['document_specific'] = self._analyze_document_specific(image, document_type)
        
        return analysis
    
    def _analyze_basic_properties(self, image: np.ndarray) -> Dict:
        """기본 이미지 속성 분석"""
        height, width = image.shape[:2]
        
        return {
            'width': int(width),
            'height': int(height),
            'aspect_ratio': float(width / height),
            'total_pixels': int(width * height),
            'channels': int(image.shape[2] if len(image.shape) == 3 else 1),
            'color_space': 'BGR' if len(image.shape) == 3 else 'Grayscale'
        }
    
    def _analyze_resolution(self, image: np.ndarray) -> Dict:
        """해상도 품질 분석"""
        height, width = image.shape[:2]
        total_pixels = width * height
        
        # 해상도 등급 결정
        resolution_grade = 'poor'
        for grade, (min_w, min_h) in self.quality_thresholds['resolution'].items():
            if width >= min_w and height >= min_h:
                resolution_grade = grade
                break
        
        # DPI 추정 (가정: 표준 모니터 크기 기준)
        estimated_dpi = max(width, height) / 10  # 간단한 추정
        
        return {
            'width': width,
            'height': height,
            'total_pixels': total_pixels,
            'megapixels': total_pixels / 1_000_000,
            'resolution_grade': resolution_grade,
            'estimated_dpi': float(estimated_dpi),
            'sufficient_for_ocr': total_pixels >= 786432,  # 1024x768 이상
            'ocr_quality_factor': min(total_pixels / 786432, 2.0)  # 최대 2배 가중치
        }
    
    def _analyze_sharpness(self, image: np.ndarray) -> Dict:
        """이미지 선명도 분석"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 라플라시안 변수 (선명도 지표)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 소벨 필터 기반 엣지 강도
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_density = np.mean(sobel_magnitude)
        
        # 텍스트 영역의 선명도 (고주파 성분)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        high_freq_energy = np.mean(magnitude_spectrum[gray.shape[0]//4:-gray.shape[0]//4, 
                                                     gray.shape[1]//4:-gray.shape[1]//4])
        
        # 선명도 등급 결정
        sharpness_grade = 'poor'
        for grade, threshold in self.quality_thresholds['sharpness'].items():
            if laplacian_var >= threshold:
                sharpness_grade = grade
                break
        
        return {
            'laplacian_variance': float(laplacian_var),
            'edge_density': float(edge_density),
            'high_freq_energy': float(high_freq_energy),
            'sharpness_score': float(min(laplacian_var / 100 * 100, 100)),  # 0-100 정규화
            'sharpness_grade': sharpness_grade,
            'is_sharp_enough': laplacian_var >= self.quality_thresholds['sharpness']['fair']
        }
    
    def _analyze_brightness_contrast(self, image: np.ndarray) -> Dict:
        """밝기 및 대비 분석"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 밝기 통계
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # 히스토그램 분석
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist.flatten() / hist.sum()
        
        # 대비 측정 (RMS 대비)
        rms_contrast = np.sqrt(np.mean((gray - mean_brightness) ** 2))
        
        # 다이나믹 레인지
        dynamic_range = np.max(gray) - np.min(gray)
        
        # 밝기 분포 분석
        dark_pixels = np.sum(gray < 85) / gray.size  # 어두운 픽셀 비율
        bright_pixels = np.sum(gray > 170) / gray.size  # 밝은 픽셀 비율
        mid_pixels = 1 - dark_pixels - bright_pixels  # 중간 톤 비율
        
        # 품질 평가
        brightness_optimal = (
            self.quality_thresholds['brightness']['optimal_range'][0] <= 
            mean_brightness <= 
            self.quality_thresholds['brightness']['optimal_range'][1]
        )
        
        contrast_grade = 'poor'
        for grade, threshold in self.quality_thresholds['contrast'].items():
            if rms_contrast >= threshold:
                contrast_grade = grade
                break
        
        return {
            'mean_brightness': float(mean_brightness),
            'brightness_std': float(brightness_std),
            'rms_contrast': float(rms_contrast),
            'dynamic_range': float(dynamic_range),
            'dark_pixel_ratio': float(dark_pixels),
            'bright_pixel_ratio': float(bright_pixels),
            'mid_tone_ratio': float(mid_pixels),
            'brightness_optimal': brightness_optimal,
            'contrast_grade': contrast_grade,
            'brightness_score': float(100 - abs(mean_brightness - 150) / 150 * 100),  # 150이 최적
            'contrast_score': float(min(rms_contrast / 80 * 100, 100))  # 0-100 정규화
        }
    
    def _analyze_skew(self, image: np.ndarray) -> Dict:
        """이미지 기울기 감지 및 분석"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 엣지 감지
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 허프 변환으로 직선 감지
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:20]:  # 상위 20개 직선만 분석
                angle = theta * 180 / np.pi
                # 수직선과 수평선 구분
                if 45 <= angle <= 135:  # 수직선 근처
                    angles.append(90 - angle)
                else:  # 수평선 근처
                    angles.append(angle if angle <= 90 else angle - 180)
            
            if angles:
                # 각도 클러스터링 (간단한 방법)
                angles = np.array(angles)
                median_angle = np.median(angles)
                skew_angle = median_angle
            else:
                skew_angle = 0.0
        else:
            skew_angle = 0.0
        
        # 기울기 등급 결정
        abs_skew = abs(skew_angle)
        skew_grade = 'poor'
        for grade, threshold in self.quality_thresholds['skew'].items():
            if abs_skew <= threshold:
                skew_grade = grade
                break
        
        return {
            'skew_angle': float(skew_angle),
            'skew_magnitude': float(abs_skew),
            'skew_grade': skew_grade,
            'correction_needed': abs_skew > 1.0,
            'skew_score': float(max(0, 100 - abs_skew * 20))  # 0도=100점, 5도=0점
        }
    
    def _detect_text_regions(self, image: np.ndarray) -> Dict:
        """텍스트 영역 감지 및 분석"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 텍스트 영역 감지 (MSER 알고리즘)
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        # 텍스트 후보 영역 필터링
        text_candidates = []
        total_area = gray.shape[0] * gray.shape[1]
        
        for region in regions:
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # 텍스트 영역 조건 필터링
            if (10 <= w <= gray.shape[1] * 0.8 and  # 너비 조건
                5 <= h <= gray.shape[0] * 0.3 and   # 높이 조건
                0.1 <= aspect_ratio <= 20 and       # 종횡비 조건
                area >= 50):                        # 최소 면적
                text_candidates.append((x, y, w, h))
        
        # 텍스트 영역 통계
        if text_candidates:
            total_text_area = sum(w * h for x, y, w, h in text_candidates)
            text_coverage = total_text_area / total_area
            avg_text_height = np.mean([h for x, y, w, h in text_candidates])
            text_density = len(text_candidates) / total_area * 1000000  # per million pixels
        else:
            text_coverage = 0
            avg_text_height = 0
            text_density = 0
        
        return {
            'text_regions_count': len(text_candidates),
            'text_coverage_ratio': float(text_coverage),
            'average_text_height': float(avg_text_height),
            'text_density': float(text_density),
            'text_regions': text_candidates,
            'has_sufficient_text': text_coverage > 0.05  # 5% 이상
        }
    
    def _analyze_background(self, image: np.ndarray) -> Dict:
        """배경 복잡도 및 특성 분석"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 배경 복잡도 측정
        # 1. 표준편차 기반 복잡도
        background_complexity = np.std(gray)
        
        # 2. 엣지 밀도
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 3. 텍스처 분석 (Local Binary Pattern 유사)
        kernel = np.ones((3,3), np.uint8)
        local_variance = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        texture_complexity = np.std(local_variance)
        
        # 4. 색상 다양성 (컬러 이미지의 경우)
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            color_variance = np.std(hsv[:,:,1])  # 채도 분산
        else:
            color_variance = 0
        
        # 배경 타입 분류
        if background_complexity < 20 and edge_density < 0.1:
            background_type = 'uniform'
        elif background_complexity < 40 and edge_density < 0.2:
            background_type = 'simple'
        elif background_complexity < 60 and edge_density < 0.3:
            background_type = 'moderate'
        else:
            background_type = 'complex'
        
        return {
            'background_complexity': float(background_complexity),
            'edge_density': float(edge_density),
            'texture_complexity': float(texture_complexity),
            'color_variance': float(color_variance),
            'background_type': background_type,
            'ocr_friendly': background_complexity < 40 and edge_density < 0.2
        }
    
    def _analyze_document_specific(self, image: np.ndarray, document_type: str) -> Dict:
        """문서 타입별 특화 분석"""
        if document_type not in self.document_profiles:
            return {'analysis': 'generic'}
        
        profile = self.document_profiles[document_type]
        
        if document_type == 'ppt_slide':
            return self._analyze_ppt_slide(image, profile)
        elif document_type == 'certificate':
            return self._analyze_certificate(image, profile)
        elif document_type == 'catalog':
            return self._analyze_catalog(image, profile)
        else:
            return self._analyze_generic_document(image, profile)
    
    def _analyze_ppt_slide(self, image: np.ndarray, profile: Dict) -> Dict:
        """PPT 슬라이드 특화 분석"""
        # 제목 영역 감지 (상단 1/4 영역)
        height = image.shape[0]
        title_region = image[:height//4, :]
        
        # 불릿 포인트 감지
        bullet_patterns = self._detect_bullet_patterns(image)
        
        # 폰트 크기 추정
        estimated_font_sizes = self._estimate_font_sizes(image)
        
        return {
            'document_type': 'ppt_slide',
            'title_region_quality': self._assess_region_quality(title_region),
            'bullet_points_detected': len(bullet_patterns),
            'estimated_font_sizes': estimated_font_sizes,
            'slide_layout_score': self._calculate_slide_layout_score(image),
            'readability_score': self._calculate_readability_score(image)
        }
    
    def _analyze_certificate(self, image: np.ndarray, profile: Dict) -> Dict:
        """인증서 특화 분석"""
        # 헤더/푸터 영역 분석
        height = image.shape[0]
        header_region = image[:height//6, :]
        footer_region = image[-height//6:, :]
        
        # 시리얼 넘버 패턴 감지
        serial_patterns = self._detect_serial_patterns(image)
        
        # 스탬프/서명 영역 감지
        stamp_regions = self._detect_stamp_regions(image)
        
        return {
            'document_type': 'certificate',
            'header_quality': self._assess_region_quality(header_region),
            'footer_quality': self._assess_region_quality(footer_region),
            'serial_patterns_found': len(serial_patterns),
            'stamp_regions_detected': len(stamp_regions),
            'formal_layout_score': self._calculate_formal_layout_score(image)
        }
    
    def _predict_ocr_accuracy(self, image: np.ndarray, quality_analysis: Dict) -> Dict:
        """OCR 정확도 사전 예측"""
        # 주요 품질 지표 추출
        resolution_score = quality_analysis['resolution_analysis']['ocr_quality_factor']
        sharpness_score = quality_analysis['sharpness_analysis']['sharpness_score']
        brightness_score = quality_analysis['brightness_contrast']['brightness_score']
        contrast_score = quality_analysis['brightness_contrast']['contrast_score']
        skew_score = quality_analysis['skew_analysis']['skew_score']
        text_coverage = quality_analysis['text_regions']['text_coverage_ratio']
        background_friendly = quality_analysis['background_analysis']['ocr_friendly']
        
        # 가중치 적용 예측 모델
        weights = {
            'resolution': 0.15,
            'sharpness': 0.25,
            'brightness': 0.15,
            'contrast': 0.20,
            'skew': 0.15,
            'text_coverage': 0.05,
            'background': 0.05
        }
        
        # 기본 정확도 계산
        base_accuracy = (
            resolution_score * weights['resolution'] +
            sharpness_score * weights['sharpness'] +
            brightness_score * weights['brightness'] +
            contrast_score * weights['contrast'] +
            skew_score * weights['skew'] +
            min(text_coverage * 1000, 100) * weights['text_coverage'] +
            (100 if background_friendly else 50) * weights['background']
        )
        
        # 문서 타입별 보정
        doc_specific = quality_analysis.get('document_specific', {})
        doc_type = doc_specific.get('document_type', 'generic')
        
        type_multipliers = {
            'ppt_slide': 1.1,      # PPT는 일반적으로 OCR 친화적
            'certificate': 0.95,   # 인증서는 복잡한 레이아웃
            'catalog': 0.85,       # 카탈로그는 이미지가 많음
            'document': 1.0        # 일반 문서
        }
        
        predicted_accuracy = base_accuracy * type_multipliers.get(doc_type, 1.0)
        predicted_accuracy = np.clip(predicted_accuracy, 10, 95)
        
        # 신뢰도 계산
        confidence_factors = [
            sharpness_score > 70,
            contrast_score > 50,
            skew_score > 80,
            background_friendly,
            text_coverage > 0.1
        ]
        
        confidence = sum(confidence_factors) / len(confidence_factors)
        
        return {
            'predicted_accuracy': float(predicted_accuracy),
            'confidence': float(confidence),
            'accuracy_grade': self._grade_predicted_accuracy(predicted_accuracy),
            'contributing_factors': {
                'resolution_contribution': float(resolution_score * weights['resolution']),
                'sharpness_contribution': float(sharpness_score * weights['sharpness']),
                'brightness_contribution': float(brightness_score * weights['brightness']),
                'contrast_contribution': float(contrast_score * weights['contrast']),
                'skew_contribution': float(skew_score * weights['skew'])
            },
            'limiting_factors': self._identify_limiting_factors(quality_analysis)
        }
    
    def _detect_failed_regions(self, image: np.ndarray, quality_analysis: Dict) -> List[Dict]:
        """OCR 실패 가능성이 높은 영역 감지"""
        failed_regions = []
        
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        height, width = gray.shape
        
        # 1. 극도로 어둡거나 밝은 영역
        too_dark = gray < 30
        too_bright = gray > 225
        
        # 연결된 컴포넌트 찾기
        dark_contours, _ = cv2.findContours(
            too_dark.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bright_contours, _ = cv2.findContours(
            too_bright.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 큰 영역만 선택
        for contours, reason in [(dark_contours, 'too_dark'), (bright_contours, 'too_bright')]:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # 최소 면적 임계값
                    x, y, w, h = cv2.boundingRect(contour)
                    failed_regions.append({
                        'type': 'brightness_issue',
                        'reason': reason,
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'area': float(area),
                        'severity': 'high' if area > 2000 else 'medium'
                    })
        
        # 2. 심하게 기울어진 텍스트 영역
        if abs(quality_analysis['skew_analysis']['skew_angle']) > 3:
            failed_regions.append({
                'type': 'skew_issue',
                'reason': 'excessive_skew',
                'bbox': [0, 0, width, height],  # 전체 이미지
                'skew_angle': quality_analysis['skew_analysis']['skew_angle'],
                'severity': 'high'
            })
        
        # 3. 흐릿한 영역 (지역별 선명도 분석)
        blurry_regions = self._detect_blurry_regions(gray)
        failed_regions.extend(blurry_regions)
        
        # 4. 배경과 텍스트 대비가 낮은 영역
        low_contrast_regions = self._detect_low_contrast_regions(gray)
        failed_regions.extend(low_contrast_regions)
        
        return failed_regions
    
    def _detect_blurry_regions(self, gray: np.ndarray) -> List[Dict]:
        """흐릿한 영역 감지"""
        blurry_regions = []
        
        # 이미지를 타일로 분할
        tile_size = 100
        height, width = gray.shape
        
        for y in range(0, height - tile_size, tile_size // 2):
            for x in range(0, width - tile_size, tile_size // 2):
                tile = gray[y:y+tile_size, x:x+tile_size]
                
                # 라플라시안 변수로 선명도 측정
                laplacian_var = cv2.Laplacian(tile, cv2.CV_64F).var()
                
                if laplacian_var < 10:  # 임계값 이하면 흐릿함
                    blurry_regions.append({
                        'type': 'blur_issue',
                        'reason': 'local_blur',
                        'bbox': [x, y, tile_size, tile_size],
                        'blur_score': float(laplacian_var),
                        'severity': 'high' if laplacian_var < 5 else 'medium'
                    })
        
        return blurry_regions
    
    def _detect_low_contrast_regions(self, gray: np.ndarray) -> List[Dict]:
        """낮은 대비 영역 감지"""
        low_contrast_regions = []
        
        # 이미지를 타일로 분할
        tile_size = 80
        height, width = gray.shape
        
        for y in range(0, height - tile_size, tile_size // 2):
            for x in range(0, width - tile_size, tile_size // 2):
                tile = gray[y:y+tile_size, x:x+tile_size]
                
                # RMS 대비 측정
                mean_val = np.mean(tile)
                rms_contrast = np.sqrt(np.mean((tile - mean_val) ** 2))
                
                if rms_contrast < 15:  # 낮은 대비
                    low_contrast_regions.append({
                        'type': 'contrast_issue',
                        'reason': 'low_contrast',
                        'bbox': [x, y, tile_size, tile_size],
                        'contrast_score': float(rms_contrast),
                        'severity': 'high' if rms_contrast < 8 else 'medium'
                    })
        
        return low_contrast_regions
    
    def _generate_improvement_suggestions(self, 
                                          quality_analysis: Dict, 
                                          failed_regions: List[Dict], 
                                          document_type: str) -> List[Dict]:
        """품질 개선 제안 생성"""
        suggestions = []
        
        # 해상도 개선 제안
        resolution = quality_analysis['resolution_analysis']
        if resolution['resolution_grade'] in ['poor', 'fair']:
            suggestions.append({
                'category': 'resolution',
                'priority': 'high',
                'issue': f"해상도가 낮습니다 ({resolution['width']}x{resolution['height']})",
                'suggestion': "더 높은 해상도로 촬영하거나 스캔해주세요 (최소 1024x768 권장)",
                'expected_improvement': "OCR 정확도 15-25% 향상"
            })
        
        # 선명도 개선 제안
        sharpness = quality_analysis['sharpness_analysis']
        if not sharpness['is_sharp_enough']:
            suggestions.append({
                'category': 'sharpness',
                'priority': 'high',
                'issue': f"이미지가 흐릿합니다 (선명도: {sharpness['sharpness_score']:.1f})",
                'suggestion': "카메라 초점을 맞추고 손떨림을 방지해주세요",
                'expected_improvement': "텍스트 인식률 20-30% 향상"
            })
        
        # 밝기/대비 개선 제안
        brightness = quality_analysis['brightness_contrast']
        if not brightness['brightness_optimal']:
            suggestions.append({
                'category': 'lighting',
                'priority': 'medium',
                'issue': f"밝기가 적절하지 않습니다 (현재: {brightness['mean_brightness']:.0f})",
                'suggestion': "조명을 개선하거나 노출을 조정해주세요 (권장: 120-200)",
                'expected_improvement': "가독성 10-15% 향상"
            })
        
        # 기울기 보정 제안
        skew = quality_analysis['skew_analysis']
        if skew['correction_needed']:
            suggestions.append({
                'category': 'orientation',
                'priority': 'medium',
                'issue': f"이미지가 {skew['skew_angle']:.1f}도 기울어져 있습니다",
                'suggestion': "문서를 수직으로 맞춰서 다시 촬영하거나 자동 회전을 적용하세요",
                'expected_improvement': "레이아웃 인식률 15-20% 향상"
            })
        
        # 배경 개선 제안
        background = quality_analysis['background_analysis']
        if not background['ocr_friendly']:
            suggestions.append({
                'category': 'background',
                'priority': 'medium',
                'issue': "배경이 복잡하여 텍스트 인식이 어려울 수 있습니다",
                'suggestion': "단순한 배경에서 촬영하거나 스캐너 사용을 권장합니다",
                'expected_improvement': "전체 정확도 10-20% 향상"
            })
        
        # 실패 영역별 구체적 제안
        if failed_regions:
            high_severity_regions = [r for r in failed_regions if r.get('severity') == 'high']
            if high_severity_regions:
                suggestions.append({
                    'category': 'failed_regions',
                    'priority': 'high',
                    'issue': f"{len(high_severity_regions)}개의 심각한 문제 영역이 발견되었습니다",
                    'suggestion': "문제 영역을 피하거나 조명/각도를 조정해주세요",
                    'expected_improvement': "해당 영역 인식률 대폭 개선"
                })
        
        # 문서 타입별 특화 제안
        if document_type == 'ppt_slide':
            suggestions.append({
                'category': 'ppt_optimization',
                'priority': 'low',
                'issue': "PPT 슬라이드 최적화",
                'suggestion': "제목과 본문을 명확히 구분하고, 폰트 크기를 충분히 크게 설정하세요",
                'expected_improvement': "구조 인식률 향상"
            })
        
        return suggestions
    
    def _calculate_overall_assessment(self, quality_analysis: Dict, ocr_prediction: Dict) -> Dict:
        """종합 품질 평가"""
        # 주요 지표들 가중 평균
        scores = {
            'resolution': quality_analysis['resolution_analysis']['ocr_quality_factor'] * 50,
            'sharpness': quality_analysis['sharpness_analysis']['sharpness_score'],
            'brightness': quality_analysis['brightness_contrast']['brightness_score'],
            'contrast': quality_analysis['brightness_contrast']['contrast_score'],
            'skew': quality_analysis['skew_analysis']['skew_score']
        }
        
        weights = {
            'resolution': 0.15,
            'sharpness': 0.30,
            'brightness': 0.20,
            'contrast': 0.20,
            'skew': 0.15
        }
        
        overall_score = sum(scores[key] * weights[key] for key in weights.keys())
        overall_score = np.clip(overall_score, 0, 100)
        
        # 등급 결정
        if overall_score >= 85:
            grade = 'excellent'
            description = '최적의 OCR 품질 - 높은 정확도 보장'
        elif overall_score >= 70:
            grade = 'good'
            description = '양호한 품질 - 대부분의 텍스트 인식 가능'
        elif overall_score >= 55:
            grade = 'fair'
            description = '보통 품질 - 일부 개선 필요'
        else:
            grade = 'poor'
            description = '품질 개선 필요 - 재촬영 권장'
        
        return {
            'overall_score': float(overall_score),
            'grade': grade,
            'description': description,
            'predicted_ocr_accuracy': ocr_prediction['predicted_accuracy'],
            'component_scores': scores,
            'recommendation': self._get_overall_recommendation(overall_score, ocr_prediction)
        }
    
    def _get_processing_recommendations(self, quality_analysis: Dict) -> Dict:
        """이미지 전처리 권장사항"""
        recommendations = {
            'preprocessing_steps': [],
            'parameter_suggestions': {},
            'automated_fixes': []
        }
        
        # 기울기 보정
        skew_angle = quality_analysis['skew_analysis']['skew_angle']
        if abs(skew_angle) > 0.5:
            recommendations['preprocessing_steps'].append('deskew')
            recommendations['parameter_suggestions']['rotation_angle'] = -skew_angle
            recommendations['automated_fixes'].append(f"{skew_angle:.1f}도 회전 보정")
        
        # 밝기/대비 조정
        brightness = quality_analysis['brightness_contrast']
        if not brightness['brightness_optimal']:
            recommendations['preprocessing_steps'].append('brightness_adjustment')
            target_brightness = 150
            current_brightness = brightness['mean_brightness']
            brightness_delta = target_brightness - current_brightness
            recommendations['parameter_suggestions']['brightness_delta'] = brightness_delta
            recommendations['automated_fixes'].append(f"밝기 {brightness_delta:+.0f} 조정")
        
        if brightness['contrast_grade'] in ['poor', 'fair']:
            recommendations['preprocessing_steps'].append('contrast_enhancement')
            recommendations['parameter_suggestions']['contrast_factor'] = 1.3
            recommendations['automated_fixes'].append("대비 30% 향상")
        
        # 노이즈 제거
        if not quality_analysis['sharpness_analysis']['is_sharp_enough']:
            recommendations['preprocessing_steps'].append('noise_reduction')
            recommendations['automated_fixes'].append("노이즈 제거 필터 적용")
        
        # 해상도 보간
        resolution = quality_analysis['resolution_analysis']
        if resolution['resolution_grade'] == 'poor':
            recommendations['preprocessing_steps'].append('upscaling')
            recommendations['parameter_suggestions']['scale_factor'] = 2.0
            recommendations['automated_fixes'].append("2배 해상도 업스케일링")
        
        return recommendations
    
    # 문서 타입 감지 및 특화 분석 헬퍼 함수들
    def _detect_document_type(self, image: np.ndarray) -> str:
        """문서 타입 자동 감지"""
        # 간단한 휴리스틱 기반 분류
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # PPT 슬라이드 (일반적으로 16:9 또는 4:3)
        if 1.2 <= aspect_ratio <= 1.8:
            return 'ppt_slide'
        
        # 인증서 (일반적으로 A4 세로)
        elif 0.7 <= aspect_ratio <= 0.8:
            return 'certificate'
        
        # 정사각형에 가까우면 카탈로그일 가능성
        elif 0.8 <= aspect_ratio <= 1.2:
            return 'catalog'
        
        else:
            return 'document'
    
    def _detect_bullet_patterns(self, image: np.ndarray) -> List[Tuple]:
        """불릿 포인트 패턴 감지"""
        # 원형 또는 점 형태의 패턴 감지
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 허프 원 변환으로 원형 불릿 감지
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=20, minRadius=2, maxRadius=10
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return [(x, y, r) for x, y, r in circles]
        
        return []
    
    def _estimate_font_sizes(self, image: np.ndarray) -> List[int]:
        """폰트 크기 추정"""
        # 텍스트 영역의 높이를 기반으로 폰트 크기 추정
        text_regions = self._detect_text_regions(image)['text_regions']
        
        font_sizes = []
        for x, y, w, h in text_regions:
            # 높이를 픽셀에서 포인트로 변환 (대략적)
            estimated_font_size = int(h * 0.75)  # 경험적 변환
            if 6 <= estimated_font_size <= 72:  # 합리적 범위
                font_sizes.append(estimated_font_size)
        
        return font_sizes
    
    def _detect_serial_patterns(self, image: np.ndarray) -> List[str]:
        """시리얼 번호 패턴 감지"""
        # 간단한 구현 - 실제로는 OCR 결과를 정규식으로 분석
        return []  # 향후 구현
    
    def _detect_stamp_regions(self, image: np.ndarray) -> List[Tuple]:
        """스탬프/서명 영역 감지"""
        # 원형 또는 특이한 형태의 영역 감지
        return []  # 향후 구현
    
    # 품질 평가 헬퍼 함수들
    def _assess_region_quality(self, region: np.ndarray) -> Dict:
        """특정 영역의 품질 평가"""
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        return {
            'mean_brightness': float(np.mean(gray)),
            'contrast': float(np.std(gray)),
            'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var())
        }
    
    def _calculate_slide_layout_score(self, image: np.ndarray) -> float:
        """슬라이드 레이아웃 점수 계산"""
        # 텍스트 영역의 분포와 정렬 상태 평가
        text_regions = self._detect_text_regions(image)['text_regions']
        
        if not text_regions:
            return 0.0
        
        # 텍스트 영역들의 정렬 상태 확인
        left_edges = [x for x, y, w, h in text_regions]
        alignment_score = 100 - (np.std(left_edges) / np.mean(left_edges) * 100 if left_edges else 0)
        
        return np.clip(alignment_score, 0, 100)
    
    def _calculate_readability_score(self, image: np.ndarray) -> float:
        """가독성 점수 계산"""
        # 대비, 선명도, 폰트 크기를 종합한 가독성 평가
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        contrast = np.std(gray)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        readability = (contrast / 100 * 50) + (min(sharpness / 100, 1) * 50)
        return np.clip(readability, 0, 100)
    
    def _calculate_formal_layout_score(self, image: np.ndarray) -> float:
        """공식 문서 레이아웃 점수 계산"""
        # 인증서 등의 공식 문서 레이아웃 평가
        return 75.0  # 기본값 (향후 고도화)
    
    def _grade_predicted_accuracy(self, accuracy: float) -> str:
        """예측 정확도 등급"""
        if accuracy >= 90:
            return 'excellent'
        elif accuracy >= 80:
            return 'good'
        elif accuracy >= 70:
            return 'fair'
        else:
            return 'poor'
    
    def _identify_limiting_factors(self, quality_analysis: Dict) -> List[str]:
        """품질을 제한하는 주요 요인 식별"""
        limiting_factors = []
        
        if not quality_analysis['sharpness_analysis']['is_sharp_enough']:
            limiting_factors.append('흐릿한 이미지')
        
        if not quality_analysis['brightness_contrast']['brightness_optimal']:
            limiting_factors.append('부적절한 밝기')
        
        if quality_analysis['brightness_contrast']['contrast_grade'] == 'poor':
            limiting_factors.append('낮은 대비')
        
        if quality_analysis['skew_analysis']['correction_needed']:
            limiting_factors.append('이미지 기울기')
        
        if not quality_analysis['background_analysis']['ocr_friendly']:
            limiting_factors.append('복잡한 배경')
        
        return limiting_factors
    
    def _get_overall_recommendation(self, overall_score: float, ocr_prediction: Dict) -> str:
        """종합 권장사항"""
        if overall_score >= 85:
            return "현재 품질이 우수합니다. 바로 OCR 처리를 진행하세요."
        elif overall_score >= 70:
            return "양호한 품질입니다. 필요시 간단한 전처리 후 OCR을 진행하세요."
        elif overall_score >= 55:
            return "몇 가지 개선이 필요합니다. 제안된 개선사항을 적용한 후 OCR을 진행하세요."
        else:
            return "품질이 낮습니다. 재촬영하거나 대폭적인 개선이 필요합니다."
    
    def _get_image_info(self, image: np.ndarray) -> Dict:
        """기본 이미지 정보"""
        return {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'size_mb': image.nbytes / (1024 * 1024)
        }
    
    def _create_error_result(self, error_message: str) -> Dict:
        """오류 결과 생성"""
        return {
            'error': True,
            'error_message': error_message,
            'overall_assessment': {
                'overall_score': 0,
                'grade': 'error',
                'description': '분석 실패'
            }
        }
    
    # 이미지 전처리 및 최적화 함수들
    def optimize_image_for_ocr(self, image_path: str, output_path: str = None) -> Dict:
        """
        OCR을 위한 이미지 자동 최적화
        
        Args:
            image_path: 원본 이미지 경로
            output_path: 최적화된 이미지 저장 경로 (None이면 자동 생성)
            
        Returns:
            Dict: 최적화 결과
        """
        try:
            # 품질 분석
            quality_result = self.validate_image_quality(image_path)
            
            if quality_result.get('error'):
                return quality_result
            
            # 이미지 로드
            image = cv2.imread(image_path)
            optimized_image = image.copy()
            
            # 전처리 단계들
            processing_steps = quality_result['processing_recommendations']['preprocessing_steps']
            applied_optimizations = []
            
            # 기울기 보정
            if 'deskew' in processing_steps:
                angle = quality_result['processing_recommendations']['parameter_suggestions']['rotation_angle']
                optimized_image = self._rotate_image(optimized_image, angle)
                applied_optimizations.append(f"기울기 보정: {angle:.1f}도")
            
            # 밝기 조정
            if 'brightness_adjustment' in processing_steps:
                delta = quality_result['processing_recommendations']['parameter_suggestions']['brightness_delta']
                optimized_image = self._adjust_brightness(optimized_image, delta)
                applied_optimizations.append(f"밝기 조정: {delta:+.0f}")
            
            # 대비 향상
            if 'contrast_enhancement' in processing_steps:
                factor = quality_result['processing_recommendations']['parameter_suggestions']['contrast_factor']
                optimized_image = self._enhance_contrast(optimized_image, factor)
                applied_optimizations.append(f"대비 향상: {factor:.1f}배")
            
            # 노이즈 제거
            if 'noise_reduction' in processing_steps:
                optimized_image = self._reduce_noise(optimized_image)
                applied_optimizations.append("노이즈 제거")
            
            # 해상도 향상
            if 'upscaling' in processing_steps:
                scale = quality_result['processing_recommendations']['parameter_suggestions']['scale_factor']
                optimized_image = self._upscale_image(optimized_image, scale)
                applied_optimizations.append(f"해상도 향상: {scale}배")
            
            # 최적화된 이미지 저장
            if output_path is None:
                output_path = image_path.replace('.', '_optimized.')
            
            cv2.imwrite(output_path, optimized_image)
            
            # 최적화 후 품질 재평가
            optimized_quality = self.validate_image_quality(output_path)
            
            return {
                'success': True,
                'original_path': image_path,
                'optimized_path': output_path,
                'applied_optimizations': applied_optimizations,
                'quality_improvement': {
                    'before': quality_result['overall_assessment']['overall_score'],
                    'after': optimized_quality['overall_assessment']['overall_score'],
                    'improvement': optimized_quality['overall_assessment']['overall_score'] - 
                                   quality_result['overall_assessment']['overall_score']
                },
                'ocr_accuracy_improvement': {
                    'before': quality_result['ocr_prediction']['predicted_accuracy'],
                    'after': optimized_quality['ocr_prediction']['predicted_accuracy'],
                    'improvement': optimized_quality['ocr_prediction']['predicted_accuracy'] - 
                                   quality_result['ocr_prediction']['predicted_accuracy']
                }
            }
            
        except Exception as e:
            logger.error(f"이미지 최적화 오류: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """이미지 회전"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def _adjust_brightness(self, image: np.ndarray, delta: float) -> np.ndarray:
        """밝기 조정"""
        adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=delta)
        return adjusted
    
    def _enhance_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """대비 향상"""
        enhanced = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return enhanced
    
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """노이즈 제거"""
        # 바이래터럴 필터 적용
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised
    
    def _upscale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """이미지 해상도 향상"""
        height, width = image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return upscaled


# 사용 예시 및 테스트 함수
def test_ocr_quality_validator():
    """OCRQualityValidator 테스트 함수"""
    validator = OCRQualityValidator()
    
    # 테스트용 이미지 생성
    test_image = np.ones((800, 1200, 3), dtype=np.uint8) * 255  # 흰 배경
    
    # 텍스트 영역 시뮬레이션 (검은 박스들)
    cv2.rectangle(test_image, (100, 100), (1000, 150), (0, 0, 0), -1)  # 제목
    cv2.rectangle(test_image, (100, 200), (800, 230), (0, 0, 0), -1)   # 텍스트 1
    cv2.rectangle(test_image, (100, 250), (900, 280), (0, 0, 0), -1)   # 텍스트 2
    
    # 임시 파일로 저장
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        cv2.imwrite(tmp_file.name, test_image)
        
        # 품질 검증 실행
        result = validator.validate_image_quality(tmp_file.name, 'ppt_slide')
        
        print("📄 OCR 품질 검증 테스트 결과:")
        print(f"전체 품질 점수: {result['overall_assessment']['overall_score']:.1f}/100")
        print(f"품질 등급: {result['overall_assessment']['grade']}")
        print(f"예상 OCR 정확도: {result['ocr_prediction']['predicted_accuracy']:.1f}%")
        print(f"문서 타입: {result['document_type']}")
        print(f"개선 제안 수: {len(result['improvement_suggestions'])}")
        
        return result


if __name__ == "__main__":
    # 테스트 실행
    test_result = test_ocr_quality_validator()
