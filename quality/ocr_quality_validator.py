"""
👁️ OCR Quality Validator v2.1
OCR 품질 실시간 검증 및 현장 최적화 모듈

주요 기능:
- PPT/문서 인식률 실시간 측정
- 텍스트 신뢰도 점수 계산
- 이미지 전처리 품질 평가
- 주얼리 전문용어 인식 정확도
- 재촬영 권장 알고리즘
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
warnings.filterwarnings("ignore")

class OCRQualityValidator:
    """OCR 품질 실시간 검증기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # OCR 품질 기준값
        self.quality_thresholds = {
            'confidence_excellent': 90.0,  # 신뢰도 90% 이상 = 우수
            'confidence_good': 80.0,       # 신뢰도 80-90% = 양호
            'confidence_fair': 70.0,       # 신뢰도 70-80% = 보통
            'confidence_poor': 60.0,       # 신뢰도 60% 미만 = 불량
            
            'resolution_min': 150,         # 최소 DPI
            'resolution_optimal': 300,     # 최적 DPI
            
            'contrast_min': 50,            # 최소 대비
            'contrast_optimal': 100,       # 최적 대비
            
            'blur_threshold': 100,         # 블러 임계값 (라플라시안 분산)
        }
        
        # 주얼리 업계 전문용어 사전
        self.jewelry_terms = {
            'english': [
                'diamond', 'gold', 'silver', 'platinum', 'ruby', 'emerald', 'sapphire',
                'carat', 'cut', 'clarity', 'color', 'certificate', 'appraisal',
                'karat', 'setting', 'prong', 'bezel', 'pavé', 'tennis', 'solitaire',
                'vintage', 'antique', 'fine', 'jewelry', 'gemstone', 'precious',
                'brilliant', 'marquise', 'oval', 'pear', 'heart', 'asscher',
                'cushion', 'radiant', 'princess', 'round', 'GIA', 'AGS', 'AIGS'
            ],
            'korean': [
                '다이아몬드', '금', '은', '백금', '루비', '에메랄드', '사파이어',
                '캐럿', '컷', '투명도', '색상', '감정서', '평가서',
                '카라트', '세팅', '프롱', '베젤', '파베', '테니스', '솔리테어',
                '빈티지', '앤틱', '파인', '주얼리', '보석', '귀금속',
                '브릴리언트', '마키즈', '오벌', '페어', '하트', '아셔',
                '쿠션', '래디언트', '프린세스', '라운드', '지아', '에이지에스'
            ]
        }
        
        # OCR 설정
        self.ocr_config = {
            'default': '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789가-힣.,()%-:/',
            'numbers': '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.,',
            'text_only': '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz가-힣 '
        }

    def validate_ocr_quality(self, 
                           image_path: str = None, 
                           image_data: np.ndarray = None,
                           ocr_text: str = None,
                           analysis_type: str = 'comprehensive') -> Dict:
        """
        OCR 품질 종합 검증
        
        Args:
            image_path: 이미지 파일 경로
            image_data: 이미지 데이터 (numpy array)
            ocr_text: 이미 추출된 OCR 텍스트 (선택사항)
            analysis_type: 분석 유형 ('quick', 'comprehensive', 'jewelry_focused')
            
        Returns:
            Dict: OCR 품질 검증 결과
        """
        try:
            # 이미지 로드
            if image_data is None:
                if image_path is None:
                    raise ValueError("image_path 또는 image_data 중 하나는 필수입니다")
                image_data = cv2.imread(image_path)
                if image_data is None:
                    raise ValueError(f"이미지 로드 실패: {image_path}")
            
            # 기본 정보
            results = {
                'timestamp': self._get_timestamp(),
                'image_path': image_path or 'real_time_data',
                'analysis_type': analysis_type,
                'image_shape': image_data.shape
            }
            
            # 이미지 품질 분석
            image_quality = self.analyze_image_quality(image_data)
            results.update(image_quality)
            
            # OCR 수행 및 신뢰도 분석
            if ocr_text is None:
                ocr_result = self.perform_ocr_with_confidence(image_data)
            else:
                ocr_result = {'text': ocr_text, 'confidence': None}
            results.update(ocr_result)
            
            # 텍스트 품질 분석
            if ocr_result.get('text'):
                text_quality = self.analyze_text_quality(
                    ocr_result['text'], 
                    analysis_type
                )
                results.update(text_quality)
            
            # 주얼리 특화 분석
            if analysis_type in ['comprehensive', 'jewelry_focused']:
                jewelry_analysis = self.analyze_jewelry_terminology(
                    ocr_result.get('text', '')
                )
                results['jewelry_analysis'] = jewelry_analysis
            
            # 전체 OCR 품질 점수 계산
            overall_score = self.calculate_overall_ocr_score(results)
            results['overall_quality'] = overall_score
            
            # 개선 권장사항 생성
            recommendations = self.generate_ocr_recommendations(results)
            results['recommendations'] = recommendations
            
            return results
            
        except Exception as e:
            self.logger.error(f"OCR 품질 검증 오류: {str(e)}")
            return {
                'error': str(e),
                'overall_quality': {'score': 0, 'level': 'error'}
            }

    def analyze_image_quality(self, image_data: np.ndarray) -> Dict:
        """이미지 품질 분석"""
        try:
            # 그레이스케일 변환
            if len(image_data.shape) == 3:
                gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_data.copy()
            
            # 해상도 분석
            height, width = gray.shape
            total_pixels = height * width
            
            # 대비 분석
            contrast = self._calculate_contrast(gray)
            
            # 블러 분석 (라플라시안 분산)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 노이즈 분석
            noise_level = self._calculate_noise_level(gray)
            
            # 밝기 분포 분석
            brightness_stats = self._analyze_brightness_distribution(gray)
            
            # 텍스트 영역 감지
            text_regions = self._detect_text_regions(gray)
            
            return {
                'image_resolution': {
                    'width': width,
                    'height': height,
                    'total_pixels': total_pixels,
                    'estimated_dpi': self._estimate_dpi(width, height)
                },
                'contrast_score': round(contrast, 1),
                'blur_score': round(blur_score, 1),
                'noise_level': round(noise_level, 3),
                'brightness_stats': brightness_stats,
                'text_regions_count': len(text_regions),
                'text_coverage_ratio': self._calculate_text_coverage(text_regions, gray.shape)
            }
            
        except Exception as e:
            self.logger.error(f"이미지 품질 분석 오류: {str(e)}")
            return {
                'image_resolution': {'width': 0, 'height': 0},
                'contrast_score': 0.0,
                'blur_score': 0.0
            }

    def perform_ocr_with_confidence(self, image_data: np.ndarray) -> Dict:
        """신뢰도 정보와 함께 OCR 수행"""
        try:
            # 이미지 전처리
            processed_image = self._preprocess_for_ocr(image_data)
            
            # OCR 수행 (신뢰도 정보 포함)
            data = pytesseract.image_to_data(
                processed_image, 
                config=self.ocr_config['default'],
                output_type=pytesseract.Output.DICT
            )
            
            # 텍스트와 신뢰도 추출
            text_parts = []
            confidences = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # 유효한 신뢰도만
                    text = data['text'][i].strip()
                    if text:
                        text_parts.append(text)
                        confidences.append(int(data['conf'][i]))
            
            # 전체 텍스트 조합
            full_text = ' '.join(text_parts)
            
            # 평균 신뢰도 계산
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # 신뢰도 분포 분석
            confidence_distribution = self._analyze_confidence_distribution(confidences)
            
            return {
                'text': full_text,
                'confidence': round(avg_confidence, 1),
                'word_count': len(text_parts),
                'confidence_distribution': confidence_distribution,
                'low_confidence_words': [
                    text_parts[i] for i, conf in enumerate(confidences) 
                    if conf < self.quality_thresholds['confidence_fair']
                ]
            }
            
        except Exception as e:
            self.logger.error(f"OCR 수행 오류: {str(e)}")
            return {
                'text': '',
                'confidence': 0.0,
                'word_count': 0
            }

    def analyze_text_quality(self, text: str, analysis_type: str) -> Dict:
        """텍스트 품질 분석"""
        try:
            # 기본 텍스트 통계
            char_count = len(text)
            word_count = len(text.split())
            line_count = len(text.split('\n'))
            
            # 언어 감지
            language_info = self._detect_language(text)
            
            # 특수문자/숫자 비율
            special_char_ratio = self._calculate_special_char_ratio(text)
            digit_ratio = self._calculate_digit_ratio(text)
            
            # 가독성 점수
            readability_score = self._calculate_readability_score(text)
            
            # 오타/이상 패턴 감지
            anomaly_detection = self._detect_text_anomalies(text)
            
            return {
                'text_statistics': {
                    'character_count': char_count,
                    'word_count': word_count,
                    'line_count': line_count,
                    'avg_word_length': round(char_count / max(word_count, 1), 1)
                },
                'language_info': language_info,
                'content_ratios': {
                    'special_char_ratio': round(special_char_ratio, 3),
                    'digit_ratio': round(digit_ratio, 3),
                    'alpha_ratio': round(1 - special_char_ratio - digit_ratio, 3)
                },
                'readability_score': round(readability_score, 3),
                'anomaly_detection': anomaly_detection
            }
            
        except Exception as e:
            self.logger.error(f"텍스트 품질 분석 오류: {str(e)}")
            return {
                'text_statistics': {'character_count': 0, 'word_count': 0},
                'readability_score': 0.0
            }

    def analyze_jewelry_terminology(self, text: str) -> Dict:
        """주얼리 전문용어 분석"""
        try:
            text_lower = text.lower()
            
            # 영어 전문용어 매칭
            english_matches = []
            for term in self.jewelry_terms['english']:
                if term.lower() in text_lower:
                    english_matches.append(term)
            
            # 한국어 전문용어 매칭
            korean_matches = []
            for term in self.jewelry_terms['korean']:
                if term in text:
                    korean_matches.append(term)
            
            # 숫자 패턴 분석 (캐럿, 가격 등)
            number_patterns = self._extract_jewelry_numbers(text)
            
            # 브랜드/인증기관 감지
            certifications = self._detect_certifications(text)
            
            # 주얼리 특화 점수 계산
            jewelry_score = self._calculate_jewelry_score(
                english_matches, korean_matches, number_patterns, certifications
            )
            
            return {
                'english_terms_found': english_matches,
                'korean_terms_found': korean_matches,
                'term_count': len(english_matches) + len(korean_matches),
                'number_patterns': number_patterns,
                'certifications': certifications,
                'jewelry_relevance_score': round(jewelry_score, 3),
                'domain_confidence': self._classify_domain_confidence(jewelry_score)
            }
            
        except Exception as e:
            self.logger.error(f"주얼리 용어 분석 오류: {str(e)}")
            return {
                'english_terms_found': [],
                'korean_terms_found': [],
                'jewelry_relevance_score': 0.0
            }

    def calculate_overall_ocr_score(self, results: Dict) -> Dict:
        """전체 OCR 품질 점수 계산"""
        try:
            # 가중치 설정
            weights = {
                'image_quality': 0.3,    # 이미지 품질 30%
                'ocr_confidence': 0.4,   # OCR 신뢰도 40%
                'text_quality': 0.2,     # 텍스트 품질 20%
                'jewelry_relevance': 0.1 # 주얼리 관련성 10%
            }
            
            # 개별 점수 정규화
            image_score = self._normalize_image_score(results)
            confidence_score = self._normalize_confidence_score(results.get('confidence', 0))
            text_score = self._normalize_text_score(results)
            jewelry_score = results.get('jewelry_analysis', {}).get('jewelry_relevance_score', 0)
            
            # 가중 평균 계산
            overall_score = (
                image_score * weights['image_quality'] +
                confidence_score * weights['ocr_confidence'] +
                text_score * weights['text_quality'] +
                jewelry_score * weights['jewelry_relevance']
            )
            
            # 등급 분류
            level, status, color = self._classify_ocr_quality_level(overall_score)
            
            return {
                'score': round(overall_score, 3),
                'percentage': round(overall_score * 100, 1),
                'level': level,
                'status': status,
                'color': color,
                'components': {
                    'image_score': round(image_score, 3),
                    'confidence_score': round(confidence_score, 3),
                    'text_score': round(text_score, 3),
                    'jewelry_score': round(jewelry_score, 3)
                }
            }
            
        except Exception as e:
            self.logger.error(f"전체 OCR 점수 계산 오류: {str(e)}")
            return {
                'score': 0.0,
                'level': 'error',
                'status': '오류'
            }

    def generate_ocr_recommendations(self, results: Dict) -> List[Dict]:
        """OCR 품질 개선 권장사항 생성"""
        recommendations = []
        
        try:
            overall_quality = results.get('overall_quality', {})
            confidence = results.get('confidence', 0)
            blur_score = results.get('blur_score', 0)
            contrast_score = results.get('contrast_score', 0)
            
            # 신뢰도 기반 권장사항
            if confidence < self.quality_thresholds['confidence_poor']:
                recommendations.append({
                    'type': 'critical',
                    'icon': '🔴',
                    'title': 'OCR 신뢰도 매우 낮음',
                    'message': '조명을 개선하고 카메라를 안정적으로 고정한 후 재촬영하세요',
                    'action': 'improve_lighting_and_stability'
                })
            elif confidence < self.quality_thresholds['confidence_fair']:
                recommendations.append({
                    'type': 'warning',
                    'icon': '🟡',
                    'title': 'OCR 신뢰도 개선 필요',
                    'message': '문서를 평평하게 펴고 수직으로 촬영해보세요',
                    'action': 'flatten_document_and_perpendicular_shot'
                })
            
            # 블러 기반 권장사항
            if blur_score < self.quality_thresholds['blur_threshold']:
                recommendations.append({
                    'type': 'warning',
                    'icon': '🟠',
                    'title': '이미지 흐림 감지',
                    'message': '카메라를 고정하고 초점을 맞춘 후 촬영하세요',
                    'action': 'fix_camera_and_focus'
                })
            
            # 대비 기반 권장사항
            if contrast_score < self.quality_thresholds['contrast_min']:
                recommendations.append({
                    'type': 'warning',
                    'icon': '🟠',
                    'title': '이미지 대비 낮음',
                    'message': '조명을 개선하거나 배경과 텍스트의 대비를 높여주세요',
                    'action': 'improve_contrast'
                })
            
            # 전체 품질 기반 권장사항
            if overall_quality.get('level') == 'poor':
                recommendations.append({
                    'type': 'critical',
                    'icon': '🔴',
                    'title': '재촬영 권장',
                    'message': '현재 OCR 품질이 좋지 않습니다. 환경을 개선한 후 다시 촬영해보세요',
                    'action': 'retry_capture'
                })
            elif overall_quality.get('level') == 'excellent':
                recommendations.append({
                    'type': 'success',
                    'icon': '🟢',
                    'title': '최적 품질',
                    'message': '현재 설정을 유지하여 계속 촬영하세요',
                    'action': 'maintain_current_settings'
                })
            
            # 주얼리 특화 권장사항
            jewelry_analysis = results.get('jewelry_analysis', {})
            if jewelry_analysis.get('jewelry_relevance_score', 0) < 0.3:
                recommendations.append({
                    'type': 'info',
                    'icon': '💎',
                    'title': '주얼리 전문용어 부족',
                    'message': '주얼리 관련 문서인지 확인해주세요',
                    'action': 'verify_jewelry_document'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"OCR 권장사항 생성 오류: {str(e)}")
            return [{
                'type': 'error',
                'icon': '❌',
                'title': 'OCR 분석 오류',
                'message': 'OCR 품질 분석 중 오류가 발생했습니다',
                'action': 'retry_ocr_analysis'
            }]

    # === 내부 유틸리티 함수들 ===
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """OCR을 위한 이미지 전처리"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 노이즈 제거
        denoised = cv2.medianBlur(gray, 3)
        
        # 대비 개선
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 이진화
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """이미지 대비 계산"""
        return float(np.std(image))
    
    def _calculate_noise_level(self, image: np.ndarray) -> float:
        """노이즈 레벨 계산"""
        # 라플라시안 필터를 이용한 노이즈 추정
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        return float(laplacian_var) / 10000  # 정규화
    
    def _analyze_brightness_distribution(self, image: np.ndarray) -> Dict:
        """밝기 분포 분석"""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        return {
            'mean_brightness': round(float(np.mean(image)), 1),
            'brightness_std': round(float(np.std(image)), 1),
            'dark_pixel_ratio': round(float(np.sum(image < 85) / image.size), 3),
            'bright_pixel_ratio': round(float(np.sum(image > 170) / image.size), 3)
        }
    
    def _detect_text_regions(self, image: np.ndarray) -> List:
        """텍스트 영역 감지"""
        # MSER을 이용한 텍스트 영역 감지
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(image)
        
        # 텍스트 같은 영역 필터링
        text_regions = []
        for region in regions:
            if len(region) > 50 and len(region) < 2000:  # 적당한 크기의 영역만
                text_regions.append(region)
        
        return text_regions
    
    def _calculate_text_coverage(self, text_regions: List, image_shape: Tuple) -> float:
        """텍스트 영역 커버리지 계산"""
        if not text_regions:
            return 0.0
        
        total_pixels = image_shape[0] * image_shape[1]
        text_pixels = sum(len(region) for region in text_regions)
        
        return min(1.0, text_pixels / total_pixels)
    
    def _estimate_dpi(self, width: int, height: int) -> int:
        """DPI 추정 (A4 기준)"""
        # A4 크기 기준으로 DPI 추정
        a4_width_inch = 8.27
        a4_height_inch = 11.69
        
        dpi_w = width / a4_width_inch
        dpi_h = height / a4_height_inch
        
        return int((dpi_w + dpi_h) / 2)
    
    def _analyze_confidence_distribution(self, confidences: List[int]) -> Dict:
        """신뢰도 분포 분석"""
        if not confidences:
            return {'high': 0, 'medium': 0, 'low': 0}
        
        high_conf = sum(1 for c in confidences if c >= 80)
        medium_conf = sum(1 for c in confidences if 60 <= c < 80)
        low_conf = sum(1 for c in confidences if c < 60)
        total = len(confidences)
        
        return {
            'high': round(high_conf / total, 3),
            'medium': round(medium_conf / total, 3),
            'low': round(low_conf / total, 3)
        }
    
    def _detect_language(self, text: str) -> Dict:
        """언어 감지"""
        # 한글 문자 비율
        korean_chars = len(re.findall(r'[가-힣]', text))
        # 영문 문자 비율
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        # 숫자 비율
        digit_chars = len(re.findall(r'[0-9]', text))
        
        total_chars = max(1, len(re.sub(r'\s', '', text)))
        
        return {
            'korean_ratio': round(korean_chars / total_chars, 3),
            'english_ratio': round(english_chars / total_chars, 3),
            'digit_ratio': round(digit_chars / total_chars, 3),
            'primary_language': 'korean' if korean_chars > english_chars else 'english'
        }
    
    def _calculate_special_char_ratio(self, text: str) -> float:
        """특수문자 비율 계산"""
        special_chars = len(re.findall(r'[^a-zA-Z0-9가-힣\s]', text))
        total_chars = max(1, len(text))
        return special_chars / total_chars
    
    def _calculate_digit_ratio(self, text: str) -> float:
        """숫자 비율 계산"""
        digits = len(re.findall(r'[0-9]', text))
        total_chars = max(1, len(text))
        return digits / total_chars
    
    def _calculate_readability_score(self, text: str) -> float:
        """가독성 점수 계산"""
        if not text.strip():
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # 평균 단어 길이
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # 단어 길이 기준 가독성 (3-7자가 적당)
        length_score = max(0, 1 - abs(avg_word_length - 5) / 10)
        
        # 공백/구두점 비율 (적당한 비율이 좋음)
        spaces = text.count(' ')
        punctuation = len(re.findall(r'[.,!?;:]', text))
        structure_score = min(1.0, (spaces + punctuation) / len(words))
        
        return (length_score + structure_score) / 2
    
    def _detect_text_anomalies(self, text: str) -> Dict:
        """텍스트 이상 패턴 감지"""
        anomalies = []
        
        # 연속된 동일 문자
        repeated_chars = re.findall(r'(.)\1{3,}', text)
        if repeated_chars:
            anomalies.append('repeated_characters')
        
        # 비정상적인 대소문자 패턴
        if re.search(r'[a-z][A-Z][a-z][A-Z]', text):
            anomalies.append('irregular_case_pattern')
        
        # 과도한 특수문자
        special_char_ratio = self._calculate_special_char_ratio(text)
        if special_char_ratio > 0.3:
            anomalies.append('excessive_special_characters')
        
        # 숫자만 있는 긴 문자열
        if re.search(r'\d{15,}', text):
            anomalies.append('excessive_consecutive_digits')
        
        return {
            'anomalies_found': anomalies,
            'anomaly_count': len(anomalies),
            'has_anomalies': len(anomalies) > 0
        }
    
    def _extract_jewelry_numbers(self, text: str) -> Dict:
        """주얼리 관련 숫자 패턴 추출"""
        patterns = {
            'carat_weights': re.findall(r'(\d+\.?\d*)\s*(?:ct|carat|캐럿|카라트)', text, re.IGNORECASE),
            'prices': re.findall(r'[₩$¥€]\s*[\d,]+(?:\.\d{2})?', text),
            'percentages': re.findall(r'\d+(?:\.\d+)?\s*%', text),
            'measurements': re.findall(r'\d+(?:\.\d+)?\s*(?:mm|cm|inch)', text, re.IGNORECASE)
        }
        
        # 빈 리스트 제거
        patterns = {k: v for k, v in patterns.items() if v}
        
        return patterns
    
    def _detect_certifications(self, text: str) -> List[str]:
        """인증기관/브랜드 감지"""
        certifications = []
        
        cert_patterns = [
            r'GIA\b', r'AGS\b', r'AIGS\b', r'SSEF\b', r'Gübelin\b',
            r'Cartier\b', r'Tiffany\b', r'Harry Winston\b', r'Bulgari\b',
            r'지아\b', r'에이지에스\b', r'까르띠에\b', r'티파니\b'
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            certifications.extend(matches)
        
        return list(set(certifications))  # 중복 제거
    
    def _calculate_jewelry_score(self, english_terms: List, korean_terms: List, 
                                numbers: Dict, certifications: List) -> float:
        """주얼리 특화 점수 계산"""
        # 용어 점수 (최대 0.4)
        term_score = min(0.4, (len(english_terms) + len(korean_terms)) * 0.05)
        
        # 숫자 패턴 점수 (최대 0.3)
        number_score = min(0.3, len(numbers) * 0.1)
        
        # 인증 점수 (최대 0.3)
        cert_score = min(0.3, len(certifications) * 0.15)
        
        return term_score + number_score + cert_score
    
    def _classify_domain_confidence(self, score: float) -> str:
        """도메인 신뢰도 분류"""
        if score >= 0.7:
            return 'high_jewelry_relevance'
        elif score >= 0.4:
            return 'medium_jewelry_relevance'
        elif score >= 0.1:
            return 'low_jewelry_relevance'
        else:
            return 'non_jewelry_content'
    
    def _normalize_image_score(self, results: Dict) -> float:
        """이미지 품질 점수 정규화"""
        contrast = results.get('contrast_score', 0)
        blur = results.get('blur_score', 0)
        
        # 대비 점수 (0-1)
        contrast_normalized = min(1.0, contrast / 100)
        
        # 블러 점수 (0-1, 높을수록 좋음)
        blur_normalized = min(1.0, blur / 200)
        
        return (contrast_normalized + blur_normalized) / 2
    
    def _normalize_confidence_score(self, confidence: float) -> float:
        """신뢰도 점수 정규화"""
        return min(1.0, confidence / 100)
    
    def _normalize_text_score(self, results: Dict) -> float:
        """텍스트 품질 점수 정규화"""
        readability = results.get('readability_score', 0)
        anomaly_detection = results.get('anomaly_detection', {})
        
        # 이상 패턴이 있으면 점수 감점
        anomaly_penalty = 0.1 * anomaly_detection.get('anomaly_count', 0)
        
        return max(0.0, readability - anomaly_penalty)
    
    def _classify_ocr_quality_level(self, score: float) -> Tuple[str, str, str]:
        """OCR 품질 등급 분류"""
        if score >= 0.9:
            return 'excellent', '우수', '🟢'
        elif score >= 0.8:
            return 'good', '양호', '🟡'
        elif score >= 0.7:
            return 'fair', '보통', '🟠'
        else:
            return 'poor', '불량', '🔴'
    
    def _get_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 사용 예제
if __name__ == "__main__":
    validator = OCRQualityValidator()
    
    print("👁️ OCR Quality Validator v2.1 - 테스트 시작")
    print("=" * 50)
    
    # 실제 사용 시에는 이미지 파일 경로를 제공
    # result = validator.validate_ocr_quality("test_document.jpg")
    # print(f"전체 OCR 품질: {result['overall_quality']['percentage']}%")
    # print(f"OCR 신뢰도: {result['confidence']}%")
    
    print("모듈 로드 완료 ✅")
