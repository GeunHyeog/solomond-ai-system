"""
📸 Image Quality Assessor v2.1
이미지 품질 실시간 평가 및 현장 최적화 모듈

주요 기능:
- 해상도/선명도 실시간 측정
- 조명/대비 품질 분석
- 블러/노이즈 감지
- 주얼리 촬영 최적화 가이드
- 재촬영 권장 알고리즘
"""

import cv2
import numpy as np
from PIL import Image, ImageStat, ImageFilter, ImageEnhance
from typing import Dict, List, Tuple, Optional, Union
import logging
import math
import warnings
warnings.filterwarnings("ignore")

class ImageQualityAssessor:
    """이미지 품질 실시간 평가기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 이미지 품질 기준값
        self.quality_thresholds = {
            # 해상도 기준
            'resolution_min': 800,        # 최소 너비/높이
            'resolution_good': 1200,      # 권장 너비/높이
            'resolution_excellent': 1920, # 최적 너비/높이
            
            # 선명도 기준 (라플라시안 분산)
            'sharpness_poor': 50,         # 50 미만 = 흐림
            'sharpness_fair': 100,        # 50-100 = 보통
            'sharpness_good': 200,        # 100-200 = 양호
            'sharpness_excellent': 300,   # 200+ = 우수
            
            # 대비 기준
            'contrast_poor': 30,          # 30 미만 = 낮음
            'contrast_fair': 50,          # 30-50 = 보통
            'contrast_good': 80,          # 50-80 = 양호
            'contrast_excellent': 120,    # 80+ = 우수
            
            # 밝기 기준 (0-255)
            'brightness_too_dark': 50,    # 50 미만 = 너무 어두움
            'brightness_dark': 80,        # 50-80 = 어두움
            'brightness_optimal_min': 100, # 100-180 = 최적
            'brightness_optimal_max': 180,
            'brightness_bright': 200,     # 180-200 = 밝음
            'brightness_too_bright': 220, # 200+ = 너무 밝음
            
            # 노이즈 기준
            'noise_excellent': 5,         # 5 미만 = 우수
            'noise_good': 10,             # 5-10 = 양호
            'noise_fair': 20,             # 10-20 = 보통
            'noise_poor': 30,             # 20+ = 불량
        }
        
        # 주얼리 촬영 특화 설정
        self.jewelry_settings = {
            'preferred_backgrounds': ['white', 'black', 'neutral'],
            'optimal_lighting_zones': [(100, 180)],  # 최적 밝기 구간
            'macro_focus_threshold': 0.3,            # 매크로 초점 임계값
            'reflection_detection_threshold': 240,   # 반사 감지 임계값
        }

    def assess_image_quality(self, 
                           image_path: str = None, 
                           image_data: np.ndarray = None,
                           assessment_type: str = 'comprehensive') -> Dict:
        """
        이미지 품질 종합 평가
        
        Args:
            image_path: 이미지 파일 경로
            image_data: 이미지 데이터 (numpy array)
            assessment_type: 평가 유형 ('quick', 'comprehensive', 'jewelry_focused')
            
        Returns:
            Dict: 이미지 품질 평가 결과
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
                'assessment_type': assessment_type,
                'image_shape': image_data.shape
            }
            
            # 기본 품질 분석
            basic_quality = self.analyze_basic_quality(image_data)
            results.update(basic_quality)
            
            # 고급 품질 분석
            if assessment_type in ['comprehensive', 'jewelry_focused']:
                advanced_quality = self.analyze_advanced_quality(image_data)
                results.update(advanced_quality)
            
            # 주얼리 특화 분석
            if assessment_type == 'jewelry_focused':
                jewelry_analysis = self.analyze_jewelry_photography(image_data)
                results['jewelry_analysis'] = jewelry_analysis
            
            # 전체 이미지 품질 점수 계산
            overall_score = self.calculate_overall_image_score(results)
            results['overall_quality'] = overall_score
            
            # 개선 권장사항 생성
            recommendations = self.generate_image_recommendations(results)
            results['recommendations'] = recommendations
            
            return results
            
        except Exception as e:
            self.logger.error(f"이미지 품질 평가 오류: {str(e)}")
            return {
                'error': str(e),
                'overall_quality': {'score': 0, 'level': 'error'}
            }

    def analyze_basic_quality(self, image_data: np.ndarray) -> Dict:
        """기본 이미지 품질 분석"""
        try:
            # 그레이스케일 변환
            if len(image_data.shape) == 3:
                gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_data.copy()
            
            # 해상도 분석
            height, width = gray.shape
            resolution_analysis = self.analyze_resolution(width, height)
            
            # 선명도 분석 (라플라시안 분산)
            sharpness_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_level = self._classify_sharpness(sharpness_score)
            
            # 대비 분석
            contrast_score = float(np.std(gray))
            contrast_level = self._classify_contrast(contrast_score)
            
            # 밝기 분석
            brightness_analysis = self.analyze_brightness(gray)
            
            # 노이즈 분석
            noise_analysis = self.analyze_noise(gray)
            
            return {
                'resolution': resolution_analysis,
                'sharpness': {
                    'score': round(sharpness_score, 1),
                    'level': sharpness_level
                },
                'contrast': {
                    'score': round(contrast_score, 1),
                    'level': contrast_level
                },
                'brightness': brightness_analysis,
                'noise': noise_analysis
            }
            
        except Exception as e:
            self.logger.error(f"기본 품질 분석 오류: {str(e)}")
            return {
                'resolution': {'width': 0, 'height': 0},
                'sharpness': {'score': 0, 'level': 'poor'},
                'contrast': {'score': 0, 'level': 'poor'}
            }

    def analyze_advanced_quality(self, image_data: np.ndarray) -> Dict:
        """고급 이미지 품질 분석"""
        try:
            # 그레이스케일 변환
            if len(image_data.shape) == 3:
                gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_data.copy()
            
            # 히스토그램 분석
            histogram_analysis = self.analyze_histogram(gray)
            
            # 엣지 품질 분석
            edge_analysis = self.analyze_edge_quality(gray)
            
            # 텍스처 분석
            texture_analysis = self.analyze_texture(gray)
            
            # 컬러 분석 (컬러 이미지인 경우)
            color_analysis = {}
            if len(image_data.shape) == 3:
                color_analysis = self.analyze_color_quality(image_data)
            
            # 구도 분석
            composition_analysis = self.analyze_composition(gray)
            
            # 초점 분석
            focus_analysis = self.analyze_focus_quality(gray)
            
            return {
                'histogram': histogram_analysis,
                'edge_quality': edge_analysis,
                'texture': texture_analysis,
                'color': color_analysis,
                'composition': composition_analysis,
                'focus': focus_analysis
            }
            
        except Exception as e:
            self.logger.error(f"고급 품질 분석 오류: {str(e)}")
            return {
                'histogram': {},
                'edge_quality': {},
                'texture': {}
            }

    def analyze_jewelry_photography(self, image_data: np.ndarray) -> Dict:
        """주얼리 촬영 특화 분석"""
        try:
            # 그레이스케일 변환
            if len(image_data.shape) == 3:
                gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
                color = image_data.copy()
            else:
                gray = image_data.copy()
                color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # 배경 분석
            background_analysis = self.analyze_background(color)
            
            # 반사/하이라이트 분석
            reflection_analysis = self.analyze_reflections(gray)
            
            # 세부 디테일 분석 (주얼리의 세밀한 부분)
            detail_analysis = self.analyze_detail_quality(gray)
            
            # 조명 품질 분석
            lighting_analysis = self.analyze_lighting_quality(gray)
            
            # 매크로 촬영 품질
            macro_analysis = self.analyze_macro_quality(gray)
            
            # 주얼리 적합성 점수
            jewelry_suitability = self.calculate_jewelry_suitability(
                background_analysis, reflection_analysis, detail_analysis,
                lighting_analysis, macro_analysis
            )
            
            return {
                'background': background_analysis,
                'reflections': reflection_analysis,
                'detail_quality': detail_analysis,
                'lighting': lighting_analysis,
                'macro_quality': macro_analysis,
                'jewelry_suitability': jewelry_suitability
            }
            
        except Exception as e:
            self.logger.error(f"주얼리 촬영 분석 오류: {str(e)}")
            return {
                'background': {},
                'jewelry_suitability': {'score': 0, 'level': 'poor'}
            }

    def analyze_resolution(self, width: int, height: int) -> Dict:
        """해상도 분석"""
        total_pixels = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        # 해상도 등급 분류
        min_dimension = min(width, height)
        if min_dimension >= self.quality_thresholds['resolution_excellent']:
            resolution_level = 'excellent'
        elif min_dimension >= self.quality_thresholds['resolution_good']:
            resolution_level = 'good'
        elif min_dimension >= self.quality_thresholds['resolution_min']:
            resolution_level = 'fair'
        else:
            resolution_level = 'poor'
        
        return {
            'width': width,
            'height': height,
            'total_pixels': total_pixels,
            'aspect_ratio': round(aspect_ratio, 2),
            'megapixels': round(total_pixels / 1000000, 1),
            'level': resolution_level
        }

    def analyze_brightness(self, gray_image: np.ndarray) -> Dict:
        """밝기 분석"""
        mean_brightness = float(np.mean(gray_image))
        brightness_std = float(np.std(gray_image))
        
        # 밝기 히스토그램
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        
        # 밝기 분포 분석
        dark_pixels = np.sum(gray_image < 85) / gray_image.size
        bright_pixels = np.sum(gray_image > 170) / gray_image.size
        mid_pixels = 1 - dark_pixels - bright_pixels
        
        # 밝기 등급 분류
        if mean_brightness < self.quality_thresholds['brightness_too_dark']:
            brightness_level = 'too_dark'
        elif mean_brightness < self.quality_thresholds['brightness_dark']:
            brightness_level = 'dark'
        elif mean_brightness <= self.quality_thresholds['brightness_optimal_max']:
            brightness_level = 'optimal'
        elif mean_brightness < self.quality_thresholds['brightness_bright']:
            brightness_level = 'bright'
        else:
            brightness_level = 'too_bright'
        
        return {
            'mean': round(mean_brightness, 1),
            'std': round(brightness_std, 1),
            'level': brightness_level,
            'distribution': {
                'dark_ratio': round(dark_pixels, 3),
                'mid_ratio': round(mid_pixels, 3),
                'bright_ratio': round(bright_pixels, 3)
            }
        }

    def analyze_noise(self, gray_image: np.ndarray) -> Dict:
        """노이즈 분석"""
        # 가우시안 블러 적용 후 차이 계산
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        noise_image = cv2.absdiff(gray_image, blurred)
        noise_level = float(np.mean(noise_image))
        
        # 고주파 노이즈 분석
        high_freq_noise = self._calculate_high_frequency_noise(gray_image)
        
        # 노이즈 등급 분류
        if noise_level < self.quality_thresholds['noise_excellent']:
            noise_grade = 'excellent'
        elif noise_level < self.quality_thresholds['noise_good']:
            noise_grade = 'good'
        elif noise_level < self.quality_thresholds['noise_fair']:
            noise_grade = 'fair'
        else:
            noise_grade = 'poor'
        
        return {
            'level': round(noise_level, 2),
            'grade': noise_grade,
            'high_frequency_noise': round(high_freq_noise, 2)
        }

    def analyze_histogram(self, gray_image: np.ndarray) -> Dict:
        """히스토그램 분석"""
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # 히스토그램 통계
        peak_value = int(np.argmax(hist))
        histogram_spread = float(np.std(np.arange(256), weights=hist))
        
        # 다이나믹 레인지
        non_zero_indices = np.nonzero(hist)[0]
        if len(non_zero_indices) > 0:
            dynamic_range = int(non_zero_indices[-1] - non_zero_indices[0])
        else:
            dynamic_range = 0
        
        # 히스토그램 균등성 (엔트로피)
        hist_normalized = hist / np.sum(hist)
        hist_normalized = hist_normalized[hist_normalized > 0]
        entropy = float(-np.sum(hist_normalized * np.log2(hist_normalized)))
        
        return {
            'peak_value': peak_value,
            'spread': round(histogram_spread, 1),
            'dynamic_range': dynamic_range,
            'entropy': round(entropy, 2)
        }

    def analyze_edge_quality(self, gray_image: np.ndarray) -> Dict:
        """엣지 품질 분석"""
        # Canny 엣지 검출
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)
        
        # Sobel 엣지 강도
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_strength = float(np.mean(sobel_magnitude))
        
        return {
            'density': round(edge_density, 4),
            'strength': round(edge_strength, 1),
            'quality_score': round((edge_density * 1000 + edge_strength) / 2, 1)
        }

    def analyze_texture(self, gray_image: np.ndarray) -> Dict:
        """텍스처 분석"""
        # Local Binary Pattern을 간단히 구현
        texture_variance = self._calculate_texture_variance(gray_image)
        
        # 텍스처 에너지 (GLCM 기반 간단 버전)
        texture_energy = self._calculate_texture_energy(gray_image)
        
        return {
            'variance': round(texture_variance, 2),
            'energy': round(texture_energy, 4),
            'richness_score': round((texture_variance + texture_energy * 1000) / 2, 1)
        }

    def analyze_color_quality(self, color_image: np.ndarray) -> Dict:
        """컬러 품질 분석"""
        # BGR을 HSV로 변환
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # 채도 분석
        saturation = hsv[:, :, 1]
        mean_saturation = float(np.mean(saturation))
        
        # 색상 분포 분석
        hue = hsv[:, :, 0]
        hue_variance = float(np.var(hue))
        
        # 컬러 균형 분석
        b, g, r = cv2.split(color_image)
        color_balance = {
            'red_mean': float(np.mean(r)),
            'green_mean': float(np.mean(g)),
            'blue_mean': float(np.mean(b))
        }
        
        # 화이트 밸런스 점수 (RGB 평균의 균등성)
        rgb_means = [color_balance['red_mean'], color_balance['green_mean'], color_balance['blue_mean']]
        white_balance_score = 1.0 - (np.std(rgb_means) / np.mean(rgb_means))
        
        return {
            'saturation': {
                'mean': round(mean_saturation, 1),
                'level': 'high' if mean_saturation > 100 else 'medium' if mean_saturation > 50 else 'low'
            },
            'hue_variance': round(hue_variance, 1),
            'color_balance': {k: round(v, 1) for k, v in color_balance.items()},
            'white_balance_score': round(white_balance_score, 3)
        }

    def analyze_composition(self, gray_image: np.ndarray) -> Dict:
        """구도 분석"""
        height, width = gray_image.shape
        
        # 무게 중심 계산
        moments = cv2.moments(gray_image)
        if moments['m00'] != 0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
        else:
            center_x, center_y = width // 2, height // 2
        
        # 황금비율 점들과의 거리
        golden_points = [
            (width * 0.618, height * 0.618),
            (width * 0.382, height * 0.618),
            (width * 0.618, height * 0.382),
            (width * 0.382, height * 0.382)
        ]
        
        min_distance_to_golden = min([
            math.sqrt((center_x - gx)**2 + (center_y - gy)**2)
            for gx, gy in golden_points
        ])
        
        # 구도 점수 (가까울수록 높은 점수)
        composition_score = max(0, 1 - min_distance_to_golden / (width * 0.5))
        
        return {
            'center_x': center_x,
            'center_y': center_y,
            'golden_ratio_distance': round(min_distance_to_golden, 1),
            'composition_score': round(composition_score, 3)
        }

    def analyze_focus_quality(self, gray_image: np.ndarray) -> Dict:
        """초점 품질 분석"""
        # 이미지를 여러 구역으로 나누어 각 구역의 선명도 측정
        height, width = gray_image.shape
        
        # 3x3 그리드로 나누기
        focus_map = {}
        grid_height = height // 3
        grid_width = width // 3
        
        for i in range(3):
            for j in range(3):
                y1 = i * grid_height
                y2 = min((i + 1) * grid_height, height)
                x1 = j * grid_width
                x2 = min((j + 1) * grid_width, width)
                
                region = gray_image[y1:y2, x1:x2]
                region_sharpness = cv2.Laplacian(region, cv2.CV_64F).var()
                focus_map[f'region_{i}_{j}'] = round(region_sharpness, 1)
        
        # 중앙 영역 초점 품질
        center_region = gray_image[height//4:3*height//4, width//4:3*width//4]
        center_focus = cv2.Laplacian(center_region, cv2.CV_64F).var()
        
        # 초점 일관성 (분산이 낮을수록 일관성 높음)
        focus_values = list(focus_map.values())
        focus_consistency = 1.0 / (1.0 + np.std(focus_values) / np.mean(focus_values))
        
        return {
            'center_focus': round(center_focus, 1),
            'focus_map': focus_map,
            'focus_consistency': round(focus_consistency, 3),
            'overall_focus_score': round(center_focus * focus_consistency / 100, 3)
        }

    def analyze_background(self, color_image: np.ndarray) -> Dict:
        """배경 분석 (주얼리 촬영용)"""
        # 테두리 영역을 배경으로 간주
        height, width = color_image.shape[:2]
        border_size = min(height, width) // 20
        
        # 테두리 영역 추출
        top_border = color_image[:border_size, :]
        bottom_border = color_image[-border_size:, :]
        left_border = color_image[:, :border_size]
        right_border = color_image[:, -border_size:]
        
        # 모든 테두리 영역 결합
        background_pixels = np.concatenate([
            top_border.reshape(-1, 3),
            bottom_border.reshape(-1, 3),
            left_border.reshape(-1, 3),
            right_border.reshape(-1, 3)
        ])
        
        # 배경색 분석
        mean_bg_color = np.mean(background_pixels, axis=0)
        bg_color_std = np.std(background_pixels, axis=0)
        
        # 배경 균등성 (표준편차가 낮을수록 균등)
        bg_uniformity = 1.0 / (1.0 + np.mean(bg_color_std) / 255)
        
        # 배경색 분류
        bg_brightness = np.mean(mean_bg_color)
        if bg_brightness > 200:
            bg_type = 'white'
        elif bg_brightness < 50:
            bg_type = 'black'
        elif np.std(mean_bg_color) < 20:
            bg_type = 'neutral'
        else:
            bg_type = 'colored'
        
        return {
            'mean_color': [round(float(c), 1) for c in mean_bg_color],
            'uniformity': round(bg_uniformity, 3),
            'type': bg_type,
            'suitability': bg_type in self.jewelry_settings['preferred_backgrounds']
        }

    def analyze_reflections(self, gray_image: np.ndarray) -> Dict:
        """반사/하이라이트 분석"""
        # 매우 밝은 영역 감지 (반사 영역)
        reflection_threshold = self.jewelry_settings['reflection_detection_threshold']
        reflection_mask = gray_image > reflection_threshold
        reflection_ratio = float(np.sum(reflection_mask) / gray_image.size)
        
        # 반사 영역의 연속성 분석
        contours, _ = cv2.findContours(
            reflection_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        large_reflections = [c for c in contours if cv2.contourArea(c) > 100]
        
        return {
            'reflection_ratio': round(reflection_ratio, 4),
            'num_reflection_areas': len(large_reflections),
            'has_excessive_reflections': reflection_ratio > 0.05,
            'reflection_quality': 'good' if 0.001 < reflection_ratio < 0.02 else 'poor'
        }

    def analyze_detail_quality(self, gray_image: np.ndarray) -> Dict:
        """세부 디테일 품질 분석"""
        # 고주파 성분 분석 (세밀한 디테일)
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        high_pass = cv2.filter2D(gray_image, -1, kernel)
        detail_score = float(np.std(high_pass))
        
        # 미세 텍스처 분석
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        micro_detail = float(np.mean(gradient_magnitude))
        
        return {
            'detail_score': round(detail_score, 1),
            'micro_detail': round(micro_detail, 1),
            'detail_quality': 'excellent' if detail_score > 20 else 'good' if detail_score > 10 else 'poor'
        }

    def analyze_lighting_quality(self, gray_image: np.ndarray) -> Dict:
        """조명 품질 분석"""
        # 조명 균등성 분석
        # 이미지를 여러 구역으로 나누어 밝기 분포 확인
        height, width = gray_image.shape
        
        # 4x4 그리드로 나누기
        lighting_map = []
        grid_height = height // 4
        grid_width = width // 4
        
        for i in range(4):
            for j in range(4):
                y1 = i * grid_height
                y2 = min((i + 1) * grid_height, height)
                x1 = j * grid_width
                x2 = min((j + 1) * grid_width, width)
                
                region = gray_image[y1:y2, x1:x2]
                region_brightness = float(np.mean(region))
                lighting_map.append(region_brightness)
        
        # 조명 균등성 (표준편차가 낮을수록 균등)
        lighting_uniformity = 1.0 / (1.0 + np.std(lighting_map) / np.mean(lighting_map))
        
        # 전체 이미지의 그라디언트 분석 (급격한 명암 변화)
        gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        lighting_gradient = float(np.mean(np.sqrt(gradient_x**2 + gradient_y**2)))
        
        return {
            'uniformity': round(lighting_uniformity, 3),
            'gradient_score': round(lighting_gradient, 1),
            'quality': 'excellent' if lighting_uniformity > 0.8 else 'good' if lighting_uniformity > 0.6 else 'poor'
        }

    def analyze_macro_quality(self, gray_image: np.ndarray) -> Dict:
        """매크로 촬영 품질 분석"""
        # 중앙 영역의 선명도 vs 가장자리 영역의 선명도 비교
        height, width = gray_image.shape
        
        # 중앙 50% 영역
        center_h1, center_h2 = height // 4, 3 * height // 4
        center_w1, center_w2 = width // 4, 3 * width // 4
        center_region = gray_image[center_h1:center_h2, center_w1:center_w2]
        center_sharpness = cv2.Laplacian(center_region, cv2.CV_64F).var()
        
        # 가장자리 영역들
        edge_regions = [
            gray_image[:height//8, :],                    # 상단
            gray_image[-height//8:, :],                   # 하단
            gray_image[:, :width//8],                     # 좌측
            gray_image[:, -width//8:]                     # 우측
        ]
        
        edge_sharpness = []
        for region in edge_regions:
            if region.size > 0:
                sharpness = cv2.Laplacian(region, cv2.CV_64F).var()
                edge_sharpness.append(sharpness)
        
        avg_edge_sharpness = np.mean(edge_sharpness) if edge_sharpness else 0
        
        # 매크로 품질 비율 (중앙이 가장자리보다 선명해야 함)
        if avg_edge_sharpness > 0:
            macro_ratio = center_sharpness / avg_edge_sharpness
        else:
            macro_ratio = center_sharpness / 100
        
        return {
            'center_sharpness': round(center_sharpness, 1),
            'edge_sharpness': round(avg_edge_sharpness, 1),
            'macro_ratio': round(macro_ratio, 2),
            'quality': 'excellent' if macro_ratio > 2.0 else 'good' if macro_ratio > 1.5 else 'poor'
        }

    def calculate_jewelry_suitability(self, background: Dict, reflections: Dict, 
                                    detail: Dict, lighting: Dict, macro: Dict) -> Dict:
        """주얼리 촬영 적합성 점수 계산"""
        # 각 요소별 점수 (0-1)
        background_score = 1.0 if background.get('suitability', False) else 0.3
        background_score *= background.get('uniformity', 0)
        
        reflection_score = 1.0 if reflections.get('reflection_quality') == 'good' else 0.3
        
        detail_score = {
            'excellent': 1.0,
            'good': 0.8,
            'poor': 0.3
        }.get(detail.get('detail_quality', 'poor'), 0.3)
        
        lighting_score = {
            'excellent': 1.0,
            'good': 0.8,
            'poor': 0.3
        }.get(lighting.get('quality', 'poor'), 0.3)
        
        macro_score = {
            'excellent': 1.0,
            'good': 0.8,
            'poor': 0.3
        }.get(macro.get('quality', 'poor'), 0.3)
        
        # 가중 평균
        weights = {
            'background': 0.25,
            'reflection': 0.20,
            'detail': 0.25,
            'lighting': 0.20,
            'macro': 0.10
        }
        
        overall_score = (
            background_score * weights['background'] +
            reflection_score * weights['reflection'] +
            detail_score * weights['detail'] +
            lighting_score * weights['lighting'] +
            macro_score * weights['macro']
        )
        
        # 등급 분류
        if overall_score >= 0.9:
            level = 'excellent'
            status = '최적'
        elif overall_score >= 0.75:
            level = 'good'
            status = '양호'
        elif overall_score >= 0.6:
            level = 'fair'
            status = '보통'
        else:
            level = 'poor'
            status = '불량'
        
        return {
            'score': round(overall_score, 3),
            'percentage': round(overall_score * 100, 1),
            'level': level,
            'status': status,
            'components': {
                'background': round(background_score, 3),
                'reflection': round(reflection_score, 3),
                'detail': round(detail_score, 3),
                'lighting': round(lighting_score, 3),
                'macro': round(macro_score, 3)
            }
        }

    def calculate_overall_image_score(self, results: Dict) -> Dict:
        """전체 이미지 품질 점수 계산"""
        try:
            # 기본 요소 점수
            resolution_score = self._normalize_resolution_score(results.get('resolution', {}))
            sharpness_score = self._normalize_sharpness_score(results.get('sharpness', {}).get('score', 0))
            contrast_score = self._normalize_contrast_score(results.get('contrast', {}).get('score', 0))
            brightness_score = self._normalize_brightness_score(results.get('brightness', {}))
            noise_score = self._normalize_noise_score(results.get('noise', {}))
            
            # 기본 점수 가중 평균
            basic_weights = {
                'resolution': 0.2,
                'sharpness': 0.25,
                'contrast': 0.2,
                'brightness': 0.2,
                'noise': 0.15
            }
            
            basic_score = (
                resolution_score * basic_weights['resolution'] +
                sharpness_score * basic_weights['sharpness'] +
                contrast_score * basic_weights['contrast'] +
                brightness_score * basic_weights['brightness'] +
                noise_score * basic_weights['noise']
            )
            
            # 주얼리 특화 점수 (있는 경우)
            jewelry_analysis = results.get('jewelry_analysis', {})
            if jewelry_analysis:
                jewelry_score = jewelry_analysis.get('jewelry_suitability', {}).get('score', 0)
                # 기본 점수와 주얼리 점수를 7:3 비율로 결합
                overall_score = basic_score * 0.7 + jewelry_score * 0.3
            else:
                overall_score = basic_score
            
            # 등급 분류
            if overall_score >= 0.9:
                level, status, color = 'excellent', '우수', '🟢'
            elif overall_score >= 0.8:
                level, status, color = 'good', '양호', '🟡'
            elif overall_score >= 0.7:
                level, status, color = 'fair', '보통', '🟠'
            else:
                level, status, color = 'poor', '불량', '🔴'
            
            return {
                'score': round(overall_score, 3),
                'percentage': round(overall_score * 100, 1),
                'level': level,
                'status': status,
                'color': color,
                'components': {
                    'basic_score': round(basic_score, 3),
                    'resolution': round(resolution_score, 3),
                    'sharpness': round(sharpness_score, 3),
                    'contrast': round(contrast_score, 3),
                    'brightness': round(brightness_score, 3),
                    'noise': round(noise_score, 3)
                }
            }
            
        except Exception as e:
            self.logger.error(f"전체 이미지 점수 계산 오류: {str(e)}")
            return {
                'score': 0.0,
                'level': 'error',
                'status': '오류'
            }

    def generate_image_recommendations(self, results: Dict) -> List[Dict]:
        """이미지 품질 개선 권장사항 생성"""
        recommendations = []
        
        try:
            # 해상도 권장사항
            resolution = results.get('resolution', {})
            if resolution.get('level') == 'poor':
                recommendations.append({
                    'type': 'critical',
                    'icon': '📏',
                    'title': '해상도 부족',
                    'message': '더 높은 해상도로 촬영하거나 카메라를 피사체에 더 가까이 하세요',
                    'action': 'increase_resolution'
                })
            
            # 선명도 권장사항
            sharpness = results.get('sharpness', {})
            if sharpness.get('level') in ['poor', 'fair']:
                recommendations.append({
                    'type': 'warning',
                    'icon': '🔍',
                    'title': '이미지 흐림',
                    'message': '삼각대를 사용하고 초점을 정확히 맞춘 후 촬영하세요',
                    'action': 'improve_focus_and_stability'
                })
            
            # 대비 권장사항
            contrast = results.get('contrast', {})
            if contrast.get('level') in ['poor', 'fair']:
                recommendations.append({
                    'type': 'warning',
                    'icon': '🌓',
                    'title': '대비 부족',
                    'message': '조명을 개선하거나 후처리에서 대비를 높여보세요',
                    'action': 'improve_contrast'
                })
            
            # 밝기 권장사항
            brightness = results.get('brightness', {})
            brightness_level = brightness.get('level')
            if brightness_level == 'too_dark':
                recommendations.append({
                    'type': 'warning',
                    'icon': '💡',
                    'title': '이미지 너무 어두움',
                    'message': '조명을 추가하거나 노출을 증가시키세요',
                    'action': 'increase_lighting'
                })
            elif brightness_level == 'too_bright':
                recommendations.append({
                    'type': 'warning',
                    'icon': '☀️',
                    'title': '이미지 너무 밝음',
                    'message': '조명을 줄이거나 노출을 감소시키세요',
                    'action': 'reduce_lighting'
                })
            
            # 노이즈 권장사항
            noise = results.get('noise', {})
            if noise.get('grade') in ['poor', 'fair']:
                recommendations.append({
                    'type': 'info',
                    'icon': '🔧',
                    'title': '노이즈 감지',
                    'message': 'ISO를 낮추거나 더 밝은 환경에서 촬영하세요',
                    'action': 'reduce_noise'
                })
            
            # 주얼리 특화 권장사항
            jewelry_analysis = results.get('jewelry_analysis', {})
            if jewelry_analysis:
                jewelry_suitability = jewelry_analysis.get('jewelry_suitability', {})
                if jewelry_suitability.get('level') in ['poor', 'fair']:
                    recommendations.append({
                        'type': 'info',
                        'icon': '💎',
                        'title': '주얼리 촬영 최적화 필요',
                        'message': '배경, 조명, 반사를 개선하여 주얼리가 더 잘 보이도록 하세요',
                        'action': 'optimize_jewelry_photography'
                    })
                
                # 배경 권장사항
                background = jewelry_analysis.get('background', {})
                if not background.get('suitability', True):
                    recommendations.append({
                        'type': 'info',
                        'icon': '🎭',
                        'title': '배경 개선 권장',
                        'message': '흰색, 검은색 또는 중성색 배경을 사용하세요',
                        'action': 'improve_background'
                    })
                
                # 반사 권장사항
                reflections = jewelry_analysis.get('reflections', {})
                if reflections.get('has_excessive_reflections', False):
                    recommendations.append({
                        'type': 'info',
                        'icon': '✨',
                        'title': '과도한 반사 감지',
                        'message': '조명 각도를 조정하거나 확산 조명을 사용하세요',
                        'action': 'reduce_reflections'
                    })
            
            # 전체 품질 기반 권장사항
            overall_quality = results.get('overall_quality', {})
            if overall_quality.get('level') == 'poor':
                recommendations.append({
                    'type': 'critical',
                    'icon': '🔴',
                    'title': '재촬영 권장',
                    'message': '현재 이미지 품질이 좋지 않습니다. 설정을 개선한 후 다시 촬영하세요',
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
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"이미지 권장사항 생성 오류: {str(e)}")
            return [{
                'type': 'error',
                'icon': '❌',
                'title': '이미지 분석 오류',
                'message': '이미지 품질 분석 중 오류가 발생했습니다',
                'action': 'retry_image_analysis'
            }]

    # === 내부 유틸리티 함수들 ===
    
    def _classify_sharpness(self, sharpness_score: float) -> str:
        """선명도 등급 분류"""
        if sharpness_score >= self.quality_thresholds['sharpness_excellent']:
            return 'excellent'
        elif sharpness_score >= self.quality_thresholds['sharpness_good']:
            return 'good'
        elif sharpness_score >= self.quality_thresholds['sharpness_fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _classify_contrast(self, contrast_score: float) -> str:
        """대비 등급 분류"""
        if contrast_score >= self.quality_thresholds['contrast_excellent']:
            return 'excellent'
        elif contrast_score >= self.quality_thresholds['contrast_good']:
            return 'good'
        elif contrast_score >= self.quality_thresholds['contrast_fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _calculate_high_frequency_noise(self, image: np.ndarray) -> float:
        """고주파 노이즈 계산"""
        # 고주파 통과 필터
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        high_pass = cv2.filter2D(image, -1, kernel)
        return float(np.std(high_pass))
    
    def _calculate_texture_variance(self, image: np.ndarray) -> float:
        """텍스처 분산 계산"""
        # 3x3 윈도우에서 각 픽셀의 분산 계산
        kernel = np.ones((3,3), np.float32) / 9
        mean_filtered = cv2.filter2D(image.astype(np.float32), -1, kernel)
        variance = np.mean((image.astype(np.float32) - mean_filtered) ** 2)
        return float(variance)
    
    def _calculate_texture_energy(self, image: np.ndarray) -> float:
        """텍스처 에너지 계산"""
        # 간단한 GLCM 기반 에너지 계산
        # 픽셀 값 차이의 제곱합
        diff_h = np.diff(image, axis=1)
        diff_v = np.diff(image, axis=0)
        energy = np.mean(diff_h**2) + np.mean(diff_v**2)
        return float(energy) / 1000  # 정규화
    
    def _normalize_resolution_score(self, resolution: Dict) -> float:
        """해상도 점수 정규화"""
        level = resolution.get('level', 'poor')
        score_map = {
            'excellent': 1.0,
            'good': 0.8,
            'fair': 0.6,
            'poor': 0.3
        }
        return score_map.get(level, 0.3)
    
    def _normalize_sharpness_score(self, sharpness: float) -> float:
        """선명도 점수 정규화"""
        return min(1.0, sharpness / 300)
    
    def _normalize_contrast_score(self, contrast: float) -> float:
        """대비 점수 정규화"""
        return min(1.0, contrast / 120)
    
    def _normalize_brightness_score(self, brightness: Dict) -> float:
        """밝기 점수 정규화"""
        level = brightness.get('level', 'too_dark')
        score_map = {
            'optimal': 1.0,
            'dark': 0.7,
            'bright': 0.7,
            'too_dark': 0.3,
            'too_bright': 0.3
        }
        return score_map.get(level, 0.3)
    
    def _normalize_noise_score(self, noise: Dict) -> float:
        """노이즈 점수 정규화 (노이즈가 적을수록 높은 점수)"""
        grade = noise.get('grade', 'poor')
        score_map = {
            'excellent': 1.0,
            'good': 0.8,
            'fair': 0.6,
            'poor': 0.3
        }
        return score_map.get(grade, 0.3)
    
    def _get_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 사용 예제
if __name__ == "__main__":
    assessor = ImageQualityAssessor()
    
    print("📸 Image Quality Assessor v2.1 - 테스트 시작")
    print("=" * 50)
    
    # 실제 사용 시에는 이미지 파일 경로를 제공
    # result = assessor.assess_image_quality("test_jewelry.jpg", assessment_type="jewelry_focused")
    # print(f"전체 이미지 품질: {result['overall_quality']['percentage']}%")
    # print(f"주얼리 적합성: {result['jewelry_analysis']['jewelry_suitability']['percentage']}%")
    
    print("모듈 로드 완료 ✅")
