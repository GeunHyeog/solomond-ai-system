"""
🔬 Solomond AI v2.1 - 품질 검증 엔진
음성 노이즈, OCR 품질, 이미지 품질을 실시간으로 분석하고 검증하는 시스템

Author: 전근혁 (Solomond)
Created: 2025.07.11
Version: 2.1.0
"""

import numpy as np
import cv2
import librosa
import pytesseract
from PIL import Image
import easyocr
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import time

@dataclass
class QualityScore:
    """품질 점수 데이터 클래스"""
    overall_score: float  # 전체 품질 점수 (0-100)
    audio_score: float    # 음성 품질 점수
    ocr_score: float      # OCR 품질 점수
    image_score: float    # 이미지 품질 점수
    details: Dict[str, Any]  # 세부 분석 결과
    recommendations: List[str]  # 개선 권장사항
    timestamp: float

class AudioQualityChecker:
    """음성 품질 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_audio_quality(self, audio_path: str) -> Dict[str, Any]:
        """음성 파일의 품질을 종합 분석"""
        try:
            # 음성 로드
            y, sr = librosa.load(audio_path, sr=None)
            
            # 1. SNR (Signal-to-Noise Ratio) 계산
            snr_db = self.calculate_snr(y, sr)
            
            # 2. 음성 명료도 분석
            clarity_score = self.analyze_speech_clarity(y, sr)
            
            # 3. 배경 노이즈 레벨 측정
            noise_level = self.measure_background_noise(y, sr)
            
            # 4. 음성 연속성 분석
            continuity_score = self.analyze_speech_continuity(y, sr)
            
            # 5. 주파수 분포 분석
            freq_quality = self.analyze_frequency_distribution(y, sr)
            
            # 종합 점수 계산
            overall_score = self.calculate_audio_overall_score({
                'snr': snr_db,
                'clarity': clarity_score,
                'noise_level': noise_level,
                'continuity': continuity_score,
                'frequency': freq_quality
            })
            
            return {
                'snr_db': snr_db,
                'clarity_score': clarity_score,
                'noise_level': noise_level,
                'continuity_score': continuity_score,
                'frequency_quality': freq_quality,
                'overall_score': overall_score,
                'duration': len(y) / sr,
                'sample_rate': sr,
                'recommendations': self.generate_audio_recommendations(overall_score, snr_db, noise_level)
            }
            
        except Exception as e:
            self.logger.error(f"음성 품질 분석 실패: {e}")
            return {
                'error': str(e),
                'overall_score': 0,
                'recommendations': ['음성 파일을 확인하고 다시 업로드해주세요.']
            }
    
    def calculate_snr(self, y: np.ndarray, sr: int) -> float:
        """SNR (Signal-to-Noise Ratio) 계산"""
        try:
            # RMS 에너지 계산
            rms_energy = librosa.feature.rms(y=y)[0]
            
            # 음성 구간과 무음 구간 분리
            intervals = librosa.effects.split(y, top_db=20)
            
            if len(intervals) == 0:
                return 0.0
            
            # 음성 구간의 평균 에너지
            signal_energy = []
            for start, end in intervals:
                signal_energy.extend(rms_energy[start//512:end//512])
            
            # 전체 구간에서 음성 구간 제외한 노이즈 구간
            noise_energy = []
            last_end = 0
            for start, end in intervals:
                if start > last_end:
                    noise_energy.extend(rms_energy[last_end//512:start//512])
                last_end = end
            
            if len(noise_energy) == 0:
                return 30.0  # 노이즈가 없다면 높은 SNR
            
            signal_power = np.mean(signal_energy) ** 2
            noise_power = np.mean(noise_energy) ** 2
            
            if noise_power == 0:
                return 30.0
            
            snr = 10 * np.log10(signal_power / noise_power)
            return max(0, min(40, snr))  # 0-40dB 범위로 제한
            
        except Exception as e:
            self.logger.error(f"SNR 계산 실패: {e}")
            return 0.0
    
    def analyze_speech_clarity(self, y: np.ndarray, sr: int) -> float:
        """음성 명료도 분석 (0-100점)"""
        try:
            # 1. 스펙트럴 센트로이드 (음성의 밝기)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            centroid_score = min(100, np.mean(spectral_centroids) / 30)
            
            # 2. 영교차율 (음성의 안정성)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_score = min(100, (1 - np.std(zcr)) * 100)
            
            # 3. 스펙트럴 대역폭 (음성의 선명도)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            bandwidth_score = min(100, np.mean(spectral_bandwidth) / 50)
            
            # 가중 평균으로 명료도 점수 계산
            clarity_score = (centroid_score * 0.4 + zcr_score * 0.3 + bandwidth_score * 0.3)
            return max(0, min(100, clarity_score))
            
        except Exception as e:
            self.logger.error(f"음성 명료도 분석 실패: {e}")
            return 50.0
    
    def measure_background_noise(self, y: np.ndarray, sr: int) -> float:
        """배경 노이즈 레벨 측정 (낮을수록 좋음, 0-100)"""
        try:
            # 무음 구간 감지
            intervals = librosa.effects.split(y, top_db=20)
            
            # 무음 구간의 RMS 에너지 계산
            noise_segments = []
            last_end = 0
            
            for start, end in intervals:
                if start > last_end + sr * 0.5:  # 0.5초 이상의 무음 구간
                    noise_segment = y[last_end:start]
                    if len(noise_segment) > sr * 0.1:  # 0.1초 이상
                        noise_segments.append(noise_segment)
                last_end = end
            
            if not noise_segments:
                return 10.0  # 무음 구간이 없으면 낮은 노이즈로 가정
            
            # 노이즈 레벨 계산
            all_noise = np.concatenate(noise_segments)
            noise_rms = np.sqrt(np.mean(all_noise ** 2))
            
            # 0-100 스케일로 변환 (낮을수록 좋음)
            noise_level = min(100, noise_rms * 1000)
            return noise_level
            
        except Exception as e:
            self.logger.error(f"배경 노이즈 측정 실패: {e}")
            return 50.0
    
    def analyze_speech_continuity(self, y: np.ndarray, sr: int) -> float:
        """음성 연속성 분석 (0-100점)"""
        try:
            # 음성 구간 감지
            intervals = librosa.effects.split(y, top_db=20)
            
            if len(intervals) == 0:
                return 0.0
            
            # 음성 구간 길이와 간격 분석
            speech_lengths = []
            silence_lengths = []
            
            for i, (start, end) in enumerate(intervals):
                speech_lengths.append((end - start) / sr)
                
                if i > 0:
                    prev_end = intervals[i-1][1]
                    silence_lengths.append((start - prev_end) / sr)
            
            # 연속성 점수 계산
            avg_speech_length = np.mean(speech_lengths)
            avg_silence_length = np.mean(silence_lengths) if silence_lengths else 0
            
            # 적절한 발화 길이와 침묵 길이 기준
            speech_score = min(100, avg_speech_length * 20)  # 5초가 만점
            silence_score = max(0, 100 - avg_silence_length * 50)  # 2초 이상 침묵 시 감점
            
            continuity_score = (speech_score * 0.7 + silence_score * 0.3)
            return max(0, min(100, continuity_score))
            
        except Exception as e:
            self.logger.error(f"음성 연속성 분석 실패: {e}")
            return 50.0
    
    def analyze_frequency_distribution(self, y: np.ndarray, sr: int) -> float:
        """주파수 분포 품질 분석 (0-100점)"""
        try:
            # STFT 계산
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            
            # 주파수 대역별 에너지 분석
            freq_bins = librosa.fft_frequencies(sr=sr)
            
            # 음성 주요 주파수 대역 (80Hz-8000Hz)
            speech_mask = (freq_bins >= 80) & (freq_bins <= 8000)
            speech_energy = np.mean(magnitude[speech_mask])
            
            # 전체 에너지 대비 음성 대역 에너지 비율
            total_energy = np.mean(magnitude)
            
            if total_energy == 0:
                return 0.0
            
            freq_quality = (speech_energy / total_energy) * 100
            return max(0, min(100, freq_quality))
            
        except Exception as e:
            self.logger.error(f"주파수 분포 분석 실패: {e}")
            return 50.0
    
    def calculate_audio_overall_score(self, metrics: Dict[str, float]) -> float:
        """음성 품질 종합 점수 계산"""
        try:
            # SNR 점수 (20dB 이상이 우수)
            snr_score = min(100, max(0, metrics['snr'] * 5))
            
            # 각 지표별 가중치 적용
            weights = {
                'snr': 0.3,
                'clarity': 0.25,
                'noise_level': 0.2,  # 역산 적용 (낮을수록 좋음)
                'continuity': 0.15,
                'frequency': 0.1
            }
            
            # 노이즈 레벨은 역산 적용
            noise_score = 100 - metrics['noise_level']
            
            overall_score = (
                snr_score * weights['snr'] +
                metrics['clarity'] * weights['clarity'] +
                noise_score * weights['noise_level'] +
                metrics['continuity'] * weights['continuity'] +
                metrics['frequency'] * weights['frequency']
            )
            
            return max(0, min(100, overall_score))
            
        except Exception as e:
            self.logger.error(f"종합 점수 계산 실패: {e}")
            return 50.0
    
    def generate_audio_recommendations(self, overall_score: float, snr_db: float, noise_level: float) -> List[str]:
        """음성 품질 개선 권장사항 생성"""
        recommendations = []
        
        if overall_score < 60:
            recommendations.append("⚠️ 음성 품질이 낮습니다. 재녹음을 권장합니다.")
        
        if snr_db < 15:
            recommendations.append("🔊 배경 소음이 많습니다. 조용한 장소에서 녹음해주세요.")
        
        if noise_level > 70:
            recommendations.append("🎙️ 마이크를 입에 더 가까이 가져가세요.")
        
        if overall_score >= 85:
            recommendations.append("✅ 음성 품질이 우수합니다!")
        elif overall_score >= 70:
            recommendations.append("👍 음성 품질이 양호합니다.")
        
        return recommendations

class OCRQualityValidator:
    """OCR 품질 검증기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.easy_reader = easyocr.Reader(['ko', 'en', 'ch_sim'])
        
    def analyze_ocr_quality(self, image_path: str) -> Dict[str, Any]:
        """이미지의 OCR 품질을 종합 분석"""
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("이미지 파일을 읽을 수 없습니다.")
            
            # 1. 이미지 전처리 품질 분석
            preprocessing_score = self.analyze_preprocessing_quality(image)
            
            # 2. 텍스트 감지 품질
            detection_score = self.analyze_text_detection_quality(image)
            
            # 3. 문자 인식 정확도
            recognition_score = self.analyze_character_recognition(image)
            
            # 4. 레이아웃 분석 품질
            layout_score = self.analyze_layout_quality(image)
            
            # 5. 다중 OCR 엔진 비교
            comparison_score = self.compare_ocr_engines(image)
            
            # 종합 점수 계산
            overall_score = self.calculate_ocr_overall_score({
                'preprocessing': preprocessing_score,
                'detection': detection_score,
                'recognition': recognition_score,
                'layout': layout_score,
                'comparison': comparison_score
            })
            
            return {
                'preprocessing_score': preprocessing_score,
                'detection_score': detection_score,
                'recognition_score': recognition_score,
                'layout_score': layout_score,
                'comparison_score': comparison_score,
                'overall_score': overall_score,
                'image_dimensions': image.shape[:2],
                'recommendations': self.generate_ocr_recommendations(overall_score, preprocessing_score, detection_score)
            }
            
        except Exception as e:
            self.logger.error(f"OCR 품질 분석 실패: {e}")
            return {
                'error': str(e),
                'overall_score': 0,
                'recommendations': ['이미지 파일을 확인하고 다시 업로드해주세요.']
            }
    
    def analyze_preprocessing_quality(self, image: np.ndarray) -> float:
        """이미지 전처리 품질 분석"""
        try:
            # 1. 해상도 품질
            height, width = image.shape[:2]
            resolution_score = min(100, (width * height) / 10000)  # 100x100 = 1점
            
            # 2. 선명도 (라플라시안 분산)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(100, sharpness / 500)
            
            # 3. 대비 (히스토그램 분산)
            contrast = np.std(gray)
            contrast_score = min(100, contrast / 60)
            
            # 4. 기울기 감지
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            tilt_score = 100
            if lines is not None:
                angles = []
                for line in lines[:10]:  # 상위 10개 선만 확인
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi
                    angles.append(angle)
                
                if angles:
                    angle_variance = np.var(angles)
                    tilt_score = max(0, 100 - angle_variance)
            
            # 가중 평균
            preprocessing_score = (
                resolution_score * 0.3 +
                sharpness_score * 0.3 +
                contrast_score * 0.2 +
                tilt_score * 0.2
            )
            
            return max(0, min(100, preprocessing_score))
            
        except Exception as e:
            self.logger.error(f"전처리 품질 분석 실패: {e}")
            return 50.0
    
    def analyze_text_detection_quality(self, image: np.ndarray) -> float:
        """텍스트 감지 품질 분석"""
        try:
            # EasyOCR로 텍스트 영역 감지
            results = self.easy_reader.readtext(image)
            
            if not results:
                return 0.0
            
            # 1. 감지된 텍스트 영역 수
            detection_count_score = min(100, len(results) * 10)
            
            # 2. 신뢰도 평균
            confidences = [result[2] for result in results]
            avg_confidence = np.mean(confidences) if confidences else 0
            confidence_score = avg_confidence * 100
            
            # 3. 텍스트 영역 크기 일관성
            areas = []
            for result in results:
                bbox = result[0]
                width = max(point[0] for point in bbox) - min(point[0] for point in bbox)
                height = max(point[1] for point in bbox) - min(point[1] for point in bbox)
                areas.append(width * height)
            
            area_consistency = 100 - (np.std(areas) / np.mean(areas) * 100) if areas else 0
            area_consistency = max(0, min(100, area_consistency))
            
            # 종합 점수
            detection_score = (
                detection_count_score * 0.3 +
                confidence_score * 0.5 +
                area_consistency * 0.2
            )
            
            return max(0, min(100, detection_score))
            
        except Exception as e:
            self.logger.error(f"텍스트 감지 품질 분석 실패: {e}")
            return 50.0
    
    def analyze_character_recognition(self, image: np.ndarray) -> float:
        """문자 인식 정확도 분석"""
        try:
            # 1. EasyOCR 결과
            easy_results = self.easy_reader.readtext(image)
            
            # 2. Tesseract 결과
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            tess_text = pytesseract.image_to_string(gray, lang='kor+eng')
            
            # 3. 인식 일관성 검사
            easy_text = ' '.join([result[1] for result in easy_results])
            
            # 문자 수 비교
            easy_char_count = len(easy_text.strip())
            tess_char_count = len(tess_text.strip())
            
            if easy_char_count == 0 and tess_char_count == 0:
                return 0.0
            
            # 공통 문자 비율
            common_chars = set(easy_text.lower()) & set(tess_text.lower())
            total_chars = set(easy_text.lower()) | set(tess_text.lower())
            
            similarity_score = len(common_chars) / len(total_chars) * 100 if total_chars else 0
            
            # 신뢰도 기반 점수
            confidence_score = np.mean([result[2] for result in easy_results]) * 100 if easy_results else 0
            
            # 종합 점수
            recognition_score = (similarity_score * 0.6 + confidence_score * 0.4)
            
            return max(0, min(100, recognition_score))
            
        except Exception as e:
            self.logger.error(f"문자 인식 정확도 분석 실패: {e}")
            return 50.0
    
    def analyze_layout_quality(self, image: np.ndarray) -> float:
        """레이아웃 분석 품질"""
        try:
            # EasyOCR로 텍스트 영역 정보 획득
            results = self.easy_reader.readtext(image)
            
            if len(results) < 2:
                return 50.0  # 텍스트가 너무 적으면 중간 점수
            
            # 1. 텍스트 정렬 분석
            y_positions = []
            for result in results:
                bbox = result[0]
                y_center = sum(point[1] for point in bbox) / len(bbox)
                y_positions.append(y_center)
            
            # Y 좌표 기준 정렬 품질
            y_variance = np.var(y_positions)
            alignment_score = max(0, 100 - y_variance / 100)
            
            # 2. 텍스트 간격 일관성
            y_sorted = sorted(y_positions)
            gaps = [y_sorted[i+1] - y_sorted[i] for i in range(len(y_sorted)-1)]
            
            if gaps:
                gap_consistency = 100 - (np.std(gaps) / np.mean(gaps) * 100)
                gap_consistency = max(0, min(100, gap_consistency))
            else:
                gap_consistency = 50.0
            
            # 3. 텍스트 밀도
            image_area = image.shape[0] * image.shape[1]
            text_area = 0
            
            for result in results:
                bbox = result[0]
                width = max(point[0] for point in bbox) - min(point[0] for point in bbox)
                height = max(point[1] for point in bbox) - min(point[1] for point in bbox)
                text_area += width * height
            
            density_score = min(100, (text_area / image_area) * 500)
            
            # 종합 레이아웃 점수
            layout_score = (
                alignment_score * 0.4 +
                gap_consistency * 0.3 +
                density_score * 0.3
            )
            
            return max(0, min(100, layout_score))
            
        except Exception as e:
            self.logger.error(f"레이아웃 품질 분석 실패: {e}")
            return 50.0
    
    def compare_ocr_engines(self, image: np.ndarray) -> float:
        """다중 OCR 엔진 비교 분석"""
        try:
            # 1. EasyOCR 결과
            easy_results = self.easy_reader.readtext(image)
            easy_text = ' '.join([result[1] for result in easy_results])
            easy_confidence = np.mean([result[2] for result in easy_results]) if easy_results else 0
            
            # 2. Tesseract 결과
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Tesseract 신뢰도 포함 결과
            tess_data = pytesseract.image_to_data(gray, lang='kor+eng', output_type=pytesseract.Output.DICT)
            tess_confidences = [int(conf) for conf in tess_data['conf'] if int(conf) > 0]
            tess_confidence = np.mean(tess_confidences) / 100 if tess_confidences else 0
            
            # 3. 엔진 간 일치도
            tess_text = pytesseract.image_to_string(gray, lang='kor+eng')
            
            # 텍스트 유사도 계산 (간단한 문자 기반)
            easy_chars = set(easy_text.lower().replace(' ', ''))
            tess_chars = set(tess_text.lower().replace(' ', ''))
            
            if not easy_chars and not tess_chars:
                return 0.0
            
            intersection = easy_chars & tess_chars
            union = easy_chars | tess_chars
            
            similarity = len(intersection) / len(union) if union else 0
            
            # 종합 비교 점수
            comparison_score = (
                easy_confidence * 100 * 0.3 +
                tess_confidence * 100 * 0.3 +
                similarity * 100 * 0.4
            )
            
            return max(0, min(100, comparison_score))
            
        except Exception as e:
            self.logger.error(f"OCR 엔진 비교 실패: {e}")
            return 50.0
    
    def calculate_ocr_overall_score(self, metrics: Dict[str, float]) -> float:
        """OCR 품질 종합 점수 계산"""
        try:
            weights = {
                'preprocessing': 0.2,
                'detection': 0.25,
                'recognition': 0.3,
                'layout': 0.15,
                'comparison': 0.1
            }
            
            overall_score = sum(metrics[key] * weights[key] for key in weights.keys())
            return max(0, min(100, overall_score))
            
        except Exception as e:
            self.logger.error(f"OCR 종합 점수 계산 실패: {e}")
            return 50.0
    
    def generate_ocr_recommendations(self, overall_score: float, preprocessing_score: float, detection_score: float) -> List[str]:
        """OCR 품질 개선 권장사항 생성"""
        recommendations = []
        
        if overall_score < 60:
            recommendations.append("⚠️ OCR 품질이 낮습니다. 이미지를 다시 촬영해주세요.")
        
        if preprocessing_score < 50:
            recommendations.append("📸 이미지가 흐릿합니다. 초점을 맞춰서 다시 촬영해주세요.")
        
        if detection_score < 50:
            recommendations.append("🔍 텍스트 감지가 어렵습니다. 조명을 밝게 하고 정면에서 촬영해주세요.")
        
        if preprocessing_score < 60:
            recommendations.append("📐 이미지가 기울어져 있습니다. 수평으로 맞춰서 촬영해주세요.")
        
        if overall_score >= 85:
            recommendations.append("✅ OCR 품질이 우수합니다!")
        elif overall_score >= 70:
            recommendations.append("👍 OCR 품질이 양호합니다.")
        
        return recommendations

class ImageQualityAssessor:
    """이미지 품질 평가기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_image_quality(self, image_path: str) -> Dict[str, Any]:
        """이미지 품질 종합 분석"""
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("이미지 파일을 읽을 수 없습니다.")
            
            # 1. 해상도 품질
            resolution_score = self.analyze_resolution(image)
            
            # 2. 선명도 분석
            sharpness_score = self.analyze_sharpness(image)
            
            # 3. 밝기 및 대비
            brightness_score = self.analyze_brightness_contrast(image)
            
            # 4. 컬러 품질
            color_score = self.analyze_color_quality(image)
            
            # 5. 노이즈 레벨
            noise_score = self.analyze_noise_level(image)
            
            # 종합 점수 계산
            overall_score = self.calculate_image_overall_score({
                'resolution': resolution_score,
                'sharpness': sharpness_score,
                'brightness': brightness_score,
                'color': color_score,
                'noise': noise_score
            })
            
            return {
                'resolution_score': resolution_score,
                'sharpness_score': sharpness_score,
                'brightness_score': brightness_score,
                'color_score': color_score,
                'noise_score': noise_score,
                'overall_score': overall_score,
                'image_dimensions': image.shape,
                'file_size': Path(image_path).stat().st_size,
                'recommendations': self.generate_image_recommendations(overall_score, sharpness_score, brightness_score)
            }
            
        except Exception as e:
            self.logger.error(f"이미지 품질 분석 실패: {e}")
            return {
                'error': str(e),
                'overall_score': 0,
                'recommendations': ['이미지 파일을 확인하고 다시 업로드해주세요.']
            }
    
    def analyze_resolution(self, image: np.ndarray) -> float:
        """해상도 품질 분석"""
        height, width = image.shape[:2]
        total_pixels = height * width
        
        # 권장 해상도 기준 (1920x1080 = 100점)
        target_pixels = 1920 * 1080
        resolution_score = min(100, (total_pixels / target_pixels) * 100)
        
        return resolution_score
    
    def analyze_sharpness(self, image: np.ndarray) -> float:
        """선명도 분석 (라플라시안 분산 기반)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 경험적 임계값 (500이 좋은 선명도)
        sharpness_score = min(100, laplacian_var / 500 * 100)
        
        return max(0, sharpness_score)
    
    def analyze_brightness_contrast(self, image: np.ndarray) -> float:
        """밝기 및 대비 분석"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 밝기 (평균값, 128이 이상적)
        mean_brightness = np.mean(gray)
        brightness_score = 100 - abs(mean_brightness - 128) / 128 * 100
        
        # 대비 (표준편차, 60 이상이 좋음)
        contrast = np.std(gray)
        contrast_score = min(100, contrast / 60 * 100)
        
        # 히스토그램 분포 분석
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist / np.sum(hist)
        
        # 동적 범위 (0-255 전체 사용 시 100점)
        non_zero_bins = np.count_nonzero(hist_normalized)
        dynamic_range_score = non_zero_bins / 256 * 100
        
        # 종합 점수
        overall_brightness_score = (
            brightness_score * 0.4 +
            contrast_score * 0.4 +
            dynamic_range_score * 0.2
        )
        
        return max(0, min(100, overall_brightness_score))
    
    def analyze_color_quality(self, image: np.ndarray) -> float:
        """컬러 품질 분석"""
        # HSV 색공간 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 채도 분석
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation)
        saturation_score = min(100, avg_saturation / 255 * 150)  # 채도는 약간 높은 게 좋음
        
        # 색상 분포 균등성
        hue = hsv[:, :, 0]
        hue_hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
        hue_normalized = hue_hist / np.sum(hue_hist)
        
        # 엔트로피 계산 (색상 다양성)
        entropy = -np.sum(hue_normalized * np.log2(hue_normalized + 1e-10))
        entropy_score = min(100, entropy / 7 * 100)  # 최대 엔트로피는 약 7
        
        # 컬러 캐스트 검사 (RGB 채널 균형)
        b, g, r = cv2.split(image)
        r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
        
        # RGB 균형도 (편차가 적을수록 좋음)
        rgb_std = np.std([r_mean, g_mean, b_mean])
        balance_score = max(0, 100 - rgb_std / 50 * 100)
        
        # 종합 컬러 점수
        color_score = (
            saturation_score * 0.4 +
            entropy_score * 0.3 +
            balance_score * 0.3
        )
        
        return max(0, min(100, color_score))
    
    def analyze_noise_level(self, image: np.ndarray) -> float:
        """노이즈 레벨 분석"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러 적용 후 차이 계산
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_map = cv2.absdiff(gray, blurred)
        
        # 노이즈 강도 계산
        noise_level = np.mean(noise_map)
        
        # 점수화 (낮은 노이즈가 높은 점수)
        noise_score = max(0, 100 - noise_level / 10 * 100)
        
        return noise_score
    
    def calculate_image_overall_score(self, metrics: Dict[str, float]) -> float:
        """이미지 품질 종합 점수 계산"""
        weights = {
            'resolution': 0.2,
            'sharpness': 0.3,
            'brightness': 0.25,
            'color': 0.15,
            'noise': 0.1
        }
        
        overall_score = sum(metrics[key] * weights[key] for key in weights.keys())
        return max(0, min(100, overall_score))
    
    def generate_image_recommendations(self, overall_score: float, sharpness_score: float, brightness_score: float) -> List[str]:
        """이미지 품질 개선 권장사항 생성"""
        recommendations = []
        
        if overall_score < 60:
            recommendations.append("⚠️ 이미지 품질이 낮습니다. 다시 촬영해주세요.")
        
        if sharpness_score < 50:
            recommendations.append("📷 이미지가 흐릿합니다. 초점을 맞춰서 촬영해주세요.")
        
        if brightness_score < 50:
            recommendations.append("💡 조명이 부족합니다. 밝은 곳에서 촬영해주세요.")
        
        if overall_score >= 85:
            recommendations.append("✅ 이미지 품질이 우수합니다!")
        elif overall_score >= 70:
            recommendations.append("👍 이미지 품질이 양호합니다.")
        
        return recommendations

class QualityAnalyzerV21:
    """v2.1 통합 품질 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.audio_checker = AudioQualityChecker()
        self.ocr_validator = OCRQualityValidator()
        self.image_assessor = ImageQualityAssessor()
    
    def analyze_file_quality(self, file_path: str, file_type: str) -> QualityScore:
        """파일 품질 종합 분석"""
        try:
            start_time = time.time()
            
            # 파일 타입별 분석
            if file_type.startswith('audio'):
                audio_result = self.audio_checker.analyze_audio_quality(file_path)
                result = QualityScore(
                    overall_score=audio_result.get('overall_score', 0),
                    audio_score=audio_result.get('overall_score', 0),
                    ocr_score=0,
                    image_score=0,
                    details=audio_result,
                    recommendations=audio_result.get('recommendations', []),
                    timestamp=start_time
                )
            
            elif file_type.startswith('image'):
                image_result = self.image_assessor.analyze_image_quality(file_path)
                ocr_result = self.ocr_validator.analyze_ocr_quality(file_path)
                
                # 이미지 + OCR 종합 점수
                combined_score = (image_result.get('overall_score', 0) * 0.6 + 
                                ocr_result.get('overall_score', 0) * 0.4)
                
                result = QualityScore(
                    overall_score=combined_score,
                    audio_score=0,
                    ocr_score=ocr_result.get('overall_score', 0),
                    image_score=image_result.get('overall_score', 0),
                    details={
                        'image_analysis': image_result,
                        'ocr_analysis': ocr_result
                    },
                    recommendations=(image_result.get('recommendations', []) + 
                                   ocr_result.get('recommendations', [])),
                    timestamp=start_time
                )
            
            else:
                # 기타 파일 타입 (문서 등)
                result = QualityScore(
                    overall_score=75,  # 기본 점수
                    audio_score=0,
                    ocr_score=0,
                    image_score=0,
                    details={'message': '지원되는 품질 분석 타입이 아닙니다.'},
                    recommendations=['파일이 정상적으로 처리될 예정입니다.'],
                    timestamp=start_time
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"파일 품질 분석 실패: {e}")
            return QualityScore(
                overall_score=0,
                audio_score=0,
                ocr_score=0,
                image_score=0,
                details={'error': str(e)},
                recommendations=['파일을 확인하고 다시 업로드해주세요.'],
                timestamp=time.time()
            )
    
    def analyze_batch_quality(self, file_paths: List[str]) -> Dict[str, Any]:
        """배치 파일 품질 분석"""
        try:
            results = {}
            overall_scores = []
            all_recommendations = []
            
            for file_path in file_paths:
                # 파일 타입 추정
                file_ext = Path(file_path).suffix.lower()
                if file_ext in ['.mp3', '.wav', '.m4a', '.aac']:
                    file_type = 'audio'
                elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    file_type = 'image'
                else:
                    file_type = 'document'
                
                quality_score = self.analyze_file_quality(file_path, file_type)
                results[file_path] = quality_score
                
                overall_scores.append(quality_score.overall_score)
                all_recommendations.extend(quality_score.recommendations)
            
            # 배치 통계
            batch_stats = {
                'total_files': len(file_paths),
                'average_quality': np.mean(overall_scores) if overall_scores else 0,
                'min_quality': min(overall_scores) if overall_scores else 0,
                'max_quality': max(overall_scores) if overall_scores else 0,
                'quality_std': np.std(overall_scores) if overall_scores else 0,
                'high_quality_count': sum(1 for score in overall_scores if score >= 80),
                'low_quality_count': sum(1 for score in overall_scores if score < 60),
                'recommendations': list(set(all_recommendations))  # 중복 제거
            }
            
            return {
                'individual_results': results,
                'batch_statistics': batch_stats,
                'processing_complete': True
            }
            
        except Exception as e:
            self.logger.error(f"배치 품질 분석 실패: {e}")
            return {
                'error': str(e),
                'processing_complete': False
            }
    
    def get_quality_report(self, analysis_results: Dict[str, Any]) -> str:
        """품질 분석 리포트 생성"""
        try:
            if 'error' in analysis_results:
                return f"❌ 품질 분석 실패: {analysis_results['error']}"
            
            stats = analysis_results['batch_statistics']
            
            # 품질 등급 결정
            avg_quality = stats['average_quality']
            if avg_quality >= 90:
                grade = "🏆 최우수"
                grade_color = "🟢"
            elif avg_quality >= 80:
                grade = "✅ 우수"
                grade_color = "🟢"
            elif avg_quality >= 70:
                grade = "👍 양호"
                grade_color = "🟡"
            elif avg_quality >= 60:
                grade = "⚠️ 보통"
                grade_color = "🟡"
            else:
                grade = "❌ 개선필요"
                grade_color = "🔴"
            
            report = f"""
📊 **품질 분석 리포트**

{grade_color} **종합 품질**: {grade} ({avg_quality:.1f}/100점)

📈 **통계 정보**
• 전체 파일: {stats['total_files']}개
• 평균 품질: {avg_quality:.1f}점
• 최고 품질: {stats['max_quality']:.1f}점
• 최저 품질: {stats['min_quality']:.1f}점
• 고품질 파일: {stats['high_quality_count']}개 (80점 이상)
• 개선필요 파일: {stats['low_quality_count']}개 (60점 미만)

💡 **개선 권장사항**
{chr(10).join('• ' + rec for rec in stats['recommendations'][:5])}
            """
            
            return report.strip()
            
        except Exception as e:
            self.logger.error(f"품질 리포트 생성 실패: {e}")
            return "품질 리포트 생성 중 오류가 발생했습니다."

# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 품질 분석기 초기화
    analyzer = QualityAnalyzerV21()
    
    # 샘플 파일 분석
    # quality_score = analyzer.analyze_file_quality("sample_audio.mp3", "audio")
    # print(f"품질 점수: {quality_score.overall_score}")
    # print(f"권장사항: {quality_score.recommendations}")
    
    print("✅ 품질 검증 엔진 v2.1 로드 완료!")
