"""
솔로몬드 AI 시스템 - 품질 분석 엔진
현장 녹화/사진의 품질 분석, 노이즈 검출, PPT OCR 품질 평가
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import json
import re
from pathlib import Path

# 음성 분석용
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# 이미지 분석용
try:
    from PIL import Image, ImageStat, ImageFilter
    import pytesseract
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class QualityAnalyzer:
    """품질 분석 및 개선 제안 클래스"""
    
    def __init__(self):
        # 품질 임계값 설정
        self.quality_thresholds = {
            "audio": {
                "noise_level": 0.3,     # 30% 이하가 양호
                "clarity_score": 0.7,   # 70% 이상이 양호
                "volume_consistency": 0.8  # 80% 이상이 양호
            },
            "image": {
                "blur_threshold": 100,   # 라플라시안 분산
                "brightness_range": (50, 200),  # 적정 밝기
                "contrast_min": 50,     # 최소 대비
                "text_confidence": 0.6  # OCR 신뢰도
            },
            "ppt": {
                "text_density": 0.1,    # 텍스트 밀도 최소값
                "geometric_score": 0.7, # 기하학적 구조 점수
                "color_contrast": 3.0   # 색상 대비 비율
            }
        }
        
        # PPT 특화 패턴
        self.ppt_patterns = [
            r'\d+\.',  # 번호 목록
            r'[▶▪►•]',  # 불릿 포인트
            r'제\s*\d+\s*장',  # 장 제목
            r'목\s*차|개\s*요|결\s*론',  # 일반적인 PPT 구조
            r'\d{4}년|\d{1,2}월|\d{1,2}일',  # 날짜
            r'[A-Z]{2,}',  # 대문자 약어
        ]
        
        logging.info("품질 분석 엔진 초기화 완료")
    
    async def analyze_audio_quality(self, 
                                  audio_data: bytes, 
                                  filename: str,
                                  sample_rate: int = 22050) -> Dict:
        """
        음성 품질 분석
        
        Args:
            audio_data: 음성 바이너리 데이터
            filename: 파일명
            sample_rate: 샘플링 레이트
            
        Returns:
            음성 품질 분석 결과
        """
        if not LIBROSA_AVAILABLE:
            return {
                "success": False,
                "error": "librosa가 필요합니다. pip install librosa로 설치하세요.",
                "filename": filename
            }
        
        try:
            print(f"🔊 음성 품질 분석 시작: {filename}")
            
            # 임시 파일로 저장 후 분석
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                # librosa로 음성 로드
                y, sr = librosa.load(temp_path, sr=sample_rate)
                
                # 1. 노이즈 레벨 분석
                noise_level = self._analyze_noise_level(y)
                
                # 2. 음성 명료도 분석
                clarity_score = self._analyze_speech_clarity(y, sr)
                
                # 3. 볼륨 일관성 분석
                volume_consistency = self._analyze_volume_consistency(y)
                
                # 4. 주파수 분석
                frequency_analysis = self._analyze_frequency_spectrum(y, sr)
                
                # 5. 전체 품질 점수 계산
                overall_quality = self._calculate_audio_quality_score(
                    noise_level, clarity_score, volume_consistency
                )
                
                # 6. 개선 제안 생성
                improvement_suggestions = self._generate_audio_improvements(
                    noise_level, clarity_score, volume_consistency
                )
                
                result = {
                    "success": True,
                    "filename": filename,
                    "quality_metrics": {
                        "noise_level": round(noise_level, 3),
                        "clarity_score": round(clarity_score, 3),
                        "volume_consistency": round(volume_consistency, 3),
                        "overall_quality": round(overall_quality, 3)
                    },
                    "frequency_analysis": frequency_analysis,
                    "quality_assessment": self._assess_audio_quality(overall_quality),
                    "improvement_suggestions": improvement_suggestions,
                    "analysis_time": datetime.now().isoformat()
                }
                
                print(f"✅ 음성 품질 분석 완료: {overall_quality:.1%} 품질")
                return result
                
            finally:
                # 임시 파일 정리
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logging.error(f"음성 품질 분석 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "filename": filename
            }
    
    def _analyze_noise_level(self, audio: np.ndarray) -> float:
        """노이즈 레벨 분석"""
        try:
            # RMS 에너지 계산
            rms = librosa.feature.rms(y=audio)[0]
            
            # 조용한 구간 (하위 10%) 식별
            quiet_threshold = np.percentile(rms, 10)
            quiet_segments = rms < quiet_threshold
            
            if np.any(quiet_segments):
                noise_floor = np.mean(rms[quiet_segments])
                signal_peak = np.max(rms)
                
                # SNR 계산 (신호 대 잡음비)
                if noise_floor > 0:
                    snr = signal_peak / noise_floor
                    noise_level = 1.0 / (1.0 + snr)  # 0~1 범위로 정규화
                else:
                    noise_level = 0.0
            else:
                noise_level = 0.5  # 기본값
            
            return min(noise_level, 1.0)
            
        except Exception as e:
            logging.warning(f"노이즈 분석 실패: {e}")
            return 0.5
    
    def _analyze_speech_clarity(self, audio: np.ndarray, sr: int) -> float:
        """음성 명료도 분석"""
        try:
            # 스펙트럴 센트로이드 (명료도 지표)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            
            # 제로 크로싱 레이트 (음성 활동 지표)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            # MFCC 특성 (음성 특성)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_variance = np.var(mfccs, axis=1)
            
            # 명료도 점수 계산
            clarity = (
                np.mean(spectral_centroids) / 8000 * 0.4 +  # 고주파 성분
                (1 - np.mean(zcr)) * 0.3 +  # 안정성
                np.mean(mfcc_variance) / 100 * 0.3  # 특성 다양성
            )
            
            return min(max(clarity, 0.0), 1.0)
            
        except Exception as e:
            logging.warning(f"명료도 분석 실패: {e}")
            return 0.7
    
    def _analyze_volume_consistency(self, audio: np.ndarray) -> float:
        """볼륨 일관성 분석"""
        try:
            # RMS 에너지 계산
            rms = librosa.feature.rms(y=audio, hop_length=512)[0]
            
            # 활성 음성 구간만 선택 (상위 50%)
            active_threshold = np.percentile(rms, 50)
            active_rms = rms[rms >= active_threshold]
            
            if len(active_rms) > 0:
                # 표준편차를 이용한 일관성 측정
                mean_rms = np.mean(active_rms)
                std_rms = np.std(active_rms)
                
                if mean_rms > 0:
                    consistency = 1.0 - (std_rms / mean_rms)
                    return max(consistency, 0.0)
            
            return 0.5
            
        except Exception as e:
            logging.warning(f"볼륨 일관성 분석 실패: {e}")
            return 0.7
    
    def _analyze_frequency_spectrum(self, audio: np.ndarray, sr: int) -> Dict:
        """주파수 스펙트럼 분석"""
        try:
            # FFT 계산
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            magnitude = np.abs(fft)
            
            # 주요 주파수 대역 분석
            low_freq = np.sum(magnitude[(freqs >= 80) & (freqs <= 250)])    # 저음
            mid_freq = np.sum(magnitude[(freqs >= 250) & (freqs <= 2000)])  # 중음
            high_freq = np.sum(magnitude[(freqs >= 2000) & (freqs <= 8000)]) # 고음
            
            total_energy = low_freq + mid_freq + high_freq
            
            if total_energy > 0:
                return {
                    "low_frequency_ratio": round(low_freq / total_energy, 3),
                    "mid_frequency_ratio": round(mid_freq / total_energy, 3),
                    "high_frequency_ratio": round(high_freq / total_energy, 3),
                    "dominant_frequency": round(freqs[np.argmax(magnitude[:len(freqs)//2])], 1)
                }
            else:
                return {"error": "주파수 분석 실패"}
                
        except Exception as e:
            logging.warning(f"주파수 분석 실패: {e}")
            return {"error": str(e)}
    
    def _calculate_audio_quality_score(self, 
                                     noise_level: float, 
                                     clarity_score: float, 
                                     volume_consistency: float) -> float:
        """전체 음성 품질 점수 계산"""
        # 가중 평균으로 종합 점수 계산
        weights = {"noise": 0.3, "clarity": 0.4, "consistency": 0.3}
        
        quality_score = (
            (1.0 - noise_level) * weights["noise"] +
            clarity_score * weights["clarity"] +
            volume_consistency * weights["consistency"]
        )
        
        return quality_score
    
    def _assess_audio_quality(self, quality_score: float) -> str:
        """음성 품질 평가"""
        if quality_score >= 0.8:
            return "우수한 품질 - 현장 녹화 치고 매우 좋음"
        elif quality_score >= 0.6:
            return "양호한 품질 - 사용 가능한 수준"
        elif quality_score >= 0.4:
            return "보통 품질 - 노이즈 제거 권장"
        else:
            return "낮은 품질 - 재녹화 또는 전문 처리 필요"
    
    def _generate_audio_improvements(self, 
                                   noise_level: float, 
                                   clarity_score: float, 
                                   volume_consistency: float) -> List[str]:
        """음성 개선 제안 생성"""
        suggestions = []
        
        if noise_level > self.quality_thresholds["audio"]["noise_level"]:
            suggestions.append("🔧 노이즈 제거 필터 적용 권장")
            suggestions.append("📱 다음에는 마이크를 화자에게 더 가까이 설치")
        
        if clarity_score < self.quality_thresholds["audio"]["clarity_score"]:
            suggestions.append("🎚️ 고주파 강화 및 이퀄라이저 조정 권장")
            suggestions.append("🗣️ 화자가 더 명확하게 발음하도록 안내")
        
        if volume_consistency < self.quality_thresholds["audio"]["volume_consistency"]:
            suggestions.append("📊 자동 음량 정규화 적용 권장")
            suggestions.append("🎤 마이크 거리를 일정하게 유지")
        
        if not suggestions:
            suggestions.append("✅ 현재 품질이 양호합니다!")
        
        return suggestions
    
    async def analyze_image_quality(self, 
                                  image_data: bytes, 
                                  filename: str,
                                  is_ppt_screen: bool = False) -> Dict:
        """
        이미지 품질 분석 (PPT 화면 특화 포함)
        
        Args:
            image_data: 이미지 바이너리 데이터
            filename: 파일명
            is_ppt_screen: PPT 화면 여부 (자동 감지도 포함)
            
        Returns:
            이미지 품질 분석 결과
        """
        if not PIL_AVAILABLE:
            return {
                "success": False,
                "error": "PIL/Pillow가 필요합니다.",
                "filename": filename
            }
        
        try:
            print(f"📸 이미지 품질 분석 시작: {filename}")
            
            # PIL Image 객체 생성
            image = Image.open(io.BytesIO(image_data))
            
            # 1. 기본 이미지 품질 분석
            basic_quality = self._analyze_basic_image_quality(image)
            
            # 2. PPT 화면 감지 및 분석
            ppt_analysis = self._analyze_ppt_screen(image, is_ppt_screen)
            
            # 3. OCR 품질 분석
            ocr_quality = self._analyze_ocr_quality(image, ppt_analysis["is_ppt_screen"])
            
            # 4. 전체 품질 점수 계산
            overall_quality = self._calculate_image_quality_score(
                basic_quality, ppt_analysis, ocr_quality
            )
            
            # 5. 개선 제안 생성
            improvement_suggestions = self._generate_image_improvements(
                basic_quality, ppt_analysis, ocr_quality
            )
            
            result = {
                "success": True,
                "filename": filename,
                "basic_quality": basic_quality,
                "ppt_analysis": ppt_analysis,
                "ocr_quality": ocr_quality,
                "overall_quality": round(overall_quality, 3),
                "quality_assessment": self._assess_image_quality(overall_quality, ppt_analysis["is_ppt_screen"]),
                "improvement_suggestions": improvement_suggestions,
                "analysis_time": datetime.now().isoformat()
            }
            
            print(f"✅ 이미지 품질 분석 완료: {overall_quality:.1%} 품질")
            return result
            
        except Exception as e:
            logging.error(f"이미지 품질 분석 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "filename": filename
            }
    
    def _analyze_basic_image_quality(self, image: Image) -> Dict:
        """기본 이미지 품질 분석"""
        try:
            # RGB로 변환
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # NumPy 배열로 변환
            img_array = np.array(image)
            
            # 1. 블러 검출 (라플라시안 분산)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. 밝기 분석
            brightness = np.mean(img_array)
            
            # 3. 대비 분석
            contrast = np.std(img_array)
            
            # 4. 노이즈 분석
            noise_level = self._estimate_noise_level(gray)
            
            # 5. 해상도 체크
            width, height = image.size
            resolution_score = min((width * height) / (1920 * 1080), 1.0)
            
            return {
                "blur_score": round(blur_score, 2),
                "brightness": round(brightness, 2),
                "contrast": round(contrast, 2),
                "noise_level": round(noise_level, 3),
                "resolution": {"width": width, "height": height},
                "resolution_score": round(resolution_score, 3)
            }
            
        except Exception as e:
            logging.warning(f"기본 이미지 품질 분석 실패: {e}")
            return {"error": str(e)}
    
    def _analyze_ppt_screen(self, image: Image, is_ppt_hint: bool = False) -> Dict:
        """PPT 화면 분석"""
        try:
            # 1. PPT 화면 감지
            is_ppt_screen = self._detect_ppt_screen(image) or is_ppt_hint
            
            if not is_ppt_screen:
                return {
                    "is_ppt_screen": False,
                    "confidence": 0.0,
                    "ppt_specific_analysis": {}
                }
            
            # 2. PPT 특화 분석
            img_array = np.array(image.convert('RGB'))
            
            # 텍스트 영역 비율
            text_density = self._calculate_text_density(image)
            
            # 기하학적 구조 점수 (사각형, 정렬 등)
            geometric_score = self._analyze_ppt_geometry(img_array)
            
            # 색상 대비 분석
            color_contrast = self._analyze_color_contrast(img_array)
            
            # PPT 패턴 매칭
            pattern_score = self._match_ppt_patterns(image)
            
            return {
                "is_ppt_screen": True,
                "confidence": 0.8 if is_ppt_hint else 0.6,
                "ppt_specific_analysis": {
                    "text_density": round(text_density, 3),
                    "geometric_score": round(geometric_score, 3),
                    "color_contrast": round(color_contrast, 3),
                    "pattern_score": round(pattern_score, 3)
                }
            }
            
        except Exception as e:
            logging.warning(f"PPT 분석 실패: {e}")
            return {
                "is_ppt_screen": False,
                "error": str(e)
            }
    
    def _detect_ppt_screen(self, image: Image) -> bool:
        """PPT 화면 자동 감지"""
        try:
            # 간단한 휴리스틱으로 PPT 감지
            width, height = image.size
            aspect_ratio = width / height
            
            # 일반적인 PPT 비율 (4:3, 16:9, 16:10)
            ppt_ratios = [4/3, 16/9, 16/10]
            ratio_match = any(abs(aspect_ratio - ratio) < 0.1 for ratio in ppt_ratios)
            
            # 해상도가 프레젠테이션에 적합한지
            resolution_suitable = width >= 800 and height >= 600
            
            return ratio_match and resolution_suitable
            
        except Exception:
            return False
    
    def _calculate_text_density(self, image: Image) -> float:
        """텍스트 밀도 계산"""
        try:
            # 간단한 텍스트 감지 (OCR 없이)
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # 엣지 검출로 텍스트 영역 추정
            edges = cv2.Canny(img_array, 50, 150)
            text_pixels = np.sum(edges > 0)
            total_pixels = img_array.size
            
            return text_pixels / total_pixels
            
        except Exception:
            return 0.1
    
    def _analyze_ppt_geometry(self, img_array: np.ndarray) -> float:
        """PPT 기하학적 구조 분석"""
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # 직선 검출
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            # 사각형 검출
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangles = 0
            
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:
                    rectangles += 1
            
            # 구조 점수 계산
            line_score = min(len(lines) / 20, 1.0) if lines is not None else 0
            rect_score = min(rectangles / 10, 1.0)
            
            return (line_score + rect_score) / 2
            
        except Exception:
            return 0.5
    
    def _analyze_color_contrast(self, img_array: np.ndarray) -> float:
        """색상 대비 분석"""
        try:
            # LAB 색공간으로 변환
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            
            # 밝기 대비 계산
            contrast_ratio = np.std(l_channel) / np.mean(l_channel)
            
            return min(contrast_ratio, 5.0)  # 최대 5.0으로 제한
            
        except Exception:
            return 1.0
    
    def _match_ppt_patterns(self, image: Image) -> float:
        """PPT 패턴 매칭"""
        try:
            # OCR로 텍스트 추출 (간단히)
            import pytesseract
            text = pytesseract.image_to_string(image, lang='kor+eng')
            
            # 패턴 매칭
            matches = 0
            for pattern in self.ppt_patterns:
                if re.search(pattern, text):
                    matches += 1
            
            return min(matches / len(self.ppt_patterns), 1.0)
            
        except Exception:
            return 0.3
    
    def _analyze_ocr_quality(self, image: Image, is_ppt: bool) -> Dict:
        """OCR 품질 분석"""
        try:
            # OCR 수행
            import pytesseract
            
            # 신뢰도 포함 데이터 추출
            data = pytesseract.image_to_data(
                image, 
                lang='kor+eng',
                output_type=pytesseract.Output.DICT
            )
            
            # 신뢰도 분석
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) / 100 if confidences else 0.0
            
            # 텍스트 추출
            text = pytesseract.image_to_string(image, lang='kor+eng')
            word_count = len(text.split())
            
            # PPT용 특별 분석
            if is_ppt:
                ppt_readability = self._analyze_ppt_readability(text, confidences)
            else:
                ppt_readability = {}
            
            return {
                "average_confidence": round(avg_confidence, 3),
                "word_count": word_count,
                "text_length": len(text.strip()),
                "high_confidence_ratio": len([c for c in confidences if c >= 80]) / max(len(confidences), 1),
                "ppt_readability": ppt_readability
            }
            
        except Exception as e:
            logging.warning(f"OCR 품질 분석 실패: {e}")
            return {
                "error": str(e),
                "average_confidence": 0.0,
                "word_count": 0
            }
    
    def _analyze_ppt_readability(self, text: str, confidences: List[int]) -> Dict:
        """PPT 가독성 분석"""
        try:
            # 제목/내용 구분
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # 큰 글씨 (제목 추정) vs 작은 글씨 (내용)
            title_candidates = [line for line in lines if len(line) < 30 and any(c.isupper() for c in line)]
            content_lines = [line for line in lines if line not in title_candidates]
            
            # 구조화 정도
            structured_elements = len(re.findall(r'^\d+\.|\s*[▶▪►•]', text, re.MULTILINE))
            
            return {
                "title_count": len(title_candidates),
                "content_lines": len(content_lines),
                "structured_elements": structured_elements,
                "structure_score": min(structured_elements / max(len(lines), 1), 1.0)
            }
            
        except Exception:
            return {}
    
    def _estimate_noise_level(self, gray_image: np.ndarray) -> float:
        """이미지 노이즈 레벨 추정"""
        try:
            # 가우시안 필터 적용 후 차이로 노이즈 추정
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            noise = gray_image.astype(np.float32) - blurred.astype(np.float32)
            noise_level = np.std(noise) / 255.0
            
            return min(noise_level, 1.0)
            
        except Exception:
            return 0.1
    
    def _calculate_image_quality_score(self, 
                                     basic_quality: Dict,
                                     ppt_analysis: Dict,
                                     ocr_quality: Dict) -> float:
        """이미지 전체 품질 점수 계산"""
        try:
            # 기본 품질 점수
            blur_ok = basic_quality.get("blur_score", 0) >= self.quality_thresholds["image"]["blur_threshold"]
            brightness_ok = self.quality_thresholds["image"]["brightness_range"][0] <= basic_quality.get("brightness", 0) <= self.quality_thresholds["image"]["brightness_range"][1]
            contrast_ok = basic_quality.get("contrast", 0) >= self.quality_thresholds["image"]["contrast_min"]
            
            basic_score = (int(blur_ok) + int(brightness_ok) + int(contrast_ok)) / 3
            
            # OCR 품질 점수
            ocr_score = ocr_quality.get("average_confidence", 0)
            
            # PPT 특화 점수
            if ppt_analysis.get("is_ppt_screen", False):
                ppt_specific = ppt_analysis.get("ppt_specific_analysis", {})
                ppt_score = (
                    ppt_specific.get("text_density", 0) * 0.3 +
                    ppt_specific.get("geometric_score", 0) * 0.3 +
                    ppt_specific.get("color_contrast", 0) / 5.0 * 0.2 +
                    ppt_specific.get("pattern_score", 0) * 0.2
                )
                
                # PPT의 경우 가중치 조정
                return basic_score * 0.4 + ocr_score * 0.4 + ppt_score * 0.2
            else:
                return basic_score * 0.6 + ocr_score * 0.4
                
        except Exception:
            return 0.5
    
    def _assess_image_quality(self, quality_score: float, is_ppt: bool) -> str:
        """이미지 품질 평가"""
        prefix = "PPT 화면" if is_ppt else "일반 이미지"
        
        if quality_score >= 0.8:
            return f"우수한 {prefix} - OCR 품질 매우 좋음"
        elif quality_score >= 0.6:
            return f"양호한 {prefix} - 사용 가능한 수준"
        elif quality_score >= 0.4:
            return f"보통 {prefix} - 조명/각도 개선 필요"
        else:
            return f"낮은 {prefix} - 재촬영 권장"
    
    def _generate_image_improvements(self, 
                                   basic_quality: Dict,
                                   ppt_analysis: Dict,
                                   ocr_quality: Dict) -> List[str]:
        """이미지 개선 제안 생성"""
        suggestions = []
        
        # 기본 품질 개선
        blur_score = basic_quality.get("blur_score", 0)
        if blur_score < self.quality_thresholds["image"]["blur_threshold"]:
            suggestions.append("📷 흔들림 방지 - 삼각대 사용 또는 양손으로 고정")
        
        brightness = basic_quality.get("brightness", 0)
        brightness_range = self.quality_thresholds["image"]["brightness_range"]
        if brightness < brightness_range[0]:
            suggestions.append("💡 조명 개선 - 더 밝은 환경에서 촬영")
        elif brightness > brightness_range[1]:
            suggestions.append("🔆 노출 조정 - 플래시 끄기 또는 간접 조명 사용")
        
        contrast = basic_quality.get("contrast", 0)
        if contrast < self.quality_thresholds["image"]["contrast_min"]:
            suggestions.append("📊 대비 개선 - 배경과 텍스트 색상 차이 확보")
        
        # OCR 품질 개선
        ocr_confidence = ocr_quality.get("average_confidence", 0)
        if ocr_confidence < self.quality_thresholds["image"]["text_confidence"]:
            suggestions.append("🔤 텍스트 인식 개선 - 정면에서 촬영하고 각도 최소화")
        
        # PPT 특화 개선
        if ppt_analysis.get("is_ppt_screen", False):
            ppt_specific = ppt_analysis.get("ppt_specific_analysis", {})
            
            if ppt_specific.get("text_density", 0) < self.quality_thresholds["ppt"]["text_density"]:
                suggestions.append("📱 PPT 화면 확대 - 텍스트가 더 크게 보이도록 촬영")
            
            if ppt_specific.get("geometric_score", 0) < self.quality_thresholds["ppt"]["geometric_score"]:
                suggestions.append("📐 화면 정렬 - 스크린에 수직으로 촬영")
        
        if not suggestions:
            suggestions.append("✅ 현재 품질이 양호합니다!")
        
        return suggestions


# 전역 인스턴스
_quality_analyzer_instance = None

def get_quality_analyzer() -> QualityAnalyzer:
    """전역 품질 분석기 인스턴스 반환"""
    global _quality_analyzer_instance
    if _quality_analyzer_instance is None:
        _quality_analyzer_instance = QualityAnalyzer()
    return _quality_analyzer_instance

# 편의 함수들
async def analyze_audio_quality(audio_data: bytes, filename: str) -> Dict:
    """음성 품질 분석 편의 함수"""
    analyzer = get_quality_analyzer()
    return await analyzer.analyze_audio_quality(audio_data, filename)

async def analyze_image_quality(image_data: bytes, filename: str, is_ppt: bool = False) -> Dict:
    """이미지 품질 분석 편의 함수"""
    analyzer = get_quality_analyzer()
    return await analyzer.analyze_image_quality(image_data, filename, is_ppt)

if __name__ == "__main__":
    print("품질 분석 엔진 테스트")
    print("라이브러리 상태:", {"librosa": LIBROSA_AVAILABLE, "PIL": PIL_AVAILABLE})
