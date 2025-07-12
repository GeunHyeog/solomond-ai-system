#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
솔로몬드 AI v2.1 - 품질 검증 엔진
실시간 품질 모니터링 및 자동 개선 제안 시스템

작성자: 전근혁 (솔로몬드 대표)
생성일: 2025.07.11
목적: 현장에서 즉시 품질 확인 및 개선 권장
"""

import numpy as np
import cv2
import librosa
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Tuple, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioQualityAnalyzer:
    """음성 품질 분석기"""
    
    def __init__(self):
        self.min_snr = 20.0  # dB
        self.min_clarity = 0.8
        self.sample_rate = 22050
    
    def analyze_audio_quality(self, audio_file: str) -> Dict:
        """음성 파일 품질 분석"""
        try:
            # 음성 파일 로드
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # SNR 계산
            snr = self._calculate_snr(y)
            
            # 명료도 계산
            clarity = self._calculate_clarity(y, sr)
            
            # 배경음 레벨
            noise_level = self._estimate_noise_level(y)
            
            # 전체 품질 점수
            quality_score = self._calculate_overall_score(snr, clarity, noise_level)
            
            analysis = {
                "snr_db": round(snr, 2),
                "clarity_score": round(clarity, 3),
                "noise_level": round(noise_level, 3),
                "overall_quality": round(quality_score, 3),
                "duration_seconds": len(y) / sr,
                "sample_rate": sr,
                "recommendations": self._generate_recommendations(snr, clarity, noise_level),
                "quality_status": self._get_quality_status(quality_score),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"🎤 음성 품질 분석 완료: {quality_score:.1%}")
            return analysis
            
        except Exception as e:
            logger.error(f"음성 품질 분석 오류: {e}")
            return {"error": str(e), "quality_status": "분석 실패"}
    
    def _calculate_snr(self, y: np.ndarray) -> float:
        """Signal-to-Noise Ratio 계산"""
        # 음성 구간과 무음 구간 분리
        intervals = librosa.effects.split(y, top_db=20)
        
        if len(intervals) == 0:
            return 0.0
        
        # 신호 구간의 평균 에너지
        signal_energy = 0
        signal_samples = 0
        
        for start, end in intervals:
            signal_energy += np.sum(y[start:end] ** 2)
            signal_samples += (end - start)
        
        if signal_samples == 0:
            return 0.0
        
        signal_power = signal_energy / signal_samples
        
        # 전체 에너지에서 신호 에너지를 뺀 것이 노이즈
        total_energy = np.sum(y ** 2)
        noise_energy = total_energy - signal_energy
        noise_samples = len(y) - signal_samples
        
        if noise_samples <= 0:
            return 40.0  # 노이즈가 없으면 높은 SNR
        
        noise_power = noise_energy / noise_samples
        
        if noise_power <= 0:
            return 40.0
        
        snr = 10 * np.log10(signal_power / noise_power)
        return max(0, snr)
    
    def _calculate_clarity(self, y: np.ndarray, sr: int) -> float:
        """음성 명료도 계산"""
        # 스펙트럴 중심주파수 계산
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # MFCC 계수 계산 (음성 특징)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # 제로 크로싱 비율 (발음 명확성 지표)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # 정규화된 명료도 점수 계산
        clarity = (
            np.mean(spectral_centroids) / 4000 * 0.3 +  # 주파수 특성
            np.std(mfccs) / 50 * 0.4 +                  # 음성 특징 다양성
            np.mean(zcr) * 100 * 0.3                    # 발음 명확성
        )
        
        return min(1.0, max(0.0, clarity))
    
    def _estimate_noise_level(self, y: np.ndarray) -> float:
        """배경음 레벨 추정"""
        # 무음 구간 검출
        intervals = librosa.effects.split(y, top_db=20)
        
        if len(intervals) == 0:
            return np.std(y)  # 전체가 노이즈
        
        # 음성 구간이 아닌 부분의 표준편차 (노이즈 레벨)
        noise_segments = []
        prev_end = 0
        
        for start, end in intervals:
            if start > prev_end:
                noise_segments.extend(y[prev_end:start])
            prev_end = end
        
        if noise_segments:
            return np.std(noise_segments)
        else:
            return 0.0
    
    def _calculate_overall_score(self, snr: float, clarity: float, noise_level: float) -> float:
        """전체 품질 점수 계산"""
        snr_score = min(1.0, snr / 30.0)  # 30dB를 최대로 정규화
        clarity_score = clarity
        noise_score = max(0.0, 1.0 - noise_level * 10)  # 노이즈가 적을수록 높은 점수
        
        return (snr_score * 0.4 + clarity_score * 0.4 + noise_score * 0.2)
    
    def _generate_recommendations(self, snr: float, clarity: float, noise_level: float) -> List[str]:
        """품질 개선 권장사항 생성"""
        recommendations = []
        
        if snr < 15:
            recommendations.append("🔴 노이즈가 심합니다. 조용한 곳으로 이동하세요.")
        elif snr < 20:
            recommendations.append("🟡 배경음이 있습니다. 가능하면 더 조용한 환경에서 녹음하세요.")
        
        if clarity < 0.6:
            recommendations.append("🔴 발음이 불명확합니다. 마이크에 더 가까이 말씀하세요.")
        elif clarity < 0.8:
            recommendations.append("🟡 발음을 더 명확히 해주세요.")
        
        if noise_level > 0.1:
            recommendations.append("🔴 배경음이 큽니다. 노이즈 캔슬링 기능을 사용하세요.")
        
        if not recommendations:
            recommendations.append("🟢 음성 품질이 우수합니다. 현재 설정을 유지하세요.")
        
        return recommendations
    
    def _get_quality_status(self, score: float) -> str:
        """품질 상태 반환"""
        if score >= 0.8:
            return "우수"
        elif score >= 0.6:
            return "양호"
        elif score >= 0.4:
            return "보통"
        else:
            return "개선필요"


class OCRQualityAnalyzer:
    """OCR 품질 분석기"""
    
    def __init__(self):
        self.min_resolution = 1920
        self.min_confidence = 0.8
    
    def analyze_image_quality(self, image_path: str) -> Dict:
        """이미지 품질 분석"""
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "이미지 로드 실패", "quality_status": "분석 실패"}
            
            # 해상도 확인
            height, width = image.shape[:2]
            resolution_score = min(1.0, width / self.min_resolution)
            
            # 선명도 계산
            sharpness = self._calculate_sharpness(image)
            
            # 대비 계산
            contrast = self._calculate_contrast(image)
            
            # 조명 균일성
            lighting = self._calculate_lighting_uniformity(image)
            
            # 전체 품질 점수
            quality_score = self._calculate_image_quality_score(
                resolution_score, sharpness, contrast, lighting
            )
            
            analysis = {
                "resolution": {"width": width, "height": height, "score": round(resolution_score, 3)},
                "sharpness_score": round(sharpness, 3),
                "contrast_score": round(contrast, 3),
                "lighting_score": round(lighting, 3),
                "overall_quality": round(quality_score, 3),
                "ocr_readiness": quality_score > 0.7,
                "recommendations": self._generate_image_recommendations(
                    resolution_score, sharpness, contrast, lighting
                ),
                "quality_status": self._get_quality_status(quality_score),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"📸 이미지 품질 분석 완료: {quality_score:.1%}")
            return analysis
            
        except Exception as e:
            logger.error(f"이미지 품질 분석 오류: {e}")
            return {"error": str(e), "quality_status": "분석 실패"}
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """이미지 선명도 계산 (Laplacian variance)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        return min(1.0, sharpness / 1000.0)  # 정규화
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """이미지 대비 계산"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        return min(1.0, contrast / 128.0)  # 정규화
    
    def _calculate_lighting_uniformity(self, image: np.ndarray) -> float:
        """조명 균일성 계산"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 이미지를 블록으로 나누어 각 블록의 평균 밝기 계산
        h, w = gray.shape
        block_size = min(h, w) // 8
        
        if block_size < 10:
            return 0.5  # 이미지가 너무 작음
        
        block_means = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i + block_size, j:j + block_size]
                block_means.append(np.mean(block))
        
        if not block_means:
            return 0.5
        
        # 블록 간 밝기 차이가 적을수록 조명이 균일
        uniformity = 1.0 - (np.std(block_means) / 128.0)
        return max(0.0, min(1.0, uniformity))
    
    def _calculate_image_quality_score(self, resolution: float, sharpness: float, 
                                     contrast: float, lighting: float) -> float:
        """전체 이미지 품질 점수 계산"""
        return (resolution * 0.2 + sharpness * 0.3 + contrast * 0.3 + lighting * 0.2)
    
    def _generate_image_recommendations(self, resolution: float, sharpness: float,
                                      contrast: float, lighting: float) -> List[str]:
        """이미지 품질 개선 권장사항"""
        recommendations = []
        
        if resolution < 0.5:
            recommendations.append("🔴 해상도가 낮습니다. 더 고해상도로 촬영하세요.")
        elif resolution < 0.8:
            recommendations.append("🟡 해상도를 높이면 더 좋은 결과를 얻을 수 있습니다.")
        
        if sharpness < 0.3:
            recommendations.append("🔴 이미지가 흐릿합니다. 초점을 다시 맞춰주세요.")
        elif sharpness < 0.6:
            recommendations.append("🟡 조금 더 선명하게 촬영해주세요.")
        
        if contrast < 0.3:
            recommendations.append("🔴 대비가 부족합니다. 조명을 개선하세요.")
        elif contrast < 0.6:
            recommendations.append("🟡 조명을 조정하여 대비를 높여보세요.")
        
        if lighting < 0.4:
            recommendations.append("🔴 조명이 불균일합니다. 균일한 조명을 사용하세요.")
        elif lighting < 0.7:
            recommendations.append("🟡 조명을 더 균일하게 조정해보세요.")
        
        if not recommendations:
            recommendations.append("🟢 이미지 품질이 우수합니다. OCR 처리에 적합합니다.")
        
        return recommendations
    
    def _get_quality_status(self, score: float) -> str:
        """품질 상태 반환"""
        if score >= 0.8:
            return "우수"
        elif score >= 0.6:
            return "양호" 
        elif score >= 0.4:
            return "보통"
        else:
            return "개선필요"


class QualityManager:
    """통합 품질 관리자"""
    
    def __init__(self):
        self.audio_analyzer = AudioQualityAnalyzer()
        self.ocr_analyzer = OCRQualityAnalyzer()
        
    def comprehensive_quality_check(self, files: Dict[str, str]) -> Dict:
        """포괄적 품질 검사"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "files_analyzed": len(files),
            "audio_results": {},
            "image_results": {},
            "overall_summary": {},
            "recommendations": []
        }
        
        # 음성 파일 분석
        for file_type, file_path in files.items():
            if file_type.startswith('audio'):
                results["audio_results"][file_type] = self.audio_analyzer.analyze_audio_quality(file_path)
            elif file_type.startswith('image'):
                results["image_results"][file_type] = self.ocr_analyzer.analyze_image_quality(file_path)
        
        # 전체 요약 생성
        results["overall_summary"] = self._generate_overall_summary(results)
        
        return results
    
    def _generate_overall_summary(self, results: Dict) -> Dict:
        """전체 품질 요약 생성"""
        audio_scores = [r.get("overall_quality", 0) for r in results["audio_results"].values() if "overall_quality" in r]
        image_scores = [r.get("overall_quality", 0) for r in results["image_results"].values() if "overall_quality" in r]
        
        summary = {
            "audio_avg_quality": np.mean(audio_scores) if audio_scores else 0,
            "image_avg_quality": np.mean(image_scores) if image_scores else 0,
            "total_files": len(audio_scores) + len(image_scores),
            "ready_for_processing": True
        }
        
        # 처리 준비도 판단
        if summary["audio_avg_quality"] < 0.6 or summary["image_avg_quality"] < 0.6:
            summary["ready_for_processing"] = False
            summary["reason"] = "품질이 기준에 미달합니다. 개선 후 재시도하세요."
        
        return summary


class QualityAnalyzerV21:
    """주얼리 AI 플랫폼 v2.1 통합 품질 분석기"""
    
    def __init__(self):
        """초기화"""
        self.audio_analyzer = AudioQualityAnalyzer()
        self.ocr_analyzer = OCRQualityAnalyzer()
        self.quality_manager = QualityManager()
        self.version = "2.1.0"
        
        logger.info(f"🔬 QualityAnalyzerV21 v{self.version} 초기화 완료")
    
    def analyze_quality(self, file_path: str, file_type: str = "auto") -> Dict:
        """단일 파일 품질 분석"""
        try:
            if file_type == "auto":
                file_type = self._detect_file_type(file_path)
            
            if file_type in ["audio", "wav", "mp3", "mp4"]:
                return self.audio_analyzer.analyze_audio_quality(file_path)
            elif file_type in ["image", "jpg", "png", "jpeg"]:
                return self.ocr_analyzer.analyze_image_quality(file_path)
            else:
                return {"error": f"지원하지 않는 파일 타입: {file_type}", "quality_status": "분석 실패"}
                
        except Exception as e:
            logger.error(f"품질 분석 오류: {e}")
            return {"error": str(e), "quality_status": "분석 실패"}
    
    def batch_analyze(self, files: Dict[str, str]) -> Dict:
        """다중 파일 일괄 품질 분석"""
        return self.quality_manager.comprehensive_quality_check(files)
    
    def get_real_time_quality_metrics(self) -> Dict:
        """실시간 품질 지표 반환 (데모용)"""
        return {
            "audio_quality": {
                "snr_db": 24.5,
                "clarity": 92,
                "background_noise": "낮음",
                "status": "✅"
            },
            "ocr_quality": {
                "accuracy": 97,
                "ppt_recognition": 98,
                "table_chart": 94,
                "status": "✅"
            },
            "integration_analysis": {
                "language_consistency": 95,
                "content_connectivity": 89,
                "translation_accuracy": 93,
                "status": "✅"
            },
            "overall_status": "우수",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_quality_recommendations(self, quality_scores: Dict) -> List[str]:
        """품질 기반 권장사항 생성"""
        recommendations = []
        
        # 오디오 품질 권장사항
        audio_score = quality_scores.get("audio_quality", {}).get("snr_db", 0)
        if audio_score < 20:
            recommendations.append("🔴 노이즈 높음: 조용한 곳으로 이동 권장")
        elif audio_score < 25:
            recommendations.append("🟡 OCR 낮음: 카메라 각도 조정 필요")
        else:
            recommendations.append("🟢 품질 우수: 현재 설정 유지")
        
        return recommendations
    
    def _detect_file_type(self, file_path: str) -> str:
        """파일 확장자로 타입 감지"""
        extension = Path(file_path).suffix.lower()
        
        if extension in ['.wav', '.mp3', '.mp4', '.m4a']:
            return "audio"
        elif extension in ['.jpg', '.jpeg', '.png', '.bmp']:
            return "image"
        else:
            return "unknown"
    
    def get_version_info(self) -> Dict:
        """버전 정보 반환"""
        return {
            "version": self.version,
            "components": {
                "audio_analyzer": "AudioQualityAnalyzer",
                "ocr_analyzer": "OCRQualityAnalyzer", 
                "quality_manager": "QualityManager"
            },
            "features": [
                "실시간 음성 품질 분석",
                "OCR 이미지 품질 검증",
                "통합 품질 관리",
                "자동 개선 권장사항"
            ]
        }


# 사용 예시
if __name__ == "__main__":
    quality_analyzer = QualityAnalyzerV21()
    
    # 테스트용 더미 데이터
    test_files = {
        "audio_meeting": "sample_meeting.wav",
        "image_document": "sample_document.jpg"
    }
    
    print("🔍 품질 검증 시스템 테스트")
    print("=" * 50)
    
    # 실시간 품질 지표 확인
    metrics = quality_analyzer.get_real_time_quality_metrics()
    print("📊 실시간 품질 지표:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    
    # 버전 정보 확인
    version_info = quality_analyzer.get_version_info()
    print(f"\n✅ QualityAnalyzerV21 v{version_info['version']} 로드 완료")
    print("📊 실제 파일로 테스트하려면 파일 경로를 수정하세요")
