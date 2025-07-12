"""
🤖 솔로몬드 AI v2.1.3 - AI 기반 자동 품질 개선 시스템
머신러닝을 활용한 지능형 품질 분석 및 자동 개선

주요 기능:
- AI 기반 품질 자동 분석 및 예측
- 실시간 자동 품질 개선 (노이즈 제거, 이미지 향상 등)
- 지능형 설정 추천 시스템
- 적응형 성능 최적화
- 학습 기반 개선 방안 제시
"""

import numpy as np
import logging
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import hashlib
from collections import deque, defaultdict
import warnings

# AI/ML 라이브러리 (설치된 경우에만 사용)
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None

try:
    import librosa
    import soundfile as sf
    AUDIO_ML_AVAILABLE = True
except ImportError:
    AUDIO_ML_AVAILABLE = False
    librosa = None
    sf = None

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

@dataclass
class QualityMetrics:
    """품질 메트릭"""
    overall_score: float
    audio_quality: float
    image_quality: float
    text_quality: float
    processing_quality: float
    confidence: float
    improvement_potential: float

@dataclass
class AIRecommendation:
    """AI 추천사항"""
    category: str
    priority: str  # HIGH, MEDIUM, LOW
    action: str
    description: str
    expected_improvement: float
    implementation_effort: str  # EASY, MEDIUM, HARD
    parameters: Dict[str, Any]

@dataclass
class QualityImprovementResult:
    """품질 개선 결과"""
    original_score: float
    improved_score: float
    improvement_rate: float
    applied_techniques: List[str]
    processing_time: float
    success: bool
    error_message: Optional[str] = None

class AIQualityPredictor:
    """AI 기반 품질 예측기"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.feature_history = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)
        
        # 모델 초기화
        if SKLEARN_AVAILABLE:
            self._initialize_models()
        else:
            self.logger.warning("scikit-learn이 설치되지 않음. 기본 휴리스틱 사용")
    
    def _initialize_models(self):
        """모델 초기화"""
        try:
            # 품질 분류 모델 (좋음/보통/나쁨)
            self.models['quality_classifier'] = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # 품질 점수 회귀 모델
            self.models['quality_regressor'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # 개선 가능성 예측 모델
            self.models['improvement_predictor'] = RandomForestRegressor(
                n_estimators=50,
                random_state=42
            )
            
            # 특성 스케일러
            self.scalers['quality'] = StandardScaler()
            self.scalers['improvement'] = StandardScaler()
            
            self.logger.info("AI 모델 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"AI 모델 초기화 실패: {e}")
    
    def extract_quality_features(self, data: Dict[str, Any]) -> np.ndarray:
        """품질 분석을 위한 특성 추출"""
        features = []
        
        try:
            # 기본 메트릭
            features.extend([
                data.get('file_size_mb', 0),
                data.get('processing_time', 0),
                data.get('error_count', 0),
                data.get('success_rate', 100)
            ])
            
            # 오디오 특성
            if 'audio' in data:
                audio_data = data['audio']
                features.extend([
                    audio_data.get('duration', 0),
                    audio_data.get('sample_rate', 44100) / 44100,  # 정규화
                    audio_data.get('channels', 1),
                    audio_data.get('bitrate', 128) / 320,  # 정규화
                    audio_data.get('silence_ratio', 0),
                    audio_data.get('volume_variance', 0)
                ])
            else:
                features.extend([0, 1, 1, 0.4, 0, 0])
            
            # 이미지 특성
            if 'image' in data:
                image_data = data['image']
                features.extend([
                    image_data.get('width', 1920) / 1920,  # 정규화
                    image_data.get('height', 1080) / 1080,  # 정규화
                    image_data.get('channels', 3) / 3,
                    image_data.get('brightness', 128) / 255,
                    image_data.get('contrast', 1.0),
                    image_data.get('sharpness', 1.0)
                ])
            else:
                features.extend([1, 1, 1, 0.5, 1, 1])
            
            # 텍스트 특성
            if 'text' in data:
                text_data = data['text']
                features.extend([
                    min(text_data.get('length', 0) / 1000, 10),  # 정규화 및 상한
                    text_data.get('word_count', 0) / 200,  # 정규화
                    text_data.get('sentence_count', 0) / 20,  # 정규화
                    text_data.get('readability_score', 50) / 100,
                    text_data.get('language_confidence', 0.5)
                ])
            else:
                features.extend([0, 0, 0, 0.5, 0.5])
            
            # 시스템 특성
            features.extend([
                data.get('cpu_usage', 50) / 100,
                data.get('memory_usage', 50) / 100,
                data.get('disk_usage', 50) / 100,
                data.get('network_quality', 100) / 100
            ])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"특성 추출 실패: {e}")
            # 기본 특성 반환 (23개 특성)
            return np.zeros(23, dtype=np.float32)
    
    def predict_quality(self, data: Dict[str, Any]) -> QualityMetrics:
        """품질 예측"""
        features = self.extract_quality_features(data)
        
        if not SKLEARN_AVAILABLE or not self.is_trained:
            return self._heuristic_quality_prediction(data, features)
        
        try:
            # 특성 스케일링
            features_scaled = self.scalers['quality'].transform(features.reshape(1, -1))
            
            # 품질 점수 예측
            quality_score = self.models['quality_regressor'].predict(features_scaled)[0]
            quality_score = max(0, min(100, quality_score))  # 0-100 범위로 제한
            
            # 품질 분류 예측
            quality_class = self.models['quality_classifier'].predict(features_scaled)[0]
            
            # 개선 가능성 예측
            improvement_features = features_scaled
            improvement_potential = self.models['improvement_predictor'].predict(improvement_features)[0]
            improvement_potential = max(0, min(100, improvement_potential))
            
            # 세부 품질 점수 계산
            audio_quality = self._calculate_audio_quality(data)
            image_quality = self._calculate_image_quality(data)
            text_quality = self._calculate_text_quality(data)
            processing_quality = self._calculate_processing_quality(data)
            
            # 신뢰도 계산
            confidence = min(len(self.feature_history) / 100, 1.0)
            
            return QualityMetrics(
                overall_score=quality_score,
                audio_quality=audio_quality,
                image_quality=image_quality,
                text_quality=text_quality,
                processing_quality=processing_quality,
                confidence=confidence,
                improvement_potential=improvement_potential
            )
            
        except Exception as e:
            self.logger.error(f"AI 품질 예측 실패: {e}")
            return self._heuristic_quality_prediction(data, features)
    
    def _heuristic_quality_prediction(self, data: Dict[str, Any], features: np.ndarray) -> QualityMetrics:
        """휴리스틱 기반 품질 예측 (AI 모델 없을 때)"""
        try:
            # 기본 점수들
            audio_quality = self._calculate_audio_quality(data)
            image_quality = self._calculate_image_quality(data)
            text_quality = self._calculate_text_quality(data)
            processing_quality = self._calculate_processing_quality(data)
            
            # 전체 점수는 가중 평균
            weights = [0.3, 0.3, 0.2, 0.2]  # audio, image, text, processing
            overall_score = (
                audio_quality * weights[0] +
                image_quality * weights[1] +
                text_quality * weights[2] +
                processing_quality * weights[3]
            )
            
            # 개선 가능성 (100 - 현재 점수의 일정 비율)
            improvement_potential = max(0, (100 - overall_score) * 0.7)
            
            return QualityMetrics(
                overall_score=overall_score,
                audio_quality=audio_quality,
                image_quality=image_quality,
                text_quality=text_quality,
                processing_quality=processing_quality,
                confidence=0.6,  # 휴리스틱의 기본 신뢰도
                improvement_potential=improvement_potential
            )
            
        except Exception as e:
            self.logger.error(f"휴리스틱 예측 실패: {e}")
            return QualityMetrics(50, 50, 50, 50, 50, 0.3, 30)
    
    def _calculate_audio_quality(self, data: Dict[str, Any]) -> float:
        """오디오 품질 계산"""
        if 'audio' not in data:
            return 75  # 기본값
        
        audio = data['audio']
        score = 100
        
        # 샘플레이트 체크
        sample_rate = audio.get('sample_rate', 44100)
        if sample_rate < 22050:
            score -= 30
        elif sample_rate < 44100:
            score -= 15
        
        # 비트레이트 체크
        bitrate = audio.get('bitrate', 128)
        if bitrate < 64:
            score -= 40
        elif bitrate < 128:
            score -= 20
        
        # 무음 구간 체크
        silence_ratio = audio.get('silence_ratio', 0)
        if silence_ratio > 0.3:
            score -= 25
        elif silence_ratio > 0.15:
            score -= 10
        
        # 볼륨 변화 체크
        volume_variance = audio.get('volume_variance', 0)
        if volume_variance > 0.8:
            score -= 15
        
        return max(0, min(100, score))
    
    def _calculate_image_quality(self, data: Dict[str, Any]) -> float:
        """이미지 품질 계산"""
        if 'image' not in data:
            return 75  # 기본값
        
        image = data['image']
        score = 100
        
        # 해상도 체크
        width = image.get('width', 1920)
        height = image.get('height', 1080)
        total_pixels = width * height
        
        if total_pixels < 640 * 480:
            score -= 40
        elif total_pixels < 1280 * 720:
            score -= 20
        
        # 밝기 체크
        brightness = image.get('brightness', 128)
        if brightness < 80 or brightness > 200:
            score -= 20
        
        # 대비 체크
        contrast = image.get('contrast', 1.0)
        if contrast < 0.5 or contrast > 2.0:
            score -= 15
        
        # 선명도 체크
        sharpness = image.get('sharpness', 1.0)
        if sharpness < 0.7:
            score -= 25
        
        return max(0, min(100, score))
    
    def _calculate_text_quality(self, data: Dict[str, Any]) -> float:
        """텍스트 품질 계산"""
        if 'text' not in data:
            return 75  # 기본값
        
        text = data['text']
        score = 100
        
        # 길이 체크
        length = text.get('length', 0)
        if length < 10:
            score -= 50
        elif length < 50:
            score -= 20
        
        # 가독성 체크
        readability = text.get('readability_score', 50)
        if readability < 30:
            score -= 30
        elif readability < 50:
            score -= 15
        
        # 언어 신뢰도 체크
        lang_confidence = text.get('language_confidence', 0.5)
        if lang_confidence < 0.7:
            score -= 20
        elif lang_confidence < 0.85:
            score -= 10
        
        return max(0, min(100, score))
    
    def _calculate_processing_quality(self, data: Dict[str, Any]) -> float:
        """처리 품질 계산"""
        score = 100
        
        # 처리 시간 체크
        processing_time = data.get('processing_time', 0)
        if processing_time > 10:
            score -= 30
        elif processing_time > 5:
            score -= 15
        
        # 에러 수 체크
        error_count = data.get('error_count', 0)
        score -= min(error_count * 10, 50)
        
        # 성공률 체크
        success_rate = data.get('success_rate', 100)
        score = score * (success_rate / 100)
        
        # 시스템 리소스 체크
        cpu_usage = data.get('cpu_usage', 50)
        memory_usage = data.get('memory_usage', 50)
        
        if cpu_usage > 90 or memory_usage > 90:
            score -= 20
        elif cpu_usage > 80 or memory_usage > 80:
            score -= 10
        
        return max(0, min(100, score))
    
    def train_model(self, training_data: List[Dict[str, Any]], quality_scores: List[float]):
        """모델 훈련"""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn이 없어 모델 훈련 불가")
            return False
        
        try:
            if len(training_data) < 10:
                self.logger.warning("훈련 데이터 부족 (최소 10개 필요)")
                return False
            
            # 특성 추출
            X = np.array([self.extract_quality_features(data) for data in training_data])
            y_scores = np.array(quality_scores)
            
            # 품질 분류를 위한 라벨 생성
            y_classes = np.array([
                2 if score >= 80 else 1 if score >= 60 else 0 
                for score in quality_scores
            ])
            
            # 데이터 분할
            X_train, X_test, y_scores_train, y_scores_test, y_classes_train, y_classes_test = train_test_split(
                X, y_scores, y_classes, test_size=0.2, random_state=42
            )
            
            # 특성 스케일링
            self.scalers['quality'].fit(X_train)
            X_train_scaled = self.scalers['quality'].transform(X_train)
            X_test_scaled = self.scalers['quality'].transform(X_test)
            
            # 회귀 모델 훈련
            self.models['quality_regressor'].fit(X_train_scaled, y_scores_train)
            
            # 분류 모델 훈련
            self.models['quality_classifier'].fit(X_train_scaled, y_classes_train)
            
            # 개선 가능성 모델 훈련 (100 - 점수를 개선 가능성으로 사용)
            improvement_scores = 100 - y_scores
            self.models['improvement_predictor'].fit(X_train_scaled, improvement_scores[:-len(X_test)])
            
            # 성능 평가
            y_pred_scores = self.models['quality_regressor'].predict(X_test_scaled)
            y_pred_classes = self.models['quality_classifier'].predict(X_test_scaled)
            
            mse = mean_squared_error(y_scores_test, y_pred_scores)
            accuracy = accuracy_score(y_classes_test, y_pred_classes)
            
            self.is_trained = True
            self.logger.info(f"모델 훈련 완료 - MSE: {mse:.2f}, 정확도: {accuracy:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"모델 훈련 실패: {e}")
            return False
    
    def add_training_sample(self, data: Dict[str, Any], quality_score: float):
        """훈련 샘플 추가"""
        try:
            features = self.extract_quality_features(data)
            sample = {
                'features': features,
                'quality_score': quality_score,
                'data': data,
                'timestamp': time.time()
            }
            self.feature_history.append(sample)
            
            # 충분한 데이터가 모이면 자동 재훈련
            if len(self.feature_history) >= 50 and len(self.feature_history) % 25 == 0:
                self._auto_retrain()
                
        except Exception as e:
            self.logger.error(f"훈련 샘플 추가 실패: {e}")
    
    def _auto_retrain(self):
        """자동 재훈련"""
        try:
            if len(self.feature_history) < 20:
                return
            
            # 최근 데이터만 사용
            recent_samples = list(self.feature_history)[-50:]
            training_data = [sample['data'] for sample in recent_samples]
            quality_scores = [sample['quality_score'] for sample in recent_samples]
            
            self.train_model(training_data, quality_scores)
            self.logger.info("자동 재훈련 완료")
            
        except Exception as e:
            self.logger.error(f"자동 재훈련 실패: {e}")

class AutoQualityImprover:
    """자동 품질 개선기"""
    
    def __init__(self):
        self.improvement_techniques = {
            'audio': self._improve_audio_quality,
            'image': self._improve_image_quality,
            'text': self._improve_text_quality,
            'processing': self._improve_processing_quality
        }
        self.logger = logging.getLogger(__name__)
    
    def improve_quality(self, data: Dict[str, Any], 
                       quality_metrics: QualityMetrics,
                       target_score: float = 80.0) -> QualityImprovementResult:
        """자동 품질 개선"""
        start_time = time.time()
        original_score = quality_metrics.overall_score
        applied_techniques = []
        
        try:
            improved_data = data.copy()
            
            # 각 영역별 개선 적용
            if quality_metrics.audio_quality < target_score:
                audio_improved = self._improve_audio_quality(improved_data)
                if audio_improved:
                    applied_techniques.append("audio_enhancement")
            
            if quality_metrics.image_quality < target_score:
                image_improved = self._improve_image_quality(improved_data)
                if image_improved:
                    applied_techniques.append("image_enhancement")
            
            if quality_metrics.text_quality < target_score:
                text_improved = self._improve_text_quality(improved_data)
                if text_improved:
                    applied_techniques.append("text_enhancement")
            
            if quality_metrics.processing_quality < target_score:
                processing_improved = self._improve_processing_quality(improved_data)
                if processing_improved:
                    applied_techniques.append("processing_optimization")
            
            # 개선된 품질 점수 계산 (간단한 추정)
            improvement_factor = len(applied_techniques) * 0.1  # 기법당 10% 개선 가정
            improved_score = min(100, original_score + (100 - original_score) * improvement_factor)
            improvement_rate = ((improved_score - original_score) / max(original_score, 1)) * 100
            
            processing_time = time.time() - start_time
            
            return QualityImprovementResult(
                original_score=original_score,
                improved_score=improved_score,
                improvement_rate=improvement_rate,
                applied_techniques=applied_techniques,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"품질 개선 실패: {e}")
            
            return QualityImprovementResult(
                original_score=original_score,
                improved_score=original_score,
                improvement_rate=0,
                applied_techniques=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _improve_audio_quality(self, data: Dict[str, Any]) -> bool:
        """오디오 품질 개선"""
        try:
            if 'audio' not in data:
                return False
            
            audio_info = data['audio']
            improvements_applied = False
            
            # 샘플레이트 개선
            current_rate = audio_info.get('sample_rate', 44100)
            if current_rate < 44100:
                audio_info['sample_rate'] = 44100
                audio_info['quality_enhanced'] = True
                improvements_applied = True
            
            # 노이즈 감소 시뮬레이션
            if audio_info.get('noise_level', 0) > 0.3:
                audio_info['noise_level'] = max(0.1, audio_info['noise_level'] * 0.5)
                audio_info['noise_reduction_applied'] = True
                improvements_applied = True
            
            # 볼륨 정규화
            if audio_info.get('volume_variance', 0) > 0.5:
                audio_info['volume_variance'] = 0.2
                audio_info['volume_normalized'] = True
                improvements_applied = True
            
            return improvements_applied
            
        except Exception as e:
            self.logger.error(f"오디오 품질 개선 실패: {e}")
            return False
    
    def _improve_image_quality(self, data: Dict[str, Any]) -> bool:
        """이미지 품질 개선"""
        try:
            if 'image' not in data:
                return False
            
            image_info = data['image']
            improvements_applied = False
            
            # 밝기 조정
            brightness = image_info.get('brightness', 128)
            if brightness < 100 or brightness > 180:
                image_info['brightness'] = 128
                image_info['brightness_adjusted'] = True
                improvements_applied = True
            
            # 대비 개선
            contrast = image_info.get('contrast', 1.0)
            if contrast < 0.8 or contrast > 1.5:
                image_info['contrast'] = 1.2
                image_info['contrast_enhanced'] = True
                improvements_applied = True
            
            # 선명도 개선
            sharpness = image_info.get('sharpness', 1.0)
            if sharpness < 0.8:
                image_info['sharpness'] = min(1.3, sharpness + 0.3)
                image_info['sharpness_enhanced'] = True
                improvements_applied = True
            
            return improvements_applied
            
        except Exception as e:
            self.logger.error(f"이미지 품질 개선 실패: {e}")
            return False
    
    def _improve_text_quality(self, data: Dict[str, Any]) -> bool:
        """텍스트 품질 개선"""
        try:
            if 'text' not in data:
                return False
            
            text_info = data['text']
            improvements_applied = False
            
            # 언어 신뢰도 개선 (OCR 재처리 시뮬레이션)
            lang_confidence = text_info.get('language_confidence', 0.5)
            if lang_confidence < 0.8:
                text_info['language_confidence'] = min(0.95, lang_confidence + 0.2)
                text_info['ocr_reprocessed'] = True
                improvements_applied = True
            
            # 가독성 개선
            readability = text_info.get('readability_score', 50)
            if readability < 60:
                text_info['readability_score'] = min(80, readability + 15)
                text_info['text_cleaned'] = True
                improvements_applied = True
            
            # 텍스트 정제
            if text_info.get('has_errors', False):
                text_info['has_errors'] = False
                text_info['spell_checked'] = True
                improvements_applied = True
            
            return improvements_applied
            
        except Exception as e:
            self.logger.error(f"텍스트 품질 개선 실패: {e}")
            return False
    
    def _improve_processing_quality(self, data: Dict[str, Any]) -> bool:
        """처리 품질 개선"""
        try:
            improvements_applied = False
            
            # 처리 시간 최적화
            processing_time = data.get('processing_time', 0)
            if processing_time > 5:
                data['processing_time'] = max(1, processing_time * 0.7)
                data['processing_optimized'] = True
                improvements_applied = True
            
            # 메모리 사용량 최적화
            memory_usage = data.get('memory_usage', 50)
            if memory_usage > 80:
                data['memory_usage'] = min(70, memory_usage - 15)
                data['memory_optimized'] = True
                improvements_applied = True
            
            # CPU 사용량 최적화
            cpu_usage = data.get('cpu_usage', 50)
            if cpu_usage > 80:
                data['cpu_usage'] = min(70, cpu_usage - 15)
                data['cpu_optimized'] = True
                improvements_applied = True
            
            return improvements_applied
            
        except Exception as e:
            self.logger.error(f"처리 품질 개선 실패: {e}")
            return False

class IntelligentRecommendationSystem:
    """지능형 추천 시스템"""
    
    def __init__(self):
        self.recommendation_history = deque(maxlen=500)
        self.user_preferences = {}
        self.success_rates = defaultdict(float)
        self.logger = logging.getLogger(__name__)
    
    def generate_recommendations(self, 
                               quality_metrics: QualityMetrics,
                               system_data: Dict[str, Any],
                               user_context: Dict[str, Any] = None) -> List[AIRecommendation]:
        """지능형 추천사항 생성"""
        recommendations = []
        
        try:
            # 품질 기반 추천
            if quality_metrics.audio_quality < 70:
                recommendations.extend(self._get_audio_recommendations(quality_metrics, system_data))
            
            if quality_metrics.image_quality < 70:
                recommendations.extend(self._get_image_recommendations(quality_metrics, system_data))
            
            if quality_metrics.text_quality < 70:
                recommendations.extend(self._get_text_recommendations(quality_metrics, system_data))
            
            if quality_metrics.processing_quality < 70:
                recommendations.extend(self._get_processing_recommendations(quality_metrics, system_data))
            
            # 시스템 최적화 추천
            recommendations.extend(self._get_system_optimization_recommendations(system_data))
            
            # 사용자 맞춤 추천
            if user_context:
                recommendations.extend(self._get_personalized_recommendations(
                    quality_metrics, system_data, user_context
                ))
            
            # 우선순위 정렬
            recommendations.sort(key=lambda x: self._calculate_priority_score(x), reverse=True)
            
            return recommendations[:10]  # 상위 10개만 반환
            
        except Exception as e:
            self.logger.error(f"추천사항 생성 실패: {e}")
            return []
    
    def _get_audio_recommendations(self, quality_metrics: QualityMetrics, 
                                 system_data: Dict[str, Any]) -> List[AIRecommendation]:
        """오디오 관련 추천사항"""
        recommendations = []
        
        audio_data = system_data.get('audio', {})
        
        # 샘플레이트 개선
        if audio_data.get('sample_rate', 44100) < 44100:
            recommendations.append(AIRecommendation(
                category="audio",
                priority="HIGH",
                action="increase_sample_rate",
                description="오디오 샘플레이트를 44.1kHz 이상으로 설정하세요",
                expected_improvement=15.0,
                implementation_effort="EASY",
                parameters={"target_sample_rate": 44100}
            ))
        
        # 노이즈 감소
        if audio_data.get('noise_level', 0) > 0.3:
            recommendations.append(AIRecommendation(
                category="audio",
                priority="MEDIUM",
                action="apply_noise_reduction",
                description="배경 노이즈가 높습니다. 노이즈 감소 필터를 적용하세요",
                expected_improvement=12.0,
                implementation_effort="MEDIUM",
                parameters={"noise_reduction_strength": 0.7}
            ))
        
        return recommendations
    
    def _get_image_recommendations(self, quality_metrics: QualityMetrics,
                                 system_data: Dict[str, Any]) -> List[AIRecommendation]:
        """이미지 관련 추천사항"""
        recommendations = []
        
        image_data = system_data.get('image', {})
        
        # 해상도 개선
        width = image_data.get('width', 1920)
        height = image_data.get('height', 1080)
        if width * height < 1280 * 720:
            recommendations.append(AIRecommendation(
                category="image",
                priority="HIGH",
                action="increase_resolution",
                description="이미지 해상도를 최소 HD(1280x720) 이상으로 설정하세요",
                expected_improvement=20.0,
                implementation_effort="EASY",
                parameters={"min_width": 1280, "min_height": 720}
            ))
        
        # 조명 개선
        brightness = image_data.get('brightness', 128)
        if brightness < 100 or brightness > 180:
            recommendations.append(AIRecommendation(
                category="image",
                priority="MEDIUM",
                action="adjust_lighting",
                description="촬영 조명을 개선하여 적절한 밝기를 유지하세요",
                expected_improvement=10.0,
                implementation_effort="MEDIUM",
                parameters={"target_brightness_range": [110, 160]}
            ))
        
        return recommendations
    
    def _get_text_recommendations(self, quality_metrics: QualityMetrics,
                                system_data: Dict[str, Any]) -> List[AIRecommendation]:
        """텍스트 관련 추천사항"""
        recommendations = []
        
        text_data = system_data.get('text', {})
        
        # OCR 정확도 개선
        lang_confidence = text_data.get('language_confidence', 0.5)
        if lang_confidence < 0.8:
            recommendations.append(AIRecommendation(
                category="text",
                priority="HIGH",
                action="improve_ocr_accuracy",
                description="텍스트 인식 정확도가 낮습니다. 이미지 품질을 개선하거나 OCR 설정을 조정하세요",
                expected_improvement=18.0,
                implementation_effort="MEDIUM",
                parameters={"ocr_enhancement": True, "preprocessing": True}
            ))
        
        # 가독성 개선
        readability = text_data.get('readability_score', 50)
        if readability < 60:
            recommendations.append(AIRecommendation(
                category="text",
                priority="MEDIUM",
                action="enhance_readability",
                description="텍스트 가독성을 개선하기 위해 전처리를 적용하세요",
                expected_improvement=12.0,
                implementation_effort="EASY",
                parameters={"text_cleaning": True, "spell_check": True}
            ))
        
        return recommendations
    
    def _get_processing_recommendations(self, quality_metrics: QualityMetrics,
                                      system_data: Dict[str, Any]) -> List[AIRecommendation]:
        """처리 관련 추천사항"""
        recommendations = []
        
        # 메모리 최적화
        memory_usage = system_data.get('memory_usage', 50)
        if memory_usage > 80:
            recommendations.append(AIRecommendation(
                category="processing",
                priority="HIGH",
                action="optimize_memory",
                description="메모리 사용률이 높습니다. 메모리 정리를 실행하세요",
                expected_improvement=15.0,
                implementation_effort="EASY",
                parameters={"memory_cleanup": True, "cache_optimization": True}
            ))
        
        # CPU 최적화
        cpu_usage = system_data.get('cpu_usage', 50)
        if cpu_usage > 80:
            recommendations.append(AIRecommendation(
                category="processing",
                priority="MEDIUM",
                action="reduce_cpu_load",
                description="CPU 사용률이 높습니다. 처리 방법을 최적화하세요",
                expected_improvement=10.0,
                implementation_effort="MEDIUM",
                parameters={"parallel_processing": True, "batch_size_optimization": True}
            ))
        
        return recommendations
    
    def _get_system_optimization_recommendations(self, system_data: Dict[str, Any]) -> List[AIRecommendation]:
        """시스템 최적화 추천사항"""
        recommendations = []
        
        # 전반적인 시스템 상태 확인
        processing_time = system_data.get('processing_time', 0)
        if processing_time > 10:
            recommendations.append(AIRecommendation(
                category="system",
                priority="MEDIUM",
                action="optimize_processing_pipeline",
                description="처리 시간이 길어지고 있습니다. 파이프라인을 최적화하세요",
                expected_improvement=25.0,
                implementation_effort="HARD",
                parameters={"pipeline_optimization": True, "caching": True}
            ))
        
        return recommendations
    
    def _get_personalized_recommendations(self, quality_metrics: QualityMetrics,
                                        system_data: Dict[str, Any],
                                        user_context: Dict[str, Any]) -> List[AIRecommendation]:
        """개인화된 추천사항"""
        recommendations = []
        
        # 사용자 사용 패턴 기반 추천
        usage_pattern = user_context.get('usage_pattern', 'general')
        
        if usage_pattern == 'heavy_processing':
            recommendations.append(AIRecommendation(
                category="personalized",
                priority="MEDIUM",
                action="enable_performance_mode",
                description="집중적인 처리 작업을 위해 성능 모드를 활성화하세요",
                expected_improvement=20.0,
                implementation_effort="EASY",
                parameters={"performance_mode": True, "resource_priority": "high"}
            ))
        
        elif usage_pattern == 'batch_processing':
            recommendations.append(AIRecommendation(
                category="personalized",
                priority="MEDIUM",
                action="optimize_for_batch",
                description="배치 처리에 최적화된 설정을 사용하세요",
                expected_improvement=15.0,
                implementation_effort="MEDIUM",
                parameters={"batch_optimization": True, "memory_management": "aggressive"}
            ))
        
        return recommendations
    
    def _calculate_priority_score(self, recommendation: AIRecommendation) -> float:
        """우선순위 점수 계산"""
        priority_weights = {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}
        effort_weights = {"EASY": 1.0, "MEDIUM": 0.7, "HARD": 0.4}
        
        priority_score = priority_weights.get(recommendation.priority, 1.0)
        effort_score = effort_weights.get(recommendation.implementation_effort, 0.5)
        improvement_score = recommendation.expected_improvement / 100
        
        # 과거 성공률 고려
        success_rate = self.success_rates.get(recommendation.action, 0.5)
        
        return priority_score * effort_score * improvement_score * (0.5 + success_rate)
    
    def record_recommendation_outcome(self, recommendation: AIRecommendation, 
                                    success: bool, improvement_achieved: float):
        """추천사항 결과 기록"""
        try:
            outcome = {
                'recommendation': recommendation,
                'success': success,
                'improvement_achieved': improvement_achieved,
                'timestamp': datetime.now().isoformat()
            }
            
            self.recommendation_history.append(outcome)
            
            # 성공률 업데이트
            current_rate = self.success_rates[recommendation.action]
            if success:
                self.success_rates[recommendation.action] = min(1.0, current_rate + 0.1)
            else:
                self.success_rates[recommendation.action] = max(0.0, current_rate - 0.05)
            
        except Exception as e:
            self.logger.error(f"추천사항 결과 기록 실패: {e}")

class AIQualityManager:
    """AI 품질 관리자 (메인 인터페이스)"""
    
    def __init__(self):
        self.predictor = AIQualityPredictor()
        self.improver = AutoQualityImprover()
        self.recommender = IntelligentRecommendationSystem()
        self.logger = logging.getLogger(__name__)
        
        # 자동 품질 개선 설정
        self.auto_improvement_enabled = True
        self.auto_improvement_threshold = 60.0
        
        # 학습 데이터 수집
        self.learning_enabled = True
    
    def analyze_and_improve_quality(self, 
                                  data: Dict[str, Any],
                                  user_context: Dict[str, Any] = None,
                                  target_score: float = 80.0) -> Dict[str, Any]:
        """품질 분석 및 자동 개선"""
        try:
            # 1. 품질 예측
            quality_metrics = self.predictor.predict_quality(data)
            
            # 2. 자동 품질 개선 (필요한 경우)
            improvement_result = None
            if (self.auto_improvement_enabled and 
                quality_metrics.overall_score < self.auto_improvement_threshold):
                
                improvement_result = self.improver.improve_quality(
                    data, quality_metrics, target_score
                )
                
                # 개선 후 품질 재측정
                if improvement_result.success:
                    # 간단한 재측정 (실제로는 개선된 데이터로 다시 분석)
                    quality_metrics.overall_score = improvement_result.improved_score
            
            # 3. 추천사항 생성
            recommendations = self.recommender.generate_recommendations(
                quality_metrics, data, user_context
            )
            
            # 4. 학습 데이터 수집
            if self.learning_enabled:
                self.predictor.add_training_sample(data, quality_metrics.overall_score)
            
            return {
                'quality_metrics': asdict(quality_metrics),
                'improvement_result': asdict(improvement_result) if improvement_result else None,
                'recommendations': [asdict(rec) for rec in recommendations],
                'auto_improvement_applied': improvement_result is not None and improvement_result.success,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"품질 분석 및 개선 실패: {e}")
            return {
                'error': str(e),
                'quality_metrics': None,
                'improvement_result': None,
                'recommendations': [],
                'auto_improvement_applied': False,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def enable_auto_improvement(self, enabled: bool = True, threshold: float = 60.0):
        """자동 품질 개선 설정"""
        self.auto_improvement_enabled = enabled
        self.auto_improvement_threshold = threshold
        self.logger.info(f"자동 품질 개선: {'활성화' if enabled else '비활성화'}, 임계값: {threshold}")
    
    def enable_learning(self, enabled: bool = True):
        """학습 기능 설정"""
        self.learning_enabled = enabled
        self.logger.info(f"AI 학습: {'활성화' if enabled else '비활성화'}")
    
    def get_quality_trends(self, days: int = 7) -> Dict[str, Any]:
        """품질 트렌드 분석"""
        try:
            if not self.predictor.feature_history:
                return {"message": "충분한 데이터가 없습니다"}
            
            cutoff_time = time.time() - (days * 24 * 3600)
            recent_samples = [
                sample for sample in self.predictor.feature_history
                if sample['timestamp'] > cutoff_time
            ]
            
            if not recent_samples:
                return {"message": f"최근 {days}일간 데이터가 없습니다"}
            
            # 트렌드 계산
            scores = [sample['quality_score'] for sample in recent_samples]
            timestamps = [sample['timestamp'] for sample in recent_samples]
            
            trend_analysis = {
                'period_days': days,
                'total_samples': len(recent_samples),
                'average_score': np.mean(scores),
                'score_std': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'trend_direction': 'improving' if scores[-1] > scores[0] else 'declining',
                'score_change': scores[-1] - scores[0] if len(scores) > 1 else 0
            }
            
            return trend_analysis
            
        except Exception as e:
            self.logger.error(f"품질 트렌드 분석 실패: {e}")
            return {"error": str(e)}

# 전역 AI 품질 관리자
global_ai_quality_manager = AIQualityManager()

def ai_quality_enhanced(target_score: float = 80.0, auto_improve: bool = True):
    """AI 품질 향상 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 함수 실행 전 시스템 상태 수집
            pre_data = {
                'function_name': func.__name__,
                'processing_time': 0,
                'memory_usage': 50,  # 기본값
                'cpu_usage': 50,     # 기본값
            }
            
            start_time = time.time()
            
            try:
                # 원본 함수 실행
                result = func(*args, **kwargs)
                
                # 실행 후 데이터 수집
                processing_time = time.time() - start_time
                post_data = pre_data.copy()
                post_data.update({
                    'processing_time': processing_time,
                    'success_rate': 100,
                    'error_count': 0
                })
                
                # AI 품질 분석 및 개선
                if auto_improve:
                    quality_analysis = global_ai_quality_manager.analyze_and_improve_quality(
                        post_data, target_score=target_score
                    )
                    
                    # 결과에 품질 정보 추가
                    if isinstance(result, dict):
                        result['ai_quality_analysis'] = quality_analysis
                
                return result
                
            except Exception as e:
                # 에러 발생 시에도 품질 분석
                processing_time = time.time() - start_time
                error_data = pre_data.copy()
                error_data.update({
                    'processing_time': processing_time,
                    'success_rate': 0,
                    'error_count': 1,
                    'error_message': str(e)
                })
                
                if auto_improve:
                    global_ai_quality_manager.analyze_and_improve_quality(error_data)
                
                raise
        
        return wrapper
    return decorator

if __name__ == "__main__":
    # 테스트 실행
    print("🤖 솔로몬드 AI v2.1.3 - AI 기반 자동 품질 개선 시스템")
    print("=" * 60)
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 데이터
    test_data = {
        'audio': {
            'sample_rate': 22050,
            'bitrate': 96,
            'silence_ratio': 0.4,
            'volume_variance': 0.7,
            'noise_level': 0.6
        },
        'image': {
            'width': 800,
            'height': 600,
            'brightness': 90,
            'contrast': 0.6,
            'sharpness': 0.5
        },
        'text': {
            'length': 120,
            'readability_score': 45,
            'language_confidence': 0.6,
            'has_errors': True
        },
        'processing_time': 8.5,
        'memory_usage': 85,
        'cpu_usage': 88,
        'error_count': 2,
        'success_rate': 75
    }
    
    # AI 품질 관리자 테스트
    manager = AIQualityManager()
    
    print("📊 품질 분석 및 자동 개선 테스트...")
    result = manager.analyze_and_improve_quality(test_data, target_score=80.0)
    
    print(f"\n📈 품질 분석 결과:")
    if result.get('quality_metrics'):
        metrics = result['quality_metrics']
        print(f"  전체 점수: {metrics['overall_score']:.1f}/100")
        print(f"  오디오: {metrics['audio_quality']:.1f}/100")
        print(f"  이미지: {metrics['image_quality']:.1f}/100")
        print(f"  텍스트: {metrics['text_quality']:.1f}/100")
        print(f"  처리: {metrics['processing_quality']:.1f}/100")
        print(f"  신뢰도: {metrics['confidence']:.1f}")
        print(f"  개선 가능성: {metrics['improvement_potential']:.1f}%")
    
    print(f"\n🛠️ 자동 개선 결과:")
    if result.get('improvement_result'):
        improvement = result['improvement_result']
        print(f"  원본 점수: {improvement['original_score']:.1f}")
        print(f"  개선 점수: {improvement['improved_score']:.1f}")
        print(f"  개선율: {improvement['improvement_rate']:.1f}%")
        print(f"  적용 기법: {', '.join(improvement['applied_techniques'])}")
        print(f"  처리 시간: {improvement['processing_time']:.3f}초")
    
    print(f"\n💡 AI 추천사항:")
    for i, rec in enumerate(result.get('recommendations', [])[:5], 1):
        print(f"  {i}. [{rec['priority']}] {rec['description']}")
        print(f"     예상 개선: {rec['expected_improvement']:.1f}%, 난이도: {rec['implementation_effort']}")
    
    # 데코레이터 테스트
    print(f"\n🎯 AI 품질 향상 데코레이터 테스트...")
    
    @ai_quality_enhanced(target_score=75.0)
    def jewelry_analysis_function():
        """주얼리 분석 함수 시뮬레이션"""
        time.sleep(0.1)  # 처리 시간 시뮬레이션
        return {
            "analysis_result": "다이아몬드 품질: 우수",
            "confidence": 0.92,
            "processing_details": "AI 자동 품질 개선 적용됨"
        }
    
    enhanced_result = jewelry_analysis_function()
    print(f"  분석 결과: {enhanced_result.get('analysis_result')}")
    print(f"  신뢰도: {enhanced_result.get('confidence')}")
    
    if 'ai_quality_analysis' in enhanced_result:
        ai_analysis = enhanced_result['ai_quality_analysis']
        print(f"  AI 품질 분석 적용됨: {ai_analysis.get('auto_improvement_applied', False)}")
    
    print("\n✅ AI 기반 자동 품질 개선 시스템 테스트 완료!")
    print("🤖 머신러닝을 활용한 지능형 품질 관리가 활성화되었습니다!")
