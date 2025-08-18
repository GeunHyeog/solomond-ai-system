#!/usr/bin/env python3
"""
지능형 예측 알고리즘 엔진 v2.6
AI 모델별 세분화 및 처리 시간 최적화
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
from collections import deque
import threading
import psutil

@dataclass
class ProcessingTimeMetrics:
    """처리 시간 메트릭 데이터 클래스"""
    model_name: str
    file_type: str
    file_size_mb: float
    processing_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: str
    success: bool
    error_message: Optional[str] = None

@dataclass
class PredictionResult:
    """예측 결과 데이터 클래스"""
    model_name: str
    file_type: str
    file_size_mb: float
    predicted_time_ms: float
    confidence_score: float
    predicted_memory_mb: float
    predicted_cpu_percent: float
    optimization_suggestions: List[str]
    timestamp: str

@dataclass
class ModelPerformanceProfile:
    """모델 성능 프로필 데이터 클래스"""
    model_name: str
    base_processing_rate: float  # ms per MB
    memory_efficiency: float     # MB per processing MB
    cpu_intensity: float         # CPU% during processing
    error_rate: float           # 0.0-1.0
    last_updated: str
    sample_count: int
    confidence_level: float

class IntelligentPredictionEngine:
    """지능형 예측 알고리즘 엔진"""
    
    def __init__(self, history_retention_hours: int = 168):  # 1주일
        self.history_retention_hours = history_retention_hours
        self.logger = self._setup_logging()
        
        # 메트릭 저장소
        self.processing_history = deque(maxlen=10000)
        
        # AI 모델별 성능 프로필 (Ollama 7개 모델 + 기타)
        self.model_profiles = self._initialize_model_profiles()
        
        # 파일 타입별 기본 처리율
        self.file_type_rates = {
            'audio/wav': {'base_rate': 8.0, 'complexity_factor': 1.2},
            'audio/mp3': {'base_rate': 6.0, 'complexity_factor': 1.0},
            'audio/m4a': {'base_rate': 10.0, 'complexity_factor': 1.5},
            'image/jpeg': {'base_rate': 2.0, 'complexity_factor': 0.8},
            'image/png': {'base_rate': 3.0, 'complexity_factor': 1.0},
            'video/mp4': {'base_rate': 15.0, 'complexity_factor': 2.0},
            'text/plain': {'base_rate': 0.5, 'complexity_factor': 0.3}
        }
        
        # 예측 정확도 추적
        self.prediction_accuracy = {
            'total_predictions': 0,
            'accurate_predictions': 0,
            'accuracy_threshold': 0.2,  # 20% 오차 허용
            'current_accuracy': 0.0
        }
        
        # 최적화 추천 엔진
        self.optimization_rules = self._initialize_optimization_rules()
        
        # 스레드 안전성
        self.lock = threading.Lock()
        
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.IntelligentPredictionEngine')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_model_profiles(self) -> Dict[str, ModelPerformanceProfile]:
        """AI 모델별 성능 프로필 초기화"""
        profiles = {}
        
        # Ollama 7개 모델 프로필 (사용자 실제 모델 기반)
        ollama_models = {
            'gemma3:27b': {
                'base_processing_rate': 25.0,  # 고품질이지만 느림
                'memory_efficiency': 3.5,
                'cpu_intensity': 85.0,
                'error_rate': 0.02
            },
            'qwen3:8b': {
                'base_processing_rate': 12.0,  # 균형잡힌 성능
                'memory_efficiency': 2.0,
                'cpu_intensity': 60.0,
                'error_rate': 0.03
            },
            'qwen2.5:7b': {
                'base_processing_rate': 10.0,  # 빠른 처리
                'memory_efficiency': 1.8,
                'cpu_intensity': 55.0,
                'error_rate': 0.04
            },
            'gemma3:4b': {
                'base_processing_rate': 8.0,   # 가장 빠름
                'memory_efficiency': 1.2,
                'cpu_intensity': 40.0,
                'error_rate': 0.05
            },
            'solar:latest': {
                'base_processing_rate': 15.0,
                'memory_efficiency': 2.5,
                'cpu_intensity': 70.0,
                'error_rate': 0.03
            },
            'mistral:latest': {
                'base_processing_rate': 14.0,
                'memory_efficiency': 2.2,
                'cpu_intensity': 65.0,
                'error_rate': 0.04
            },
            'llama3.2:latest': {
                'base_processing_rate': 18.0,
                'memory_efficiency': 2.8,
                'cpu_intensity': 75.0,
                'error_rate': 0.03
            }
        }
        
        # 전통적 분석 도구들
        traditional_models = {
            'whisper_cpu': {
                'base_processing_rate': 8.0,
                'memory_efficiency': 1.0,
                'cpu_intensity': 45.0,
                'error_rate': 0.01
            },
            'easyocr': {
                'base_processing_rate': 2.0,
                'memory_efficiency': 0.8,
                'cpu_intensity': 30.0,
                'error_rate': 0.02
            },
            'transformers_cpu': {
                'base_processing_rate': 20.0,
                'memory_efficiency': 2.0,
                'cpu_intensity': 70.0,
                'error_rate': 0.02
            }
        }
        
        # 프로필 생성
        all_models = {**ollama_models, **traditional_models}
        
        for model_name, config in all_models.items():
            profiles[model_name] = ModelPerformanceProfile(
                model_name=model_name,
                base_processing_rate=config['base_processing_rate'],
                memory_efficiency=config['memory_efficiency'],
                cpu_intensity=config['cpu_intensity'],
                error_rate=config['error_rate'],
                last_updated=datetime.now().isoformat(),
                sample_count=0,
                confidence_level=0.5  # 초기 신뢰도
            )
        
        return profiles
    
    def _initialize_optimization_rules(self) -> List[Dict[str, Any]]:
        """최적화 규칙 초기화"""
        return [
            {
                'condition': 'high_memory_usage',
                'threshold': 80.0,
                'suggestions': [
                    '메모리 사용량이 높습니다. 더 작은 모델 사용을 고려하세요',
                    'GEMMA3:4B 또는 QWEN2.5:7B 모델로 전환을 권장합니다',
                    '파일을 더 작은 단위로 분할하여 처리하세요'
                ]
            },
            {
                'condition': 'slow_processing',
                'threshold': 30000.0,  # 30초
                'suggestions': [
                    '처리 시간이 깁니다. 더 빠른 모델을 사용하세요',
                    'GEMMA3:4B 모델이 가장 빠른 처리 속도를 제공합니다',
                    'CPU 모드 대신 GPU 사용을 검토하세요'
                ]
            },
            {
                'condition': 'high_cpu_usage',
                'threshold': 90.0,
                'suggestions': [
                    'CPU 사용률이 높습니다. 병렬 처리를 줄이세요',
                    '다른 프로세스를 종료하고 다시 시도하세요',
                    '더 가벼운 모델(QWEN2.5:7B)을 사용하세요'
                ]
            },
            {
                'condition': 'large_file_size',
                'threshold': 100.0,  # 100MB
                'suggestions': [
                    '대용량 파일입니다. 스트리밍 처리를 권장합니다',
                    '파일을 10MB 단위로 분할하여 처리하세요',
                    'GEMMA3:27B 대신 QWEN3:8B 사용을 권장합니다'
                ]
            },
            {
                'condition': 'frequent_errors',
                'threshold': 0.1,  # 10% 에러율
                'suggestions': [
                    '에러가 자주 발생합니다. 파일 형식을 확인하세요',
                    '더 안정적인 모델(Whisper, EasyOCR)을 사용하세요',
                    '시스템 리소스가 충분한지 확인하세요'
                ]
            }
        ]
    
    def record_processing_metrics(self, metrics: ProcessingTimeMetrics) -> None:
        """처리 메트릭 기록"""
        with self.lock:
            self.processing_history.append(metrics)
            
            # 모델 프로필 업데이트
            if metrics.model_name in self.model_profiles:
                self._update_model_profile(metrics)
            
            self.logger.info(f"📊 메트릭 기록: {metrics.model_name} - {metrics.processing_time_ms:.1f}ms")
    
    def _update_model_profile(self, metrics: ProcessingTimeMetrics) -> None:
        """모델 프로필 업데이트 (이동 평균)"""
        profile = self.model_profiles[metrics.model_name]
        
        if metrics.success and metrics.file_size_mb > 0:
            # 처리율 업데이트 (이동 평균)
            current_rate = metrics.processing_time_ms / metrics.file_size_mb
            alpha = 0.1  # 학습률
            
            profile.base_processing_rate = (
                (1 - alpha) * profile.base_processing_rate + 
                alpha * current_rate
            )
            
            # 메모리 효율성 업데이트
            memory_rate = metrics.memory_usage_mb / metrics.file_size_mb
            profile.memory_efficiency = (
                (1 - alpha) * profile.memory_efficiency + 
                alpha * memory_rate
            )
            
            # CPU 강도 업데이트
            profile.cpu_intensity = (
                (1 - alpha) * profile.cpu_intensity + 
                alpha * metrics.cpu_usage_percent
            )
        
        # 에러율 업데이트
        if not metrics.success:
            profile.error_rate = min(1.0, profile.error_rate + 0.01)
        else:
            profile.error_rate = max(0.0, profile.error_rate - 0.001)
        
        # 샘플 수 및 신뢰도 업데이트
        profile.sample_count += 1
        profile.confidence_level = min(1.0, profile.sample_count / 100.0)
        profile.last_updated = datetime.now().isoformat()
    
    def predict_processing_time(self, 
                              model_name: str, 
                              file_type: str, 
                              file_size_mb: float,
                              current_system_load: Optional[Dict] = None) -> PredictionResult:
        """처리 시간 예측"""
        
        # 시스템 부하 정보 가져오기
        if current_system_load is None:
            current_system_load = self._get_current_system_load()
        
        # 기본 예측 계산
        base_prediction = self._calculate_base_prediction(model_name, file_type, file_size_mb)
        
        # 시스템 부하 보정
        load_adjusted_prediction = self._adjust_for_system_load(base_prediction, current_system_load)
        
        # 파일 크기 보정
        size_adjusted_prediction = self._adjust_for_file_size(load_adjusted_prediction, file_size_mb)
        
        # 신뢰도 계산
        confidence_score = self._calculate_prediction_confidence(model_name, file_type)
        
        # 최적화 제안 생성
        optimization_suggestions = self._generate_optimization_suggestions(
            model_name, file_type, file_size_mb, size_adjusted_prediction, current_system_load
        )
        
        result = PredictionResult(
            model_name=model_name,
            file_type=file_type,
            file_size_mb=file_size_mb,
            predicted_time_ms=size_adjusted_prediction['processing_time_ms'],
            confidence_score=confidence_score,
            predicted_memory_mb=size_adjusted_prediction['memory_usage_mb'],
            predicted_cpu_percent=size_adjusted_prediction['cpu_usage_percent'],
            optimization_suggestions=optimization_suggestions,
            timestamp=datetime.now().isoformat()
        )
        
        self.logger.info(f"🔮 예측 완료: {model_name} - {file_size_mb:.1f}MB → {result.predicted_time_ms:.1f}ms")
        return result
    
    def _calculate_base_prediction(self, model_name: str, file_type: str, file_size_mb: float) -> Dict[str, float]:
        """기본 예측 계산"""
        # 모델 프로필 가져오기
        if model_name in self.model_profiles:
            profile = self.model_profiles[model_name]
        else:
            # 기본 프로필 사용
            profile = ModelPerformanceProfile(
                model_name="default",
                base_processing_rate=15.0,
                memory_efficiency=2.0,
                cpu_intensity=60.0,
                error_rate=0.05,
                last_updated=datetime.now().isoformat(),
                sample_count=0,
                confidence_level=0.3
            )
        
        # 파일 타입별 보정 인수
        file_config = self.file_type_rates.get(file_type, {
            'base_rate': 10.0, 
            'complexity_factor': 1.0
        })
        
        # 기본 처리 시간 계산
        base_time = profile.base_processing_rate * file_size_mb * file_config['complexity_factor']
        
        # 메모리 사용량 예측
        memory_usage = profile.memory_efficiency * file_size_mb
        
        # CPU 사용률 예측
        cpu_usage = profile.cpu_intensity
        
        return {
            'processing_time_ms': base_time,
            'memory_usage_mb': memory_usage,
            'cpu_usage_percent': cpu_usage
        }
    
    def _adjust_for_system_load(self, base_prediction: Dict[str, float], system_load: Dict) -> Dict[str, float]:
        """시스템 부하에 따른 예측 보정"""
        adjusted = base_prediction.copy()
        
        # CPU 부하 보정
        cpu_load_factor = 1.0 + (system_load.get('cpu_percent', 0) / 100.0)
        adjusted['processing_time_ms'] *= cpu_load_factor
        
        # 메모리 부하 보정
        memory_load_factor = 1.0 + (system_load.get('memory_percent', 0) / 200.0)
        adjusted['memory_usage_mb'] *= memory_load_factor
        
        # 디스크 I/O 보정
        disk_load_factor = 1.0 + (system_load.get('disk_usage', 0) / 300.0)
        adjusted['processing_time_ms'] *= disk_load_factor
        
        return adjusted
    
    def _adjust_for_file_size(self, prediction: Dict[str, float], file_size_mb: float) -> Dict[str, float]:
        """파일 크기에 따른 비선형 보정"""
        adjusted = prediction.copy()
        
        # 대용량 파일에 대한 비선형 보정
        if file_size_mb > 50:
            # 50MB 이상에서는 처리 시간이 비선형적으로 증가
            size_penalty = 1.0 + ((file_size_mb - 50) / 100.0) * 0.3
            adjusted['processing_time_ms'] *= size_penalty
            
        elif file_size_mb < 1:
            # 소용량 파일에 대한 최소 처리 시간 보장
            adjusted['processing_time_ms'] = max(adjusted['processing_time_ms'], 500.0)
        
        return adjusted
    
    def _calculate_prediction_confidence(self, model_name: str, file_type: str) -> float:
        """예측 신뢰도 계산"""
        base_confidence = 0.5
        
        # 모델 프로필 기반 신뢰도
        if model_name in self.model_profiles:
            profile = self.model_profiles[model_name]
            model_confidence = profile.confidence_level
        else:
            model_confidence = 0.3
        
        # 파일 타입 기반 신뢰도
        if file_type in self.file_type_rates:
            type_confidence = 0.8
        else:
            type_confidence = 0.4
        
        # 전체 예측 정확도 기반 신뢰도
        accuracy_confidence = self.prediction_accuracy['current_accuracy']
        
        # 가중 평균 계산
        final_confidence = (
            0.4 * model_confidence + 
            0.3 * type_confidence + 
            0.3 * accuracy_confidence
        )
        
        return min(1.0, max(0.1, final_confidence))
    
    def _get_current_system_load(self) -> Dict[str, float]:
        """현재 시스템 부하 정보"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0.0
            }
        except Exception as e:
            self.logger.warning(f"시스템 부하 측정 실패: {e}")
            return {'cpu_percent': 50.0, 'memory_percent': 50.0, 'disk_usage': 30.0}
    
    def _generate_optimization_suggestions(self, 
                                         model_name: str, 
                                         file_type: str, 
                                         file_size_mb: float,
                                         prediction: Dict[str, float],
                                         system_load: Dict) -> List[str]:
        """최적화 제안 생성"""
        suggestions = []
        
        # 규칙 기반 제안
        for rule in self.optimization_rules:
            condition = rule['condition']
            threshold = rule['threshold']
            
            if condition == 'high_memory_usage' and prediction['memory_usage_mb'] > threshold:
                suggestions.extend(rule['suggestions'])
            elif condition == 'slow_processing' and prediction['processing_time_ms'] > threshold:
                suggestions.extend(rule['suggestions'])
            elif condition == 'high_cpu_usage' and system_load.get('cpu_percent', 0) > threshold:
                suggestions.extend(rule['suggestions'])
            elif condition == 'large_file_size' and file_size_mb > threshold:
                suggestions.extend(rule['suggestions'])
        
        # 모델별 특화 제안
        if model_name == 'gemma3:27b' and file_size_mb > 50:
            suggestions.append("GEMMA3:27B는 대용량 파일에서 메모리를 많이 사용합니다. QWEN3:8B 사용을 권장합니다")
        
        if model_name == 'gemma3:4b' and prediction['processing_time_ms'] < 2000:
            suggestions.append("작은 파일에는 GEMMA3:4B가 매우 효율적입니다")
        
        # 중복 제거 및 상위 5개만 반환
        unique_suggestions = list(set(suggestions))
        return unique_suggestions[:5]
    
    def validate_prediction(self, prediction: PredictionResult, actual_metrics: ProcessingTimeMetrics) -> float:
        """예측 정확도 검증"""
        # 처리 시간 오차 계산
        time_error = abs(prediction.predicted_time_ms - actual_metrics.processing_time_ms) / actual_metrics.processing_time_ms
        
        # 정확도 업데이트
        with self.lock:
            self.prediction_accuracy['total_predictions'] += 1
            
            if time_error <= self.prediction_accuracy['accuracy_threshold']:
                self.prediction_accuracy['accurate_predictions'] += 1
            
            # 현재 정확도 계산
            if self.prediction_accuracy['total_predictions'] > 0:
                self.prediction_accuracy['current_accuracy'] = (
                    self.prediction_accuracy['accurate_predictions'] / 
                    self.prediction_accuracy['total_predictions']
                )
        
        self.logger.info(f"🎯 예측 검증: 오차 {time_error:.2%}, 전체 정확도 {self.prediction_accuracy['current_accuracy']:.2%}")
        
        return time_error
    
    def get_model_recommendations(self, file_type: str, file_size_mb: float, priority: str = 'balanced') -> List[Dict[str, Any]]:
        """상황별 최적 모델 추천"""
        recommendations = []
        
        # 모든 모델에 대해 예측 수행
        for model_name in self.model_profiles.keys():
            prediction = self.predict_processing_time(model_name, file_type, file_size_mb)
            
            recommendations.append({
                'model_name': model_name,
                'predicted_time_ms': prediction.predicted_time_ms,
                'predicted_memory_mb': prediction.predicted_memory_mb,
                'confidence_score': prediction.confidence_score,
                'profile': self.model_profiles[model_name]
            })
        
        # 우선순위에 따른 정렬
        if priority == 'speed':
            recommendations.sort(key=lambda x: x['predicted_time_ms'])
        elif priority == 'memory':
            recommendations.sort(key=lambda x: x['predicted_memory_mb'])
        elif priority == 'accuracy':
            recommendations.sort(key=lambda x: x['profile'].error_rate)
        else:  # balanced
            # 종합 점수 계산 (시간, 메모리, 정확도의 가중 평균)
            for rec in recommendations:
                # 정규화 점수 (0-1)
                time_score = 1.0 / (1.0 + rec['predicted_time_ms'] / 10000.0)
                memory_score = 1.0 / (1.0 + rec['predicted_memory_mb'] / 1000.0)
                accuracy_score = 1.0 - rec['profile'].error_rate
                confidence_score = rec['confidence_score']
                
                rec['overall_score'] = (
                    0.3 * time_score + 
                    0.2 * memory_score + 
                    0.3 * accuracy_score + 
                    0.2 * confidence_score
                )
            
            recommendations.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return recommendations[:3]  # 상위 3개만 반환
    
    def export_performance_data(self, output_path: str) -> None:
        """성능 데이터 내보내기"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'model_profiles': {name: asdict(profile) for name, profile in self.model_profiles.items()},
            'file_type_rates': self.file_type_rates,
            'prediction_accuracy': self.prediction_accuracy,
            'processing_history_count': len(self.processing_history),
            'recent_metrics': [asdict(metrics) for metrics in list(self.processing_history)[-100:]]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📊 성능 데이터 내보내기 완료: {output_path}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보"""
        return {
            'total_models': len(self.model_profiles),
            'supported_file_types': len(self.file_type_rates),
            'processing_history_count': len(self.processing_history),
            'prediction_accuracy': self.prediction_accuracy['current_accuracy'],
            'most_accurate_model': max(
                self.model_profiles.items(), 
                key=lambda x: x[1].confidence_level
            )[0] if self.model_profiles else 'none',
            'fastest_model': min(
                self.model_profiles.items(), 
                key=lambda x: x[1].base_processing_rate
            )[0] if self.model_profiles else 'none'
        }

# 전역 예측 엔진 인스턴스
_global_prediction_engine = None

def get_global_prediction_engine() -> IntelligentPredictionEngine:
    """전역 예측 엔진 인스턴스 반환"""
    global _global_prediction_engine
    if _global_prediction_engine is None:
        _global_prediction_engine = IntelligentPredictionEngine()
    return _global_prediction_engine

# 사용 예시
if __name__ == "__main__":
    engine = IntelligentPredictionEngine()
    
    # 테스트 예측
    prediction = engine.predict_processing_time('qwen3:8b', 'audio/wav', 25.0)
    
    print("🔮 지능형 예측 결과:")
    print(f"모델: {prediction.model_name}")
    print(f"파일: {prediction.file_type} ({prediction.file_size_mb}MB)")
    print(f"예상 처리 시간: {prediction.predicted_time_ms:.1f}ms")
    print(f"예상 메모리 사용: {prediction.predicted_memory_mb:.1f}MB")
    print(f"신뢰도: {prediction.confidence_score:.2%}")
    print(f"최적화 제안: {len(prediction.optimization_suggestions)}개")
    
    # 모델 추천
    recommendations = engine.get_model_recommendations('audio/wav', 25.0, 'balanced')
    print(f"\n🏆 추천 모델 (상위 {len(recommendations)}개):")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['model_name']} - {rec['predicted_time_ms']:.1f}ms")