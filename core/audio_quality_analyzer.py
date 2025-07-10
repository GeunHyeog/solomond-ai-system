"""
주얼리 AI 플랫폼 v2.1 - 음성 품질 분석 엔진
==============================================

실시간 음성 품질 분석, 노이즈 측정, 명료도 평가, 배경소음 분류
주얼리 업계 회의/세미나/강의 환경에 특화된 품질 예측 시스템

Author: 전근혁 (solomond.jgh@gmail.com)
Created: 2025.07.10
Version: 2.1.0
"""

import numpy as np
import librosa
import librosa.display
import scipy.signal
import scipy.stats
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# 음성 처리 라이브러리
try:
    import noisereduce as nr
    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    NOISE_REDUCE_AVAILABLE = False
    
try:
    import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    import pystoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioQualityAnalyzer:
    """
    음성 품질 종합 분석 엔진
    
    주요 기능:
    - 실시간 노이즈 레벨 측정 (dB 단위)
    - 음성 명료도 점수 계산 (0-100점)
    - 배경소음 환경 자동 분류
    - STT 정확도 사전 예측
    - 품질 개선 제안 생성
    """
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        """
        AudioQualityAnalyzer 초기화
        
        Args:
            sample_rate: 샘플링 레이트 (기본: 16kHz)
            chunk_size: 분석 청크 크기 (기본: 1024)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.analysis_results = {}
        
        # 주얼리 업계 특화 환경 임계값
        self.quality_thresholds = {
            'excellent': {'snr': 25, 'clarity': 85, 'noise_db': -40},
            'good': {'snr': 15, 'clarity': 70, 'noise_db': -30}, 
            'fair': {'snr': 10, 'clarity': 55, 'noise_db': -20},
            'poor': {'snr': 5, 'clarity': 40, 'noise_db': -10}
        }
        
        # 배경소음 환경별 특성
        self.environment_profiles = {
            'office': {
                'freq_range': (200, 2000),
                'typical_snr': (15, 25),
                'noise_pattern': 'constant_low'
            },
            'exhibition_hall': {
                'freq_range': (300, 3000), 
                'typical_snr': (8, 18),
                'noise_pattern': 'variable_high'
            },
            'conference_room': {
                'freq_range': (250, 2500),
                'typical_snr': (18, 30),
                'noise_pattern': 'constant_medium'
            },
            'auditorium': {
                'freq_range': (100, 4000),
                'typical_snr': (5, 15),
                'noise_pattern': 'reverb_high'
            },
            'outdoor': {
                'freq_range': (50, 5000),
                'typical_snr': (0, 12),
                'noise_pattern': 'wind_traffic'
            }
        }
        
        logger.info("AudioQualityAnalyzer 초기화 완료")
    
    def analyze_audio_file(self, audio_path: str) -> Dict:
        """
        오디오 파일 종합 품질 분석
        
        Args:
            audio_path: 분석할 오디오 파일 경로
            
        Returns:
            Dict: 종합 품질 분석 결과
        """
        try:
            # 오디오 로드
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            logger.info(f"오디오 파일 로드 완료: {audio_path}")
            logger.info(f"길이: {len(y)/sr:.2f}초, 샘플레이트: {sr}Hz")
            
            # 종합 분석 실행
            analysis_result = self._comprehensive_analysis(y, sr)
            
            # 결과 저장
            self.analysis_results[audio_path] = analysis_result
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"오디오 분석 오류: {str(e)}")
            return self._create_error_result(str(e))
    
    def _comprehensive_analysis(self, y: np.ndarray, sr: int) -> Dict:
        """
        오디오 신호 종합 분석
        
        Args:
            y: 오디오 신호 배열
            sr: 샘플링 레이트
            
        Returns:
            Dict: 종합 분석 결과
        """
        analysis = {}
        
        # 1. 기본 오디오 정보
        analysis['basic_info'] = self._analyze_basic_info(y, sr)
        
        # 2. 노이즈 레벨 측정
        analysis['noise_analysis'] = self._analyze_noise_level(y, sr)
        
        # 3. 음성 명료도 분석
        analysis['clarity_analysis'] = self._analyze_clarity(y, sr)
        
        # 4. 주파수 도메인 분석
        analysis['frequency_analysis'] = self._analyze_frequency_domain(y, sr)
        
        # 5. 배경소음 환경 분류
        analysis['environment_classification'] = self._classify_environment(y, sr)
        
        # 6. STT 정확도 예측
        analysis['stt_prediction'] = self._predict_stt_accuracy(analysis)
        
        # 7. 품질 종합 평가
        analysis['overall_quality'] = self._calculate_overall_quality(analysis)
        
        # 8. 개선 제안 생성
        analysis['improvement_suggestions'] = self._generate_improvement_suggestions(analysis)
        
        # 9. 실시간 경고 체크
        analysis['quality_warnings'] = self._check_quality_warnings(analysis)
        
        return analysis
    
    def _analyze_basic_info(self, y: np.ndarray, sr: int) -> Dict:
        """기본 오디오 정보 분석"""
        return {
            'duration': len(y) / sr,
            'sample_rate': sr,
            'channels': 1,  # 모노로 로드됨
            'total_samples': len(y),
            'rms_energy': np.sqrt(np.mean(y**2)),
            'peak_amplitude': np.max(np.abs(y)),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y))
        }
    
    def _analyze_noise_level(self, y: np.ndarray, sr: int) -> Dict:
        """
        노이즈 레벨 정량적 측정
        - dB 단위 노이즈 레벨
        - 신호 대 잡음비(SNR)
        - 노이즈 분포 분석
        """
        # RMS 기반 노이즈 레벨 계산
        rms_values = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        
        # dB 변환 (참조: 1.0 = 0dB)
        rms_db = 20 * np.log10(rms_values + 1e-10)  # 로그(0) 방지
        
        # 노이즈 플로어 추정 (하위 10% percentile)
        noise_floor_db = np.percentile(rms_db, 10)
        
        # 신호 레벨 추정 (상위 10% percentile)
        signal_level_db = np.percentile(rms_db, 90)
        
        # SNR 계산
        snr = signal_level_db - noise_floor_db
        
        # 스펙트럴 중심 기반 노이즈 분석
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        return {
            'noise_floor_db': float(noise_floor_db),
            'signal_level_db': float(signal_level_db),
            'snr_db': float(snr),
            'rms_mean_db': float(np.mean(rms_db)),
            'rms_std_db': float(np.std(rms_db)),
            'spectral_noise_indicator': float(np.std(spectral_centroids)),
            'noise_consistency': float(1.0 - np.std(rms_db) / (np.mean(rms_db) + 1e-10)),
            'noise_grade': self._grade_noise_level(snr, noise_floor_db)
        }
    
    def _analyze_clarity(self, y: np.ndarray, sr: int) -> Dict:
        """
        음성 명료도 분석 (0-100점)
        - 스펙트럴 명료도
        - 포만트 명확성
        - 조화 구조 분석
        """
        # 스펙트럴 특성 분석
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # 스펙트럴 명료도 점수 (0-100)
        spectral_clarity = 100 * (1 - np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-10))
        spectral_clarity = np.clip(spectral_clarity, 0, 100)
        
        # 대역폭 기반 명료도
        bandwidth_clarity = 100 * (1 - np.mean(spectral_bandwidth) / (sr/4))  # Nyquist 기준
        bandwidth_clarity = np.clip(bandwidth_clarity, 0, 100)
        
        # 스펙트럴 대비 명료도
        contrast_clarity = np.mean(spectral_contrast) * 20  # 정규화
        contrast_clarity = np.clip(contrast_clarity, 0, 100)
        
        # MFCC 기반 음성 특성
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_clarity = 100 * (1 - np.std(mfccs) / (np.mean(np.abs(mfccs)) + 1e-10))
        mfcc_clarity = np.clip(mfcc_clarity, 0, 100)
        
        # 조화 구조 분석
        harmonic, percussive = librosa.effects.hpss(y)
        harmonic_ratio = np.sum(harmonic**2) / (np.sum(y**2) + 1e-10)
        harmonic_clarity = harmonic_ratio * 100
        
        # 종합 명료도 점수
        overall_clarity = np.mean([
            spectral_clarity * 0.3,
            bandwidth_clarity * 0.2, 
            contrast_clarity * 0.2,
            mfcc_clarity * 0.2,
            harmonic_clarity * 0.1
        ])
        
        return {
            'spectral_clarity': float(spectral_clarity),
            'bandwidth_clarity': float(bandwidth_clarity),
            'contrast_clarity': float(contrast_clarity),
            'mfcc_clarity': float(mfcc_clarity),
            'harmonic_clarity': float(harmonic_clarity),
            'overall_clarity_score': float(overall_clarity),
            'clarity_grade': self._grade_clarity(overall_clarity)
        }
    
    def _analyze_frequency_domain(self, y: np.ndarray, sr: int) -> Dict:
        """주파수 도메인 상세 분석"""
        # FFT 분석
        fft_values = fft(y)
        freqs = fftfreq(len(fft_values), 1/sr)
        magnitude = np.abs(fft_values)
        
        # 주요 주파수 대역 에너지 분석
        bands = {
            'low': (0, 500),      # 저주파 (배경소음)
            'mid': (500, 2000),   # 중주파 (음성 주파수)
            'high': (2000, 8000)  # 고주파 (자음, 명료도)
        }
        
        band_energies = {}
        for band_name, (low_freq, high_freq) in bands.items():
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_energies[f'{band_name}_energy'] = float(np.sum(magnitude[mask]))
        
        # 스펙트럴 특성
        spectral_features = {
            'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
            'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
            'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
            'spectral_flatness': float(np.mean(librosa.feature.spectral_flatness(y=y)))
        }
        
        return {**band_energies, **spectral_features}
    
    def _classify_environment(self, y: np.ndarray, sr: int) -> Dict:
        """
        배경소음 환경 자동 분류
        - 사무실, 전시장, 회의실, 강당, 야외 등
        """
        # 각 환경별 특성 점수 계산
        environment_scores = {}
        
        for env_name, profile in self.environment_profiles.items():
            score = self._calculate_environment_score(y, sr, profile)
            environment_scores[env_name] = score
        
        # 가장 높은 점수의 환경 선택
        predicted_environment = max(environment_scores, key=environment_scores.get)
        confidence = environment_scores[predicted_environment]
        
        # 환경별 특화 분석
        env_analysis = self._analyze_environment_specific(y, sr, predicted_environment)
        
        return {
            'predicted_environment': predicted_environment,
            'confidence': float(confidence),
            'environment_scores': environment_scores,
            'environment_analysis': env_analysis,
            'environment_recommendations': self._get_environment_recommendations(predicted_environment)
        }
    
    def _calculate_environment_score(self, y: np.ndarray, sr: int, profile: Dict) -> float:
        """특정 환경 프로필과의 매칭 점수 계산"""
        # 주파수 범위 매칭
        freq_low, freq_high = profile['freq_range']
        
        # 해당 주파수 대역의 에너지 비율
        fft_values = fft(y)
        freqs = fftfreq(len(fft_values), 1/sr)
        magnitude = np.abs(fft_values)
        
        target_mask = (freqs >= freq_low) & (freqs <= freq_high)
        target_energy = np.sum(magnitude[target_mask])
        total_energy = np.sum(magnitude) + 1e-10
        
        freq_score = target_energy / total_energy
        
        # SNR 매칭
        noise_analysis = self._analyze_noise_level(y, sr)
        snr = noise_analysis['snr_db']
        snr_low, snr_high = profile['typical_snr']
        
        if snr_low <= snr <= snr_high:
            snr_score = 1.0
        else:
            snr_score = max(0, 1.0 - abs(snr - np.mean([snr_low, snr_high])) / 20)
        
        # 노이즈 패턴 매칭 (기본 구현)
        pattern_score = 0.5  # 향후 고도화 예정
        
        # 종합 점수
        return freq_score * 0.5 + snr_score * 0.3 + pattern_score * 0.2
    
    def _analyze_environment_specific(self, y: np.ndarray, sr: int, environment: str) -> Dict:
        """환경별 특화 분석"""
        if environment == 'exhibition_hall':
            return {
                'crowd_noise_level': self._analyze_crowd_noise(y, sr),
                'reverberation_time': self._estimate_reverberation(y, sr),
                'ambient_noise_type': 'high_variable'
            }
        elif environment == 'conference_room':
            return {
                'room_acoustics': self._analyze_room_acoustics(y, sr),
                'echo_presence': self._detect_echo(y, sr),
                'ambient_noise_type': 'low_constant'
            }
        # 다른 환경들도 추가 구현 예정
        else:
            return {'general_analysis': True}
    
    def _predict_stt_accuracy(self, analysis: Dict) -> Dict:
        """
        STT 정확도 사전 예측
        - 음성 품질 기반 예측 모델
        """
        # 주요 품질 지표 추출
        snr = analysis['noise_analysis']['snr_db']
        clarity = analysis['clarity_analysis']['overall_clarity_score']
        noise_floor = analysis['noise_analysis']['noise_floor_db']
        
        # 환경 보정 팩터
        environment = analysis['environment_classification']['predicted_environment']
        env_factor = {
            'office': 1.0,
            'conference_room': 0.95,
            'exhibition_hall': 0.8,
            'auditorium': 0.85,
            'outdoor': 0.7
        }.get(environment, 0.9)
        
        # STT 정확도 예측 (경험적 모델)
        base_accuracy = 50 + (snr * 1.5) + (clarity * 0.3) + abs(noise_floor * 0.5)
        predicted_accuracy = base_accuracy * env_factor
        predicted_accuracy = np.clip(predicted_accuracy, 10, 95)  # 10-95% 범위
        
        # 신뢰도 계산
        confidence_factors = [
            snr > 15,  # 좋은 SNR
            clarity > 70,  # 높은 명료도
            noise_floor < -30,  # 낮은 노이즈 플로어
            environment in ['office', 'conference_room']  # 좋은 환경
        ]
        
        confidence = sum(confidence_factors) / len(confidence_factors)
        
        return {
            'predicted_accuracy': float(predicted_accuracy),
            'confidence': float(confidence),
            'factors': {
                'snr_contribution': float(snr * 1.5),
                'clarity_contribution': float(clarity * 0.3),
                'noise_contribution': float(abs(noise_floor * 0.5)),
                'environment_factor': float(env_factor)
            },
            'accuracy_grade': self._grade_stt_prediction(predicted_accuracy)
        }
    
    def _calculate_overall_quality(self, analysis: Dict) -> Dict:
        """종합 품질 점수 계산 (0-100)"""
        # 가중치 설정
        weights = {
            'noise_quality': 0.3,      # 노이즈 품질
            'clarity_quality': 0.3,    # 명료도
            'environment_quality': 0.2, # 환경 적합성
            'stt_prediction': 0.2      # STT 예측 정확도
        }
        
        # 각 항목별 점수 정규화
        scores = {
            'noise_quality': self._normalize_snr_to_score(analysis['noise_analysis']['snr_db']),
            'clarity_quality': analysis['clarity_analysis']['overall_clarity_score'],
            'environment_quality': analysis['environment_classification']['confidence'] * 100,
            'stt_prediction': analysis['stt_prediction']['predicted_accuracy']
        }
        
        # 가중 평균 계산
        overall_score = sum(scores[key] * weights[key] for key in weights.keys())
        overall_score = np.clip(overall_score, 0, 100)
        
        return {
            'overall_score': float(overall_score),
            'component_scores': scores,
            'weights_used': weights,
            'quality_grade': self._grade_overall_quality(overall_score),
            'quality_level': self._get_quality_level(overall_score)
        }
    
    def _generate_improvement_suggestions(self, analysis: Dict) -> List[Dict]:
        """품질 개선 제안 생성"""
        suggestions = []
        
        # 노이즈 관련 제안
        noise_analysis = analysis['noise_analysis']
        if noise_analysis['snr_db'] < 15:
            suggestions.append({
                'category': 'noise_reduction',
                'priority': 'high',
                'issue': f"SNR이 {noise_analysis['snr_db']:.1f}dB로 낮습니다",
                'suggestion': "마이크를 화자에게 더 가깝게 배치하거나 배경소음을 줄여주세요",
                'expected_improvement': "STT 정확도 10-15% 향상"
            })
        
        # 명료도 관련 제안  
        clarity_analysis = analysis['clarity_analysis']
        if clarity_analysis['overall_clarity_score'] < 70:
            suggestions.append({
                'category': 'clarity_improvement',
                'priority': 'medium',
                'issue': f"음성 명료도가 {clarity_analysis['overall_clarity_score']:.1f}점으로 낮습니다",
                'suggestion': "화자에게 더 명확한 발음을 요청하거나 녹음 환경을 개선해주세요",
                'expected_improvement': "텍스트 인식률 5-10% 향상"
            })
        
        # 환경 관련 제안
        env_analysis = analysis['environment_classification']
        if env_analysis['predicted_environment'] in ['exhibition_hall', 'outdoor']:
            suggestions.append({
                'category': 'environment_optimization',
                'priority': 'medium',
                'issue': f"{env_analysis['predicted_environment']} 환경은 녹음에 적합하지 않습니다",
                'suggestion': "가능하다면 더 조용한 환경으로 이동하거나 지향성 마이크를 사용해주세요",
                'expected_improvement': "전체 품질 15-25% 향상"
            })
        
        return suggestions
    
    def _check_quality_warnings(self, analysis: Dict) -> List[Dict]:
        """실시간 품질 경고 체크"""
        warnings = []
        
        # 중요 품질 이슈 체크
        overall_score = analysis['overall_quality']['overall_score']
        
        if overall_score < 50:
            warnings.append({
                'level': 'critical',
                'message': "음성 품질이 매우 낮습니다. 분석 결과의 정확도가 크게 떨어질 수 있습니다.",
                'action': "녹음을 중단하고 환경을 개선해주세요"
            })
        elif overall_score < 70:
            warnings.append({
                'level': 'warning', 
                'message': "음성 품질이 권장 수준보다 낮습니다.",
                'action': "가능하다면 녹음 조건을 개선해주세요"
            })
        
        # STT 정확도 경고
        stt_accuracy = analysis['stt_prediction']['predicted_accuracy']
        if stt_accuracy < 80:
            warnings.append({
                'level': 'warning',
                'message': f"예상 STT 정확도: {stt_accuracy:.1f}%",
                'action': "음성 품질 개선을 권장합니다"
            })
        
        return warnings
    
    # 유틸리티 함수들
    def _normalize_snr_to_score(self, snr: float) -> float:
        """SNR을 0-100 점수로 정규화"""
        # SNR 0dB = 점수 0, SNR 30dB = 점수 100
        score = (snr + 10) * 100 / 40  # -10dB~30dB를 0~100으로 매핑
        return np.clip(score, 0, 100)
    
    def _grade_noise_level(self, snr: float, noise_floor: float) -> str:
        """노이즈 레벨 등급 결정"""
        if snr > 25 and noise_floor < -40:
            return 'excellent'
        elif snr > 15 and noise_floor < -30:
            return 'good'
        elif snr > 10 and noise_floor < -20:
            return 'fair'
        else:
            return 'poor'
    
    def _grade_clarity(self, clarity_score: float) -> str:
        """명료도 등급 결정"""
        if clarity_score >= 85:
            return 'excellent'
        elif clarity_score >= 70:
            return 'good'
        elif clarity_score >= 55:
            return 'fair'
        else:
            return 'poor'
    
    def _grade_stt_prediction(self, accuracy: float) -> str:
        """STT 예측 정확도 등급"""
        if accuracy >= 90:
            return 'excellent'
        elif accuracy >= 80:
            return 'good'
        elif accuracy >= 70:
            return 'fair'
        else:
            return 'poor'
    
    def _grade_overall_quality(self, score: float) -> str:
        """종합 품질 등급"""
        if score >= 85:
            return 'excellent'
        elif score >= 70:
            return 'good'
        elif score >= 55:
            return 'fair'
        else:
            return 'poor'
    
    def _get_quality_level(self, score: float) -> str:
        """품질 레벨 (사용자 친화적)"""
        if score >= 85:
            return '최고 품질 - 완벽한 분석 가능'
        elif score >= 70:
            return '양호한 품질 - 높은 정확도 기대'
        elif score >= 55:
            return '보통 품질 - 기본 분석 가능'
        else:
            return '개선 필요 - 품질 향상 권장'
    
    def _analyze_crowd_noise(self, y: np.ndarray, sr: int) -> float:
        """군중 소음 레벨 분석 (전시장용)"""
        # 스펙트럴 중심의 변동성으로 군중 소음 감지
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        crowd_indicator = np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-10)
        return float(crowd_indicator * 100)
    
    def _estimate_reverberation(self, y: np.ndarray, sr: int) -> float:
        """잔향 시간 추정"""
        # 에너지 감쇠 기반 잔향 추정 (단순화된 방법)
        rms_values = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        
        # 피크 이후 감쇠 시간 추정
        peak_idx = np.argmax(rms_values)
        if peak_idx < len(rms_values) - 10:
            decay_values = rms_values[peak_idx:peak_idx+10]
            decay_rate = np.polyfit(range(len(decay_values)), np.log(decay_values + 1e-10), 1)[0]
            rt60_estimate = -13.8 / decay_rate if decay_rate < 0 else 0
            return float(np.clip(rt60_estimate, 0, 5))  # 0-5초 범위
        return 0.0
    
    def _analyze_room_acoustics(self, y: np.ndarray, sr: int) -> Dict:
        """회의실 음향 특성 분석"""
        return {
            'reverberation_time': self._estimate_reverberation(y, sr),
            'echo_presence': self._detect_echo(y, sr),
            'room_size_estimate': 'medium'  # 향후 고도화
        }
    
    def _detect_echo(self, y: np.ndarray, sr: int) -> bool:
        """에코 감지"""
        # 자기 상관 함수로 에코 패턴 감지
        correlation = np.correlate(y, y, mode='full')
        correlation = correlation[len(correlation)//2:]
        
        # 지연된 피크 찾기 (50ms 이후)
        min_delay_samples = int(0.05 * sr)  # 50ms
        if len(correlation) > min_delay_samples:
            delayed_peaks = correlation[min_delay_samples:]
            echo_strength = np.max(delayed_peaks) / (np.max(correlation) + 1e-10)
            return echo_strength > 0.3
        
        return False
    
    def _get_environment_recommendations(self, environment: str) -> List[str]:
        """환경별 개선 권장사항"""
        recommendations = {
            'office': [
                "에어컨이나 컴퓨터 팬 소음을 최소화하세요",
                "문을 닫아 복도 소음을 차단하세요"
            ],
            'exhibition_hall': [
                "지향성 마이크를 사용하세요",
                "가능하다면 조용한 구역으로 이동하세요",
                "핸드헬드 마이크를 화자 입에 가깝게 배치하세요"
            ],
            'conference_room': [
                "테이블 중앙에 마이크를 배치하세요",
                "에어컨 바람 소리를 확인하세요"
            ],
            'auditorium': [
                "강단 마이크를 사용하세요",
                "좌석 앞쪽에 녹음 장비를 배치하세요"
            ],
            'outdoor': [
                "실내로 이동할 것을 강력히 권장합니다",
                "바람막이나 윈드스크린을 사용하세요"
            ]
        }
        
        return recommendations.get(environment, ["일반적인 녹음 환경 개선을 권장합니다"])
    
    def _create_error_result(self, error_message: str) -> Dict:
        """오류 발생 시 기본 결과 반환"""
        return {
            'error': True,
            'error_message': error_message,
            'overall_quality': {
                'overall_score': 0,
                'quality_grade': 'error',
                'quality_level': '분석 실패'
            },
            'improvement_suggestions': [{
                'category': 'error',
                'priority': 'critical',
                'issue': '파일 분석 실패',
                'suggestion': '파일 형식 및 코덱을 확인해주세요',
                'expected_improvement': '분석 재시도 필요'
            }]
        }
    
    # 실시간 분석을 위한 스트리밍 함수들
    def analyze_audio_stream(self, audio_chunk: np.ndarray, sr: int) -> Dict:
        """
        실시간 오디오 스트림 분석
        
        Args:
            audio_chunk: 실시간 오디오 청크
            sr: 샘플링 레이트
            
        Returns:
            Dict: 실시간 품질 분석 결과
        """
        if len(audio_chunk) < self.chunk_size:
            return {'status': 'insufficient_data'}
        
        # 빠른 품질 체크 (주요 지표만)
        quick_analysis = {
            'timestamp': len(audio_chunk) / sr,
            'rms_level': float(np.sqrt(np.mean(audio_chunk**2))),
            'peak_level': float(np.max(np.abs(audio_chunk))),
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio_chunk)))
        }
        
        # 실시간 노이즈 체크
        rms_db = 20 * np.log10(quick_analysis['rms_level'] + 1e-10)
        quick_analysis['rms_db'] = rms_db
        
        # 실시간 품질 경고
        if rms_db < -50:
            quick_analysis['warning'] = 'too_quiet'
        elif rms_db > -5:
            quick_analysis['warning'] = 'too_loud'
        else:
            quick_analysis['warning'] = None
        
        return quick_analysis
    
    def get_quality_recommendations(self) -> Dict:
        """품질 개선을 위한 종합 권장사항"""
        return {
            'microphone_setup': [
                "마이크를 화자로부터 15-30cm 거리에 배치",
                "지향성 마이크 사용 권장 (카디오이드 패턴)",
                "마이크 입력 레벨을 -12dB~-6dB로 설정"
            ],
            'environment_optimization': [
                "배경소음이 적은 환경 선택",
                "딱딱한 표면의 반사음 최소화",
                "에어컨, 프로젝터 등 기계 소음 확인"
            ],
            'recording_practices': [
                "녹음 전 테스트 녹음으로 품질 확인",
                "화자에게 명확한 발음 요청",
                "중요한 내용은 반복 설명 요청"
            ],
            'post_processing': [
                "노이즈 제거 필터 적용 고려",
                "볼륨 정규화로 일정한 레벨 유지",
                "품질이 낮은 구간은 재녹음 고려"
            ]
        }


# 사용 예시 및 테스트 함수
def test_audio_quality_analyzer():
    """AudioQualityAnalyzer 테스트 함수"""
    analyzer = AudioQualityAnalyzer()
    
    # 테스트용 합성 오디오 생성
    sample_rate = 16000
    duration = 5  # 5초
    t = np.linspace(0, duration, sample_rate * duration)
    
    # 음성 시뮬레이션 (복합 사인파 + 노이즈)
    voice_signal = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz 톤
    noise_signal = np.random.normal(0, 0.1, len(t))   # 배경 노이즈
    test_audio = voice_signal + noise_signal
    
    # 테스트 분석 실행
    try:
        # 임시 파일로 저장
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, test_audio, sample_rate)
            
            # 분석 실행
            result = analyzer.analyze_audio_file(tmp_file.name)
            
            print("🎤 음성 품질 분석 테스트 결과:")
            print(f"전체 품질 점수: {result['overall_quality']['overall_score']:.1f}/100")
            print(f"품질 등급: {result['overall_quality']['quality_grade']}")
            print(f"예상 STT 정확도: {result['stt_prediction']['predicted_accuracy']:.1f}%")
            print(f"감지된 환경: {result['environment_classification']['predicted_environment']}")
            
            return result
            
    except Exception as e:
        print(f"테스트 실패: {str(e)}")
        return None


if __name__ == "__main__":
    # 테스트 실행
    test_result = test_audio_quality_analyzer()
