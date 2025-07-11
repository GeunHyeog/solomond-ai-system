"""
🎙️ Audio Quality Checker v2.1
음성 품질 실시간 분석 및 현장 최적화 모듈

주요 기능:
- SNR (Signal-to-Noise Ratio) 실시간 측정 
- 배경 노이즈 레벨 분석
- 음성 명료도 점수 계산
- 현장 녹음 품질 즉시 검증
- 재녹음 권장 알고리즘
"""

import numpy as np
import librosa
import scipy.signal
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings("ignore")

class AudioQualityChecker:
    """음성 품질 실시간 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 품질 기준값 (dB)
        self.quality_thresholds = {
            'snr_excellent': 25.0,    # SNR 25dB 이상 = 우수
            'snr_good': 20.0,         # SNR 20-25dB = 양호  
            'snr_fair': 15.0,         # SNR 15-20dB = 보통
            'snr_poor': 10.0,         # SNR 10dB 미만 = 불량
            
            'noise_low': -40.0,       # 노이즈 -40dB 이하 = 낮음
            'noise_medium': -30.0,    # 노이즈 -30~-40dB = 보통
            'noise_high': -20.0,      # 노이즈 -20dB 이상 = 높음
            
            'clarity_excellent': 0.9, # 명료도 90% 이상 = 우수
            'clarity_good': 0.8,      # 명료도 80-90% = 양호
            'clarity_fair': 0.7,      # 명료도 70-80% = 보통
        }
        
        # 주얼리 업계 특화 키워드 (명료도 측정용)
        self.jewelry_keywords = [
            'diamond', 'gold', 'silver', 'platinum', 'gemstone',
            'carat', 'cut', 'clarity', 'color', 'certificate',
            '다이아몬드', '금', '은', '백금', '보석',
            '캐럿', '컷', '투명도', '색상', '감정서'
        ]

    def analyze_audio_quality(self, 
                            audio_path: str = None, 
                            audio_data: np.ndarray = None, 
                            sr: int = 22050) -> Dict:
        """
        음성 품질 종합 분석
        
        Args:
            audio_path: 오디오 파일 경로
            audio_data: 오디오 데이터 (numpy array)
            sr: 샘플링 레이트
            
        Returns:
            Dict: 품질 분석 결과
        """
        try:
            # 오디오 로드
            if audio_data is None:
                if audio_path is None:
                    raise ValueError("audio_path 또는 audio_data 중 하나는 필수입니다")
                audio_data, sr = librosa.load(audio_path, sr=sr)
            
            # 기본 분석
            results = {
                'timestamp': self._get_timestamp(),
                'duration': len(audio_data) / sr,
                'sample_rate': sr,
                'file_path': audio_path or 'real_time_data'
            }
            
            # SNR 분석
            snr_result = self.calculate_snr(audio_data, sr)
            results.update(snr_result)
            
            # 노이즈 분석  
            noise_result = self.analyze_background_noise(audio_data, sr)
            results.update(noise_result)
            
            # 명료도 분석
            clarity_result = self.calculate_speech_clarity(audio_data, sr)
            results.update(clarity_result)
            
            # 전체 품질 점수 계산
            overall_score = self.calculate_overall_quality_score(results)
            results['overall_quality'] = overall_score
            
            # 권장사항 생성
            recommendations = self.generate_recommendations(results)
            results['recommendations'] = recommendations
            
            return results
            
        except Exception as e:
            self.logger.error(f"음성 품질 분석 오류: {str(e)}")
            return {
                'error': str(e),
                'overall_quality': {'score': 0, 'level': 'error'}
            }

    def calculate_snr(self, audio_data: np.ndarray, sr: int) -> Dict:
        """SNR (Signal-to-Noise Ratio) 계산"""
        try:
            # 음성 구간과 무음 구간 분리
            voice_segments, silence_segments = self._detect_voice_silence(audio_data, sr)
            
            if len(voice_segments) == 0 or len(silence_segments) == 0:
                return {
                    'snr_db': 0.0,
                    'snr_level': 'unknown',
                    'signal_power': 0.0,
                    'noise_power': 0.0
                }
            
            # 신호 파워 계산 (음성 구간)
            signal_power = np.mean([np.mean(segment**2) for segment in voice_segments])
            
            # 노이즈 파워 계산 (무음 구간)
            noise_power = np.mean([np.mean(segment**2) for segment in silence_segments])
            
            # SNR 계산 (dB)
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
            else:
                snr_db = 60.0  # 노이즈가 거의 없는 경우
            
            # SNR 등급 분류
            snr_level = self._classify_snr_level(snr_db)
            
            return {
                'snr_db': round(snr_db, 1),
                'snr_level': snr_level,
                'signal_power': float(signal_power),
                'noise_power': float(noise_power)
            }
            
        except Exception as e:
            self.logger.error(f"SNR 계산 오류: {str(e)}")
            return {
                'snr_db': 0.0,
                'snr_level': 'error',
                'signal_power': 0.0,
                'noise_power': 0.0
            }

    def analyze_background_noise(self, audio_data: np.ndarray, sr: int) -> Dict:
        """배경 노이즈 분석"""
        try:
            # 스펙트럼 분석
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            
            # 노이즈 프로파일 추출 (저주파 + 고주파 노이즈)
            low_freq_noise = np.mean(magnitude[:10, :])    # 저주파 (0-1kHz)
            mid_freq_noise = np.mean(magnitude[10:50, :])  # 중주파 (1-5kHz)
            high_freq_noise = np.mean(magnitude[50:, :])   # 고주파 (5kHz+)
            
            # 전체 노이즈 레벨
            total_noise_db = 20 * np.log10(np.mean(magnitude) + 1e-10)
            
            # 노이즈 유형 분석
            noise_types = self._analyze_noise_types(audio_data, sr)
            
            # 노이즈 등급 분류
            noise_level = self._classify_noise_level(total_noise_db)
            
            return {
                'noise_db': round(total_noise_db, 1),
                'noise_level': noise_level,
                'low_freq_noise': float(low_freq_noise),
                'mid_freq_noise': float(mid_freq_noise),
                'high_freq_noise': float(high_freq_noise),
                'noise_types': noise_types
            }
            
        except Exception as e:
            self.logger.error(f"노이즈 분석 오류: {str(e)}")
            return {
                'noise_db': 0.0,
                'noise_level': 'error',
                'noise_types': []
            }

    def calculate_speech_clarity(self, audio_data: np.ndarray, sr: int) -> Dict:
        """음성 명료도 계산"""
        try:
            # 음성 특징 추출
            features = self._extract_speech_features(audio_data, sr)
            
            # 주파수 분포 분석 (음성 대역 집중도)
            speech_band_energy = self._calculate_speech_band_energy(audio_data, sr)
            
            # 음성 일관성 분석
            consistency_score = self._calculate_speech_consistency(audio_data, sr)
            
            # 전체 명료도 점수 계산
            clarity_score = (
                features['spectral_centroid_score'] * 0.3 +
                features['zero_crossing_score'] * 0.2 +
                speech_band_energy * 0.3 +
                consistency_score * 0.2
            )
            
            # 명료도 등급 분류
            clarity_level = self._classify_clarity_level(clarity_score)
            
            return {
                'clarity_score': round(clarity_score, 3),
                'clarity_level': clarity_level,
                'clarity_percentage': round(clarity_score * 100, 1),
                'speech_features': features,
                'speech_band_energy': round(speech_band_energy, 3),
                'consistency_score': round(consistency_score, 3)
            }
            
        except Exception as e:
            self.logger.error(f"명료도 계산 오류: {str(e)}")
            return {
                'clarity_score': 0.0,
                'clarity_level': 'error',
                'clarity_percentage': 0.0
            }

    def calculate_overall_quality_score(self, results: Dict) -> Dict:
        """전체 품질 점수 계산"""
        try:
            # 가중치 설정
            weights = {
                'snr': 0.4,        # SNR 40%
                'noise': 0.3,      # 노이즈 30%
                'clarity': 0.3     # 명료도 30%
            }
            
            # 개별 점수 정규화 (0-1)
            snr_normalized = self._normalize_snr_score(results.get('snr_db', 0))
            noise_normalized = self._normalize_noise_score(results.get('noise_db', 0))
            clarity_normalized = results.get('clarity_score', 0)
            
            # 가중 평균 계산
            overall_score = (
                snr_normalized * weights['snr'] +
                noise_normalized * weights['noise'] +
                clarity_normalized * weights['clarity']
            )
            
            # 등급 분류
            if overall_score >= 0.9:
                level = 'excellent'
                status = '우수'
                color = '🟢'
            elif overall_score >= 0.8:
                level = 'good'  
                status = '양호'
                color = '🟡'
            elif overall_score >= 0.7:
                level = 'fair'
                status = '보통'
                color = '🟠'
            else:
                level = 'poor'
                status = '불량'
                color = '🔴'
            
            return {
                'score': round(overall_score, 3),
                'percentage': round(overall_score * 100, 1),
                'level': level,
                'status': status,
                'color': color,
                'components': {
                    'snr_score': round(snr_normalized, 3),
                    'noise_score': round(noise_normalized, 3),
                    'clarity_score': round(clarity_normalized, 3)
                }
            }
            
        except Exception as e:
            self.logger.error(f"전체 품질 점수 계산 오류: {str(e)}")
            return {
                'score': 0.0,
                'level': 'error',
                'status': '오류'
            }

    def generate_recommendations(self, results: Dict) -> List[Dict]:
        """품질 개선 권장사항 생성"""
        recommendations = []
        
        try:
            overall_quality = results.get('overall_quality', {})
            snr_db = results.get('snr_db', 0)
            noise_db = results.get('noise_db', 0)
            clarity_score = results.get('clarity_score', 0)
            
            # SNR 기반 권장사항
            if snr_db < self.quality_thresholds['snr_poor']:
                recommendations.append({
                    'type': 'critical',
                    'icon': '🔴',
                    'title': 'SNR 매우 낮음',
                    'message': '마이크를 입에 더 가까이 하거나 조용한 곳으로 이동하세요',
                    'action': 'move_closer_or_quiet_place'
                })
            elif snr_db < self.quality_thresholds['snr_fair']:
                recommendations.append({
                    'type': 'warning',
                    'icon': '🟡',
                    'title': 'SNR 개선 필요',
                    'message': '말하는 소리를 조금 더 크게 하거나 배경음을 줄여보세요',
                    'action': 'speak_louder_or_reduce_background'
                })
            
            # 노이즈 기반 권장사항  
            if noise_db > self.quality_thresholds['noise_high']:
                recommendations.append({
                    'type': 'critical',
                    'icon': '🔴',
                    'title': '배경 노이즈 높음',
                    'message': '에어컨, 팬 등을 끄거나 더 조용한 장소로 이동하세요',
                    'action': 'reduce_background_noise'
                })
            elif noise_db > self.quality_thresholds['noise_medium']:
                recommendations.append({
                    'type': 'warning',
                    'icon': '🟠',
                    'title': '배경 노이즈 보통',
                    'message': '가능하면 조용한 환경에서 녹음해보세요',
                    'action': 'find_quieter_environment'
                })
            
            # 명료도 기반 권장사항
            if clarity_score < self.quality_thresholds['clarity_fair']:
                recommendations.append({
                    'type': 'warning',
                    'icon': '🟠',
                    'title': '음성 명료도 낮음',
                    'message': '더 또렷하게 발음하고 적당한 속도로 말씀해주세요',
                    'action': 'speak_more_clearly'
                })
            
            # 전체 품질 기반 권장사항
            if overall_quality.get('level') == 'poor':
                recommendations.append({
                    'type': 'critical',
                    'icon': '🔴',
                    'title': '재녹음 권장',
                    'message': '현재 음질이 좋지 않습니다. 환경을 개선한 후 다시 녹음해보세요',
                    'action': 'retry_recording'
                })
            elif overall_quality.get('level') == 'excellent':
                recommendations.append({
                    'type': 'success',
                    'icon': '🟢',
                    'title': '최적 품질',
                    'message': '현재 설정을 유지하여 계속 녹음하세요',
                    'action': 'maintain_current_settings'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"권장사항 생성 오류: {str(e)}")
            return [{
                'type': 'error',
                'icon': '❌',
                'title': '분석 오류',
                'message': '품질 분석 중 오류가 발생했습니다',
                'action': 'retry_analysis'
            }]

    # === 내부 유틸리티 함수들 ===
    
    def _detect_voice_silence(self, audio_data: np.ndarray, sr: int) -> Tuple[List, List]:
        """음성/무음 구간 감지"""
        # 에너지 기반 VAD (Voice Activity Detection)
        frame_length = int(0.025 * sr)  # 25ms 프레임
        hop_length = int(0.010 * sr)    # 10ms 홉
        
        # 단구간 에너지 계산
        energy = []
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i + frame_length]
            energy.append(np.sum(frame**2))
        
        energy = np.array(energy)
        
        # 임계값 설정 (에너지의 중간값 기준)
        threshold = np.median(energy) * 2
        
        # 음성/무음 구간 분류
        voice_segments = []
        silence_segments = []
        
        for i, e in enumerate(energy):
            start_idx = i * hop_length
            end_idx = start_idx + frame_length
            
            if end_idx <= len(audio_data):
                segment = audio_data[start_idx:end_idx]
                
                if e > threshold:
                    voice_segments.append(segment)
                else:
                    silence_segments.append(segment)
        
        return voice_segments, silence_segments
    
    def _analyze_noise_types(self, audio_data: np.ndarray, sr: int) -> List[str]:
        """노이즈 유형 분석"""
        noise_types = []
        
        # 스펙트럼 분석
        stft = librosa.stft(audio_data)
        magnitude = np.abs(stft)
        
        # 50Hz/60Hz 험 감지
        freqs = librosa.fft_frequencies(sr=sr)
        hum_indices = [
            np.argmin(np.abs(freqs - 50)),   # 50Hz
            np.argmin(np.abs(freqs - 60)),   # 60Hz
            np.argmin(np.abs(freqs - 100)),  # 100Hz
            np.argmin(np.abs(freqs - 120))   # 120Hz
        ]
        
        hum_energy = np.mean([np.mean(magnitude[idx, :]) for idx in hum_indices])
        total_energy = np.mean(magnitude)
        
        if hum_energy / total_energy > 0.1:
            noise_types.append('electrical_hum')
        
        # 화이트 노이즈 감지
        high_freq_energy = np.mean(magnitude[100:, :])
        if high_freq_energy / total_energy > 0.3:
            noise_types.append('white_noise')
        
        # 환경음 감지 (저주파 에너지)
        low_freq_energy = np.mean(magnitude[:20, :])
        if low_freq_energy / total_energy > 0.4:
            noise_types.append('environmental')
        
        return noise_types
    
    def _extract_speech_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """음성 특징 추출"""
        # 스펙트럴 중심 (Spectral Centroid)
        spec_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        centroid_mean = np.mean(spec_centroid)
        centroid_score = min(1.0, centroid_mean / 3000)  # 3kHz 기준 정규화
        
        # 영교차율 (Zero Crossing Rate)
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        zcr_mean = np.mean(zcr)
        zcr_score = min(1.0, zcr_mean * 10)  # 정규화
        
        return {
            'spectral_centroid': round(centroid_mean, 1),
            'spectral_centroid_score': round(centroid_score, 3),
            'zero_crossing_rate': round(zcr_mean, 4),
            'zero_crossing_score': round(zcr_score, 3)
        }
    
    def _calculate_speech_band_energy(self, audio_data: np.ndarray, sr: int) -> float:
        """음성 대역 에너지 비율 계산"""
        stft = librosa.stft(audio_data)
        magnitude = np.abs(stft)
        
        freqs = librosa.fft_frequencies(sr=sr)
        
        # 음성 주요 대역 (300Hz - 3400Hz)
        speech_band_indices = np.where((freqs >= 300) & (freqs <= 3400))[0]
        speech_energy = np.mean(magnitude[speech_band_indices, :])
        
        # 전체 에너지
        total_energy = np.mean(magnitude)
        
        if total_energy > 0:
            return speech_energy / total_energy
        else:
            return 0.0
    
    def _calculate_speech_consistency(self, audio_data: np.ndarray, sr: int) -> float:
        """음성 일관성 점수 계산"""
        # 오디오를 여러 구간으로 나누어 일관성 측정
        segment_length = int(1.0 * sr)  # 1초 구간
        num_segments = len(audio_data) // segment_length
        
        if num_segments < 2:
            return 1.0
        
        segment_energies = []
        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            segment = audio_data[start:end]
            energy = np.mean(segment**2)
            segment_energies.append(energy)
        
        # 에너지 변동 계수 계산
        if len(segment_energies) > 1:
            std_dev = np.std(segment_energies)
            mean_energy = np.mean(segment_energies)
            
            if mean_energy > 0:
                cv = std_dev / mean_energy  # 변동계수
                consistency = max(0.0, 1.0 - cv)  # 변동이 적을수록 일관성 높음
            else:
                consistency = 0.0
        else:
            consistency = 1.0
        
        return consistency
    
    def _classify_snr_level(self, snr_db: float) -> str:
        """SNR 등급 분류"""
        if snr_db >= self.quality_thresholds['snr_excellent']:
            return 'excellent'
        elif snr_db >= self.quality_thresholds['snr_good']:
            return 'good'
        elif snr_db >= self.quality_thresholds['snr_fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _classify_noise_level(self, noise_db: float) -> str:
        """노이즈 등급 분류"""
        if noise_db <= self.quality_thresholds['noise_low']:
            return 'low'
        elif noise_db <= self.quality_thresholds['noise_medium']:
            return 'medium'
        else:
            return 'high'
    
    def _classify_clarity_level(self, clarity_score: float) -> str:
        """명료도 등급 분류"""
        if clarity_score >= self.quality_thresholds['clarity_excellent']:
            return 'excellent'
        elif clarity_score >= self.quality_thresholds['clarity_good']:
            return 'good'
        elif clarity_score >= self.quality_thresholds['clarity_fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _normalize_snr_score(self, snr_db: float) -> float:
        """SNR 점수 정규화 (0-1)"""
        # 0dB = 0점, 30dB = 1점으로 정규화
        return max(0.0, min(1.0, snr_db / 30.0))
    
    def _normalize_noise_score(self, noise_db: float) -> float:
        """노이즈 점수 정규화 (0-1, 노이즈가 낮을수록 높은 점수)"""
        # -50dB = 1점, 0dB = 0점으로 정규화
        return max(0.0, min(1.0, (-noise_db) / 50.0))
    
    def _get_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 사용 예제
if __name__ == "__main__":
    checker = AudioQualityChecker()
    
    # 테스트용 예시
    print("🎙️ Audio Quality Checker v2.1 - 테스트 시작")
    print("=" * 50)
    
    # 실제 사용 시에는 오디오 파일 경로를 제공
    # result = checker.analyze_audio_quality("test_audio.wav")
    # print(f"전체 품질 점수: {result['overall_quality']['percentage']}%")
    # print(f"품질 등급: {result['overall_quality']['status']}")
    
    print("모듈 로드 완료 ✅")
