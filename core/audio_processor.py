#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고급 오디오 전처리 모듈 - 솔로몬드 AI 시스템 확장

주얼리 AI STT를 위한 오디오 전처리 및 최적화 모듈
- 노이즈 제거 및 품질 향상
- 주얼리 회의/세미나 환경 최적화
- 모바일 녹음 품질 개선
- Whisper STT 성능 극대화

Author: 전근혁 (솔로몬드 대표)
Date: 2025.07.08
"""

import logging
import os
import tempfile
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import scipy.signal as signal
from scipy.io import wavfile
import librosa

logger = logging.getLogger(__name__)

class JewelryAudioProcessor:
    """주얼리 업계 특화 오디오 전처리 및 최적화 클래스"""
    
    def __init__(self):
        """오디오 처리기 초기화"""
        self.target_sr = 16000  # Whisper 최적 샘플링 레이트
        self.target_channels = 1  # 모노
        self.target_bit_depth = 16  # 16-bit
        
        # 주얼리 업계 특화 설정
        self.speech_freq_range = (300, 3400)  # 음성 주파수 대역
        self.noise_gate_threshold = -50  # dB
        self.jewelry_room_acoustics = {
            'conference_room': {'reverb_reduction': True, 'echo_suppression': True},
            'showroom': {'background_noise': 'high', 'glass_reflection': True},
            'workshop': {'machinery_noise': True, 'metal_clanking': True},
            'mobile': {'wind_noise': True, 'handling_noise': True}
        }
        
        logger.info("주얼리 업계 특화 오디오 처리기 초기화 완료")
    
    def analyze_jewelry_audio_environment(self, audio_path: str) -> Dict[str, Any]:
        """주얼리 업계 환경 특화 오디오 분석"""
        try:
            # 기본 오디오 분석
            audio = AudioSegment.from_file(audio_path)
            y, sr = librosa.load(audio_path, sr=None)
            
            # 기본 정보
            duration = len(audio) / 1000.0
            sample_rate = audio.frame_rate
            channels = audio.channels
            
            # 주얼리 업계 특화 분석
            analysis = {
                'basic_info': {
                    'duration': round(duration, 2),
                    'sample_rate': sample_rate,
                    'channels': channels,
                    'file_size': os.path.getsize(audio_path)
                },
                'audio_quality': {},
                'jewelry_environment': {},
                'recommendations': []
            }
            
            # 음성 에너지 분석
            rms_energy = np.sqrt(np.mean(y**2))
            max_amplitude = np.max(np.abs(y))
            
            # 스펙트럼 분석
            frequencies, times, spectrogram = signal.spectrogram(y, sr)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
            # 주파수 대역별 에너지 분석
            speech_band_energy = self._analyze_frequency_band(y, sr, 300, 3400)
            jewelry_clanking_energy = self._analyze_frequency_band(y, sr, 2000, 8000)  # 금속 소리
            glass_reflection_energy = self._analyze_frequency_band(y, sr, 1000, 4000)  # 유리 반사
            
            analysis['audio_quality'] = {
                'rms_energy': round(rms_energy, 4),
                'max_amplitude': round(max_amplitude, 4),
                'spectral_centroid': round(spectral_centroid, 2),
                'speech_clarity': round(speech_band_energy, 4)
            }
            
            # 주얼리 환경 특성 분석
            environment_type = self._identify_jewelry_environment(
                speech_band_energy, jewelry_clanking_energy, glass_reflection_energy
            )
            
            analysis['jewelry_environment'] = {
                'environment_type': environment_type,
                'speech_band_energy': round(speech_band_energy, 4),
                'jewelry_clanking_detected': jewelry_clanking_energy > 0.01,
                'glass_reflection_detected': glass_reflection_energy > 0.02,
                'reverb_level': self._estimate_reverb_level(y, sr)
            }
            
            # 주얼리 특화 권장사항
            analysis['recommendations'] = self._generate_jewelry_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"주얼리 오디오 환경 분석 오류: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_frequency_band(self, y: np.ndarray, sr: int, low_freq: int, high_freq: int) -> float:
        """특정 주파수 대역의 에너지 분석"""
        try:
            # FFT 수행
            fft = np.fft.fft(y)
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            
            # 해당 주파수 대역 인덱스 찾기
            low_idx = np.argmin(np.abs(freqs - low_freq))
            high_idx = np.argmin(np.abs(freqs - high_freq))
            
            # 해당 대역의 에너지 계산
            band_energy = np.mean(np.abs(fft[low_idx:high_idx])**2)
            total_energy = np.mean(np.abs(fft)**2)
            
            return band_energy / total_energy if total_energy > 0 else 0
            
        except Exception as e:
            logger.warning(f"주파수 대역 분석 오류: {str(e)}")
            return 0
    
    def _identify_jewelry_environment(self, speech_energy: float, metal_energy: float, glass_energy: float) -> str:
        """주얼리 업계 환경 유형 식별"""
        if metal_energy > 0.02:
            return 'workshop'  # 작업장 (금속 가공 소리)
        elif glass_energy > 0.03:
            return 'showroom'  # 쇼룸 (유리 진열장 반사)
        elif speech_energy > 0.1:
            return 'conference_room'  # 회의실 (깨끗한 음성)
        else:
            return 'mobile'  # 모바일 녹음
    
    def _estimate_reverb_level(self, y: np.ndarray, sr: int) -> str:
        """리버브 레벨 추정"""
        try:
            # 간단한 리버브 감지 (에코 지연 시간 분석)
            autocorr = np.correlate(y, y, mode='full')
            center = len(autocorr) // 2
            
            # 100ms 이후의 자기상관 피크 찾기
            delay_samples = int(0.1 * sr)  # 100ms
            late_autocorr = autocorr[center + delay_samples:]
            
            if len(late_autocorr) > 0:
                max_late_corr = np.max(np.abs(late_autocorr)) / np.max(np.abs(autocorr))
                
                if max_late_corr > 0.3:
                    return 'high'
                elif max_late_corr > 0.1:
                    return 'medium'
                else:
                    return 'low'
            
            return 'low'
            
        except Exception as e:
            logger.warning(f"리버브 추정 오류: {str(e)}")
            return 'unknown'
    
    def _generate_jewelry_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """주얼리 환경 특화 권장사항 생성"""
        recommendations = []
        
        basic_info = analysis.get('basic_info', {})
        audio_quality = analysis.get('audio_quality', {})
        jewelry_env = analysis.get('jewelry_environment', {})
        
        # 기본 품질 권장사항
        if basic_info.get('sample_rate', 0) != self.target_sr:
            recommendations.append(f"샘플링 레이트를 {self.target_sr}Hz로 최적화 권장")
        
        if basic_info.get('channels', 1) > 1:
            recommendations.append("모노 채널로 변환하여 처리 효율성 향상")
        
        # 주얼리 환경별 권장사항
        env_type = jewelry_env.get('environment_type', 'unknown')
        
        if env_type == 'workshop':
            recommendations.append("작업장 환경: 금속 가공 노이즈 제거 필터 적용")
            recommendations.append("고주파 노이즈 억제 및 음성 대역 강화")
        
        elif env_type == 'showroom':
            recommendations.append("쇼룸 환경: 유리 반사음 억제 및 에코 제거")
            recommendations.append("중간 주파수 대역 노이즈 필터링")
        
        elif env_type == 'conference_room':
            recommendations.append("회의실 환경: 리버브 감소 및 음성 명료도 향상")
        
        elif env_type == 'mobile':
            recommendations.append("모바일 녹음: 핸들링 노이즈 및 바람 소리 제거")
            recommendations.append("자동 게인 조절 및 압축 적용")
        
        # 음질별 권장사항
        speech_clarity = audio_quality.get('speech_clarity', 0)
        if speech_clarity < 0.05:
            recommendations.append("음성 명료도 낮음: 노이즈 제거 및 음성 증폭 필요")
        
        return recommendations
    
    def preprocess_jewelry_audio(self, audio_path: str, 
                                environment_type: str = 'auto',
                                enhancement_level: str = 'medium') -> str:
        """
        주얼리 업계 특화 오디오 전처리
        
        Args:
            audio_path: 입력 오디오 파일 경로
            environment_type: 환경 유형 ('auto', 'conference_room', 'showroom', 'workshop', 'mobile')
            enhancement_level: 향상 레벨 ('light', 'medium', 'aggressive')
            
        Returns:
            전처리된 오디오 파일 경로
        """
        try:
            logger.info(f"주얼리 특화 오디오 전처리 시작: {audio_path}")
            
            # 환경 자동 감지
            if environment_type == 'auto':
                analysis = self.analyze_jewelry_audio_environment(audio_path)
                environment_type = analysis.get('jewelry_environment', {}).get('environment_type', 'mobile')
            
            # 원본 오디오 로딩
            audio = AudioSegment.from_file(audio_path)
            
            # 1. 기본 포맷 최적화
            audio = self._optimize_format(audio)
            
            # 2. 환경별 특화 처리
            audio = self._apply_environment_specific_processing(audio, environment_type)
            
            # 3. 향상 레벨별 처리
            audio = self._apply_enhancement_level(audio, enhancement_level)
            
            # 4. 최종 품질 향상
            audio = self._final_quality_enhancement(audio)
            
            # 5. 임시 파일로 저장
            with tempfile.NamedTemporaryFile(
                suffix='.wav', 
                delete=False,
                dir=os.path.dirname(audio_path)
            ) as tmp_file:
                processed_path = tmp_file.name
            
            # WAV 포맷으로 내보내기
            audio.export(processed_path, format="wav")
            
            logger.info(f"주얼리 특화 전처리 완료: {processed_path}")
            return processed_path
            
        except Exception as e:
            logger.error(f"주얼리 오디오 전처리 오류: {str(e)}")
            return audio_path  # 실패 시 원본 경로 반환
    
    def _optimize_format(self, audio: AudioSegment) -> AudioSegment:
        """기본 포맷 최적화"""
        # 샘플링 레이트 변환
        if audio.frame_rate != self.target_sr:
            audio = audio.set_frame_rate(self.target_sr)
        
        # 모노 변환
        if audio.channels > 1:
            audio = audio.set_channels(self.target_channels)
        
        # 비트 깊이 설정
        if audio.sample_width != self.target_bit_depth // 8:
            audio = audio.set_sample_width(self.target_bit_depth // 8)
        
        return audio
    
    def _apply_environment_specific_processing(self, audio: AudioSegment, env_type: str) -> AudioSegment:
        """환경별 특화 처리"""
        if env_type == 'workshop':
            # 작업장: 금속 노이즈 제거
            audio = audio.high_pass_filter(400)  # 저주파 기계 소음 제거
            audio = audio.low_pass_filter(6000)  # 고주파 금속 소음 제거
            
        elif env_type == 'showroom':
            # 쇼룸: 유리 반사 및 에코 처리
            audio = audio.high_pass_filter(200)
            audio = audio.low_pass_filter(7000)
            # 간단한 컴프레서로 에코 억제
            audio = compress_dynamic_range(audio, threshold=-25.0, ratio=3.0)
            
        elif env_type == 'conference_room':
            # 회의실: 리버브 감소
            audio = audio.high_pass_filter(100)
            audio = compress_dynamic_range(audio, threshold=-20.0, ratio=2.5)
            
        elif env_type == 'mobile':
            # 모바일: 핸들링 노이즈 및 바람 소리
            audio = audio.high_pass_filter(150)  # 바람 소리 제거
            audio = audio.low_pass_filter(8000)
            
        return audio
    
    def _apply_enhancement_level(self, audio: AudioSegment, level: str) -> AudioSegment:
        """향상 레벨별 처리"""
        if level == 'light':
            # 가벼운 처리
            audio = normalize(audio, headroom=0.1)
            
        elif level == 'medium':
            # 중간 처리
            audio = normalize(audio)
            audio = compress_dynamic_range(audio, threshold=-20.0, ratio=3.0)
            
        elif level == 'aggressive':
            # 적극적 처리
            audio = normalize(audio)
            audio = compress_dynamic_range(audio, threshold=-15.0, ratio=4.0)
            # 노이즈 게이트 적용
            audio = self._apply_noise_gate(audio)
            
        return audio
    
    def _apply_noise_gate(self, audio: AudioSegment) -> AudioSegment:
        """노이즈 게이트 적용"""
        try:
            chunks = []
            chunk_length = 100  # ms
            
            for i in range(0, len(audio), chunk_length):
                chunk = audio[i:i + chunk_length]
                if chunk.dBFS > self.noise_gate_threshold:
                    chunks.append(chunk)
                else:
                    # 조용한 부분은 볼륨 감소
                    chunks.append(chunk - 15)  # 15dB 감소
            
            return sum(chunks) if chunks else audio
            
        except Exception as e:
            logger.warning(f"노이즈 게이트 적용 오류: {str(e)}")
            return audio
    
    def _final_quality_enhancement(self, audio: AudioSegment) -> AudioSegment:
        """최종 품질 향상"""
        try:
            # 음성 주파수 대역 강화 (300-3400Hz)
            # 간단한 EQ 효과
            audio = audio + 1  # 1dB 부스트
            
            # 최종 정규화
            audio = normalize(audio, headroom=0.5)
            
            return audio
            
        except Exception as e:
            logger.warning(f"최종 품질 향상 오류: {str(e)}")
            return audio
    
    def extract_audio_features(self, audio_path: str) -> Dict[str, float]:
        """주얼리 업계 특화 오디오 특성 추출"""
        try:
            y, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # 기본 특성
            features = {
                'duration': len(y) / sr,
                'rms_energy': float(np.sqrt(np.mean(y**2))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
            }
            
            # 스펙트럼 특성
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            
            features.update({
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            })
            
            # 주얼리 특화 특성
            speech_energy = self._analyze_frequency_band(y, sr, 300, 3400)
            jewelry_noise = self._analyze_frequency_band(y, sr, 2000, 8000)
            
            features.update({
                'speech_band_ratio': speech_energy,
                'jewelry_noise_ratio': jewelry_noise,
                'voice_activity_ratio': self._estimate_voice_activity(y, sr)
            })
            
            return features
            
        except Exception as e:
            logger.error(f"특성 추출 오류: {str(e)}")
            return {}
    
    def _estimate_voice_activity(self, y: np.ndarray, sr: int) -> float:
        """음성 활동 비율 추정"""
        try:
            # 간단한 에너지 기반 음성 활동 감지
            frame_length = int(0.025 * sr)  # 25ms
            hop_length = int(0.01 * sr)   # 10ms
            
            frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
            energy = np.sum(frames**2, axis=0)
            
            # 임계값 이상의 프레임 비율
            threshold = np.percentile(energy, 30)  # 30th percentile을 임계값으로
            voice_frames = np.sum(energy > threshold)
            total_frames = len(energy)
            
            return voice_frames / total_frames if total_frames > 0 else 0
            
        except Exception as e:
            logger.warning(f"음성 활동 추정 오류: {str(e)}")
            return 0
    
    def batch_process_jewelry_audio(self, audio_files: List[str], 
                                   output_dir: str = None) -> List[Dict[str, Any]]:
        """주얼리 오디오 파일 일괄 처리"""
        results = []
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for audio_file in audio_files:
            try:
                logger.info(f"일괄 처리 중: {audio_file}")
                
                # 분석
                analysis = self.analyze_jewelry_audio_environment(audio_file)
                
                # 전처리
                processed_file = self.preprocess_jewelry_audio(audio_file)
                
                # 결과 저장
                result = {
                    'original_file': audio_file,
                    'processed_file': processed_file,
                    'analysis': analysis,
                    'status': 'success'
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"일괄 처리 오류 ({audio_file}): {str(e)}")
                results.append({
                    'original_file': audio_file,
                    'processed_file': None,
                    'analysis': None,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def cleanup_temp_files(self, file_paths: List[str]):
        """임시 파일 정리"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path) and 'tmp' in file_path:
                    os.remove(file_path)
                    logger.debug(f"임시 파일 삭제: {file_path}")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {file_path}, 오류: {str(e)}")
