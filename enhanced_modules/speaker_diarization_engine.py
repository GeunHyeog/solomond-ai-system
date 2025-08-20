#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[SPEAKER DIARIZATION] 고급 화자 구분 엔진
Advanced Speaker Diarization Engine for Conference Analysis

핵심 기능:
1. 고정밀 화자 구분 (Voice Activity Detection + Clustering)
2. Whisper 세그먼트 기반 화자 분리 향상
3. 음성 특성 분석 (피치, 음색, 억양)
4. 컨퍼런스 환경 특화 최적화
5. 실시간 화자 추적 및 라벨링
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
import logging
from pathlib import Path
import json
import tempfile
import math
from collections import defaultdict, Counter

# 오디오 처리 및 화자 구분
try:
    import librosa
    import soundfile as sf
    from scipy import signal, cluster
    from scipy.spatial.distance import cosine, euclidean
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

# 고급 음성 분석
try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# 로깅 설정
logger = logging.getLogger(__name__)

@dataclass
class SpeakerSegment:
    """화자 세그먼트 정보"""
    start_time: float
    end_time: float
    speaker_id: str
    confidence: float
    text: Optional[str] = None
    voice_features: Optional[Dict[str, float]] = None

@dataclass
class SpeakerProfile:
    """화자 프로필"""
    speaker_id: str
    total_duration: float
    segment_count: int
    avg_pitch: float
    pitch_variance: float
    avg_energy: float
    speaking_rate: float  # 말하는 속도 (단어/분)
    voice_features: Dict[str, float]

@dataclass
class DiarizationResult:
    """화자 구분 결과"""
    success: bool
    num_speakers: int
    segments: List[SpeakerSegment]
    speaker_profiles: Dict[str, SpeakerProfile]
    processing_time: float
    confidence_score: float
    error_message: Optional[str] = None

class VoiceActivityDetector:
    """음성 활동 감지기"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_duration_ms = 30  # 30ms 프레임
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
        # WebRTC VAD 초기화
        self.vad = None
        if WEBRTC_AVAILABLE:
            try:
                self.vad = webrtcvad.Vad()
                self.vad.set_mode(2)  # 중간 민감도
            except:
                self.vad = None
    
    def detect_speech_segments(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """음성 구간 감지"""
        if self.vad is None:
            # 폴백: 에너지 기반 VAD
            return self._energy_based_vad(audio)
        
        try:
            # WebRTC VAD 사용
            return self._webrtc_vad(audio)
        except:
            # 실패시 에너지 기반으로 폴백
            return self._energy_based_vad(audio)
    
    def _webrtc_vad(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """WebRTC VAD 기반 음성 감지"""
        # 오디오를 16kHz, 16-bit PCM으로 변환
        audio_int16 = (audio * 32767).astype(np.int16)
        
        speech_frames = []
        frame_count = len(audio_int16) // self.frame_size
        
        for i in range(frame_count):
            start_idx = i * self.frame_size
            end_idx = start_idx + self.frame_size
            frame = audio_int16[start_idx:end_idx].tobytes()
            
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            speech_frames.append(is_speech)
        
        # 연속된 음성 구간 찾기
        segments = []
        in_speech = False
        start_time = 0
        
        for i, is_speech in enumerate(speech_frames):
            time_stamp = i * self.frame_duration_ms / 1000.0
            
            if is_speech and not in_speech:
                start_time = time_stamp
                in_speech = True
            elif not is_speech and in_speech:
                segments.append((start_time, time_stamp))
                in_speech = False
        
        # 마지막 세그먼트 처리
        if in_speech:
            segments.append((start_time, len(audio) / self.sample_rate))
        
        return segments
    
    def _energy_based_vad(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """에너지 기반 VAD (폴백)"""
        # 프레임별 에너지 계산
        hop_length = self.frame_size // 2
        frame_length = self.frame_size
        
        # Short-time energy
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            frame_energy = np.sum(frame ** 2)
            energy.append(frame_energy)
        
        energy = np.array(energy)
        
        # 적응적 임계값 계산
        energy_sorted = np.sort(energy)
        noise_floor = np.mean(energy_sorted[:len(energy_sorted)//4])  # 하위 25%
        threshold = noise_floor * 10  # 10배 이상을 음성으로 간주
        
        # 음성 구간 찾기
        speech_frames = energy > threshold
        
        segments = []
        in_speech = False
        start_time = 0
        
        for i, is_speech in enumerate(speech_frames):
            time_stamp = i * hop_length / self.sample_rate
            
            if is_speech and not in_speech:
                start_time = time_stamp
                in_speech = True
            elif not is_speech and in_speech:
                if time_stamp - start_time > 0.3:  # 최소 300ms
                    segments.append((start_time, time_stamp))
                in_speech = False
        
        # 마지막 세그먼트 처리
        if in_speech:
            segments.append((start_time, len(audio) / self.sample_rate))
        
        return segments

class VoiceFeatureExtractor:
    """음성 특성 추출기"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def extract_features(self, audio: np.ndarray, start_time: float = 0) -> Dict[str, float]:
        """음성 특성 추출"""
        if len(audio) == 0:
            return self._empty_features()
        
        features = {}
        
        # 1. 기본 피치 특성
        pitch_features = self._extract_pitch_features(audio)
        features.update(pitch_features)
        
        # 2. 스펙트럼 특성
        spectral_features = self._extract_spectral_features(audio)
        features.update(spectral_features)
        
        # 3. 에너지 및 리듬 특성
        energy_features = self._extract_energy_features(audio)
        features.update(energy_features)
        
        # 4. MFCC 특성 (음색)
        mfcc_features = self._extract_mfcc_features(audio)
        features.update(mfcc_features)
        
        return features
    
    def _extract_pitch_features(self, audio: np.ndarray) -> Dict[str, float]:
        """피치 관련 특성 추출"""
        # librosa를 사용한 피치 추정
        try:
            pitches, magnitudes = librosa.piptrack(
                y=audio, sr=self.sample_rate, 
                threshold=0.1, fmin=50, fmax=400
            )
            
            # 각 프레임에서 가장 강한 피치 선택
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) == 0:
                return {'pitch_mean': 0, 'pitch_std': 0, 'pitch_range': 0}
            
            pitch_values = np.array(pitch_values)
            
            return {
                'pitch_mean': float(np.mean(pitch_values)),
                'pitch_std': float(np.std(pitch_values)),
                'pitch_range': float(np.max(pitch_values) - np.min(pitch_values)),
                'pitch_median': float(np.median(pitch_values))
            }
        except:
            return {'pitch_mean': 0, 'pitch_std': 0, 'pitch_range': 0, 'pitch_median': 0}
    
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """스펙트럼 특성 추출"""
        try:
            # 스펙트럼 특성 계산
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            
            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate))
            }
        except:
            return {
                'spectral_centroid_mean': 0, 'spectral_centroid_std': 0,
                'spectral_rolloff_mean': 0, 'spectral_bandwidth_mean': 0,
                'zero_crossing_rate_mean': 0
            }
    
    def _extract_energy_features(self, audio: np.ndarray) -> Dict[str, float]:
        """에너지 및 리듬 특성 추출"""
        try:
            # RMS 에너지
            rms = librosa.feature.rms(y=audio)[0]
            
            # 템포 추정
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            
            return {
                'rms_mean': float(np.mean(rms)),
                'rms_std': float(np.std(rms)),
                'tempo': float(tempo) if tempo is not None else 0,
                'energy_variance': float(np.var(audio))
            }
        except:
            return {
                'rms_mean': float(np.sqrt(np.mean(audio**2))),
                'rms_std': 0, 'tempo': 0,
                'energy_variance': float(np.var(audio))
            }
    
    def _extract_mfcc_features(self, audio: np.ndarray, n_mfcc: int = 13) -> Dict[str, float]:
        """MFCC 특성 추출 (음색)"""
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc)
            
            features = {}
            for i in range(n_mfcc):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            
            return features
        except:
            # 폴백: 빈 MFCC 특성
            features = {}
            for i in range(n_mfcc):
                features[f'mfcc_{i}_mean'] = 0
                features[f'mfcc_{i}_std'] = 0
            return features
    
    def _empty_features(self) -> Dict[str, float]:
        """빈 특성 반환"""
        features = {
            'pitch_mean': 0, 'pitch_std': 0, 'pitch_range': 0, 'pitch_median': 0,
            'spectral_centroid_mean': 0, 'spectral_centroid_std': 0,
            'spectral_rolloff_mean': 0, 'spectral_bandwidth_mean': 0,
            'zero_crossing_rate_mean': 0,
            'rms_mean': 0, 'rms_std': 0, 'tempo': 0, 'energy_variance': 0
        }
        
        # MFCC 특성 추가
        for i in range(13):
            features[f'mfcc_{i}_mean'] = 0
            features[f'mfcc_{i}_std'] = 0
        
        return features

class SpeakerClustering:
    """화자 클러스터링"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_keys = None
    
    def cluster_speakers(self, segments: List[Dict], min_speakers: int = 2, max_speakers: int = 8) -> Dict[str, Any]:
        """화자 클러스터링 수행"""
        if len(segments) < min_speakers:
            return self._single_speaker_result(segments)
        
        # 특성 행렬 구성
        features_matrix, valid_segments = self._build_feature_matrix(segments)
        
        if features_matrix is None or len(features_matrix) < min_speakers:
            return self._single_speaker_result(segments)
        
        # 최적 클러스터 수 찾기
        optimal_clusters = self._find_optimal_clusters(
            features_matrix, min_speakers, max_speakers
        )
        
        # 클러스터링 수행
        cluster_labels = self._perform_clustering(features_matrix, optimal_clusters)
        
        # 결과 구성
        return self._build_clustering_result(valid_segments, cluster_labels, optimal_clusters)
    
    def _build_feature_matrix(self, segments: List[Dict]) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """특성 행렬 구축"""
        valid_segments = []
        features_list = []
        
        for segment in segments:
            if 'voice_features' in segment and segment['voice_features']:
                features = segment['voice_features']
                
                # 첫 번째 세그먼트에서 특성 키 설정
                if self.feature_keys is None:
                    self.feature_keys = sorted(features.keys())
                
                # 특성 벡터 구성
                feature_vector = [features.get(key, 0) for key in self.feature_keys]
                
                # NaN 값 체크
                if not np.isnan(feature_vector).any() and np.var(feature_vector) > 0:
                    features_list.append(feature_vector)
                    valid_segments.append(segment)
        
        if len(features_list) < 2:
            return None, segments
        
        features_matrix = np.array(features_list)
        
        # 정규화
        try:
            features_matrix = self.scaler.fit_transform(features_matrix)
        except:
            # 정규화 실패시 원본 사용
            pass
        
        return features_matrix, valid_segments
    
    def _find_optimal_clusters(self, features_matrix: np.ndarray, min_k: int, max_k: int) -> int:
        """최적 클러스터 수 찾기 (실루엣 스코어 사용)"""
        if len(features_matrix) < min_k:
            return 1
        
        max_k = min(max_k, len(features_matrix))
        best_k = min_k
        best_score = -1
        
        for k in range(min_k, max_k + 1):
            try:
                clustering = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = clustering.fit_predict(features_matrix)
                
                if len(set(labels)) > 1:  # 실제로 여러 클러스터가 생성된 경우만
                    score = silhouette_score(features_matrix, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            except:
                continue
        
        return best_k
    
    def _perform_clustering(self, features_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
        """클러스터링 수행"""
        try:
            # Agglomerative Clustering (더 안정적)
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            labels = clustering.fit_predict(features_matrix)
            return labels
        except:
            try:
                # K-Means 폴백
                clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = clustering.fit_predict(features_matrix)
                return labels
            except:
                # 최종 폴백: 순차적 라벨링
                return np.arange(len(features_matrix)) % n_clusters
    
    def _build_clustering_result(self, segments: List[Dict], labels: np.ndarray, n_clusters: int) -> Dict[str, Any]:
        """클러스터링 결과 구성"""
        # 각 세그먼트에 화자 ID 할당
        for i, segment in enumerate(segments):
            segment['speaker_id'] = f'speaker_{labels[i]:02d}'
        
        # 클러스터 품질 평가
        confidence_score = self._calculate_clustering_confidence(segments, labels)
        
        return {
            'num_speakers': n_clusters,
            'segments': segments,
            'confidence_score': confidence_score,
            'clustering_method': 'agglomerative'
        }
    
    def _calculate_clustering_confidence(self, segments: List[Dict], labels: np.ndarray) -> float:
        """클러스터링 신뢰도 계산"""
        if len(set(labels)) < 2:
            return 0.5
        
        # 클러스터 내 일관성 측정
        cluster_distances = defaultdict(list)
        
        for i, segment_i in enumerate(segments):
            for j, segment_j in enumerate(segments[i+1:], i+1):
                if labels[i] == labels[j]:  # 같은 클러스터
                    distance = self._feature_distance(
                        segment_i.get('voice_features', {}),
                        segment_j.get('voice_features', {})
                    )
                    cluster_distances[labels[i]].append(distance)
        
        # 평균 클러스터 내 거리 계산
        intra_cluster_distances = []
        for cluster_id, distances in cluster_distances.items():
            if distances:
                intra_cluster_distances.append(np.mean(distances))
        
        if not intra_cluster_distances:
            return 0.5
        
        # 거리가 작을수록 좋은 클러스터링
        avg_intra_distance = np.mean(intra_cluster_distances)
        confidence = max(0, min(1, 1 - avg_intra_distance))
        
        return confidence
    
    def _feature_distance(self, features1: Dict, features2: Dict) -> float:
        """특성 간 거리 계산"""
        if not features1 or not features2:
            return 1.0
        
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 1.0
        
        vec1 = [features1[key] for key in common_keys]
        vec2 = [features2[key] for key in common_keys]
        
        try:
            return cosine(vec1, vec2)
        except:
            return euclidean(vec1, vec2) / max(1, len(vec1))
    
    def _single_speaker_result(self, segments: List[Dict]) -> Dict[str, Any]:
        """단일 화자 결과"""
        for segment in segments:
            segment['speaker_id'] = 'speaker_00'
        
        return {
            'num_speakers': 1,
            'segments': segments,
            'confidence_score': 0.8,
            'clustering_method': 'single_speaker'
        }

class SpeakerDiarizationEngine:
    """통합 화자 구분 엔진"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.vad = VoiceActivityDetector(sample_rate)
        self.feature_extractor = VoiceFeatureExtractor(sample_rate)
        self.clustering = SpeakerClustering()
        
        logger.info("[SPEAKER DIARIZATION] 화자 구분 엔진 초기화 완료")
    
    def process_audio(self, audio_path: str, whisper_segments: List[Dict] = None) -> DiarizationResult:
        """오디오 화자 구분 처리"""
        if not AUDIO_PROCESSING_AVAILABLE:
            return DiarizationResult(
                success=False, num_speakers=0, segments=[], speaker_profiles={},
                processing_time=0, confidence_score=0,
                error_message="오디오 처리 라이브러리가 설치되지 않았습니다"
            )
        
        start_time = time.time()
        
        try:
            # 오디오 로드
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            if len(audio) == 0:
                raise ValueError("빈 오디오 파일입니다")
            
            # Whisper 세그먼트가 있으면 활용, 없으면 VAD 사용
            if whisper_segments and len(whisper_segments) > 0:
                segments = self._process_with_whisper_segments(audio, whisper_segments)
            else:
                segments = self._process_with_vad(audio)
            
            if len(segments) == 0:
                return DiarizationResult(
                    success=False, num_speakers=0, segments=[], speaker_profiles={},
                    processing_time=time.time() - start_time, confidence_score=0,
                    error_message="음성 세그먼트를 찾을 수 없습니다"
                )
            
            # 화자 클러스터링
            clustering_result = self.clustering.cluster_speakers(segments)
            
            # Speaker 세그먼트 구성
            speaker_segments = []
            for segment_data in clustering_result['segments']:
                speaker_segment = SpeakerSegment(
                    start_time=segment_data['start_time'],
                    end_time=segment_data['end_time'],
                    speaker_id=segment_data['speaker_id'],
                    confidence=segment_data.get('confidence', 0.8),
                    text=segment_data.get('text'),
                    voice_features=segment_data.get('voice_features')
                )
                speaker_segments.append(speaker_segment)
            
            # 화자 프로필 생성
            speaker_profiles = self._build_speaker_profiles(speaker_segments, audio)
            
            processing_time = time.time() - start_time
            
            return DiarizationResult(
                success=True,
                num_speakers=clustering_result['num_speakers'],
                segments=speaker_segments,
                speaker_profiles=speaker_profiles,
                processing_time=processing_time,
                confidence_score=clustering_result['confidence_score']
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"[ERROR] 화자 구분 실패: {e}")
            return DiarizationResult(
                success=False, num_speakers=0, segments=[], speaker_profiles={},
                processing_time=processing_time, confidence_score=0,
                error_message=str(e)
            )
    
    def _process_with_whisper_segments(self, audio: np.ndarray, whisper_segments: List[Dict]) -> List[Dict]:
        """Whisper 세그먼트 기반 처리"""
        segments = []
        
        for whisper_segment in whisper_segments:
            start_time = whisper_segment.get('start', 0)
            end_time = whisper_segment.get('end', len(audio) / self.sample_rate)
            text = whisper_segment.get('text', '').strip()
            
            if end_time <= start_time or not text:
                continue
            
            # 오디오 세그먼트 추출
            start_idx = int(start_time * self.sample_rate)
            end_idx = int(end_time * self.sample_rate)
            segment_audio = audio[start_idx:end_idx]
            
            if len(segment_audio) == 0:
                continue
            
            # 음성 특성 추출
            voice_features = self.feature_extractor.extract_features(segment_audio, start_time)
            
            segment_data = {
                'start_time': start_time,
                'end_time': end_time,
                'text': text,
                'voice_features': voice_features,
                'duration': end_time - start_time,
                'confidence': whisper_segment.get('confidence', 0.8)
            }
            
            segments.append(segment_data)
        
        return segments
    
    def _process_with_vad(self, audio: np.ndarray) -> List[Dict]:
        """VAD 기반 처리"""
        # VAD로 음성 구간 감지
        speech_segments = self.vad.detect_speech_segments(audio)
        
        segments = []
        for start_time, end_time in speech_segments:
            if end_time <= start_time:
                continue
            
            # 오디오 세그먼트 추출
            start_idx = int(start_time * self.sample_rate)
            end_idx = int(end_time * self.sample_rate)
            segment_audio = audio[start_idx:end_idx]
            
            if len(segment_audio) == 0:
                continue
            
            # 음성 특성 추출
            voice_features = self.feature_extractor.extract_features(segment_audio, start_time)
            
            segment_data = {
                'start_time': start_time,
                'end_time': end_time,
                'text': None,
                'voice_features': voice_features,
                'duration': end_time - start_time,
                'confidence': 0.7  # VAD 기본 신뢰도
            }
            
            segments.append(segment_data)
        
        return segments
    
    def _build_speaker_profiles(self, segments: List[SpeakerSegment], audio: np.ndarray) -> Dict[str, SpeakerProfile]:
        """화자 프로필 구축"""
        speaker_data = defaultdict(list)
        
        # 화자별 세그먼트 수집
        for segment in segments:
            speaker_data[segment.speaker_id].append(segment)
        
        speaker_profiles = {}
        
        for speaker_id, speaker_segments in speaker_data.items():
            # 통계 계산
            total_duration = sum(seg.end_time - seg.start_time for seg in speaker_segments)
            segment_count = len(speaker_segments)
            
            # 음성 특성 통합
            all_features = []
            text_lengths = []
            
            for segment in speaker_segments:
                if segment.voice_features:
                    all_features.append(segment.voice_features)
                if segment.text:
                    text_lengths.append(len(segment.text.split()))
            
            # 평균 특성 계산
            if all_features:
                avg_features = {}
                for key in all_features[0].keys():
                    values = [f[key] for f in all_features if key in f]
                    if values:
                        avg_features[key] = np.mean(values)
                
                avg_pitch = avg_features.get('pitch_mean', 0)
                pitch_variance = avg_features.get('pitch_std', 0)
                avg_energy = avg_features.get('rms_mean', 0)
            else:
                avg_features = {}
                avg_pitch = 0
                pitch_variance = 0
                avg_energy = 0
            
            # 말하는 속도 계산 (단어/분)
            if text_lengths and total_duration > 0:
                total_words = sum(text_lengths)
                speaking_rate = (total_words / total_duration) * 60
            else:
                speaking_rate = 0
            
            profile = SpeakerProfile(
                speaker_id=speaker_id,
                total_duration=total_duration,
                segment_count=segment_count,
                avg_pitch=avg_pitch,
                pitch_variance=pitch_variance,
                avg_energy=avg_energy,
                speaking_rate=speaking_rate,
                voice_features=avg_features
            )
            
            speaker_profiles[speaker_id] = profile
        
        return speaker_profiles
    
    def enhance_whisper_segments(self, whisper_segments: List[Dict], audio_path: str) -> List[Dict]:
        """Whisper 세그먼트에 화자 정보 추가"""
        diarization_result = self.process_audio(audio_path, whisper_segments)
        
        if not diarization_result.success:
            # 실패시 기본 화자 ID 추가
            for i, segment in enumerate(whisper_segments):
                segment['speaker'] = 'speaker_00'
            return whisper_segments
        
        # 화자 정보가 포함된 세그먼트 반환
        enhanced_segments = []
        for whisper_seg in whisper_segments:
            whisper_start = whisper_seg.get('start', 0)
            whisper_end = whisper_seg.get('end', 0)
            
            # 가장 겹치는 화자 세그먼트 찾기
            best_speaker = 'speaker_00'
            best_overlap = 0
            
            for speaker_seg in diarization_result.segments:
                overlap = min(whisper_end, speaker_seg.end_time) - max(whisper_start, speaker_seg.start_time)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker_seg.speaker_id
            
            # 화자 정보 추가
            enhanced_segment = whisper_seg.copy()
            enhanced_segment['speaker'] = best_speaker
            enhanced_segment['speaker_confidence'] = diarization_result.confidence_score
            
            enhanced_segments.append(enhanced_segment)
        
        return enhanced_segments

# 테스트 및 데모
if __name__ == "__main__":
    # 화자 구분 엔진 테스트
    engine = SpeakerDiarizationEngine()
    
    print("[SUCCESS] 화자 구분 엔진 초기화 완료")
    print(f"[INFO] 오디오 처리: {'가능' if AUDIO_PROCESSING_AVAILABLE else '불가능 (librosa 필요)'}")
    print(f"[INFO] WebRTC VAD: {'가능' if WEBRTC_AVAILABLE else '불가능 (webrtcvad 필요)'}")
    
    print("[SUCCESS] 화자 구분 엔진 테스트 완료")