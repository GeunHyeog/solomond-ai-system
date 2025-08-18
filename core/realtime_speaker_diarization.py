#!/usr/bin/env python3
"""
실시간 추적 통합 화자 분리 시스템
오디오에서 화자를 분리하고 각 화자별 발언 내용을 STT와 매핑하여 제공
"""

import os
import time
import tempfile
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json

# 오디오 처리를 위한 라이브러리들
try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

class RealtimeSpeakerDiarization:
    """실시간 추적 통합 화자 분리 시스템"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 실시간 추적 시스템 로드 - 강화된 모듈 검색 시스템
        try:
            import sys
            import os
            from pathlib import Path
            
            # 현재 파일의 디렉토리
            current_dir = Path(__file__).parent
            
            # 프로젝트 루트 디렉토리
            project_root = current_dir.parent
            
            # sys.path에 필요한 경로들 추가
            paths_to_add = [
                str(current_dir),
                str(project_root),
                str(project_root / "core")
            ]
            
            for path in paths_to_add:
                if path not in sys.path:
                    sys.path.insert(0, path)
            
            # 동적 임포트로 모듈 로드
            import importlib.util
            
            # realtime_progress_tracker 모듈 로드
            tracker_path = current_dir / "realtime_progress_tracker.py"
            if tracker_path.exists():
                spec = importlib.util.spec_from_file_location("realtime_progress_tracker", tracker_path)
                tracker_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(tracker_module)
                self.progress_tracker = getattr(tracker_module, 'global_progress_tracker', None)
            else:
                self.progress_tracker = None
            
            # mcp_auto_problem_solver 모듈 로드
            solver_path = current_dir / "mcp_auto_problem_solver.py"
            if solver_path.exists():
                spec = importlib.util.spec_from_file_location("mcp_auto_problem_solver", solver_path)
                solver_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(solver_module)
                self.problem_solver = getattr(solver_module, 'global_mcp_solver', None)
            else:
                self.problem_solver = None
            
            # 로드 성공 여부 확인
            if self.progress_tracker is not None and self.problem_solver is not None:
                self.realtime_tracking_available = True
                self.logger.info("🎯 실시간 추적 시스템 연동 완료")
            else:
                self.realtime_tracking_available = False
                self.logger.warning("일부 실시간 추적 모듈 로드 실패")
        except ImportError as e:
            self.progress_tracker = None
            self.problem_solver = None
            self.realtime_tracking_available = False
            self.logger.warning(f"실시간 추적 시스템 로드 실패: {e}")
        
        # 화자 식별 시스템 로드 - 동일한 동적 로딩 시스템 사용
        try:
            # current_dir이 이미 정의되어 있지 않은 경우를 대비
            if 'current_dir' not in locals():
                current_dir = Path(__file__).parent
                
            speaker_id_path = current_dir / "speaker_identification.py"
            if speaker_id_path.exists():
                spec = importlib.util.spec_from_file_location("speaker_identification", speaker_id_path)
                speaker_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(speaker_module)
                self.speaker_identifier = getattr(speaker_module, 'global_speaker_identifier', None)
                
                if self.speaker_identifier is not None:
                    self.speaker_identification_available = True
                    self.logger.info("🎭 화자 식별 시스템 연동 완료")
                else:
                    self.speaker_identification_available = False
                    self.logger.warning("화자 식별 시스템 인스턴스 로드 실패")
            else:
                self.speaker_identifier = None
                self.speaker_identification_available = False
                self.logger.warning("speaker_identification.py 파일을 찾을 수 없음")
        except ImportError as e:
            self.speaker_identifier = None
            self.speaker_identification_available = False
            self.logger.warning(f"화자 식별 시스템 로드 실패: {e}")
        
        # 오디오 분석 설정
        self.sample_rate = 16000
        self.frame_duration = 0.03  # 30ms 프레임
        self.hop_length = int(self.sample_rate * self.frame_duration)
        
        # 화자 분리 설정
        self.max_speakers = 6  # 최대 화자 수
        self.min_segment_duration = 1.0  # 최소 발언 지속 시간 (초)
        self.energy_threshold = 0.01  # 음성 활동 감지 임계값
        
        # VAD (Voice Activity Detection) 설정
        if WEBRTC_AVAILABLE:
            self.vad = webrtcvad.Vad(2)  # 중간 민감도
            self.vad_available = True
        else:
            self.vad = None
            self.vad_available = False
        
        self._check_dependencies()
        self.logger.info("🎤 실시간 화자 분리 시스템 초기화 완료")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.RealtimeSpeakerDiarization')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _check_dependencies(self):
        """의존성 확인"""
        deps_status = {
            "librosa": LIBROSA_AVAILABLE,
            "sklearn": SKLEARN_AVAILABLE, 
            "webrtcvad": WEBRTC_AVAILABLE,
            "realtime_tracking": self.realtime_tracking_available,
            "speaker_identification": self.speaker_identification_available
        }
        
        self.logger.info(f"📦 의존성 상태: {deps_status}")
        
        if not LIBROSA_AVAILABLE:
            self.logger.warning("⚠️ librosa 미설치 - 오디오 처리 기능 제한")
        if not SKLEARN_AVAILABLE:
            self.logger.warning("⚠️ sklearn 미설치 - 화자 클러스터링 기능 제한")
        if not WEBRTC_AVAILABLE:
            self.logger.warning("⚠️ webrtcvad 미설치 - 음성 활동 감지 기능 제한")
    
    def analyze_speakers_in_audio(self, audio_file: str, transcript: str = "", 
                                progress_container=None) -> Dict[str, Any]:
        """오디오 파일에서 화자 분리 및 발언 매핑"""
        
        if self.realtime_tracking_available and progress_container:
            self.progress_tracker.update_progress_with_time(
                "🎤 화자 분리 분석 시작",
                f"파일: {os.path.basename(audio_file)}"
            )
        
        start_time = time.time()
        
        try:
            # 1. 오디오 전처리 및 로드
            if self.realtime_tracking_available and progress_container:
                self.progress_tracker.update_progress_with_time(
                    "📊 오디오 파일 로드 및 전처리",
                    "음성 신호 분석 중..."
                )
            
            audio_data, segments = self._load_and_preprocess_audio(audio_file)
            
            if audio_data is None:
                return self._create_error_result("오디오 로드 실패", audio_file)
            
            # 2. 음성 활동 감지 (VAD)
            if self.realtime_tracking_available and progress_container:
                self.progress_tracker.update_progress_with_time(
                    "🔊 음성 활동 구간 감지",
                    f"총 {len(audio_data)/self.sample_rate:.1f}초 분석 중..."
                )
            
            voice_segments = self._detect_voice_activity(audio_data)
            
            # 3. 화자 특성 추출
            if self.realtime_tracking_available and progress_container:
                self.progress_tracker.update_progress_with_time(
                    "🎯 화자 특성 추출",
                    f"{len(voice_segments)}개 음성 구간 분석..."
                )
            
            speaker_features = self._extract_speaker_features(audio_data, voice_segments)
            
            # 4. 화자 클러스터링
            if self.realtime_tracking_available and progress_container:
                self.progress_tracker.update_progress_with_time(
                    "👥 화자 클러스터링",
                    "유사한 음성 특성 그룹핑..."
                )
            
            speaker_labels, optimal_speakers = self._cluster_speakers(speaker_features)
            
            # 5. 화자별 시간 구간 생성
            if self.realtime_tracking_available and progress_container:
                self.progress_tracker.update_progress_with_time(
                    "⏰ 화자별 시간 구간 생성",
                    f"{optimal_speakers}명 화자 구간 매핑..."
                )
            
            speaker_timeline = self._create_speaker_timeline(voice_segments, speaker_labels)
            
            # 6. STT 결과와 화자 매핑
            if self.realtime_tracking_available and progress_container:
                self.progress_tracker.update_progress_with_time(
                    "📝 화자별 발언 내용 매핑",
                    "STT 결과와 화자 시간대 매칭..."
                )
            
            speaker_statements = self._map_transcript_to_speakers(transcript, speaker_timeline)
            
            # 7. 화자 식별 및 역할 분석
            if self.realtime_tracking_available and progress_container:
                self.progress_tracker.update_progress_with_time(
                    "🎭 화자 식별 및 역할 분석",
                    "텍스트 기반 화자 정보 추출..."
                )
            
            speaker_identification = self._identify_speakers_in_statements(speaker_statements, transcript)
            
            # 8. 결과 통합 및 사용자 친화적 요약 생성
            processing_time = time.time() - start_time
            
            result = {
                "status": "success",
                "audio_file": audio_file,
                "processing_time": round(processing_time, 2),
                "speaker_count": optimal_speakers,
                "total_duration": len(audio_data) / self.sample_rate,
                "voice_activity_ratio": self._calculate_voice_activity_ratio(voice_segments, len(audio_data)),
                "speaker_timeline": speaker_timeline,
                "speaker_statements": speaker_statements,
                "speaker_identification": speaker_identification,
                "analysis_quality": self._assess_analysis_quality(speaker_features, voice_segments),
                "user_summary": self._create_user_friendly_summary(speaker_statements, speaker_identification, optimal_speakers),
                "detailed_breakdown": self._create_detailed_breakdown(speaker_statements, speaker_timeline),
                "timestamp": datetime.now().isoformat()
            }
            
            if self.realtime_tracking_available and progress_container:
                self.progress_tracker.update_progress_with_time(
                    f"✅ 화자 분리 완료: {optimal_speakers}명 식별",
                    f"처리 시간: {processing_time:.2f}초"
                )
            
            self.logger.info(f"🎉 화자 분리 분석 완료: {optimal_speakers}명, {processing_time:.2f}초")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"화자 분리 분석 오류: {str(e)}"
            self.logger.error(error_msg)
            
            # MCP 자동 문제 해결 시도
            if self.realtime_tracking_available and self.problem_solver:
                try:
                    problem_result = self.problem_solver.detect_and_solve_problems(
                        memory_usage_mb=100,
                        processing_time=processing_time,
                        file_info={'name': audio_file, 'size_mb': 0},
                        error_message=str(e)
                    )
                    
                    if problem_result['solutions_found']:
                        self.logger.info(f"자동 해결책 {len(problem_result['solutions_found'])}개 발견")
                        if progress_container:
                            with progress_container.container():
                                import streamlit as st
                                st.warning("화자 분리 중 문제가 발생했습니다. 자동 해결책을 확인하세요:")
                                for i, solution in enumerate(problem_result['solutions_found'][:2], 1):
                                    st.write(f"{i}. {solution.get('title', '해결책')}")
                                    if solution.get('url'):
                                        st.write(f"   참고: {solution['url']}")
                except Exception as mcp_error:
                    self.logger.warning(f"MCP 자동 문제 해결 실패: {mcp_error}")
            
            return self._create_error_result(error_msg, audio_file, processing_time)
    
    def _load_and_preprocess_audio(self, audio_file: str) -> Tuple[Optional[np.ndarray], Optional[List]]:
        """오디오 로드 및 전처리"""
        
        if not LIBROSA_AVAILABLE:
            self.logger.error("librosa가 설치되지 않아 오디오 처리 불가")
            return None, None
        
        try:
            # 오디오 로드
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # 무음 제거 및 정규화
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            y_normalized = librosa.util.normalize(y_trimmed)
            
            # 프레임 단위로 분할
            frames = librosa.util.frame(y_normalized, frame_length=self.hop_length, 
                                      hop_length=self.hop_length, axis=0)
            
            return y_normalized, frames
            
        except Exception as e:
            self.logger.error(f"오디오 전처리 실패: {e}")
            return None, None
    
    def _detect_voice_activity(self, audio_data: np.ndarray) -> List[Dict[str, float]]:
        """음성 활동 구간 감지"""
        
        voice_segments = []
        
        if self.vad_available and WEBRTC_AVAILABLE:
            # WebRTC VAD 사용
            frame_duration = 0.03  # 30ms
            frame_size = int(self.sample_rate * frame_duration)
            
            current_segment_start = None
            
            for i in range(0, len(audio_data) - frame_size, frame_size):
                frame = audio_data[i:i + frame_size]
                
                # 16-bit PCM으로 변환
                frame_int16 = (frame * 32767).astype(np.int16)
                frame_bytes = frame_int16.tobytes()
                
                try:
                    is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
                    time_position = i / self.sample_rate
                    
                    if is_speech and current_segment_start is None:
                        current_segment_start = time_position
                    elif not is_speech and current_segment_start is not None:
                        # 음성 구간 종료
                        duration = time_position - current_segment_start
                        if duration >= self.min_segment_duration:
                            voice_segments.append({
                                "start": current_segment_start,
                                "end": time_position,
                                "duration": duration
                            })
                        current_segment_start = None
                        
                except Exception as e:
                    # VAD 오류시 에너지 기반 fallback
                    pass
            
            # 마지막 구간 처리
            if current_segment_start is not None:
                final_time = len(audio_data) / self.sample_rate
                duration = final_time - current_segment_start
                if duration >= self.min_segment_duration:
                    voice_segments.append({
                        "start": current_segment_start,
                        "end": final_time,
                        "duration": duration
                    })
        
        # VAD가 없거나 실패한 경우 에너지 기반 감지
        if not voice_segments:
            voice_segments = self._energy_based_vad(audio_data)
        
        return voice_segments
    
    def _energy_based_vad(self, audio_data: np.ndarray) -> List[Dict[str, float]]:
        """에너지 기반 음성 활동 감지"""
        
        if not LIBROSA_AVAILABLE:
            return []
        
        # 짧은 프레임으로 에너지 계산
        frame_length = int(self.sample_rate * 0.025)  # 25ms
        hop_length = int(self.sample_rate * 0.01)     # 10ms
        
        # 스펙트럴 센트로이드와 영교차율 계산
        energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, 
                                   hop_length=hop_length)[0]
        
        # 임계값 이상인 구간 찾기
        threshold = np.percentile(energy, 30)  # 하위 30% 제거
        voice_mask = energy > threshold
        
        # 연속 구간 찾기
        voice_segments = []
        in_voice = False
        start_time = 0
        
        for i, is_voice in enumerate(voice_mask):
            time_position = i * hop_length / self.sample_rate
            
            if is_voice and not in_voice:
                start_time = time_position
                in_voice = True
            elif not is_voice and in_voice:
                duration = time_position - start_time
                if duration >= self.min_segment_duration:
                    voice_segments.append({
                        "start": start_time,
                        "end": time_position,
                        "duration": duration
                    })
                in_voice = False
        
        # 마지막 구간 처리
        if in_voice:
            final_time = len(audio_data) / self.sample_rate
            duration = final_time - start_time
            if duration >= self.min_segment_duration:
                voice_segments.append({
                    "start": start_time,
                    "end": final_time,
                    "duration": duration
                })
        
        return voice_segments
    
    def _extract_speaker_features(self, audio_data: np.ndarray, voice_segments: List[Dict]) -> List[np.ndarray]:
        """각 음성 구간에서 화자 특성 추출"""
        
        if not LIBROSA_AVAILABLE:
            return []
        
        features = []
        
        for segment in voice_segments:
            start_sample = int(segment["start"] * self.sample_rate)
            end_sample = int(segment["end"] * self.sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            
            if len(segment_audio) < self.sample_rate * 0.5:  # 0.5초 미만은 스킵
                continue
            
            # MFCC 특성 추출 (화자 식별에 유용)
            mfccs = librosa.feature.mfcc(y=segment_audio, sr=self.sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # 스펙트럴 특성 추출
            spectral_centroids = librosa.feature.spectral_centroid(y=segment_audio, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=segment_audio, sr=self.sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(segment_audio)
            
            # 피치 특성 (기본 주파수)
            pitches, magnitudes = librosa.piptrack(y=segment_audio, sr=self.sample_rate)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            # 특성 벡터 생성
            feature_vector = np.concatenate([
                mfcc_mean,
                mfcc_std,
                [np.mean(spectral_centroids)],
                [np.mean(spectral_rolloff)],
                [np.mean(zero_crossing_rate)],
                [pitch_mean]
            ])
            
            features.append(feature_vector)
        
        return features
    
    def _cluster_speakers(self, features: List[np.ndarray]) -> Tuple[List[int], int]:
        """화자 특성을 클러스터링하여 화자 분리"""
        
        if not SKLEARN_AVAILABLE or not features:
            # 기본 동작: 모든 구간을 단일 화자로 처리
            return [0] * len(features), 1
        
        features_array = np.array(features)
        
        # 특성 정규화
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_array)
        
        # 최적의 클러스터 수 찾기 (Elbow method)
        best_k = 1
        min_inertia = float('inf')
        
        for k in range(1, min(self.max_speakers + 1, len(features) + 1)):
            if k > len(features):
                break
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_normalized)
            
            if k == 1:
                best_k = 1
                best_labels = labels
            else:
                # 실루엣 스코어나 inertia로 최적 클러스터 수 결정
                inertia = kmeans.inertia_
                if inertia < min_inertia and k <= 4:  # 일반적으로 4명 이하
                    min_inertia = inertia
                    best_k = k
                    best_labels = labels
        
        return best_labels.tolist(), best_k
    
    def _create_speaker_timeline(self, voice_segments: List[Dict], speaker_labels: List[int]) -> List[Dict]:
        """화자별 시간라인 생성"""
        
        timeline = []
        
        for i, (segment, label) in enumerate(zip(voice_segments, speaker_labels)):
            timeline.append({
                "segment_id": i + 1,
                "speaker_id": f"SPEAKER_{label:02d}",
                "start_time": segment["start"],
                "end_time": segment["end"],
                "duration": segment["duration"],
                "start_formatted": self._format_time(segment["start"]),
                "end_formatted": self._format_time(segment["end"]),
                "duration_formatted": f"{segment['duration']:.1f}초"
            })
        
        return timeline
    
    def _map_transcript_to_speakers(self, transcript: str, speaker_timeline: List[Dict]) -> Dict[str, List[Dict]]:
        """STT 결과를 화자별로 매핑"""
        
        if not transcript or not speaker_timeline:
            return {}
        
        # 간단한 시간 기반 매핑 (실제로는 더 정교한 알고리즘 필요)
        sentences = self._split_transcript_by_sentences(transcript)
        total_duration = speaker_timeline[-1]["end_time"] if speaker_timeline else 0
        
        speaker_statements = {}
        sentence_duration = total_duration / len(sentences) if sentences else 0
        
        sentence_index = 0
        for segment in speaker_timeline:
            speaker_id = segment["speaker_id"]
            
            if speaker_id not in speaker_statements:
                speaker_statements[speaker_id] = []
            
            # 해당 시간대에 해당하는 문장들 찾기
            segment_start = segment["start_time"]
            segment_end = segment["end_time"]
            
            # 간단한 매핑: 시간 비율로 문장 할당
            start_sentence_idx = int((segment_start / total_duration) * len(sentences))
            end_sentence_idx = int((segment_end / total_duration) * len(sentences))
            
            start_sentence_idx = max(0, min(start_sentence_idx, len(sentences) - 1))
            end_sentence_idx = max(start_sentence_idx, min(end_sentence_idx, len(sentences)))
            
            segment_sentences = sentences[start_sentence_idx:end_sentence_idx + 1]
            
            if segment_sentences:
                speaker_statements[speaker_id].append({
                    "segment_id": segment["segment_id"],
                    "start_time": segment["start_formatted"],
                    "end_time": segment["end_formatted"],
                    "duration": segment["duration_formatted"],
                    "content": " ".join(segment_sentences),
                    "sentence_count": len(segment_sentences)
                })
        
        return speaker_statements
    
    def _identify_speakers_in_statements(self, speaker_statements: Dict, full_transcript: str) -> Dict[str, Any]:
        """화자별 발언에서 화자 정보 식별"""
        
        if not self.speaker_identification_available:
            return {"status": "speaker_identification_unavailable"}
        
        try:
            # 전체 텍스트에서 화자 식별
            identification_result = self.speaker_identifier.analyze_speakers(full_transcript)
            
            # 각 화자별 발언 내용 분석
            speaker_details = {}
            
            for speaker_id, statements in speaker_statements.items():
                combined_text = " ".join([stmt["content"] for stmt in statements])
                
                # 개별 화자 분석
                individual_analysis = self.speaker_identifier.analyze_speakers(combined_text)
                
                speaker_details[speaker_id] = {
                    "total_statements": len(statements),
                    "total_content_length": len(combined_text),
                    "identified_names": individual_analysis.get("identified_speakers", []),
                    "expert_roles": individual_analysis.get("expert_roles", {}),
                    "key_statements": individual_analysis.get("key_statements", {}),
                    "analysis_confidence": individual_analysis.get("analysis_confidence", 0.0)
                }
            
            return {
                "status": "success",
                "global_analysis": identification_result,
                "speaker_details": speaker_details
            }
            
        except Exception as e:
            self.logger.error(f"화자 식별 분석 실패: {e}")
            return {"status": "error", "error": str(e)}
    
    def _split_transcript_by_sentences(self, transcript: str) -> List[str]:
        """텍스트를 문장 단위로 분할"""
        
        import re
        
        # 한국어 문장 분할
        sentences = re.split(r'[.!?]\s+', transcript)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]  # 너무 짧은 문장 제외
        
        return sentences
    
    def _calculate_voice_activity_ratio(self, voice_segments: List[Dict], total_samples: int) -> float:
        """음성 활동 비율 계산"""
        
        total_voice_duration = sum(segment["duration"] for segment in voice_segments)
        total_duration = total_samples / self.sample_rate
        
        return round(total_voice_duration / total_duration, 3) if total_duration > 0 else 0.0
    
    def _assess_analysis_quality(self, features: List[np.ndarray], voice_segments: List[Dict]) -> Dict[str, Any]:
        """분석 품질 평가"""
        
        quality_score = 0.0
        quality_factors = []
        
        # 음성 구간 수 평가
        if len(voice_segments) >= 3:
            quality_score += 0.3
            quality_factors.append("충분한 음성 구간 감지")
        elif len(voice_segments) >= 1:
            quality_score += 0.1
            quality_factors.append("최소 음성 구간 감지")
        
        # 특성 추출 품질 평가
        if len(features) >= 3:
            quality_score += 0.3
            quality_factors.append("충분한 화자 특성 추출")
        
        # 총 음성 길이 평가
        total_voice_duration = sum(segment["duration"] for segment in voice_segments)
        if total_voice_duration >= 30:
            quality_score += 0.2
            quality_factors.append("충분한 음성 길이")
        elif total_voice_duration >= 10:
            quality_score += 0.1
            quality_factors.append("최소 음성 길이")
        
        # 의존성 상태 평가
        if LIBROSA_AVAILABLE and SKLEARN_AVAILABLE:
            quality_score += 0.2
            quality_factors.append("필수 라이브러리 사용 가능")
        
        return {
            "quality_score": round(min(quality_score, 1.0), 2),
            "quality_level": "높음" if quality_score >= 0.8 else "중간" if quality_score >= 0.5 else "낮음",
            "quality_factors": quality_factors
        }
    
    def _create_user_friendly_summary(self, speaker_statements: Dict, speaker_identification: Dict, 
                                    speaker_count: int) -> str:
        """사용자 친화적 요약 생성"""
        
        summary_parts = []
        
        # 기본 정보
        summary_parts.append(f"🎤 **화자 분리 결과: {speaker_count}명 감지**\n")
        
        # 각 화자별 요약
        for speaker_id, statements in speaker_statements.items():
            speaker_num = speaker_id.replace("SPEAKER_", "").lstrip("0") or "1"
            total_statements = len(statements)
            total_duration = sum(float(stmt["duration"].replace("초", "")) for stmt in statements)
            
            summary_parts.append(f"👤 **화자 {speaker_num}**")
            summary_parts.append(f"   📝 발언 구간: {total_statements}개")
            summary_parts.append(f"   ⏱️ 총 발언 시간: {total_duration:.1f}초")
            
            # 주요 발언 샘플
            if statements:
                first_statement = statements[0]["content"][:100]
                summary_parts.append(f"   💬 주요 발언: \"{first_statement}...\"")
            
            # 식별된 정보 (있는 경우)
            if speaker_identification.get("status") == "success":
                speaker_details = speaker_identification.get("speaker_details", {}).get(speaker_id, {})
                identified_names = speaker_details.get("identified_names", [])
                if identified_names:
                    name = identified_names[0].get("name", "")
                    summary_parts.append(f"   🏷️ 식별된 이름: {name}")
            
            summary_parts.append("")
        
        # 전체 대화 특성
        if speaker_count > 1:
            summary_parts.append(f"🗣️ **대화 특성**: {speaker_count}명 간의 대화로 분석됨")
        else:
            summary_parts.append("🎤 **발표 특성**: 단일 화자 발표로 분석됨")
        
        return "\\n".join(summary_parts)
    
    def _create_detailed_breakdown(self, speaker_statements: Dict, speaker_timeline: List[Dict]) -> List[Dict]:
        """상세 분석 결과 생성"""
        
        breakdown = []
        
        for segment in speaker_timeline:
            speaker_id = segment["speaker_id"]
            segment_id = segment["segment_id"]
            
            # 해당 세그먼트의 발언 내용 찾기
            segment_content = ""
            if speaker_id in speaker_statements:
                for stmt in speaker_statements[speaker_id]:
                    if stmt["segment_id"] == segment_id:
                        segment_content = stmt["content"]
                        break
            
            breakdown.append({
                "segment_id": segment_id,
                "speaker_id": speaker_id,
                "time_range": f"{segment['start_formatted']} - {segment['end_formatted']}",
                "duration": segment["duration_formatted"],
                "content": segment_content,
                "content_length": len(segment_content),
                "word_count": len(segment_content.split()) if segment_content else 0
            })
        
        return breakdown
    
    def _format_time(self, seconds: float) -> str:
        """시간을 읽기 쉬운 형태로 포맷"""
        
        if seconds < 60:
            return f"{seconds:.1f}초"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}분 {secs:.1f}초"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}시간 {minutes}분 {secs:.1f}초"
    
    def _create_error_result(self, error_msg: str, audio_file: str, processing_time: float = 0) -> Dict[str, Any]:
        """오류 결과 생성"""
        
        return {
            "status": "error",
            "error": error_msg,
            "audio_file": audio_file,
            "processing_time": processing_time,
            "speaker_count": 0,
            "user_summary": f"❌ 화자 분리 분석 실패: {error_msg}",
            "timestamp": datetime.now().isoformat(),
            "dependencies_status": {
                "librosa": LIBROSA_AVAILABLE,
                "sklearn": SKLEARN_AVAILABLE,
                "webrtcvad": WEBRTC_AVAILABLE
            }
        }
    
    def get_installation_guide(self) -> Dict[str, Any]:
        """설치 가이드 제공"""
        
        missing_packages = []
        
        if not LIBROSA_AVAILABLE:
            missing_packages.append({
                "package": "librosa",
                "command": "pip install librosa",
                "purpose": "오디오 신호 처리 및 특성 추출"
            })
        
        if not SKLEARN_AVAILABLE:
            missing_packages.append({
                "package": "scikit-learn",
                "command": "pip install scikit-learn",
                "purpose": "화자 특성 클러스터링"
            })
        
        if not WEBRTC_AVAILABLE:
            missing_packages.append({
                "package": "webrtcvad",
                "command": "pip install webrtcvad",
                "purpose": "음성 활동 감지 (VAD)"
            })
        
        return {
            "available": LIBROSA_AVAILABLE and SKLEARN_AVAILABLE,
            "missing_packages": missing_packages,
            "install_all": "pip install librosa scikit-learn webrtcvad",
            "realtime_tracking": self.realtime_tracking_available,
            "speaker_identification": self.speaker_identification_available,
            "recommended_setup": [
                "1. pip install librosa scikit-learn webrtcvad",
                "2. 실시간 추적 시스템 활성화 확인",
                "3. 화자 식별 시스템 연동 확인",
                "4. 테스트 오디오로 기능 검증"
            ]
        }


# 전역 인스턴스 생성
global_speaker_diarization = RealtimeSpeakerDiarization()

def analyze_speakers_realtime(audio_file: str, transcript: str = "", progress_container=None) -> Dict[str, Any]:
    """실시간 추적 화자 분리 분석 함수"""
    return global_speaker_diarization.analyze_speakers_in_audio(audio_file, transcript, progress_container)