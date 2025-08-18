#!/usr/bin/env python3
"""
멀티모달 화자 분리 시스템
- 음성 특징 (29차원) + 시각적 특징 (얼굴 인식) 융합
- The Rise of the Eco-Friendly Luxury Consumer 3명 발표자 분석 특화
"""

import os
import sys
import cv2
import json
import time
import librosa
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import subprocess
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Whisper STT
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

class MultimodalSpeakerDiarization:
    """멀티모달 화자 분리 시스템"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 음성 분석 설정
        self.sample_rate = 16000
        self.hop_length = 512
        self.n_mfcc = 13
        
        # 영상 분석 설정
        self.face_cascade = None
        self.frame_sample_rate = 1  # 1초마다 1프레임 샘플링
        
        # 화자 분리 설정
        self.min_speakers = 2
        self.max_speakers = 6
        self.min_segment_duration = 2.0  # 최소 세그먼트 길이 (초)
        
        # 모델 초기화
        self._init_models()
        
    def _setup_logging(self):
        """로깅 설정"""
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def _init_models(self):
        """모델 초기화"""
        print("Initializing Multimodal Speaker Diarization System...")
        
        # Load OpenCV face recognition model
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("Face recognition model loaded successfully")
        except Exception as e:
            print(f"Failed to load face recognition model: {e}")
        
        # Load Whisper model
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base")
                print("Whisper STT model loaded successfully")
            except Exception as e:
                print(f"Failed to load Whisper model: {e}")
                self.whisper_model = None
        else:
            print("Whisper library not available")
            self.whisper_model = None
    
    def analyze_video_multimodal(self, video_path: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Multimodal video analysis"""
        print(f"Starting multimodal video analysis: {os.path.basename(video_path)}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        start_time = time.time()
        
        # 1. Extract video information
        video_info = self._extract_video_info(video_path)
        print(f"Video info: {video_info['duration']:.1f}s, {video_info['fps']:.1f}fps")
        
        # 2. Extract and analyze audio
        audio_path = self._extract_audio(video_path)
        audio_analysis = self._analyze_audio_multimodal(audio_path)
        
        # 3. Visual speaker analysis
        visual_analysis = self._analyze_visual_speakers(video_path)
        
        # 4. Multimodal fusion
        fused_analysis = self._fuse_multimodal_features(audio_analysis, visual_analysis)
        
        # 5. Context-based refinement
        if context:
            fused_analysis = self._apply_context(fused_analysis, context)
        
        processing_time = time.time() - start_time
        
        result = {
            "video_info": video_info,
            "audio_analysis": audio_analysis,
            "visual_analysis": visual_analysis,
            "multimodal_result": fused_analysis,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"Multimodal analysis completed ({processing_time:.1f}s)")
        return result
    
    def _extract_video_info(self, video_path: str) -> Dict[str, Any]:
        """영상 정보 추출"""
        cap = cv2.VideoCapture(video_path)
        
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": 0,
            "file_size": os.path.getsize(video_path) / (1024 * 1024)  # MB
        }
        
        if info["fps"] > 0:
            info["duration"] = info["frame_count"] / info["fps"]
        
        cap.release()
        return info
    
    def _extract_audio(self, video_path: str) -> str:
        """영상에서 음성 추출"""
        audio_path = video_path.replace('.MOV', '_audio.wav').replace('.mp4', '_audio.wav')
        
        try:
            # FFmpeg를 사용한 음성 추출
            cmd = [
                'ffmpeg', '-i', video_path,
                '-ar', str(self.sample_rate),
                '-ac', '1',  # 모노
                '-y',  # 덮어쓰기
                audio_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"Audio extraction completed: {os.path.basename(audio_path)}")
            return audio_path
            
        except subprocess.CalledProcessError as e:
            print(f"Audio extraction failed: {e}")
            return None
    
    def _analyze_audio_multimodal(self, audio_path: str) -> Dict[str, Any]:
        """멀티모달 음성 분석"""
        if not audio_path or not os.path.exists(audio_path):
            return {"error": "음성 파일 없음"}
        
        print("Extracting 29-dimensional voice features...")
        
        # 1. Load audio data
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(y) / sr
        
        # 2. STT processing
        transcription = self._transcribe_audio(audio_path)
        
        # 3. Extract 29-dimensional voice features
        voice_features = self._extract_voice_features_29d(y, sr)
        
        # 4. Speaker diarization
        speaker_segments = self._identify_speakers_multimodal(voice_features, transcription)
        
        return {
            "duration": duration,
            "transcription": transcription,
            "voice_features": voice_features,
            "speaker_segments": speaker_segments,
            "feature_dimensions": len(voice_features[0]) if voice_features else 0
        }
    
    def _extract_voice_features_29d(self, y: np.ndarray, sr: int) -> List[List[float]]:
        """29차원 음성 특징 벡터 추출"""
        
        # 세그먼트 단위로 분할 (2초씩)
        segment_length = int(2 * sr)  # 2초
        hop_length = int(1 * sr)      # 1초 hop
        
        features = []
        
        for start in range(0, len(y) - segment_length, hop_length):
            end = start + segment_length
            segment = y[start:end]
            
            if len(segment) < segment_length:
                # 패딩
                segment = np.pad(segment, (0, segment_length - len(segment)))
            
            # 29차원 특징 추출
            feature_vector = []
            
            # 1-13: MFCC (13차원)
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=self.n_mfcc)
            feature_vector.extend(np.mean(mfccs, axis=1))
            
            # 14-15: 스펙트럴 센트로이드 (평균, 표준편차)
            centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
            feature_vector.extend([np.mean(centroid), np.std(centroid)])
            
            # 16-17: 스펙트럴 롤오프 (평균, 표준편차)
            rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
            feature_vector.extend([np.mean(rolloff), np.std(rolloff)])
            
            # 18-19: 제로 크로싱 레이트 (평균, 표준편차)
            zcr = librosa.feature.zero_crossing_rate(segment)
            feature_vector.extend([np.mean(zcr), np.std(zcr)])
            
            # 20-21: RMS 에너지 (평균, 표준편차)
            rms = librosa.feature.rms(y=segment)
            feature_vector.extend([np.mean(rms), np.std(rms)])
            
            # 22-23: 스펙트럴 대역폭 (평균, 표준편차)
            bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
            feature_vector.extend([np.mean(bandwidth), np.std(bandwidth)])
            
            # 24-25: 피치 (평균, 표준편차)
            pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                feature_vector.extend([np.mean(pitch_values), np.std(pitch_values)])
            else:
                feature_vector.extend([0, 0])
            
            # 26-29: 추가 스펙트럴 특징들
            contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)
            feature_vector.extend([
                np.mean(contrast),
                np.std(contrast),
                np.mean(segment**2),  # 파워
                len([x for x in segment if abs(x) > 0.01]) / len(segment)  # 활성도
            ])
            
            features.append(feature_vector)
        
        return features
    
    def _transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """음성을 텍스트로 변환"""
        if not self.whisper_model:
            return {"segments": [], "text": ""}
        
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result
        except Exception as e:
            print(f"⚠️ STT 처리 실패: {e}")
            return {"segments": [], "text": ""}
    
    def _analyze_visual_speakers(self, video_path: str) -> Dict[str, Any]:
        """시각적 화자 분석"""
        print("Visual speaker analysis in progress...")
        
        if not self.face_cascade:
            return {"error": "Face recognition model not available"}
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        face_detections = []
        frame_samples = []
        
        # 1초마다 프레임 샘플링
        sample_interval = int(fps / self.frame_sample_rate)
        
        for frame_idx in range(0, frame_count, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            timestamp = frame_idx / fps
            
            # 얼굴 감지
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            
            face_info = {
                "timestamp": timestamp,
                "frame_idx": frame_idx,
                "face_count": len(faces),
                "faces": []
            }
            
            for (x, y, w, h) in faces:
                face_data = {
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "center": [int(x + w/2), int(y + h/2)],
                    "area": int(w * h),
                    "confidence": 1.0  # 기본값
                }
                face_info["faces"].append(face_data)
            
            face_detections.append(face_info)
            
            # 대표 프레임 저장 (매 30초마다)
            if int(timestamp) % 30 == 0:
                frame_samples.append({
                    "timestamp": timestamp,
                    "face_count": len(faces),
                    "frame_path": f"frame_{int(timestamp)}.jpg"
                })
        
        cap.release()
        
        # 화자 수 추정 (얼굴 감지 기반)
        face_counts = [detection["face_count"] for detection in face_detections]
        estimated_speakers = max(face_counts) if face_counts else 1
        
        # 화자 전환 감지
        speaker_transitions = self._detect_speaker_transitions(face_detections)
        
        return {
            "estimated_speakers": estimated_speakers,
            "face_detections": face_detections,
            "speaker_transitions": speaker_transitions,
            "frame_samples": frame_samples,
            "total_frames_analyzed": len(face_detections)
        }
    
    def _detect_speaker_transitions(self, face_detections: List[Dict]) -> List[Dict]:
        """화자 전환 감지"""
        transitions = []
        prev_count = 0
        
        for detection in face_detections:
            current_count = detection["face_count"]
            
            # 얼굴 수 변화 감지
            if current_count != prev_count and current_count > 0:
                transitions.append({
                    "timestamp": detection["timestamp"],
                    "from_speakers": prev_count,
                    "to_speakers": current_count,
                    "transition_type": "visual_change"
                })
            
            prev_count = current_count
        
        return transitions
    
    def _identify_speakers_multimodal(self, voice_features: List, transcription: Dict) -> Dict[str, Any]:
        """멀티모달 화자 분리"""
        if not voice_features or len(voice_features) < 2:
            return {"speakers": 1, "segments": [], "quality_score": 0}
        
        try:
            # 특징 벡터 정규화
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(voice_features)
            
            # PCA 차원 축소
            n_components = min(10, len(voice_features) - 1, len(voice_features[0]))
            if n_components < 2:
                return {"speakers": 1, "segments": [], "quality_score": 0}
            
            pca = PCA(n_components=n_components)
            features_pca = pca.fit_transform(features_scaled)
            
            # 최적 클러스터 수 찾기
            best_score = -1
            best_n_speakers = 1
            
            for n_speakers in range(self.min_speakers, min(self.max_speakers + 1, len(voice_features))):
                kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features_pca)
                
                if len(set(labels)) > 1:
                    score = silhouette_score(features_pca, labels)
                    if score > best_score:
                        best_score = score
                        best_n_speakers = n_speakers
            
            # 최적 클러스터링 수행
            kmeans = KMeans(n_clusters=best_n_speakers, random_state=42, n_init=10)
            speaker_labels = kmeans.fit_predict(features_pca)
            
            # 세그먼트 생성
            segments = []
            for i, label in enumerate(speaker_labels):
                segments.append({
                    "start": i * 1.0,  # 1초 간격
                    "end": (i + 1) * 1.0,
                    "speaker": f"Speaker_{label + 1}",
                    "confidence": 0.8
                })
            
            return {
                "speakers": best_n_speakers,
                "segments": segments,
                "quality_score": best_score,
                "method": "voice_based_29d"
            }
            
        except Exception as e:
            print(f"⚠️ 음성 기반 화자 분리 실패: {e}")
            return {"speakers": 1, "segments": [], "quality_score": 0}
    
    def _fuse_multimodal_features(self, audio_analysis: Dict, visual_analysis: Dict) -> Dict[str, Any]:
        """멀티모달 특징 융합"""
        print("Fusing multimodal features...")
        
        # 음성 기반 화자 수
        audio_speakers = audio_analysis.get("speaker_segments", {}).get("speakers", 1)
        
        # 시각적 기반 화자 수
        visual_speakers = visual_analysis.get("estimated_speakers", 1)
        
        # 융합 결정 로직
        if visual_speakers >= 3 and audio_speakers < 3:
            # 시각적으로 3명 이상 감지되면 시각적 결과 우선
            final_speakers = visual_speakers
            confidence = "visual_priority"
        elif audio_speakers >= 2 and visual_speakers == 1:
            # 음성으로 2명 이상 감지되고 시각적으로 1명이면 음성 결과 우선
            final_speakers = audio_speakers
            confidence = "audio_priority"
        else:
            # 평균값 사용
            final_speakers = max(audio_speakers, visual_speakers)
            confidence = "consensus"
        
        # 시각적 전환점과 음성 세그먼트 매칭
        transitions = visual_analysis.get("speaker_transitions", [])
        audio_segments = audio_analysis.get("speaker_segments", {}).get("segments", [])
        
        # 정제된 세그먼트 생성
        refined_segments = self._create_refined_segments(audio_segments, transitions, final_speakers)
        
        return {
            "final_speaker_count": final_speakers,
            "confidence_method": confidence,
            "audio_speakers": audio_speakers,
            "visual_speakers": visual_speakers,
            "refined_segments": refined_segments,
            "visual_transitions": transitions,
            "fusion_quality": self._calculate_fusion_quality(audio_analysis, visual_analysis)
        }
    
    def _create_refined_segments(self, audio_segments: List, visual_transitions: List, speaker_count: int) -> List[Dict]:
        """정제된 세그먼트 생성"""
        refined = []
        
        # 시각적 전환점을 기준으로 세그먼트 조정
        for segment in audio_segments:
            # 해당 시간대의 시각적 전환 확인
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", 0)
            
            # 시각적 전환점과 매칭
            matching_transition = None
            for transition in visual_transitions:
                if segment_start <= transition["timestamp"] <= segment_end:
                    matching_transition = transition
                    break
            
            refined_segment = {
                "start": segment_start,
                "end": segment_end,
                "speaker": segment.get("speaker", "Speaker_1"),
                "confidence": segment.get("confidence", 0.5),
                "modality": "multimodal",
                "visual_support": matching_transition is not None
            }
            
            if matching_transition:
                refined_segment["visual_transition"] = matching_transition
                refined_segment["confidence"] = min(0.9, refined_segment["confidence"] + 0.2)
            
            refined.append(refined_segment)
        
        return refined
    
    def _calculate_fusion_quality(self, audio_analysis: Dict, visual_analysis: Dict) -> float:
        """융합 품질 점수 계산"""
        score = 0.5  # 기본 점수
        
        # 음성 품질 점수
        audio_quality = audio_analysis.get("speaker_segments", {}).get("quality_score", 0)
        if audio_quality > 0:
            score += audio_quality * 0.3
        
        # 시각적 감지 품질
        visual_detections = len(visual_analysis.get("face_detections", []))
        if visual_detections > 0:
            score += min(0.3, visual_detections / 100)
        
        # 전환점 일치도
        transitions = len(visual_analysis.get("speaker_transitions", []))
        if transitions > 0:
            score += min(0.2, transitions / 10)
        
        return min(1.0, score)
    
    def _apply_context(self, analysis: Dict, context: Dict) -> Dict[str, Any]:
        """컨텍스트 정보 적용"""
        if not context:
            return analysis
        
        # 컨퍼런스명에서 화자 수 힌트
        conference_name = context.get("conference_name", "")
        if "3명" in conference_name or "three" in conference_name.lower():
            if analysis["final_speaker_count"] != 3:
                analysis["context_adjusted"] = True
                analysis["original_speaker_count"] = analysis["final_speaker_count"]
                analysis["final_speaker_count"] = 3
                analysis["confidence_method"] += "_context_adjusted"
        
        # 발표자 정보 적용
        speaker_info = context.get("speaker_info", {})
        if speaker_info:
            analysis["expected_speakers"] = speaker_info
        
        return analysis
    
    def save_analysis_result(self, result: Dict[str, Any], output_path: str = None) -> str:
        """분석 결과 저장"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"multimodal_analysis_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"📊 분석 결과 저장: {output_path}")
        return output_path

def main():
    """Main execution function"""
    print("=== Multimodal Speaker Diarization System ===")
    
    # System initialization
    analyzer = MultimodalSpeakerDiarization()
    
    # Test video path
    video_path = "C:/Users/PC_58410/solomond-ai-system/user_files/JGA2025_D1/IMG_0032.MOV"
    
    # Context information
    context = {
        "conference_name": "The Rise of the Eco-Friendly Luxury Consumer",
        "expected_speakers": 3,
        "speaker_info": {
            "total_speakers": 3,
            "presentation_type": "sequential"
        }
    }
    
    if os.path.exists(video_path):
        try:
            # Execute multimodal analysis
            result = analyzer.analyze_video_multimodal(video_path, context)
            
            # Print results
            print("\n" + "="*50)
            print("MULTIMODAL SPEAKER DIARIZATION RESULTS")
            print("="*50)
            
            multimodal = result["multimodal_result"]
            print(f"Final speaker count: {multimodal['final_speaker_count']} speakers")
            print(f"Confidence method: {multimodal['confidence_method']}")
            print(f"Audio-based: {multimodal['audio_speakers']} speakers")
            print(f"Visual-based: {multimodal['visual_speakers']} speakers")
            print(f"Fusion quality: {multimodal['fusion_quality']:.3f}")
            
            # Save results
            output_file = analyzer.save_analysis_result(result)
            print(f"\nAnalysis completed! Result file: {output_file}")
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Video file not found: {video_path}")

if __name__ == "__main__":
    main()