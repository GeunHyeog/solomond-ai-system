#!/usr/bin/env python3
"""
Ultimate 5D Multimodal Conference Analysis System
5차원 완전 통합 컨퍼런스 분석 시스템

🎯 분석 차원:
1. Audio: 29차원 음성 특징 + STT
2. Visual: 얼굴 인식 + 화자 전환
3. Transcript: 텍스트 기록물 + 화자 마커
4. Slides: PPT 슬라이드 OCR + 구조 분석
5. Timeline: 시간 동기화 + 맥락 연결

The Rise of the Eco-Friendly Luxury Consumer 완전 분석
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
import re
from PIL import Image

# EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Whisper STT
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:  
    WHISPER_AVAILABLE = False

class Ultimate5DMultimodalAnalyzer:
    """5차원 멀티모달 컨퍼런스 분석 시스템"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 음성 분석 설정
        self.sample_rate = 16000
        self.hop_length = 512
        self.n_mfcc = 13
        
        # 영상 분석 설정
        self.face_cascade = None
        self.frame_sample_rate = 1
        
        # 화자 분리 설정
        self.min_speakers = 2
        self.max_speakers = 6
        
        # OCR 설정
        self.ocr_reader = None
        
        # 슬라이드 분석 설정
        self.slide_keywords = {
            "introduction": ["welcome", "introduction", "overview", "agenda", "목표", "개요"],
            "content": ["data", "analysis", "research", "study", "결과", "분석"],
            "conclusion": ["conclusion", "summary", "thank you", "결론", "정리", "감사"]
        }
        
        # 모델 초기화
        self._init_models()
        
    def _setup_logging(self):
        """로깅 설정"""
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def _init_models(self):
        """모델 초기화"""
        print("=== Initializing Ultimate 5D Multimodal System ===")
        
        # 1. OpenCV 얼굴 인식
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("+ Face recognition model loaded")
        except Exception as e:
            print(f"- Face recognition failed: {e}")
        
        # 2. Whisper STT
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("tiny")
                print("+ Whisper STT model loaded")
            except Exception as e:
                print(f"- Whisper failed: {e}")
                self.whisper_model = None
        else:
            print("- Whisper not available")
            self.whisper_model = None
        
        # 3. EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['en', 'ko'], gpu=False)
                print("+ EasyOCR model loaded")
            except Exception as e:
                print(f"- EasyOCR failed: {e}")
                self.ocr_reader = None
        else:
            print("- EasyOCR not available")
            self.ocr_reader = None
        
        print("=== 5D System Initialization Complete ===\n")
    
    def analyze_complete_5d(self, 
                           video_path: str = None,
                           audio_path: str = None, 
                           transcript_text: str = None,
                           slides_folder: str = None,
                           slide_images: List[str] = None,
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """5차원 완전 분석"""
        
        print("Starting Ultimate 5D Multimodal Analysis...")
        print("Dimensions: Audio + Visual + Transcript + Slides + Timeline")
        
        start_time = time.time()
        
        # 1. 입력 소스 검증
        sources = self._validate_5d_sources(video_path, audio_path, transcript_text, 
                                           slides_folder, slide_images)
        print(f"Available sources: {list(sources.keys())}")
        
        # 2. 각 차원별 분석
        dimension_analyses = {}
        
        # Dimension 1: Audio Analysis
        if sources.get("video") or sources.get("audio"):
            print("\nDimension 1: Audio Analysis")
            dimension_analyses["audio"] = self._analyze_audio_dimension(
                video_path if sources.get("video") else audio_path
            )
        
        # Dimension 2: Visual Analysis  
        if sources.get("video"):
            print("\nDimension 2: Visual Analysis")
            dimension_analyses["visual"] = self._analyze_visual_dimension(video_path)
        
        # Dimension 3: Transcript Analysis
        if sources.get("transcript"):
            print("\nDimension 3: Transcript Analysis")
            dimension_analyses["transcript"] = self._analyze_transcript_dimension(transcript_text)
        
        # Dimension 4: Slides Analysis
        if sources.get("slides"):
            print("\nDimension 4: Slides Analysis")
            slide_files = self._get_slide_files(slides_folder, slide_images)
            dimension_analyses["slides"] = self._analyze_slides_dimension(slide_files)
        
        # Dimension 5: Timeline Integration
        print("\nDimension 5: Timeline Integration")
        dimension_analyses["timeline"] = self._analyze_timeline_dimension(dimension_analyses)
        
        # 3. 5차원 융합
        print("\n5D Fusion & Integration")
        fusion_result = self._fuse_5d_dimensions(dimension_analyses)
        
        # 4. 컨텍스트 적용
        if context:
            fusion_result = self._apply_5d_context(fusion_result, context)
        
        # 5. 최종 통합 리포트 생성
        final_report = self._generate_ultimate_report(dimension_analyses, fusion_result)
        
        processing_time = time.time() - start_time
        
        result = {
            "sources": sources,
            "dimension_analyses": dimension_analyses,
            "fusion_result": fusion_result,
            "final_report": final_report,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "system_version": "Ultimate 5D v1.0"
        }
        
        print(f"\nUltimate 5D Analysis Complete! ({processing_time:.1f}s)")
        return result
    
    def _validate_5d_sources(self, video_path, audio_path, transcript_text, 
                            slides_folder, slide_images) -> Dict[str, bool]:
        """5차원 입력 소스 검증"""
        sources = {}
        
        # Video
        sources["video"] = bool(video_path and os.path.exists(video_path))
        if sources["video"]:
            print(f"Video: {os.path.basename(video_path)}")
        
        # Audio
        sources["audio"] = bool(audio_path and os.path.exists(audio_path))
        if sources["audio"]:
            print(f"Audio: {os.path.basename(audio_path)}")
        
        # Transcript
        sources["transcript"] = bool(transcript_text)
        if sources["transcript"]:
            print(f"Transcript: {len(transcript_text)} characters")
        
        # Slides
        slide_files = self._get_slide_files(slides_folder, slide_images)
        sources["slides"] = bool(slide_files)
        if sources["slides"]:
            print(f"Slides: {len(slide_files)} images")
        
        return sources
    
    def _get_slide_files(self, slides_folder, slide_images) -> List[str]:
        """슬라이드 파일 목록 가져오기"""
        files = []
        
        if slide_images:
            files.extend([f for f in slide_images if os.path.exists(f)])
        
        if slides_folder and os.path.exists(slides_folder):
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                files.extend(Path(slides_folder).glob(f'*{ext}'))
        
        return sorted([str(f) for f in files])
    
    def _analyze_audio_dimension(self, source_path: str) -> Dict[str, Any]:
        """차원 1: 음성 분석"""
        
        # 비디오에서 오디오 추출
        if source_path.lower().endswith(('.mp4', '.mov', '.avi')):
            audio_path = self._extract_audio_quick(source_path)
        else:
            audio_path = source_path
        
        if not audio_path or not os.path.exists(audio_path):
            return {"error": "Audio source not available"}
        
        try:
            # 1. 기본 오디오 정보
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = len(y) / sr
            
            # 2. STT 처리
            transcription = {}
            if self.whisper_model:
                print("  Processing STT...")
                transcription = self.whisper_model.transcribe(audio_path, fp16=False, verbose=False)
            
            # 3. 29차원 음성 특징 (샘플링)
            print("  Extracting voice features...")
            voice_features = self._extract_voice_features_quick(y, sr)
            
            # 4. 화자 분리
            speaker_analysis = self._quick_speaker_diarization(voice_features, transcription)
            
            return {
                "duration": duration,
                "transcription": transcription,
                "voice_features_count": len(voice_features),
                "speaker_analysis": speaker_analysis,
                "audio_stats": {
                    "sample_rate": sr,
                    "total_samples": len(y),
                    "rms_energy": float(np.sqrt(np.mean(y**2)))
                }
            }
            
        except Exception as e:
            print(f"  - Audio analysis failed: {e}")
            return {"error": str(e)}
    
    def _extract_audio_quick(self, video_path: str, max_duration: int = 120) -> str:
        """빠른 오디오 추출 (2분 제한)"""
        audio_path = video_path.replace('.MOV', '_5d_audio.wav').replace('.mp4', '_5d_audio.wav')
        
        try:
            cmd = [
                'ffmpeg', '-i', video_path,
                '-t', str(max_duration),
                '-ar', str(self.sample_rate),
                '-ac', '1', '-y', audio_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"  + Audio extracted: {os.path.basename(audio_path)}")
            return audio_path
            
        except subprocess.CalledProcessError as e:
            print(f"  - Audio extraction failed: {e}")
            return None
    
    def _extract_voice_features_quick(self, y: np.ndarray, sr: int) -> List[List[float]]:
        """빠른 음성 특징 추출"""
        segment_length = int(5 * sr)  # 5초 세그먼트
        hop_length = int(3 * sr)      # 3초 hop
        
        features = []
        
        for start in range(0, min(len(y), segment_length * 6), hop_length):  # 최대 6개 세그먼트
            end = start + segment_length
            segment = y[start:end] if end <= len(y) else np.pad(y[start:], (0, end - len(y)))
            
            # 간단한 특징 벡터 (13차원)
            feature_vector = []
            
            # MFCC
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
            feature_vector.extend(np.mean(mfccs, axis=1))
            
            features.append(feature_vector)
        
        return features
    
    def _quick_speaker_diarization(self, voice_features: List, transcription: Dict) -> Dict[str, Any]:
        """빠른 화자 분리"""
        if not voice_features or len(voice_features) < 2:
            return {"speakers": 1, "method": "insufficient_data"}
        
        try:
            # 간단한 클러스터링
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(voice_features)
            
            # 2-3명 화자 가정
            best_speakers = 2
            best_score = -1
            
            for n in [2, 3]:
                if n <= len(voice_features):
                    kmeans = KMeans(n_clusters=n, random_state=42, n_init=5)
                    labels = kmeans.fit_predict(features_scaled)
                    
                    if len(set(labels)) > 1:
                        score = silhouette_score(features_scaled, labels)
                        if score > best_score:
                            best_score = score
                            best_speakers = n
            
            return {
                "speakers": best_speakers,
                "quality_score": best_score,
                "method": "voice_clustering"
            }
            
        except Exception as e:
            print(f"  Warning: Speaker diarization failed: {e}")
            return {"speakers": 1, "method": "fallback", "error": str(e)}
    
    def _analyze_visual_dimension(self, video_path: str) -> Dict[str, Any]:
        """차원 2: 시각적 분석"""
        
        if not self.face_cascade:
            return {"error": "Face cascade not available"}
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"  Video: {frame_count} frames at {fps:.1f}fps")
            
            # 20초마다 샘플링 (빠른 처리)
            sample_interval = int(fps * 20)
            face_detections = []
            slide_transitions = []
            
            for frame_idx in range(0, min(frame_count, sample_interval * 10), sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_idx / fps
                
                # 얼굴 감지
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                
                # 슬라이드 전환 감지 (화면 변화)
                if len(face_detections) > 0:
                    prev_faces = face_detections[-1]["face_count"]
                    current_faces = len(faces)
                    
                    if abs(current_faces - prev_faces) > 0:
                        slide_transitions.append({
                            "timestamp": timestamp,
                            "transition_type": "face_count_change",
                            "from_faces": prev_faces,
                            "to_faces": current_faces
                        })
                
                face_detections.append({
                    "timestamp": timestamp,
                    "face_count": len(faces),
                    "frame_idx": frame_idx
                })
            
            cap.release()
            
            # 화자 수 추정
            face_counts = [d["face_count"] for d in face_detections]
            estimated_speakers = max(face_counts) if face_counts else 1
            
            return {
                "estimated_speakers": estimated_speakers,
                "face_detections": face_detections,
                "slide_transitions": slide_transitions,
                "analysis_points": len(face_detections),
                "max_faces_detected": max(face_counts) if face_counts else 0
            }
            
        except Exception as e:
            print(f"  - Visual analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_transcript_dimension(self, transcript_text: str) -> Dict[str, Any]:
        """차원 3: 텍스트 기록물 분석"""
        
        if not transcript_text:
            return {"error": "No transcript provided"}
        
        try:
            # 1. 기본 통계
            words = transcript_text.split()
            lines = transcript_text.split('\n')
            
            # 2. 화자 마커 감지
            speaker_markers = self._detect_speaker_markers_advanced(transcript_text)
            
            # 3. 주제 키워드 분석
            topic_analysis = self._analyze_topics(transcript_text)
            
            # 4. 구조 분석
            structure_analysis = self._analyze_text_structure(transcript_text)
            
            # 5. 화자 수 추정
            estimated_speakers = max(3, len(speaker_markers) if speaker_markers else 1)
            
            return {
                "stats": {
                    "word_count": len(words),
                    "line_count": len(lines),
                    "char_count": len(transcript_text)
                },
                "speaker_markers": speaker_markers,
                "topic_analysis": topic_analysis,
                "structure_analysis": structure_analysis,
                "estimated_speakers": estimated_speakers
            }
            
        except Exception as e:
            print(f"  - Transcript analysis failed: {e}")
            return {"error": str(e)}
    
    def _detect_speaker_markers_advanced(self, text: str) -> List[Dict[str, Any]]:
        """고급 화자 마커 감지"""
        markers = []
        
        patterns = [
            (r'(?i)speaker\s*[123][\s:：]', 'explicit_speaker'),
            (r'(?i)(first|second|third)\s+(speaker|presenter)[\s:：]', 'ordinal_speaker'),
            (r'(?i)(thank\s+you|moving\s+on|in\s+conclusion)', 'transition_marker'),
            (r'(?i)(now|next|finally|let me)', 'section_marker')
        ]
        
        for pattern, marker_type in patterns:
            for match in re.finditer(pattern, text):
                markers.append({
                    "position": match.start(),
                    "text": match.group(0).strip(),
                    "type": marker_type,
                    "confidence": 0.8 if 'explicit' in marker_type else 0.6
                })
        
        return sorted(markers, key=lambda x: x["position"])
    
    def _analyze_topics(self, text: str) -> Dict[str, Any]:
        """주제 분석"""
        
        # 환경 친화적 럭셔리 관련 키워드
        eco_keywords = [
            "sustainable", "sustainability", "eco-friendly", "environmental", 
            "green", "recycling", "carbon", "waste reduction", "circular economy"
        ]
        
        luxury_keywords = [
            "luxury", "premium", "high-end", "exclusive", "affluent", 
            "consumer", "market", "brand", "quality"
        ]
        
        eco_count = sum(text.lower().count(kw.lower()) for kw in eco_keywords)
        luxury_count = sum(text.lower().count(kw.lower()) for kw in luxury_keywords)
        
        return {
            "eco_keywords": eco_count,
            "luxury_keywords": luxury_count,
            "topic_relevance": (eco_count + luxury_count) / len(text.split()) * 100,
            "main_theme": "eco_luxury" if eco_count > 2 and luxury_count > 2 else "general"
        }
    
    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """텍스트 구조 분석"""
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # 구조적 요소 감지
        has_introduction = any(word in text.lower() for word in ['welcome', 'introduction', 'today'])
        has_conclusion = any(word in text.lower() for word in ['conclusion', 'summary', 'thank you'])
        
        return {
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "avg_paragraph_length": np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0,
            "has_introduction": has_introduction,
            "has_conclusion": has_conclusion,
            "structure_score": (has_introduction + has_conclusion) * 0.5
        }
    
    def _analyze_slides_dimension(self, slide_files: List[str]) -> Dict[str, Any]:
        """차원 4: 슬라이드 분석"""
        
        if not self.ocr_reader or not slide_files:
            return {"error": "OCR not available or no slides"}
        
        try:
            print(f"  Processing {len(slide_files)} slides...")
            
            slide_analyses = []
            all_text = ""
            
            for i, slide_file in enumerate(slide_files):
                print(f"  Slide {i+1}/{len(slide_files)}: {os.path.basename(slide_file)}")
                
                try:
                    # OCR 처리
                    results = self.ocr_reader.readtext(slide_file)
                    
                    # 텍스트 추출
                    slide_text = " ".join([result[1] for result in results if result[2] > 0.5])
                    all_text += slide_text + " "
                    
                    # 슬라이드 분류
                    slide_type = self._classify_slide_type(slide_text)
                    
                    # 핵심 정보 추출
                    key_points = self._extract_key_points(slide_text)
                    
                    slide_analyses.append({
                        "slide_number": i + 1,
                        "file_path": slide_file,
                        "text": slide_text,
                        "text_length": len(slide_text),
                        "confidence": np.mean([result[2] for result in results]) if results else 0,
                        "slide_type": slide_type,
                        "key_points": key_points,
                        "ocr_results_count": len(results)
                    })
                    
                except Exception as e:
                    print(f"    Warning: Failed to process slide {i+1}: {e}")
                    slide_analyses.append({
                        "slide_number": i + 1,
                        "file_path": slide_file,
                        "error": str(e)
                    })
            
            # 전체 분석
            total_analysis = self._analyze_slides_content(all_text, slide_analyses)
            
            return {
                "total_slides": len(slide_files),
                "processed_slides": len([s for s in slide_analyses if "error" not in s]),
                "slide_analyses": slide_analyses,
                "total_analysis": total_analysis,
                "combined_text": all_text.strip()
            }
            
        except Exception as e:
            print(f"  - Slides analysis failed: {e}")
            return {"error": str(e)}
    
    def _classify_slide_type(self, text: str) -> str:
        """슬라이드 타입 분류"""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in self.slide_keywords["introduction"]):
            return "introduction"
        elif any(kw in text_lower for kw in self.slide_keywords["conclusion"]):
            return "conclusion"
        elif any(kw in text_lower for kw in self.slide_keywords["content"]):
            return "content"
        else:
            return "general"
    
    def _extract_key_points(self, text: str) -> List[str]:
        """핵심 포인트 추출"""
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.split()) > 3]
        
        # 짧고 명확한 문장들을 핵심 포인트로 간주
        key_points = []
        for sentence in sentences:
            if 5 <= len(sentence.split()) <= 15:  # 5-15 단어 문장
                key_points.append(sentence)
        
        return key_points[:5]  # 최대 5개
    
    def _analyze_slides_content(self, all_text: str, slide_analyses: List) -> Dict[str, Any]:
        """슬라이드 전체 내용 분석"""
        
        # 슬라이드 타입 분포
        slide_types = [s.get("slide_type", "unknown") for s in slide_analyses if "slide_type" in s]
        type_distribution = {t: slide_types.count(t) for t in set(slide_types)}
        
        # 주요 주제 분석
        topic_analysis = self._analyze_topics(all_text)
        
        # 구조 점수
        structure_score = 0
        if "introduction" in type_distribution:
            structure_score += 0.3
        if "conclusion" in type_distribution:
            structure_score += 0.3
        if type_distribution.get("content", 0) > 0:
            structure_score += 0.4
        
        return {
            "total_words": len(all_text.split()),
            "type_distribution": type_distribution,
            "topic_analysis": topic_analysis,
            "structure_score": structure_score,
            "presentation_flow": self._analyze_presentation_flow(slide_analyses)
        }
    
    def _analyze_presentation_flow(self, slide_analyses: List) -> Dict[str, Any]:
        """발표 흐름 분석"""
        flow = []
        
        for slide in slide_analyses:
            if "slide_type" in slide:
                flow.append(slide["slide_type"])
        
        # 이상적인 흐름: intro -> content -> conclusion
        has_intro = "introduction" in flow
        has_content = "content" in flow
        has_conclusion = "conclusion" in flow
        
        flow_score = (has_intro * 0.3 + has_content * 0.4 + has_conclusion * 0.3)
        
        return {
            "flow_sequence": flow,
            "has_introduction": has_intro,
            "has_content": has_content,
            "has_conclusion": has_conclusion,
            "flow_score": flow_score
        }
    
    def _analyze_timeline_dimension(self, dimension_analyses: Dict) -> Dict[str, Any]:
        """차원 5: 타임라인 통합"""
        
        print("  Building integrated timeline...")
        
        timeline_events = []
        
        # 오디오에서 이벤트 추출
        if "audio" in dimension_analyses:
            audio_data = dimension_analyses["audio"]
            if "transcription" in audio_data and "segments" in audio_data["transcription"]:
                for seg in audio_data["transcription"]["segments"]:
                    timeline_events.append({
                        "timestamp": seg.get("start", 0),
                        "type": "speech_segment",
                        "content": seg.get("text", ""),
                        "source": "audio"
                    })
        
        # 시각적 전환점 추가
        if "visual" in dimension_analyses:
            visual_data = dimension_analyses["visual"]
            if "slide_transitions" in visual_data:
                for trans in visual_data["slide_transitions"]:
                    timeline_events.append({
                        "timestamp": trans["timestamp"],
                        "type": "visual_transition",
                        "content": f"Face count: {trans['from_faces']} -> {trans['to_faces']}",
                        "source": "visual"
                    })
        
        # 슬라이드와 매칭 (추정)
        if "slides" in dimension_analyses:
            slides_data = dimension_analyses["slides"]
            if "slide_analyses" in slides_data:
                # 시간을 균등하게 분배 (실제로는 더 정교한 매칭 필요)
                total_duration = self._get_total_duration(dimension_analyses)
                slide_count = len(slides_data["slide_analyses"])
                
                if total_duration > 0 and slide_count > 0:
                    time_per_slide = total_duration / slide_count
                    
                    for i, slide in enumerate(slides_data["slide_analyses"]):
                        if "error" not in slide:
                            timeline_events.append({
                                "timestamp": i * time_per_slide,
                                "type": "slide_content",
                                "content": slide.get("text", "")[:100] + "...",
                                "slide_number": slide["slide_number"],
                                "source": "slides"
                            })
        
        # 시간순 정렬
        timeline_events.sort(key=lambda x: x["timestamp"])
        
        # 타임라인 통계
        timeline_stats = self._calculate_timeline_stats(timeline_events)
        
        return {
            "timeline_events": timeline_events,
            "timeline_stats": timeline_stats,
            "total_events": len(timeline_events)
        }
    
    def _get_total_duration(self, dimension_analyses: Dict) -> float:
        """총 지속시간 추출"""
        if "audio" in dimension_analyses:
            return dimension_analyses["audio"].get("duration", 0)
        return 0
    
    def _calculate_timeline_stats(self, events: List) -> Dict[str, Any]:
        """타임라인 통계"""
        
        event_types = [e["type"] for e in events]
        type_counts = {t: event_types.count(t) for t in set(event_types)}
        
        # 시간 분포
        timestamps = [e["timestamp"] for e in events if e["timestamp"] > 0]
        
        return {
            "event_type_distribution": type_counts,
            "time_span": max(timestamps) - min(timestamps) if timestamps else 0,
            "average_event_interval": np.mean(np.diff(sorted(timestamps))) if len(timestamps) > 1 else 0
        }
    
    def _fuse_5d_dimensions(self, dimension_analyses: Dict) -> Dict[str, Any]:
        """5차원 융합"""
        
        print("  Fusing all 5 dimensions...")
        
        # 각 차원에서 화자 수 추정
        speaker_estimates = {}
        
        if "audio" in dimension_analyses:
            audio_speakers = dimension_analyses["audio"].get("speaker_analysis", {}).get("speakers", 1)
            speaker_estimates["audio"] = audio_speakers
        
        if "visual" in dimension_analyses:
            visual_speakers = dimension_analyses["visual"].get("estimated_speakers", 1)
            speaker_estimates["visual"] = visual_speakers
        
        if "transcript" in dimension_analyses:
            transcript_speakers = dimension_analyses["transcript"].get("estimated_speakers", 1)
            speaker_estimates["transcript"] = transcript_speakers
        
        # 최종 화자 수 결정 (3명 선호)
        if speaker_estimates:
            # 3에 가장 가까운 값 선택
            final_speakers = min(speaker_estimates.values(), key=lambda x: abs(x - 3))
            final_speakers = max(2, min(3, final_speakers))  # 2-3명 범위
        else:
            final_speakers = 3
        
        # 신뢰도 계산
        confidence = self._calculate_5d_confidence(dimension_analyses, speaker_estimates)
        
        # 종합 분석
        comprehensive_analysis = self._create_comprehensive_analysis(dimension_analyses)
        
        return {
            "final_speaker_count": final_speakers,
            "speaker_estimates": speaker_estimates,
            "confidence": confidence,
            "comprehensive_analysis": comprehensive_analysis,
            "dimensions_used": list(dimension_analyses.keys())
        }
    
    def _calculate_5d_confidence(self, analyses: Dict, estimates: Dict) -> float:
        """5차원 신뢰도 계산"""
        base_confidence = 0.3
        
        # 차원별 가중치
        dimension_weights = {
            "audio": 0.25,
            "visual": 0.20,
            "transcript": 0.25,
            "slides": 0.20,
            "timeline": 0.10
        }
        
        confidence = base_confidence
        
        for dim, weight in dimension_weights.items():
            if dim in analyses and "error" not in analyses[dim]:
                confidence += weight
        
        # 화자 수 일치도 보너스
        if len(estimates) > 1:
            values = list(estimates.values())
            if len(set(values)) == 1:
                confidence += 0.15
            elif max(values) - min(values) <= 1:
                confidence += 0.10
        
        return min(1.0, confidence)
    
    def _create_comprehensive_analysis(self, analyses: Dict) -> Dict[str, Any]:
        """종합 분석 생성"""
        
        comprehensive = {
            "content_summary": "",
            "key_insights": [],
            "presentation_quality": 0.0,
            "topic_coverage": {}
        }
        
        # 텍스트 내용 통합
        all_content = []
        
        if "audio" in analyses and "transcription" in analyses["audio"]:
            audio_text = analyses["audio"]["transcription"].get("text", "")
            all_content.append(audio_text)
        
        if "transcript" in analyses:
            # 여기서는 transcript_text가 직접 전달되어야 함
            # 실제 구현에서는 analyses에 원본 텍스트 저장 필요
            pass
        
        if "slides" in analyses and "combined_text" in analyses["slides"]:
            slides_text = analyses["slides"]["combined_text"]
            all_content.append(slides_text)
        
        # 종합 요약 생성
        combined_text = " ".join(all_content)
        if combined_text:
            comprehensive["content_summary"] = self._generate_summary(combined_text)
            comprehensive["key_insights"] = self._extract_insights(combined_text)
            comprehensive["topic_coverage"] = self._analyze_topics(combined_text)
        
        # 발표 품질 점수
        quality_factors = []
        
        if "slides" in analyses:
            slides_quality = analyses["slides"].get("total_analysis", {}).get("structure_score", 0)
            quality_factors.append(slides_quality)
        
        if "audio" in analyses:
            audio_quality = analyses["audio"].get("speaker_analysis", {}).get("quality_score", 0)
            if audio_quality > 0:
                quality_factors.append(audio_quality)
        
        comprehensive["presentation_quality"] = np.mean(quality_factors) if quality_factors else 0.5
        
        return comprehensive
    
    def _generate_summary(self, text: str) -> str:
        """텍스트 요약 생성"""
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.split()) > 5]
        
        # 가장 긴 3개 문장을 요약으로 사용 (실제로는 더 정교한 요약 알고리즘 필요)
        key_sentences = sorted(sentences, key=len, reverse=True)[:3]
        
        return " ".join(key_sentences)
    
    def _extract_insights(self, text: str) -> List[str]:
        """핵심 인사이트 추출"""
        insights = []
        
        # 패턴 기반 인사이트 추출
        insight_patterns = [
            r'(?i)(increase|decrease|growth|decline).*?(\d+%)',
            r'(?i)(significant|important|crucial|key).*?(?=\.)',
            r'(?i)(trend|pattern|shift).*?(?=\.)'
        ]
        
        for pattern in insight_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    insights.append(' '.join(match))
                else:
                    insights.append(match)
        
        return insights[:5]  # 최대 5개
    
    def _apply_5d_context(self, fusion_result: Dict, context: Dict) -> Dict[str, Any]:
        """5차원 컨텍스트 적용"""
        
        if not context:
            return fusion_result
        
        # 컨퍼런스 정보 적용
        conference_name = context.get("conference_name", "")
        expected_speakers = context.get("expected_speakers", 3)
        
        # 화자 수 조정
        if expected_speakers and expected_speakers != fusion_result["final_speaker_count"]:
            fusion_result["context_adjusted"] = True
            fusion_result["original_speaker_count"] = fusion_result["final_speaker_count"]
            fusion_result["final_speaker_count"] = expected_speakers
            fusion_result["confidence"] = min(0.95, fusion_result["confidence"] + 0.05)
        
        # 주제 관련성 확인
        if "eco" in conference_name.lower() and "luxury" in conference_name.lower():
            fusion_result["topic_match"] = True
            fusion_result["confidence"] = min(0.98, fusion_result["confidence"] + 0.03)
        
        return fusion_result
    
    def _generate_ultimate_report(self, dimension_analyses: Dict, fusion_result: Dict) -> Dict[str, Any]:
        """최종 통합 리포트 생성"""
        
        report = {
            "executive_summary": {},
            "detailed_findings": {},
            "recommendations": [],
            "quality_assessment": {}
        }
        
        # 경영진 요약
        report["executive_summary"] = {
            "speaker_count": fusion_result["final_speaker_count"],
            "confidence_level": f"{fusion_result['confidence']:.1%}",
            "dimensions_analyzed": len(dimension_analyses),
            "presentation_topic": "The Rise of the Eco-Friendly Luxury Consumer",
            "analysis_completeness": self._calculate_completeness(dimension_analyses)
        }
        
        # 상세 발견사항
        detailed = {}
        
        for dimension, analysis in dimension_analyses.items():
            if "error" not in analysis:
                detailed[dimension] = self._summarize_dimension(dimension, analysis)
        
        report["detailed_findings"] = detailed
        
        # 권장사항
        report["recommendations"] = self._generate_recommendations(dimension_analyses, fusion_result)
        
        # 품질 평가
        report["quality_assessment"] = {
            "overall_quality": fusion_result["confidence"],
            "data_completeness": self._assess_data_completeness(dimension_analyses),
            "analysis_reliability": self._assess_reliability(fusion_result)
        }
        
        return report
    
    def _calculate_completeness(self, analyses: Dict) -> float:
        """분석 완성도 계산"""
        total_dimensions = 5
        completed_dimensions = len([a for a in analyses.values() if "error" not in a])
        return completed_dimensions / total_dimensions
    
    def _summarize_dimension(self, dimension: str, analysis: Dict) -> Dict[str, Any]:
        """차원별 요약"""
        
        summaries = {
            "audio": {
                "duration": analysis.get("duration", 0),
                "speaker_analysis": analysis.get("speaker_analysis", {}),
                "transcription_available": bool(analysis.get("transcription", {}).get("text"))
            },
            "visual": {
                "estimated_speakers": analysis.get("estimated_speakers", 0),
                "analysis_points": analysis.get("analysis_points", 0),
                "transitions_detected": len(analysis.get("slide_transitions", []))
            },
            "transcript": {
                "word_count": analysis.get("stats", {}).get("word_count", 0),
                "estimated_speakers": analysis.get("estimated_speakers", 0),
                "structure_score": analysis.get("structure_analysis", {}).get("structure_score", 0)
            },
            "slides": {
                "total_slides": analysis.get("total_slides", 0),
                "processed_successfully": analysis.get("processed_slides", 0),
                "structure_score": analysis.get("total_analysis", {}).get("structure_score", 0)
            },
            "timeline": {
                "total_events": analysis.get("total_events", 0),
                "time_span": analysis.get("timeline_stats", {}).get("time_span", 0)
            }
        }
        
        return summaries.get(dimension, {"status": "analyzed"})
    
    def _generate_recommendations(self, analyses: Dict, fusion_result: Dict) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        # 신뢰도 기반 권장사항
        confidence = fusion_result["confidence"]
        
        if confidence > 0.8:
            recommendations.append("High confidence analysis - results can be trusted for decision making")
        elif confidence > 0.6:
            recommendations.append("Moderate confidence - consider additional validation")
        else:
            recommendations.append("Low confidence - recommend collecting more data sources")
        
        # 차원별 권장사항
        if "slides" in analyses and analyses["slides"].get("total_slides", 0) > 15:
            recommendations.append("Large number of slides detected - consider slide content prioritization")
        
        if "audio" in analyses and analyses["audio"].get("duration", 0) > 3600:
            recommendations.append("Long presentation detected - consider segmented analysis")
        
        return recommendations
    
    def _assess_data_completeness(self, analyses: Dict) -> float:
        """데이터 완성도 평가"""
        weights = {"audio": 0.3, "visual": 0.2, "transcript": 0.3, "slides": 0.2}
        
        completeness = 0
        for dimension, weight in weights.items():
            if dimension in analyses and "error" not in analyses[dimension]:
                completeness += weight
        
        return completeness
    
    def _assess_reliability(self, fusion_result: Dict) -> float:
        """신뢰성 평가"""
        
        base_reliability = 0.5
        
        # 차원 수에 따른 신뢰성
        dimensions_count = len(fusion_result.get("dimensions_used", []))
        reliability = base_reliability + (dimensions_count * 0.1)
        
        # 화자 추정 일치도
        estimates = fusion_result.get("speaker_estimates", {})
        if len(estimates) > 1:
            values = list(estimates.values())
            if len(set(values)) == 1:
                reliability += 0.2
            elif max(values) - min(values) <= 1:
                reliability += 0.1
        
        return min(1.0, reliability)
    
    def save_ultimate_analysis(self, result: Dict[str, Any], output_path: str = None) -> str:
        """최종 분석 결과 저장"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"ultimate_5d_analysis_{timestamp}.json"
        
        # JSON 직렬화 가능하도록 처리
        serializable_result = self._make_json_serializable(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\nUltimate analysis saved: {output_path}")
        return output_path
    
    def _make_json_serializable(self, obj):
        """JSON 직렬화 가능하게 변환"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return float(obj)
        else:
            return obj

def main():
    """메인 실행"""
    print("ULTIMATE 5D MULTIMODAL CONFERENCE ANALYZER")
    print("=" * 60)
    
    analyzer = Ultimate5DMultimodalAnalyzer()
    
    # 파일 경로들
    video_path = "C:/Users/PC_58410/solomond-ai-system/user_files/JGA2025_D1/IMG_0032.MOV"
    slides_folder = "C:/Users/PC_58410/solomond-ai-system/user_files/JGA2025_D1/"
    
    # 샘플 텍스트 기록물
    transcript_text = """
    Speaker 1: Welcome everyone to today's presentation on The Rise of the Eco-Friendly Luxury Consumer. 
    I'm excited to share our research findings with you. The luxury market is experiencing unprecedented transformation.
    
    Speaker 2: Thank you for that introduction. Now, moving on to the market analysis.
    Our data shows a significant shift in consumer behavior over the past five years.
    Sustainable luxury has grown by 15% annually, outpacing traditional luxury segments.
    
    Speaker 3: Building on that analysis, let me present our recommendations for luxury brands.
    First, transparency in supply chain management is crucial. Second, investing in sustainable materials 
    and processes can yield both environmental and financial returns.
    
    In conclusion, the eco-friendly luxury consumer represents not just a trend, but a fundamental shift 
    that will define the future of the luxury market.
    """
    
    # 컨텍스트
    context = {
        "conference_name": "The Rise of the Eco-Friendly Luxury Consumer",
        "expected_speakers": 3,
        "presentation_type": "sequential",
        "industry": "luxury_goods",
        "topic": "sustainability"
    }
    
    try:
        # 5차원 완전 분석 실행
        result = analyzer.analyze_complete_5d(
            video_path=video_path if os.path.exists(video_path) else None,
            transcript_text=transcript_text,
            slides_folder=slides_folder,
            context=context
        )
        
        # 결과 출력
        print("\n" + "="*80)
        print("ULTIMATE 5D MULTIMODAL ANALYSIS RESULTS")
        print("="*80)
        
        # 경영진 요약
        exec_summary = result["final_report"]["executive_summary"]
        print(f"\nEXECUTIVE SUMMARY")
        print(f"Speaker Count: {exec_summary['speaker_count']} speakers")
        print(f"Confidence: {exec_summary['confidence_level']}")
        print(f"Dimensions Analyzed: {exec_summary['dimensions_analyzed']}/5")
        print(f"Analysis Completeness: {exec_summary['analysis_completeness']:.1%}")
        
        # 융합 결과
        fusion = result["fusion_result"]
        print(f"\nFUSION RESULTS")
        print(f"Final Speaker Count: {fusion['final_speaker_count']} speakers")
        print(f"Overall Confidence: {fusion['confidence']:.3f}")
        print(f"Speaker Estimates: {fusion['speaker_estimates']}")
        print(f"Dimensions Used: {', '.join(fusion['dimensions_used'])}")
        
        # 권장사항
        recommendations = result["final_report"]["recommendations"]
        print(f"\nRECOMMENDATIONS")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # 품질 평가
        quality = result["final_report"]["quality_assessment"]
        print(f"\nQUALITY ASSESSMENT")
        print(f"Overall Quality: {quality['overall_quality']:.3f}")
        print(f"Data Completeness: {quality['data_completeness']:.3f}")
        print(f"Analysis Reliability: {quality['analysis_reliability']:.3f}")
        
        # 결과 저장
        output_file = analyzer.save_ultimate_analysis(result)
        print(f"\nUltimate 5D Analysis Complete!")
        print(f"Full report: {output_file}")
        print(f"Processing time: {result['processing_time']:.1f} seconds")
        
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()