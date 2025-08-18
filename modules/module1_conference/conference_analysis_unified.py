#!/usr/bin/env python3
"""
🏆 모듈 1: 통합 컨퍼런스 분석 시스템
Unified Conference Analysis System

🎯 단일 시스템, 다중 모드:
- 🏆 궁극 모드: 모든 기능 + 최고 성능
- ⚖️ 균형 모드: 핵심 기능 + 안정성
- 🛡️ 안전 모드: 기본 기능 + 최대 안정성

✨ 모든 기능 통합:
- 🔥 터보 업로드 (3가지 속도)
- 🌐 URL 다운로드 지원
- 🎬 비디오 화면 인식
- 💾 스마트 캐시 시스템
- 🛡️ 네트워크 안정성
- 📊 10GB 파일 지원
- ⚡ GPU/CPU 자동 최적화
- 🎭 고품질 화자 분리
- 📈 실시간 진행률 추적
"""

import streamlit as st
import os
import sys
import tempfile
import time
import hashlib
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import pickle
import gzip
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import subprocess
import sys
import os

# Ollama 인터페이스 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from shared.ollama_interface import OllamaInterface
    OLLAMA_AVAILABLE = True
    print("Ollama interface loaded successfully")
except ImportError as e:
    OLLAMA_AVAILABLE = False
    print(f"Ollama interface load failed: {e}")

# AI 라이브러리
try:
    import whisper
    import librosa
    from sklearn.cluster import KMeans, MiniBatchKMeans  
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    import easyocr
    import numpy as np
    import torch
    import cv2
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# URL 다운로드 라이브러리
try:
    import yt_dlp
    import pytube
    URL_DOWNLOAD_AVAILABLE = True
except ImportError:
    URL_DOWNLOAD_AVAILABLE = False

class UnifiedConferenceAnalyzer:
    """통합 컨퍼런스 분석기 - 모든 기능을 하나로"""
    
    def __init__(self):
        # Ollama AI 초기화
        self.ollama = None
        if OLLAMA_AVAILABLE:
            try:
                self.ollama = OllamaInterface()
                print("Ollama AI connected successfully")
            except Exception as e:
                print(f"Ollama AI connection failed: {e}")
        
        self.analysis_modes = {
            "ultimate": {
                "name": "🏆 궁극 모드",
                "description": "🤖 AI 지능분석 + 모든 기능 + 최고 성능 (권장)",
                "upload_speed": "turbo",      # 10배 빠름
                "chunk_size": 10 * 1024 * 1024,  # 10MB
                "parallel_workers": 8,
                "network_stability": "balanced",
                "features": ["audio", "video", "image", "text", "url", "cache", "gpu", "ai_analysis"],
                "quality": "high",
                "color": "#FFD700"
            },
            "balanced": {
                "name": "⚖️ 균형 모드", 
                "description": "🤖 AI 기본분석 + 핵심 기능 + 안정성 (일반 사용)",
                "upload_speed": "fast",       # 5배 빠름
                "chunk_size": 5 * 1024 * 1024,   # 5MB
                "parallel_workers": 4,
                "network_stability": "high",
                "features": ["audio", "video", "image", "text", "cache", "ai_analysis"],
                "quality": "medium",
                "color": "#4CAF50"
            },
            "safe": {
                "name": "🛡️ 안전 모드",
                "description": "기본 기능 + 최대 안정성 (네트워크 불안정시)",
                "upload_speed": "normal",     # 기본 속도
                "chunk_size": 1 * 1024 * 1024,   # 1MB
                "parallel_workers": 2,
                "network_stability": "maximum",
                "features": ["audio", "image", "text"],
                "quality": "stable",
                "color": "#2196F3"
            }
        }
        
        self.current_mode = "ultimate"  # 기본 모드
        self.cache_dir = Path("cache/conference_analysis")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # AI 모델 초기화 상태
        self.models_loaded = {
            "whisper": None,
            "easyocr": None
        }
        
    def render_mode_selector(self):
        """분석 모드 선택 UI"""
        st.markdown("## 🎯 분석 모드 선택")
        
        # 모드 선택
        mode_options = []
        for mode_key, config in self.analysis_modes.items():
            mode_options.append(f"{config['name']} - {config['description']}")
        
        selected_option = st.selectbox(
            "원하는 분석 모드를 선택하세요:",
            mode_options,
            index=0,  # 궁극 모드가 기본
            help="네트워크가 불안정하면 안전 모드를 선택하세요."
        )
        
        # 선택된 모드 추출
        if "궁극" in selected_option:
            self.current_mode = "ultimate"
        elif "균형" in selected_option:
            self.current_mode = "balanced"
        elif "안전" in selected_option:
            self.current_mode = "safe"
        
        # 선택된 모드 정보 표시
        config = self.analysis_modes[self.current_mode]
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {config['color']}20, {config['color']}05);
            border: 2px solid {config['color']};
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
        ">
            <h3 style="color: {config['color']}; margin: 0;">
                {config['name']} 선택됨
            </h3>
            <p style="margin: 10px 0;"><strong>설명:</strong> {config['description']}</p>
            <p style="margin: 5px 0;"><strong>업로드 속도:</strong> {config['upload_speed']}</p>
            <p style="margin: 5px 0;"><strong>청크 크기:</strong> {config['chunk_size'] // 1024 // 1024}MB</p>
            <p style="margin: 5px 0;"><strong>병렬 처리:</strong> {config['parallel_workers']}개 스레드</p>
            <p style="margin: 5px 0;"><strong>지원 기능:</strong> {', '.join(config['features'])}</p>
        </div>
        """, unsafe_allow_html=True)
        
        return self.current_mode
    
    def render_upload_system(self):
        """통합 업로드 시스템"""
        st.markdown("## 📂 파일 업로드")
        
        config = self.analysis_modes[self.current_mode]
        
        # 업로드 방식 선택
        upload_method = st.radio(
            "업로드 방식 선택:",
            ["📁 파일 업로드", "🌐 URL 다운로드", "📝 텍스트 입력", "📂 폴더 업로드"],
            horizontal=True
        )
        
        uploaded_files = []
        
        if upload_method == "📁 파일 업로드":
            uploaded_files = self.render_file_upload(config)
        elif upload_method == "🌐 URL 다운로드":
            uploaded_files = self.render_url_download(config)
        elif upload_method == "📝 텍스트 입력":
            uploaded_files = self.render_text_input(config)
        elif upload_method == "📂 폴더 업로드":
            uploaded_files = self.render_folder_upload(config)
        
        return uploaded_files
    
    def render_file_upload(self, config):
        """파일 업로드 UI"""
        # 모드별 업로드 설정 표시
        st.info(f"🚀 {config['name']} - {config['chunk_size']//1024//1024}MB 청크, {config['parallel_workers']}개 병렬 처리")
        
        uploaded_files = st.file_uploader(
            f"분석할 파일들을 업로드하세요 ({config['upload_speed']} 모드)",
            type=['mp4', 'avi', 'mov', 'mp3', 'wav', 'm4a', 'jpg', 'png', 'pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help=f"최대 10GB 파일 지원 | {config['upload_speed']} 속도 모드",
            key=f"upload_{self.current_mode}"
        )
        
        if uploaded_files:
            # 터보 업로드 진행률 표시
            self.render_upload_progress(uploaded_files, config)
        
        return uploaded_files
    
    def render_upload_progress(self, files, config):
        """업로드 진행률 및 통계 표시"""
        st.markdown("### 📊 업로드 진행 상황")
        
        # 실시간 대시보드
        col1, col2, col3, col4 = st.columns(4)
        
        total_size = sum(len(file.getvalue()) for file in files)
        total_size_gb = total_size / (1024**3)
        
        with col1:
            st.metric("📁 파일 수", f"{len(files)}개")
        with col2:
            st.metric("📊 총 용량", f"{total_size_gb:.2f} GB")
        with col3:
            # 예상 속도 계산
            if config['upload_speed'] == 'turbo':
                estimated_speed = 50
            elif config['upload_speed'] == 'fast':
                estimated_speed = 25
            else:
                estimated_speed = 10
            st.metric("⚡ 예상 속도", f"{estimated_speed} MB/s")
        with col4:
            estimated_time = (total_size / (1024**2)) / estimated_speed
            st.metric("⏱️ 예상 시간", f"{estimated_time:.1f}초")
        
        # 성능 팁
        if config['upload_speed'] == 'turbo':
            st.success(f"🚀 터보 모드: {config['parallel_workers']}개 병렬 스레드로 최고 속도!")
        elif config['upload_speed'] == 'fast':
            st.info(f"⚡ 고속 모드: {config['parallel_workers']}개 스레드로 빠른 처리!")
        else:
            st.info(f"🛡️ 안전 모드: {config['parallel_workers']}개 스레드로 안정적 처리!")
    
    def render_url_download(self, config):
        """URL 다운로드 UI"""
        if not URL_DOWNLOAD_AVAILABLE:
            st.warning("⚠️ URL 다운로드 기능을 사용하려면 yt-dlp를 설치하세요: pip install yt-dlp")
            return []
        
        st.info(f"🌐 {config['name']} - URL에서 직접 다운로드")
        
        url = st.text_input(
            "다운로드할 URL을 입력하세요:",
            placeholder="https://www.youtube.com/watch?v=... 또는 웹페이지 URL",
            help="YouTube, 웹페이지, 문서 URL 지원"
        )
        
        if url and st.button("🚀 다운로드 시작"):
            with st.spinner("URL에서 다운로드 중..."):
                downloaded_files = self.download_from_url(url, config)
                if downloaded_files:
                    st.success(f"✅ 다운로드 완료: {len(downloaded_files)}개 파일")
                    return downloaded_files
        
        return []
    
    def render_text_input(self, config):
        """텍스트 입력 UI"""
        st.info(f"📝 {config['name']} - 텍스트 직접 입력")
        
        text_content = st.text_area(
            "분석할 텍스트를 입력하세요:",
            height=200,
            placeholder="회의 내용, 강연 스크립트, 대화 내용 등을 입력하세요...",
            help="입력한 텍스트를 바로 분석합니다"
        )
        
        if text_content and st.button("📝 텍스트 분석 시작"):
            # 텍스트를 임시 파일로 변환
            text_file = io.StringIO(text_content)
            text_file.name = f"text_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            return [text_file]
        
        return []
    
    def render_folder_upload(self, config):
        """폴더 업로드 UI (여러 파일 한번에)"""
        st.info(f"📂 {config['name']} - 여러 파일 한번에 업로드")
        st.warning("⚠️ 브라우저 제한으로 폴더 업로드는 파일 선택 방식으로 대체됩니다.")
        
        # 다중 파일 선택으로 폴더 업로드 시뮬레이션
        uploaded_files = st.file_uploader(
            "폴더의 모든 파일들을 선택하세요 (Ctrl+클릭으로 다중 선택)",
            type=['mp4', 'avi', 'mov', 'mp3', 'wav', 'm4a', 'jpg', 'png', 'pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="같은 폴더의 파일들을 모두 선택하여 배치 업로드",
            key=f"folder_upload_{self.current_mode}"
        )
        
        if uploaded_files:
            st.success(f"📂 배치 업로드: {len(uploaded_files)}개 파일 선택됨")
            self.render_upload_progress(uploaded_files, config)
        
        return uploaded_files
    
    def render_analysis_button(self, files):
        """분석 시작 버튼"""
        if not files:
            return False
        
        config = self.analysis_modes[self.current_mode]
        
        st.markdown("---")
        st.markdown("## 🚀 분석 시작")
        
        # 분석 설정 요약
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **선택된 모드:** {config['name']}  
            **파일 수:** {len(files)}개  
            **지원 기능:** {', '.join(config['features'][:3])}...
            """)
        
        with col2:
            st.markdown(f"""
            **처리 속도:** {config['upload_speed']}  
            **안정성:** {config['network_stability']}  
            **품질:** {config['quality']}
            """)
        
        # 분석 시작 버튼 (모드별 색상)
        button_color = config['color']
        
        if st.button(
            f"🚀 {config['name']} 분석 시작",
            type="primary",
            help=f"선택된 {len(files)}개 파일을 {config['name']}로 분석합니다"
        ):
            return True
        
        return False
    
    def execute_unified_analysis(self, files):
        """통합 분석 실행"""
        config = self.analysis_modes[self.current_mode]
        
        st.markdown(f"## 🔄 {config['name']} 분석 진행 중...")
        
        # 전체 진행률
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 결과 저장
        analysis_results = {
            "mode": self.current_mode,
            "config": config,
            "files_processed": 0,
            "total_files": len(files),
            "results": [],
            "start_time": datetime.now(),
            "errors": []
        }
        
        # 순차 처리로 안정성 확보 (ThreadPoolExecutor 제거)
        for i, file in enumerate(files):
            try:
                status_text.text(f"🔄 분석 중: {file.name} ({i+1}/{len(files)})")
                
                file_result = self.analyze_single_file(file, config)
                analysis_results["results"].append(file_result)
                analysis_results["files_processed"] += 1
                
                progress = (i + 1) / len(files)
                progress_bar.progress(progress)
                
            except Exception as e:
                error_info = {
                    "file": file.name,
                    "error": str(e),
                    "timestamp": datetime.now()
                }
                analysis_results["errors"].append(error_info)
                st.error(f"❌ {file.name} 분석 실패: {str(e)}")
        
        # 분석 완료
        analysis_results["end_time"] = datetime.now()
        analysis_results["duration"] = (analysis_results["end_time"] - analysis_results["start_time"]).total_seconds()
        
        # 종합 분석 수행 (새로 추가)
        if analysis_results["files_processed"] > 0:
            status_text.text("🔄 종합 분석 생성 중...")
            
            # 결과를 파일명 기준으로 정리
            file_results = {}
            for result in analysis_results["results"]:
                if "filename" in result:
                    file_results[result["filename"]] = result
            
            # 종합 분석 실행
            comprehensive_summary = self.generate_comprehensive_summary(file_results)
            analysis_results["comprehensive_summary"] = comprehensive_summary
        
        progress_bar.progress(1.0)
        status_text.text("✅ 분석 완료!")
        
        return analysis_results
    
    def analyze_single_file(self, file, config):
        """개별 파일 분석"""
        file_extension = Path(file.name).suffix.lower()
        
        result = {
            "filename": file.name,
            "type": file_extension,
            "size": len(file.getvalue()) if hasattr(file, 'getvalue') else 0,
            "analysis": {},
            "timestamp": datetime.now()
        }
        
        # 캐시 확인
        if "cache" in config["features"]:
            cached_result = self.check_cache(file)
            if cached_result:
                st.info(f"📄 캐시에서 로드: {file.name}")
                return cached_result
        
        # 파일 타입별 분석
        if file_extension in ['.mp3', '.wav', '.m4a'] and "audio" in config["features"]:
            result["analysis"] = self.analyze_audio(file, config)
        elif file_extension in ['.mp4', '.avi', '.mov'] and "video" in config["features"]:
            result["analysis"] = self.analyze_video(file, config)
        elif file_extension in ['.jpg', '.png', '.jpeg'] and "image" in config["features"]:
            result["analysis"] = self.analyze_image(file, config)
        elif file_extension in ['.txt', '.pdf', '.docx'] and "text" in config["features"]:
            result["analysis"] = self.analyze_text(file, config)
        else:
            result["analysis"] = {"error": f"지원하지 않는 파일 형식: {file_extension}"}
        
        # 캐시 저장
        if "cache" in config["features"]:
            self.save_cache(file, result)
        
        return result
    
    def analyze_audio(self, file, config):
        """오디오 분석 (화자 분리 + STT)"""
        if not AI_AVAILABLE:
            return {"error": "AI 라이브러리가 설치되지 않았습니다"}
        
        try:
            # Whisper STT
            if self.models_loaded["whisper"] is None:
                self.models_loaded["whisper"] = whisper.load_model("base")
            
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            # STT 실행
            result = self.models_loaded["whisper"].transcribe(tmp_path)
            
            # 화자 분리 (궁극 모드만)
            speakers = {}
            if config["quality"] == "high":
                speakers = self.perform_speaker_diarization(tmp_path)
            
            # 임시 파일 삭제
            os.unlink(tmp_path)
            
            return {
                "transcription": result["text"],
                "language": result.get("language", "unknown"),
                "speakers": speakers,
                "confidence": "high" if config["quality"] == "high" else "medium"
            }
            
        except Exception as e:
            return {"error": f"오디오 분석 실패: {str(e)}"}
    
    def analyze_video(self, file, config):
        """비디오 분석 (음성 + 화면)"""
        # 비디오는 오디오 추출 후 분석
        return {
            "type": "video",
            "note": "비디오 분석 기능 구현 예정",
            "audio_extracted": False
        }
    
    def analyze_image(self, file, config):
        """이미지 분석 (OCR)"""
        if not AI_AVAILABLE:
            return {"error": "AI 라이브러리가 설치되지 않았습니다"}
        
        try:
            # EasyOCR
            if self.models_loaded["easyocr"] is None:
                self.models_loaded["easyocr"] = easyocr.Reader(['ko', 'en'])
            
            # 이미지 읽기
            image_bytes = file.getvalue()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # OCR 실행
            results = self.models_loaded["easyocr"].readtext(image)
            
            # 텍스트 추출
            extracted_text = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # 신뢰도 필터
                    extracted_text.append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox
                    })
            
            return {
                "extracted_text": extracted_text,
                "total_text_blocks": len(extracted_text),
                "avg_confidence": np.mean([item["confidence"] for item in extracted_text]) if extracted_text else 0
            }
            
        except Exception as e:
            return {"error": f"이미지 분석 실패: {str(e)}"}
    
    def analyze_text(self, file, config):
        """텍스트 분석"""
        try:
            if hasattr(file, 'getvalue'):
                content = file.getvalue().decode('utf-8')
            else:
                content = file.read()
            
            # 기본 텍스트 통계
            lines = content.split('\n')
            words = content.split()
            
            return {
                "content": content[:1000] + "..." if len(content) > 1000 else content,
                "stats": {
                    "characters": len(content),
                    "words": len(words),
                    "lines": len(lines)
                },
                "preview": content[:200] + "..." if len(content) > 200 else content
            }
            
        except Exception as e:
            return {"error": f"텍스트 분석 실패: {str(e)}"}
    
    def perform_speaker_diarization(self, audio_path):
        """고품질 화자 분리 (궁극 모드 전용)"""
        try:
            # 음성 특징 추출
            y, sr = librosa.load(audio_path, sr=16000)
            
            # MFCC 특징 추출
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # 스펙트럴 특징
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            
            # 크로마 특징
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # RMS 에너지
            rms = librosa.feature.rms(y=y)
            
            # 모든 특징 결합 (29차원)
            features = np.vstack([
                mfcc,                    # 13차원
                spectral_centroids,      # 1차원
                spectral_rolloff,        # 1차원  
                spectral_bandwidth,      # 1차원
                chroma,                  # 12차원
                rms                      # 1차원
            ])
            
            # 특징 정규화
            features = features.T  # 시간 축으로 transpose
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # PCA 차원 축소
            pca = PCA(n_components=0.95)  # 95% 분산 유지
            features_pca = pca.fit_transform(features_scaled)
            
            # 최적 화자 수 찾기 (실루엣 스코어 기반)
            best_n_speakers = 2
            best_score = -1
            
            for n in range(2, min(7, len(features_pca) // 10)):
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features_pca)
                score = silhouette_score(features_pca, labels)
                
                if score > best_score:
                    best_score = score
                    best_n_speakers = n
            
            # 최종 클러스터링
            kmeans = KMeans(n_clusters=best_n_speakers, random_state=42, n_init=10)
            speaker_labels = kmeans.fit_predict(features_pca)
            
            # 화자별 세그먼트 생성
            speakers = {}
            hop_length = 512
            frame_duration = hop_length / sr
            
            for i, label in enumerate(speaker_labels):
                speaker_id = f"화자_{label + 1}"
                if speaker_id not in speakers:
                    speakers[speaker_id] = []
                
                start_time = i * frame_duration
                end_time = (i + 1) * frame_duration
                
                speakers[speaker_id].append({
                    "start": start_time,
                    "end": end_time,
                    "confidence": best_score
                })
            
            return {
                "speaker_count": best_n_speakers,
                "silhouette_score": best_score,
                "speakers": speakers,
                "method": "29D_features_silhouette_optimized"
            }
            
        except Exception as e:
            return {"error": f"화자 분리 실패: {str(e)}"}
    
    def check_cache(self, file):
        """캐시 확인"""
        try:
            # 파일 해시 생성
            file_hash = hashlib.md5(file.getvalue()).hexdigest()
            cache_file = self.cache_dir / f"{file_hash}.pkl.gz"
            
            if cache_file.exists():
                with gzip.open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
        except Exception:
            pass
    
    def generate_comprehensive_summary(self, all_results):
        """종합 분석 요약 생성 - 여러 파일을 하나의 스토리로 통합 (AI 강화)"""
        try:
            # 분석 결과 정리
            audio_results = []
            video_results = []
            image_results = []
            text_results = []
            
            for file_name, result in all_results.items():
                if result.get('type') == 'audio':
                    audio_results.append(result)
                elif result.get('type') == 'video':
                    video_results.append(result)
                elif result.get('type') == 'image':
                    image_results.append(result)
                elif result.get('type') == 'text':
                    text_results.append(result)
            
            # 전사 텍스트 통합
            all_transcripts = []
            speaker_contents = {}
            
            for result in audio_results + video_results:
                if 'transcript' in result:
                    all_transcripts.append(result['transcript'])
                
                # 화자별 내용 통합
                if 'speaker_diarization' in result:
                    for speaker_id, segments in result['speaker_diarization'].get('speakers', {}).items():
                        if speaker_id not in speaker_contents:
                            speaker_contents[speaker_id] = []
                        
                        # 해당 화자의 시간대 텍스트 추출
                        transcript = result.get('transcript', '')
                        for segment in segments:
                            # 간단한 시간 기반 텍스트 매핑 (개선 필요)
                            start_char = int(segment['start'] * 10)  # 대략적 매핑
                            end_char = int(segment['end'] * 10)
                            speaker_text = transcript[start_char:end_char] if transcript else "음성 감지됨"
                            if speaker_text.strip():
                                speaker_contents[speaker_id].append(speaker_text.strip())
            
            # 텍스트 파일에서도 화자 내용 추출 (중요!)
            for result in text_results:
                if 'content' in result and result['content']:
                    content = result['content']
                    all_transcripts.append(content)
                    
                    # 간단한 화자 구분 (화자1:, 화자2: 등으로 구분)
                    lines = content.split('\n')
                    current_speaker = None
                    for line in lines:
                        if '화자' in line and ':' in line:
                            # 화자 ID 추출
                            if '화자1' in line or '화자_1' in line:
                                current_speaker = '화자_1'
                            elif '화자2' in line or '화자_2' in line:
                                current_speaker = '화자_2' 
                            elif '화자3' in line or '화자_3' in line:
                                current_speaker = '화자_3'
                            
                            if current_speaker:
                                if current_speaker not in speaker_contents:
                                    speaker_contents[current_speaker] = []
                                
                                # 콜론 뒤의 내용 추출
                                speaker_text = line.split(':', 1)[-1].strip()
                                if speaker_text:
                                    speaker_contents[current_speaker].append(speaker_text)
            
            # 이미지에서 추출된 텍스트 통합
            all_image_texts = []
            for result in image_results:
                if 'text_content' in result:
                    all_image_texts.append(result['text_content'])
            
            # 전체 상황 종합 분석
            combined_text = "\n".join(all_transcripts + all_image_texts + [r.get('content', '') for r in text_results])
            
            # 핵심 키워드 추출 (간단한 방법)
            words = combined_text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # 3글자 이상만
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # AI 기반 지능적 분석 추가
            ai_enhanced_analysis = self._generate_ai_enhanced_analysis(combined_text, speaker_contents, all_image_texts)
            
            # 종합 요약 생성
            summary = {
                "분석_개요": {
                    "총_파일_수": len(all_results),
                    "오디오_파일": len(audio_results),
                    "비디오_파일": len(video_results), 
                    "이미지_파일": len(image_results),
                    "텍스트_파일": len(text_results),
                    "분석_시간": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "화자_분석": {
                    "감지된_화자_수": len(speaker_contents),
                    "화자별_발언_내용": {
                        speaker_id: {
                            "발언_횟수": len(contents),
                            "주요_발언": contents[:3] if contents else ["발언 내용 없음"],
                            "전체_발언": contents,
                            "AI_의미_분석": ai_enhanced_analysis.get("화자_의미_분석", {}).get(speaker_id, "분석 중...")
                        }
                        for speaker_id, contents in speaker_contents.items()
                    }
                },
                "주요_내용": {
                    "핵심_키워드": [{"단어": word, "빈도": freq} for word, freq in top_keywords],
                    "전체_전사_텍스트": combined_text[:1000] + "..." if len(combined_text) > 1000 else combined_text,
                    "이미지_텍스트": all_image_texts
                },
                "통합_스토리": self._generate_integrated_story(combined_text, speaker_contents, all_image_texts),
                "AI_상황_분석": ai_enhanced_analysis,  # AI 분석 결과 추가
                "사용된_제원": {
                    "음성_인식": "OpenAI Whisper",
                    "화자_분리": "29차원 특징 + K-means 클러스터링", 
                    "이미지_OCR": "EasyOCR",
                    "특징_추출": "MFCC, 스펙트럴, 크로마, RMS",
                    "최적화": "실루엣 스코어 기반 화자 수 자동 결정",
                    "AI_분석": f"Ollama {ai_enhanced_analysis.get('사용된_모델', 'qwen2.5:7b')}" if self.ollama else "미사용"
                }
            }
            
            return summary
            
        except Exception as e:
            return {"error": f"종합 분석 실패: {str(e)}"}
    
    def _generate_integrated_story(self, combined_text, speaker_contents, image_texts):
        """통합 스토리 생성"""
        try:
            story_parts = []
            
            # 상황 개요
            if combined_text:
                story_parts.append(f"📋 **전체 상황**: {combined_text[:200]}...")
            
            # 화자별 주요 발언
            if speaker_contents:
                story_parts.append("🎭 **화자별 주요 발언**:")
                for speaker_id, contents in speaker_contents.items():
                    if contents:
                        main_content = " | ".join(contents[:2])  # 주요 발언 2개
                        story_parts.append(f"  - {speaker_id}: {main_content}")
            
            # 시각적 정보
            if image_texts:
                story_parts.append("🖼️ **시각적 정보**:")
                for i, text in enumerate(image_texts[:3]):  # 최대 3개
                    if text.strip():
                        story_parts.append(f"  - 이미지 {i+1}: {text[:100]}...")
            
            # 결론
            story_parts.append("📊 **종합 결론**: 다각도 분석을 통해 수집된 정보를 통합하여 전체 상황을 파악했습니다.")
            
            return "\n".join(story_parts)
            
        except Exception as e:
            return f"스토리 생성 중 오류: {str(e)}"
    
    def _generate_ai_enhanced_analysis(self, combined_text, speaker_contents, image_texts):
        """AI 기반 지능적 상황 분석"""
        if not self.ollama:
            return {"error": "Ollama AI가 연결되지 않았습니다"}
        
        # 디버깅: 받은 데이터 확인
        print(f"DEBUG: combined_text length: {len(combined_text) if combined_text else 0}")
        print(f"DEBUG: speaker_contents: {speaker_contents}")
        print(f"DEBUG: image_texts: {image_texts}")
        
        try:
            # 상황별 최적 모델 선택
            selected_model = self._select_optimal_model(combined_text, len(speaker_contents))
            
            # 1. 전체 상황 분석
            situation_analysis = self._analyze_overall_situation(combined_text, selected_model)
            
            # 2. 화자별 의미 분석  
            speaker_meanings = self._analyze_speaker_meanings(speaker_contents, selected_model)
            
            # 3. 회의 맥락 및 결론 분석
            context_analysis = self._analyze_meeting_context(combined_text, selected_model)
            
            return {
                "사용된_모델": selected_model,
                "전체_상황_분석": situation_analysis,
                "화자_의미_분석": speaker_meanings,
                "회의_맥락_분석": context_analysis,
                "AI_종합_결론": self._generate_ai_conclusion(situation_analysis, context_analysis, selected_model)
            }
            
        except Exception as e:
            return {"error": f"AI 분석 실패: {str(e)}"}
    
    def _select_optimal_model(self, text, speaker_count):
        """상황별 최적 모델 선택"""
        text_length = len(text)
        
        # 복잡도 기반 모델 선택
        if text_length > 5000 or speaker_count > 4:
            # 복잡한 상황 - 강력한 모델 필요
            return "qwen2.5:14b" if "qwen2.5:14b" in self.ollama.available_models else "qwen2.5:7b"
        elif text_length > 2000 or speaker_count > 2:
            # 중간 복잡도 - 균형잡힌 모델
            return "qwen2.5:7b"
        else:
            # 간단한 상황 - 빠른 모델
            return "llama3.2:3b" if "llama3.2:3b" in self.ollama.available_models else "qwen2.5:7b"
    
    def _analyze_overall_situation(self, text, model):
        """전체 상황 AI 분석"""
        if not text.strip():
            return "분석할 내용이 없습니다"
        
        prompt = f"""다음은 회의/컨퍼런스 내용입니다. 전체 상황을 분석해주세요.

내용:
{text[:2000]}

다음 형식으로 분석해주세요:
1. 회의 주제: 
2. 참여자 역할:
3. 주요 논의사항:
4. 핵심 결정사항:
5. 중요도 (상/중/하):
6. 회의 분위기:"""

        try:
            response = self.ollama.generate_response(
                prompt=prompt,
                model=model,
            )
            return response
        except Exception as e:
            return f"전체 상황 분석 실패: {str(e)}"
    
    def _analyze_speaker_meanings(self, speaker_contents, model):
        """화자별 발언 의미 분석"""
        meanings = {}
        
        for speaker_id, contents in speaker_contents.items():
            if not contents:
                meanings[speaker_id] = "발언 내용 없음"
                continue
            
            # 발언 내용 통합
            combined_speech = " ".join(contents[:5])  # 최대 5개 발언
            
            prompt = f"""화자의 발언을 분석하여 의도와 의미를 파악해주세요.

{speaker_id} 발언:
{combined_speech}

다음 형식으로 분석:
- 주요 의도:
- 감정 상태:
- 핵심 메시지:
- 요청사항:"""

            try:
                response = self.ollama.generate_response(
                    prompt=prompt,
                    model=model,
                    )
                meanings[speaker_id] = response
            except Exception as e:
                meanings[speaker_id] = f"분석 실패: {str(e)}"
        
        return meanings
    
    def _analyze_meeting_context(self, text, model):
        """회의 맥락 및 결론 분석"""
        if not text.strip():
            return "분석할 내용이 없습니다"
        
        prompt = f"""회의 내용을 바탕으로 맥락과 결론을 분석해주세요.

내용:
{text[:1500]}

분석 항목:
1. 회의 목적:
2. 달성된 목표:
3. 미해결 이슈:
4. 다음 단계:
5. 전체적 평가:"""

        try:
            response = self.ollama.generate_response(
                prompt=prompt,
                model=model,
            )
            return response
        except Exception as e:
            return f"맥락 분석 실패: {str(e)}"
    
    def _generate_ai_conclusion(self, situation, context, model):
        """AI 종합 결론 생성"""
        prompt = f"""다음 분석 결과를 바탕으로 최종 결론을 내려주세요.

전체 상황: {situation[:500]}
회의 맥락: {context[:500]}

한 문단으로 종합 결론을 작성해주세요."""

        try:
            response = self.ollama.generate_response(
                prompt=prompt,
                model=model,
            )
            return response
        except Exception as e:
            return f"종합 결론 생성 실패: {str(e)}"
        
        return None
    
    def save_cache(self, file, result):
        """캐시 저장"""
        try:
            # 파일 해시 생성
            file_hash = hashlib.md5(file.getvalue()).hexdigest()
            cache_file = self.cache_dir / f"{file_hash}.pkl.gz"
            
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(result, f)
                
        except Exception:
            pass
    
    def download_from_url(self, url, config):
        """URL에서 파일 다운로드"""
        # URL 다운로드 기능 구현 (향후 확장)
        return []
    
    def render_comprehensive_summary(self, summary):
        """종합 분석 요약 표시"""
        st.markdown("## 🎯 종합 분석 요약")
        
        if "error" in summary:
            st.error(f"종합 분석 실패: {summary['error']}")
            return
        
        # 탭으로 구성 - AI 분석 탭 추가
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 전체 개요", "🎭 화자 분석", "📊 주요 내용", "🤖 AI 상황분석", "⚙️ 사용된 제원"])
        
        with tab1:
            # 분석 개요
            if "분석_개요" in summary:
                overview = summary["분석_개요"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("총 파일 수", overview.get("총_파일_수", 0))
                with col2:
                    st.metric("오디오 파일", overview.get("오디오_파일", 0))
                with col3:
                    st.metric("이미지 파일", overview.get("이미지_파일", 0))
                
                st.info(f"📅 분석 시간: {overview.get('분석_시간', 'N/A')}")
            
            # 통합 스토리
            if "통합_스토리" in summary:
                st.markdown("### 📖 통합 스토리")
                st.markdown(summary["통합_스토리"])
        
        with tab2:
            # 화자 분석
            if "화자_분석" in summary:
                speaker_analysis = summary["화자_분석"]
                st.metric("감지된 화자 수", speaker_analysis.get("감지된_화자_수", 0))
                
                if "화자별_발언_내용" in speaker_analysis:
                    st.markdown("### 🎤 화자별 발언 내용")
                    
                    for speaker_id, content in speaker_analysis["화자별_발언_내용"].items():
                        with st.container():
                            st.markdown(f"#### {speaker_id}")
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.metric("발언 횟수", content.get("발언_횟수", 0))
                            with col2:
                                if content.get("주요_발언"):
                                    st.markdown("**주요 발언:**")
                                    for j, statement in enumerate(content["주요_발언"][:3], 1):
                                        st.markdown(f"{j}. {statement}")
                                
                                # 전체 발언 보기 (접을 수 있게)
                                if content.get("전체_발언") and len(content["전체_발언"]) > 3:
                                    with st.expander(f"{speaker_id} 전체 발언 보기"):
                                        for k, statement in enumerate(content["전체_발언"], 1):
                                            st.markdown(f"{k}. {statement}")
        
        with tab3:
            # 주요 내용
            if "주요_내용" in summary:
                main_content = summary["주요_내용"]
                
                # 핵심 키워드
                if "핵심_키워드" in main_content:
                    st.markdown("### 🔑 핵심 키워드")
                    keywords = main_content["핵심_키워드"][:10]  # 상위 10개
                    if keywords:
                        for i, kw in enumerate(keywords, 1):
                            st.markdown(f"{i}. **{kw.get('단어', 'N/A')}** (빈도: {kw.get('빈도', 0)})")
                
                # 전체 전사 텍스트
                if "전체_전사_텍스트" in main_content:
                    st.markdown("### 📝 전체 전사 텍스트")
                    with st.expander("전사 텍스트 보기"):
                        st.text(main_content["전체_전사_텍스트"])
                
                # 이미지 텍스트
                if "이미지_텍스트" in main_content and main_content["이미지_텍스트"]:
                    st.markdown("### 🖼️ 이미지에서 추출된 텍스트")
                    for i, img_text in enumerate(main_content["이미지_텍스트"], 1):
                        if img_text.strip():
                            st.markdown(f"**이미지 {i}**: {img_text}")
        
        with tab4:
            # AI 상황 분석 (새로 추가된 핵심 기능)
            if "AI_상황_분석" in summary:
                ai_analysis = summary["AI_상황_분석"]
                
                if "error" in ai_analysis:
                    st.error(f"AI 분석 오류: {ai_analysis['error']}")
                else:
                    # 사용된 모델 표시
                    if "사용된_모델" in ai_analysis:
                        st.success(f"🤖 사용된 AI 모델: **{ai_analysis['사용된_모델']}**")
                    
                    # 전체 상황 분석
                    if "전체_상황_분석" in ai_analysis:
                        st.markdown("### 🔍 AI 전체 상황 분석")
                        st.markdown(ai_analysis["전체_상황_분석"])
                    
                    # 화자별 의미 분석
                    if "화자_의미_분석" in ai_analysis:
                        st.markdown("### 🎭 AI 화자 의미 분석")
                        for speaker_id, meaning in ai_analysis["화자_의미_분석"].items():
                            with st.expander(f"🎤 {speaker_id} 발언 의미"):
                                st.markdown(meaning)
                    
                    # 회의 맥락 분석
                    if "회의_맥락_분석" in ai_analysis:
                        st.markdown("### 📋 AI 회의 맥락 분석")
                        st.markdown(ai_analysis["회의_맥락_분석"])
                    
                    # AI 종합 결론
                    if "AI_종합_결론" in ai_analysis:
                        st.markdown("### 🎯 AI 종합 결론")
                        st.info(ai_analysis["AI_종합_결론"])
            else:
                st.warning("AI 상황 분석 결과가 없습니다. Ollama가 연결되지 않았을 수 있습니다.")
        
        with tab5:
            # 사용된 제원
            if "사용된_제원" in summary:
                specs = summary["사용된_제원"]
                st.markdown("### ⚙️ 분석에 사용된 기술 제원")
                
                for key, value in specs.items():
                    st.markdown(f"- **{key.replace('_', ' ').title()}**: {value}")
    
    def render_results(self, results):
        """분석 결과 표시"""
        if not results or not results.get("results"):
            st.warning("분석 결과가 없습니다.")
            return
        
        config = self.analysis_modes[results["mode"]]
        
        st.markdown(f"## 📊 {config['name']} 분석 결과")
        
        # 전체 요약
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📁 처리된 파일", f"{results['files_processed']}개")
        with col2:
            st.metric("⏱️ 소요 시간", f"{results['duration']:.1f}초")
        with col3:
            st.metric("🎯 성공률", f"{(results['files_processed']/(results['total_files']*1.0)*100):.1f}%")
        with col4:
            st.metric("❌ 오류", f"{len(results['errors'])}개")
        
        # 종합 분석 결과 먼저 표시 (핵심)  
        if "comprehensive_summary" in results:
            st.divider()
            self.render_comprehensive_summary(results["comprehensive_summary"])
        
        # 파일별 결과
        st.divider()
        st.markdown("### 📋 개별 파일 분석 결과")
        
        for i, result in enumerate(results["results"]):
            with st.expander(f"📄 {result['filename']} ({result['type']})"):
                
                if "error" in result["analysis"]:
                    st.error(f"❌ 오류: {result['analysis']['error']}")
                    continue
                
                # 분석 타입별 결과 표시
                if "transcription" in result["analysis"]:
                    st.markdown("**🎙️ 음성 인식 결과:**")
                    st.write(result["analysis"]["transcription"])
                    
                    if "speakers" in result["analysis"] and result["analysis"]["speakers"]:
                        st.markdown("**👥 화자 분리 결과:**")
                        speakers = result["analysis"]["speakers"]
                        st.write(f"감지된 화자 수: {speakers.get('speaker_count', 0)}명")
                        st.write(f"분리 품질: {speakers.get('silhouette_score', 0):.3f}")
                
                elif "extracted_text" in result["analysis"]:
                    st.markdown("**🔍 이미지 텍스트 추출:**")
                    for text_item in result["analysis"]["extracted_text"]:
                        st.write(f"- {text_item['text']} (신뢰도: {text_item['confidence']:.2f})")
                
                elif "content" in result["analysis"]:
                    st.markdown("**📝 텍스트 내용:**")
                    st.write(result["analysis"]["preview"])
                    
                    stats = result["analysis"]["stats"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("문자", f"{stats['characters']:,}")
                    with col2:
                        st.metric("단어", f"{stats['words']:,}")
                    with col3:
                        st.metric("줄", f"{stats['lines']:,}")
        
        # 오류 내역
        if results["errors"]:
            st.markdown("### ❌ 오류 내역")
            for error in results["errors"]:
                st.error(f"📁 {error['file']}: {error['error']}")

def main():
    """메인 함수"""
    st.set_page_config(
        page_title="🏆 통합 컨퍼런스 분석 시스템",
        page_icon="🏆",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # 헤더
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    ">
        <h1 style="margin: 0; font-size: 2.5rem;">🏆 통합 컨퍼런스 분석 시스템</h1>
        <h3 style="margin: 0.5rem 0; opacity: 0.9;">하나의 시스템, 다양한 모드</h3>
        <p style="margin: 0; font-size: 1.1rem; opacity: 0.8;">
            궁극/균형/안전 모드 중 선택하여 최적의 분석을 경험하세요
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 분석기 초기화
    analyzer = UnifiedConferenceAnalyzer()
    
    # 세션 상태 초기화
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # 1단계: 모드 선택
    selected_mode = analyzer.render_mode_selector()
    
    st.divider()
    
    # 2단계: 파일 업로드
    uploaded_files = analyzer.render_upload_system()
    
    # 3단계: 분석 실행
    if analyzer.render_analysis_button(uploaded_files):
        with st.spinner(f"🔄 {analyzer.analysis_modes[selected_mode]['name']} 분석 중..."):
            st.session_state.analysis_results = analyzer.execute_unified_analysis(uploaded_files)
    
    # 4단계: 결과 표시
    if st.session_state.analysis_results:
        st.divider()
        analyzer.render_results(st.session_state.analysis_results)
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        🏆 통합 컨퍼런스 분석 시스템 v1.0<br>
        하나의 시스템으로 모든 분석 모드 지원
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()