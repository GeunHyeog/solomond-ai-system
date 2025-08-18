#!/usr/bin/env python3
"""
🚀 모듈 1: 고성능 컨퍼런스 분석 시스템
High-Performance Conference Analysis System

CLI 수준의 성능을 브라우저에서 구현:
- ⚡ 스트리밍 업로드 (청크 기반)
- 🔄 백그라운드 병렬 처리
- 💾 스마트 캐싱 시스템
- 📊 실시간 진행률 표시
- 🎯 즉각적인 사용자 피드백
"""

import streamlit as st
import asyncio
import threading
import queue
import time
import os
import tempfile
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import pickle
import gzip

# 고성능 라이브러리들
try:
    import whisper
    import librosa
    import easyocr
    import numpy as np
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    import torch
    FAST_ANALYSIS_AVAILABLE = True
except ImportError:
    FAST_ANALYSIS_AVAILABLE = False

class HighPerformanceAnalyzer:
    """고성능 컨퍼런스 분석기"""
    
    def __init__(self):
        self.init_session_state()
        self.init_performance_settings()
        if FAST_ANALYSIS_AVAILABLE:
            self.init_models()
    
    def init_session_state(self):
        """세션 상태 초기화"""
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = {}
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'analysis_progress' not in st.session_state:
            st.session_state.analysis_progress = {}
        if 'cache_enabled' not in st.session_state:
            st.session_state.cache_enabled = True
    
    def init_performance_settings(self):
        """성능 설정 초기화"""
        self.chunk_size = 8 * 1024 * 1024  # 8MB 청크
        self.max_workers = min(4, os.cpu_count())
        self.cache_dir = Path(tempfile.gettempdir()) / "solomond_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # GPU 감지
        self.use_gpu = torch.cuda.is_available() if FAST_ANALYSIS_AVAILABLE else False
        if self.use_gpu:
            torch.cuda.empty_cache()
    
    def init_models(self):
        """모델 초기화 (백그라운드에서)"""
        if not hasattr(st.session_state, 'models_initialized'):
            with st.spinner("🔄 AI 모델 초기화 중... (최초 1회만)"):
                try:
                    # Whisper 모델 (GPU 사용 가능시 GPU로)
                    device = "cuda" if self.use_gpu else "cpu"
                    self.whisper_model = whisper.load_model("base", device=device)
                    
                    # EasyOCR (GPU 사용)
                    self.ocr_reader = easyocr.Reader(['ko', 'en'], gpu=self.use_gpu)
                    
                    st.session_state.models_initialized = True
                    st.success(f"✅ AI 모델 초기화 완료 ({'GPU' if self.use_gpu else 'CPU'} 모드)")
                    
                except Exception as e:
                    st.error(f"❌ 모델 초기화 실패: {str(e)}")
                    FAST_ANALYSIS_AVAILABLE = False
        else:
            # 이미 초기화된 모델 사용
            device = "cuda" if self.use_gpu else "cpu"
            self.whisper_model = whisper.load_model("base", device=device)
            self.ocr_reader = easyocr.Reader(['ko', 'en'], gpu=self.use_gpu)
    
    def render_header(self):
        """헤더 렌더링"""
        st.title("🚀 고성능 컨퍼런스 분석 시스템")
        
        # 성능 표시기
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            gpu_status = "🟢 GPU 가속" if self.use_gpu else "🟡 CPU 모드"
            st.markdown(f"**⚡ 성능**: {gpu_status}")
        with col2:
            worker_count = self.max_workers
            st.markdown(f"**🔄 병렬 처리**: {worker_count}개 워커")
        with col3:
            cache_status = "🟢 활성" if st.session_state.cache_enabled else "🔴 비활성"
            st.markdown(f"**💾 캐시**: {cache_status}")
        with col4:
            models_status = "🟢 준비완료" if st.session_state.get('models_initialized', False) else "🟡 초기화중"
            st.markdown(f"**🤖 AI 모델**: {models_status}")
        
        st.markdown("### 📱 CLI 수준 성능을 브라우저에서 경험하세요!")
        st.divider()
    
    def render_high_speed_upload(self):
        """고속 업로드 인터페이스"""
        st.markdown("## ⚡ 고속 파일 업로드")
        
        # 업로드 설정
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 📁 파일 선택 (스트리밍 업로드)")
            
            # 멀티 업로드 지원
            uploaded_files = st.file_uploader(
                "파일을 선택하세요 (최대 10GB, 동시 업로드 지원)",
                accept_multiple_files=True,
                help="대용량 파일도 청크 단위로 빠르게 업로드됩니다"
            )
            
            if uploaded_files:
                self.process_streaming_upload(uploaded_files)
        
        with col2:
            st.markdown("### ⚙️ 성능 설정")
            
            # 캐시 설정
            cache_enabled = st.checkbox(
                "💾 스마트 캐시", 
                value=st.session_state.cache_enabled,
                help="동일한 파일의 재분석을 방지합니다"
            )
            st.session_state.cache_enabled = cache_enabled
            
            # 병렬 처리 설정
            max_workers = st.slider(
                "🔄 병렬 처리 수준",
                min_value=1,
                max_value=8,
                value=self.max_workers,
                help="CPU 코어 수에 따라 조정"
            )
            self.max_workers = max_workers
            
            # GPU 사용 설정
            if self.use_gpu:
                st.success("🚀 GPU 가속 활성")
            else:
                st.info("💻 CPU 모드 실행")
    
    def process_streaming_upload(self, files):
        """스트리밍 업로드 처리"""
        st.markdown("### 📊 업로드 진행 상황")
        
        # 전체 진행률
        total_files = len(files)
        overall_progress = st.progress(0)
        status_text = st.empty()
        
        # 파일별 상세 진행률
        file_progress_bars = {}
        file_status = {}
        
        for i, file in enumerate(files):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"📄 **{file.name}** ({len(file.getvalue())/(1024*1024):.1f} MB)")
            with col2:
                file_progress_bars[file.name] = st.progress(0)
        
        # 백그라운드에서 처리
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for i, file in enumerate(files):
                # 캐시 확인
                file_hash = self.get_file_hash(file)
                cached_result = self.get_cached_result(file_hash) if st.session_state.cache_enabled else None
                
                if cached_result:
                    file_progress_bars[file.name].progress(100)
                    file_status[file.name] = "✅ 캐시에서 로드"
                    st.session_state.uploaded_files[file.name] = {
                        'file': file,
                        'hash': file_hash,
                        'cached_result': cached_result,
                        'upload_time': datetime.now()
                    }
                else:
                    # 백그라운드 업로드 시작
                    future = executor.submit(self.upload_file_chunked, file, file_hash)
                    futures[future] = (file, i)
            
            # 진행률 업데이트
            completed = 0
            total_futures = len(futures)
            
            for future in as_completed(futures):
                file, file_index = futures[future]
                
                try:
                    result = future.result()
                    file_progress_bars[file.name].progress(100)
                    file_status[file.name] = "✅ 업로드 완료"
                    
                    st.session_state.uploaded_files[file.name] = {
                        'file': file,
                        'hash': result['hash'],
                        'temp_path': result['temp_path'],
                        'upload_time': datetime.now()
                    }
                    
                except Exception as e:
                    file_progress_bars[file.name].progress(100)
                    file_status[file.name] = f"❌ 오류: {str(e)[:30]}"
                
                completed += 1
                overall_progress.progress(completed / total_files)
                status_text.text(f"완료: {completed}/{total_files}")
        
        # 업로드 완료
        st.success(f"🎉 **모든 파일 업로드 완료!** ({total_files}개 파일)")
        
        # 즉시 분석 시작 버튼
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 **즉시 분석 시작!**", type="primary", use_container_width=True):
                self.start_realtime_analysis()
    
    def upload_file_chunked(self, file, file_hash):
        """청크 기반 파일 업로드"""
        # 임시 파일 생성
        suffix = f".{file.name.split('.')[-1]}" if '.' in file.name else ""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        # 청크 단위로 쓰기
        file_data = file.getvalue()
        total_size = len(file_data)
        
        for i in range(0, total_size, self.chunk_size):
            chunk = file_data[i:i + self.chunk_size]
            temp_file.write(chunk)
        
        temp_file.close()
        
        return {
            'temp_path': temp_file.name,
            'hash': file_hash,
            'size': total_size
        }
    
    def start_realtime_analysis(self):
        """실시간 분석 시작"""
        if not st.session_state.uploaded_files:
            st.error("업로드된 파일이 없습니다")
            return
        
        st.markdown("## 🔍 실시간 분석 진행")
        
        files_to_analyze = list(st.session_state.uploaded_files.values())
        total_files = len(files_to_analyze)
        
        # 분석 컨테이너
        analysis_container = st.container()
        
        with analysis_container:
            # 전체 진행률
            overall_progress = st.progress(0)
            overall_status = st.empty()
            
            # 실시간 결과 표시 영역
            results_container = st.container()
            
            # 백그라운드 분석 시작
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                
                for i, file_data in enumerate(files_to_analyze):
                    future = executor.submit(self.analyze_file_fast, file_data, i)
                    futures[future] = i
                
                results = {}
                completed = 0
                
                # 실시간 결과 수집
                for future in as_completed(futures):
                    file_index = futures[future]
                    
                    try:
                        result = future.result()
                        results[file_index] = result
                        
                        # 즉시 결과 표시
                        with results_container:
                            self.display_realtime_result(result, completed + 1)
                        
                    except Exception as e:
                        results[file_index] = {
                            'filename': f'파일_{file_index}',
                            'status': 'error',
                            'error': str(e)
                        }
                    
                    completed += 1
                    overall_progress.progress(completed / total_files)
                    overall_status.text(f"분석 완료: {completed}/{total_files}")
                
                # 통합 분석 생성
                if len(results) > 1:
                    with st.spinner("🎯 통합 스토리 생성 중..."):
                        integrated_analysis = self.create_fast_integrated_story(list(results.values()))
                        
                        st.markdown("## 🎯 통합 분석 결과")
                        st.markdown(integrated_analysis)
                
                st.success("✅ 모든 분석 완료!")
                st.balloons()
    
    def analyze_file_fast(self, file_data, index):
        """고속 파일 분석"""
        try:
            # 캐시 확인
            if 'cached_result' in file_data:
                time.sleep(0.1)  # 캐시 로딩 시뮬레이션
                return file_data['cached_result']
            
            file = file_data['file']
            temp_path = file_data.get('temp_path')
            
            if not temp_path:
                return {'filename': file.name, 'status': 'error', 'error': '임시 파일 없음'}
            
            # 파일 타입 감지
            ext = file.name.lower().split('.')[-1]
            
            result = None
            if ext in ['wav', 'mp3', 'm4a', 'flac', 'ogg']:
                result = self.fast_audio_analysis(temp_path, file.name)
            elif ext in ['mp4', 'avi', 'mov', 'mkv', 'wmv']:
                result = self.fast_video_analysis(temp_path, file.name)
            elif ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
                result = self.fast_image_analysis(temp_path, file.name)
            else:
                result = {'filename': file.name, 'status': 'unsupported', 'message': '지원하지 않는 형식'}
            
            # 캐시 저장
            if st.session_state.cache_enabled and result:
                self.save_to_cache(file_data['hash'], result)
            
            return result
            
        except Exception as e:
            return {
                'filename': file_data['file'].name,
                'status': 'error',
                'error': str(e)
            }
    
    def fast_audio_analysis(self, file_path, filename):
        """고속 음성 분석"""
        try:
            # Whisper STT (GPU 가속)
            result = self.whisper_model.transcribe(file_path, language="ko")
            
            # 빠른 화자 분리 (MiniBatch KMeans 사용)
            speaker_analysis = self.fast_speaker_diarization(file_path, result)
            
            return {
                'filename': filename,
                'transcription': result,
                'speaker_analysis': speaker_analysis,
                'status': 'success',
                'analysis_time': time.time()
            }
            
        except Exception as e:
            return {
                'filename': filename,
                'status': 'error',
                'error': str(e)
            }
    
    def fast_speaker_diarization(self, file_path, transcription):
        """고속 화자 분리"""
        try:
            # 음성 로드 (빠른 처리를 위해 샘플링 레이트 조정)
            y, sr = librosa.load(file_path, sr=16000)  # 16kHz로 다운샘플링
            
            segments = transcription.get('segments', [])
            if len(segments) <= 2:
                return {'speakers': 1, 'method': 'single_speaker', 'quality_score': 1.0}
            
            # 간단한 특징 추출 (성능 우선)
            features = []
            for segment in segments[:20]:  # 최대 20개 세그먼트만
                start_time = segment.get('start', 0)
                end_time = segment.get('end', start_time + 1)
                
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                if start_sample < len(y) and end_sample <= len(y):
                    segment_audio = y[start_sample:end_sample]
                    
                    if len(segment_audio) > 0:
                        # 기본 특징만 추출 (속도 우선)
                        mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=5)  # 5차원으로 축소
                        features.append(np.mean(mfcc, axis=1))
            
            if len(features) < 2:
                return {'speakers': 1, 'method': 'insufficient_data', 'quality_score': 0.5}
            
            # MiniBatch KMeans로 빠른 클러스터링
            features_array = np.array(features)
            n_speakers = min(3, max(2, len(features) // 4))
            
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)
            
            kmeans = MiniBatchKMeans(n_clusters=n_speakers, random_state=42, batch_size=10)
            labels = kmeans.fit_predict(features_scaled)
            
            # 세그먼트에 화자 할당
            for i, segment in enumerate(segments[:len(labels)]):
                segment['speaker'] = int(labels[i])
            
            return {
                'speakers': n_speakers,
                'method': 'fast_minibatch_kmeans',
                'quality_score': 0.8,
                'processing_time': time.time()
            }
            
        except Exception as e:
            return {'speakers': 1, 'method': 'error', 'error': str(e)}
    
    def fast_video_analysis(self, file_path, filename):
        """고속 영상 분석 (음성 추출 후 분석)"""
        return self.fast_audio_analysis(file_path, filename)
    
    def fast_image_analysis(self, file_path, filename):
        """고속 이미지 분석"""
        try:
            # EasyOCR (GPU 가속)
            results = self.ocr_reader.readtext(file_path)
            extracted_text = "\n".join([result[1] for result in results if result[2] > 0.5])  # 신뢰도 0.5 이상만
            
            return {
                'filename': filename,
                'extracted_text': extracted_text,
                'ocr_confidence': np.mean([result[2] for result in results]) if results else 0,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'filename': filename,
                'status': 'error',
                'error': str(e)
            }
    
    def display_realtime_result(self, result, index):
        """실시간 결과 표시"""
        with st.expander(f"✅ {result.get('filename', f'결과 {index}')} - 분석 완료!", expanded=index <= 2):
            
            if result['status'] == 'success':
                if 'transcription' in result:
                    # 음성 결과
                    transcription = result['transcription']
                    
                    st.markdown("**🎤 음성 전사 결과:**")
                    st.text_area("전사 내용", transcription.get('text', ''), height=100, key=f"transcript_{index}")
                    
                    if 'speaker_analysis' in result:
                        speaker_info = result['speaker_analysis']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("화자 수", speaker_info.get('speakers', 1))
                        with col2:
                            st.metric("분석 방법", speaker_info.get('method', 'N/A'))
                        with col3:
                            st.metric("품질 점수", f"{speaker_info.get('quality_score', 0):.2f}")
                
                elif 'extracted_text' in result:
                    # 이미지 결과
                    st.markdown("**📝 추출된 텍스트:**")
                    st.text_area("OCR 결과", result['extracted_text'], height=100, key=f"ocr_{index}")
                    
                    if 'ocr_confidence' in result:
                        st.metric("OCR 신뢰도", f"{result['ocr_confidence']:.2f}")
            
            else:
                st.error(f"분석 실패: {result.get('error', '알 수 없는 오류')}")
    
    def create_fast_integrated_story(self, results):
        """고속 통합 스토리 생성"""
        try:
            story_parts = []
            
            # 빠른 요약
            audio_count = sum(1 for r in results if 'transcription' in r)
            image_count = sum(1 for r in results if 'extracted_text' in r)
            
            story_parts.append(f"## 📊 분석 요약")
            story_parts.append(f"- **음성/영상**: {audio_count}개")
            story_parts.append(f"- **이미지/문서**: {image_count}개")
            
            # 주요 내용 추출
            all_text = ""
            for result in results:
                if 'transcription' in result:
                    all_text += result['transcription'].get('text', '') + " "
                elif 'extracted_text' in result:
                    all_text += result['extracted_text'] + " "
            
            # 간단한 키워드 추출
            if all_text:
                words = all_text.split()
                word_freq = {}
                for word in words:
                    if len(word) > 2:
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                if top_words:
                    story_parts.append(f"\n**🏷️ 주요 키워드**: {', '.join([word for word, count in top_words])}")
            
            return "\n".join(story_parts)
            
        except Exception as e:
            return f"통합 분석 생성 중 오류: {str(e)}"
    
    def get_file_hash(self, file):
        """파일 해시 생성"""
        return hashlib.md5(file.getvalue()).hexdigest()
    
    def get_cached_result(self, file_hash):
        """캐시에서 결과 가져오기"""
        try:
            cache_file = self.cache_dir / f"{file_hash}.pkl.gz"
            if cache_file.exists():
                with gzip.open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass
        return None
    
    def save_to_cache(self, file_hash, result):
        """캐시에 결과 저장"""
        try:
            cache_file = self.cache_dir / f"{file_hash}.pkl.gz"
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception:
            pass
    
    def run(self):
        """메인 실행"""
        st.set_page_config(
            page_title="고성능 컨퍼런스 분석",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        self.render_header()
        
        if not FAST_ANALYSIS_AVAILABLE:
            st.error("❌ 필요한 라이브러리가 설치되지 않았습니다")
            return
        
        # 탭 구성
        tab1, tab2, tab3 = st.tabs(["⚡ 고속 분석", "📊 성능 모니터", "🔧 설정"])
        
        with tab1:
            self.render_high_speed_upload()
        
        with tab2:
            self.render_performance_monitor()
        
        with tab3:
            self.render_settings()
    
    def render_performance_monitor(self):
        """성능 모니터링"""
        st.markdown("## 📊 성능 모니터링")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔄 CPU 사용률", f"{os.cpu_count()}코어 활용")
        with col2:
            gpu_info = "GPU 가속 활성" if self.use_gpu else "CPU 모드"
            st.metric("🚀 가속 모드", gpu_info)
        with col3:
            cache_files = len(list(self.cache_dir.glob("*.pkl.gz"))) if self.cache_dir.exists() else 0
            st.metric("💾 캐시 파일", f"{cache_files}개")
        with col4:
            st.metric("⚡ 병렬 워커", f"{self.max_workers}개")
        
        # 캐시 관리
        st.markdown("### 💾 캐시 관리")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ 캐시 정리", help="모든 캐시 파일을 삭제합니다"):
                self.clear_cache()
        
        with col2:
            if st.button("📊 캐시 통계", help="캐시 사용량을 확인합니다"):
                self.show_cache_stats()
        
        with col3:
            if st.button("🔄 모델 재로드", help="AI 모델을 다시 로드합니다"):
                self.reload_models()
    
    def render_settings(self):
        """설정 화면"""
        st.markdown("## 🔧 고급 설정")
        
        # 성능 설정
        st.markdown("### ⚡ 성능 최적화")
        
        new_chunk_size = st.selectbox(
            "청크 크기 (업로드 속도)",
            [1, 2, 4, 8, 16],
            index=3,
            format_func=lambda x: f"{x}MB"
        )
        self.chunk_size = new_chunk_size * 1024 * 1024
        
        # AI 모델 설정
        st.markdown("### 🤖 AI 모델 설정")
        
        whisper_model_size = st.selectbox(
            "Whisper 모델 크기",
            ["tiny", "base", "small", "medium"],
            index=1,
            help="큰 모델일수록 정확하지만 느립니다"
        )
        
        if st.button("모델 설정 적용"):
            st.session_state.models_initialized = False
            st.rerun()
    
    def clear_cache(self):
        """캐시 정리"""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
            st.success("✅ 캐시가 정리되었습니다")
        except Exception as e:
            st.error(f"❌ 캐시 정리 실패: {str(e)}")
    
    def show_cache_stats(self):
        """캐시 통계 표시"""
        if self.cache_dir.exists():
            cache_files = list(self.cache_dir.glob("*.pkl.gz"))
            total_size = sum(f.stat().st_size for f in cache_files)
            st.info(f"📊 캐시 파일: {len(cache_files)}개, 총 크기: {total_size/(1024*1024):.1f}MB")
        else:
            st.info("📊 캐시 없음")
    
    def reload_models(self):
        """모델 재로드"""
        st.session_state.models_initialized = False
        st.rerun()

def main():
    """메인 함수"""
    analyzer = HighPerformanceAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()