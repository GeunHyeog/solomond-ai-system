#!/usr/bin/env python3
"""
🤖 완전 자동화 ULTIMATE 컨퍼런스 분석 시스템
- 파일 업로드 → 자동 분석 → 완성된 결과까지 원클릭
- 모든 Yes 처리 자동화
- 오류 없는 완전 자동 실행
"""

import streamlit as st
import os
import sys
import tempfile
import time
import hashlib
import threading
import json
import pickle
import gzip
import io
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import requests

# 고성능 라이브러리
try:
    import whisper
    import librosa
    import easyocr
    import numpy as np
    import cv2
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    ULTIMATE_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ 필수 라이브러리 누락: {e}")
    ULTIMATE_AVAILABLE = False

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from shared.ollama_interface import OllamaInterface
    from core.comprehensive_message_extractor import ComprehensiveMessageExtractor
    from core.ollama_enhanced_extractor import OllamaEnhancedExtractor
    from core.optimized_ai_loader import optimized_loader
    from core.smart_memory_manager import get_memory_stats
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False

# 페이지 설정
st.set_page_config(
    page_title="🤖 완전 자동화 ULTIMATE 분석",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AutoCompleteAnalysisEngine:
    """완전 자동화 분석 엔진"""
    
    def __init__(self):
        self.cache_dir = Path("cache/auto_complete")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_files_dir = Path("user_files")
        self.user_files_dir.mkdir(parents=True, exist_ok=True)
        
        # AI 컴포넌트 초기화 (자동)
        self.auto_initialize_components()
        
        # 자동화 설정
        self.auto_settings = {
            'auto_start_analysis': True,
            'auto_apply_all_features': True,
            'auto_generate_reports': True,
            'auto_save_results': True,
            'skip_confirmations': True
        }
        
        st.success("🤖 완전 자동화 엔진 초기화 완료!")
    
    def auto_initialize_components(self):
        """AI 컴포넌트 자동 초기화"""
        try:
            if COMPONENTS_AVAILABLE:
                self.ollama = OllamaInterface()
                self.message_extractor = ComprehensiveMessageExtractor()
                self.enhanced_extractor = OllamaEnhancedExtractor()
                st.success("✅ 모든 AI 컴포넌트 자동 로드 완료")
            else:
                st.warning("⚠️ 일부 컴포넌트 누락 - 기본 모드로 실행")
        except Exception as e:
            st.error(f"❌ 컴포넌트 초기화 실패: {e}")
    
    def scan_user_files_automatically(self) -> List[Path]:
        """user_files 폴더 자동 스캔"""
        if not self.user_files_dir.exists():
            st.info("📁 user_files 폴더가 없습니다. 수동 업로드를 사용하세요.")
            return []
        
        # 지원 파일 형식
        supported_extensions = [
            '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac',  # 오디오
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif',  # 이미지
            '.mp4', '.avi', '.mov', '.mkv', '.wmv'  # 비디오
        ]
        
        files = []
        for ext in supported_extensions:
            files.extend(self.user_files_dir.glob(f"*{ext}"))
            files.extend(self.user_files_dir.glob(f"*{ext.upper()}"))
        
        return sorted(files)[:10]  # 최대 10개 파일
    
    def auto_analyze_all_files(self, files: List[Path]) -> Dict[str, Any]:
        """모든 파일 완전 자동 분석"""
        if not files:
            return {'error': '분석할 파일이 없습니다.'}
        
        start_time = time.time()
        
        # 진행률 표시
        progress_container = st.container()
        with progress_container:
            st.subheader("🚀 완전 자동 분석 진행 중...")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        all_results = {
            'session_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
            'total_files': len(files),
            'audio_analysis': {},
            'image_analysis': {},
            'video_analysis': {},
            'combined_insights': {},
            'auto_settings': self.auto_settings
        }
        
        # 각 파일 자동 처리
        for i, file_path in enumerate(files):
            try:
                progress = (i + 1) / len(files)
                progress_bar.progress(progress)
                status_text.text(f"📁 분석 중: {file_path.name} ({i+1}/{len(files)})")
                
                # 파일 내용 읽기
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                # 파일 타입별 자동 분석
                file_ext = file_path.suffix.lower()
                if file_ext in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']:
                    result = self.auto_analyze_audio(file_content, file_path.name)
                    if 'error' not in result:
                        all_results['audio_analysis'][file_path.name] = result
                
                elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                    result = self.auto_analyze_image(file_content, file_path.name)
                    if 'error' not in result:
                        all_results['image_analysis'][file_path.name] = result
                
                elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                    result = self.auto_analyze_video(file_content, file_path.name)
                    if 'error' not in result:
                        all_results['video_analysis'][file_path.name] = result
                
                # 짧은 대기 (UI 반응성)
                time.sleep(0.1)
                
            except Exception as e:
                st.warning(f"⚠️ {file_path.name} 처리 중 오류: {e}")
        
        # 자동 종합 분석 (Ollama AI)
        progress_bar.progress(0.9)
        status_text.text("🤖 AI 종합 분석 중...")
        
        if COMPONENTS_AVAILABLE and (all_results['audio_analysis'] or all_results['image_analysis']):
            combined_insights = self.enhanced_extractor.extract_ultimate_insights(all_results)
            all_results['combined_insights'] = combined_insights
        
        # 최종 정리
        all_results['processing_time'] = time.time() - start_time
        all_results['timestamp'] = datetime.now().isoformat()
        all_results['auto_completed'] = True
        
        progress_bar.progress(1.0)
        status_text.text("✅ 완전 자동 분석 완료!")
        
        return all_results
    
    def auto_analyze_audio(self, content: bytes, filename: str) -> Dict[str, Any]:
        """오디오 파일 자동 분석"""
        if not ULTIMATE_AVAILABLE:
            return {'error': 'Ultimate 라이브러리 필요'}
        
        try:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            # Whisper STT (자동 최적 모델 선택)
            model_size = "small" if len(content) < 10*1024*1024 else "base"  # 10MB 기준
            
            with optimized_loader.get_whisper_model(model_size) as whisper_model:
                stt_result = whisper_model.transcribe(tmp_path, language='ko')
            
            # 오디오 특징 추출 (자동)
            y, sr = librosa.load(tmp_path)
            
            # 기본 특징
            duration = len(y) / sr
            rms_energy = np.sqrt(np.mean(y**2))
            zero_crossings = np.sum(np.abs(np.diff(np.sign(y)))) / 2
            
            # 자동 화자 분리 (간단한 버전)
            segments = stt_result.get('segments', [])
            speaker_count = min(max(1, len(segments) // 5), 4)  # 자동 추정
            
            # 정리
            os.unlink(tmp_path)
            
            return {
                'filename': filename,
                'duration': duration,
                'transcript': stt_result['text'],
                'language': stt_result.get('language', 'unknown'),
                'segments_count': len(segments),
                'estimated_speakers': speaker_count,
                'audio_quality': {
                    'rms_energy': float(rms_energy),
                    'zero_crossing_rate': float(zero_crossings / len(y)),
                    'quality_score': min(100, max(0, 100 - (zero_crossings / len(y)) * 1000))
                },
                'auto_analysis': True
            }
            
        except Exception as e:
            return {'error': f'오디오 분석 실패: {str(e)}'}
    
    def auto_analyze_image(self, content: bytes, filename: str) -> Dict[str, Any]:
        """이미지 파일 자동 분석"""
        if not ULTIMATE_AVAILABLE:
            return {'error': 'Ultimate 라이브러리 필요'}
        
        try:
            # 이미지 로드
            nparr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {'error': '이미지 로드 실패'}
            
            # EasyOCR 자동 분석 (한국어+영어)
            with optimized_loader.get_easyocr_reader(['en', 'ko']) as reader:
                ocr_results = reader.readtext(img)
            
            # 텍스트 추출 및 정리
            extracted_text = ""
            text_blocks = []
            confidences = []
            
            for (bbox, text, confidence) in ocr_results:
                text_blocks.append({
                    'text': text,
                    'confidence': confidence
                })
                extracted_text += text + " "
                confidences.append(confidence)
            
            # 이미지 기본 정보
            height, width = img.shape[:2]
            
            return {
                'filename': filename,
                'dimensions': {'width': width, 'height': height},
                'extracted_text': extracted_text.strip(),
                'text_blocks': text_blocks,
                'total_text_blocks': len(text_blocks),
                'avg_confidence': np.mean(confidences) if confidences else 0,
                'auto_analysis': True
            }
            
        except Exception as e:
            return {'error': f'이미지 분석 실패: {str(e)}'}
    
    def auto_analyze_video(self, content: bytes, filename: str) -> Dict[str, Any]:
        """비디오 파일 자동 분석 (기본)"""
        try:
            return {
                'filename': filename,
                'size_bytes': len(content),
                'status': '비디오 분석 준비됨',
                'note': '향후 비디오 분석 기능 확장 예정',
                'auto_analysis': True
            }
        except Exception as e:
            return {'error': f'비디오 분석 실패: {str(e)}'}

def main():
    """메인 애플리케이션 - 완전 자동화"""
    
    # 헤더
    st.markdown("""
    # 🤖 완전 자동화 ULTIMATE 분석 시스템
    ### 파일 업로드 → 자동 분석 → 완성된 결과까지 원클릭!
    """)
    
    # 자동화 상태 표시
    st.info("🎯 **완전 자동화 모드**: 모든 확인 단계를 자동으로 처리합니다.")
    
    # 시스템 상태 체크
    if not ULTIMATE_AVAILABLE:
        st.error("❌ Ultimate 라이브러리가 설치되지 않았습니다.")
        st.stop()
    
    # 자동화 엔진 초기화
    if 'auto_engine' not in st.session_state:
        st.session_state.auto_engine = AutoCompleteAnalysisEngine()
    
    engine = st.session_state.auto_engine
    
    # 탭으로 구분
    tabs = st.tabs(["🤖 완전 자동 분석", "📁 수동 업로드", "⚙️ 설정"])
    
    with tabs[0]:  # 완전 자동 분석
        st.subheader("🚀 user_files 폴더 자동 분석")
        
        # 폴더 파일 자동 스캔
        user_files = engine.scan_user_files_automatically()
        
        if user_files:
            st.success(f"📁 {len(user_files)}개 파일 자동 발견!")
            
            # 파일 목록 표시
            with st.expander("📋 발견된 파일 목록", expanded=True):
                for i, file_path in enumerate(user_files, 1):
                    file_size = file_path.stat().st_size / (1024*1024)  # MB
                    st.write(f"**{i}.** {file_path.name} ({file_size:.2f} MB)")
            
            # 완전 자동 분석 시작 버튼
            if st.button("🤖 완전 자동 분석 시작", type="primary", use_container_width=True):
                
                # 자동 분석 실행
                results = engine.auto_analyze_all_files(user_files)
                
                if 'error' not in results:
                    st.success(f"🎉 완전 자동 분석 완료! (소요시간: {results['processing_time']:.2f}초)")
                    
                    # 결과를 세션에 저장
                    st.session_state.auto_results = results
                    
                    # 결과 표시
                    display_auto_complete_results(results)
                else:
                    st.error(f"❌ 자동 분석 실패: {results['error']}")
        else:
            st.info("📁 user_files 폴더에 분석할 파일이 없습니다.")
            st.markdown("**📋 지원 파일 형식:**")
            st.markdown("- 🎵 **오디오**: mp3, wav, m4a, flac, ogg, aac")
            st.markdown("- 🖼️ **이미지**: jpg, jpeg, png, bmp, tiff, gif")  
            st.markdown("- 🎬 **비디오**: mp4, avi, mov, mkv, wmv")
    
    with tabs[1]:  # 수동 업로드
        st.subheader("📁 수동 파일 업로드")
        
        uploaded_files = st.file_uploader(
            "분석할 파일들을 선택하세요 (다중 선택 가능)",
            type=['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac', 
                  'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif',
                  'mp4', 'avi', 'mov', 'mkv', 'wmv'],
            accept_multiple_files=True,
            help="여러 파일을 한 번에 선택할 수 있습니다"
        )
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)}개 파일 업로드됨!")
            
            # 파일 정보 표시
            total_size = sum(len(f.getvalue()) for f in uploaded_files) / (1024*1024)
            st.write(f"**총 크기**: {total_size:.2f} MB")
            
            if st.button("🚀 업로드된 파일 자동 분석", type="primary", use_container_width=True):
                
                # 업로드된 파일들을 임시로 처리
                temp_results = {
                    'session_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
                    'total_files': len(uploaded_files),
                    'audio_analysis': {},
                    'image_analysis': {},
                    'video_analysis': {},
                    'combined_insights': {}
                }
                
                # 진행률 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                start_time = time.time()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"분석 중: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                    
                    file_content = uploaded_file.getvalue()
                    file_ext = Path(uploaded_file.name).suffix.lower()
                    
                    # 파일 타입별 분석
                    if file_ext in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']:
                        result = engine.auto_analyze_audio(file_content, uploaded_file.name)
                        if 'error' not in result:
                            temp_results['audio_analysis'][uploaded_file.name] = result
                    
                    elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                        result = engine.auto_analyze_image(file_content, uploaded_file.name)
                        if 'error' not in result:
                            temp_results['image_analysis'][uploaded_file.name] = result
                    
                    elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                        result = engine.auto_analyze_video(file_content, uploaded_file.name)
                        if 'error' not in result:
                            temp_results['video_analysis'][uploaded_file.name] = result
                
                # 종합 분석
                if COMPONENTS_AVAILABLE:
                    status_text.text("🤖 AI 종합 분석 중...")
                    combined_insights = engine.enhanced_extractor.extract_ultimate_insights(temp_results)
                    temp_results['combined_insights'] = combined_insights
                
                temp_results['processing_time'] = time.time() - start_time
                temp_results['timestamp'] = datetime.now().isoformat()
                
                progress_bar.progress(1.0)
                status_text.text("✅ 분석 완료!")
                
                st.success(f"🎉 분석 완료! (소요시간: {temp_results['processing_time']:.2f}초)")
                
                # 결과 표시
                display_auto_complete_results(temp_results)
    
    with tabs[2]:  # 설정
        st.subheader("⚙️ 자동화 설정")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 분석 설정")
            auto_audio = st.checkbox("🎵 오디오 자동 분석", value=True)
            auto_image = st.checkbox("🖼️ 이미지 자동 분석", value=True)
            auto_video = st.checkbox("🎬 비디오 자동 분석", value=True)
        
        with col2:
            st.subheader("🤖 AI 설정")
            use_ollama = st.checkbox("🦙 Ollama AI 종합 분석", value=COMPONENTS_AVAILABLE)
            auto_insights = st.checkbox("💡 자동 인사이트 생성", value=True)
            auto_reports = st.checkbox("📊 자동 보고서 생성", value=True)
        
        st.info("💡 모든 설정이 자동으로 활성화되어 있습니다.")

def display_auto_complete_results(results: Dict[str, Any]):
    """완전 자동화 결과 표시"""
    
    st.header("🎉 완전 자동 분석 결과")
    
    # 요약 정보
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 파일", results['total_files'])
    with col2:
        st.metric("오디오 파일", len(results['audio_analysis']))
    with col3:
        st.metric("이미지 파일", len(results['image_analysis']))
    with col4:
        st.metric("처리 시간", f"{results.get('processing_time', 0):.2f}초")
    
    # 탭으로 결과 구분
    result_tabs = st.tabs(["🏆 종합 인사이트", "🎵 오디오 결과", "🖼️ 이미지 결과", "📊 전체 데이터"])
    
    with result_tabs[0]:  # 종합 인사이트
        if 'combined_insights' in results and results['combined_insights']:
            display_combined_insights(results['combined_insights'])
        else:
            st.info("🤖 AI 종합 인사이트가 생성되지 않았습니다.")
    
    with result_tabs[1]:  # 오디오 결과
        if results['audio_analysis']:
            for filename, analysis in results['audio_analysis'].items():
                with st.expander(f"🎵 {filename}", expanded=True):
                    if 'error' not in analysis:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("길이", f"{analysis.get('duration', 0):.1f}초")
                        with col2:
                            st.metric("화자 수", analysis.get('estimated_speakers', 0))
                        with col3:
                            quality = analysis.get('audio_quality', {})
                            st.metric("음질 점수", f"{quality.get('quality_score', 0):.1f}")
                        
                        st.subheader("📝 음성 인식 결과")
                        st.text_area("인식된 텍스트", analysis.get('transcript', ''), height=100)
                    else:
                        st.error(f"❌ 분석 실패: {analysis['error']}")
        else:
            st.info("분석된 오디오 파일이 없습니다.")
    
    with result_tabs[2]:  # 이미지 결과  
        if results['image_analysis']:
            for filename, analysis in results['image_analysis'].items():
                with st.expander(f"🖼️ {filename}", expanded=True):
                    if 'error' not in analysis:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            dims = analysis.get('dimensions', {})
                            st.metric("크기", f"{dims.get('width', 0)}x{dims.get('height', 0)}")
                        with col2:
                            st.metric("텍스트 블록", analysis.get('total_text_blocks', 0))
                        with col3:
                            st.metric("평균 신뢰도", f"{analysis.get('avg_confidence', 0):.2f}")
                        
                        extracted_text = analysis.get('extracted_text', '')
                        if extracted_text:
                            st.subheader("📝 추출된 텍스트")
                            st.text_area("OCR 결과", extracted_text, height=100)
                        else:
                            st.info("추출된 텍스트가 없습니다.")
                    else:
                        st.error(f"❌ 분석 실패: {analysis['error']}")
        else:
            st.info("분석된 이미지 파일이 없습니다.")
    
    with result_tabs[3]:  # 전체 데이터
        st.json(results)

def display_combined_insights(insights: Dict[str, Any]):
    """종합 인사이트 표시"""
    
    if 'executive_summary' in insights:
        st.subheader("📋 경영진 요약")
        st.info(insights['executive_summary'])
    
    if 'key_findings' in insights and insights['key_findings']:
        st.subheader("🔍 핵심 발견사항")
        for i, finding in enumerate(insights['key_findings'], 1):
            st.write(f"**{i}.** {finding}")
    
    if 'business_recommendations' in insights and insights['business_recommendations']:
        st.subheader("💼 비즈니스 권장사항")
        for i, rec in enumerate(insights['business_recommendations'], 1):
            st.success(f"**권장 {i}**: {rec}")
    
    if 'next_actions' in insights and insights['next_actions']:
        st.subheader("🎯 다음 액션")
        for action in insights['next_actions']:
            priority = action.get('priority', '보통')
            if priority == '긴급':
                st.error(f"🔥 **{action.get('action', '')}** - {action.get('description', '')}")
            elif priority == '높음':
                st.warning(f"⚡ **{action.get('action', '')}** - {action.get('description', '')}")
            else:
                st.info(f"📌 **{action.get('action', '')}** - {action.get('description', '')}")

if __name__ == "__main__":
    main()