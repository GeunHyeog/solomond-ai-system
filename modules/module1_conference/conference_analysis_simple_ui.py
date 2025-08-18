#!/usr/bin/env python3
"""
🎯 모듈 1: 간단하고 직관적인 컨퍼런스 분석 시스템
Simple and Intuitive Conference Analysis System

사용자 워크플로우:
1️⃣ 파일 업로드 → 2️⃣ 분석 실행 → 3️⃣ 결과 확인
"""

import streamlit as st
import os
import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

# 기존 분석 엔진 import
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from modules.module1_conference.conference_analysis import ConferenceAnalysisSystem
    ANALYSIS_ENGINE_AVAILABLE = True
except ImportError:
    ANALYSIS_ENGINE_AVAILABLE = False

class SimpleConferenceAnalyzer:
    """간단하고 직관적인 컨퍼런스 분석기"""
    
    def __init__(self):
        self.init_session_state()
        if ANALYSIS_ENGINE_AVAILABLE:
            self.analysis_engine = ConferenceAnalysisSystem()
    
    def init_session_state(self):
        """세션 상태 초기화"""
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 1
    
    def render_header(self):
        """헤더 렌더링"""
        st.title("🎯 컨퍼런스 분석 시스템")
        st.markdown("### 📝 음성 → 텍스트 → 화자별 분석까지 원클릭으로!")
        
        # 진행 단계 표시
        col1, col2, col3 = st.columns(3)
        
        step_1_status = "✅" if st.session_state.current_step >= 1 else "1️⃣"
        step_2_status = "✅" if st.session_state.current_step >= 2 else "2️⃣" 
        step_3_status = "✅" if st.session_state.current_step >= 3 else "3️⃣"
        
        with col1:
            st.markdown(f"### {step_1_status} 파일 업로드")
            if st.session_state.current_step == 1:
                st.markdown("👈 **현재 단계**")
                
        with col2:
            st.markdown(f"### {step_2_status} 분석 실행")
            if st.session_state.current_step == 2:
                st.markdown("👈 **현재 단계**")
                
        with col3:
            st.markdown(f"### {step_3_status} 결과 확인")
            if st.session_state.current_step == 3:
                st.markdown("👈 **현재 단계**")
        
        st.divider()
    
    def render_step_1_upload(self):
        """1단계: 파일 업로드"""
        if st.session_state.current_step != 1:
            return
            
        st.markdown("## 1️⃣ 분석할 콘텐츠를 선택하세요")
        
        # 업로드 방식 선택
        upload_method = st.radio(
            "📥 업로드 방식을 선택하세요:",
            ["📁 파일 업로드", "🌐 URL 링크", "📂 폴더 업로드", "✏️ 직접 입력"],
            horizontal=True,
            help="다양한 방식으로 콘텐츠를 업로드할 수 있습니다"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if upload_method == "📁 파일 업로드":
                self.render_file_upload()
            elif upload_method == "🌐 URL 링크":
                self.render_url_upload()
            elif upload_method == "📂 폴더 업로드":
                self.render_folder_upload()
            else:  # 직접 입력
                self.render_direct_input()
                
        with col2:
            self.render_upload_info(upload_method)
    
    def render_file_upload(self):
        """파일 업로드 인터페이스"""
        st.markdown("### 📁 파일 선택 (모든 형식 지원)")
        
        # 파일 타입 선택
        file_type = st.selectbox(
            "파일 형식을 선택하세요:",
            [
                "🎬 영상 파일 (MP4, AVI, MOV, MKV, WMV)",
                "🎤 음성 파일 (WAV, MP3, M4A, FLAC, OGG)",
                "🖼️ 이미지 파일 (PNG, JPG, JPEG, GIF, BMP)",
                "📄 문서 파일 (PDF, DOCX, PPTX, TXT)",
                "🗂️ 모든 파일 (자동 감지)"
            ],
            help="분석하고 싶은 파일의 형식을 먼저 선택해주세요"
        )
        
        # 고용량 파일 안내
        st.info("💾 **고용량 파일 지원**: 최대 5GB까지 업로드 가능 (진행률 표시됨)")
        
        # 파일 확장자 매핑
        file_extensions = {
            "🎬 영상": ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'],
            "🎤 음성": ['wav', 'mp3', 'm4a', 'flac', 'ogg', 'aac', 'wma'],
            "🖼️ 이미지": ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'svg'],
            "📄 문서": ['pdf', 'docx', 'pptx', 'txt', 'rtf', 'odt'],
            "🗂️ 모든": None  # 모든 파일 허용
        }
        
        # 선택된 파일 타입에 따른 확장자
        file_key = next(key for key in file_extensions.keys() if key in file_type)
        allowed_extensions = file_extensions[file_key]
        
        # 파일 업로드
        uploaded_files = st.file_uploader(
            f"{file_type} 선택 (여러 파일 가능, 최대 5GB)",
            type=allowed_extensions,
            accept_multiple_files=True,
            help="고용량 파일도 지원됩니다. 업로드 중 진행률이 표시됩니다."
        )
        
        if uploaded_files:
            self.process_uploaded_files(uploaded_files, file_type)
    
    def render_url_upload(self):
        """URL 업로드 인터페이스"""
        st.markdown("### 🌐 URL 링크 분석")
        
        url_type = st.selectbox(
            "URL 타입을 선택하세요:",
            [
                "🎥 YouTube 동영상",
                "🎵 SoundCloud 음성",
                "📰 웹페이지 (뉴스, 블로그)",
                "📄 온라인 문서 (PDF, Google Docs)",
                "🔗 일반 URL (자동 감지)"
            ]
        )
        
        url_input = st.text_input(
            "URL을 입력하세요:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="YouTube, 웹페이지, 온라인 문서 등 다양한 URL을 지원합니다"
        )
        
        if url_input:
            if st.button("🔍 URL 분석 시작", type="primary"):
                self.process_url_content(url_input, url_type)
    
    def render_folder_upload(self):
        """폴더 업로드 인터페이스"""
        st.markdown("### 📂 폴더 전체 업로드")
        
        st.warning("💡 **폴더 업로드 방법**: ZIP 파일로 압축해서 업로드해주세요")
        
        zip_file = st.file_uploader(
            "ZIP 폴더 파일을 선택하세요:",
            type=['zip'],
            help="폴더를 ZIP으로 압축한 후 업로드하면 내부 파일들을 자동으로 분석합니다"
        )
        
        if zip_file:
            self.process_zip_folder(zip_file)
    
    def render_direct_input(self):
        """직접 입력 인터페이스"""
        st.markdown("### ✏️ 텍스트 직접 입력")
        
        input_type = st.selectbox(
            "입력 형식:",
            ["📝 회의록", "💬 대화 기록", "📄 일반 텍스트", "🎭 화자별 대화"]
        )
        
        if "화자별" in input_type:
            st.markdown("**화자별 대화 형식 예시:**")
            st.code("""
화자1: 안녕하세요, 오늘 회의를 시작하겠습니다.
화자2: 네, 준비되었습니다.
화자1: 첫 번째 안건은 프로젝트 진행상황입니다.
            """)
        
        text_content = st.text_area(
            "텍스트를 입력하세요:",
            height=200,
            placeholder="분석할 텍스트 내용을 입력해주세요...",
            help="회의록, 대화 기록 등을 직접 입력할 수 있습니다"
        )
        
        if text_content.strip():
            self.process_direct_text(text_content, input_type)
    
    def render_upload_info(self, upload_method):
        """업로드 방식별 정보 표시"""
        st.markdown("### ℹ️ 지원 기능")
        
        if upload_method == "📁 파일 업로드":
            st.markdown("""
            **🎬 영상 분석:**
            - 🎤 음성 추출 및 전사
            - 🎭 화자 분리 분석
            - 📊 시간대별 분석
            
            **🎤 음성 분석:**
            - 📝 음성 → 텍스트 변환
            - 🎭 화자별 구분
            - 📈 발언 비율 분석
            
            **📄 문서 분석:**
            - 📝 텍스트 추출 (PDF, DOCX)
            - 🔍 내용 요약 및 분석
            - 📊 구조화된 정보 추출
            """)
            
        elif upload_method == "🌐 URL 링크":
            st.markdown("""
            **🎥 YouTube:**
            - 자동 자막 추출
            - 음성 다운로드 및 분석
            - 댓글 분석 (선택사항)
            
            **📰 웹페이지:**
            - 본문 텍스트 추출
            - 구조화된 내용 분석
            - 요약 및 키워드 추출
            """)
            
        elif upload_method == "📂 폴더 업로드":
            st.markdown("""
            **📂 일괄 처리:**
            - 여러 파일 동시 분석
            - 파일 타입별 자동 분류
            - 통합 리포트 생성
            - 진행률 실시간 표시
            """)
            
        else:  # 직접 입력
            st.markdown("""
            **✏️ 텍스트 분석:**
            - 화자별 발언 구분
            - 대화 패턴 분석
            - 핵심 주제 추출
            - 감정 분석 (베타)
            """)
    
    def process_uploaded_files(self, files, file_type):
        """업로드된 파일 처리"""
        total_size = sum(len(file.getvalue()) for file in files)
        total_size_mb = total_size / (1024 * 1024)
        
        st.success(f"✅ {len(files)}개 파일 업로드 완료!")
        st.info(f"📊 총 용량: {total_size_mb:.1f} MB")
        
        # 파일 목록 표시
        with st.expander("📋 업로드된 파일 목록", expanded=True):
            for i, file in enumerate(files):
                file_size = len(file.getvalue()) / (1024 * 1024)
                file_icon = self.get_file_icon(file.name)
                st.markdown(f"{file_icon} **{file.name}** ({file_size:.1f} MB)")
        
        # 세션에 저장
        st.session_state.uploaded_files = {
            'files': files,
            'type': file_type,
            'method': 'file_upload',
            'total_size_mb': total_size_mb,
            'upload_time': datetime.now()
        }
        
        self.show_next_step_button()
    
    def process_url_content(self, url, url_type):
        """URL 콘텐츠 처리"""
        with st.spinner(f"🔍 {url} 분석 중..."):
            # URL 유효성 검사
            if not url.startswith(('http://', 'https://')):
                st.error("❌ 올바른 URL 형식이 아닙니다. http:// 또는 https://로 시작해야 합니다.")
                return
            
            # 임시로 URL 정보 저장 (실제 다운로드는 분석 단계에서)
            st.session_state.uploaded_files = {
                'url': url,
                'type': url_type,
                'method': 'url_upload',
                'upload_time': datetime.now()
            }
            
            st.success(f"✅ URL이 등록되었습니다: {url}")
            self.show_next_step_button()
    
    def process_zip_folder(self, zip_file):
        """ZIP 폴더 처리"""
        import zipfile
        import io
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_file.getvalue())) as z:
                file_list = z.namelist()
                
            st.success(f"✅ ZIP 파일 분석 완료!")
            st.info(f"📂 내부 파일 {len(file_list)}개 발견")
            
            # 파일 목록 표시
            with st.expander("📋 ZIP 내부 파일 목록", expanded=True):
                for file_name in file_list[:20]:  # 최대 20개만 표시
                    if not file_name.endswith('/'):  # 폴더 제외
                        file_icon = self.get_file_icon(file_name)
                        st.markdown(f"{file_icon} {file_name}")
                
                if len(file_list) > 20:
                    st.markdown(f"... 외 {len(file_list) - 20}개 파일")
            
            # 세션에 저장
            st.session_state.uploaded_files = {
                'zip_file': zip_file,
                'file_list': file_list,
                'type': 'ZIP 폴더',
                'method': 'folder_upload',
                'upload_time': datetime.now()
            }
            
            self.show_next_step_button()
            
        except Exception as e:
            st.error(f"❌ ZIP 파일 처리 중 오류: {str(e)}")
    
    def process_direct_text(self, text, input_type):
        """직접 입력 텍스트 처리"""
        word_count = len(text.split())
        char_count = len(text)
        
        st.success(f"✅ 텍스트 입력 완료!")
        st.info(f"📊 단어 수: {word_count}개, 글자 수: {char_count}자")
        
        # 세션에 저장
        st.session_state.uploaded_files = {
            'text_content': text,
            'type': input_type,
            'method': 'direct_input',
            'word_count': word_count,
            'char_count': char_count,
            'upload_time': datetime.now()
        }
        
        self.show_next_step_button()
    
    def get_file_icon(self, filename):
        """파일 확장자별 아이콘 반환"""
        ext = filename.lower().split('.')[-1]
        
        icon_map = {
            # 영상
            'mp4': '🎬', 'avi': '🎬', 'mov': '🎬', 'mkv': '🎬', 'wmv': '🎬',
            # 음성
            'wav': '🎤', 'mp3': '🎵', 'm4a': '🎵', 'flac': '🎵', 'ogg': '🎵',
            # 이미지
            'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️', 'bmp': '🖼️',
            # 문서
            'pdf': '📄', 'docx': '📝', 'pptx': '📊', 'txt': '📄',
        }
        
        return icon_map.get(ext, '📁')
    
    def show_next_step_button(self):
        """다음 단계 버튼 표시"""
        if st.button("➡️ 다음 단계: 분석 실행", type="primary", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()
                uploaded_files = st.file_uploader(
                    "음성 파일을 선택하세요 (여러 파일 가능)",
                    type=['wav', 'mp3', 'm4a', 'mp4'],
                    accept_multiple_files=True,
                    help="회의 녹음, 인터뷰, 대화 등의 음성 파일을 업로드하세요"
                )
                file_category = "audio"
                
            elif "이미지" in file_type:
                uploaded_files = st.file_uploader(
                    "이미지 파일을 선택하세요 (여러 파일 가능)",
                    type=['png', 'jpg', 'jpeg'],
                    accept_multiple_files=True,
                    help="프레젠테이션 슬라이드, 화이트보드, 문서 등의 이미지를 업로드하세요"
                )
                file_category = "image"
                
            else:  # 텍스트
                uploaded_files = st.file_uploader(
                    "텍스트 파일을 선택하세요",
                    type=['txt'],
                    help="회의록, 대화록 등의 텍스트 파일을 업로드하세요"
                )
                if uploaded_files:
                    uploaded_files = [uploaded_files]  # 리스트로 변환
                file_category = "text"
        
        with col2:
            st.markdown("### ℹ️ 분석 가능한 내용")
            
            if "음성" in file_type:
                st.markdown("""
                **🎤 음성 분석으로 얻을 수 있는 정보:**
                - 📝 음성 → 텍스트 변환
                - 🎭 화자별 구분 및 분석
                - ⏰ 발언 시간 및 순서
                - 📊 화자별 발언 비율
                - 🔍 주요 키워드 추출
                """)
            elif "이미지" in file_type:
                st.markdown("""
                **🖼️ 이미지 분석으로 얻을 수 있는 정보:**
                - 📝 이미지 내 텍스트 추출 (OCR)
                - 📋 문서 내용 분석
                - 📊 표와 차트 인식
                - 🔍 주요 정보 요약
                """)
            else:
                st.markdown("""
                **📄 텍스트 분석으로 얻을 수 있는 정보:**
                - 🎭 화자별 발언 구분
                - 📊 대화 흐름 분석
                - 🔍 핵심 주제 추출
                - 📝 요약 및 정리
                """)
        
        # 업로드된 파일 처리
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)}개 파일이 업로드되었습니다!")
            
            # 파일 정보 표시
            for i, file in enumerate(uploaded_files):
                file_size = len(file.getvalue()) / (1024 * 1024)  # MB
                st.markdown(f"📄 **{file.name}** ({file_size:.1f} MB)")
            
            # 세션에 저장
            st.session_state.uploaded_files = {
                'files': uploaded_files,
                'category': file_category,
                'upload_time': datetime.now()
            }
            
            # 다음 단계로 버튼
            if st.button("➡️ 다음 단계: 분석 실행", type="primary", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
        else:
            st.info("👆 위에서 파일을 선택해주세요")
    
    def render_step_2_analysis(self):
        """2단계: 분석 실행"""
        if st.session_state.current_step != 2:
            return
            
        st.markdown("## 2️⃣ 분석을 실행하세요")
        
        if not st.session_state.uploaded_files:
            st.error("파일이 업로드되지 않았습니다. 1단계로 돌아가주세요.")
            if st.button("⬅️ 1단계로 돌아가기"):
                st.session_state.current_step = 1
                st.rerun()
            return
        
        uploaded_data = st.session_state.uploaded_files
        files = uploaded_data['files']
        category = uploaded_data['category']
        
        # 분석 설정
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### 📁 업로드된 파일 ({len(files)}개)")
            for file in files:
                st.markdown(f"- 📄 {file.name}")
            
            st.markdown("### ⚙️ 분석 설정")
            
            if category == "audio":
                st.markdown("**🎤 음성 분석 옵션:**")
                enable_speaker_diarization = st.checkbox(
                    "🎭 화자 분리 분석", 
                    value=True, 
                    help="여러 사람이 말하는 경우 화자별로 구분하여 분석합니다"
                )
                
                language = st.selectbox(
                    "음성 언어",
                    ["auto", "ko", "en", "ja", "zh"],
                    format_func=lambda x: {
                        "auto": "🌐 자동 감지",
                        "ko": "🇰🇷 한국어", 
                        "en": "🇺🇸 영어",
                        "ja": "🇯🇵 일본어",
                        "zh": "🇨🇳 중국어"
                    }[x]
                )
                
        with col2:
            st.markdown("### 🚀 분석 실행")
            st.markdown("준비가 완료되었습니다!")
            
            if st.button("🔍 지금 분석 시작!", type="primary", use_container_width=True):
                self.run_analysis(files, category)
        
        # 하단 버튼
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ 이전 단계"):
                st.session_state.current_step = 1
                st.rerun()
    
    def run_analysis(self, files, category):
        """분석 실행"""
        if not ANALYSIS_ENGINE_AVAILABLE:
            st.error("❌ 분석 엔진을 사용할 수 없습니다.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 분석을 시작합니다...")
            progress_bar.progress(10)
            
            results = []
            
            for i, file in enumerate(files):
                file_progress = (i + 1) / len(files)
                
                status_text.text(f"🔍 {file.name} 분석 중... ({i+1}/{len(files)})")
                progress_bar.progress(int(20 + file_progress * 60))
                
                # 임시 파일로 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    if category == "audio":
                        result = self.analysis_engine._process_audio_full_quality(tmp_path, file.name)
                    elif category == "image":
                        result = self.analysis_engine._process_image(tmp_path, file.name)
                    else:  # text
                        content = file.getvalue().decode('utf-8')
                        result = self.analysis_engine._process_text(content, file.name)
                    
                    results.append(result)
                    
                finally:
                    # 임시 파일 삭제
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            
            status_text.text("✅ 분석이 완료되었습니다!")
            progress_bar.progress(100)
            
            # 결과 저장
            st.session_state.analysis_results = {
                'files_analyzed': len(files),
                'category': category,
                'results': results,
                'analysis_time': datetime.now()
            }
            
            time.sleep(1)
            st.session_state.current_step = 3
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ 분석 중 오류가 발생했습니다: {str(e)}")
            status_text.text("❌ 분석 실패")
    
    def render_step_3_results(self):
        """3단계: 결과 확인"""
        if st.session_state.current_step != 3:
            return
            
        st.markdown("## 3️⃣ 분석 결과")
        
        if not st.session_state.analysis_results:
            st.error("분석 결과가 없습니다.")
            return
        
        results_data = st.session_state.analysis_results
        
        # 결과 요약
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📁 분석 파일", f"{results_data['files_analyzed']}개")
        with col2:
            st.metric("📊 분석 유형", results_data['category'])
        with col3:
            st.metric("⏰ 분석 시간", results_data['analysis_time'].strftime("%H:%M"))
        with col4:
            st.metric("✅ 상태", "완료")
        
        st.divider()
        
        # 결과 내용
        for i, result in enumerate(results_data['results']):
            with st.expander(f"📄 {result.get('filename', f'파일 {i+1}')} 분석 결과", expanded=i==0):
                
                if results_data['category'] == 'audio':
                    self.render_audio_result(result)
                elif results_data['category'] == 'image':
                    self.render_image_result(result)
                else:
                    self.render_text_result(result)
        
        # 하단 액션 버튼들
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 새로운 분석", use_container_width=True):
                # 세션 초기화
                st.session_state.uploaded_files = []
                st.session_state.analysis_results = None
                st.session_state.current_step = 1
                st.rerun()
        
        with col2:
            if st.button("📥 결과 다운로드", use_container_width=True):
                self.download_results()
        
        with col3:
            if st.button("📊 상세 분석", use_container_width=True):
                st.info("상세 분석 기능은 준비 중입니다.")
    
    def render_audio_result(self, result):
        """음성 분석 결과 렌더링"""
        # 전사 결과
        if 'transcription' in result and result['transcription']:
            transcription = result['transcription']
            
            # 화자별 대화 내용
            if 'segments' in transcription:
                st.markdown("### 🎭 화자별 대화 내용")
                
                for segment in transcription['segments']:
                    speaker_id = segment.get('speaker', 0)
                    speaker_name = f"화자 {speaker_id + 1}"
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', 0)
                    text = segment.get('text', '').strip()
                    
                    if text:
                        # 화자별 색상
                        colors = ['🔵', '🔴', '🟢', '🟡', '🟣', '🟠']
                        color = colors[speaker_id % len(colors)]
                        
                        st.markdown(f"""
                        <div style="margin: 8px 0; padding: 12px; border-left: 4px solid #2196F3; background: linear-gradient(90deg, rgba(33,150,243,0.1) 0%, rgba(255,255,255,0) 100%);">
                            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                <span style="font-weight: bold; color: #1976D2;">{color} {speaker_name}</span>
                                <span style="margin-left: 10px; font-size: 0.85em; color: #666;">[{start_time:.1f}s - {end_time:.1f}s]</span>
                            </div>
                            <div style="font-size: 1.05em; line-height: 1.4;">{text}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # 화자 분석 통계
        if 'speaker_analysis' in result and result['speaker_analysis']:
            speaker_analysis = result['speaker_analysis']
            
            st.markdown("### 📊 화자 분석 요약")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎤 감지된 화자 수", speaker_analysis.get('speakers', 'N/A'))
            with col2:
                st.metric("🎯 분석 품질", f"{speaker_analysis.get('quality_score', 0):.2f}")
            with col3:
                st.metric("⚙️ 분석 방법", speaker_analysis.get('method', 'N/A'))
    
    def render_image_result(self, result):
        """이미지 분석 결과 렌더링"""
        if 'extracted_text' in result:
            st.markdown("### 📝 추출된 텍스트")
            st.text_area("OCR 결과", result['extracted_text'], height=200)
    
    def render_text_result(self, result):
        """텍스트 분석 결과 렌더링"""
        if 'processed_text' in result:
            st.markdown("### 📝 처리된 텍스트")
            st.text_area("분석 결과", result['processed_text'], height=200)
    
    def download_results(self):
        """결과 다운로드"""
        if st.session_state.analysis_results:
            results_json = json.dumps(st.session_state.analysis_results, default=str, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 JSON 파일로 다운로드",
                data=results_json,
                file_name=f"conference_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def run(self):
        """메인 실행"""
        # 페이지 설정
        st.set_page_config(
            page_title="컨퍼런스 분석 시스템",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # 헤더
        self.render_header()
        
        # 단계별 렌더링
        if st.session_state.current_step == 1:
            self.render_step_1_upload()
        elif st.session_state.current_step == 2:
            self.render_step_2_analysis()
        elif st.session_state.current_step == 3:
            self.render_step_3_results()

def main():
    """메인 함수"""
    analyzer = SimpleConferenceAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()