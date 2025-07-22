#!/usr/bin/env python3
"""
솔로몬드 AI v2.3 - 실제 분석 통합 버전
가짜 분석을 실제 분석으로 완전 교체

주요 개선사항 (v2.3.1):
- 배치 분석 완료 후 결과 요약 및 미리보기 개선
- 개별 파일 텍스트 내용 실시간 표시 추가
- 분석 결과 탭 UI/UX 대폭 개선 (필터링, 페이징, 다운로드)
- 다운로드 기능 즉시 활성화 및 다양한 형식 지원
- 사용자 경험 개선 (애니메이션, 명확한 안내 메시지)
"""

import streamlit as st
import sys
import os
import asyncio
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

# NumPy 임포트 (옵션)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# 시각화 라이브러리들 (옵션)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 실제 분석 엔진 import
try:
    from core.real_analysis_engine import global_analysis_engine, analyze_file_real
    REAL_ANALYSIS_AVAILABLE = True
    print("[SUCCESS] 실제 분석 엔진 로드 완료")
except ImportError as e:
    REAL_ANALYSIS_AVAILABLE = False
    print(f"[ERROR] 실제 분석 엔진 로드 실패: {e}")

# 대용량 파일 핸들러 import
try:
    from core.large_file_handler import large_file_handler
    LARGE_FILE_HANDLER_AVAILABLE = True
    print("[SUCCESS] 대용량 파일 핸들러 로드 완료")
except ImportError as e:
    LARGE_FILE_HANDLER_AVAILABLE = False
    print(f"[ERROR] 대용량 파일 핸들러 로드 실패: {e}")

# 강의 내용 컴파일러 import
try:
    from core.lecture_content_compiler import compile_comprehensive_lecture
    LECTURE_COMPILER_AVAILABLE = True
    print("[SUCCESS] 강의 내용 컴파일러 로드 완료")
except ImportError as e:
    LECTURE_COMPILER_AVAILABLE = False
    print(f"[ERROR] 강의 내용 컴파일러 로드 실패: {e}")

try:
    from core.performance_monitor import global_performance_monitor, get_system_performance, get_current_success_rate
    PERFORMANCE_MONITOR_AVAILABLE = True
    print("[SUCCESS] 성능 모니터링 시스템 로드 완료")
except ImportError as e:
    PERFORMANCE_MONITOR_AVAILABLE = False
    print(f"[ERROR] 성능 모니터링 시스템 로드 실패: {e}")

# 기존 모듈들
try:
    from core.hybrid_llm_manager_v23 import HybridLLMManager
    HYBRID_LLM_AVAILABLE = True
except ImportError:
    HYBRID_LLM_AVAILABLE = False
    
# Streamlit 설정
st.set_page_config(
    page_title="솔로몬드 AI v2.3 - 실제 분석",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

def convert_numpy_types(obj):
    """NumPy 타입을 JSON 직렬화 가능한 Python 기본 타입으로 변환"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif NUMPY_AVAILABLE:
        # NumPy가 사용 가능한 경우에만 NumPy 타입 체크
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
    
    # NumPy가 없거나 기타 타입인 경우 그대로 반환
    return obj

class SolomondRealAnalysisUI:
    """솔로몬드 AI v2.3 실제 분석 UI - 4단계 워크플로우"""
    
    def __init__(self):
        self.setup_logging()
        self.analysis_engine = global_analysis_engine if REAL_ANALYSIS_AVAILABLE else None
        self.session_stats = {
            "files_analyzed": 0,
            "total_processing_time": 0,
            "successful_analyses": 0,
            "session_start": datetime.now()
        }
        
        # 4단계 워크플로우 상태 관리
        if 'workflow_step' not in st.session_state:
            st.session_state.workflow_step = 1
        if 'project_info' not in st.session_state:
            st.session_state.project_info = {}
        if 'uploaded_files_data' not in st.session_state:
            st.session_state.uploaded_files_data = []
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
        if 'final_report' not in st.session_state:
            st.session_state.final_report = None
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """메인 실행 - 4단계 워크플로우"""
        
        # 헤더
        st.markdown("""
        # 💎 솔로몬드 AI v2.3 - 스마트 분석 워크플로우
        
        **🚀 4단계 프로세스:** 기본정보 → 업로드 → 검토 → 보고서
        """)
        
        # 워크플로우 진행 상태 표시
        self.display_workflow_progress()
        
        # 현재 단계에 따른 렌더링
        if st.session_state.workflow_step == 1:
            self.render_step1_basic_info()
        elif st.session_state.workflow_step == 2:
            self.render_step2_upload()
        elif st.session_state.workflow_step == 3:
            self.render_step3_review()
        elif st.session_state.workflow_step == 4:
            self.render_step4_report()
        
        # 하단에 전체 시스템 상태 표시
        with st.expander("🔧 시스템 상태 확인"):
            self.display_system_status()
    
    def display_workflow_progress(self):
        """워크플로우 진행 상태 표시"""
        steps = [
            "1️⃣ 기본정보",
            "2️⃣ 업로드", 
            "3️⃣ 검토",
            "4️⃣ 보고서"
        ]
        
        cols = st.columns(4)
        for i, (col, step) in enumerate(zip(cols, steps)):
            with col:
                if i + 1 == st.session_state.workflow_step:
                    st.markdown(f"**🔸 {step}**")
                elif i + 1 < st.session_state.workflow_step:
                    st.markdown(f"✅ {step}")
                else:
                    st.markdown(f"⚪ {step}")
        
        st.markdown("---")
    
    def render_navigation_bar(self, current_step: int):
        """표준 네비게이션 바 렌더링"""
        st.markdown("---")
        
        # 이전/다음 단계 버튼 로직
        prev_step = current_step - 1 if current_step > 1 else None
        next_step = current_step + 1 if current_step < 4 else None
        
        # 단계별 조건 검사
        can_go_next = self._can_proceed_to_next_step(current_step)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if prev_step:
                step_names = {1: "기본정보", 2: "업로드", 3: "검토", 4: "보고서"}
                if st.button(f"⬅️ 이전 단계 ({step_names[prev_step]})", type="secondary"):
                    st.session_state.workflow_step = prev_step
                    st.rerun()
        
        with col3:
            if next_step and can_go_next:
                step_names = {2: "업로드", 3: "검토", 4: "보고서"}
                button_text = f"➡️ 다음 단계 ({step_names[next_step]})"
                if current_step == 3:
                    button_text = "📋 최종 보고서 생성"
                
                if st.button(button_text, type="primary"):
                    st.session_state.workflow_step = next_step
                    if current_step == 3:
                        st.success("✅ 분석 완료! 최종 보고서를 생성합니다.")
                    st.rerun()
            elif next_step and not can_go_next:
                # 조건 미충족 시 안내 메시지
                if current_step == 1:
                    st.info("기본정보를 입력하거나 '건너뛰기' 버튼을 사용하세요.")
                elif current_step == 2:
                    st.info("파일을 업로드하거나 동영상 URL을 입력해주세요.")
                elif current_step == 3:
                    st.info("분석을 완료한 후 다음 단계로 진행할 수 있습니다.")
    
    def _can_proceed_to_next_step(self, current_step: int) -> bool:
        """다음 단계로 진행 가능한지 확인"""
        if current_step == 1:
            # Step 1은 항상 건너뛸 수 있음
            return True
        elif current_step == 2:
            # Step 2는 파일이 업로드되었거나 YouTube URL이 있어야 함
            return bool(st.session_state.uploaded_files_data)
        elif current_step == 3:
            # Step 3는 분석 결과가 있어야 함
            return bool(st.session_state.analysis_results)
        return False
    
    def render_step1_basic_info(self):
        """1단계: 기본정보 입력"""
        st.markdown("## 1️⃣ 프로젝트 기본정보 (선택사항)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 프로젝트 정보")
            project_name = st.text_input(
                "프로젝트명", 
                value=st.session_state.project_info.get('name', ''),
                placeholder="예: 2024년 주얼리 트렌드 분석"
            )
            
            project_type = st.selectbox(
                "분석 유형",
                ["일반 분석", "주얼리 전문 분석", "고객 피드백 분석", "시장조사 분석", "교육/훈련 자료 분석"],
                index=["일반 분석", "주얼리 전문 분석", "고객 피드백 분석", "시장조사 분석", "교육/훈련 자료 분석"].index(
                    st.session_state.project_info.get('type', '일반 분석')
                )
            )
            
            priority = st.select_slider(
                "우선순위",
                options=["낮음", "보통", "높음", "긴급"],
                value=st.session_state.project_info.get('priority', '보통')
            )
        
        with col2:
            st.markdown("### 🎯 분석 목표")
            objective = st.text_area(
                "분석 목적 및 목표",
                value=st.session_state.project_info.get('objective', ''),
                placeholder="예: 고객 음성 데이터에서 주얼리 선호도 패턴 분석",
                height=80
            )
            
            target_language = st.selectbox(
                "주요 입력 언어",
                ["자동 감지", "한국어", "영어", "중국어", "일본어", "스페인어"],
                index=["자동 감지", "한국어", "영어", "중국어", "일본어", "스페인어"].index(
                    st.session_state.project_info.get('target_language', '자동 감지')
                )
            )
        
        # 새로운 섹션: 참석자 및 상황 정보
        st.markdown("### 👥 참석자 및 상황 정보 (분석 품질 향상)")
        
        col3, col4 = st.columns(2)
        
        with col3:
            participants = st.text_area(
                "참석자 정보",
                value=st.session_state.project_info.get('participants', ''),
                placeholder="예: 김철수 (마케팅 팀장), 박영희 (디자인 실장), 고객 A, B, C",
                height=80,
                help="참석자 이름과 역할을 입력하면 음성 인식 정확도가 향상됩니다"
            )
            
            speakers = st.text_input(
                "주요 발표자",
                value=st.session_state.project_info.get('speakers', ''),
                placeholder="예: 김철수, 박영희",
                help="주요 발표자를 명시하면 화자 구분과 내용 분석이 개선됩니다"
            )
        
        with col4:
            event_context = st.text_area(
                "상황 및 배경",
                value=st.session_state.project_info.get('event_context', ''),
                placeholder="예: 2024년 Q1 주얼리 트렌드 세미나, 고객 피드백 수집 회의",
                height=80,
                help="상황 정보는 분석 결과의 해석과 강의 내용 생성에 활용됩니다"
            )
            
            topic_keywords = st.text_input(
                "주요 주제 키워드",
                value=st.session_state.project_info.get('topic_keywords', ''),
                placeholder="예: 다이아몬드, 골드, 트렌드, 브랜딩, 고객만족",
                help="예상되는 주제 키워드를 입력하면 OCR과 STT 정확도가 향상됩니다"
            )
        
        # 다각도 분석 설정
        st.markdown("### 🔄 다각도 분석 설정")
        
        col5, col6 = st.columns(2)
        
        with col5:
            enable_multi_angle = st.checkbox(
                "다각도 종합 분석 활성화",
                value=st.session_state.project_info.get('enable_multi_angle', True),
                help="동일 상황의 여러 파일(영상, 이미지, 음성)을 종합하여 분석합니다"
            )
            
            output_format = st.multiselect(
                "원하는 출력 형식",
                ["요약 텍스트", "키워드 추출", "감정 분석", "카테고리 분류", "통계 차트", "종합 강의 자료"],
                default=st.session_state.project_info.get('output_format', ["요약 텍스트", "키워드 추출", "종합 강의 자료"])
            )
        
        with col6:
            analysis_depth = st.select_slider(
                "분석 깊이",
                options=["기본", "상세", "심층", "전문가급"],
                value=st.session_state.project_info.get('analysis_depth', '상세'),
                help="깊이가 높을수록 더 상세한 분석과 인사이트를 제공합니다"
            )
        
        # 🎯 분석 모드 선택 (독립 섹션으로 분리하여 직관성 개선)
        st.markdown("---")
        st.markdown("### 🎯 분석 모드 선택")
        
        col7, col8 = st.columns([3, 2])
        
        with col7:
            st.markdown("**어떤 방식으로 파일들을 분석할까요?**")
            analysis_mode = st.radio(
                "분석 방식 선택",
                options=[
                    "🚀 **배치 종합 분석** (권장) - 모든 파일을 통합하여 고품질 분석",
                    "📁 **개별 파일 분석** - 각 파일을 독립적으로 분석"
                ],
                index=0 if st.session_state.project_info.get('correlation_analysis', True) else 1,
                label_visibility="collapsed"
            )
            
            correlation_analysis = "배치 종합 분석" in analysis_mode
        
        with col8:
            if correlation_analysis:
                st.success("""
                ✨ **배치 종합 분석의 장점:**
                - 파일 간 상관관계 분석
                - 중복 내용 자동 제거
                - 컨텍스트 통합으로 정확도 향상
                - 종합적인 인사이트 도출
                """)
            else:
                st.warning("""
                📁 **개별 파일 분석 특징:**
                - 파일별 독립적 처리
                - 빠른 처리 속도
                - 상관관계 분석 불가
                - 제한적인 통합 인사이트
                """)
        
        # 확장된 기본정보 저장
        st.session_state.project_info = {
            'name': project_name,
            'type': project_type,
            'priority': priority,
            'objective': objective,
            'target_language': target_language,
            'participants': participants,
            'speakers': speakers,
            'event_context': event_context,
            'topic_keywords': topic_keywords,
            'enable_multi_angle': enable_multi_angle,
            'analysis_depth': analysis_depth,
            'correlation_analysis': correlation_analysis,
            'output_format': output_format,
            'created_time': datetime.now().isoformat()
        }
        
        # 건너뛰기 버튼 (중앙에 별도로)
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📋 기본정보 건너뛰고 업로드 시작", type="primary", use_container_width=True):
                st.session_state.workflow_step = 2
                st.rerun()
        
        # 표준 네비게이션 바
        self.render_navigation_bar(1)
    
    def render_step2_upload(self):
        """2단계: 파일 업로드"""
        st.markdown("## 2️⃣ 다중 파일 업로드")
        
        # 프로젝트 정보 요약 표시
        if st.session_state.project_info.get('name'):
            with st.expander("📋 프로젝트 정보 요약"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**프로젝트명:** {st.session_state.project_info.get('name', 'N/A')}")
                    st.write(f"**분석 유형:** {st.session_state.project_info.get('type', 'N/A')}")
                with col2:
                    st.write(f"**우선순위:** {st.session_state.project_info.get('priority', 'N/A')}")
                    st.write(f"**주요 언어:** {st.session_state.project_info.get('target_language', 'N/A')}")
        
        # 지원 파일 형식 안내
        with st.expander("📁 지원하는 파일 형식"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                **🎤 음성/동영상:**
                - MP3, WAV, FLAC, M4A
                - MP4, MOV, AVI
                - 유튜브 URL
                """)
            with col2:
                st.markdown("""
                **🖼️ 이미지:**
                - JPG, JPEG, PNG
                - BMP, TIFF, WEBP
                - PDF (이미지 포함)
                """)
            with col3:
                st.markdown("""
                **📄 문서:**
                - PDF 문서
                - Word (DOCX)
                - 텍스트 (TXT)
                """)
        
        # 파일 업로드 인터페이스
        st.markdown("### 📤 파일 업로드")
        
        # 대용량 파일 업로드 지원 (5GB까지)
        if LARGE_FILE_HANDLER_AVAILABLE:
            st.info("💪 **대용량 파일 지원**: 동영상 파일 최대 5GB까지 업로드 가능 (자동 청크 처리)")
            
        uploaded_files = st.file_uploader(
            "파일들을 선택하세요 (여러 개 동시 선택 가능, 동영상 최대 5GB/파일)",
            type=['wav', 'mp3', 'flac', 'm4a', 'mp4', 'mov', 'avi', 
                  'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp',
                  'pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Ctrl/Cmd + 클릭으로 여러 파일 선택 가능. 대용량 동영상은 자동으로 청크 단위 처리됩니다."
        )
        
        # 동영상 URL 입력
        st.markdown("### 🎬 동영상 URL 추가")
        video_urls = st.text_area(
            "동영상 URL (YouTube, Brightcove 등 - 한 줄에 하나씩)",
            placeholder="https://www.youtube.com/watch?v=example1\nhttps://players.brightcove.net/1659762912/default_default/index.html?videoId=6374563565112\nhttps://youtu.be/example3",
            height=120,
            help="지원 플랫폼: YouTube, Brightcove, 기타 직접 동영상 링크"
        )
        
        # 업로드된 파일 정보 표시
        if uploaded_files or video_urls.strip():
            st.markdown("### 📋 업로드된 파일 목록")
            
            total_files = 0
            total_size = 0
            file_categories = {"audio": [], "video": [], "image": [], "document": [], "youtube": []}
            
            # 업로드된 파일 분류 및 대용량 파일 감지
            large_files_detected = []
            if uploaded_files:
                for file in uploaded_files:
                    try:
                        file_size_mb = len(file.getvalue()) / (1024 * 1024)
                        file_size_gb = file_size_mb / 1024
                        total_size += file_size_mb
                        total_files += 1
                        
                        file_ext = file.name.split('.')[-1].lower() if '.' in file.name else 'unknown'
                    except Exception as e:
                        st.error(f"❌ 파일 처리 오류 ({file.name}): {str(e)}")
                        st.info("💡 **해결 방법**: 파일이 손상되지 않았는지 확인하고, 다른 형식으로 변환해보세요")
                        continue
                    
                    # 대용량 동영상 파일 감지 (1GB 이상)
                    is_large_video = file_ext in ['mp4', 'mov', 'avi'] and file_size_gb >= 1.0
                    if is_large_video:
                        large_files_detected.append((file.name, file_size_gb))
                    
                    if file_ext in ['wav', 'mp3', 'flac', 'm4a']:
                        file_categories["audio"].append((file.name, file_size_mb))
                    elif file_ext in ['mp4', 'mov', 'avi']:
                        size_display = f"{file_size_gb:.2f}GB" if file_size_gb >= 1.0 else f"{file_size_mb:.1f}MB"
                        file_categories["video"].append((file.name, size_display, is_large_video))
                    elif file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']:
                        file_categories["image"].append((file.name, file_size_mb))
                    elif file_ext in ['pdf', 'docx', 'txt']:
                        file_categories["document"].append((file.name, file_size_mb))
            
            # 동영상 URL 처리
            if video_urls.strip():
                urls = [url.strip() for url in video_urls.strip().split('\n') if url.strip()]
                for url in urls:
                    if 'youtube.com' in url or 'youtu.be' in url:
                        file_categories["youtube"].append((url, 0))
                        total_files += 1
                    elif 'brightcove.net' in url:
                        file_categories["brightcove"] = file_categories.get("brightcove", [])
                        file_categories["brightcove"].append((url, 0))
                        total_files += 1
                    elif any(domain in url.lower() for domain in ['vimeo.com', 'dailymotion.com', '.mp4', '.mov', '.avi']):
                        file_categories["other_video"] = file_categories.get("other_video", [])
                        file_categories["other_video"].append((url, 0))
                        total_files += 1
            
            # 파일 정보 표시
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📁 총 파일", f"{total_files}개")
            with col2:
                st.metric("💾 총 크기", f"{total_size:.2f} MB")
            with col3:
                languages = ["자동 감지", "한국어", "영어", "중국어", "일본어"]
                # 프로젝트 정보에서 언어 설정 가져오기 (호환성 확보)
                saved_language = st.session_state.project_info.get('target_language', '자동 감지')
                default_index = 0
                try:
                    if saved_language in languages:
                        default_index = languages.index(saved_language)
                except (ValueError, TypeError):
                    default_index = 0
                
                analysis_language = st.selectbox(
                    "분석 언어", 
                    languages,
                    index=default_index
                )
            
            # 대용량 파일 경고 표시
            if large_files_detected and LARGE_FILE_HANDLER_AVAILABLE:
                st.warning(f"🚨 **대용량 동영상 파일 감지**: {len(large_files_detected)}개 파일이 1GB 이상입니다. 자동으로 청크 단위 처리가 적용됩니다.")
                with st.expander("📊 대용량 파일 상세 정보"):
                    for filename, size_gb in large_files_detected:
                        st.write(f"🎬 {filename}: {size_gb:.2f}GB")
                        st.markdown("  - ✅ 청크 단위 업로드")
                        st.markdown("  - ✅ 오디오 자동 추출")
                        st.markdown("  - ✅ 메모리 효율적 처리")
            elif large_files_detected and not LARGE_FILE_HANDLER_AVAILABLE:
                st.error(f"❌ **대용량 파일 처리 불가**: {len(large_files_detected)}개의 대용량 파일이 있지만 대용량 파일 핸들러가 비활성화되어 있습니다.")
            
            # 파일 미리보기 및 카테고리별 목록
            self.render_file_preview(file_categories, uploaded_files)
            
            # 분석 시작 버튼
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 전체 파일 분석 시작", type="primary", use_container_width=True):
                    # 파일 데이터 저장
                    st.session_state.uploaded_files_data = {
                        'files': uploaded_files,
                        'video_urls': video_urls.strip().split('\n') if video_urls.strip() else [],
                        'analysis_language': analysis_language,
                        'total_files': total_files,
                        'total_size': total_size,
                        'categories': file_categories
                    }
                    st.session_state.workflow_step = 3
                    st.success(f"✅ {total_files}개 파일 업로드 완료! 분석을 시작합니다.")
                    st.rerun()
        
        # 표준 네비게이션 바
        self.render_navigation_bar(2)
    
    def render_file_preview(self, file_categories: Dict, uploaded_files: List):
        """파일 미리보기 및 상세 정보 표시"""
        st.markdown("### 📋 업로드된 파일 미리보기")
        
        # 파일 딕셔너리 생성 (파일명을 키로 사용)
        file_dict = {}
        if uploaded_files:
            for file in uploaded_files:
                file_dict[file.name] = file
        
        # 카테고리별 파일 목록과 미리보기
        for category, files in file_categories.items():
            if not files:
                continue
                
            category_names = {
                "audio": "🎤 음성 파일",
                "video": "🎬 동영상 파일", 
                "image": "🖼️ 이미지 파일",
                "document": "📄 문서 파일",
                "youtube": "🎬 YouTube URL",
                "brightcove": "📺 Brightcove URL",
                "other_video": "🌐 기타 동영상 URL"
            }
            
            with st.expander(f"{category_names[category]} ({len(files)}개)", expanded=True):
                # 이미지 파일의 경우 썸네일 표시
                if category == "image":
                    # 이미지를 그리드로 표시
                    cols = st.columns(min(3, len(files)))
                    for idx, file_info in enumerate(files):
                        name, size_mb = file_info
                        col_idx = idx % 3
                        
                        with cols[col_idx]:
                            # 파일 정보 표시
                            st.write(f"**{name}**")
                            st.caption(f"📏 크기: {size_mb:.2f} MB")
                            
                            # 이미지 썸네일 표시
                            if name in file_dict:
                                try:
                                    st.image(file_dict[name], width=200, caption=name)
                                except Exception as e:
                                    st.error(f"⚠️ 이미지 미리보기 실패: {str(e)}")
                
                # 오디오 파일의 경우 상세 정보와 재생
                elif category == "audio":
                    for file_info in files:
                        name, size_mb = file_info
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"🎵 **{name}**")
                            st.caption(f"📏 크기: {size_mb:.2f} MB")
                            
                            # 오디오 파일 정보 추출 시도
                            if name in file_dict:
                                try:
                                    # 임시로 파일 저장하여 정보 추출
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{name.split('.')[-1]}") as tmp_file:
                                        tmp_file.write(file_dict[name].getvalue())
                                        tmp_path = tmp_file.name
                                    
                                    # 오디오 정보 표시 (실제 분석 엔진 활용)
                                    try:
                                        from core.audio_converter import get_audio_info
                                        audio_info = get_audio_info(tmp_path)
                                        if audio_info['is_valid']:
                                            st.caption(f"⏱️ 길이: {audio_info['duration_seconds']:.1f}초")
                                            st.caption(f"🎵 샘플링: {audio_info['sample_rate']}Hz")
                                            st.caption(f"📻 채널: {audio_info['channels']}ch")
                                    except ImportError:
                                        st.caption("📊 상세 정보: 분석 엔진 로드 후 확인 가능")
                                    except Exception:
                                        st.caption("📊 상세 정보: 분석 중 확인됩니다")
                                    
                                    # 임시 파일 정리
                                    try:
                                        os.unlink(tmp_path)
                                    except:
                                        pass
                                        
                                except Exception as e:
                                    st.caption(f"⚠️ 정보 추출 실패: {str(e)}")
                        
                        with col2:
                            # 오디오 재생 위젯 (지원되는 형식만)
                            if name in file_dict and name.lower().endswith(('.wav', '.mp3')):
                                try:
                                    st.audio(file_dict[name])
                                except Exception:
                                    st.caption("🎵 재생기 로드 실패")
                
                # 동영상 파일의 경우 상세 정보
                elif category == "video":
                    for file_info in files:
                        name, size_display, is_large = file_info
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            icon = "🎬🚀" if is_large else "🎬"
                            note = " (대용량 자동 처리)" if is_large else ""
                            st.write(f"{icon} **{name}**{note}")
                            st.caption(f"📏 크기: {size_display}")
                            
                            if is_large:
                                st.info("✨ 대용량 파일로 자동 청크 처리됩니다")
                        
                        with col2:
                            if name in file_dict:
                                try:
                                    # 비디오 미리보기 (작은 크기로)
                                    st.video(file_dict[name])
                                except Exception:
                                    st.caption("🎬 미리보기 로드 실패")
                
                # 문서 파일의 경우
                elif category == "document":
                    for file_info in files:
                        name, size_mb = file_info
                        st.write(f"📄 **{name}**")
                        st.caption(f"📏 크기: {size_mb:.2f} MB")
                        
                        # 파일 형식별 설명
                        if name.lower().endswith('.pdf'):
                            st.caption("📑 PDF 문서 - OCR 텍스트 추출 예정")
                        elif name.lower().endswith('.docx'):
                            st.caption("📝 Word 문서 - 텍스트 추출 예정") 
                        elif name.lower().endswith('.txt'):
                            st.caption("📋 텍스트 파일 - 직접 읽기")
                
                # 온라인 동영상 URL의 경우
                elif category in ["youtube", "brightcove", "other_video"]:
                    for file_info in files:
                        name = file_info[0]
                        
                        if category == "youtube":
                            st.write(f"🎬 **YouTube URL**")
                            st.caption("🎬 비디오 정보 및 오디오 다운로드 예정")
                        elif category == "brightcove":
                            st.write(f"📺 **Brightcove URL**")
                            st.caption("🎬 Brightcove 플레이어에서 동영상 분석 예정")
                        elif category == "other_video":
                            st.write(f"🌐 **동영상 URL**")
                            st.caption("🎬 직접 링크에서 동영상 다운로드 및 분석 예정")
                        
                        st.code(name)
                        
                        # URL 유효성 간단 체크
                        if name.startswith(('http://', 'https://')):
                            st.success("✅ 유효한 URL 형식")
                        else:
                            st.warning("⚠️ URL이 http:// 또는 https://로 시작하지 않습니다")
    
    def render_step3_review(self):
        """3단계: 중간 검토"""
        st.markdown("## 3️⃣ 분석 진행 및 중간 검토")
        
        if not st.session_state.uploaded_files_data:
            st.error("업로드된 파일이 없습니다. 이전 단계로 돌아가세요.")
            return
        
        # 분석 진행 상황 표시
        st.markdown("### 🔄 분석 진행 상황")
        
        uploaded_data = st.session_state.uploaded_files_data
        
        # 분석 실행 전 시스템 상태 체크
        analysis_ready, dependency_status = self.check_analysis_readiness()
        
        # 의존성 상태 표시
        if not analysis_ready:
            st.error("🚨 분석 시스템 준비 불완료")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**시스템 상태:**")
                for component, status in dependency_status.items():
                    icon = "✅" if status else "❌"
                    st.markdown(f"{icon} {component}")
            with col2:
                st.markdown("**해결 방법:**")
                if not dependency_status.get('whisper', False):
                    st.markdown("- `pip install openai-whisper` 실행")
                if not dependency_status.get('easyocr', False):
                    st.markdown("- `pip install easyocr` 실행")
                if not dependency_status.get('transformers', False):
                    st.markdown("- `pip install transformers` 실행")
        
        # 실제 분석 실행
        analysis_button_disabled = not analysis_ready or len(uploaded_data.get('files', [])) == 0
        
        if st.button("▶️ 분석 실행", type="primary", disabled=analysis_button_disabled):
            if analysis_ready:
                with st.spinner("🔄 포괄적 분석을 시작합니다..."):
                    results = self.execute_comprehensive_analysis()
                    st.session_state.analysis_results = results
            else:
                st.error("분석 시스템이 준비되지 않았습니다. 위의 의존성을 먼저 설치해주세요.")
        
        # 기존 분석 결과가 있으면 표시
        if st.session_state.analysis_results:
            st.markdown("### 📊 중간 분석 결과")
            
            # 분석 완료 통계
            total_results = len(st.session_state.analysis_results)
            successful_results = len([r for r in st.session_state.analysis_results if r.get('status') == 'success'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("처리된 파일", f"{total_results}개")
            with col2:
                st.metric("성공", f"{successful_results}개")
            with col3:
                success_rate = (successful_results / total_results * 100) if total_results > 0 else 0
                st.metric("성공률", f"{success_rate:.1f}%")
            
            # 결과 미리보기
            for i, result in enumerate(st.session_state.analysis_results[:5]):  # 최대 5개만 표시
                with st.expander(f"📄 {result.get('file_name', f'파일 {i+1}')} - {result.get('analysis_type', 'unknown')}"):
                    if result.get('status') == 'success':
                        if result.get('full_text'):
                            st.write("**추출된 텍스트:**")
                            preview_text = result['full_text'][:300] + ("..." if len(result['full_text']) > 300 else "")
                            st.text_area("추출된 텍스트 미리보기", value=preview_text, height=100, disabled=True, key=f"preview_{i}", label_visibility="collapsed")
                        
                        if result.get('summary'):
                            st.write("**AI 요약:**")
                            st.info(result['summary'])
                        
                        if result.get('jewelry_keywords'):
                            st.write("**키워드:**")
                            for keyword in result['jewelry_keywords'][:5]:
                                st.badge(keyword)
                    else:
                        st.error(f"❌ 분석 실패: {result.get('error', 'Unknown error')}")
            
            if len(st.session_state.analysis_results) > 5:
                st.info(f"추가 {len(st.session_state.analysis_results) - 5}개 결과가 더 있습니다.")
            
        # 표준 네비게이션 바
        self.render_navigation_bar(3)
    
    def render_step4_report(self):
        """4단계: 최종 보고서"""
        st.markdown("## 4️⃣ 최종 분석 보고서")
        
        if not st.session_state.analysis_results:
            st.error("분석 결과가 없습니다. 이전 단계로 돌아가서 분석을 진행해주세요.")
            return
        
        # 최종 보고서 생성
        if not st.session_state.final_report:
            with st.spinner("📊 최종 보고서 생성 중..."):
                st.session_state.final_report = self.generate_final_report()
        
        if st.session_state.final_report:
            report = st.session_state.final_report
            
            # 보고서 헤더
            st.markdown("### 📋 프로젝트 분석 보고서")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**프로젝트:** {report['project_name']}")
                st.markdown(f"**분석 일시:** {report['analysis_date']}")
                st.markdown(f"**총 처리 파일:** {report['total_files']}개")
            with col2:
                st.markdown(f"**성공률:** {report['success_rate']:.1f}%")
                st.markdown(f"**처리 시간:** {report['total_time']:.1f}초")
            
            st.markdown("---")
            
            # 핵심 요약
            st.markdown("### 🎯 핵심 요약")
            st.markdown(report['executive_summary'])
            
            # 주요 발견사항
            if report['key_findings']:
                st.markdown("### 🔍 주요 발견사항")
                for i, finding in enumerate(report['key_findings'], 1):
                    st.markdown(f"{i}. {finding}")
            
            # 키워드 클라우드
            if report['top_keywords']:
                st.markdown("### 🏷️ 주요 키워드")
                col1, col2, col3 = st.columns(3)
                for i, (keyword, count) in enumerate(report['top_keywords'][:15]):
                    with [col1, col2, col3][i % 3]:
                        st.metric(keyword, f"{count}회")
            
            # 📊 고급 분석 대시보드
            st.markdown("### 📊 분석 대시보드")
            self.render_advanced_dashboard(report)
            
            # 파일별 상세 결과
            with st.expander("📄 파일별 상세 분석 결과"):
                for result in st.session_state.analysis_results:
                    if result.get('status') == 'success':
                        st.markdown(f"**{result['file_name']}**")
                        if result.get('full_text'):
                            st.text_area(
                                "추출된 텍스트",
                                value=result['full_text'][:500] + ("..." if len(result['full_text']) > 500 else ""),
                                height=100,
                                disabled=True,
                                key=f"detail_{result['file_name']}"
                            )
                        if result.get('summary'):
                            st.info(f"**요약:** {result['summary']}")
                        st.markdown("---")
            
            # 결론 및 제안사항
            if report['conclusions']:
                st.markdown("### 💡 결론 및 제안사항")
                for i, conclusion in enumerate(report['conclusions'], 1):
                    st.markdown(f"{i}. {conclusion}")
            
            # 종합 강의 내용
            if LECTURE_COMPILER_AVAILABLE:
                st.markdown("### 🎓 종합 강의 내용")
                
                if not hasattr(st.session_state, 'comprehensive_lecture') or st.session_state.comprehensive_lecture is None:
                    if st.button("📚 종합 강의 내용 생성", type="secondary"):
                        st.session_state.comprehensive_lecture = self.generate_comprehensive_lecture()
                        if st.session_state.comprehensive_lecture:
                            st.success("✅ 종합 강의 내용 생성 완료!")
                            st.rerun()
                else:
                    # 강의 내용 표시
                    lecture = st.session_state.comprehensive_lecture
                    
                    # 강의 제목
                    st.markdown(f"#### 📖 {lecture['title']}")
                    
                    # 강의 개요
                    with st.expander("📋 강의 개요", expanded=True):
                        st.markdown(lecture['overview'])
                    
                    # 주요 주제
                    if lecture['main_topics']:
                        with st.expander("🎯 주요 주제"):
                            for i, topic in enumerate(lecture['main_topics'], 1):
                                st.markdown(f"{i}. {topic}")
                    
                    # 핵심 인사이트
                    if lecture['key_insights']:
                        with st.expander("💡 핵심 인사이트"):
                            for i, insight in enumerate(lecture['key_insights'], 1):
                                st.markdown(f"{i}. {insight}")
                    
                    # 실용적 응용 방안
                    if lecture['practical_applications']:
                        with st.expander("🛠️ 실용적 응용 방안"):
                            for i, application in enumerate(lecture['practical_applications'], 1):
                                st.markdown(f"{i}. {application}")
                    
                    # 세부 내용 (카테고리별)
                    if lecture['detailed_content']:
                        with st.expander("📚 세부 내용 (카테고리별)"):
                            for category, content in lecture['detailed_content'].items():
                                if content['summary']:
                                    st.markdown(f"**{category.replace('_', ' ').title()}**")
                                    st.markdown(content['summary'])
                                    
                                    if content['key_points']:
                                        st.markdown("주요 포인트:")
                                        for point in content['key_points'][:3]:  # 상위 3개만 표시
                                            st.markdown(f"• {point}")
                                    st.markdown("---")
                    
                    # 결론
                    if lecture['conclusion']:
                        with st.expander("🎯 강의 결론"):
                            st.markdown(lecture['conclusion'])
                    
                    # 품질 및 메타데이터
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("품질 점수", f"{lecture['metadata']['quality_score']:.1f}/100")
                    with col2:
                        st.metric("처리 파일 수", lecture['metadata']['total_files'])
                    with col3:
                        st.metric("컴파일 시간", f"{lecture['metadata']['compilation_time']:.1f}초")
                    
                    # 강의 내용 다운로드
                    if st.button("🔄 강의 내용 재생성", type="secondary"):
                        st.session_state.comprehensive_lecture = None
                        st.rerun()
            
            # 다운로드 옵션
            st.markdown("### 📥 보고서 다운로드")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 완전한 보고서 JSON
                report_json = json.dumps(convert_numpy_types({
                    'report': report,
                    'detailed_results': st.session_state.analysis_results,
                    'project_info': st.session_state.project_info
                }), indent=2, ensure_ascii=False)
                
                st.download_button(
                    "📊 완전한 보고서 (JSON)",
                    data=report_json,
                    file_name=f"분석보고서_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                # 요약 보고서 텍스트
                summary_text = f"""
# {report['project_name']} 분석 보고서

## 분석 개요
- 분석 일시: {report['analysis_date']}
- 총 처리 파일: {report['total_files']}개
- 성공률: {report['success_rate']:.1f}%

## 핵심 요약
{report['executive_summary']}

## 주요 발견사항
{chr(10).join([f'{i}. {finding}' for i, finding in enumerate(report['key_findings'], 1)])}

## 주요 키워드
{', '.join([keyword for keyword, _ in report['top_keywords'][:10]])}

## 결론 및 제안사항
{chr(10).join([f'{i}. {conclusion}' for i, conclusion in enumerate(report['conclusions'], 1)])}
"""
                
                st.download_button(
                    "📄 요약 보고서 (텍스트)",
                    data=summary_text,
                    file_name=f"요약보고서_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col3:
                # 강의 내용 다운로드 (있는 경우)
                if hasattr(st.session_state, 'comprehensive_lecture') and st.session_state.comprehensive_lecture:
                    lecture = st.session_state.comprehensive_lecture
                    
                    # 강의 내용을 텍스트로 변환
                    lecture_text = f"""
# {lecture['title']}

## 강의 개요
{lecture['overview']}

## 주요 주제
{chr(10).join([f'{i}. {topic}' for i, topic in enumerate(lecture['main_topics'], 1)])}

## 핵심 인사이트
{chr(10).join([f'{i}. {insight}' for i, insight in enumerate(lecture['key_insights'], 1)])}

## 실용적 응용 방안
{chr(10).join([f'{i}. {app}' for i, app in enumerate(lecture['practical_applications'], 1)])}

## 결론
{lecture['conclusion']}

---
생성 일시: {lecture['metadata']['compilation_date']}
품질 점수: {lecture['metadata']['quality_score']}/100
"""
                    
                    st.download_button(
                        "🎓 종합 강의 내용",
                        data=lecture_text,
                        file_name=f"종합강의_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                else:
                    # 전체 추출 텍스트 (강의 내용이 없는 경우)
                    all_texts = []
                    for result in st.session_state.analysis_results:
                        if result.get('status') == 'success' and result.get('full_text'):
                            all_texts.append(f"=== {result['file_name']} ===\n{result['full_text']}\n")
                    
                    combined_text = "\n".join(all_texts)
                    
                    st.download_button(
                        "📝 전체 추출 텍스트",
                        data=combined_text,
                        file_name=f"전체텍스트_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        
        # 성능 분석 대시보드
        if PERFORMANCE_MONITOR_AVAILABLE:
            st.markdown("---")
            st.markdown("### 📊 시스템 성능 분석")
            
            try:
                performance_summary = get_system_performance()
                
                # 성능 메트릭 표시
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    overall = performance_summary["overall_performance"]
                    rate = overall["success_rate"]
                    if rate >= 85:
                        st.metric("🎯 전체 성공률", f"{rate}%", delta="우수")
                    elif rate >= 70:
                        st.metric("🎯 전체 성공률", f"{rate}%", delta="양호")
                    else:
                        st.metric("🎯 전체 성공률", f"{rate}%", delta="개선필요")
                
                with col2:
                    recent = performance_summary["recent_performance"]
                    st.metric("📈 최근 성공률", f"{recent['success_rate']}%", 
                             delta=f"최근 {recent['total_analyses']}개")
                
                with col3:
                    total_processed = performance_summary["system_stats"]["total_files_processed"]
                    st.metric("📁 총 처리 파일", f"{total_processed}개")
                
                with col4:
                    errors = performance_summary["error_analysis"]["total_errors"]
                    if errors == 0:
                        st.metric("🛡️ 총 오류", "0개", delta="안정")
                    else:
                        st.metric("🛡️ 총 오류", f"{errors}개", delta="점검필요")
                
                # 파일 타입별 성능
                if performance_summary["file_type_performance"]:
                    with st.expander("📈 파일 타입별 성능 상세"):
                        for file_type, perf in performance_summary["file_type_performance"].items():
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"**{file_type.upper()} 파일**")
                            with col2:
                                st.markdown(f"성공률: {perf['success_rate']}%")
                            with col3:
                                st.markdown(f"평균 시간: {perf['avg_processing_time']}초")
                
                # 성능 개선 추천사항
                recommendations = global_performance_monitor.get_recommendations()
                if recommendations:
                    with st.expander("💡 성능 개선 추천사항"):
                        for rec in recommendations:
                            if rec["priority"] == "high":
                                st.error(f"🔴 **{rec['category']}**: {rec['recommendation']}")
                            elif rec["priority"] == "medium":
                                st.warning(f"🟡 **{rec['category']}**: {rec['recommendation']}")
                            else:
                                st.info(f"🔵 **{rec['category']}**: {rec['recommendation']}")
                
            except Exception as e:
                st.error(f"⚠️ 성능 분석 오류: {str(e)}")
        
        # 새 프로젝트 시작
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🆕 새 프로젝트 시작", type="primary", use_container_width=True):
                # 세션 상태 초기화
                st.session_state.workflow_step = 1
                st.session_state.project_info = {}
                st.session_state.uploaded_files_data = []
                st.session_state.analysis_results = []
                st.session_state.final_report = None
                st.success("✅ 새 프로젝트가 시작되었습니다!")
                st.rerun()
        
        # 표준 네비게이션 바
        self.render_navigation_bar(4)
    
    def render_multifile_analysis_tab(self):
        """멀티파일 분석 탭"""
        
        st.markdown("## 📁 멀티파일 배치 분석")
        st.markdown("**🚀 모든 지원 형식을 한번에 업로드하여 배치 분석**")
        
        if not REAL_ANALYSIS_AVAILABLE:
            st.error("❌ 실제 분석 엔진이 로드되지 않았습니다.")
            return
        
        # 지원 형식 안내
        with st.expander("📋 지원하는 파일 형식"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🎤 음성 파일:**")
                st.markdown("- WAV, MP3, FLAC, M4A, MP4")
                st.markdown("- Whisper STT로 실제 변환")
            
            with col2:
                st.markdown("**🖼️ 이미지 파일:**")
                st.markdown("- JPG, JPEG, PNG, BMP, TIFF")
                st.markdown("- EasyOCR로 실제 텍스트 추출")
        
        # 멀티파일 업로드
        uploaded_files = st.file_uploader(
            "파일들을 선택하세요 (여러 개 동시 선택 가능)",
            type=['wav', 'mp3', 'flac', 'm4a', 'mp4', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Ctrl/Cmd + 클릭으로 여러 파일 선택 가능"
        )
        
        if uploaded_files:
            # 업로드된 파일 정보 표시
            st.markdown("### 📋 업로드된 파일 목록")
            
            audio_files = []
            image_files = []
            total_size = 0
            
            # 파일 분류
            for file in uploaded_files:
                file_size = len(file.getvalue()) / (1024 * 1024)
                total_size += file_size
                
                file_ext = file.name.split('.')[-1].lower()
                
                if file_ext in ['wav', 'mp3', 'flac', 'm4a', 'mp4']:
                    audio_files.append(file)
                elif file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
                    image_files.append(file)
            
            # 파일 정보 표시
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎤 음성 파일", f"{len(audio_files)}개")
            
            with col2:
                st.metric("🖼️ 이미지 파일", f"{len(image_files)}개")
            
            with col3:
                st.metric("📦 총 크기", f"{total_size:.2f} MB")
            
            # 파일 목록 상세 표시
            if audio_files or image_files:
                with st.expander("🔍 파일 상세 정보"):
                    
                    if audio_files:
                        st.markdown("**🎤 음성 파일들:**")
                        for i, file in enumerate(audio_files, 1):
                            file_size = len(file.getvalue()) / (1024 * 1024)
                            st.write(f"{i}. {file.name} ({file_size:.2f} MB)")
                    
                    if image_files:
                        st.markdown("**🖼️ 이미지 파일들:**")
                        for i, file in enumerate(image_files, 1):
                            file_size = len(file.getvalue()) / (1024 * 1024)
                            st.write(f"{i}. {file.name} ({file_size:.2f} MB)")
            
            # 분석 설정
            st.markdown("### ⚙️ 배치 분석 설정")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                audio_language = st.selectbox(
                    "음성 언어",
                    ["ko", "en", "auto"],
                    help="모든 음성 파일에 적용"
                )
            
            with col2:
                whisper_model = st.selectbox(
                    "Whisper 모델",
                    ["tiny", "base", "small", "medium"],
                    index=1,
                    help="정확도 vs 속도"
                )
            
            with col3:
                cpu_mode = st.checkbox(
                    "CPU 모드 강제",
                    value=True,
                    help="GPU 메모리 부족 방지"
                )
            
            # 배치 분석 시작
            if st.button("🚀 멀티파일 배치 분석 시작", type="primary"):
                self.process_multifile_analysis(
                    audio_files, image_files, 
                    audio_language, whisper_model, cpu_mode
                )
        
        else:
            st.info("📁 여러 파일을 선택하여 배치 분석을 시작하세요.")
            st.markdown("**💡 사용법:**")
            st.markdown("1. 위의 파일 업로드 버튼 클릭")
            st.markdown("2. Ctrl/Cmd + 클릭으로 여러 파일 선택")
            st.markdown("3. 음성과 이미지 파일을 함께 선택 가능")
            st.markdown("4. 설정 확인 후 배치 분석 시작")
    
    def process_multifile_analysis(self, audio_files: List, image_files: List, 
                                 language: str, model_size: str, cpu_mode: bool):
        """멀티파일 배치 분석 처리"""
        
        total_files = len(audio_files) + len(image_files)
        
        if total_files == 0:
            st.warning("⚠️ 분석할 파일이 없습니다.")
            return
        
        # CPU 모드 설정
        if cpu_mode:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # 진행 상황 표시
        st.markdown("### 🔄 배치 분석 진행 상황")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        batch_results = []
        processed_count = 0
        
        # 배치 분석 시작 시간
        batch_start_time = time.time()
        
        try:
            # 음성 파일 분석
            for i, audio_file in enumerate(audio_files):
                
                status_text.text(f"🎤 음성 분석 중: {audio_file.name} ({i+1}/{len(audio_files)})")
                
                # 임시 파일 생성
                with tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=f".{audio_file.name.split('.')[-1]}"
                ) as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # 실제 분석 실행
                    result = self.analysis_engine.analyze_audio_file(tmp_file_path, language)
                    result['batch_index'] = processed_count + 1
                    result['file_type'] = 'audio'
                    batch_results.append(result)
                    
                    # 결과 실시간 표시
                    with results_container:
                        if result.get('status') == 'success':
                            st.success(f"✅ {audio_file.name}: {result['text_length']}글자 추출 ({result['processing_time']}초)")
                            # 성공한 결과의 텍스트 미리보기 표시
                            if result.get('full_text'):
                                preview_text = result['full_text'][:200] + ("..." if len(result['full_text']) > 200 else "")
                                st.text_area(
                                    f"추출된 텍스트 미리보기: {audio_file.name}",
                                    value=preview_text,
                                    height=80,
                                    disabled=True,
                                    key=f"audio_preview_{processed_count}"
                                )
                        else:
                            st.error(f"❌ {audio_file.name}: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    error_result = {
                        'status': 'error',
                        'error': str(e),
                        'file_name': audio_file.name,
                        'batch_index': processed_count + 1,
                        'file_type': 'audio'
                    }
                    batch_results.append(error_result)
                    
                    with results_container:
                        st.error(f"❌ {audio_file.name}: {str(e)}")
                
                finally:
                    # 임시 파일 정리
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
                
                # 진행률 업데이트
                processed_count += 1
                progress_bar.progress(processed_count / total_files)
            
            # 이미지 파일 분석
            for i, image_file in enumerate(image_files):
                
                status_text.text(f"🖼️ 이미지 분석 중: {image_file.name} ({i+1}/{len(image_files)})")
                
                # 임시 파일 생성
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f".{image_file.name.split('.')[-1]}"
                ) as tmp_file:
                    tmp_file.write(image_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # 실제 분석 실행
                    result = self.analysis_engine.analyze_image_file(tmp_file_path)
                    result['batch_index'] = processed_count + 1
                    result['file_type'] = 'image'
                    batch_results.append(result)
                    
                    # 결과 실시간 표시
                    with results_container:
                        if result.get('status') == 'success':
                            st.success(f"✅ {image_file.name}: {result['blocks_detected']}개 블록 추출 ({result['processing_time']}초)")
                            # 성공한 결과의 텍스트 미리보기 표시
                            if result.get('full_text'):
                                preview_text = result['full_text'][:200] + ("..." if len(result['full_text']) > 200 else "")
                                st.text_area(
                                    f"추출된 텍스트 미리보기: {image_file.name}",
                                    value=preview_text,
                                    height=80,
                                    disabled=True,
                                    key=f"image_preview_{processed_count}"
                                )
                        else:
                            st.error(f"❌ {image_file.name}: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    error_result = {
                        'status': 'error',
                        'error': str(e),
                        'file_name': image_file.name,
                        'batch_index': processed_count + 1,
                        'file_type': 'image'
                    }
                    batch_results.append(error_result)
                    
                    with results_container:
                        st.error(f"❌ {image_file.name}: {str(e)}")
                
                finally:
                    # 임시 파일 정리
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
                
                # 진행률 업데이트
                processed_count += 1
                progress_bar.progress(processed_count / total_files)
            
            # 배치 분석 완료
            batch_end_time = time.time()
            total_batch_time = batch_end_time - batch_start_time
            
            # 최종 결과 요약
            self.display_batch_results_summary(batch_results, total_batch_time)
            
            # 세션에 결과 저장 (NumPy 타입 변환 후)
            if 'analysis_results' not in st.session_state:
                st.session_state.analysis_results = []
            
            # NumPy 타입을 JSON 호환 타입으로 변환
            converted_batch_results = convert_numpy_types(batch_results)
            st.session_state.analysis_results.extend(converted_batch_results)
            
            # 성공 메시지와 함께 링크 제공
            status_text.text("✅ 멀티파일 배치 분석 완료!")
            st.balloons()  # 축하 애니메이션
            
            # 결과 확인 링크
            st.markdown("### 🎉 분석 완료!")
            st.info("📊 **분석 결과** 탭에서 모든 결과를 확인하고 다운로드할 수 있습니다.")
            
        except Exception as e:
            st.error(f"❌ 배치 분석 중 오류: {str(e)}")
            self.logger.error(f"배치 분석 오류: {e}")
    
    def display_batch_results_summary(self, batch_results: List[Dict], total_time: float):
        """배치 분석 결과 요약 표시"""
        
        st.markdown("### 📊 배치 분석 완료 요약")
        
        # 통계 계산
        total_files = len(batch_results)
        successful_files = len([r for r in batch_results if r.get('status') == 'success'])
        audio_files = len([r for r in batch_results if r.get('file_type') == 'audio'])
        image_files = len([r for r in batch_results if r.get('file_type') == 'image'])
        
        # 메트릭 표시
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 파일", f"{total_files}개")
        
        with col2:
            st.metric("성공", f"{successful_files}개")
        
        with col3:
            success_rate = (successful_files / total_files * 100) if total_files > 0 else 0
            st.metric("성공률", f"{success_rate:.1f}%")
        
        with col4:
            st.metric("총 처리시간", f"{total_time:.1f}초")
        
        # 파일 타입별 통계
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎤 음성 파일 결과:**")
            audio_success = len([r for r in batch_results 
                               if r.get('file_type') == 'audio' and r.get('status') == 'success'])
            st.write(f"- 처리: {audio_files}개")
            st.write(f"- 성공: {audio_success}개")
            
            if audio_success > 0:
                total_text = sum(r.get('text_length', 0) for r in batch_results 
                               if r.get('file_type') == 'audio' and r.get('status') == 'success')
                st.write(f"- 총 추출 텍스트: {total_text}글자")
        
        with col2:
            st.markdown("**🖼️ 이미지 파일 결과:**")
            image_success = len([r for r in batch_results 
                               if r.get('file_type') == 'image' and r.get('status') == 'success'])
            st.write(f"- 처리: {image_files}개")
            st.write(f"- 성공: {image_success}개")
            
            if image_success > 0:
                total_blocks = sum(r.get('blocks_detected', 0) for r in batch_results 
                                 if r.get('file_type') == 'image' and r.get('status') == 'success')
                st.write(f"- 총 텍스트 블록: {total_blocks}개")
        
        # 개별 결과 미리보기 추가
        st.markdown("### 📋 개별 분석 결과 미리보기")
        
        successful_results = [r for r in batch_results if r.get('status') == 'success']
        
        if successful_results:
            for i, result in enumerate(successful_results, 1):
                with st.expander(f"📄 {result.get('file_name', f'파일 {i}')} - {result.get('file_type', '').upper()} 결과"):
                    
                    # 기본 정보
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**파일:** {result.get('file_name', 'Unknown')}")
                    with col2:
                        st.write(f"**타입:** {result.get('file_type', 'Unknown').upper()}")
                    with col3:
                        st.write(f"**처리 시간:** {result.get('processing_time', 'N/A')}초")
                    
                    # 추출된 텍스트 표시
                    if result.get('full_text'):
                        st.markdown("**📄 추출된 텍스트:**")
                        text_preview = result['full_text'][:300] + ("..." if len(result['full_text']) > 300 else "")
                        st.text_area(
                            "텍스트 미리보기",
                            value=text_preview,
                            height=100,
                            disabled=True,
                            key=f"batch_preview_{i}"
                        )
                        
                        # 전체 텍스트 표시 옵션
                        if len(result['full_text']) > 300:
                            if st.button(f"📖 전체 텍스트 보기", key=f"show_full_{i}"):
                                st.text_area(
                                    "전체 텍스트",
                                    value=result['full_text'],
                                    height=200,
                                    disabled=True,
                                    key=f"batch_full_{i}"
                                )
                    
                    # 요약 표시
                    if result.get('summary'):
                        st.markdown("**📋 AI 요약:**")
                        st.info(result['summary'])
                    
                    # 주얼리 키워드 표시
                    if result.get('jewelry_keywords'):
                        st.markdown("**💎 주얼리 키워드:**")
                        keyword_text = ", ".join(result['jewelry_keywords'])
                        st.write(keyword_text)
        
        # 전체 결과 다운로드 - 더 명확한 형태로 제공
        st.markdown("### 📥 결과 다운로드")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON 다운로드 (NumPy 타입 변환)
            batch_data = {
                'batch_summary': {
                    'total_files': total_files,
                    'successful_files': successful_files,
                    'success_rate': success_rate,
                    'total_processing_time': total_time,
                    'audio_files': audio_files,
                    'image_files': image_files,
                    'analysis_date': datetime.now().isoformat()
                },
                'individual_results': batch_results
            }
            # NumPy 타입 변환 후 JSON 직렬화
            converted_batch_data = convert_numpy_types(batch_data)
            batch_json = json.dumps(converted_batch_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                "📄 JSON 형식 다운로드",
                data=batch_json,
                file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="모든 분석 결과를 JSON 형식으로 다운로드"
            )
        
        with col2:
            # 텍스트만 다운로드
            text_content = "\n\n" + "="*50 + "\n"
            text_content += f"솔로몬드 AI 배치 분석 결과\n"
            text_content += f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            text_content += f"총 파일: {total_files}개, 성공: {successful_files}개\n"
            text_content += "="*50 + "\n\n"
            
            for i, result in enumerate(successful_results, 1):
                text_content += f"{i}. {result.get('file_name', f'파일 {i}')} ({result.get('file_type', '').upper()})\n"
                text_content += "-" * 30 + "\n"
                if result.get('full_text'):
                    text_content += result['full_text'] + "\n"
                if result.get('summary'):
                    text_content += f"\n[요약] {result['summary']}\n"
                if result.get('jewelry_keywords'):
                    text_content += f"\n[키워드] {', '.join(result['jewelry_keywords'])}\n"
                text_content += "\n" + "="*50 + "\n\n"
            
            st.download_button(
                "📝 텍스트 형식 다운로드",
                data=text_content,
                file_name=f"batch_texts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                help="추출된 텍스트만 텍스트 파일로 다운로드"
            )

    def display_system_status(self):
        """시스템 상태 표시"""
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if REAL_ANALYSIS_AVAILABLE:
                st.success("✅ 실제 분석 엔진")
            else:
                st.error("❌ 실제 분석 엔진")
        
        with col2:
            try:
                import whisper
                st.success("✅ Whisper STT")
            except ImportError:
                st.error("❌ Whisper STT")
        
        with col3:
            try:
                import easyocr
                st.success("✅ EasyOCR")
            except ImportError:
                st.error("❌ EasyOCR")
        
        with col4:
            try:
                from transformers import pipeline
                st.success("✅ Transformers")
            except ImportError:
                st.warning("⚠️ Transformers")
        
        with col5:
            st.markdown("**📈 성능**")
            if PERFORMANCE_MONITOR_AVAILABLE:
                try:
                    success_rate = get_current_success_rate()
                    if success_rate["total_analyses"] > 0:
                        rate = success_rate["success_rate"]
                        if rate >= 85:
                            st.success(f"✅ {rate}%")
                        elif rate >= 70:
                            st.warning(f"⚠️ {rate}%")
                        else:
                            st.error(f"❌ {rate}%")
                        st.caption(f"총 {success_rate['total_analyses']}개")
                    else:
                        st.info("📊 분석 대기")
                except Exception:
                    st.caption("⚠️ 모니터링 오류")
            else:
                st.caption("❌ 모니터링 불가")
    
    def render_audio_analysis_tab(self):
        """음성 분석 탭"""
        
        st.markdown("## 🎤 실제 음성 분석 (Whisper STT)")
        
        if not REAL_ANALYSIS_AVAILABLE:
            st.error("❌ 실제 분석 엔진이 로드되지 않았습니다.")
            return
        
        # 파일 업로드
        uploaded_file = st.file_uploader(
            "음성 파일 업로드",
            type=['wav', 'mp3', 'flac', 'm4a', 'mp4'],
            help="지원 형식: WAV, MP3, FLAC, M4A, MP4"
        )
        
        if uploaded_file is not None:
            # 파일 정보 표시
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📁 파일: {uploaded_file.name} ({file_size:.2f} MB)")
            
            # 분석 설정
            col1, col2 = st.columns(2)
            with col1:
                language = st.selectbox(
                    "언어 선택",
                    ["ko", "en", "auto"],
                    help="ko: 한국어, en: 영어, auto: 자동 감지"
                )
            
            with col2:
                whisper_model = st.selectbox(
                    "Whisper 모델",
                    ["tiny", "base", "small", "medium"],
                    index=1,
                    help="tiny: 빠름, medium: 정확"
                )
            
            # 분석 시작
            if st.button("🎯 실제 음성 분석 시작", type="primary"):
                self.process_audio_analysis(uploaded_file, language, whisper_model)
    
    def process_audio_analysis(self, uploaded_file, language: str, model_size: str):
        """실제 음성 분석 처리"""
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            with st.spinner("🎤 Whisper STT 실제 분석 중..."):
                
                # 실제 분석 실행
                start_time = time.time()
                result = self.analysis_engine.analyze_audio_file(tmp_file_path, language)
                processing_time = time.time() - start_time
                
                # 결과 표시
                if result.get("status") == "success":
                    st.success(f"✅ 음성 분석 완료! ({result['processing_time']}초)")
                    
                    # 분석 결과 표시
                    self.display_audio_results(result)
                    
                    # 세션 통계 업데이트
                    self.update_session_stats(processing_time, True)
                    
                    # 결과를 세션에 저장 (NumPy 타입 변환 후)
                    if 'analysis_results' not in st.session_state:
                        st.session_state.analysis_results = []
                    converted_result = convert_numpy_types(result)
                    st.session_state.analysis_results.append(converted_result)
                    
                else:
                    st.error(f"❌ 음성 분석 실패: {result.get('error', 'Unknown error')}")
                    self.update_session_stats(processing_time, False)
                    
        except Exception as e:
            st.error(f"❌ 분석 중 오류: {str(e)}")
            self.logger.error(f"음성 분석 오류: {e}")
        
        finally:
            # 임시 파일 정리
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    def display_audio_results(self, result: Dict[str, Any]):
        """음성 분석 결과 표시 - 사용자 친화적 버전"""
        
        # 🚀 향상된 결과 표시 엔진 사용
        try:
            from core.user_friendly_presenter import show_enhanced_analysis_result
            show_enhanced_analysis_result(result, st.session_state.project_info)
            return
        except ImportError:
            pass  # 기존 방식으로 fallback
        
        # 기존 기본 정보 표시 (fallback)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("처리 시간", f"{result['processing_time']}초")
        
        with col2:
            st.metric("감지 언어", result['detected_language'])
        
        with col3:
            st.metric("텍스트 길이", f"{result['text_length']}자")
        
        with col4:
            st.metric("세그먼트", f"{result['segments_count']}개")
        
        # 💡 핵심 개선: 종합 메시지 분석 표시
        if result.get('comprehensive_messages') and result['comprehensive_messages'].get('status') == 'success':
            st.markdown("### 🎯 **이 사람들이 말한 내용 요약**")
            
            comp_msg = result['comprehensive_messages']
            main_summary = comp_msg.get('main_summary', {})
            
            # 핵심 한 줄 요약
            if main_summary.get('one_line_summary'):
                st.success(f"**📢 핵심 메시지:** {main_summary['one_line_summary']}")
            
            # 고객 상태 및 중요도
            col1, col2 = st.columns(2)
            with col1:
                if main_summary.get('customer_status'):
                    st.info(f"**👤 고객 상태:** {main_summary['customer_status']}")
            with col2:
                if main_summary.get('urgency_indicator'):
                    urgency_colors = {'높음': '🔴', '보통': '🟡', '낮음': '🟢'}
                    urgency_emoji = urgency_colors.get(main_summary['urgency_indicator'], '⚪')
                    st.info(f"**⚡ 긴급도:** {urgency_emoji} {main_summary['urgency_indicator']}")
            
            # 주요 포인트
            if main_summary.get('key_points'):
                st.markdown("**🔍 주요 포인트:**")
                for point in main_summary['key_points'][:3]:  # 상위 3개만
                    st.markdown(f"• {point}")
            
            # 추천 액션
            if main_summary.get('recommended_actions'):
                st.markdown("**💼 추천 액션:**")
                for action in main_summary['recommended_actions']:
                    st.markdown(f"{action}")
            
            # 상세 분석 (접을 수 있게)
            with st.expander("🔬 상세 대화 분석"):
                conv_analysis = comp_msg.get('conversation_analysis', {})
                
                # 화자 분석
                if conv_analysis.get('speakers'):
                    speakers_info = conv_analysis['speakers']
                    st.markdown("**👥 대화 참여자:**")
                    if speakers_info.get('speaker_distribution'):
                        for speaker, count in speakers_info['speaker_distribution'].items():
                            st.markdown(f"• {speaker}: {count}회 발언")
                
                # 대화 의도
                if conv_analysis.get('intent'):
                    intent_info = conv_analysis['intent']
                    st.markdown(f"**🎯 대화 의도:** {intent_info.get('description', '')}")
                    st.markdown(f"**📊 신뢰도:** {intent_info.get('confidence', 0)*100:.0f}%")
        
        # 추출된 텍스트 (기술적 상세정보로 이동)
        with st.expander("📄 추출된 원본 텍스트"):
            st.text_area(
                "전체 텍스트",
                value=result['full_text'],
                height=200,
                disabled=True
            )
        
        # 기존 요약 (fallback)
        if result.get('summary') and not (result.get('comprehensive_messages') and result['comprehensive_messages'].get('status') == 'success'):
            st.markdown("### 📋 AI 요약")
            st.info(result['summary'])
        
        # 주얼리 키워드
        if result.get('jewelry_keywords'):
            st.markdown("### 💎 주얼리 키워드")
            for keyword in result['jewelry_keywords']:
                st.badge(keyword)
        
        # 상세 세그먼트 (확장 가능)
        with st.expander("🔍 상세 세그먼트 정보"):
            for i, segment in enumerate(result.get('segments', []), 1):
                st.write(f"**{i}. [{segment['start']:.1f}s - {segment['end']:.1f}s]**")
                st.write(segment['text'])
                st.write("---")
    
    def render_image_analysis_tab(self):
        """이미지 분석 탭"""
        
        st.markdown("## 🖼️ 실제 이미지 분석 (EasyOCR)")
        
        if not REAL_ANALYSIS_AVAILABLE:
            st.error("❌ 실제 분석 엔진이 로드되지 않았습니다.")
            return
        
        # GPU 메모리 경고
        st.warning("⚠️ GPU 메모리 부족시 CPU 모드로 실행됩니다. 처리 시간이 길어질 수 있습니다.")
        
        # 파일 업로드
        uploaded_file = st.file_uploader(
            "이미지 파일 업로드",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="지원 형식: JPG, PNG, BMP, TIFF"
        )
        
        if uploaded_file is not None:
            # 이미지 미리보기
            st.image(uploaded_file, caption="업로드된 이미지", use_container_width=True)
            
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📁 파일: {uploaded_file.name} ({file_size:.2f} MB)")
            
            # 분석 시작
            if st.button("🎯 실제 OCR 분석 시작", type="primary"):
                self.process_image_analysis(uploaded_file)
    
    def process_image_analysis(self, uploaded_file):
        """실제 이미지 분석 처리"""
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            with st.spinner("🖼️ EasyOCR 실제 분석 중..."):
                
                # CPU 모드 강제 설정 (GPU 메모리 부족 방지)
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                
                # 실제 분석 실행
                start_time = time.time()
                result = self.analysis_engine.analyze_image_file(tmp_file_path)
                processing_time = time.time() - start_time
                
                # 결과 표시
                if result.get("status") == "success":
                    st.success(f"✅ 이미지 분석 완료! ({result['processing_time']}초)")
                    
                    # 분석 결과 표시
                    self.display_image_results(result)
                    
                    # 세션 통계 업데이트
                    self.update_session_stats(processing_time, True)
                    
                    # 결과를 세션에 저장 (NumPy 타입 변환 후)
                    if 'analysis_results' not in st.session_state:
                        st.session_state.analysis_results = []
                    converted_result = convert_numpy_types(result)
                    st.session_state.analysis_results.append(converted_result)
                    
                else:
                    st.error(f"❌ 이미지 분석 실패: {result.get('error', 'Unknown error')}")
                    self.update_session_stats(processing_time, False)
                    
        except Exception as e:
            st.error(f"❌ 분석 중 오류: {str(e)}")
            self.logger.error(f"이미지 분석 오류: {e}")
        
        finally:
            # 임시 파일 정리
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    def display_image_results(self, result: Dict[str, Any]):
        """이미지 분석 결과 표시 - 사용자 친화적 버전"""
        
        # 🚀 향상된 결과 표시 엔진 사용
        try:
            from core.user_friendly_presenter import show_enhanced_analysis_result
            show_enhanced_analysis_result(result, st.session_state.project_info)
            return
        except ImportError:
            pass  # 기존 방식으로 fallback
        
        # 기존 기본 정보 표시 (fallback)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("처리 시간", f"{result['processing_time']}초")
        
        with col2:
            st.metric("텍스트 블록", f"{result['blocks_detected']}개")
        
        with col3:
            st.metric("평균 신뢰도", f"{result['average_confidence']:.3f}")
        
        with col4:
            st.metric("파일 크기", f"{result['file_size_mb']} MB")
        
        # 💡 핵심 개선: 종합 메시지 분석 표시
        if result.get('comprehensive_messages') and result['comprehensive_messages'].get('status') == 'success':
            st.markdown("### 🎯 **이미지에서 추출한 핵심 내용**")
            
            comp_msg = result['comprehensive_messages']
            main_summary = comp_msg.get('main_summary', {})
            
            # 핵심 한 줄 요약
            if main_summary.get('one_line_summary'):
                st.success(f"**📢 핵심 메시지:** {main_summary['one_line_summary']}")
            
            # 고객 상태 및 중요도
            col1, col2 = st.columns(2)
            with col1:
                if main_summary.get('customer_status'):
                    st.info(f"**👤 고객 상태:** {main_summary['customer_status']}")
            with col2:
                if main_summary.get('urgency_indicator'):
                    urgency_colors = {'높음': '🔴', '보통': '🟡', '낮음': '🟢'}
                    urgency_emoji = urgency_colors.get(main_summary['urgency_indicator'], '⚪')
                    st.info(f"**⚡ 긴급도:** {urgency_emoji} {main_summary['urgency_indicator']}")
            
            # 주요 포인트
            if main_summary.get('key_points'):
                st.markdown("**🔍 주요 포인트:**")
                for point in main_summary['key_points'][:3]:  # 상위 3개만
                    st.markdown(f"• {point}")
            
            # 추천 액션
            if main_summary.get('recommended_actions'):
                st.markdown("**💼 추천 액션:**")
                for action in main_summary['recommended_actions']:
                    st.markdown(f"{action}")
        
        # 추출된 텍스트 (기술적 상세정보로 이동)
        with st.expander("📄 추출된 원본 텍스트"):
            st.text_area(
                "OCR 결과",
                value=result['full_text'],
                height=150,
                disabled=True
            )
        
        # 기존 요약 (fallback)
        if result.get('summary') and not (result.get('comprehensive_messages') and result['comprehensive_messages'].get('status') == 'success'):
            st.markdown("### 📋 AI 요약")
            st.info(result['summary'])
        
        # 주얼리 키워드
        if result.get('jewelry_keywords'):
            st.markdown("### 💎 주얼리 키워드")
            for keyword in result['jewelry_keywords']:
                st.badge(keyword)
        
        # 상세 결과 (확장 가능)
        with st.expander("🔍 상세 OCR 결과"):
            for i, item in enumerate(result.get('detailed_results', []), 1):
                st.write(f"**{i}. 신뢰도: {item['confidence']:.3f}**")
                st.write(f"텍스트: {item['text']}")
                st.write("---")
    
    def render_results_tab(self):
        """분석 결과 탭"""
        
        st.markdown("## 📊 분석 결과 모음")
        
        # 세션 통계
        self.display_session_stats()
        
        # 저장된 결과들
        if 'analysis_results' in st.session_state and st.session_state.analysis_results:
            
            st.markdown("### 📋 분석 기록")
            
            # 결과 필터링 옵션
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_type = st.selectbox(
                    "파일 타입 필터",
                    ["전체", "음성", "이미지"],
                    help="특정 타입의 결과만 표시"
                )
            
            with col2:
                show_mode = st.selectbox(
                    "표시 모드",
                    ["요약", "전체 텍스트"],
                    help="텍스트 표시 방식 선택"
                )
            
            with col3:
                results_per_page = st.selectbox(
                    "페이지당 결과 수",
                    [5, 10, 20, "전체"],
                    index=1
                )
            
            # 필터링된 결과
            filtered_results = st.session_state.analysis_results
            if filter_type == "음성":
                filtered_results = [r for r in filtered_results if r.get('analysis_type') == 'audio' or r.get('file_type') == 'audio']
            elif filter_type == "이미지":
                filtered_results = [r for r in filtered_results if r.get('analysis_type') == 'image' or r.get('file_type') == 'image']
            
            # 페이징 처리
            if results_per_page != "전체":
                page_size = int(results_per_page)
                total_pages = (len(filtered_results) + page_size - 1) // page_size
                if total_pages > 1:
                    page_num = st.selectbox(f"페이지 (총 {total_pages}페이지)", range(1, total_pages + 1))
                    start_idx = (page_num - 1) * page_size
                    end_idx = start_idx + page_size
                    filtered_results = filtered_results[start_idx:end_idx]
            
            # 전체 다운로드 버튼 (상단에 배치)
            if len(st.session_state.analysis_results) > 1:
                st.markdown("### 📥 전체 결과 다운로드")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # 전체 JSON 다운로드 (NumPy 타입 변환)
                    all_results_data = {
                        'export_info': {
                            'total_results': len(st.session_state.analysis_results),
                            'export_date': datetime.now().isoformat(),
                            'export_source': '솔로몬드 AI v2.3'
                        },
                        'results': st.session_state.analysis_results
                    }
                    # NumPy 타입 변환 후 JSON 직렬화
                    converted_all_results = convert_numpy_types(all_results_data)
                    all_results_json = json.dumps(converted_all_results, indent=2, ensure_ascii=False)
                    
                    st.download_button(
                        "📄 전체 JSON 다운로드",
                        data=all_results_json,
                        file_name=f"solomond_all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col2:
                    # 전체 텍스트 다운로드
                    all_text_content = f"솔로몬드 AI v2.3 - 전체 분석 결과\n"
                    all_text_content += f"내보내기 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    all_text_content += f"총 결과 수: {len(st.session_state.analysis_results)}개\n"
                    all_text_content += "=" * 60 + "\n\n"
                    
                    for i, result in enumerate(st.session_state.analysis_results, 1):
                        all_text_content += f"{i}. {result.get('file_name', f'파일 {i}')}\n"
                        all_text_content += f"타입: {result.get('analysis_type', result.get('file_type', 'Unknown'))}\n"
                        all_text_content += f"분석 시간: {result.get('timestamp', 'N/A')}\n"
                        all_text_content += "-" * 40 + "\n"
                        
                        if result.get('full_text'):
                            all_text_content += "[추출된 텍스트]\n"
                            all_text_content += result['full_text'] + "\n\n"
                        
                        if result.get('summary'):
                            all_text_content += "[AI 요약]\n"
                            all_text_content += result['summary'] + "\n\n"
                        
                        if result.get('jewelry_keywords'):
                            all_text_content += "[주얼리 키워드]\n"
                            all_text_content += ", ".join(result['jewelry_keywords']) + "\n\n"
                        
                        all_text_content += "=" * 60 + "\n\n"
                    
                    st.download_button(
                        "📝 전체 텍스트 다운로드",
                        data=all_text_content,
                        file_name=f"solomond_all_texts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                with col3:
                    # 결과 초기화
                    if st.button("🗑️ 모든 결과 초기화"):
                        st.session_state.analysis_results = []
                        st.rerun()
            
            st.markdown("---")
            
            # 개별 결과 표시
            for i, result in enumerate(filtered_results):
                
                # 파일 타입 결정
                file_type = result.get('analysis_type', result.get('file_type', 'unknown'))
                type_icon = "🎤" if file_type in ['audio', 'Audio'] else "🖼️" if file_type in ['image', 'Image'] else "📄"
                
                with st.expander(f"{type_icon} {result.get('file_name', f'파일 {i+1}')} - {file_type.upper()}"):
                    
                    # 기본 정보
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**📁 파일:** {result.get('file_name', 'Unknown')}")
                        st.write(f"**📊 타입:** {file_type.upper()}")
                    
                    with col2:
                        timestamp = result.get('timestamp', 'N/A')
                        st.write(f"**🕐 분석 시간:** {timestamp}")
                        
                        # 처리 시간 안전 처리
                        processing_time = result.get('processing_time')
                        if processing_time is not None:
                            st.write(f"**⏱️ 처리 시간:** {processing_time}초")
                        else:
                            alt_time = result.get('duration') or result.get('elapsed_time') or result.get('execution_time')
                            if alt_time:
                                st.write(f"**⏱️ 처리 시간:** {alt_time}초")
                            else:
                                st.write("**⏱️ 처리 시간:** 측정되지 않음")
                    
                    with col3:
                        # 개별 다운로드 버튼 (NumPy 타입 변환)
                        converted_result = convert_numpy_types(result)
                        json_str = json.dumps(converted_result, indent=2, ensure_ascii=False)
                        st.download_button(
                            "📥 개별 다운로드",
                            data=json_str,
                            file_name=f"analysis_{result.get('file_name', f'result_{i}')}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            mime="application/json",
                            key=f"download_{i}"
                        )
                    
                    # 추출된 텍스트 표시
                    if result.get('full_text'):
                        st.markdown("### 📄 추출된 텍스트")
                        
                        full_text = result['full_text']
                        
                        if show_mode == "요약" and len(full_text) > 500:
                            # 요약 모드: 처음 500자만 표시
                            preview_text = full_text[:500] + "..."
                            st.text_area(
                                "텍스트 미리보기 (처음 500자)",
                                value=preview_text,
                                height=150,
                                disabled=True,
                                key=f"text_preview_{i}"
                            )
                            
                            # 전체 텍스트 보기 버튼
                            if st.button(f"📖 전체 텍스트 보기 ({len(full_text)}자)", key=f"show_full_{i}"):
                                st.text_area(
                                    "전체 텍스트",
                                    value=full_text,
                                    height=300,
                                    disabled=True,
                                    key=f"text_full_{i}"
                                )
                        else:
                            # 전체 텍스트 모드 또는 짧은 텍스트
                            st.text_area(
                                f"전체 텍스트 ({len(full_text)}자)",
                                value=full_text,
                                height=200 if len(full_text) < 1000 else 300,
                                disabled=True,
                                key=f"text_full_{i}"
                            )
                    
                    # 요약 표시
                    if result.get('summary'):
                        st.markdown("### 📋 AI 요약")
                        st.info(result['summary'])
                    
                    # 주얼리 키워드 표시
                    if result.get('jewelry_keywords'):
                        st.markdown("### 💎 주얼리 키워드")
                        # 키워드를 배지 스타일로 표시
                        keywords_html = ""
                        for keyword in result['jewelry_keywords']:
                            keywords_html += f'<span style="background-color: #e1f5fe; color: #01579b; padding: 2px 8px; margin: 2px; border-radius: 12px; font-size: 12px;">{keyword}</span> '
                        st.markdown(keywords_html, unsafe_allow_html=True)
                    
                    # 추가 정보 (확장 가능)
                    with st.expander("🔍 상세 정보"):
                        # 오디오 관련 정보
                        if file_type in ['audio', 'Audio']:
                            if result.get('detected_language'):
                                st.write(f"**감지된 언어:** {result['detected_language']}")
                            if result.get('segments_count'):
                                st.write(f"**세그먼트 수:** {result['segments_count']}개")
                            if result.get('text_length'):
                                st.write(f"**텍스트 길이:** {result['text_length']}자")
                        
                        # 이미지 관련 정보
                        elif file_type in ['image', 'Image']:
                            if result.get('blocks_detected'):
                                st.write(f"**감지된 텍스트 블록:** {result['blocks_detected']}개")
                            if result.get('average_confidence'):
                                st.write(f"**평균 신뢰도:** {result['average_confidence']:.3f}")
                            if result.get('file_size_mb'):
                                st.write(f"**파일 크기:** {result['file_size_mb']} MB")
                        
                        # 기타 정보
                        other_info = {k: v for k, v in result.items() 
                                    if k not in ['file_name', 'analysis_type', 'file_type', 'timestamp', 
                                                'processing_time', 'full_text', 'summary', 'jewelry_keywords',
                                                'detected_language', 'segments_count', 'text_length',
                                                'blocks_detected', 'average_confidence', 'file_size_mb']}
                        
                        if other_info:
                            st.json(other_info)
        
        else:
            st.info("📝 아직 분석 결과가 없습니다. 음성 또는 이미지 분석을 실행해보세요.")
            
            # 사용법 안내
            st.markdown("### 💡 사용법 안내")
            with st.expander("📖 분석 결과 확인 방법"):
                st.markdown("""
                **1. 분석 실행 후 결과 확인:**
                - 멀티파일 분석, 음성 분석, 이미지 분석을 실행하면 여기에 결과가 저장됩니다.
                
                **2. 결과 필터링:**
                - 파일 타입별로 결과를 필터링할 수 있습니다.
                - 표시 모드를 선택하여 요약 또는 전체 텍스트를 볼 수 있습니다.
                
                **3. 결과 다운로드:**
                - 개별 결과를 JSON 형식으로 다운로드할 수 있습니다.
                - 전체 결과를 JSON 또는 텍스트 형식으로 일괄 다운로드할 수 있습니다.
                
                **4. 결과 관리:**
                - 불필요한 결과는 "모든 결과 초기화" 버튼으로 삭제할 수 있습니다.
                """)
    
    def display_session_stats(self):
        """세션 통계 표시"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("분석한 파일", f"{self.session_stats['files_analyzed']}개")
        
        with col2:
            st.metric("성공한 분석", f"{self.session_stats['successful_analyses']}개")
        
        with col3:
            success_rate = 0
            if self.session_stats['files_analyzed'] > 0:
                success_rate = (self.session_stats['successful_analyses'] / self.session_stats['files_analyzed']) * 100
            st.metric("성공률", f"{success_rate:.1f}%")
        
        with col4:
            st.metric("총 처리 시간", f"{self.session_stats['total_processing_time']:.1f}초")
    
    def render_settings_tab(self):
        """설정 탭"""
        
        st.markdown("## ⚙️ 시스템 설정")
        
        # 분석 엔진 통계
        if REAL_ANALYSIS_AVAILABLE and self.analysis_engine:
            st.markdown("### 📊 분석 엔진 통계")
            
            try:
                stats = self.analysis_engine.get_analysis_stats()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.json({
                        "전체 분석 파일": stats.get('total_files', 0),
                        "성공한 분석": stats.get('successful_analyses', 0),
                        "성공률": f"{stats.get('success_rate', 0):.1f}%"
                    })
                
                with col2:
                    st.json({
                        "총 처리 시간": f"{stats.get('total_processing_time', 0):.1f}초",
                        "평균 처리 시간": f"{stats.get('average_processing_time', 0):.1f}초",
                        "마지막 분석": stats.get('last_analysis_time', 'N/A')
                    })
            
            except Exception as e:
                st.error(f"통계 로드 실패: {e}")
        
        # 시스템 정보
        st.markdown("### 🖥️ 시스템 정보")
        
        system_info = {
            "실제 분석 엔진": "✅ 사용 가능" if REAL_ANALYSIS_AVAILABLE else "❌ 사용 불가",
            "Whisper STT": "✅ 설치됨" if self._check_module('whisper') else "❌ 미설치",
            "EasyOCR": "✅ 설치됨" if self._check_module('easyocr') else "❌ 미설치",
            "Transformers": "✅ 설치됨" if self._check_module('transformers') else "❌ 미설치",
            "Google Gemini": "✅ 설치됨" if self._check_module('google.generativeai') else "❌ 미설치"
        }
        
        st.json(system_info)
        
        # 메모리 최적화 옵션
        st.markdown("### 🔧 성능 최적화")
        
        if st.button("🗑️ GPU 메모리 정리"):
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    st.success("✅ GPU 메모리가 정리되었습니다.")
                else:
                    st.info("ℹ️ CUDA가 사용 불가능합니다.")
            except ImportError:
                st.warning("⚠️ PyTorch가 설치되지 않았습니다.")
    
    def _check_module(self, module_name: str) -> bool:
        """모듈 설치 여부 확인"""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def update_session_stats(self, processing_time: float, success: bool):
        """세션 통계 업데이트"""
        self.session_stats['files_analyzed'] += 1
        self.session_stats['total_processing_time'] += processing_time
        if success:
            self.session_stats['successful_analyses'] += 1
    
    def _check_module_availability(self, module_name):
        """모듈 가용성 체크"""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def check_analysis_readiness(self):
        """분석 시스템 준비 상태 체크"""
        dependency_status = {
            'whisper': self._check_module_availability('whisper'),
            'easyocr': self._check_module_availability('easyocr'),
            'transformers': self._check_module_availability('transformers'),
            'numpy': self._check_module_availability('numpy'),
            'librosa': self._check_module_availability('librosa'),
            'ffmpeg': self._check_ffmpeg_availability()
        }
        
        # 필수 의존성 확인 (whisper, easyocr는 필수)
        critical_dependencies = ['whisper', 'easyocr', 'numpy']
        analysis_ready = all(dependency_status.get(dep, False) for dep in critical_dependencies)
        
        return analysis_ready, dependency_status
    
    def _check_ffmpeg_availability(self):
        """FFmpeg 설치 상태 확인"""
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False
    
    def _preload_analysis_models(self):
        """분석 모델들을 사전 로딩하여 실제 분석 시 지연 최소화"""
        try:
            if REAL_ANALYSIS_AVAILABLE and self.analysis_engine:
                # 진행 상황을 위한 임시 컨테이너
                model_status = st.empty()
                
                # Whisper 모델 로딩
                model_status.text("🎤 Whisper STT 모델 로딩 중...")
                if not self.analysis_engine.whisper_model:
                    self.analysis_engine._lazy_load_whisper()
                
                # EasyOCR 모델 로딩  
                model_status.text("🖼️ EasyOCR 모델 로딩 중...")
                if not self.analysis_engine.ocr_reader:
                    self.analysis_engine._lazy_load_ocr()
                
                # NLP 모델 로딩 (선택적)
                model_status.text("🧠 NLP 모델 로딩 중...")
                self.analysis_engine._lazy_load_nlp()
                
                model_status.text("✅ 모든 모델 준비 완료!")
                import time
                time.sleep(0.5)  # 사용자가 메시지를 볼 수 있도록
                model_status.empty()
                
        except Exception as e:
            self.logger.warning(f"모델 사전 로딩 중 오류: {e}")
            # 오류가 있어도 분석은 계속 진행 (lazy loading 방식으로)
    
    def execute_comprehensive_analysis(self):
        """🚀 배치 종합 분석 실행 - 모든 파일을 통합 분석"""
        if not st.session_state.uploaded_files_data:
            return []
        
        uploaded_files_data = st.session_state.uploaded_files_data
        
        # 🎯 배치 분석 vs 개별 분석 선택
        enable_batch_analysis = st.session_state.project_info.get('correlation_analysis', True)
        
        if enable_batch_analysis:
            st.success("🚀 **배치 종합 분석 시작**: 모든 파일을 통합하여 최고 품질의 분석을 수행합니다")
            with st.container():
                st.markdown("### 📊 배치 분석 진행 상황")
                return self._execute_batch_comprehensive_analysis()
        else:
            st.warning("📁 **개별 분석 모드**: 파일별로 독립적으로 분석합니다 (품질 제한적)")
            return self._execute_individual_analysis()
    
    def _execute_batch_comprehensive_analysis(self):
        """배치 종합 분석 - 모든 파일을 통합 처리"""
        uploaded_files_data = st.session_state.uploaded_files_data
        all_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1️⃣ 단계: 파일 분류 및 전처리
        status_text.text("🔍 1단계: 파일 분류 및 전처리 중...")
        file_categories = self._categorize_and_preprocess_files(uploaded_files_data)
        progress_bar.progress(0.2)
        
        # 2️⃣ 단계: 통합 컨텍스트 구성
        status_text.text("🧠 2단계: 통합 컨텍스트 구성 중...")
        integrated_context = self._build_integrated_context(file_categories)
        progress_bar.progress(0.4)
        
        # 3️⃣ 단계: 배치 분석 실행
        status_text.text("⚡ 3단계: 배치 통합 분석 실행 중...")
        batch_results = self._execute_batch_analysis(file_categories, integrated_context)
        progress_bar.progress(0.8)
        
        # 4️⃣ 단계: 결과 통합 및 최적화
        status_text.text("🎯 4단계: 결과 통합 및 최적화 중...")
        final_results = self._integrate_and_optimize_results(batch_results, integrated_context)
        progress_bar.progress(1.0)
        
        status_text.text("✅ 배치 종합 분석 완료!")
        return final_results
    
    def _execute_individual_analysis(self):
        """기존 개별 분석 방식 (호환성 유지)"""
        uploaded_files_data = st.session_state.uploaded_files_data
        all_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 모델은 실제 사용 시점에 lazy loading으로 로딩 (서버 시작 시간 단축)
        status_text.text("🔧 분석 준비 중...")
        
        total_items = len(uploaded_files_data.get('files', [])) + len(uploaded_files_data.get('video_urls', []))
        current_item = 0
        
        # 업로드된 파일 분석
        for uploaded_file in uploaded_files_data.get('files', []):
            current_item += 1
            progress_bar.progress(current_item / total_items)
            status_text.text(f"🔄 분석 중: {uploaded_file.name} ({current_item}/{total_items})")
            
            tmp_file_path = None
            audio_file_path = None
            is_large_video = False
            try:
                # 파일 타입 결정
                file_ext = uploaded_file.name.split('.')[-1].lower()
                file_size_gb = len(uploaded_file.getvalue()) / (1024 * 1024 * 1024)
                is_large_video = file_ext in ['mp4', 'mov', 'avi'] and file_size_gb >= 1.0
                
                if file_ext in ['wav', 'mp3', 'flac', 'm4a']:
                    file_type = "audio"
                elif file_ext in ['mp4', 'mov', 'avi']:
                    file_type = "video"
                elif file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
                    file_type = "image"
                else:
                    file_type = "unknown"
                
                # 대용량 비디오 파일 처리
                if is_large_video and LARGE_FILE_HANDLER_AVAILABLE:
                    status_text.text(f"🚀 대용량 파일 처리 중: {uploaded_file.name} ({file_size_gb:.2f}GB)")
                    
                    # 청크 단위로 저장
                    def progress_callback(progress):
                        progress_bar.progress((current_item - 1 + progress * 0.5) / total_items)
                        status_text.text(f"📥 업로드 중: {uploaded_file.name} ({progress*100:.1f}%)")
                    
                    tmp_file_path = large_file_handler.save_uploaded_file_chunked(uploaded_file, progress_callback)
                    
                    if tmp_file_path:
                        # 동영상에서 오디오 추출
                        status_text.text(f"🎵 오디오 추출 중: {uploaded_file.name}")
                        audio_file_path = large_file_handler.extract_audio_from_video(tmp_file_path)
                        
                        if audio_file_path:
                            # 추출된 오디오로 변경하여 분석
                            tmp_file_path = audio_file_path
                            file_type = "audio"
                            status_text.text(f"🔄 오디오 분석 중: {uploaded_file.name}")
                        else:
                            raise Exception("동영상에서 오디오 추출 실패")
                    else:
                        raise Exception("대용량 파일 저장 실패")
                        
                # 일반 파일 처리
                else:
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                        
                    # 작은 동영상 파일도 오디오 추출 필요
                    if file_type == "video":
                        if LARGE_FILE_HANDLER_AVAILABLE:
                            status_text.text(f"🎵 오디오 추출 중: {uploaded_file.name}")
                            audio_file_path = large_file_handler.extract_audio_from_video(tmp_file_path)
                            if audio_file_path:
                                tmp_file_path = audio_file_path
                                file_type = "audio"
                        else:
                            # FFmpeg 없이는 동영상 직접 처리 불가
                            file_type = "unsupported_video"
                
                # 실제 분석 수행
                if REAL_ANALYSIS_AVAILABLE and file_type in ["audio", "image"]:
                    language = uploaded_files_data.get('analysis_language', 'auto')
                    
                    # 🧠 강화된 프로젝트 컨텍스트 정보 준비
                    context = self._prepare_enhanced_context()
                    
                    # 분석 진행 상황 업데이트 (컨텍스트 정보 포함)
                    self._update_analysis_status(uploaded_file.name, context)
                    
                    result = analyze_file_real(tmp_file_path, file_type, language, context)
                    
                    # NumPy 타입을 Python 기본 타입으로 변환
                    result = convert_numpy_types(result)
                    
                    # 동영상에서 추출된 오디오인 경우 원본 파일명 정보 추가
                    if audio_file_path and result.get('status') == 'success':
                        result['original_video_file'] = uploaded_file.name
                        result['extracted_audio'] = True
                        result['large_file_processed'] = is_large_video
                        
                else:
                    if not REAL_ANALYSIS_AVAILABLE:
                        error_msg = "분석 엔진이 로드되지 않았습니다. 필수 패키지(whisper, easyocr)를 설치해주세요."
                    elif file_type == "unsupported_video":
                        error_msg = "동영상 처리를 위해 FFmpeg가 필요합니다. 대용량 파일 핸들러를 설치해주세요."
                    else:
                        error_msg = f"지원하지 않는 파일 형식: {file_type}. 지원 형식: 음성(wav, mp3, m4a), 동영상(mp4, mov, avi), 이미지(jpg, png, bmp)"
                    
                    result = {
                        "status": "error",
                        "error": error_msg,
                        "file_name": uploaded_file.name,
                        "file_type": file_type,
                        "suggested_action": "FFmpeg 설치 또는 지원되는 형식으로 변환해주세요." if file_type == "unsupported_video" else "파일 형식을 확인하거나 지원되는 형식으로 변환해주세요."
                    }
                
                all_results.append(result)
                    
            except Exception as e:
                self.logger.error(f"파일 분석 실패: {uploaded_file.name}: {e}")
                all_results.append({
                    "status": "error",
                    "error": str(e),
                    "file_name": uploaded_file.name
                })
            finally:
                # 임시 파일들 확실히 정리
                cleanup_files = []
                if tmp_file_path and os.path.exists(tmp_file_path):
                    cleanup_files.append(tmp_file_path)
                if audio_file_path and audio_file_path != tmp_file_path and os.path.exists(audio_file_path):
                    cleanup_files.append(audio_file_path)
                
                for file_path in cleanup_files:
                    try:
                        os.unlink(file_path)
                        self.logger.debug(f"임시 파일 정리 완료: {file_path}")
                    except Exception as cleanup_error:
                        self.logger.warning(f"임시 파일 정리 실패: {file_path}: {cleanup_error}")
                
                # 대용량 파일 핸들러의 정리 작업도 수행
                if LARGE_FILE_HANDLER_AVAILABLE and is_large_video:
                    try:
                        large_file_handler.cleanup_temp_files(max_age_hours=1)  # 1시간 이상 된 파일만 정리
                    except Exception as cleanup_error:
                        self.logger.warning(f"대용량 파일 핸들러 정리 실패: {cleanup_error}")
        
        # 동영상 URL 분석 (YouTube, Brightcove 등)
        for url in uploaded_files_data.get('video_urls', []):
            current_item += 1
            progress_bar.progress(current_item / total_items)
            # URL 타입에 따른 상태 메시지
            if 'youtube.com' in url or 'youtu.be' in url:
                status_text.text(f"🔄 YouTube 분석 중: {url[:50]}... ({current_item}/{total_items})")
            elif 'brightcove.net' in url:
                status_text.text(f"🔄 Brightcove 분석 중: {url[:50]}... ({current_item}/{total_items})")
            else:
                status_text.text(f"🔄 동영상 URL 분석 중: {url[:50]}... ({current_item}/{total_items})")
            
            # YouTube 분석은 향후 구현 예정
            all_results.append({
                "status": "pending",
                "message": "YouTube 분석 기능은 향후 구현 예정입니다.",
                "url": url
            })
        
        progress_bar.progress(1.0)
        status_text.text("✅ 모든 분석 완료!")
        
        return all_results
    
    def generate_final_report(self):
        """최종 분석 보고서 생성"""
        if not st.session_state.analysis_results:
            return None
        
        results = st.session_state.analysis_results
        project_info = st.session_state.project_info
        
        # 기본 통계 계산
        total_files = len(results)
        successful_analyses = len([r for r in results if r.get('status') == 'success'])
        success_rate = (successful_analyses / total_files * 100) if total_files > 0 else 0
        
        # 총 처리 시간 계산
        total_time = sum([r.get('processing_time', 0) for r in results if r.get('processing_time')])
        
        # 모든 텍스트 수집
        all_texts = []
        for result in results:
            if result.get('status') == 'success' and result.get('full_text'):
                all_texts.append(result['full_text'])
        
        combined_text = ' '.join(all_texts)
        
        # 키워드 추출 및 빈도 계산
        import re
        from collections import Counter
        
        # 한국어와 영어 키워드 추출 (2글자 이상)
        korean_words = re.findall(r'[가-힣]{2,}', combined_text)
        english_words = re.findall(r'[A-Za-z]{3,}', combined_text.lower())
        
        all_keywords = korean_words + english_words
        keyword_freq = Counter(all_keywords)
        top_keywords = keyword_freq.most_common(20)
        
        # 주얼리 관련 키워드 수집
        jewelry_keywords = []
        for result in results:
            if result.get('jewelry_keywords'):
                jewelry_keywords.extend(result['jewelry_keywords'])
        
        unique_jewelry_keywords = list(set(jewelry_keywords))
        
        # 핵심 요약 생성
        executive_summary = self._generate_executive_summary(
            total_files, successful_analyses, success_rate, 
            len(combined_text), unique_jewelry_keywords
        )
        
        # 주요 발견사항 생성
        key_findings = self._generate_key_findings(results, unique_jewelry_keywords)
        
        # 결론 및 제안사항 생성
        conclusions = self._generate_conclusions(results, success_rate, unique_jewelry_keywords)
        
        # 최종 보고서 구조 생성
        report = {
            'project_name': project_info.get('project_name', '분석 프로젝트'),
            'analysis_date': datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분'),
            'total_files': total_files,
            'successful_analyses': successful_analyses,
            'success_rate': success_rate,
            'total_time': total_time,
            'total_text_length': len(combined_text),
            'executive_summary': executive_summary,
            'key_findings': key_findings,
            'top_keywords': top_keywords,
            'jewelry_keywords': unique_jewelry_keywords,
            'conclusions': conclusions,
            'analysis_details': {
                'audio_files': len([r for r in results if r.get('analysis_type') == 'real_whisper_stt']),
                'image_files': len([r for r in results if r.get('analysis_type') == 'real_easyocr']),
                'total_processing_time': total_time,
                'average_confidence': self._calculate_average_confidence(results)
            }
        }
        
        return report
    
    def generate_comprehensive_lecture(self):
        """종합 강의 내용 생성"""
        if not st.session_state.analysis_results:
            return None
        
        if not LECTURE_COMPILER_AVAILABLE:
            st.error("강의 내용 컴파일러가 사용 불가능합니다.")
            return None
        
        try:
            # 분석 결과들 준비
            analysis_results = st.session_state.analysis_results
            
            # 성공한 결과만 필터링 (부분 성공 포함)
            valid_results = [
                result for result in analysis_results 
                if result.get('status') in ['success', 'partial_success']
            ]
            
            if not valid_results:
                st.warning("강의 내용을 생성할 유효한 분석 결과가 없습니다.")
                return None
            
            # 프로젝트 정보에서 제목 가져오기
            project_info = st.session_state.get('project_info', {})
            custom_title = project_info.get('project_name')
            
            # 강의 내용 컴파일
            with st.spinner("🎓 종합 강의 내용 생성 중..."):
                lecture_result = compile_comprehensive_lecture(valid_results, custom_title)
            
            if lecture_result.get('status') == 'success':
                return lecture_result['lecture_content']
            else:
                st.error(f"강의 내용 생성 실패: {lecture_result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            st.error(f"강의 내용 생성 중 오류: {str(e)}")
            return None
    
    def _generate_executive_summary(self, total_files, successful, success_rate, text_length, jewelry_keywords):
        """핵심 요약 생성"""
        # 추출된 텍스트의 첫 500자를 미리보기로 추가
        results = st.session_state.analysis_results
        content_preview = ""
        main_language = "한국어"
        
        # 실제 추출된 텍스트에서 주요 내용 파악
        all_texts = []
        for result in results:
            if result.get('status') == 'success' and result.get('full_text'):
                text = result['full_text'].strip()
                if text:
                    all_texts.append(text)
                    # 첫 번째 의미있는 텍스트를 미리보기로 사용
                    if not content_preview and len(text) > 10:
                        content_preview = text[:300] + "..." if len(text) > 300 else text
                        # 언어 감지
                        import re
                        korean_chars = len(re.findall(r'[가-힣]', text))
                        english_chars = len(re.findall(r'[a-zA-Z]', text))
                        if english_chars > korean_chars:
                            main_language = "영어"
        
        # 주제 및 키워드 기반 컨텐츠 분류
        combined_text = ' '.join(all_texts).lower()
        content_type = "일반 내용"
        
        if any(keyword in combined_text for keyword in ['seminar', '세미나', 'conference', '컨퍼런스', 'presentation', '발표']):
            content_type = "세미나/컨퍼런스"
        elif any(keyword in combined_text for keyword in ['jewelry', '주얼리', 'diamond', '다이아몬드', 'gold', '금']):
            content_type = "주얼리 관련"
        elif any(keyword in combined_text for keyword in ['business', '비즈니스', 'market', '시장', 'trend', '트렌드']):
            content_type = "비즈니스 분석"
        
        summary_parts = [
            f"📊 **분석 개요**: 총 {total_files}개 파일 중 {successful}개 파일에서 성공적으로 데이터를 추출했습니다. (성공률: {success_rate:.1f}%)",
            f"📝 **추출된 내용**: {main_language} 텍스트 {text_length:,}자 분량의 {content_type} 내용이 감지되었습니다."
        ]
        
        if content_preview:
            summary_parts.append(f"🎯 **주요 내용 미리보기**: \"{content_preview}\"")
        
        if jewelry_keywords:
            summary_parts.append(f"💎 **주얼리 키워드**: {len(jewelry_keywords)}개의 관련 키워드가 발견되었습니다. ({', '.join(jewelry_keywords[:3])}...)")
        
        if success_rate >= 90:
            summary_parts.append("✅ **품질 평가**: 분석 품질이 매우 우수합니다.")
        elif success_rate >= 70:
            summary_parts.append("⚠️ **품질 평가**: 분석 품질이 양호합니다.")
        else:
            summary_parts.append("❌ **품질 평가**: 일부 파일에서 분석 어려움이 있었습니다.")
        
        return '\n\n'.join(summary_parts)
    
    def _generate_key_findings(self, results, jewelry_keywords):
        """주요 발견사항 생성"""
        findings = []
        
        # 파일 형식별 분석
        audio_results = [r for r in results if r.get('analysis_type') == 'real_whisper_stt']
        image_results = [r for r in results if r.get('analysis_type') == 'real_easyocr']
        
        # 실제 추출된 텍스트 내용 분석
        all_successful_texts = []
        detected_languages = []
        
        for result in results:
            if result.get('status') == 'success' and result.get('full_text'):
                text = result['full_text'].strip()
                if text:
                    all_successful_texts.append(text)
                    # 언어 감지 정보 수집
                    if result.get('detected_language'):
                        detected_languages.append(result['detected_language'])
        
        if all_successful_texts:
            combined_content = ' '.join(all_successful_texts)
            
            # 언어 분석
            if detected_languages:
                lang_counts = {}
                for lang in detected_languages:
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                main_lang = max(lang_counts.items(), key=lambda x: x[1])[0]
                findings.append(f"🗣️ **주요 언어**: {main_lang} ({lang_counts[main_lang]}개 파일)")
            
            # 내용 길이 및 품질 분석
            total_length = len(combined_content)
            word_count = len(combined_content.split())
            findings.append(f"📝 **텍스트 분량**: 총 {total_length:,}자, 약 {word_count:,}단어 추출")
            
            # 주요 주제 키워드 분석 (한국어 + 영어)
            import re
            from collections import Counter
            
            # 한국어 명사 추출 (2글자 이상)
            korean_words = re.findall(r'[가-힣]{2,}', combined_content)
            # 영어 단어 추출 (3글자 이상)
            english_words = re.findall(r'[A-Za-z]{3,}', combined_content.lower())
            
            # 빈도 분석
            all_words = korean_words + english_words
            if all_words:
                word_freq = Counter(all_words)
                top_words = word_freq.most_common(8)
                if top_words:
                    findings.append(f"🔍 **핵심 키워드**: {', '.join([f'{word}({count})' for word, count in top_words[:5]])}")
            
            # 특정 주제 감지
            topic_keywords = {
                '세미나/교육': ['seminar', 'conference', 'presentation', 'education', 'training', 'workshop', '세미나', '교육', '발표', '강의', '워크샵'],
                '비즈니스': ['business', 'market', 'sales', 'customer', 'company', 'industry', '비즈니스', '시장', '고객', '회사', '산업'],
                '기술/IT': ['technology', 'software', 'digital', 'system', 'platform', '기술', '소프트웨어', '디지털', '시스템', '플랫폼'],
                '주얼리': ['jewelry', 'diamond', 'gold', 'silver', 'gem', 'precious', '주얼리', '다이아몬드', '금', '은', '보석']
            }
            
            detected_topics = []
            for topic, keywords in topic_keywords.items():
                matches = sum(1 for keyword in keywords if keyword.lower() in combined_content.lower())
                if matches >= 2:  # 2개 이상의 관련 키워드가 있으면 해당 주제로 분류
                    detected_topics.append(f"{topic}({matches}개 키워드)")
            
            if detected_topics:
                findings.append(f"🎯 **주제 분류**: {', '.join(detected_topics)}")
        
        # 기술적 분석 결과
        if audio_results:
            audio_success = len([r for r in audio_results if r.get('status') == 'success'])
            total_audio_time = sum([r.get('processing_time', 0) for r in audio_results if r.get('processing_time')])
            findings.append(f"🎤 **음성 분석**: {len(audio_results)}개 파일 중 {audio_success}개 성공 (총 처리시간: {total_audio_time:.1f}초)")
        
        if image_results:
            image_success = len([r for r in image_results if r.get('status') == 'success'])
            avg_confidence = sum([r.get('average_confidence', 0) for r in image_results if r.get('average_confidence')]) / len(image_results) if image_results else 0
            findings.append(f"🖼️ **이미지 분석**: {len(image_results)}개 파일 중 {image_success}개 성공 (평균 신뢰도: {avg_confidence:.1f}%)")
        
        if jewelry_keywords:
            findings.append(f"💎 **주얼리 전문용어**: {', '.join(jewelry_keywords[:5])}{'...' if len(jewelry_keywords) > 5 else ''}")
        
        return findings
    
    def _generate_conclusions(self, results, success_rate, jewelry_keywords):
        """결론 및 제안사항 생성"""
        conclusions = []
        
        if success_rate >= 90:
            conclusions.append("분석 시스템이 안정적으로 작동하고 있으며, 대부분의 파일에서 성공적으로 데이터를 추출했습니다.")
        elif success_rate >= 70:
            conclusions.append("분석 시스템이 전반적으로 잘 작동하고 있으나, 일부 파일 형식이나 품질 개선이 필요할 수 있습니다.")
        else:
            conclusions.append("분석 실패율이 높으므로 파일 품질이나 형식을 점검하고, 시스템 설정을 조정할 필요가 있습니다.")
        
        if jewelry_keywords:
            conclusions.append("주얼리 관련 컨텐츠가 충분히 감지되어 도메인 특화 분석이 효과적으로 수행되었습니다.")
            conclusions.append("향후 주얼리 전문 용어 사전을 확장하여 더 정확한 키워드 추출이 가능할 것입니다.")
        
        # 개선 제안
        error_results = [r for r in results if r.get('status') == 'error']
        if error_results:
            error_types = [r.get('error', '') for r in error_results]
            if any('m4a' in error.lower() for error in error_types):
                conclusions.append("M4A 파일 처리 개선을 위해 FFmpeg 설정을 최적화하거나 WAV 형식 사전 변환을 고려하세요.")
        
        conclusions.append("정기적인 배치 분석을 통해 비즈니스 인사이트를 지속적으로 확보할 수 있습니다.")
        
        return conclusions
    
    def _calculate_average_confidence(self, results):
        """평균 신뢰도 계산"""
        confidence_scores = []
        for result in results:
            if result.get('average_confidence'):
                confidence_scores.append(result['average_confidence'])
        
        return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    def render_advanced_dashboard(self, report: Dict[str, Any]):
        """고급 분석 대시보드 렌더링"""
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
            st.warning("⚠️ 고급 차트 기능을 위해 plotly와 pandas 설치가 필요합니다.")
            st.code("pip install plotly pandas")
            return
        
        try:
            # 탭으로 대시보드 구성
            tab1, tab2, tab3, tab4 = st.tabs(["📈 파일 분석", "⏱️ 처리 시간", "🏷️ 키워드 분석", "📊 성능 지표"])
            
            with tab1:
                self._render_file_analysis_charts(report)
            
            with tab2:
                self._render_processing_time_charts(report)
            
            with tab3:
                self._render_keyword_analysis_charts(report)
            
            with tab4:
                self._render_performance_metrics(report)
                
        except Exception as e:
            st.error(f"대시보드 렌더링 오류: {str(e)}")
    
    def _render_file_analysis_charts(self, report: Dict[str, Any]):
        """파일 분석 차트"""
        col1, col2 = st.columns(2)
        
        with col1:
            # 파일 타입별 분포
            if report.get('file_type_distribution'):
                file_types = list(report['file_type_distribution'].keys())
                file_counts = list(report['file_type_distribution'].values())
                
                fig = px.pie(
                    values=file_counts, 
                    names=file_types,
                    title="📁 파일 타입 분포",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 분석 성공률
            if hasattr(report, 'success_rate') and hasattr(report, 'total_files'):
                success_count = int((report['success_rate'] / 100) * report['total_files'])
                failed_count = report['total_files'] - success_count
                
                fig = go.Figure(data=[
                    go.Bar(name='성공', x=['분석 결과'], y=[success_count], marker_color='green'),
                    go.Bar(name='실패', x=['분석 결과'], y=[failed_count], marker_color='red')
                ])
                fig.update_layout(
                    title="✅ 분석 성공률",
                    barmode='stack',
                    yaxis_title="파일 수"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 파일 크기 분포
        if st.session_state.analysis_results:
            file_sizes = []
            file_names = []
            
            for result in st.session_state.analysis_results:
                if result.get('file_size_mb'):
                    file_sizes.append(result['file_size_mb'])
                    file_names.append(result.get('file_name', 'Unknown'))
            
            if file_sizes:
                fig = px.histogram(
                    x=file_sizes,
                    nbins=10,
                    title="📏 파일 크기 분포 (MB)",
                    labels={'x': '파일 크기 (MB)', 'y': '파일 수'}
                )
                fig.update_traces(marker_color='lightblue')
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_processing_time_charts(self, report: Dict[str, Any]):
        """처리 시간 차트"""
        if not st.session_state.analysis_results:
            st.info("처리 시간 데이터가 없습니다.")
            return
        
        # 파일별 처리 시간
        processing_times = []
        file_names = []
        file_types = []
        
        for result in st.session_state.analysis_results:
            if result.get('processing_time'):
                processing_times.append(result['processing_time'])
                file_names.append(result.get('file_name', 'Unknown')[:20] + '...')
                file_types.append(result.get('file_type', 'unknown'))
        
        if processing_times:
            col1, col2 = st.columns(2)
            
            with col1:
                # 파일별 처리 시간 막대 차트
                fig = px.bar(
                    x=file_names,
                    y=processing_times,
                    title="⏱️ 파일별 처리 시간",
                    labels={'x': '파일명', 'y': '처리 시간 (초)'},
                    color=processing_times,
                    color_continuous_scale='Viridis'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 파일 타입별 평균 처리 시간
                if PANDAS_AVAILABLE:
                    df = pd.DataFrame({
                        'file_type': file_types,
                        'processing_time': processing_times
                    })
                    avg_times = df.groupby('file_type')['processing_time'].mean().reset_index()
                    
                    fig = px.bar(
                        avg_times,
                        x='file_type',
                        y='processing_time',
                        title="📊 파일 타입별 평균 처리 시간",
                        labels={'file_type': '파일 타입', 'processing_time': '평균 처리 시간 (초)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # 처리 시간 통계
        if processing_times:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⚡ 최단 시간", f"{min(processing_times):.1f}초")
            with col2:
                st.metric("🐌 최장 시간", f"{max(processing_times):.1f}초")
            with col3:
                st.metric("📊 평균 시간", f"{sum(processing_times)/len(processing_times):.1f}초")
            with col4:
                st.metric("🕐 총 처리 시간", f"{sum(processing_times):.1f}초")
    
    def _render_keyword_analysis_charts(self, report: Dict[str, Any]):
        """키워드 분석 차트"""
        if not report.get('top_keywords'):
            st.info("키워드 데이터가 없습니다.")
            return
        
        # 상위 키워드 막대 차트
        keywords = [item[0] for item in report['top_keywords'][:20]]
        counts = [item[1] for item in report['top_keywords'][:20]]
        
        fig = px.bar(
            x=counts,
            y=keywords,
            orientation='h',
            title="🏷️ 상위 키워드 빈도",
            labels={'x': '출현 빈도', 'y': '키워드'},
            color=counts,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # 키워드 트렌드 (시간별 - 가능한 경우)
        if len(report['top_keywords']) >= 10:
            # 워드클라우드 스타일 시각화 (Plotly로 구현)
            fig = go.Figure()
            
            for i, (keyword, count) in enumerate(report['top_keywords'][:20]):
                fig.add_trace(go.Scatter(
                    x=[i % 5], 
                    y=[i // 5],
                    text=keyword,
                    mode='text',
                    textfont=dict(size=min(30, 10 + count * 2)),
                    showlegend=False
                ))
            
            fig.update_layout(
                title="☁️ 키워드 클라우드",
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_performance_metrics(self, report: Dict[str, Any]):
        """성능 지표 차트"""
        col1, col2 = st.columns(2)
        
        with col1:
            # 성능 게이지 차트
            success_rate = report.get('success_rate', 0)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = success_rate,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "성공률 (%)"},
                delta = {'reference': 90},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 신뢰도 분포
            if st.session_state.analysis_results:
                confidences = []
                for result in st.session_state.analysis_results:
                    if result.get('average_confidence'):
                        confidences.append(result['average_confidence'])
                
                if confidences:
                    fig = px.histogram(
                        x=confidences,
                        nbins=15,
                        title="🎯 신뢰도 분포",
                        labels={'x': '신뢰도', 'y': '파일 수'},
                        color_discrete_sequence=['lightcoral']
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # 종합 성능 레이더 차트
        if st.session_state.analysis_results:
            categories = ['속도', '정확도', '안정성', '효율성', '사용성']
            
            # 성능 점수 계산 (0-100)
            processing_times = [r.get('processing_time', 0) for r in st.session_state.analysis_results if r.get('processing_time')]
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
            speed_score = max(0, 100 - min(100, avg_time * 10))  # 시간이 짧을수록 높은 점수
            
            accuracy_score = min(100, report.get('success_rate', 0))
            stability_score = min(100, (report.get('success_rate', 0) + 20))  # 성공률 기반
            efficiency_score = min(100, speed_score * 0.7 + accuracy_score * 0.3)
            usability_score = 85  # 고정값 (UI 복잡도 등 고려)
            
            values = [speed_score, accuracy_score, stability_score, efficiency_score, usability_score]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='성능 지표'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                title="📊 종합 성능 평가",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _prepare_enhanced_context(self) -> Dict[str, Any]:
        """강화된 분석 컨텍스트 준비 - 사전정보 최대 활용"""
        base_context = st.session_state.get('project_info', {})
        
        enhanced_context = {
            # 기본 프로젝트 정보
            'project_name': base_context.get('name', ''),
            'project_type': base_context.get('type', ''),
            'objective': base_context.get('objective', ''),
            'target_language': base_context.get('target_language', 'auto'),
            
            # 🆕 참석자 및 발표자 정보 (텍스트 보정에 활용)
            'participants': self._extract_names(base_context.get('participants', '')),
            'speakers': self._extract_names(base_context.get('speakers', '')),
            
            # 🆕 주제 및 키워드 정보 (컨텍스트 분석에 활용)
            'topic_keywords': self._extract_keywords(base_context.get('topic_keywords', '')),
            'event_context': base_context.get('event_context', ''),
            
            # 🆕 분석 설정 정보
            'analysis_depth': base_context.get('analysis_depth', '상세'),
            'enable_multi_angle': base_context.get('enable_multi_angle', True),
            'correlation_analysis': base_context.get('correlation_analysis', True),
            
            # 🆕 이미 분석된 파일들의 정보 (상관관계 분석용)
            'previous_results': self._get_previous_analysis_summary(),
            
            # 🆕 실시간 분석 가이드라인
            'analysis_guidelines': self._generate_analysis_guidelines(base_context)
        }
        
        return enhanced_context
    
    def _extract_names(self, names_text: str) -> List[str]:
        """참석자/발표자 이름 추출 및 정규화"""
        if not names_text.strip():
            return []
        
        # 쉼표, 세미콜론, 줄바꿈으로 구분
        names = []
        for separator in [',', ';', '\n']:
            names_text = names_text.replace(separator, '|')
        
        for name in names_text.split('|'):
            name = name.strip()
            if name and len(name) >= 2:  # 최소 2글자 이상
                names.append(name)
        
        return names
    
    def _extract_keywords(self, keywords_text: str) -> List[str]:
        """주제 키워드 추출 및 정규화"""
        if not keywords_text.strip():
            return []
        
        keywords = []
        for separator in [',', ';', '\n', ' ']:
            keywords_text = keywords_text.replace(separator, '|')
        
        for keyword in keywords_text.split('|'):
            keyword = keyword.strip()
            if keyword and len(keyword) >= 2:  # 최소 2글자 이상
                keywords.append(keyword)
        
        return keywords
    
    def _get_previous_analysis_summary(self) -> Dict[str, Any]:
        """이전 분석 결과 요약 (상관관계 분석용)"""
        previous_results = getattr(st.session_state, 'analysis_results', [])
        
        if not previous_results:
            return {}
        
        # 이전 결과에서 중요 정보 추출
        summary = {
            'total_files_analyzed': len(previous_results),
            'common_keywords': [],
            'frequent_participants': [],
            'main_topics': []
        }
        
        # 공통 키워드 및 참석자 추출
        all_keywords = []
        all_texts = []
        
        for result in previous_results:
            if result.get('status') == 'success':
                if result.get('jewelry_keywords'):
                    all_keywords.extend(result['jewelry_keywords'])
                if result.get('full_text'):
                    all_texts.append(result['full_text'])
        
        # 빈도 기반 중요 정보 추출
        if all_keywords:
            from collections import Counter
            keyword_counts = Counter(all_keywords)
            summary['common_keywords'] = [k for k, v in keyword_counts.most_common(5)]
        
        return summary
    
    def _generate_analysis_guidelines(self, context: Dict[str, Any]) -> List[str]:
        """컨텍스트 기반 분석 가이드라인 생성"""
        guidelines = []
        
        # 프로젝트 타입별 가이드라인
        project_type = context.get('type', '')
        if '회의' in project_type:
            guidelines.append("회의록 형식의 구조화된 요약 생성")
            guidelines.append("참석자별 발언 내용 구분")
        elif '강의' in project_type or '세미나' in project_type:
            guidelines.append("교육 자료로 활용 가능한 체계적 정리")
            guidelines.append("핵심 개념과 실용적 응용 방안 도출")
        
        # 참석자 정보가 있을 경우
        if context.get('participants'):
            guidelines.append(f"참석자 이름 정확성 검증: {', '.join(context['participants'][:3])}")
        
        # 키워드 정보가 있을 경우
        if context.get('topic_keywords'):
            guidelines.append(f"핵심 키워드 중심 분석: {', '.join(context['topic_keywords'][:3])}")
        
        return guidelines
    
    def _update_analysis_status(self, filename: str, context: Dict[str, Any]):
        """분석 진행 상황을 컨텍스트 정보와 함께 업데이트"""
        
        # 컨텍스트 정보 표시
        context_info = []
        if context.get('participants'):
            context_info.append(f"👥 참석자 {len(context['participants'])}명")
        if context.get('speakers'):
            context_info.append(f"🎤 발표자 {len(context['speakers'])}명")
        if context.get('topic_keywords'):
            context_info.append(f"🔑 키워드 {len(context['topic_keywords'])}개")
        
        if context_info:
            context_display = " | ".join(context_info)
            st.info(f"📋 **활용 중인 사전정보**: {context_display}")
        
        # 분석 가이드라인 표시
        if context.get('analysis_guidelines'):
            with st.expander("🎯 적용 중인 분석 가이드라인", expanded=False):
                for guideline in context['analysis_guidelines']:
                    st.write(f"• {guideline}")
    
    def _categorize_and_preprocess_files(self, uploaded_files_data) -> Dict[str, List]:
        """파일 분류 및 전처리"""
        categories = {
            'audio_files': [],
            'video_files': [],  
            'image_files': [],
            'document_files': [],
            'video_urls': uploaded_files_data.get('video_urls', [])
        }
        
        for uploaded_file in uploaded_files_data.get('files', []):
            file_ext = uploaded_file.name.split('.')[-1].lower()
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            
            file_info = {
                'file': uploaded_file,
                'name': uploaded_file.name,
                'extension': file_ext,
                'size_mb': file_size_mb,
                'temp_path': None
            }
            
            if file_ext in ['wav', 'mp3', 'flac', 'm4a']:
                categories['audio_files'].append(file_info)
            elif file_ext in ['mp4', 'mov', 'avi']:
                categories['video_files'].append(file_info)
            elif file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
                categories['image_files'].append(file_info)
            elif file_ext in ['pdf', 'docx', 'txt']:
                categories['document_files'].append(file_info)
        
        return categories
    
    def _build_integrated_context(self, file_categories) -> Dict[str, Any]:
        """모든 파일 정보를 통합한 컨텍스트 구성"""
        base_context = self._prepare_enhanced_context()
        
        integrated_context = {
            **base_context,
            'file_categories': file_categories,
            'total_files': sum(len(files) for files in file_categories.values() if isinstance(files, list)),
            'file_distribution': {
                'audio': len(file_categories.get('audio_files', [])),
                'video': len(file_categories.get('video_files', [])),
                'image': len(file_categories.get('image_files', [])),
                'document': len(file_categories.get('document_files', [])),
                'video_url': len(file_categories.get('video_urls', []))
            },
            'analysis_strategy': self._determine_analysis_strategy(file_categories),
            'cross_reference_enabled': True,
            'batch_processing': True
        }
        
        return integrated_context
    
    def _determine_analysis_strategy(self, file_categories) -> str:
        """파일 구성에 따른 최적 분석 전략 결정"""
        audio_count = len(file_categories.get('audio_files', []))
        video_count = len(file_categories.get('video_files', []))
        image_count = len(file_categories.get('image_files', []))
        
        if video_count > 0 and (audio_count > 0 or image_count > 0):
            return "multimodal_integrated"  # 다중모달 통합 분석
        elif audio_count > 0 and image_count > 0:
            return "audio_visual_correlation"  # 음성-시각 상관관계
        elif video_count > 1:
            return "multi_video_synthesis"  # 다중 영상 종합
        elif image_count > 3:
            return "sequential_image_analysis"  # 연속 이미지 분석
        else:
            return "standard_batch"  # 표준 배치 분석
    
    def _execute_batch_analysis(self, file_categories, integrated_context) -> Dict[str, Any]:
        """배치 통합 분석 실행"""
        batch_results = {
            'audio_results': [],
            'video_results': [],
            'image_results': [],
            'document_results': [],
            'youtube_results': [],
            'cross_correlations': [],
            'integrated_insights': {}
        }
        
        # 🎤 음성 파일 배치 분석
        if file_categories.get('audio_files'):
            batch_results['audio_results'] = self._batch_analyze_audio_files(
                file_categories['audio_files'], integrated_context
            )
        
        # 🎬 영상 파일 배치 분석  
        if file_categories.get('video_files'):
            batch_results['video_results'] = self._batch_analyze_video_files(
                file_categories['video_files'], integrated_context
            )
        
        # 🖼️ 이미지 파일 배치 분석
        if file_categories.get('image_files'):
            batch_results['image_results'] = self._batch_analyze_image_files(
                file_categories['image_files'], integrated_context
            )
        
        # 🔗 상관관계 분석
        if integrated_context.get('cross_reference_enabled'):
            batch_results['cross_correlations'] = self._analyze_cross_correlations(
                batch_results, integrated_context
            )
        
        return batch_results
    
    def _batch_analyze_audio_files(self, audio_files, context) -> List[Dict]:
        """음성 파일 배치 분석"""
        results = []
        
        # 모든 음성 파일을 임시 저장
        temp_files = []
        for file_info in audio_files:
            with tempfile.NamedTemporaryFile(suffix=f".{file_info['extension']}", delete=False) as tmp_file:
                tmp_file.write(file_info['file'].getvalue())
                temp_files.append(tmp_file.name)
                file_info['temp_path'] = tmp_file.name
        
        # 배치 STT 처리 (GPU 효율성 극대화)
        try:
            for i, file_info in enumerate(audio_files):
                st.text(f"🎤 음성 분석: {file_info['name']} ({i+1}/{len(audio_files)})")
                
                # 개별 분석 수행 (컨텍스트 포함)
                result = analyze_file_real(file_info['temp_path'], 'audio', 'auto', context)
                result['batch_index'] = i
                result['cross_reference_ready'] = True
                results.append(result)
                
        finally:
            # 임시 파일 정리
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        return results
    
    def _batch_analyze_image_files(self, image_files, context) -> List[Dict]:
        """이미지 파일 배치 분석 - GPU 최적화"""
        results = []
        
        # 이미지 파일 임시 저장
        temp_files = []
        for file_info in image_files:
            with tempfile.NamedTemporaryFile(suffix=f".{file_info['extension']}", delete=False) as tmp_file:
                tmp_file.write(file_info['file'].getvalue())
                temp_files.append(tmp_file.name)
                file_info['temp_path'] = tmp_file.name
        
        try:
            # GPU 모델 한 번만 로드하여 배치 처리
            for i, file_info in enumerate(image_files):
                st.text(f"🖼️ 이미지 분석: {file_info['name']} ({i+1}/{len(image_files)})")
                
                result = analyze_file_real(file_info['temp_path'], 'image', 'auto', context)
                result['batch_index'] = i
                result['cross_reference_ready'] = True
                results.append(result)
                
        finally:
            # 임시 파일 정리
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        return results
    
    def _batch_analyze_video_files(self, video_files, context) -> List[Dict]:
        """영상 파일 배치 분석 - 다각도 통합"""
        results = []
        
        for i, file_info in enumerate(video_files):
            st.text(f"🎬 영상 분석: {file_info['name']} ({i+1}/{len(video_files)})")
            
            # 임시 파일 저장
            with tempfile.NamedTemporaryFile(suffix=f".{file_info['extension']}", delete=False) as tmp_file:
                tmp_file.write(file_info['file'].getvalue())
                temp_path = tmp_file.name
            
            try:
                result = analyze_file_real(temp_path, 'video', 'auto', context)
                result['batch_index'] = i
                result['cross_reference_ready'] = True
                results.append(result)
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        return results
    
    def _analyze_cross_correlations(self, batch_results, context) -> List[Dict]:
        """배치 결과 간 상관관계 분석"""
        correlations = []
        
        # 모든 텍스트 수집
        all_texts = []
        all_keywords = []
        
        for category, results in batch_results.items():
            if isinstance(results, list):
                for result in results:
                    if result.get('status') == 'success':
                        if result.get('full_text'):
                            all_texts.append({
                                'text': result['full_text'],
                                'source': category,
                                'file_name': result.get('file_name', ''),
                                'type': category.replace('_results', '')
                            })
                        if result.get('jewelry_keywords'):
                            all_keywords.extend(result['jewelry_keywords'])
        
        # 공통 키워드 분석
        if all_keywords:
            from collections import Counter
            keyword_counts = Counter(all_keywords)
            common_keywords = [k for k, v in keyword_counts.most_common(10) if v > 1]
            
            correlations.append({
                'type': 'common_keywords',
                'keywords': common_keywords,
                'strength': len(common_keywords) / len(set(all_keywords)) if all_keywords else 0
            })
        
        return correlations
    
    def _integrate_and_optimize_results(self, batch_results, context) -> List[Dict]:
        """배치 결과 통합 및 최적화"""
        integrated_results = []
        
        # 모든 결과를 통합 리스트로 변환
        for category, results in batch_results.items():
            if isinstance(results, list):
                for result in results:
                    # 배치 분석 메타데이터 추가
                    result['batch_processed'] = True
                    result['correlation_analyzed'] = len(batch_results.get('cross_correlations', [])) > 0
                    result['analysis_strategy'] = context.get('analysis_strategy', 'standard_batch')
                    integrated_results.append(result)
        
        return integrated_results

def main():
    """메인 실행 함수"""
    ui = SolomondRealAnalysisUI()
    ui.run()

if __name__ == "__main__":
    main()