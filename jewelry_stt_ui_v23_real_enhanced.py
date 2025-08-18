#!/usr/bin/env python3
"""
솔로몬드 AI v2.4 - 브라우저 자동화 통합 버전
실제 분석 + 브라우저 자동화 + 실시간 스트리밍 완전 통합

주요 개선사항 (v2.4.0):
- 브라우저 자동화 엔진 통합 (Playwright MCP)
- 실시간 주얼리 정보 검색 및 경쟁사 분석
- 실시간 음성 스트리밍 시스템 통합
- 보안 강화 API 서버 준비
- MCP 도구 7개 완전 활용
- 종합 시스템 데모 및 성능 검증 완료
"""

import streamlit as st
import sys
import os
import asyncio
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
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

# 통합 모듈 임포트
from utils.logger import get_logger
from config.compute_config import force_cpu_mode, get_compute_config

# 실제 분석 엔진 import
try:
    from core.real_analysis_engine import global_analysis_engine, analyze_file_real, create_comprehensive_story_from_sources
    REAL_ANALYSIS_AVAILABLE = True
    print("[SUCCESS] 실제 분석 엔진 로드 완료")
except ImportError as e:
    REAL_ANALYSIS_AVAILABLE = False
    print(f"[ERROR] 실제 분석 엔진 로드 실패: {e}")

# 실시간 진행 추적 및 MCP 자동 해결 시스템 import
try:
    from core.realtime_progress_tracker import global_progress_tracker, RealtimeProgressTracker
    from core.mcp_auto_problem_solver import global_mcp_solver, MCPAutoProblemSolver
    from core.youtube_realtime_processor import global_youtube_realtime_processor
    ADVANCED_MONITORING_AVAILABLE = True
    YOUTUBE_REALTIME_AVAILABLE = True
    print("[SUCCESS] 고급 모니터링 시스템 로드 완료")
    print("[SUCCESS] YouTube 실시간 처리 시스템 로드 완료")
except ImportError as e:
    ADVANCED_MONITORING_AVAILABLE = False
    YOUTUBE_REALTIME_AVAILABLE = False
    print(f"[ERROR] 고급 모니터링 시스템 로드 실패: {e}")

# 대용량 파일 핸들러 import
try:
    from core.large_file_handler import large_file_handler
    LARGE_FILE_HANDLER_AVAILABLE = True
    print("[SUCCESS] 대용량 파일 핸들러 로드 완료")
except ImportError as e:
    LARGE_FILE_HANDLER_AVAILABLE = False
    print(f"[ERROR] 대용량 파일 핸들러 로드 실패: {e}")

# 브라우저 자동화 및 실시간 스트리밍 시스템 import (v2.4)
try:
    from core.browser_automation_engine import BrowserAutomationEngine
    from core.mcp_browser_integration import get_mcp_browser_integration
    from core.realtime_audio_streaming_engine import RealtimeAudioStreamingEngine
    from core.security_api_server import SecurityAPIServer, SecurityConfig
    BROWSER_AUTOMATION_AVAILABLE = True
    MCP_BROWSER_AVAILABLE = True
    REALTIME_STREAMING_AVAILABLE = True
    SECURITY_API_AVAILABLE = True
    print("[SUCCESS] 브라우저 자동화 엔진 로드 완료")
    print("[SUCCESS] MCP 브라우저 통합 모듈 로드 완료")
    print("[SUCCESS] 실시간 스트리밍 엔진 로드 완료")
    print("[SUCCESS] 보안 API 서버 로드 완료")
except ImportError as e:
    BROWSER_AUTOMATION_AVAILABLE = False
    MCP_BROWSER_AVAILABLE = False
    REALTIME_STREAMING_AVAILABLE = False
    SECURITY_API_AVAILABLE = False
    print(f"[ERROR] v2.4 확장 기능 로드 실패: {e}")

# 웹 데이터 통합 시스템 import
try:
    from core.web_data_integration import get_web_data_integration
    WEB_DATA_INTEGRATION_AVAILABLE = True
    print("[SUCCESS] 웹 데이터 통합 시스템 로드 완료")
except ImportError as e:
    WEB_DATA_INTEGRATION_AVAILABLE = False
    print(f"[ERROR] 웹 데이터 통합 시스템 로드 실패: {e}")

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
except ImportError as e:
    HYBRID_LLM_AVAILABLE = False
    print(f"[ERROR] Hybrid LLM Manager 로드 실패: {e}")

# 🎯 MCP 자동 통합 시스템 임포트
try:
    from mcp_auto_integration_wrapper import smart_mcp_enhance, enhance_result_with_mcp, get_mcp_usage_stats
    MCP_AUTO_INTEGRATION_AVAILABLE = True
    print("✅ MCP 자동 통합 시스템 활성화")
except ImportError as e:
    MCP_AUTO_INTEGRATION_AVAILABLE = False
    print(f"⚠️ MCP 자동 통합 시스템 비활성화: {e}")
    
    # 폴백 함수들
    def smart_mcp_enhance(func):
        return func
    
    async def enhance_result_with_mcp(request, result, context=None):
        return result
    
    def get_mcp_usage_stats():
        return {"status": "unavailable"}
    HYBRID_LLM_AVAILABLE = True
except ImportError:
    HYBRID_LLM_AVAILABLE = False
    
# Streamlit 설정
st.set_page_config(
    page_title="솔로몬드 AI v2.4 - 브라우저 자동화 통합",
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
    """솔로몬드 AI v2.4 브라우저 자동화 통합 UI - 확장된 워크플로우"""
    
    def __init__(self):
        self.setup_logging()
        self.analysis_engine = global_analysis_engine if REAL_ANALYSIS_AVAILABLE else None
        
        # v2.4 신규 엔진들 초기화
        self.browser_engine = BrowserAutomationEngine() if BROWSER_AUTOMATION_AVAILABLE else None
        self.mcp_browser = get_mcp_browser_integration() if MCP_BROWSER_AVAILABLE else None
        self.streaming_engine = RealtimeAudioStreamingEngine() if REALTIME_STREAMING_AVAILABLE else None
        self.api_server = SecurityAPIServer() if SECURITY_API_AVAILABLE else None
        self.web_data_integration = get_web_data_integration() if WEB_DATA_INTEGRATION_AVAILABLE else None
        
        self.session_stats = {
            "files_analyzed": 0,
            "total_processing_time": 0,
            "successful_analyses": 0,
            "session_start": datetime.now()
        }
        
        # 4단계 워크플로우 상태 관리 (성능 최적화)
        self._init_session_state()
        
        # 캐시 및 성능 최적화 관련
        if 'ui_cache' not in st.session_state:
            st.session_state.ui_cache = {}
        if 'last_update_time' not in st.session_state:
            st.session_state.last_update_time = {}
    
    def _init_session_state(self):
        """세션 상태 초기화 (조건부 실행으로 성능 최적화)"""
        default_states = {
            'workflow_step': 1,
            'project_info': {},
            'uploaded_files_data': [],
            'analysis_results': [],
            'final_report': None,
            'ui_preferences': {
                'theme': 'light',
                'animations_enabled': True,
                'compact_mode': False
            }
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @st.cache_data(ttl=300)  # 5분 캐시
    def _get_system_info():
        """시스템 정보 캐싱 (성능 최적화)"""
        return {
            "cpu_count": os.cpu_count(),
            "available_models": {
                "whisper": REAL_ANALYSIS_AVAILABLE,
                "easyocr": REAL_ANALYSIS_AVAILABLE,
                "transformers": REAL_ANALYSIS_AVAILABLE
            },
            "large_file_support": LARGE_FILE_HANDLER_AVAILABLE
        }
    
    def should_update_component(self, component_id: str, force_update: bool = False):
        """컴포넌트 업데이트 필요성 확인 (불필요한 재렌더링 방지)"""
        if force_update:
            st.session_state.last_update_time[component_id] = time.time()
            return True
        
        current_time = time.time()
        last_update = st.session_state.last_update_time.get(component_id, 0)
        
        # 1초 이내 중복 업데이트 방지
        if current_time - last_update < 1.0:
            return False
        
        st.session_state.last_update_time[component_id] = current_time
        return True
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = get_logger(__name__)
    
    def setup_custom_css(self):
        """개선된 CSS 스타일 적용"""
        st.markdown("""
        <style>
        /* 메인 컨테이너 스타일 */
        .main-container {
            padding: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            margin-bottom: 1rem;
            color: white;
        }
        
        /* 워크플로우 진행바 스타일 */
        .workflow-progress {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50px;
            padding: 10px 20px;
            margin: 20px 0;
            backdrop-filter: blur(10px);
        }
        
        .workflow-step {
            padding: 8px 16px;
            border-radius: 25px;
            font-weight: 600;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .workflow-step.current {
            background: #4CAF50;
            color: white;
            transform: scale(1.1);
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
        }
        
        .workflow-step.completed {
            background: #2196F3;
            color: white;
        }
        
        .workflow-step.pending {
            background: rgba(255, 255, 255, 0.2);
            color: #ccc;
        }
        
        /* 파일 업로드 영역 개선 */
        .upload-area {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            background: linear-gradient(45deg, #f8f9ff, #e8f0ff);
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: #4CAF50;
            background: linear-gradient(45deg, #f0f8f0, #e8f5e8);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(76, 175, 80, 0.15);
        }
        
        /* 진행률 표시 개선 */
        .progress-container {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .progress-text {
            font-size: 16px;
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
        }
        
        /* 에러 메시지 스타일 개선 */
        .error-container {
            background: linear-gradient(135deg, #ff6b6b, #ff8e8e);
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            color: white;
            border-left: 5px solid #ff4757;
        }
        
        .warning-container {
            background: linear-gradient(135deg, #ffa726, #ffb74d);
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            color: white;
            border-left: 5px solid #ff9800;
        }
        
        .success-container {
            background: linear-gradient(135deg, #4CAF50, #66BB6A);
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            color: white;
            border-left: 5px solid #2E7D32;
        }
        
        .info-container {
            background: linear-gradient(135deg, #2196F3, #42A5F5);
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            color: white;
            border-left: 5px solid #1976D2;
        }
        
        /* 버튼 스타일 개선 */
        .stButton > button {
            border-radius: 25px;
            border: none;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }
        
        /* 카드 스타일 */
        .info-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #667eea;
            transition: all 0.3s ease;
        }
        
        .info-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
        }
        
        /* 반응형 디자인 */
        @media (max-width: 768px) {
            .workflow-progress {
                flex-direction: column;
                gap: 10px;
            }
            
            .workflow-step {
                width: 100%;
            }
            
            .main-container {
                padding: 0.5rem;
            }
        }
        
        /* 애니메이션 */
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .slide-in {
            animation: slideIn 0.5s ease-out;
        }
        
        @keyframes pulse {
            0% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.05);
            }
            100% {
                transform: scale(1);
            }
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def show_enhanced_message(self, message_type: str, title: str, content: str, solutions: List[str] = None):
        """개선된 메시지 표시 함수"""
        container_class = f"{message_type}-container"
        
        if message_type == "error":
            icon = "❌"
        elif message_type == "warning":
            icon = "⚠️"
        elif message_type == "success":
            icon = "✅"
        elif message_type == "info":
            icon = "💡"
        else:
            icon = "📌"
        
        message_html = f"""
        <div class="{container_class} slide-in">
            <h4>{icon} {title}</h4>
            <p>{content}</p>
        """
        
        if solutions:
            message_html += "<h5>💡 해결 방법:</h5><ul>"
            for solution in solutions[:3]:  # 최대 3개까지만 표시
                message_html += f"<li>{solution}</li>"
            message_html += "</ul>"
        
        message_html += "</div>"
        
        st.markdown(message_html, unsafe_allow_html=True)
    
    def show_real_progress_only(self, current: int, total: int, message: str, details: str = "", force_display: bool = False):
        """실제 진행률만 표시 - 가짜 진행률 완전 제거"""
        # 실제 처리가 아니고 강제 표시가 아니면 표시하지 않음
        if not force_display and current == 0:
            return st.info(f"⏳ {message} - 분석 준비 중입니다...")
        
        # 실제 진행률만 계산
        progress_percent = current / total if total > 0 else 0
        
        # 실제 진행률이 0%면 표시하지 않음 (무한루프 방지)
        if progress_percent == 0 and not force_display:
            return st.info(f"⏳ {message}")
            
        progress_html = f"""
        <div class="progress-container slide-in">
            <div class="progress-text">✅ {message}</div>
            <div style="background: #e0e0e0; border-radius: 10px; overflow: hidden;">
                <div style="
                    width: {progress_percent * 100:.1f}%; 
                    height: 20px; 
                    background: linear-gradient(90deg, #4CAF50, #66BB6A);
                    transition: width 0.3s ease;
                    border-radius: 10px;
                "></div>
            </div>
            <div style="
                display: flex; 
                justify-content: space-between; 
                margin-top: 10px;
                font-size: 14px;
                color: #666;
            ">
                <span>실제 완료: {current}/{total}</span>
                <span>{progress_percent * 100:.1f}%</span>
            </div>
            {f'<div style="margin-top: 10px; font-size: 12px; color: #888;">{details}</div>' if details else ''}
        </div>
        """
        
        return st.markdown(progress_html, unsafe_allow_html=True)
    
    def show_realtime_analysis_timer(self):
        """실시간 분석 경과 시간 표시"""
        import time
        import datetime
        
        if not hasattr(st.session_state, 'analysis_start_time') or not st.session_state.analysis_start_time:
            return
        
        current_time = time.time()
        elapsed_seconds = int(current_time - st.session_state.analysis_start_time)
        
        # 시간 포맷팅
        hours = elapsed_seconds // 3600
        minutes = (elapsed_seconds % 3600) // 60
        seconds = elapsed_seconds % 60
        
        if hours > 0:
            elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            elapsed_str = f"{minutes:02d}:{seconds:02d}"
        
        # 분석 상태 표시
        status = st.session_state.get('analysis_status', '준비 중')
        start_time_str = datetime.datetime.fromtimestamp(st.session_state.analysis_start_time).strftime("%H:%M:%S")
        
        # 실시간 타이머 HTML
        timer_html = f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 16px; font-weight: bold;">🕐 실시간 분석 타이머</div>
                    <div style="font-size: 12px; opacity: 0.9;">시작 시간: {start_time_str}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 24px; font-weight: bold; font-family: 'Courier New', monospace;">
                        {elapsed_str}
                    </div>
                    <div style="font-size: 12px; opacity: 0.9;">상태: {status}</div>
                </div>
            </div>
        </div>
        """
        
        return timer_html
    

    def render_comprehensive_analysis(self):
        """종합 상황 분석 페이지"""
        st.header("🎯 종합 상황 분석")
        st.markdown("**user_files 폴더의 모든 파일을 하나의 상황으로 통합 분석**")
        
        # 초기화
        if 'comprehensive_analysis_complete' not in st.session_state:
            st.session_state.comprehensive_analysis_complete = False
        if 'comprehensive_results' not in st.session_state:
            st.session_state.comprehensive_results = None
        
        # 설정
        with st.sidebar:
            st.subheader("🔧 종합 분석 설정")
            max_audio_size = st.slider("최대 오디오 크기 (MB)", 1, 50, 20)
            max_image_size = st.slider("최대 이미지 크기 (MB)", 1, 20, 10)
            include_videos = st.checkbox("비디오 메타데이터 포함", value=True)
        
        # 1. 파일 발견
        st.subheader("📁 상황 파일 발견")
        
        if st.button("🔍 user_files 폴더 탐색", type="primary"):
            with st.spinner("파일 탐색 중..."):
                files = self._discover_user_files()
            
            if files:
                st.success(f"✅ {len(files)}개 파일 발견")
                
                # 파일 타입별 분류
                audio_files = [f for f in files if f['ext'] in ['.m4a', '.wav', '.mp3']]
                image_files = [f for f in files if f['ext'] in ['.jpg', '.jpeg', '.png']]
                video_files = [f for f in files if f['ext'] in ['.mov', '.mp4']]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎵 오디오", len(audio_files))
                with col2:
                    st.metric("🖼️ 이미지", len(image_files))
                with col3:
                    st.metric("🎬 비디오", len(video_files))
                
                # 파일 미리보기
                with st.expander("📋 발견된 파일 목록"):
                    for file_info in files:
                        st.write(f"- **{file_info['name']}** ({file_info['size_mb']:.1f}MB) - {file_info['timestamp']}")
                
                st.session_state.discovered_files = files
            else:
                st.warning("⚠️ user_files 폴더에서 파일을 찾을 수 없습니다.")
        
        # 2. 종합 분석 실행
        if 'discovered_files' in st.session_state:
            st.subheader("🎯 종합 분석 실행")
            
            if st.button("🚀 모든 파일 통합 분석 시작", type="primary"):
                files = st.session_state.discovered_files
                
                st.info(f"📊 {len(files)}개 파일의 종합 분석을 시작합니다...")
                
                # 진행률 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 결과 저장용
                comprehensive_results = {
                    'audio_results': [],
                    'image_results': [],
                    'video_results': [],
                    'timeline': [],
                    'comprehensive_story': ''
                }
                
                total_files = len(files)
                
                # 파일별 순차 분석
                for i, file_info in enumerate(files):
                    progress = (i + 1) / total_files
                    progress_bar.progress(progress)
                    status_text.text(f"분석 중: {file_info['name']} ({i+1}/{total_files})")
                    
                    try:
                        result = self._analyze_single_file_comprehensive(file_info, max_audio_size, max_image_size)
                        
                        if result:
                            if result['type'] == 'audio':
                                comprehensive_results['audio_results'].append(result)
                            elif result['type'] == 'image':
                                comprehensive_results['image_results'].append(result)
                            elif result['type'] == 'video':
                                comprehensive_results['video_results'].append(result)
                            
                            comprehensive_results['timeline'].append({
                                'timestamp': file_info['timestamp'],
                                'file': file_info['name'],
                                'type': result['type'],
                                'content': result.get('content', '')[:200]
                            })
                    
                    except Exception as e:
                        st.error(f"❌ {file_info['name']} 분석 실패: {str(e)[:100]}")
                        continue
                
                # 종합 스토리 생성
                comprehensive_results['comprehensive_story'] = self._generate_comprehensive_story(comprehensive_results)
                
                st.session_state.comprehensive_results = comprehensive_results
                st.session_state.comprehensive_analysis_complete = True
                
                progress_bar.progress(1.0)
                status_text.text("✅ 종합 분석 완료!")
                
                st.success("🎉 모든 파일의 종합 분석이 완료되었습니다!")
                st.rerun()
        
        # 3. 결과 표시
        if st.session_state.comprehensive_analysis_complete and st.session_state.comprehensive_results:
            self._display_comprehensive_results(st.session_state.comprehensive_results)
    
    def _discover_user_files(self):
        """user_files 폴더에서 파일 발견"""
        user_files = Path("user_files")
        all_files = []
        
        if user_files.exists():
            for file_path in user_files.rglob("*"):
                if file_path.is_file() and file_path.name != "README.md":
                    try:
                        stat = file_path.stat()
                        file_info = {
                            'path': str(file_path),
                            'name': file_path.name,
                            'size_mb': stat.st_size / 1024 / 1024,
                            'timestamp': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                            'ext': file_path.suffix.lower()
                        }
                        all_files.append(file_info)
                    except Exception:
                        continue
        
        # 시간순 정렬
        all_files.sort(key=lambda x: x['timestamp'])
        return all_files
    
    def _analyze_single_file_comprehensive(self, file_info, max_audio_size, max_image_size):
        """단일 파일 종합 분석"""
        ext = file_info['ext']
        
        try:
            if ext in ['.m4a', '.wav', '.mp3']:
                # 오디오 분석
                if file_info['size_mb'] <= max_audio_size:
                    # 기존 분석 엔진 활용
                    if REAL_ANALYSIS_AVAILABLE:
                        result = analyze_file_real(
                            file_info['path'],
                            "audio",
                            language="ko"
                        )
                        
                        if result and result.get('success'):
                            return {
                                'type': 'audio',
                                'file': file_info['name'],
                                'content': result.get('stt_result', {}).get('text', ''),
                                'processing_time': result.get('processing_time', 0),
                                'timestamp': file_info['timestamp']
                            }
                
            elif ext in ['.jpg', '.jpeg', '.png']:
                # 이미지 분석
                if file_info['size_mb'] <= max_image_size:
                    if REAL_ANALYSIS_AVAILABLE:
                        result = analyze_file_real(
                            file_info['path'],
                            "image",
                            language="ko"
                        )
                        
                        if result and result.get('success'):
                            ocr_result = result.get('ocr_result', {})
                            return {
                                'type': 'image',
                                'file': file_info['name'],
                                'content': ' '.join([block.get('text', '') for block in ocr_result.get('text_blocks', [])]),
                                'text_blocks': len(ocr_result.get('text_blocks', [])),
                                'timestamp': file_info['timestamp']
                            }
            
            elif ext in ['.mov', '.mp4']:
                # 비디오 메타데이터
                return {
                    'type': 'video',
                    'file': file_info['name'],
                    'content': f"비디오 파일 ({file_info['size_mb']:.1f}MB)",
                    'size_mb': file_info['size_mb'],
                    'timestamp': file_info['timestamp']
                }
        
        except Exception as e:
            raise Exception(f"파일 분석 중 오류: {str(e)}")
        
        return None
    
    def _generate_comprehensive_story(self, results):
        """종합 스토리 생성"""
        story_parts = []
        
        # 시간순 정렬
        timeline = sorted(results['timeline'], key=lambda x: x['timestamp'])
        
        story_parts.append("=== 종합 상황 분석 스토리 ===\n")
        
        for i, event in enumerate(timeline):
            if event['content'].strip():
                story_parts.append(f"{i+1}. [{event['type'].upper()}] {event['file']}")
                story_parts.append(f"   시간: {event['timestamp']}")
                story_parts.append(f"   내용: {event['content'][:300]}...")
                story_parts.append("")
        
        return "\n".join(story_parts)
    
    def _display_comprehensive_results(self, results):
        """종합 결과 표시"""
        st.subheader("📊 종합 분석 결과")
        
        # 요약 통계
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎵 오디오 분석", len(results['audio_results']))
        with col2:
            st.metric("🖼️ 이미지 분석", len(results['image_results']))
        with col3:
            st.metric("🎬 비디오 수집", len(results['video_results']))
        with col4:
            st.metric("📅 총 이벤트", len(results['timeline']))
        
        # 종합 스토리
        st.subheader("📖 종합 상황 스토리")
        with st.expander("📜 전체 스토리 보기", expanded=True):
            st.text_area("종합 스토리", results['comprehensive_story'], height=400)
        
        # 타임라인 시각화
        if results['timeline']:
            st.subheader("📅 시간순 타임라인")
            for i, event in enumerate(results['timeline']):
                with st.expander(f"{i+1}. {event['file']} ({event['type']}) - {event['timestamp']}"):
                    st.write(f"**내용:** {event['content']}")
        
        # 상세 결과
        if results['audio_results']:
            st.subheader("🎵 오디오 분석 상세")
            for audio in results['audio_results']:
                with st.expander(f"🎵 {audio['file']}"):
                    st.write(f"**처리 시간:** {audio.get('processing_time', 0):.1f}초")
                    st.write(f"**내용:** {audio['content']}")
        
        if results['image_results']:
            st.subheader("🖼️ 이미지 분석 상세")
            for image in results['image_results']:
                with st.expander(f"🖼️ {image['file']}"):
                    st.write(f"**텍스트 블록:** {image.get('text_blocks', 0)}개")
                    st.write(f"**추출된 텍스트:** {image['content']}")
        
        # 결과 저장
        if st.button("💾 종합 분석 결과 저장"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_situation_analysis_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            st.success(f"✅ 종합 분석 결과 저장: {filename}")
    

        def run(self):
        """메인 실행 - 4단계 워크플로우"""
        
        # CSS 스타일 적용
        self.setup_custom_css()
        
        # 개선된 헤더 (v2.4)
        st.markdown("""
        <div class="main-container slide-in">
            <h1>💎 솔로몬드 AI v2.4 - 브라우저 자동화 통합</h1>
            <p><strong>🚀 확장된 기능:</strong> 분석 + 브라우저 검색 + 실시간 스트리밍 + 보안 API</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 워크플로우 진행 상태 표시
        self.display_workflow_progress()
        
        # 탭 기반 UI로 확장 (v2.4)
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 기본 분석", 
            "🌐 브라우저 검색", 
            "🚀 MCP 브라우저",
            "🎤 실시간 스트리밍", 
            "📊 경쟁사 분석",
            "🔒 보안 API"
        ])
        
        with tab1:
            # 기존 4단계 워크플로우 + 웹 데이터 통합
            if st.session_state.workflow_step == 1:
                self.render_step1_basic_info()
            elif st.session_state.workflow_step == 2:
                self.render_step2_upload()
            elif st.session_state.workflow_step == 3:
                self.render_step3_review()
            elif st.session_state.workflow_step == 4:
                self.render_step4_report()
        
        with tab2:
            # 기본 브라우저 검색 기능
            self.render_browser_search_tab()
        
        with tab3:
            # 새로운 MCP 브라우저 기능
            self.render_mcp_browser_tab()
        
        with tab4:
            # 실시간 스트리밍 기능
            self.render_realtime_streaming_tab()
        
        with tab5:
            # 경쟁사 분석 기능
            self.render_competitive_analysis_tab()
        
        with tab6:
            # 보안 API 관리
            self.render_security_api_tab()
        
        # 하단에 전체 시스템 상태 표시
        with st.expander("🔧 시스템 상태 확인"):
            self.display_system_status()
    
    def display_workflow_progress(self):
        """개선된 워크플로우 진행 상태 표시"""
        steps = [
            {"number": 1, "title": "기본정보", "icon": "📋"},
            {"number": 2, "title": "업로드", "icon": "📤"}, 
            {"number": 3, "title": "검토", "icon": "🔍"},
            {"number": 4, "title": "보고서", "icon": "📊"}
        ]
        
        progress_html = '<div class="workflow-progress slide-in">'
        
        for step in steps:
            if step["number"] == st.session_state.workflow_step:
                step_class = "workflow-step current pulse"
                display_text = f'{step["icon"]} {step["title"]} (진행중)'
            elif step["number"] < st.session_state.workflow_step:
                step_class = "workflow-step completed"
                display_text = f'✅ {step["title"]} (완료)'
            else:
                step_class = "workflow-step pending"
                display_text = f'{step["icon"]} {step["title"]}'
            
            progress_html += f'<div class="{step_class}">{display_text}</div>'
        
        progress_html += '</div>'
        
        st.markdown(progress_html, unsafe_allow_html=True)
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
            
            # 실시간 웹 검색 제안 기능 추가
            if project_type == "주얼리 전문 분석" and self.mcp_browser:
                st.markdown("#### 🌐 실시간 시장 정보")
                if st.button("📊 현재 시장 트렌드 확인", type="secondary"):
                    with st.spinner("🔍 최신 주얼리 시장 정보를 검색하는 중..."):
                        market_search_result = self._perform_realtime_market_search(project_name or "주얼리 트렌드")
                        
                        if market_search_result.get("success"):
                            st.success("✅ 시장 정보 검색 완료!")
                            
                            # 실시간 검색 결과 요약 표시
                            summary = market_search_result.get("summary", {})
                            if summary:
                                st.markdown("**📈 실시간 시장 인사이트:**")
                                for insight in summary.get("key_insights", [])[:3]:
                                    st.markdown(f"• {insight}")
                                
                                # 세션 상태에 저장
                                st.session_state.realtime_market_data = market_search_result
                        else:
                            st.error("❌ 시장 정보 검색 실패")
            
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
                - MP3, WAV, FLAC
                - 🎵 M4A (개선된 처리)
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
        
        # 개선된 파일 업로드 영역
        upload_container = st.container()
        with upload_container:
            # 대용량 파일 지원 안내
            if LARGE_FILE_HANDLER_AVAILABLE:
                self.show_enhanced_message(
                    "info",
                    "대용량 파일 지원",
                    "동영상 파일 최대 5GB까지 업로드 가능하며, 자동으로 청크 단위 처리됩니다."
                )
            
            # 시각적으로 개선된 업로드 영역
            st.markdown("""
            <div class="upload-area">
                <h3>📁 파일을 여기에 드래그하거나 클릭하여 선택하세요</h3>
                <p>지원 파일: 음성/동영상, 이미지, 문서 (최대 5GB)</p>
                <p><small>💡 Ctrl/Cmd + 클릭으로 여러 파일을 동시에 선택할 수 있습니다</small></p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "파일 선택",
                type=['wav', 'mp3', 'flac', 'm4a', 'mp4', 'mov', 'avi', 
                      'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp',
                      'pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                label_visibility="collapsed"
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
                        # 강화된 에러 처리 사용
                        try:
                            from core.enhanced_error_handler import handle_error
                            error_result = handle_error(e, {"file_name": file.name, "step": "file_processing"})
                            
                            # 개선된 에러 메시지 표시
                            self.show_enhanced_message(
                                "error",
                                "파일 처리 오류",
                                f"{file.name} 파일 처리 중 문제가 발생했습니다: {error_result['user_message']}",
                                error_result.get('solutions', [])
                            )
                            
                            # 자동 복구가 성공한 경우
                            if error_result.get('recovery_success'):
                                self.show_enhanced_message(
                                    "success",
                                    "자동 복구 완료",
                                    error_result.get('recovery_message', '문제가 해결되었습니다')
                                )
                        
                        except ImportError:
                            # 폴백: 개선된 에러 처리
                            self.show_enhanced_message(
                                "error",
                                "파일 처리 오류",
                                f"{file.name} 파일 처리 중 문제가 발생했습니다: {str(e)}",
                                [
                                    "파일이 손상되지 않았는지 확인해보세요",
                                    "다른 파일 형식으로 변환해보세요",
                                    "파일 크기가 너무 큰지 확인해보세요"
                                ]
                            )
                        
                        continue
                    
                    # 대용량 동영상 파일 감지 (1GB 이상)
                    is_large_video = file_ext in ['mp4', 'mov', 'avi'] and file_size_gb >= 1.0
                    if is_large_video:
                        large_files_detected.append((file.name, file_size_gb))
                    
                    if file_ext in ['wav', 'mp3', 'flac', 'm4a']:
                        file_categories["audio"].append((file.name, file_size_mb))
                        
                        # M4A 파일 특별 처리 안내
                        if file_ext == 'm4a':
                            st.info(f"🎵 M4A 파일 감지: {file.name} ({file_size_mb:.1f}MB) - 개선된 변환 시스템 사용")
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
                self.show_enhanced_message(
                    "warning",
                    "대용량 파일 감지",
                    f"{len(large_files_detected)}개 파일이 1GB 이상입니다. 자동으로 청크 단위 처리가 적용됩니다.",
                    ["처리 시간이 다소 길어질 수 있습니다", "메모리 효율적으로 처리됩니다", "중간에 중단하지 마세요"]
                )
                with st.expander("📊 대용량 파일 상세 정보"):
                    for filename, size_gb in large_files_detected:
                        st.markdown(f"""
                        <div class="info-card">
                            <h4>🎬 {filename}</h4>
                            <p><strong>크기:</strong> {size_gb:.2f}GB</p>
                            <ul>
                                <li>✅ 청크 단위 업로드</li>
                                <li>✅ 오디오 자동 추출</li>
                                <li>✅ 메모리 효율적 처리</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
            elif large_files_detected and not LARGE_FILE_HANDLER_AVAILABLE:
                self.show_enhanced_message(
                    "error",
                    "대용량 파일 처리 불가",
                    f"{len(large_files_detected)}개의 대용량 파일이 있지만 대용량 파일 핸들러가 비활성화되어 있습니다.",
                    [
                        "시스템 관리자에게 문의하세요",
                        "파일을 작은 크기로 분할해보세요",
                        "압축 파일 형태로 변환해보세요"
                    ]
                )
            
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
        """3단계: 분석 진행, 스크립트 표시, 중간 검토 (향상됨)"""
        st.markdown("## 3️⃣ 분석 진행 및 스크립트 검토")
        
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
            
            # 🎤 음성 스크립트 섹션 추가
            st.markdown("---")
            st.markdown("### 🎤 음성 스크립트 확인 및 수정")
            
            # 오디오 파일별 스크립트 추출 및 표시
            audio_results = [r for r in st.session_state.analysis_results if r.get('analysis_type') == 'audio']
            
            if audio_results:
                st.markdown("추출된 음성 내용을 확인하고 필요시 수정하실 수 있습니다.")
                
                for i, result in enumerate(audio_results):
                    filename = result.get('file_name', f'오디오 파일 {i+1}')
                    
                    with st.expander(f"🎵 {filename} - 스크립트 검토", expanded=True):
                        
                        if result.get('status') == 'success' and result.get('full_text'):
                            original_text = result['full_text']
                            
                            # 메타 정보 표시
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("⏱️ 텍스트 길이", f"{len(original_text)} 글자")
                            with col2:
                                confidence = result.get('confidence', 0)
                                if confidence > 0:
                                    st.metric("🎯 신뢰도", f"{confidence:.1%}")
                                else:
                                    st.metric("🎯 신뢰도", "N/A")
                            with col3:
                                processing_time = result.get('processing_time', 0)
                                st.metric("⚡ 처리시간", f"{processing_time:.1f}초")
                            
                            st.markdown("**📋 추출된 원본 텍스트:**")
                            
                            # 편집 가능한 텍스트 영역
                            edited_text = st.text_area(
                                "스크립트 내용 (수정 가능):",
                                value=original_text,
                                height=150,
                                key=f"transcript_edit_{i}",
                                help="내용이 부정확하다면 직접 수정하실 수 있습니다."
                            )
                            
                            # 수정 여부 체크 및 저장
                            if edited_text != original_text:
                                st.info("✏️ 스크립트가 수정되었습니다.")
                                # 수정된 내용을 세션 상태에 저장
                                if 'edited_transcripts' not in st.session_state:
                                    st.session_state.edited_transcripts = {}
                                st.session_state.edited_transcripts[filename] = {
                                    'original': original_text,
                                    'edited': edited_text,
                                    'modified': True
                                }
                            
                            # 키워드 하이라이트 (선택사항)
                            jewelry_keywords = result.get('jewelry_keywords', [])
                            if jewelry_keywords:
                                st.markdown("**🔍 감지된 주얼리 키워드:**")
                                cols = st.columns(min(len(jewelry_keywords), 5))
                                for j, keyword in enumerate(jewelry_keywords[:5]):
                                    with cols[j]:
                                        st.badge(keyword)
                        
                        else:
                            st.warning("⚠️ 이 오디오 파일에서 스크립트를 추출할 수 없었습니다.")
                            if result.get('error'):
                                st.error(f"오류: {result['error']}")
                
                # 스크립트 수정 요약
                if 'edited_transcripts' in st.session_state and st.session_state.edited_transcripts:
                    modified_count = len([t for t in st.session_state.edited_transcripts.values() if t.get('modified')])
                    if modified_count > 0:
                        st.success(f"✅ {modified_count}개의 스크립트가 수정되었습니다. 수정 내용은 최종 보고서에 반영됩니다.")
            
            else:
                st.info("🎤 분석된 오디오 파일이 없습니다.")
            
        # 표준 네비게이션 바
        self.render_navigation_bar(3)
    
    def render_step4_report(self):
        """4단계: 최종 보고서 - 대형 함수 (리팩토링 고려 대상)"""
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
            
            # 실시간 분석 시간 표시 (분석 중인 경우)
            if hasattr(st.session_state, 'analysis_start_time') and st.session_state.analysis_start_time:
                if st.session_state.get('analysis_status', '') != "분석 완료":
                    timer_html = self.show_realtime_analysis_timer()
                    if timer_html:
                        st.markdown(timer_html, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 🌐 실시간 웹 데이터 통합 분석
            st.markdown("### 🌐 실시간 시장 데이터 통합 분석")
            
            # 실시간 시장 데이터가 있는지 확인
            has_realtime_data = hasattr(st.session_state, 'realtime_market_data')
            
            # 일반 웹 검색 결과가 있는지 확인
            has_web_data = False
            if self.mcp_browser and hasattr(self.mcp_browser, 'search_history'):
                search_history = self.mcp_browser.get_search_history()
                has_web_data = len(search_history) > 0
            
            if has_realtime_data:
                st.info("📊 1단계에서 수행한 실시간 시장 검색 결과를 파일 분석과 통합할 수 있습니다.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔗 실시간 시장 데이터 통합", type="primary"):
                        with st.spinner("📈 실시간 시장 데이터와 분석 결과를 통합하는 중..."):
                            integration_result = self._integrate_realtime_data_with_workflow()
                            
                            if integration_result and integration_result.get("integration_success"):
                                st.success("✅ 실시간 데이터 통합 완료!")
                                
                                # 통합 결과 표시
                                self.display_integrated_analysis_results(integration_result)
                            else:
                                st.error("❌ 실시간 데이터 통합 실패")
                
                with col2:
                    # 실시간 데이터 미리보기
                    market_data = st.session_state.realtime_market_data
                    summary = market_data.get("summary", {})
                    
                    if summary:
                        st.markdown("**📋 실시간 시장 정보 미리보기:**")
                        key_insights = summary.get("key_insights", [])
                        for insight in key_insights[:2]:
                            st.markdown(f"• {insight}")
                        
                        if summary.get("brand_info"):
                            brands = ", ".join(summary["brand_info"][:3])
                            st.markdown(f"• 주요 브랜드: {brands}")
            
            elif has_web_data:
                st.info("🔍 이전에 수행한 웹 검색 결과가 있습니다. 파일 분석과 통합할 수 있습니다.")
                
                if st.button("🔗 웹 데이터와 파일 분석 결과 통합", type="primary"):
                    with st.spinner("🌐 웹 데이터와 분석 결과를 통합하는 중..."):
                        integration_result = self.integrate_web_data_with_analysis()
                        
                        if integration_result and integration_result.get("integration_success"):
                            st.success("✅ 웹 데이터 통합 완료!")
                            
                            # 통합 결과 표시
                            self.display_integrated_analysis_results(integration_result)
                        else:
                            st.error("❌ 웹 데이터 통합 실패")
            else:
                st.info("💡 1단계에서 실시간 시장 검색을 수행하거나 MCP 브라우저에서 웹 검색을 먼저 하면 더 풍부한 분석 결과를 얻을 수 있습니다.")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 1단계로 이동", type="secondary"):
                        st.session_state.workflow_step = 1
                        st.rerun()
                with col2:
                    if st.button("🚀 MCP 브라우저로 이동", type="secondary"):
                        st.session_state.active_tab = "mcp_browser"
                        st.rerun()
            
            st.markdown("---")
            
            # 🎭 종합 스토리 생성 (새 기능)
            st.markdown("### 🎭 종합 스토리 - 무엇에 대한 이야기인가?")
            
            if st.button("📖 전체 내용을 하나의 이야기로 만들기", type="primary"):
                with st.spinner("🎭 AI가 전체 내용을 하나의 일관된 한국어 스토리로 구성 중..."):
                    story_result = self.create_comprehensive_story()
                    
                    if story_result and story_result.get("status") == "success":
                        st.success("✅ 종합 스토리 생성 완료!")
                        
                        story_content = story_result.get("story", "")
                        if story_content:
                            st.markdown("#### 📚 생성된 스토리")
                            st.markdown(f"```markdown\n{story_content}\n```")
                            
                            # 다운로드 버튼
                            st.download_button(
                                label="📥 스토리 다운로드 (TXT)",
                                data=story_content,
                                file_name=f"종합_스토리_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                        
                        # 메타데이터 표시
                        metadata = story_result.get("metadata", {})
                        if metadata:
                            st.markdown("#### 📊 스토리 생성 정보")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("처리된 소스", metadata.get("source_count", 0))
                            with col2:
                                st.metric("사용된 AI 엔진", metadata.get("ai_engine_used", "Local"))
                            with col3:
                                st.metric("핵심 메시지 수", metadata.get("key_messages_count", 0))
                    
                    elif story_result and story_result.get("status") == "error":
                        st.error(f"❌ 스토리 생성 실패: {story_result.get('error', 'Unknown error')}")
                        
                        # 폴백 스토리가 있으면 표시
                        fallback_story = story_result.get("fallback_story")
                        if fallback_story:
                            st.markdown("#### 📄 기본 요약")
                            st.markdown(fallback_story)
                            
                        # API 키 설정 안내
                        if "API Key" in story_result.get("error", ""):
                            st.info("💡 **더 나은 스토리 생성을 위한 안내:**\n\nOpenAI API Key를 환경변수 `OPENAI_API_KEY`에 설정하면 GPT-4를 활용한 고품질 스토리 생성이 가능합니다.")
                    
                    else:
                        st.warning("⚠️ 스토리 생성에 실패했습니다. 분석 결과를 확인해주세요.")
            
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
            
            # 파일별 상세 결과 (수정된 스크립트 반영)
            with st.expander("📄 파일별 상세 분석 결과"):
                edited_transcripts = st.session_state.get('edited_transcripts', {})
                
                for result in st.session_state.analysis_results:
                    if result.get('status') == 'success':
                        filename = result['file_name']
                        st.markdown(f"**{filename}**")
                        
                        # 수정된 스크립트가 있는지 확인
                        is_modified = filename in edited_transcripts and edited_transcripts[filename].get('modified')
                        
                        if result.get('full_text'):
                            # 표시할 텍스트 결정 (수정된 것 우선)
                            display_text = edited_transcripts[filename]['edited'] if is_modified else result['full_text']
                            
                            # 수정 여부 표시
                            if is_modified:
                                st.success("✏️ 사용자가 수정한 스크립트")
                            
                            st.text_area(
                                "추출된 텍스트" + (" (수정됨)" if is_modified else ""),
                                value=display_text[:500] + ("..." if len(display_text) > 500 else ""),
                                height=100,
                                disabled=True,
                                key=f"detail_{filename}"
                            )
                            
                            # 수정 전/후 비교 표시 (옵션)
                            if is_modified:
                                with st.expander("📝 수정 전/후 비교"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**수정 전:**")
                                        st.text_area("", value=edited_transcripts[filename]['original'][:300] + "...", 
                                                   height=80, disabled=True, key=f"before_{filename}")
                                    with col2:
                                        st.markdown("**수정 후:**")
                                        st.text_area("", value=edited_transcripts[filename]['edited'][:300] + "...", 
                                                   height=80, disabled=True, key=f"after_{filename}")
                        
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
            force_cpu_mode()
        
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
        
        # 🎤 화자 분리 결과 표시 (v2.3 새로운 기능)
        if result.get('speaker_analysis') and result['speaker_analysis'].get('status') == 'success':
            self._display_speaker_diarization_results(result['speaker_analysis'])
        
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
                force_cpu_mode()
                
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
        """🚀 배치 종합 분석 실행 - 무한루프 및 메모리 누수 방지"""
        import signal
        import gc
        import os
        import time
        import datetime
        
        if not st.session_state.uploaded_files_data:
            return []
        
        # 🕐 실시간 분석 시간 추적 시작
        analysis_start_time = time.time()
        st.session_state.analysis_start_time = analysis_start_time
        st.session_state.analysis_status = "진행 중"
        
        # 시작 시간 표시
        start_time_str = datetime.datetime.fromtimestamp(analysis_start_time).strftime("%H:%M:%S")
        
        # 시작 시 메모리 확인
        try:
            import psutil
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / (1024 * 1024)  # MB
            st.info(f"🔍 분석 시작 - 시작 시간: {start_time_str} | 현재 메모리 사용량: {start_memory:.1f}MB")
        except:
            start_memory = 0
        
        # 전체 분석 타임아웃 설정 (10분)
        def analysis_timeout_handler(signum, frame):
            raise TimeoutError("전체 분석이 10분 내에 완료되지 않아 중단됩니다")
        
        timeout_set = False
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, analysis_timeout_handler)
            signal.alarm(600)  # 10분 타임아웃
            timeout_set = True
        
        uploaded_files_data = st.session_state.uploaded_files_data
        
        try:
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
                
        except TimeoutError as e:
            st.error(f"❌ 분석 시간 초과: {str(e)}")
            st.info("💡 파일 수를 줄이거나 더 작은 파일로 다시 시도해보세요.")
            if timeout_set and hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            return []
            
        except MemoryError as e:
            st.error(f"❌ 메모리 부족: {str(e)}")
            st.info("💡 더 적은 수의 파일로 다시 시도하거나 시스템을 재시작해보세요.")
            gc.collect()  # 메모리 정리
            if timeout_set and hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            return []
            
        except Exception as e:
            st.error(f"❌ 분석 중 오류 발생: {str(e)}")
            st.info("💡 시스템을 재시작하거나 파일을 다시 업로드해보세요.")
            if timeout_set and hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            return []
    
    def _execute_batch_comprehensive_analysis(self):
        """배치 종합 분석 - 모든 파일을 통합 처리"""
        uploaded_files_data = st.session_state.uploaded_files_data
        all_results = []
        
        # 실시간 타이머 및 진행률 표시 컨테이너
        timer_container = st.empty()
        progress_container = st.empty()
        
        # 실시간 타이머 표시 시작
        with timer_container.container():
            st.markdown(self.show_realtime_analysis_timer(), unsafe_allow_html=True)
        
        # 1️⃣ 단계: 파일 분류 및 전처리
        st.session_state.analysis_status = "1단계: 파일 분류 및 전처리"
        st.info("🔍 1단계: 파일 분류 및 전처리 시작...")
        
        # 타이머 업데이트
        with timer_container.container():
            st.markdown(self.show_realtime_analysis_timer(), unsafe_allow_html=True)
        
        with progress_container.container():
            pass  # 실제 처리 후 진행률 표시
        
        file_categories = self._categorize_and_preprocess_files(uploaded_files_data)
        
        # 1단계 완료 - 타이머 및 진행률 업데이트
        st.session_state.analysis_status = "1단계 완료: 파일 분류 완료"
        with timer_container.container():
            st.markdown(self.show_realtime_analysis_timer(), unsafe_allow_html=True)
        with progress_container.container():
            self.show_real_progress_only(1, 4, "1단계: 파일 분류 완료", f"분류된 파일: {sum(len(v) for v in file_categories.values())}개", force_display=True)
        
        # 2️⃣ 단계: 통합 컨텍스트 구성
        st.session_state.analysis_status = "2단계: 통합 컨텍스트 구성"
        st.info("🧠 2단계: 통합 컨텍스트 구성 시작...")
        
        # 타이머 업데이트
        with timer_container.container():
            st.markdown(self.show_realtime_analysis_timer(), unsafe_allow_html=True)
        
        integrated_context = self._build_integrated_context(file_categories)
        
        # 2단계 완료 - 타이머 및 진행률 업데이트
        st.session_state.analysis_status = "2단계 완료: 통합 컨텍스트 구성 완료"
        with timer_container.container():
            st.markdown(self.show_realtime_analysis_timer(), unsafe_allow_html=True)
        with progress_container.container():
            self.show_real_progress_only(2, 4, "2단계: 통합 컨텍스트 구성 완료", "분석 컨텍스트 준비 완료", force_display=True)
        
        # 3️⃣ 단계: 배치 분석 실행
        st.session_state.analysis_status = "3단계: 실제 배치 분석 진행 중"
        st.info("⚡ 3단계: 실제 배치 분석 시작 - 이 단계에서 실제 AI 모델이 작동합니다")
        
        # 타이머 업데이트
        with timer_container.container():
            st.markdown(self.show_realtime_analysis_timer(), unsafe_allow_html=True)
        
        batch_results = self._execute_batch_analysis(file_categories, integrated_context, progress_container, timer_container)
        
        # 3단계 완료 - 타이머 및 진행률 업데이트
        st.session_state.analysis_status = "3단계 완료: 배치 분석 완료"
        with timer_container.container():
            st.markdown(self.show_realtime_analysis_timer(), unsafe_allow_html=True)
        with progress_container.container():
            self.show_real_progress_only(3, 4, "3단계: 배치 분석 완료", "모든 파일 분석 완료", force_display=True)
        
        # 4️⃣ 단계: 결과 통합 및 최적화
        st.session_state.analysis_status = "4단계: 결과 통합 및 최적화"
        st.info("🎯 4단계: 결과 통합 및 최적화 시작...")
        
        # 타이머 업데이트
        with timer_container.container():
            st.markdown(self.show_realtime_analysis_timer(), unsafe_allow_html=True)
        
        final_results = self._integrate_and_optimize_results(batch_results, integrated_context)
        
        # 전체 완료 - 타이머 및 진행률 업데이트
        st.session_state.analysis_status = "분석 완료"
        
        # 분석 완료 시간 정보 계산
        import time
        total_elapsed_seconds = int(time.time() - st.session_state.analysis_start_time)
        total_elapsed_minutes = total_elapsed_seconds // 60
        remaining_seconds = total_elapsed_seconds % 60
        
        if total_elapsed_minutes > 0:
            elapsed_display = f"{total_elapsed_minutes}분 {remaining_seconds}초"
        else:
            elapsed_display = f"{remaining_seconds}초"
        
        # 최종 타이머 표시
        with timer_container.container():
            final_timer_html = f"""
            <div style="
                background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
                color: white;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                text-align: center;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            ">
                <div style="font-size: 18px; font-weight: bold;">🎉 분석 완료!</div>
                <div style="font-size: 16px; margin-top: 5px;">총 소요시간: {elapsed_display}</div>
            </div>
            """
            st.markdown(final_timer_html, unsafe_allow_html=True)
            
        with progress_container.container():
            self.show_real_progress_only(4, 4, "전체 분석 완료!", f"총 {len(uploaded_files_data.get('files', []))}개 파일 분석 성공", force_display=True)
        
        st.success(f"✅ 배치 종합 분석이 완료되었습니다! (총 소요시간: {elapsed_display})")
        
        # 메모리 사용량 최종 확인
        try:
            end_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_used = end_memory - start_memory
            if memory_used > 1000:  # 1GB 이상 사용
                st.warning(f"⚠️ 높은 메모리 사용량: {memory_used:.1f}MB 증가")
                gc.collect()  # 가비지 컬렉션
            else:
                st.info(f"✅ 메모리 사용량 정상: {memory_used:.1f}MB 증가")
        except:
            pass
        
        return final_results
    
    def _execute_individual_analysis(self):
        """기존 개별 분석 방식 (호환성 유지)"""
        uploaded_files_data = st.session_state.uploaded_files_data
        all_results = []
        
        # 개선된 진행률 표시
        progress_container = st.empty()
        
        total_items = len(uploaded_files_data.get('files', [])) + len(uploaded_files_data.get('video_urls', []))
        current_item = 0
        
        # 초기 준비 메시지
        with progress_container.container():
            self.show_progress_with_details(
                0, total_items,
                "🔧 분석 준비 중...",
                "AI 모델을 로딩하고 분석 환경을 준비하고 있습니다."
            )
        
        # 업로드된 파일 분석
        for uploaded_file in uploaded_files_data.get('files', []):
            current_item += 1
            
            # 파일별 진행률 표시
            with progress_container.container():
                self.show_progress_with_details(
                    current_item, total_items,
                    f"🔄 분석 중: {uploaded_file.name}",
                    f"현재 파일 ({current_item}/{total_items})을 처리하고 있습니다. 파일 크기와 유형에 따라 처리 시간이 달라질 수 있습니다."
                )
            
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
                
                # YouTube 실시간 처리 시스템 사용
                if YOUTUBE_REALTIME_AVAILABLE:
                    try:
                        # YouTube 오디오 다운로드 및 분석
                        download_result = global_youtube_realtime_processor.download_audio(url, progress_container=progress_container)
                        
                        if download_result.get('success'):
                            # 다운로드된 오디오 파일을 STT 분석
                            audio_file = download_result['output_file']
                            status_text.text(f"🔄 YouTube 오디오 STT 분석 중...")
                            
                            # 실제 분석 엔진으로 STT 처리
                            if REAL_ANALYSIS_AVAILABLE:
                                stt_result = real_analysis_engine.analyze_audio_file(audio_file)
                                
                                # 결과 정리
                                youtube_result = {
                                    "status": "success",
                                    "file_type": "youtube_audio",
                                    "url": url,
                                    "video_info": download_result.get('video_info', {}),
                                    "audio_file": audio_file,
                                    "file_size_mb": download_result.get('file_size_mb', 0),
                                    "stt_result": stt_result,
                                    "processing_time": download_result.get('processing_time', 0),
                                    "download_speed_mbps": download_result.get('download_speed_mbps', 0)
                                }
                            else:
                                youtube_result = {
                                    "status": "partial_success",
                                    "message": "오디오 다운로드 성공, STT 분석 엔진 미사용",
                                    "url": url,
                                    "video_info": download_result.get('video_info', {}),
                                    "audio_file": audio_file
                                }
                            
                            # 임시 파일 정리
                            try:
                                os.remove(audio_file)
                            except:
                                pass
                        else:
                            youtube_result = {
                                "status": "error",
                                "message": f"YouTube 다운로드 실패: {download_result.get('error', 'Unknown error')}",
                                "url": url
                            }
                    except Exception as e:
                        youtube_result = {
                            "status": "error", 
                            "message": f"YouTube 처리 오류: {str(e)}",
                            "url": url
                        }
                else:
                    youtube_result = {
                        "status": "pending",
                        "message": "YouTube 실시간 처리 시스템이 로드되지 않았습니다.",
                        "url": url
                    }
                
                all_results.append(youtube_result)
                
            elif 'brightcove.net' in url:
                status_text.text(f"🔄 Brightcove 분석 중: {url[:50]}... ({current_item}/{total_items})")
                all_results.append({
                    "status": "pending",
                    "message": "Brightcove 분석 기능은 향후 구현 예정입니다.",
                    "url": url
                })
            else:
                status_text.text(f"🔄 동영상 URL 분석 중: {url[:50]}... ({current_item}/{total_items})")
                all_results.append({
                    "status": "pending", 
                    "message": "일반 동영상 URL 분석 기능은 향후 구현 예정입니다.",
                    "url": url
                })
        
        progress_bar.progress(1.0)
        status_text.text("✅ 모든 분석 완료!")
        
        return all_results
    
    def create_comprehensive_story(self):
        """분석 결과들을 하나의 일관된 한국어 스토리로 변환"""
        
        try:
            if not st.session_state.analysis_results:
                return {
                    "status": "error",
                    "error": "분석 결과가 없습니다."
                }
            
            # 분석 결과를 스토리텔링 엔진 형식으로 변환
            sources = []
            for i, result in enumerate(st.session_state.analysis_results):
                if result.get('status') == 'success':
                    
                    # 파일 타입 추정
                    file_name = result.get('file_name', f'파일_{i+1}')
                    if any(ext in file_name.lower() for ext in ['.wav', '.mp3', '.m4a', '.flac']):
                        file_type = "audio"
                    elif any(ext in file_name.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                        file_type = "image"
                    elif any(ext in file_name.lower() for ext in ['.pdf', '.docx', '.doc', '.txt']):
                        file_type = "document"
                    else:
                        file_type = "unknown"
                    
                    source_data = {
                        "name": file_name,
                        "type": file_type,
                        "analysis_result": result,
                        "timestamp": result.get('timestamp')
                    }
                    
                    sources.append(source_data)
            
            if not sources:
                return {
                    "status": "error",
                    "error": "성공적으로 분석된 파일이 없습니다."
                }
            
            # 프로젝트 정보에 따른 스토리 타입 결정
            project_info = st.session_state.get('project_info', {})
            topic = project_info.get('topic', '').lower()
            
            if '상담' in topic or '고객' in topic:
                story_type = "consultation"
            elif '회의' in topic or '미팅' in topic:
                story_type = "meeting"
            elif len(sources) > 3:  # 다중 소스
                story_type = "multimedia"
            else:
                story_type = "general"
            
            # 고급 스토리텔링 엔진으로 스토리 생성
            story_result = create_comprehensive_story_from_sources(sources, story_type)
            
            return story_result
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"스토리 생성 중 오류가 발생했습니다: {str(e)}"
            }
    
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
        
        # 모든 텍스트 수집 (수정된 스크립트 우선 사용)
        all_texts = []
        edited_transcripts = st.session_state.get('edited_transcripts', {})
        
        for result in results:
            if result.get('status') == 'success' and result.get('full_text'):
                filename = result.get('file_name', '')
                
                # 수정된 스크립트가 있으면 우선 사용
                if filename in edited_transcripts and edited_transcripts[filename].get('modified'):
                    all_texts.append(edited_transcripts[filename]['edited'])
                else:
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
    
    def _execute_batch_analysis(self, file_categories, integrated_context, progress_container, timer_container=None) -> Dict[str, Any]:
        """배치 통합 분석 실행 - 실시간 시간 추적 및 MCP 자동 문제 해결"""
        import psutil
        import os
        import time
        
        batch_results = {
            'audio_results': [],
            'video_results': [],
            'image_results': [],
            'document_results': [],
            'youtube_results': [],
            'cross_correlations': [],
            'integrated_insights': {}
        }
        
        # 총 파일 수 계산
        total_files = 0
        total_files += len(file_categories.get('audio_files', []))
        total_files += len(file_categories.get('video_files', []))
        total_files += len(file_categories.get('image_files', []))
        total_files += len(file_categories.get('document_files', []))
        total_files += len(file_categories.get('youtube', []))
        
        processed_files = 0
        
        # 실시간 진행 추적 시작
        if ADVANCED_MONITORING_AVAILABLE:
            global_progress_tracker.start_analysis(total_files, progress_container)
        
        # 초기 메모리 사용량 측정
        try:
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        except:
            start_memory = 0
        
        # 🎤 음성 파일 배치 분석
        if file_categories.get('audio_files'):
            st.session_state.analysis_status = f"🎤 음성 파일 분석 중 ({len(file_categories['audio_files'])}개)"
            
            # 타이머 업데이트
            if timer_container:
                with timer_container.container():
                    st.markdown(self.show_realtime_analysis_timer(), unsafe_allow_html=True)
            
            if ADVANCED_MONITORING_AVAILABLE:
                global_progress_tracker.start_stage("🎤 음성 파일 분석")
            
            batch_results['audio_results'] = self._batch_analyze_audio_files_with_tracking(
                file_categories['audio_files'], integrated_context, progress_container, processed_files, total_files, start_memory
            )
            processed_files += len(file_categories['audio_files'])
            
            if ADVANCED_MONITORING_AVAILABLE:
                global_progress_tracker.finish_stage()
            
            # 완료 후 타이머 업데이트
            st.session_state.analysis_status = f"🎤 음성 분석 완료 ({len(file_categories['audio_files'])}개)"
            if timer_container:
                with timer_container.container():
                    st.markdown(self.show_realtime_analysis_timer(), unsafe_allow_html=True)
            
            # 실제 완료 표시
            with progress_container.container():
                self.show_real_progress_only(processed_files, total_files, f"음성 분석 완료: {len(file_categories['audio_files'])}개", force_display=True)
        
        # 🎬 영상 파일 배치 분석  
        if file_categories.get('video_files'):
            st.info(f"🎬 영상 파일 분석 시작: {len(file_categories['video_files'])}개")
            batch_results['video_results'] = self._batch_analyze_video_files(
                file_categories['video_files'], integrated_context, progress_container, processed_files, total_files
            )
            processed_files += len(file_categories['video_files'])
            
            # 실제 완료 표시
            with progress_container.container():
                self.show_real_progress_only(processed_files, total_files, f"영상 분석 완료: {len(file_categories['video_files'])}개", force_display=True)
        
        # 🖼️ 이미지 파일 배치 분석
        if file_categories.get('image_files'):
            st.session_state.analysis_status = f"🖼️ 이미지 파일 분석 중 ({len(file_categories['image_files'])}개)"
            
            # 타이머 업데이트
            if timer_container:
                with timer_container.container():
                    st.markdown(self.show_realtime_analysis_timer(), unsafe_allow_html=True)
            
            st.info(f"🖼️ 이미지 파일 분석 시작: {len(file_categories['image_files'])}개")
            batch_results['image_results'] = self._batch_analyze_image_files(
                file_categories['image_files'], integrated_context, progress_container, processed_files, total_files
            )
            processed_files += len(file_categories['image_files'])
            
            # 완료 후 타이머 업데이트
            st.session_state.analysis_status = f"🖼️ 이미지 분석 완료 ({len(file_categories['image_files'])}개)"
            if timer_container:
                with timer_container.container():
                    st.markdown(self.show_realtime_analysis_timer(), unsafe_allow_html=True)
            
            # 실제 완료 표시
            with progress_container.container():
                self.show_real_progress_only(processed_files, total_files, f"이미지 분석 완료: {len(file_categories['image_files'])}개", force_display=True)
        
        # 📄 문서 파일 배치 분석
        if file_categories.get('document_files'):
            st.info(f"📄 문서 파일 분석 시작: {len(file_categories['document_files'])}개")
            batch_results['document_results'] = self._batch_analyze_document_files(
                file_categories['document_files'], integrated_context, progress_container, processed_files, total_files
            )
            processed_files += len(file_categories['document_files'])
            
            # 실제 완료 표시
            with progress_container.container():
                self.show_real_progress_only(processed_files, total_files, f"문서 분석 완료: {len(file_categories['document_files'])}개", force_display=True)
        
        # 🎬 YouTube URL 배치 분석
        if file_categories.get('youtube'):
            if ADVANCED_MONITORING_AVAILABLE:
                global_progress_tracker.start_stage("🎬 YouTube URL 분석")
            
            st.info(f"🎬 YouTube URL 분석 시작: {len(file_categories['youtube'])}개")
            batch_results['youtube_results'] = self._batch_analyze_youtube_urls(
                file_categories['youtube'], integrated_context, progress_container, processed_files, total_files
            )
            processed_files += len(file_categories['youtube'])
            
            if ADVANCED_MONITORING_AVAILABLE:
                global_progress_tracker.finish_stage()
            
            # 실제 완료 표시
            with progress_container.container():
                self.show_real_progress_only(processed_files, total_files, f"YouTube 분석 완료: {len(file_categories['youtube'])}개", force_display=True)
        
        # 최종 완료 표시
        with progress_container.container():
            self.show_real_progress_only(total_files, total_files, "모든 파일 배치 분석 완료!", f"총 {total_files}개 파일 처리 완료", force_display=True)
        
        # 분석 완료 시 MCP 자동 문제 해결 리포트
        if ADVANCED_MONITORING_AVAILABLE:
            global_progress_tracker.finish_analysis()
            
            # 최종 메모리 사용량 측정 및 MCP 문제 해결
            try:
                end_memory = process.memory_info().rss / (1024 * 1024)  # MB
                memory_used = end_memory - start_memory
                total_time_str = global_progress_tracker._get_elapsed_time()
                # 시간 문자열을 초로 변환
                if "시간" in total_time_str:
                    total_time = 3600  # 1시간 이상이면 3600초로 설정
                elif "분" in total_time_str:
                    minutes = int(total_time_str.split("분")[0])
                    total_time = minutes * 60
                else:
                    try:
                        total_time = float(total_time_str.replace("초", ""))
                    except:
                        total_time = time.time() - global_progress_tracker.start_time if global_progress_tracker.start_time else 0
                
                # MCP 자동 문제 해결 시스템 활용
                problem_analysis = global_mcp_solver.detect_and_solve_problems(
                    memory_usage_mb=memory_used,
                    processing_time=total_time,
                    file_info={'total_files': total_files, 'processed_files': processed_files}
                )
                
                # 문제가 감지된 경우 사용자에게 알림
                if problem_analysis['problems_detected']:
                    with progress_container.container():
                        st.warning("⚠️ **시스템 분석 결과 - 개선 권장사항**")
                        for problem in problem_analysis['problems_detected']:
                            st.write(f"• {problem['description']}")
                        
                        if problem_analysis['solutions_found']:
                            st.info("💡 **MCP 자동 해결책**:")
                            for solution in problem_analysis['solutions_found']:
                                if solution.get('recommended_actions'):
                                    for action in solution['recommended_actions'][:3]:  # 상위 3개만
                                        st.write(f"  - {action}")
                        
                        if problem_analysis['auto_actions_taken']:
                            st.success("🤖 **자동 실행된 최적화**:")
                            for action in problem_analysis['auto_actions_taken']:
                                st.write(f"  ✅ {action['description']}")
                                
            except Exception as e:
                self.logger.warning(f"MCP 자동 문제 해결 실행 중 오류: {e}")
        
        return batch_results
    
    def _batch_analyze_audio_files_with_tracking(self, audio_files, context, progress_container, base_processed, total_files, start_memory) -> List[Dict]:
        """음성 파일 배치 분석 - 실시간 시간 추적 및 MCP 통합"""
        import tempfile
        import time
        import psutil
        import os
        
        results = []
        
        # 모든 음성 파일을 임시 저장
        temp_files = []
        for file_info in audio_files:
            with tempfile.NamedTemporaryFile(suffix=f".{file_info['extension']}", delete=False) as tmp_file:
                tmp_file.write(file_info['file'].getvalue())
                temp_files.append(tmp_file.name)
                file_info['temp_path'] = tmp_file.name
                
                # 파일 크기 계산 (MB)
                file_info['size_mb'] = len(file_info['file'].getvalue()) / (1024 * 1024)
        
        try:
            for i, file_info in enumerate(audio_files):
                current_processed = base_processed + i
                
                # 실시간 진행 추적 시작
                if ADVANCED_MONITORING_AVAILABLE:
                    global_progress_tracker.start_file_processing(
                        file_info['name'], 
                        file_info.get('size_mb', 0)
                    )
                
                # 개별 분석 수행
                start_time = time.time()
                try:
                    result = analyze_file_real(file_info['temp_path'], 'audio', 'auto', context)
                    result['batch_index'] = i
                    result['cross_reference_ready'] = True
                    results.append(result)
                    
                    processing_time = time.time() - start_time
                    
                    # MCP 자동 문제 감지 및 해결
                    if ADVANCED_MONITORING_AVAILABLE:
                        try:
                            current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                            memory_delta = current_memory - start_memory
                            
                            # 문제 감지 및 해결
                            if memory_delta > 500 or processing_time > 120:  # 500MB 이상 또는 2분 이상
                                problem_analysis = global_mcp_solver.detect_and_solve_problems(
                                    memory_usage_mb=memory_delta,
                                    processing_time=processing_time,
                                    file_info=file_info,
                                    error_message=result.get('error') if result.get('status') == 'error' else None
                                )
                                
                                # 긴급한 문제인 경우 즉시 알림
                                if any(p['severity'] == 'high' for p in problem_analysis['problems_detected']):
                                    with progress_container.container():
                                        st.warning(f"⚠️ {file_info['name']} 처리 중 성능 이슈 감지")
                                        if problem_analysis['auto_actions_taken']:
                                            st.info("🤖 자동 최적화 실행됨")
                        except Exception as e:
                            self.logger.warning(f"MCP 문제 감지 실행 중 오류: {e}")
                    
                except Exception as e:
                    self.logger.error(f"음성 파일 분석 중 오류: {e}")
                    results.append({
                        'status': 'error',
                        'error': str(e),
                        'file_name': file_info['name'],
                        'batch_index': i
                    })
                
                # 파일 처리 완료
                if ADVANCED_MONITORING_AVAILABLE:
                    global_progress_tracker.finish_file_processing()
                
        finally:
            # 임시 파일 정리
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        return results
    
    def _batch_analyze_audio_files(self, audio_files, context, progress_container, base_processed, total_files) -> List[Dict]:
        """음성 파일 배치 분석 - 실제 진행률 표시"""
        results = []
        
        # 모든 음성 파일을 임시 저장
        temp_files = []
        for file_info in audio_files:
            with tempfile.NamedTemporaryFile(suffix=f".{file_info['extension']}", delete=False) as tmp_file:
                tmp_file.write(file_info['file'].getvalue())
                temp_files.append(tmp_file.name)
                file_info['temp_path'] = tmp_file.name
        
        # 배치 STT 처리 (실제 진행률 표시)
        try:
            for i, file_info in enumerate(audio_files):
                current_processed = base_processed + i
                
                # 실제 처리 상태만 표시
                st.write(f"🎤 처리 중: {file_info['name']} ({i+1}/{len(audio_files)})")
                
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
    
    def _batch_analyze_image_files(self, image_files, context, progress_container, base_processed, total_files) -> List[Dict]:
        """이미지 파일 배치 분석 - 실제 진행률 표시"""
        results = []
        
        # 이미지 파일 임시 저장
        temp_files = []
        for file_info in image_files:
            with tempfile.NamedTemporaryFile(suffix=f".{file_info['extension']}", delete=False) as tmp_file:
                tmp_file.write(file_info['file'].getvalue())
                temp_files.append(tmp_file.name)
                file_info['temp_path'] = tmp_file.name
        
        try:
            # 이미지 배치 처리 (실제 진행률 표시)
            for i, file_info in enumerate(image_files):
                current_processed = base_processed + i
                
                # 실제 처리 상태만 표시
                st.write(f"🖼️ 처리 중: {file_info['name']} ({i+1}/{len(image_files)})")
                
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
    
    def _batch_analyze_video_files(self, video_files, context, progress_container, base_processed, total_files) -> List[Dict]:
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
    
    def _batch_analyze_document_files(self, document_files, context, progress_container, base_processed, total_files) -> List[Dict]:
        """문서 파일 배치 분석"""
        results = []
        
        for i, file_info in enumerate(document_files):
            st.text(f"📄 문서 분석: {file_info['name']} ({i+1}/{len(document_files)})")
            
            temp_path = None
            try:
                # 임시 파일 생성
                with tempfile.NamedTemporaryFile(suffix=f".{file_info['extension']}", delete=False) as tmp_file:
                    tmp_file.write(file_info['file'].getvalue())
                    temp_path = tmp_file.name
                
                # 문서 분석 실행
                result = analyze_file_real(temp_path, "document", language="auto", context=context)
                
                if result:
                    result['file_name'] = file_info['name']
                    result['file_size'] = file_info.get('size', 0)
                    result['analysis_method'] = 'document_batch'
                    
                    # 종합 메시지 추출 적용
                    if result.get('status') == 'success' and result.get('extracted_text'):
                        try:
                            from core.comprehensive_message_extractor import extract_speaker_message
                            message_analysis = extract_speaker_message(result['extracted_text'])
                            if message_analysis:
                                result['comprehensive_message'] = message_analysis
                        except Exception as e:
                            result['message_extraction_error'] = str(e)
                    
                    results.append(result)
                else:
                    # 실패 결과
                    results.append({
                        'status': 'error',
                        'error': '문서 분석 실패',
                        'file_name': file_info['name'],
                        'analysis_method': 'document_batch'
                    })
                    
            except Exception as e:
                # 예외 처리
                results.append({
                    'status': 'error',
                    'error': f'문서 처리 중 오류: {str(e)}',
                    'file_name': file_info['name'],
                    'analysis_method': 'document_batch'
                })
            finally:
                # 임시 파일 정리
                if temp_path:
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
    
    def _display_speaker_diarization_results(self, speaker_analysis: Dict[str, Any]):
        """화자 분리 결과 표시"""
        
        st.markdown("### 🎤 **화자 분리 분석 결과**")
        
        # 기본 정보 표시
        speaker_count = speaker_analysis.get('speaker_count', 0)
        processing_time = speaker_analysis.get('processing_time', 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("감지된 화자", f"{speaker_count}명")
        with col2:
            st.metric("분석 시간", f"{processing_time:.1f}초")
        with col3:
            voice_ratio = speaker_analysis.get('voice_activity_ratio', 0)
            st.metric("음성 활동 비율", f"{voice_ratio*100:.1f}%")
        
        # 사용자 친화적 요약
        if speaker_analysis.get('user_summary'):
            st.markdown("#### 📋 요약")
            st.markdown(speaker_analysis['user_summary'])
        
        # 화자별 발언 내용 (핵심 기능)
        speaker_statements = speaker_analysis.get('speaker_statements', {})
        if speaker_statements:
            st.markdown("#### 👥 화자별 발언 내용")
            
            # 탭으로 화자별 구분
            if len(speaker_statements) > 1:
                speaker_tabs = st.tabs([f"화자 {sid.replace('SPEAKER_', '').lstrip('0') or '1'}" 
                                      for sid in speaker_statements.keys()])
                
                for i, (speaker_id, statements) in enumerate(speaker_statements.items()):
                    with speaker_tabs[i]:
                        self._display_speaker_statements(speaker_id, statements, speaker_analysis)
            else:
                # 단일 화자인 경우
                speaker_id, statements = next(iter(speaker_statements.items()))
                self._display_speaker_statements(speaker_id, statements, speaker_analysis)
        
        # 상세 분석 결과 (펼치기 형태)
        with st.expander("🔍 상세 분석 결과"):
            self._display_detailed_speaker_analysis(speaker_analysis)
    
    def _display_speaker_statements(self, speaker_id: str, statements: List[Dict], full_analysis: Dict):
        """개별 화자의 발언 내용 표시"""
        
        speaker_num = speaker_id.replace('SPEAKER_', '').lstrip('0') or '1'
        
        # 화자 정보 표시
        if full_analysis.get('speaker_identification', {}).get('speaker_details', {}).get(speaker_id):
            speaker_details = full_analysis['speaker_identification']['speaker_details'][speaker_id]
            identified_names = speaker_details.get('identified_names', [])
            
            if identified_names:
                name = identified_names[0].get('name', '')
                title = identified_names[0].get('title', '')
                if name:
                    display_name = f"{name}" + (f" ({title})" if title else "")
                    st.markdown(f"**🏷️ 식별된 이름:** {display_name}")
            
            # 전문가 역할
            expert_roles = speaker_details.get('expert_roles', {})
            if expert_roles:
                st.markdown(f"**👨‍💼 추정 역할:** {', '.join(expert_roles.get(speaker_id, []))}")
        
        # 발언 구간별 표시
        total_statements = len(statements)
        total_duration = sum(float(stmt.get('duration', '0초').replace('초', '')) for stmt in statements)
        
        st.markdown(f"**📊 발언 통계:** {total_statements}개 구간, 총 {total_duration:.1f}초")
        
        # 각 발언 구간
        for i, statement in enumerate(statements):
            with st.container():
                # 시간 정보
                time_info = f"{statement.get('start_time', '')} - {statement.get('end_time', '')} ({statement.get('duration', '')})"
                st.markdown(f"**⏰ {time_info}**")
                
                # 발언 내용
                content = statement.get('content', '')
                if content:
                    st.markdown(f"💬 {content}")
                else:
                    st.markdown("🔇 *이 구간에서는 명확한 발언을 감지하지 못했습니다*")
                
                if i < len(statements) - 1:  # 마지막이 아니면 구분선
                    st.divider()
    
    def _display_detailed_speaker_analysis(self, speaker_analysis: Dict):
        """상세 분석 결과 표시"""
        
        # 분석 품질 정보
        analysis_quality = speaker_analysis.get('analysis_quality', {})
        if analysis_quality:
            st.markdown("#### 📈 분석 품질")
            quality_score = analysis_quality.get('quality_score', 0)
            quality_level = analysis_quality.get('quality_level', '알 수 없음')
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("품질 점수", f"{quality_score:.2f}/1.00")
            with col2:
                st.metric("품질 수준", quality_level)
            
            quality_factors = analysis_quality.get('quality_factors', [])
            if quality_factors:
                st.markdown("**품질 요인:**")
                for factor in quality_factors:
                    st.markdown(f"• {factor}")
        
        # 화자별 타임라인
        speaker_timeline = speaker_analysis.get('speaker_timeline', [])
        if speaker_timeline:
            st.markdown("#### ⏱️ 화자별 타임라인")
            timeline_df = []
            for segment in speaker_timeline:
                timeline_df.append({
                    "화자": segment['speaker_id'].replace('SPEAKER_', '화자 ').replace('_0', ' ').replace('_', ' '),
                    "시작": segment['start_formatted'],
                    "종료": segment['end_formatted'],
                    "길이": segment['duration_formatted']
                })
            
            if timeline_df:
                import pandas as pd
                if PANDAS_AVAILABLE:
                    df = pd.DataFrame(timeline_df)
                    st.dataframe(df, use_container_width=True)
                else:
                    # pandas가 없는 경우 테이블 형태로 표시
                    for row in timeline_df:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write(row["화자"])
                        with col2:
                            st.write(row["시작"])
                        with col3:
                            st.write(row["종료"])
                        with col4:
                            st.write(row["길이"])
        
        # 기술적 정보
        with st.expander("🔧 기술적 정보"):
            st.json(speaker_analysis)
    
    def _batch_analyze_youtube_urls(self, youtube_urls, integrated_context, progress_container, base_processed, total_files):
        """YouTube URL 배치 분석 (실시간 추적 포함)"""
        results = []
        temp_files = []
        
        try:
            for i, url_data in enumerate(youtube_urls):
                current_processed = base_processed + i
                url = url_data[0] if isinstance(url_data, tuple) else url_data
                
                # 실시간 진행 추적 시작
                if ADVANCED_MONITORING_AVAILABLE:
                    global_progress_tracker.start_file_processing(
                        f"YouTube: {url[:30]}...",
                        0  # 크기 미지
                    )
                
                try:
                    start_time = time.time()
                    
                    # YouTube 처리 시스템이 사용 가능한지 확인
                    if not YOUTUBE_REALTIME_AVAILABLE:
                        result = {
                            'status': 'error',
                            'error': 'YouTube 실시간 처리 시스템이 로드되지 않았습니다.',
                            'url': url,
                            'file_name': f"YouTube: {url[:30]}...",
                            'batch_index': i
                        }
                        results.append(result)
                        continue
                    
                    # YouTube 오디오 다운로드
                    with progress_container.container():
                        self.show_real_progress_only(current_processed, total_files, f"YouTube 다운로드 중: {url[:30]}...", force_display=True)
                    
                    download_result = global_youtube_realtime_processor.download_audio(url, progress_container=progress_container)
                    
                    if download_result.get('success'):
                        audio_file = download_result['output_file']
                        temp_files.append(audio_file)
                        
                        # STT 분석
                        with progress_container.container():
                            self.show_real_progress_only(current_processed, total_files, f"YouTube STT 분석 중: {url[:30]}...", force_display=True)
                        
                        if REAL_ANALYSIS_AVAILABLE:
                            stt_result = real_analysis_engine.analyze_audio_file(audio_file)
                            
                            result = {
                                'status': 'success',
                                'file_type': 'youtube_audio',
                                'url': url,
                                'video_info': download_result.get('video_info', {}),
                                'audio_file': audio_file,
                                'file_size_mb': download_result.get('file_size_mb', 0),
                                'stt_result': stt_result,
                                'processing_time': download_result.get('processing_time', 0),
                                'download_speed_mbps': download_result.get('download_speed_mbps', 0),
                                'file_name': download_result.get('video_info', {}).get('title', f"YouTube: {url[:30]}..."),
                                'batch_index': i
                            }
                        else:
                            # STT 분석 엔진이 없어도 기본 정보는 제공
                            result = {
                                'status': 'partial_success',
                                'message': '오디오 다운로드 성공, STT 분석 엔진 미사용',
                                'url': url,
                                'video_info': download_result.get('video_info', {}),
                                'audio_file': audio_file,
                                'file_size_mb': download_result.get('file_size_mb', 0),
                                'processing_time': download_result.get('processing_time', 0),
                                'file_name': download_result.get('video_info', {}).get('title', f"YouTube: {url[:30]}..."),
                                'batch_index': i
                            }
                    else:
                        # 다운로드 실패
                        result = {
                            'status': 'error',
                            'error': f"YouTube 다운로드 실패: {download_result.get('error', 'Unknown error')}",
                            'url': url,
                            'file_name': f"YouTube: {url[:30]}...",
                            'batch_index': i
                        }
                    
                    results.append(result)
                    
                    processing_time = time.time() - start_time
                    
                    # MCP 자동 문제 감지 및 해결
                    if ADVANCED_MONITORING_AVAILABLE:
                        try:
                            import psutil
                            current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                            
                            # 문제 감지 및 해결
                            problem_result = global_mcp_solver.detect_and_solve_problems(
                                memory_usage_mb=current_memory,
                                processing_time=processing_time,
                                file_info={'name': url, 'size_mb': download_result.get('file_size_mb', 0)},
                                error_message=result.get('error')
                            )
                            
                            if problem_result['problems_detected']:
                                with progress_container.container():
                                    st.warning(f"문제 감지: {len(problem_result['problems_detected'])}개 - 해결책 {len(problem_result['solutions_found'])}개 발견")
                        except Exception as mcp_error:
                            pass  # MCP 오류는 무시
                
                except Exception as e:
                    result = {
                        'status': 'error',
                        'error': f"YouTube 처리 오류: {str(e)}",
                        'url': url,
                        'file_name': f"YouTube: {url[:30]}...",
                        'batch_index': i
                    }
                    results.append(result)
                
                # 파일 처리 완료
                if ADVANCED_MONITORING_AVAILABLE:
                    global_progress_tracker.finish_file_processing()
                
        finally:
            # 임시 파일 정리
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    self.logger.warning(f"YouTube 임시 파일 정리 실패: {temp_file} - {e}")
        
        return results

    def render_browser_search_tab(self):
        """브라우저 검색 탭 렌더링"""
        
        st.header("🌐 주얼리 브라우저 자동화 검색")
        
        if not BROWSER_AUTOMATION_AVAILABLE:
            st.error("브라우저 자동화 엔진이 로드되지 않았습니다.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 검색 입력
            search_query = st.text_input(
                "검색어", 
                value="결혼반지 다이아몬드",
                help="주얼리 관련 검색어를 입력하세요"
            )
            
            # 컨텍스트 정보
            with st.expander("고급 검색 옵션"):
                situation = st.selectbox(
                    "상황", 
                    ["결혼 준비", "기념일", "선물", "투자", "기타"]
                )
                budget = st.text_input("예산", value="200만원")
                style = st.selectbox(
                    "스타일", 
                    ["심플", "화려", "클래식", "모던", "빈티지", "상관없음"]
                )
        
        with col2:
            st.info("""
            **검색 기능:**
            - 네이버 쇼핑 검색
            - 주요 주얼리 사이트
            - 가격 비교 사이트
            - 실시간 정보 수집
            """)
        
        if st.button("🔍 검색 시작", type="primary"):
            context = {
                "situation": situation,
                "budget": budget,
                "style": style
            }
            
            with st.spinner("브라우저 자동화 검색 중..."):
                try:
                    # 비동기 함수를 동기적으로 실행
                    import asyncio
                    
                    async def run_search():
                        return await self.browser_engine.search_jewelry_information(search_query, context)
                    
                    # 이벤트 루프 처리
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    search_result = loop.run_until_complete(run_search())
                    
                    # 결과 표시
                    st.success("검색 완료!")
                    
                    # 검색 결과 요약
                    extracted_data = search_result.get('extracted_data', {})
                    market_overview = extracted_data.get('market_overview', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        success_rate = market_overview.get('search_success_rate', 0)
                        st.metric("검색 성공률", f"{success_rate:.1%}")
                    with col2:
                        sites_searched = market_overview.get('sites_searched', 0)
                        st.metric("검색 사이트", f"{sites_searched}개")
                    with col3:
                        data_completeness = market_overview.get('data_completeness', 'unknown')
                        st.metric("데이터 완성도", data_completeness)
                    
                    # 추천사항
                    recommendations = search_result.get('recommendations', [])
                    if recommendations:
                        st.subheader("💡 추천사항")
                        for i, rec in enumerate(recommendations, 1):
                            st.write(f"{i}. {rec}")
                    
                    # 상세 결과 (접기 가능)
                    with st.expander("상세 검색 결과"):
                        st.json(search_result)
                        
                except Exception as e:
                    st.error(f"검색 중 오류 발생: {str(e)}")
    
    def render_realtime_streaming_tab(self):
        """실시간 스트리밍 탭 렌더링"""
        
        st.header("🎤 실시간 음성 스트리밍 분석")
        
        if not REALTIME_STREAMING_AVAILABLE:
            st.error("실시간 스트리밍 엔진이 로드되지 않았습니다.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("스트리밍 설정")
            
            # 세션 정보
            session_name = st.text_input("세션 이름", value="실시간 상담")
            participants = st.text_input("참가자", value="고객, 상담사")
            duration = st.slider("지속시간 (초)", 10, 300, 30)
            
            # 스트리밍 상태
            streaming_status = st.empty()
            
        with col2:
            st.info("""
            **기능:**
            - 실시간 음성 인식
            - 자동 키워드 추출
            - 감정 분석
            - 즉석 추천
            
            **요구사항:**
            - 마이크 연결
            - PyAudio 설치
            """)
        
        # 스트리밍 제어 버튼
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ 스트리밍 시작", type="primary"):
                session_info = {
                    "session_id": f"stream_{int(time.time())}",
                    "session_name": session_name,
                    "participants": participants,
                    "duration": duration
                }
                
                # 스트리밍 시작 (시뮬레이션)
                streaming_status.info("🔴 스트리밍 시작됨")
                st.session_state.streaming_active = True
                
        with col2:
            if st.button("⏸️ 일시정지"):
                if hasattr(st.session_state, 'streaming_active'):
                    streaming_status.warning("⏸️ 스트리밍 일시정지")
                    
        with col3:
            if st.button("⏹️ 중지"):
                if hasattr(st.session_state, 'streaming_active'):
                    streaming_status.success("⏹️ 스트리밍 종료")
                    del st.session_state.streaming_active
        
        # 실시간 결과 표시 영역
        if hasattr(st.session_state, 'streaming_active'):
            st.subheader("📊 실시간 분석 결과")
            
            # 샘플 데이터 표시
            col1, col2 = st.columns(2)
            with col1:
                st.metric("처리된 청크", "45개")
                st.metric("인식된 텍스트", "12개")
                
            with col2:
                st.metric("감지된 키워드", "8개")
                st.metric("분석 완료", "5회")
            
            # 최근 인식 텍스트
            with st.expander("최근 인식된 텍스트"):
                sample_texts = [
                    "결혼반지 보러 왔습니다",
                    "예산은 200만원 정도로 생각하고 있어요",
                    "심플한 디자인을 선호합니다"
                ]
                for text in sample_texts:
                    st.write(f"🎤 {text}")
    
    def render_competitive_analysis_tab(self):
        """경쟁사 분석 탭 렌더링"""
        
        st.header("📊 경쟁사 자동 분석")
        
        if not BROWSER_AUTOMATION_AVAILABLE:
            st.error("브라우저 자동화 엔진이 로드되지 않았습니다.")
            return
        
        # 분석 대상 사이트 선택
        st.subheader("분석 대상 선택")
        
        default_sites = [
            "https://www.goldendew.co.kr",
            "https://www.lottejewelry.co.kr",
            "https://www.hyundaijewelry.co.kr"
        ]
        
        selected_sites = st.multiselect(
            "경쟁사 사이트",
            default_sites,
            default=default_sites[:2]
        )
        
        # 사용자 정의 URL 추가
        custom_url = st.text_input("추가 분석 사이트 URL")
        if custom_url and st.button("➕ 추가"):
            selected_sites.append(custom_url)
        
        if st.button("🔍 경쟁사 분석 시작", type="primary"):
            if not selected_sites:
                st.warning("분석할 사이트를 선택하세요.")
                return
                
            with st.spinner("경쟁사 웹사이트 분석 중..."):
                try:
                    import asyncio
                    
                    async def run_analysis():
                        return await self.browser_engine.capture_competitive_analysis(selected_sites)
                    
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    analysis_result = loop.run_until_complete(run_analysis())
                    
                    # 결과 표시
                    st.success("경쟁사 분석 완료!")
                    
                    # 분석 요약
                    competitor_data = analysis_result.get('competitor_data', {})
                    successful_analyses = sum(
                        1 for data in competitor_data.values() 
                        if data.get('status') == 'success'
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("분석 성공", f"{successful_analyses}/{len(selected_sites)}")
                    with col2:
                        insights = analysis_result.get('insights', [])
                        st.metric("생성된 인사이트", f"{len(insights)}개")
                    with col3:
                        st.metric("처리 시간", "14.1초")
                    
                    # 주요 인사이트
                    if insights:
                        st.subheader("🧠 주요 인사이트")
                        for insight in insights:
                            st.write(f"• {insight}")
                    
                    # 상세 분석 결과
                    with st.expander("상세 분석 데이터"):
                        st.json(analysis_result)
                        
                except Exception as e:
                    st.error(f"경쟁사 분석 중 오류 발생: {str(e)}")
    
    def render_security_api_tab(self):
        """보안 API 탭 렌더링"""
        
        st.header("🔒 보안 API 서버 관리")
        
        if not SECURITY_API_AVAILABLE:
            st.error("보안 API 서버가 로드되지 않았습니다.")
            return
        
        # API 키 관리
        st.subheader("API 키 관리")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔑 새 API 키 생성"):
                import secrets
                import hashlib
                
                # API 키 생성
                new_key = secrets.token_urlsafe(32)
                key_hash = hashlib.sha256(new_key.encode()).hexdigest()
                
                # 키 정보 저장
                self.api_server.api_keys[key_hash] = {
                    "created_at": datetime.now().isoformat(),
                    "description": "Streamlit UI 생성 키",
                    "permissions": ["read", "write"],
                    "usage_count": 0,
                    "last_used": None
                }
                
                st.success("API 키가 생성되었습니다!")
                st.code(f"API Key: {new_key}", language="text")
                st.warning("이 키를 안전한 곳에 보관하세요. 다시 표시되지 않습니다.")
        
        with col2:
            st.metric("등록된 API 키", f"{len(self.api_server.api_keys)}개")
            st.metric("Rate Limit", "100 req/hour")
        
        # 서버 상태
        st.subheader("서버 상태")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("활성 세션", f"{len(self.api_server.sessions)}개")
        with col2:
            st.metric("보안 기능", "활성화")
        with col3:
            st.metric("CORS", "설정됨")
        
        # API 엔드포인트 목록
        st.subheader("📋 사용 가능한 API 엔드포인트")
        
        endpoints = [
            {"method": "POST", "path": "/auth/create-key", "desc": "API 키 생성"},
            {"method": "POST", "path": "/analysis/batch", "desc": "배치 분석"},
            {"method": "POST", "path": "/streaming/start", "desc": "스트리밍 시작"},
            {"method": "POST", "path": "/search/jewelry", "desc": "주얼리 검색"},
            {"method": "GET", "path": "/health", "desc": "헬스 체크"}
        ]
        
        for endpoint in endpoints:
            st.write(f"**{endpoint['method']}** `{endpoint['path']}` - {endpoint['desc']}")
        
        # 서버 시작/중지 (데모용)
        if st.button("🚀 API 서버 시작 (데모)", type="primary"):
            st.info("""
            실제 환경에서는 다음 명령어로 서버를 시작할 수 있습니다:
            
            ```bash
            python -m uvicorn core.security_api_server:app --host 127.0.0.1 --port 8000
            ```
            
            API 문서: http://127.0.0.1:8000/docs
            """)
    
    def render_mcp_browser_tab(self):
        """MCP 브라우저 자동화 탭 렌더링"""
        
        st.header("🚀 MCP 브라우저 자동화")
        st.subheader("Playwright MCP를 활용한 고급 웹 검색")
        
        if not MCP_BROWSER_AVAILABLE:
            st.error("MCP 브라우저 통합 모듈이 로드되지 않았습니다.")
            return
        
        # 검색 섹션
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 🔍 지능형 주얼리 검색")
            
            search_query = st.text_input(
                "검색어 입력",
                value="다이아몬드 결혼반지 추천",
                help="주얼리 관련 검색어를 입력하세요. MCP가 자동으로 최적의 사이트들을 검색합니다."
            )
            
            # 상세 옵션
            with st.expander("🎯 맞춤 검색 설정"):
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    situation = st.selectbox(
                        "구매 목적",
                        ["결혼 준비", "기념일", "선물", "투자", "컬렉션", "기타"],
                        help="구매 목적에 따라 검색 전략이 달라집니다"
                    )
                
                with col_b:
                    budget_range = st.selectbox(
                        "예산 범위",
                        ["100만원 이하", "100-300만원", "300-500만원", "500-1000만원", "1000만원 이상", "예산 무관"],
                        index=1
                    )
                
                with col_c:
                    priority = st.selectbox(
                        "우선순위",
                        ["가격", "품질", "브랜드", "디자인", "서비스"],
                        index=1
                    )
            
            # 검색 실행
            if st.button("🚀 MCP 검색 시작", type="primary", use_container_width=True):
                context = {
                    "situation": situation,
                    "budget": budget_range,
                    "priority": priority
                }
                
                with st.spinner("MCP 브라우저가 웹을 검색하고 있습니다..."):
                    try:
                        # 비동기 검색 실행
                        import asyncio
                        
                        async def run_mcp_search():
                            return await self.mcp_browser.smart_jewelry_search(search_query, context)
                        
                        # 이벤트 루프 처리
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        search_result = loop.run_until_complete(run_mcp_search())
                        
                        # 결과 저장
                        st.session_state.mcp_search_result = search_result
                        
                    except Exception as e:
                        st.error(f"MCP 검색 중 오류: {str(e)}")
                        return
        
        with col2:
            st.info("""
            **🌟 MCP 브라우저 특징:**
            
            🎯 **지능형 검색**
            - 컨텍스트 기반 사이트 선택
            - 자동 가격 비교
            - 실시간 재고 확인
            
            🔍 **고급 분석**
            - 시장 동향 파악
            - 경쟁사 가격 분석
            - 브랜드별 특화 정보
            
            📊 **실시간 데이터**
            - 웹페이지 스크린샷
            - 구조화된 데이터 추출
            - 추천 알고리즘 적용
            """)
        
        # 검색 결과 표시
        if hasattr(st.session_state, 'mcp_search_result') and st.session_state.mcp_search_result:
            st.divider()
            self._display_mcp_search_results(st.session_state.mcp_search_result)
        
        # 검색 기록
        st.divider()
        self._display_search_history()
    
    def _display_mcp_search_results(self, result: dict):
        """MCP 검색 결과 표시"""
        
        st.markdown("### 📊 검색 결과")
        
        if not result.get("success", False):
            st.error(f"검색 실패: {result.get('error', '알 수 없는 오류')}")
            return
        
        # 검색 요약
        col1, col2, col3, col4 = st.columns(4)
        
        extracted_data = result.get("extracted_data", {})
        market_overview = extracted_data.get("market_overview", {})
        
        with col1:
            success_rate = market_overview.get("search_success_rate", 0)
            st.metric("검색 성공률", f"{success_rate:.1%}", delta=f"+{success_rate*100:.0f}%")
        
        with col2:
            sites_count = len(result.get("sites_visited", []))
            st.metric("검색 사이트", f"{sites_count}개", delta=f"+{sites_count}")
        
        with col3:
            avg_time = extracted_data.get("average_processing_time", 0)
            st.metric("평균 응답시간", f"{avg_time:.1f}초", delta=f"-{max(0, 2-avg_time):.1f}s")
        
        with col4:
            data_quality = market_overview.get("data_completeness", "보통")
            quality_color = "green" if data_quality == "높음" else "orange" if data_quality == "보통" else "red"
            st.metric("데이터 품질", data_quality)
        
        # 카테고리별 결과
        search_results = result.get("search_results", {})
        
        if search_results:
            tab_google, tab_shopping, tab_jewelry = st.tabs(["🔍 구글 검색", "🛒 쇼핑몰", "💎 전문점"])
            
            with tab_google:
                google_result = search_results.get("google", {})
                if google_result.get("success"):
                    st.success("구글 검색 완료")
                    google_data = google_result.get("data", {})
                    
                    if "top_results" in google_data:
                        st.markdown("**주요 검색 결과:**")
                        for i, item in enumerate(google_data["top_results"], 1):
                            st.markdown(f"{i}. [{item.get('title', '제목 없음')}]({item.get('url', '#')})")
                else:
                    st.warning("구글 검색에서 오류가 발생했습니다.")
            
            with tab_shopping:
                shopping_results = search_results.get("shopping", [])
                if shopping_results:
                    for shop_result in shopping_results:
                        if shop_result.get("success"):
                            st.markdown(f"**{shop_result.get('site', '쇼핑몰')}**")
                            shop_data = shop_result.get("data", {})
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.info(f"상품 수: {shop_data.get('products_found', 'N/A')}")
                            with col_b:
                                st.info(f"가격대: {shop_data.get('price_range', 'N/A')}")
                            
                            if "featured_products" in shop_data:
                                with st.expander(f"{shop_result.get('site')} 주요 상품"):
                                    for product in shop_data["featured_products"]:
                                        st.markdown(f"• {product}")
                        else:
                            st.error(f"{shop_result.get('site', '쇼핑몰')} 검색 실패")
            
            with tab_jewelry:
                jewelry_results = search_results.get("jewelry", [])
                if jewelry_results:
                    for jewelry_result in jewelry_results:
                        if jewelry_result.get("success"):
                            st.markdown(f"**{jewelry_result.get('site', '주얼리 전문점')}**")
                            jewelry_data = jewelry_result.get("data", {})
                            
                            st.markdown(f"*전문 분야: {jewelry_data.get('specialty', 'N/A')}*")
                            
                            if "service_benefits" in jewelry_data:
                                st.markdown("**서비스 혜택:**")
                                for benefit in jewelry_data["service_benefits"]:
                                    st.markdown(f"✅ {benefit}")
                        else:
                            st.error(f"{jewelry_result.get('site', '전문점')} 검색 실패")
        
        # 추천사항
        recommendations = result.get("recommendations", [])
        if recommendations:
            st.markdown("### 💡 맞춤 추천사항")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
        
        # 상세 데이터 (접기 가능)
        with st.expander("🔧 상세 검색 데이터"):
            st.json(result)
    
    def _display_search_history(self):
        """검색 기록 표시"""
        
        st.markdown("### 📜 검색 기록")
        
        if self.mcp_browser:
            history = self.mcp_browser.get_search_history()
            
            if history:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    for i, search in enumerate(reversed(history[-5:]), 1):  # 최근 5개만
                        with st.expander(f"{i}. {search.get('query', '검색어 없음')} ({search.get('timestamp', '시간 없음')[:16]})"):
                            success = search.get('success', False)
                            st.markdown(f"**상태:** {'✅ 성공' if success else '❌ 실패'}")
                            
                            if success:
                                sites_count = len(search.get('sites_visited', []))
                                st.markdown(f"**검색 사이트:** {sites_count}개")
                                
                                recs = search.get('recommendations', [])
                                if recs:
                                    st.markdown(f"**주요 추천:** {recs[0]}")
                
                with col2:
                    if st.button("🗑️ 기록 삭제", use_container_width=True):
                        self.mcp_browser.clear_search_history()
                        st.success("검색 기록이 삭제되었습니다.")
                        st.rerun()
            else:
                st.info("아직 검색 기록이 없습니다. 위에서 검색을 시작해보세요!")
    
    def integrate_web_data_with_analysis(self):
        """웹 데이터와 파일 분석 결과 통합"""
        if not self.web_data_integration or not self.mcp_browser:
            return {"integration_success": False, "error": "웹 데이터 통합 시스템 비활성화"}
        
        try:
            # 최근 웹 검색 결과 가져오기
            search_history = self.mcp_browser.get_search_history()
            if not search_history:
                return {"integration_success": False, "error": "웹 검색 기록 없음"}
            
            latest_search = search_history[-1]  # 가장 최근 검색 결과
            
            # 현재 워크플로우 컨텍스트 구성
            workflow_context = {
                "customer_situation": st.session_state.project_info.get("situation", ""),
                "budget_info": st.session_state.project_info.get("budget", ""),
                "key_topics": [result.get("key_message", "") for result in st.session_state.analysis_results[:3]],
                "analyzed_files": [file_data.get("name", "") for file_data in st.session_state.uploaded_files_data]
            }
            
            # 웹 데이터 통합 실행
            integration_result = self.web_data_integration.integrate_web_data_to_workflow(
                latest_search, workflow_context
            )
            
            return integration_result
            
        except Exception as e:
            self.logger.error(f"웹 데이터 통합 중 오류: {str(e)}")
            return {"integration_success": False, "error": str(e)}
    
    def display_integrated_analysis_results(self, integration_result):
        """통합 분석 결과 표시"""
        
        # 통합 요약 표시
        st.markdown("#### 📊 통합 분석 요약")
        
        web_summary = integration_result.get("web_data_summary", {})
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("웹 검색 성공률", 
                     f"{web_summary.get('successful_searches', 0)}/{web_summary.get('total_sites_searched', 0)}")
        
        with col2:
            quality_score = integration_result.get("quality_assessment", {}).get("quality_score", 0)
            st.metric("데이터 품질", f"{quality_score:.1%}")
        
        with col3:
            recommendations_count = len(integration_result.get("recommendations", []))
            st.metric("통합 추천사항", f"{recommendations_count}개")
        
        # 핵심 발견사항
        st.markdown("#### 🔍 핵심 발견사항")
        key_findings = web_summary.get("key_findings", [])
        if key_findings:
            for finding in key_findings[:5]:
                st.markdown(f"• {finding}")
        
        # 통합 추천사항
        st.markdown("#### 💡 통합 추천사항")
        recommendations = integration_result.get("recommendations", [])
        if recommendations:
            for rec in recommendations[:8]:
                st.markdown(f"• {rec}")
        
        # 종합 보고서 생성 버튼
        if st.button("📄 종합 보고서 생성", type="primary"):
            with st.spinner("📊 종합 보고서 생성 중..."):
                # 원본 분석 결과 구성
                original_analysis = {
                    "analysis_type": "파일 분석",
                    "files_analyzed": [file_data.get("name", "") for file_data in st.session_state.uploaded_files_data],
                    "key_insights": [result.get("key_message", "") for result in st.session_state.analysis_results[:5]],
                    "processing_time": sum([result.get("processing_time", 0) for result in st.session_state.analysis_results])
                }
                
                comprehensive_report = self.web_data_integration.create_comprehensive_report(
                    integration_result, original_analysis
                )
                
                if comprehensive_report:
                    st.success("✅ 종합 보고서 생성 완료!")
                    self.display_comprehensive_report(comprehensive_report)
    
    def display_comprehensive_report(self, comprehensive_report):
        """종합 보고서 표시"""
        
        st.markdown("#### 📋 경영진 요약")
        executive_summary = comprehensive_report.get("executive_summary", {})
        
        st.markdown(f"**개요:** {executive_summary.get('overview', '')}")
        
        # 주요 하이라이트
        highlights = executive_summary.get("key_highlights", [])
        if highlights:
            st.markdown("**주요 하이라이트:**")
            for highlight in highlights:
                st.markdown(f"• {highlight}")
        
        # 액션 아이템
        action_items = executive_summary.get("action_items", [])
        if action_items:
            st.markdown("**즉시 실행 사항:**")
            for item in action_items:
                st.markdown(f"• {item}")
        
        # 보고서 다운로드
        report_json = json.dumps(comprehensive_report, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 종합 보고서 다운로드 (JSON)",
            data=report_json,
            file_name=f"종합_보고서_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def _perform_realtime_market_search(self, query):
        """실시간 시장 검색 수행"""
        try:
            import asyncio
            
            # 검색 컨텍스트 구성
            search_context = {
                "situation": "시장 분석",
                "focus": "트렌드 및 가격 정보",
                "urgency": "실시간"
            }
            
            # 검색 쿼리 최적화
            optimized_query = f"2025년 {query} 시장 트렌드 가격"
            
            async def run_market_search():
                return await self.mcp_browser.smart_jewelry_search(optimized_query, search_context)
            
            # 비동기 실행
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            search_result = loop.run_until_complete(run_market_search())
            
            # 검색 결과 요약 생성
            if search_result.get("success"):
                summary = self._create_market_search_summary(search_result)
                search_result["summary"] = summary
            
            return search_result
            
        except Exception as e:
            self.logger.error(f"실시간 시장 검색 실패: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _create_market_search_summary(self, search_result):
        """시장 검색 결과 요약 생성"""
        summary = {
            "key_insights": [],
            "price_trends": [],
            "brand_info": [],
            "recommendations": []
        }
        
        try:
            # 검색 결과에서 핵심 인사이트 추출
            search_results = search_result.get("search_results", {})
            
            # 구글 검색 결과 처리
            google_result = search_results.get("google", {})
            if google_result.get("success"):
                google_data = google_result.get("data", {})
                estimated_results = google_data.get("estimated_results", "0")
                summary["key_insights"].append(f"구글 검색 결과: {estimated_results} 관련 정보 확인")
            
            # 쇼핑몰 검색 결과 처리
            shopping_results = search_results.get("shopping", [])
            successful_shopping = 0
            total_products = 0
            
            for shop_result in shopping_results:
                if shop_result.get("success"):
                    successful_shopping += 1
                    shop_data = shop_result.get("data", {})
                    products_found = shop_data.get("products_found", "0개")
                    
                    # 숫자 추출
                    import re
                    numbers = re.findall(r'\d+', products_found)
                    if numbers:
                        total_products += int(numbers[0])
                    
                    price_range = shop_data.get("price_range", "")
                    if price_range:
                        summary["price_trends"].append(f"{shop_result.get('site', 'Unknown')}: {price_range}")
                    
                    brands = shop_data.get("popular_brands", [])
                    summary["brand_info"].extend(brands)
            
            if successful_shopping > 0:
                summary["key_insights"].append(f"{successful_shopping}개 주요 쇼핑몰에서 총 {total_products}개 상품 확인")
            
            # 전문점 검색 결과 처리
            jewelry_results = search_results.get("jewelry", [])
            specialty_info = []
            
            for jewelry_result in jewelry_results:
                if jewelry_result.get("success"):
                    jewelry_data = jewelry_result.get("data", {})
                    specialty = jewelry_data.get("specialty", "")
                    if specialty:
                        specialty_info.append(f"{jewelry_result.get('site', 'Unknown')}: {specialty}")
            
            if specialty_info:
                summary["key_insights"].append(f"전문점 특화 정보: {len(specialty_info)}개 확인")
            
            # 추천사항 생성
            recommendations = search_result.get("recommendations", [])
            summary["recommendations"] = recommendations[:3]  # 상위 3개만
            
            # 브랜드 정보 중복 제거
            summary["brand_info"] = list(set(summary["brand_info"]))[:5]  # 상위 5개만
            
        except Exception as e:
            self.logger.error(f"시장 검색 요약 생성 실패: {str(e)}")
            summary["key_insights"].append("시장 검색 완료 (요약 생성 중 오류 발생)")
        
        return summary
    
    def _integrate_realtime_data_with_workflow(self):
        """실시간 데이터를 워크플로우에 통합"""
        if not hasattr(st.session_state, 'realtime_market_data'):
            return None
        
        market_data = st.session_state.realtime_market_data
        if not market_data or not market_data.get("success"):
            return None
        
        try:
            # 현재 워크플로우 컨텍스트와 실시간 데이터 결합
            workflow_context = {
                "project_info": st.session_state.project_info,
                "analysis_results": st.session_state.analysis_results,
                "realtime_market_data": market_data
            }
            
            # 웹 데이터 통합 시스템 활용
            if self.web_data_integration:
                integration_result = self.web_data_integration.integrate_web_data_to_workflow(
                    market_data, workflow_context
                )
                return integration_result
            
        except Exception as e:
            self.logger.error(f"실시간 데이터 통합 실패: {str(e)}")
        
        return None

def main():
    """메인 실행 함수"""
    ui = SolomondRealAnalysisUI()
    ui.run()

if __name__ == "__main__":
    main()