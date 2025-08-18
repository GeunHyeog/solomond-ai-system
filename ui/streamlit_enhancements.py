"""
🎨 Streamlit UI 향상 모듈
솔로몬드 AI 시스템 - 사용자 경험 개선

목적: Streamlit UI의 시각적 완성도와 사용성 향상
기능: 커스텀 CSS, 향상된 메시지, 진행률 표시, 성능 최적화
"""

import streamlit as st
import time
import platform
import psutil
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

class StreamlitEnhancer:
    """Streamlit UI 향상 클래스"""
    
    def __init__(self):
        self.theme_colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#ff7f0e',
            'error': '#d62728',
            'info': '#17a2b8',
            'gradient_start': '#667eea',
            'gradient_end': '#764ba2'
        }
    
    def setup_custom_css(self):
        """커스텀 CSS 스타일 적용"""
        
        css = f"""
        <style>
        /* 메인 컨테이너 그라데이션 */
        .main .block-container {{
            background: linear-gradient(135deg, {self.theme_colors['gradient_start']} 0%, {self.theme_colors['gradient_end']} 100%);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-top: 1rem;
        }}
        
        /* 워크플로우 진행바 */
        .workflow-progress {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 2rem 0;
            padding: 1rem;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }}
        
        .workflow-step {{
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 1rem;
            border-radius: 8px;
            transition: all 0.3s ease;
            min-width: 150px;
        }}
        
        .workflow-step.active {{
            background: rgba(255,255,255,0.2);
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        
        .workflow-step.completed {{
            background: rgba(46,160,44,0.2);
            color: #2ca02c;
        }}
        
        .workflow-icon {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}
        
        .workflow-title {{
            font-weight: bold;
            font-size: 0.9rem;
            text-align: center;
        }}
        
        /* 향상된 메시지 박스 */
        .enhanced-message {{
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid;
            animation: slideIn 0.5s ease-out;
            backdrop-filter: blur(5px);
        }}
        
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(-20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .enhanced-message.error {{
            background: rgba(214, 39, 40, 0.1);
            border-left-color: #d62728;
            color: #d62728;
        }}
        
        .enhanced-message.success {{
            background: rgba(46, 160, 44, 0.1);
            border-left-color: #2ca02c;
            color: #2ca02c;
        }}
        
        .enhanced-message.warning {{
            background: rgba(255, 127, 14, 0.1);
            border-left-color: #ff7f0e;
            color: #ff7f0e;
        }}
        
        .enhanced-message.info {{
            background: rgba(23, 162, 184, 0.1);
            border-left-color: #17a2b8;
            color: #17a2b8;
        }}
        
        .message-title {{
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }}
        
        .message-solutions {{
            margin-top: 1rem;
            padding: 1rem;
            background: rgba(255,255,255,0.1);
            border-radius: 5px;
        }}
        
        .message-solutions h4 {{
            margin-bottom: 0.5rem;
            color: inherit;
        }}
        
        .message-solutions ul {{
            margin: 0;
            padding-left: 1.5rem;
        }}
        
        /* 진행률 표시 */
        .progress-container {{
            margin: 1rem 0;
            padding: 1rem;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
        }}
        
        .progress-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #2ca02c, #4caf50);
            border-radius: 10px;
            transition: width 0.5s ease;
            position: relative;
        }}
        
        .progress-fill::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            right: 0;
            background-image: linear-gradient(
                -45deg,
                rgba(255, 255, 255, .2) 25%,
                transparent 25%,
                transparent 50%,
                rgba(255, 255, 255, .2) 50%,
                rgba(255, 255, 255, .2) 75%,
                transparent 75%,
                transparent
            );
            background-size: 50px 50px;
            animation: move 2s linear infinite;
        }}
        
        @keyframes move {{
            0% {{ background-position: 0 0; }}
            100% {{ background-position: 50px 50px; }}
        }}
        
        /* 정보 카드 */
        .info-card {{
            background: rgba(255,255,255,0.1);
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            transition: all 0.3s ease;
            backdrop-filter: blur(5px);
        }}
        
        .info-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }}
        
        .info-icon {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
            display: block;
        }}
        
        .info-card h4 {{
            margin: 0.5rem 0;
            color: white;
            font-size: 1rem;
        }}
        
        .info-card p {{
            margin: 0;
            color: rgba(255,255,255,0.8);
            font-size: 0.9rem;
        }}
        
        /* 파일 업로드 영역 */
        .upload-area {{
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
            background: rgba(255,255,255,0.05);
        }}
        
        .upload-area:hover {{
            border-color: rgba(255,255,255,0.6);
            background: rgba(255,255,255,0.1);
        }}
        
        /* 모바일 반응형 */
        @media (max-width: 768px) {{
            .main .block-container {{
                padding: 1rem;
                margin-top: 0.5rem;
            }}
            
            .workflow-progress {{
                flex-direction: column;
                gap: 1rem;
            }}
            
            .workflow-step {{
                width: 100%;
                max-width: none;
            }}
            
            .info-card {{
                padding: 1rem;
            }}
        }}
        
        /* 다크 모드 지원 */
        @media (prefers-color-scheme: dark) {{
            .main .block-container {{
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            }}
        }}
        
        /* 접근성 개선 */
        .enhanced-message:focus,
        .workflow-step:focus,
        .info-card:focus {{
            outline: 2px solid #fff;
            outline-offset: 2px;
        }}
        
        /* 애니메이션 비활성화 옵션 */
        @media (prefers-reduced-motion: reduce) {{
            * {{
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }}
        }}
        </style>
        """
        
        st.markdown(css, unsafe_allow_html=True)
    
    def show_enhanced_message(self, message_type: str, title: str, content: str, solutions: List[str] = None):
        """향상된 메시지 표시
        
        Args:
            message_type: 'error', 'success', 'warning', 'info'
            title: 메시지 제목
            content: 메시지 내용
            solutions: 해결방안 목록 (선택적)
        """
        
        icon_map = {
            'error': '❌',
            'success': '✅', 
            'warning': '⚠️',
            'info': 'ℹ️'
        }
        
        icon = icon_map.get(message_type, 'ℹ️')
        solutions_html = ""
        
        if solutions:
            solutions_list = "\n".join([f"<li>{solution}</li>" for solution in solutions[:3]])
            solutions_html = f"""
            <div class="message-solutions">
                <h4>💡 해결방안:</h4>
                <ul>{solutions_list}</ul>
            </div>
            """
        
        message_html = f"""
        <div class="enhanced-message {message_type}">
            <div class="message-title">{icon} {title}</div>
            <div>{content}</div>
            {solutions_html}
        </div>
        """
        
        st.markdown(message_html, unsafe_allow_html=True)
    
    def show_progress_with_details(self, current: int, total: int, message: str, details: str = ""):
        """상세 진행률 표시
        
        Args:
            current: 현재 진행 수
            total: 전체 작업 수
            message: 진행 메시지
            details: 상세 설명
        """
        
        if total == 0:
            percentage = 0
        else:
            percentage = (current / total) * 100
        
        progress_html = f"""
        <div class="progress-container">
            <div class="progress-header">
                <strong>{message}</strong>
                <span>{current}/{total} ({percentage:.1f}%)</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {percentage}%"></div>
            </div>
            {f'<div style="margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.8;">{details}</div>' if details else ''}
        </div>
        """
        
        st.markdown(progress_html, unsafe_allow_html=True)
    
    def display_workflow_progress(self, current_step: int):
        """워크플로우 진행 상태 표시
        
        Args:
            current_step: 현재 단계 (1-4)
        """
        
        steps = [
            {"icon": "📝", "title": "기본정보", "step": 1},
            {"icon": "📁", "title": "파일업로드", "step": 2}, 
            {"icon": "🔍", "title": "검토", "step": 3},
            {"icon": "📊", "title": "보고서", "step": 4}
        ]
        
        steps_html = []
        for step in steps:
            if step["step"] < current_step:
                step_class = "workflow-step completed"
            elif step["step"] == current_step:
                step_class = "workflow-step active"
            else:
                step_class = "workflow-step"
            
            steps_html.append(f"""
            <div class="{step_class}">
                <div class="workflow-icon">{step["icon"]}</div>
                <div class="workflow-title">{step["title"]}</div>
            </div>
            """)
        
        workflow_html = f"""
        <div class="workflow-progress">
            {"".join(steps_html)}
        </div>
        """
        
        st.markdown(workflow_html, unsafe_allow_html=True)
    
    @st.cache_data(ttl=300)  # 5분 캐싱
    def _get_system_info(_self) -> Dict[str, str]:
        """시스템 정보 수집 (캐싱됨)"""
        
        try:
            memory_info = psutil.virtual_memory()
            memory_used = memory_info.used / (1024**3)
            memory_total = memory_info.total / (1024**3)
            
            return {
                'os': f"{platform.system()} {platform.release()}",
                'python': platform.python_version(),
                'memory': f"{memory_used:.1f}GB / {memory_total:.1f}GB"
            }
        except Exception as e:
            return {
                'os': "정보 없음",
                'python': "정보 없음", 
                'memory': "정보 없음"
            }
    
    def display_system_info(self):
        """향상된 시스템 정보 표시"""
        
        system_info = self._get_system_info()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-icon">🖥️</div>
                <div class="info-content">
                    <h4>시스템</h4>
                    <p>{system_info['os']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-icon">🐍</div>
                <div class="info-content">
                    <h4>Python</h4>
                    <p>{system_info['python']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-icon">💾</div>
                <div class="info-content">
                    <h4>메모리</h4>
                    <p>{system_info['memory']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def should_update_component(self, component_id: str, update_interval: int = 5) -> bool:
        """컴포넌트 업데이트 필요 여부 확인 (성능 최적화)
        
        Args:
            component_id: 컴포넌트 식별자
            update_interval: 업데이트 간격 (초)
            
        Returns:
            bool: 업데이트 필요 여부
        """
        
        current_time = time.time()
        last_update_key = f"last_update_{component_id}"
        
        if last_update_key not in st.session_state:
            st.session_state[last_update_key] = current_time
            return True
        
        if current_time - st.session_state[last_update_key] >= update_interval:
            st.session_state[last_update_key] = current_time
            return True
        
        return False
    
    def create_download_button(self, data: Any, filename: str, button_text: str = "다운로드", 
                             file_format: str = "json") -> bool:
        """향상된 다운로드 버튼
        
        Args:
            data: 다운로드할 데이터
            filename: 파일명
            button_text: 버튼 텍스트
            file_format: 파일 형식 ('json', 'csv', 'txt')
            
        Returns:
            bool: 다운로드 버튼 클릭 여부
        """
        
        try:
            if file_format == "json":
                file_data = json.dumps(data, ensure_ascii=False, indent=2)
                mime_type = "application/json"
            elif file_format == "csv":
                # CSV 변환 로직 (간단한 구현)
                file_data = str(data)
                mime_type = "text/csv"
            else:  # txt
                file_data = str(data)
                mime_type = "text/plain"
            
            return st.download_button(
                label=f"📥 {button_text}",
                data=file_data,
                file_name=filename,
                mime=mime_type,
                help=f"{file_format.upper()} 형식으로 다운로드"
            )
            
        except Exception as e:
            self.show_enhanced_message(
                "error",
                "다운로드 오류",
                f"파일 다운로드 중 오류가 발생했습니다: {str(e)}",
                ["파일 형식을 확인해주세요", "데이터가 올바른지 확인해주세요", "잠시 후 다시 시도해주세요"]
            )
            return False
    
    def create_file_upload_area(self, accepted_types: List[str], max_size: int = 200) -> Any:
        """향상된 파일 업로드 영역
        
        Args:
            accepted_types: 허용되는 파일 타입 목록
            max_size: 최대 파일 크기 (MB)
            
        Returns:
            업로드된 파일 객체
        """
        
        types_text = ", ".join(accepted_types)
        
        st.markdown(f"""
        <div class="upload-area">
            <h3>📁 파일 업로드</h3>
            <p>지원 형식: {types_text}</p>
            <p>최대 크기: {max_size}MB</p>
            <p>파일을 드래그하여 업로드하거나 아래 버튼을 클릭하세요</p>
        </div>
        """, unsafe_allow_html=True)
        
        return st.file_uploader(
            "파일 선택",
            type=accepted_types,
            accept_multiple_files=True,
            help=f"최대 {max_size}MB까지 업로드 가능합니다"
        )

# 전역 인스턴스
_enhancer = None

def get_streamlit_enhancer() -> StreamlitEnhancer:
    """전역 StreamlitEnhancer 인스턴스 반환"""
    global _enhancer
    if _enhancer is None:
        _enhancer = StreamlitEnhancer()
    return _enhancer

# 편의 함수들
def setup_enhanced_ui():
    """향상된 UI 설정 (메인 진입점)"""
    enhancer = get_streamlit_enhancer()
    enhancer.setup_custom_css()
    return enhancer

def show_message(message_type: str, title: str, content: str, solutions: List[str] = None):
    """메시지 표시 편의 함수"""
    enhancer = get_streamlit_enhancer()
    enhancer.show_enhanced_message(message_type, title, content, solutions)

def show_progress(current: int, total: int, message: str, details: str = ""):
    """진행률 표시 편의 함수"""
    enhancer = get_streamlit_enhancer()
    enhancer.show_progress_with_details(current, total, message, details)

def show_workflow(current_step: int):
    """워크플로우 표시 편의 함수"""
    enhancer = get_streamlit_enhancer()
    enhancer.display_workflow_progress(current_step)

# 사용 예시 (주석으로 남겨둠)
"""
사용 예시:

# 기본 설정
from ui.streamlit_enhancements import setup_enhanced_ui, show_message, show_progress

# UI 초기화
enhancer = setup_enhanced_ui()

# 메시지 표시
show_message("success", "성공", "작업이 완료되었습니다")
show_message("error", "오류", "파일을 찾을 수 없습니다", 
             ["파일 경로를 확인하세요", "파일 권한을 확인하세요"])

# 진행률 표시
show_progress(3, 10, "파일 처리 중", "현재 image_001.jpg 처리 중")

# 워크플로우 표시
show_workflow(2)  # 2단계 활성화
"""