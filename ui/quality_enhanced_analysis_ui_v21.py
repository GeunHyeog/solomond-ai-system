"""
🎨 Solomond AI v2.1 - 품질 강화 통합 분석 UI
품질 모니터링, 다국어 처리, 다중파일 통합 분석을 위한 고급 인터페이스

Author: 전근혁 (Solomond)
Created: 2025.07.11
Version: 2.1.0
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import asyncio
from pathlib import Path
import base64
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# v2.1 신규 모듈 import
import sys
sys.path.append('core')

try:
    from quality_analyzer_v21 import QualityAnalyzerV21, QualityScore
    from multilingual_processor_v21 import MultilingualProcessorV21, MultilingualSTTResult
    from multi_file_integrator_v21 import MultiFileIntegratorV21, IntegratedSession
    from korean_summary_engine_v21 import KoreanSummaryEngineV21, KoreanAnalysisResult, SummaryStyle
except ImportError as e:
    st.error(f"v2.1 모듈 import 실패: {e}")
    st.stop()

# 페이지 설정
st.set_page_config(
    page_title="🏆 솔로몬드 AI v2.1 - 품질 강화 플랫폼",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 15px 15px;
    }
    
    .quality-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .error-card {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .quality-gauge {
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
    }
    
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'v21_processor' not in st.session_state:
    st.session_state.v21_processor = {
        'quality_analyzer': QualityAnalyzerV21(),
        'multilingual_processor': MultilingualProcessorV21(),
        'file_integrator': MultiFileIntegratorV21(),
        'korean_engine': KoreanSummaryEngineV21(),
        'uploaded_files': [],
        'analysis_results': {},
        'processing_status': 'ready'
    }

def render_header():
    """메인 헤더 렌더링"""
    st.markdown("""
    <div class="main-header">
        <h1>🏆 솔로몬드 AI v2.1</h1>
        <h3>💎 주얼리 업계 품질 강화 분석 플랫폼</h3>
        <p>다국어 입력 • 품질 검증 • 통합 분석 • 한국어 요약</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.markdown("## 🎛️ 분석 설정")
        
        # 분석 모드 선택
        analysis_mode = st.selectbox(
            "📊 분석 모드",
            ["종합 분석", "품질 중심", "다국어 중심", "비즈니스 중심", "기술 중심"],
            help="분석의 주요 초점을 선택하세요"
        )
        
        # 품질 임계값 설정
        st.markdown("### 🔍 품질 기준")
        min_audio_quality = st.slider("최소 음성 품질", 0, 100, 70, help="음성 파일 최소 품질 점수")
        min_ocr_quality = st.slider("최소 OCR 품질", 0, 100, 80, help="이미지/문서 OCR 최소 품질")
        min_confidence = st.slider("최소 신뢰도", 0.0, 1.0, 0.7, help="분석 결과 최소 신뢰도")
        
        # 언어 설정
        st.markdown("### 🌍 언어 설정")
        target_languages = st.multiselect(
            "지원 언어",
            ["한국어", "English", "中文", "日本語"],
            default=["한국어", "English"],
            help="처리할 언어를 선택하세요"
        )
        
        # 출력 설정
        st.markdown("### 📄 출력 설정")
        summary_style = st.selectbox(
            "요약 스타일",
            ["종합", "경영진", "기술적", "비즈니스"],
            help="생성할 요약의 스타일을 선택하세요"
        )
        
        include_insights = st.checkbox("인사이트 포함", True)
        include_actions = st.checkbox("액션 아이템 포함", True)
        include_quality_report = st.checkbox("품질 리포트 포함", True)
        
        return {
            'analysis_mode': analysis_mode,
            'min_audio_quality': min_audio_quality,
            'min_ocr_quality': min_ocr_quality,
            'min_confidence': min_confidence,
            'target_languages': target_languages,
            'summary_style': summary_style.lower(),
            'include_insights': include_insights,
            'include_actions': include_actions,
            'include_quality_report': include_quality_report
        }

def render_file_upload():
    """파일 업로드 인터페이스"""
    st.markdown("## 📁 파일 업로드")
    
    # 지원 파일 형식 안내
    with st.expander("💡 지원 파일 형식 및 품질 가이드", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🎙️ 음성 파일**
            - MP3, WAV, M4A, AAC
            - 권장: SNR > 20dB
            - 최대 크기: 500MB
            """)
        
        with col2:
            st.markdown("""
            **📸 이미지 파일**
            - JPG, PNG, GIF, BMP
            - 권장: 1920x1080 이상
            - 텍스트는 수평으로 촬영
            """)
        
        with col3:
            st.markdown("""
            **📄 문서 파일**
            - PDF, DOCX, TXT
            - OCR 품질 자동 검증
            - 다국어 자동 감지
            """)
    
    # 파일 업로드
    uploaded_files = st.file_uploader(
        "파일을 드래그 앤 드롭하거나 선택하세요",
        type=['mp3', 'wav', 'm4a', 'aac', 'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="다중 파일 선택 가능 (최대 50개)"
    )
    
    if uploaded_files:
        st.session_state.v21_processor['uploaded_files'] = uploaded_files
        
        # 파일 정보 표시
        st.markdown("### 📋 업로드된 파일")
        
        file_data = []
        total_size = 0
        
        for i, file in enumerate(uploaded_files):
            file_size = len(file.getvalue())
            total_size += file_size
            
            file_type = "audio" if file.type.startswith('audio') else \
                       "video" if file.type.startswith('video') else \
                       "image" if file.type.startswith('image') else "document"
            
            file_data.append({
                "순번": i + 1,
                "파일명": file.name,
                "타입": file_type,
                "크기": f"{file_size / (1024*1024):.1f} MB",
                "상태": "✅ 준비됨"
            })
        
        # 파일 목록 테이블
        df = pd.DataFrame(file_data)
        st.dataframe(df, use_container_width=True)
        
        # 요약 정보
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("전체 파일", len(uploaded_files))
        with col2:
            st.metric("총 크기", f"{total_size / (1024*1024):.1f} MB")
        with col3:
            audio_count = sum(1 for f in uploaded_files if f.type.startswith('audio'))
            st.metric("음성 파일", audio_count)
        with col4:
            image_count = sum(1 for f in uploaded_files if f.type.startswith('image'))
            st.metric("이미지 파일", image_count)
        
        return True
    
    return False

def render_quality_monitor(files_analysis: Dict[str, Any]):
    """실시간 품질 모니터링"""
    st.markdown("## 🔍 실시간 품질 모니터링")
    
    if not files_analysis:
        st.info("파일 분석을 시작하면 품질 정보가 표시됩니다.")
        return
    
    # 전체 품질 점수
    overall_score = files_analysis.get('batch_statistics', {}).get('average_quality', 0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 전체 품질 게이지
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = overall_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "전체 품질 점수"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # 품질 분포
        stats = files_analysis['batch_statistics']
        quality_data = {
            '품질 등급': ['우수 (80+)', '양호 (60-80)', '개선필요 (<60)'],
            '파일 수': [
                stats.get('high_quality_count', 0),
                stats.get('total_files', 0) - stats.get('high_quality_count', 0) - stats.get('low_quality_count', 0),
                stats.get('low_quality_count', 0)
            ]
        }
        
        fig_pie = px.pie(
            values=quality_data['파일 수'],
            names=quality_data['품질 등급'],
            title="품질 등급 분포",
            color_discrete_sequence=['green', 'orange', 'red']
        )
        fig_pie.update_layout(height=250)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col3:
        # 권장사항
        recommendations = files_analysis['batch_statistics'].get('recommendations', [])
        
        st.markdown("**💡 품질 개선 권장사항**")
        if recommendations:
            for rec in recommendations[:5]:
                if '우수' in rec or '✅' in rec:
                    st.success(rec)
                elif '개선' in rec or '⚠️' in rec:
                    st.warning(rec)
                else:
                    st.info(rec)
        else:
            st.success("품질이 우수합니다!")

def render_multilingual_analysis(multilingual_results: Dict[str, Any]):
    """다국어 분석 결과 표시"""
    st.markdown("## 🌍 다국어 분석 결과")
    
    if not multilingual_results:
        st.info("다국어 분석 결과가 없습니다.")
        return
    
    # 언어 분포
    lang_dist = multilingual_results.get('language_distribution', {})
    if lang_dist:
        col1, col2 = st.columns(2)
        
        with col1:
            # 언어 분포 차트
            lang_names = {'ko': '한국어', 'en': '영어', 'zh': '중국어', 'ja': '일본어'}
            display_data = {lang_names.get(k, k): v for k, v in lang_dist.items()}
            
            fig_lang = px.bar(
                x=list(display_data.keys()),
                y=list(display_data.values()),
                title="감지된 언어 분포",
                labels={'x': '언어', 'y': '비율'}
            )
            fig_lang.update_layout(height=300)
            st.plotly_chart(fig_lang, use_container_width=True)
        
        with col2:
            # 처리 통계
            stats = multilingual_results.get('processing_statistics', {})
            
            st.markdown("**📊 처리 통계**")
            st.metric("처리된 파일", stats.get('successful_files', 0))
            st.metric("평균 신뢰도", f"{stats.get('average_confidence', 0):.1%}")
            st.metric("처리 시간", f"{stats.get('total_processing_time', 0):.1f}초")
    
    # 통합 결과
    integrated = multilingual_results.get('integrated_result', {})
    if integrated and not integrated.get('error'):
        st.markdown("### 🎯 핵심 인사이트")
        insights = integrated.get('key_insights', [])
        if insights:
            for insight in insights[:5]:
                st.markdown(f"• {insight}")
        
        # 주얼리 전문용어
        jewelry_count = integrated.get('jewelry_terms_count', 0)
        if jewelry_count > 0:
            st.success(f"💎 주얼리 전문용어 {jewelry_count}개 식별됨")

def render_integration_results(integration_results: Dict[str, Any]):
    """통합 분석 결과 표시"""
    st.markdown("## 📊 다중 파일 통합 분석")
    
    if not integration_results:
        st.info("통합 분석 결과가 없습니다.")
        return
    
    # 전체 통계
    stats = integration_results.get('processing_statistics', {})
    timeline = integration_results.get('timeline_analysis', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 세션", stats.get('total_sessions', 0))
    with col2:
        st.metric("처리된 파일", stats.get('total_files', 0))
    with col3:
        duration = timeline.get('total_duration_hours', 0)
        st.metric("분석 기간", f"{duration:.1f}시간")
    with col4:
        st.metric("처리 시간", f"{stats.get('processing_time', 0):.1f}초")
    
    # 세션별 분석
    sessions = integration_results.get('individual_sessions', [])
    if sessions:
        st.markdown("### 📋 세션별 분석 결과")
        
        session_data = []
        for i, session in enumerate(sessions):
            session_data.append({
                "세션": f"#{i+1}",
                "제목": session.title[:50] + "..." if len(session.title) > 50 else session.title,
                "타입": session.session_type,
                "파일 수": len(session.files),
                "신뢰도": f"{session.confidence_score:.1%}",
                "시작 시간": datetime.fromtimestamp(session.start_time).strftime("%m/%d %H:%M")
            })
        
        df_sessions = pd.DataFrame(session_data)
        st.dataframe(df_sessions, use_container_width=True)
        
        # 세션 상세 정보 (선택적)
        selected_session = st.selectbox("상세 보기할 세션 선택", range(len(sessions)), format_func=lambda x: f"세션 #{x+1}: {sessions[x].title[:30]}...")
        
        if selected_session is not None:
            session = sessions[selected_session]
            
            with st.expander(f"📄 세션 #{selected_session+1} 상세 정보", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**핵심 인사이트:**")
                    for insight in session.key_insights[:5]:
                        st.markdown(f"• {insight}")
                
                with col2:
                    st.markdown("**액션 아이템:**")
                    for action in session.action_items[:5]:
                        st.markdown(f"• {action}")
                
                if session.summary:
                    st.markdown("**요약:**")
                    st.text_area("", session.summary, height=150, disabled=True)

def render_korean_analysis(korean_results: KoreanAnalysisResult):
    """한국어 통합 분석 결과 표시"""
    st.markdown("## 🎯 한국어 통합 분석 결과")
    
    if not korean_results:
        st.info("한국어 분석 결과가 없습니다.")
        return
    
    # 신뢰도 및 기본 정보
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("분석 신뢰도", f"{korean_results.confidence_score:.1%}")
    with col2:
        insight_count = (len(korean_results.business_insights) + 
                        len(korean_results.technical_insights) + 
                        len(korean_results.market_insights))
        st.metric("추출된 인사이트", insight_count)
    with col3:
        st.metric("액션 아이템", len(korean_results.action_items))
    
    # 탭으로 구분된 결과 표시
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 경영진 요약", "💡 핵심 인사이트", "📋 액션 아이템", "💎 전문용어", "📄 상세 분석"])
    
    with tab1:
        if korean_results.executive_summary:
            st.markdown(korean_results.executive_summary)
        else:
            st.info("경영진 요약이 생성되지 않았습니다.")
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📈 비즈니스 인사이트**")
            for insight in korean_results.business_insights:
                st.markdown(f"• {insight}")
        
        with col2:
            st.markdown("**🔧 기술적 인사이트**")
            for insight in korean_results.technical_insights:
                st.markdown(f"• {insight}")
        
        with col3:
            st.markdown("**🌍 시장 인사이트**")
            for insight in korean_results.market_insights:
                st.markdown(f"• {insight}")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 액션 아이템**")
            for action in korean_results.action_items:
                st.markdown(f"• {action}")
        
        with col2:
            st.markdown("**✅ 주요 결정사항**")
            for decision in korean_results.key_decisions:
                st.markdown(f"• {decision}")
    
    with tab4:
        if korean_results.jewelry_terminology:
            # 전문용어 차트
            terms = list(korean_results.jewelry_terminology.keys())[:10]
            counts = list(korean_results.jewelry_terminology.values())[:10]
            
            fig_terms = px.bar(
                x=counts,
                y=terms,
                orientation='h',
                title="주얼리 전문용어 빈도",
                labels={'x': '언급 횟수', 'y': '용어'}
            )
            fig_terms.update_layout(height=400)
            st.plotly_chart(fig_terms, use_container_width=True)
        else:
            st.info("주얼리 전문용어가 감지되지 않았습니다.")
    
    with tab5:
        if korean_results.detailed_analysis:
            st.markdown(korean_results.detailed_analysis)
        else:
            st.info("상세 분석이 생성되지 않았습니다.")

def process_files_v21(uploaded_files: List, settings: Dict[str, Any]):
    """v2.1 파일 처리 메인 함수"""
    
    processor = st.session_state.v21_processor
    
    # 진행률 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: 품질 분석 (30%)
        status_text.text("🔍 파일 품질 분석 중...")
        progress_bar.progress(10)
        
        # 임시 파일 저장 및 경로 생성
        file_paths = []
        for file in uploaded_files:
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            file_paths.append(temp_path)
        
        # 품질 분석
        quality_results = processor['quality_analyzer'].analyze_batch_quality(file_paths)
        progress_bar.progress(30)
        
        # Step 2: 다국어 처리 (50%)
        status_text.text("🌍 다국어 처리 중...")
        multilingual_results = processor['multilingual_processor'].process_multilingual_content(
            file_paths, "audio"  # 주로 오디오 파일 처리
        )
        progress_bar.progress(50)
        
        # Step 3: 다중 파일 통합 (70%)
        status_text.text("📊 다중 파일 통합 분석 중...")
        integration_results = processor['file_integrator'].integrate_multiple_files(
            file_paths, 
            stt_results={}, 
            ocr_results={}
        )
        progress_bar.progress(70)
        
        # Step 4: 한국어 통합 분석 (90%)
        status_text.text("🎯 한국어 통합 분석 중...")
        
        # 통합된 내용 추출
        integrated_content = ""
        if integration_results.get('overall_integration'):
            integrated_content = integration_results['overall_integration'].get('integrated_content', '')
        
        if not integrated_content and multilingual_results.get('integrated_result'):
            integrated_content = multilingual_results['integrated_result'].get('final_korean_text', '')
        
        korean_results = processor['korean_engine'].analyze_korean_content(
            integrated_content, 
            settings['summary_style']
        )
        progress_bar.progress(90)
        
        # Step 5: 결과 정리 (100%)
        status_text.text("✅ 분석 완료!")
        progress_bar.progress(100)
        
        # 세션 상태에 결과 저장
        st.session_state.v21_processor['analysis_results'] = {
            'quality': quality_results,
            'multilingual': multilingual_results,
            'integration': integration_results,
            'korean': korean_results,
            'settings': settings,
            'timestamp': datetime.now()
        }
        
        # 임시 파일 정리
        for temp_path in file_paths:
            try:
                Path(temp_path).unlink()
            except:
                pass
        
        return True
        
    except Exception as e:
        st.error(f"처리 중 오류 발생: {e}")
        return False
    finally:
        # UI 요소 정리
        progress_bar.empty()
        status_text.empty()

def generate_comprehensive_report():
    """종합 리포트 생성"""
    results = st.session_state.v21_processor['analysis_results']
    
    if not results:
        st.warning("분석 결과가 없습니다.")
        return
    
    korean_results = results.get('korean')
    if not korean_results:
        st.error("한국어 분석 결과를 찾을 수 없습니다.")
        return
    
    # 종합 리포트 생성
    engine = st.session_state.v21_processor['korean_engine']
    report = engine.generate_comprehensive_report(korean_results)
    
    # 품질 리포트 추가
    quality_results = results.get('quality')
    if quality_results and results['settings'].get('include_quality_report'):
        quality_analyzer = st.session_state.v21_processor['quality_analyzer']
        quality_report = quality_analyzer.get_quality_report(quality_results)
        report += "\n\n" + quality_report
    
    # 다국어 처리 요약 추가
    multilingual_results = results.get('multilingual')
    if multilingual_results:
        processor = st.session_state.v21_processor['multilingual_processor']
        multilingual_summary = processor.generate_multilingual_summary(multilingual_results)
        report += "\n\n" + multilingual_summary
    
    return report

def main():
    """메인 함수"""
    render_header()
    
    # 사이드바 설정
    settings = render_sidebar()
    
    # 메인 콘텐츠
    # 파일 업로드
    files_uploaded = render_file_upload()
    
    if files_uploaded:
        # 분석 시작 버튼
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 v2.1 품질 강화 분석 시작", type="primary", use_container_width=True):
                with st.spinner("v2.1 고급 분석 진행 중..."):
                    success = process_files_v21(st.session_state.v21_processor['uploaded_files'], settings)
                    
                    if success:
                        st.success("✅ v2.1 분석이 성공적으로 완료되었습니다!")
                        st.rerun()
    
    # 분석 결과 표시
    if 'analysis_results' in st.session_state.v21_processor and st.session_state.v21_processor['analysis_results']:
        results = st.session_state.v21_processor['analysis_results']
        
        st.markdown("---")
        
        # 품질 모니터링
        render_quality_monitor(results.get('quality', {}))
        
        # 다국어 분석
        render_multilingual_analysis(results.get('multilingual', {}))
        
        # 통합 분석
        render_integration_results(results.get('integration', {}))
        
        # 한국어 분석
        render_korean_analysis(results.get('korean'))
        
        # 종합 리포트 다운로드
        st.markdown("---")
        st.markdown("## 📄 종합 리포트")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 종합 리포트 생성", type="secondary", use_container_width=True):
                with st.spinner("리포트 생성 중..."):
                    report = generate_comprehensive_report()
                    if report:
                        st.session_state['generated_report'] = report
                        st.success("리포트가 생성되었습니다!")
        
        with col2:
            if 'generated_report' in st.session_state:
                report_bytes = st.session_state['generated_report'].encode('utf-8')
                st.download_button(
                    label="💾 리포트 다운로드",
                    data=report_bytes,
                    file_name=f"솔로몬드_AI_v21_분석리포트_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        
        # 생성된 리포트 미리보기
        if 'generated_report' in st.session_state:
            with st.expander("📋 리포트 미리보기", expanded=False):
                st.markdown(st.session_state['generated_report'])

if __name__ == "__main__":
    main()
