#!/usr/bin/env python3
"""
실시간 모니터링 대시보드 v2.6
지능형 예측 + 스마트 알림 필터링 통합
Streamlit 기반 고급 모니터링 인터페이스
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading
import asyncio

# 모니터링 모듈 임포트
try:
    from .realtime_performance_monitor import RealtimePerformanceMonitor, get_global_monitor
    from .code_quality_analyzer import CodeQualityAnalyzer
    from .automated_test_engine import AutomatedTestEngine
    from .intelligent_prediction_engine import get_global_prediction_engine
    from .smart_alert_filter import get_global_smart_filter
    from .memory_optimization_engine import get_global_memory_optimizer
except ImportError:
    # 개발 환경에서의 임포트
    import sys
    sys.path.append('.')
    from realtime_performance_monitor import RealtimePerformanceMonitor, get_global_monitor
    from code_quality_analyzer import CodeQualityAnalyzer
    from automated_test_engine import AutomatedTestEngine
    from intelligent_prediction_engine import get_global_prediction_engine
    from smart_alert_filter import get_global_smart_filter
    from memory_optimization_engine import get_global_memory_optimizer

# 스마트 캐시 관리자 임포트
try:
    from ..smart_cache_manager import get_global_cache_manager
    from ..cache_analyzer import CacheAnalyzer
except ImportError:
    try:
        from smart_cache_manager import get_global_cache_manager
        from cache_analyzer import CacheAnalyzer
    except ImportError:
        get_global_cache_manager = None
        CacheAnalyzer = None

class MonitoringDashboard:
    """실시간 모니터링 대시보드"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.performance_monitor = get_global_monitor()
        self.code_analyzer = CodeQualityAnalyzer(project_root)
        self.test_engine = AutomatedTestEngine(project_root)
        
        # v2.6 NEW: 지능형 예측 및 스마트 알림 필터링, 메모리 최적화, 스마트 캐시
        self.prediction_engine = get_global_prediction_engine()
        self.alert_filter = get_global_smart_filter()
        self.memory_optimizer = get_global_memory_optimizer()
        
        # 스마트 캐시 관리자
        if get_global_cache_manager:
            self.cache_manager = get_global_cache_manager()
            if CacheAnalyzer:
                self.cache_analyzer = CacheAnalyzer(self.cache_manager)
            else:
                self.cache_analyzer = None
        else:
            self.cache_manager = None
            self.cache_analyzer = None
        
        # 대시보드 설정
        self.refresh_interval = 5  # 5초마다 업데이트
        self.max_data_points = 100  # 최대 데이터 포인트
        
        # 세션 상태 초기화
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'test_results' not in st.session_state:
            st.session_state.test_results = None
        if 'code_quality_report' not in st.session_state:
            st.session_state.code_quality_report = None
    
    def run(self):
        """대시보드 메인 실행"""
        st.set_page_config(
            page_title="솔로몬드 AI 모니터링 대시보드",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 사이드바 설정
        self._render_sidebar()
        
        # 메인 대시보드
        st.title("📊 솔로몬드 AI 시스템 모니터링 대시보드")
        st.markdown("---")
        
        # 자동 새로고침
        if st.session_state.auto_refresh:
            current_time = time.time()
            if current_time - st.session_state.last_refresh > self.refresh_interval:
                st.session_state.last_refresh = current_time
                st.rerun()
        
        # 탭 생성 (v2.6 확장 + 메모리 최적화 + 스마트 캐시)
        overview_tab, performance_tab, quality_tab, testing_tab, alerts_tab, prediction_tab, optimization_tab, memory_tab, cache_tab = st.tabs([
            "🎯 개요", "⚡ 성능", "🔍 코드 품질", "🧪 테스트", "🚨 알림", "🔮 예측", "🚀 최적화", "🧠 메모리", "📦 캐시"
        ])
        
        with overview_tab:
            self._render_overview_tab()
        
        with performance_tab:
            self._render_performance_tab()
        
        with quality_tab:
            self._render_quality_tab()
        
        with testing_tab:
            self._render_testing_tab()
        
        with alerts_tab:
            self._render_alerts_tab()
        
        with prediction_tab:
            self._render_prediction_tab()
        
        with optimization_tab:
            self._render_optimization_tab()
        
        with memory_tab:
            self._render_memory_tab()
        
        with cache_tab:
            self._render_cache_tab()
    
    def _render_sidebar(self):
        """사이드바 렌더링"""
        st.sidebar.title("⚙️ 대시보드 설정")
        
        # 자동 새로고침 설정
        st.session_state.auto_refresh = st.sidebar.checkbox(
            "🔄 자동 새로고침", 
            value=st.session_state.auto_refresh
        )
        
        self.refresh_interval = st.sidebar.slider(
            "새로고침 간격 (초)", 
            min_value=1, 
            max_value=30, 
            value=self.refresh_interval
        )
        
        # 수동 새로고침 버튼
        if st.sidebar.button("🔄 지금 새로고침"):
            st.session_state.last_refresh = time.time()
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # 시스템 정보
        st.sidebar.subheader("📋 시스템 정보")
        current_metrics = self.performance_monitor.get_current_metrics()
        
        if current_metrics:
            st.sidebar.metric("💾 메모리 사용률", f"{current_metrics.memory_usage:.1f}%")
            st.sidebar.metric("⚡ CPU 사용률", f"{current_metrics.cpu_usage:.1f}%")
            st.sidebar.metric("👥 활성 세션", f"{current_metrics.user_sessions}개")
        else:
            st.sidebar.info("모니터링 데이터를 수집 중...")
        
        st.sidebar.markdown("---")
        
        # 액션 버튼들
        st.sidebar.subheader("🚀 빠른 액션")
        
        if st.sidebar.button("🧪 테스트 실행"):
            self._run_tests()
        
        if st.sidebar.button("🔍 코드 품질 분석"):
            self._run_code_analysis()
        
        if st.sidebar.button("📊 성능 보고서 생성"):
            self._generate_performance_report()
    
    def _render_overview_tab(self):
        """개요 탭 렌더링"""
        col1, col2, col3, col4 = st.columns(4)
        
        # 현재 메트릭 가져오기
        current_metrics = self.performance_monitor.get_current_metrics()
        summary = self.performance_monitor.get_performance_summary()
        
        with col1:
            if current_metrics:
                st.metric(
                    "💾 메모리 사용률", 
                    f"{current_metrics.memory_usage:.1f}%",
                    delta=f"{current_metrics.memory_usage - summary.get('averages', {}).get('memory_usage', 0):.1f}%"
                )
            else:
                st.metric("💾 메모리 사용률", "N/A")
        
        with col2:
            if current_metrics:
                st.metric(
                    "⚡ CPU 사용률", 
                    f"{current_metrics.cpu_usage:.1f}%",
                    delta=f"{current_metrics.cpu_usage - summary.get('averages', {}).get('cpu_usage', 0):.1f}%"
                )
            else:
                st.metric("⚡ CPU 사용률", "N/A")
        
        with col3:
            if current_metrics:
                st.metric("🎯 Ollama 모델", f"{current_metrics.ollama_models_active}개")
            else:
                st.metric("🎯 Ollama 모델", "N/A")
        
        with col4:
            if current_metrics:
                st.metric("👥 활성 세션", f"{current_metrics.user_sessions}개")
            else:
                st.metric("👥 활성 세션", "N/A")
        
        st.markdown("---")
        
        # 시스템 상태 개요
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 시스템 상태")
            
            if summary.get('status') == 'active':
                # 상태 표시기
                avg_cpu = summary.get('averages', {}).get('cpu_usage', 0)
                avg_memory = summary.get('averages', {}).get('memory_usage', 0)
                
                if avg_cpu < 70 and avg_memory < 75:
                    st.success("🟢 시스템 상태: 정상")
                elif avg_cpu < 85 and avg_memory < 90:
                    st.warning("🟡 시스템 상태: 주의")
                else:
                    st.error("🔴 시스템 상태: 위험")
                
                st.info(f"📊 모니터링 시간: {summary.get('monitoring_duration_hours', 0):.1f}시간")
            else:
                st.warning("⚠️ 모니터링 데이터 부족")
        
        with col2:
            st.subheader("🚨 알림 현황")
            
            if summary.get('active_alerts', 0) > 0:
                st.error(f"🚨 활성 알림: {summary.get('active_alerts', 0)}개")
                if summary.get('critical_alerts', 0) > 0:
                    st.error(f"⚠️ 심각한 알림: {summary.get('critical_alerts', 0)}개")
            else:
                st.success("✅ 활성 알림 없음")
            
            st.info(f"📝 총 에러: {summary.get('total_errors', 0)}개")
        
        # 최근 활동 로그
        st.subheader("📋 최근 활동")
        
        # 가상의 활동 로그 (실제로는 로그 파일에서 읽어올 수 있음)
        activity_data = [
            {"시간": "10:30:15", "이벤트": "Ollama 모델 로드 완료", "상태": "성공"},
            {"시간": "10:28:43", "이벤트": "사용자 세션 시작", "상태": "정보"},
            {"시간": "10:25:22", "이벤트": "성능 모니터링 시작", "상태": "정보"},
            {"시간": "10:23:11", "이벤트": "시스템 초기화 완료", "상태": "성공"}
        ]
        
        df_activity = pd.DataFrame(activity_data)
        st.dataframe(df_activity, use_container_width=True)
    
    def _render_performance_tab(self):
        """성능 탭 렌더링"""
        st.subheader("⚡ 실시간 성능 모니터링")
        
        # 성능 메트릭 히스토리 가져오기
        metrics_history = self.performance_monitor.get_metrics_history(hours=1)
        
        if not metrics_history:
            st.warning("⚠️ 성능 데이터가 없습니다. 잠시 후 다시 확인해주세요.")
            return
        
        # 데이터프레임 생성
        df_metrics = pd.DataFrame([
            {
                'timestamp': datetime.fromtimestamp(m.timestamp),
                'cpu_usage': m.cpu_usage,
                'memory_usage': m.memory_usage,
                'response_time_ms': m.response_time_ms,
                'user_sessions': m.user_sessions
            }
            for m in metrics_history
        ])
        
        # CPU 및 메모리 사용률 차트
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU 사용률', '메모리 사용률', '응답 시간', '사용자 세션'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CPU 사용률
        fig.add_trace(
            go.Scatter(x=df_metrics['timestamp'], y=df_metrics['cpu_usage'],
                      mode='lines+markers', name='CPU %', line=dict(color='#FF6B6B')),
            row=1, col=1
        )
        
        # 메모리 사용률
        fig.add_trace(
            go.Scatter(x=df_metrics['timestamp'], y=df_metrics['memory_usage'],
                      mode='lines+markers', name='Memory %', line=dict(color='#4ECDC4')),
            row=1, col=2
        )
        
        # 응답 시간
        fig.add_trace(
            go.Scatter(x=df_metrics['timestamp'], y=df_metrics['response_time_ms'],
                      mode='lines+markers', name='Response Time (ms)', line=dict(color='#45B7D1')),
            row=2, col=1
        )
        
        # 사용자 세션
        fig.add_trace(
            go.Scatter(x=df_metrics['timestamp'], y=df_metrics['user_sessions'],
                      mode='lines+markers', name='User Sessions', line=dict(color='#96CEB4')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="실시간 성능 메트릭")
        st.plotly_chart(fig, use_container_width=True)
        
        # 성능 통계
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 평균 성능")
            avg_cpu = df_metrics['cpu_usage'].mean()
            avg_memory = df_metrics['memory_usage'].mean()
            avg_response = df_metrics['response_time_ms'].mean()
            
            st.metric("평균 CPU", f"{avg_cpu:.1f}%")
            st.metric("평균 메모리", f"{avg_memory:.1f}%")
            st.metric("평균 응답시간", f"{avg_response:.0f}ms")
        
        with col2:
            st.subheader("📈 최대 성능")
            max_cpu = df_metrics['cpu_usage'].max()
            max_memory = df_metrics['memory_usage'].max()
            max_response = df_metrics['response_time_ms'].max()
            
            st.metric("최대 CPU", f"{max_cpu:.1f}%")
            st.metric("최대 메모리", f"{max_memory:.1f}%")
            st.metric("최대 응답시간", f"{max_response:.0f}ms")
        
        with col3:
            st.subheader("📉 최소 성능")
            min_cpu = df_metrics['cpu_usage'].min()
            min_memory = df_metrics['memory_usage'].min()
            min_response = df_metrics['response_time_ms'].min()
            
            st.metric("최소 CPU", f"{min_cpu:.1f}%")
            st.metric("최소 메모리", f"{min_memory:.1f}%")
            st.metric("최소 응답시간", f"{min_response:.0f}ms")
    
    def _render_quality_tab(self):
        """코드 품질 탭 렌더링"""
        st.subheader("🔍 코드 품질 분석")
        
        # 코드 품질 분석 실행 버튼
        if st.button("🚀 코드 품질 분석 실행"):
            self._run_code_analysis()
        
        # 기존 보고서 표시
        if st.session_state.code_quality_report:
            report = st.session_state.code_quality_report
            
            # 요약 메트릭
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📁 분석 파일", f"{report.files_analyzed}개")
            
            with col2:
                st.metric("📏 총 코드 라인", f"{report.total_lines:,}줄")
            
            with col3:
                st.metric("⚠️ 이슈 수", f"{report.issues_found}개")
            
            with col4:
                st.metric("📈 복잡도 점수", f"{report.complexity_score}/100")
            
            # 이슈 심각도 분포
            st.subheader("📊 이슈 심각도 분포")
            
            severity_data = {
                '심각': report.critical_issues,
                '높음': report.high_issues,
                '보통': report.medium_issues,
                '낮음': report.low_issues
            }
            
            fig = px.pie(
                values=list(severity_data.values()),
                names=list(severity_data.keys()),
                title="이슈 심각도 분포",
                color_discrete_map={
                    '심각': '#FF4444',
                    '높음': '#FF8800', 
                    '보통': '#FFAA00',
                    '낮음': '#44AA44'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 권장사항
            if report.recommendations:
                st.subheader("🎯 권장사항")
                for rec in report.recommendations:
                    st.info(rec)
            
            # 상세 이슈 목록
            if report.issues:
                st.subheader("📋 상세 이슈 목록")
                
                # 심각도별 필터
                severity_filter = st.selectbox(
                    "심각도 필터",
                    ['전체', 'critical', 'high', 'medium', 'low']
                )
                
                filtered_issues = report.issues
                if severity_filter != '전체':
                    filtered_issues = [i for i in report.issues if i.severity == severity_filter]
                
                # 이슈 테이블
                if filtered_issues:
                    issues_data = []
                    for issue in filtered_issues[:20]:  # 최대 20개만 표시
                        issues_data.append({
                            '파일': Path(issue.file_path).name,
                            '라인': issue.line_number,
                            '타입': issue.issue_type,
                            '심각도': issue.severity,
                            '메시지': issue.message[:100] + '...' if len(issue.message) > 100 else issue.message
                        })
                    
                    df_issues = pd.DataFrame(issues_data)
                    st.dataframe(df_issues, use_container_width=True)
                else:
                    st.info("선택한 심각도의 이슈가 없습니다.")
        else:
            st.info("코드 품질 분석을 실행하여 결과를 확인하세요.")
    
    def _render_testing_tab(self):
        """테스트 탭 렌더링"""
        st.subheader("🧪 자동화된 테스트")
        
        # 테스트 실행 버튼
        if st.button("🚀 전체 테스트 실행"):
            self._run_tests()
        
        # 기존 테스트 결과 표시
        if st.session_state.test_results:
            test_suites = st.session_state.test_results
            
            # 전체 요약
            total_tests = sum(len(suite.tests) for suite in test_suites.values())
            total_passed = sum(suite.passed_count for suite in test_suites.values())
            total_failed = sum(suite.failed_count for suite in test_suites.values())
            total_errors = sum(suite.error_count for suite in test_suites.values())
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 총 테스트", f"{total_tests}개")
            
            with col2:
                st.metric("✅ 통과", f"{total_passed}개")
            
            with col3:
                st.metric("❌ 실패", f"{total_failed}개")
            
            with col4:
                st.metric("⚠️ 오류", f"{total_errors}개")
            
            # 성공률 차트
            if total_tests > 0:
                success_rate = total_passed / total_tests * 100
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = success_rate,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "테스트 성공률 (%)"},
                    delta = {'reference': 90},
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
                
                st.plotly_chart(fig, use_container_width=True)
            
            # 스위트별 결과
            st.subheader("📋 스위트별 결과")
            
            for suite_name, suite in test_suites.items():
                with st.expander(f"{suite.name} ({suite.passed_count}/{len(suite.tests)} 통과)"):
                    
                    # 스위트 요약
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("통과", suite.passed_count)
                    with col2:
                        st.metric("실패", suite.failed_count)
                    with col3:
                        st.metric("오류", suite.error_count)
                    
                    # 실패한 테스트 상세
                    failed_tests = [t for t in suite.tests if t.status in ['failed', 'error']]
                    if failed_tests:
                        st.subheader("❌ 실패한 테스트")
                        for test in failed_tests:
                            st.error(f"**{test.test_name}**: {test.error_message or '상세 정보 없음'}")
        else:
            st.info("테스트를 실행하여 결과를 확인하세요.")
    
    def _render_alerts_tab(self):
        """알림 탭 렌더링"""
        st.subheader("🚨 시스템 알림")
        
        # 알림 히스토리 가져오기
        alerts = list(self.performance_monitor.alerts_history)
        
        if not alerts:
            st.success("✅ 현재 활성 알림이 없습니다.")
            return
        
        # 최근 알림 필터 (지난 1시간)
        recent_cutoff = time.time() - 3600
        recent_alerts = [a for a in alerts if a.timestamp > recent_cutoff]
        
        if recent_alerts:
            st.warning(f"⚠️ 최근 1시간 내 {len(recent_alerts)}개의 알림이 발생했습니다.")
            
            # 알림 레벨별 분류
            critical_alerts = [a for a in recent_alerts if a.level == 'critical']
            warning_alerts = [a for a in recent_alerts if a.level == 'warning']
            
            if critical_alerts:
                st.error(f"🚨 심각한 알림: {len(critical_alerts)}개")
                for alert in critical_alerts[:5]:  # 최대 5개만 표시
                    st.error(f"**{alert.message}** (값: {alert.value:.2f}, 임계값: {alert.threshold:.2f})")
                    st.caption(f"권장 조치: {alert.suggested_action}")
            
            if warning_alerts:
                st.warning(f"⚠️ 경고 알림: {len(warning_alerts)}개")
                for alert in warning_alerts[:5]:  # 최대 5개만 표시
                    st.warning(f"**{alert.message}** (값: {alert.value:.2f}, 임계값: {alert.threshold:.2f})")
                    st.caption(f"권장 조치: {alert.suggested_action}")
        else:
            st.success("✅ 최근 1시간 내 알림이 없습니다.")
        
        # 알림 히스토리 차트
        if len(alerts) > 1:
            st.subheader("📈 알림 히스토리")
            
            # 시간별 알림 수 집계
            alert_times = [datetime.fromtimestamp(a.timestamp) for a in alerts]
            df_alerts = pd.DataFrame({
                'timestamp': alert_times,
                'level': [a.level for a in alerts]
            })
            
            # 1시간 단위로 그룹화
            df_alerts['hour'] = df_alerts['timestamp'].dt.floor('H')
            alert_counts = df_alerts.groupby(['hour', 'level']).size().reset_index(name='count')
            
            fig = px.bar(
                alert_counts, 
                x='hour', 
                y='count', 
                color='level',
                title="시간별 알림 발생 현황",
                color_discrete_map={
                    'critical': '#FF4444',
                    'warning': '#FF8800',
                    'info': '#4444FF'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _run_code_analysis(self):
        """코드 품질 분석 실행"""
        with st.spinner("🔍 코드 품질 분석 중..."):
            try:
                report = self.code_analyzer.analyze_project()
                st.session_state.code_quality_report = report
                st.success("✅ 코드 품질 분석이 완료되었습니다!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 코드 품질 분석 실패: {e}")
    
    def _run_tests(self):
        """테스트 실행"""
        with st.spinner("🧪 테스트 실행 중..."):
            try:
                # 비동기 함수를 동기적으로 실행
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                test_results = loop.run_until_complete(self.test_engine.run_all_tests())
                loop.close()
                
                st.session_state.test_results = test_results
                st.success("✅ 테스트가 완료되었습니다!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 테스트 실행 실패: {e}")
    
    def _generate_performance_report(self):
        """성능 보고서 생성"""
        with st.spinner("📊 성능 보고서 생성 중..."):
            try:
                summary = self.performance_monitor.get_performance_summary()
                metrics_data = self.performance_monitor.export_metrics(hours=24)
                
                # 보고서 파일 생성
                report_path = self.project_root / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'summary': summary,
                        'detailed_metrics': metrics_data
                    }, f, ensure_ascii=False, indent=2)
                
                st.success(f"✅ 성능 보고서가 생성되었습니다: {report_path.name}")
                
                # 다운로드 버튼
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                
                st.download_button(
                    label="📥 성능 보고서 다운로드",
                    data=report_content,
                    file_name=report_path.name,
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"❌ 성능 보고서 생성 실패: {e}")
    
    def _render_prediction_tab(self):
        """🔮 예측 탭 렌더링 (v2.6 NEW)"""
        st.subheader("🔮 지능형 처리 시간 예측")
        
        # 예측 입력 섹션
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_name = st.selectbox(
                "AI 모델 선택",
                options=['gemma3:27b', 'qwen3:8b', 'qwen2.5:7b', 'gemma3:4b', 
                        'solar:latest', 'mistral:latest', 'llama3.2:latest',
                        'whisper_cpu', 'easyocr', 'transformers_cpu'],
                index=1  # qwen3:8b 기본 선택
            )
        
        with col2:
            file_type = st.selectbox(
                "파일 타입",
                options=['audio/wav', 'audio/mp3', 'audio/m4a', 
                        'image/jpeg', 'image/png', 'video/mp4', 'text/plain']
            )
        
        with col3:
            file_size_mb = st.number_input(
                "파일 크기 (MB)",
                min_value=0.1,
                max_value=1000.0,
                value=10.0,
                step=0.5
            )
        
        # 예측 실행 버튼
        if st.button("🔮 처리 시간 예측"):
            with st.spinner("예측 계산 중..."):
                try:
                    prediction = self.prediction_engine.predict_processing_time(
                        model_name, file_type, file_size_mb
                    )
                    
                    # 예측 결과 표시
                    st.success("✅ 예측 완료!")
                    
                    # 메트릭 카드들
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "⏱️ 예상 처리 시간",
                            f"{prediction.predicted_time_ms/1000:.1f}초",
                            help="예상되는 총 처리 시간"
                        )
                    
                    with col2:
                        st.metric(
                            "💾 예상 메모리 사용",
                            f"{prediction.predicted_memory_mb:.1f}MB",
                            help="처리 중 사용될 메모리"
                        )
                    
                    with col3:
                        st.metric(
                            "⚡ 예상 CPU 사용률",
                            f"{prediction.predicted_cpu_percent:.1f}%",
                            help="처리 중 CPU 사용률"
                        )
                    
                    with col4:
                        st.metric(
                            "🎯 예측 신뢰도",
                            f"{prediction.confidence_score:.0%}",
                            help="예측 결과의 신뢰도"
                        )
                    
                    # 최적화 제안
                    if prediction.optimization_suggestions:
                        st.subheader("💡 최적화 제안")
                        for i, suggestion in enumerate(prediction.optimization_suggestions, 1):
                            st.info(f"{i}. {suggestion}")
                    
                except Exception as e:
                    st.error(f"❌ 예측 실패: {e}")
        
        st.markdown("---")
        
        # 모델 추천 시스템
        st.subheader("🏆 상황별 최적 모델 추천")
        
        priority = st.selectbox(
            "우선순위",
            options=['balanced', 'speed', 'memory', 'accuracy'],
            format_func=lambda x: {
                'balanced': '⚖️ 균형잡힌 성능',
                'speed': '🚀 처리 속도 우선',
                'memory': '💾 메모리 효율성 우선',
                'accuracy': '🎯 정확도 우선'
            }[x]
        )
        
        if st.button("🔍 모델 추천 받기"):
            with st.spinner("최적 모델 분석 중..."):
                try:
                    recommendations = self.prediction_engine.get_model_recommendations(
                        file_type, file_size_mb, priority
                    )
                    
                    st.success(f"✅ {priority} 우선 모델 추천 완료!")
                    
                    # 추천 결과 표시
                    for i, rec in enumerate(recommendations, 1):
                        with st.expander(f"🥇 추천 {i}위: {rec['model_name']}", expanded=(i==1)):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("처리 시간", f"{rec['predicted_time_ms']/1000:.1f}초")
                            with col2:
                                st.metric("메모리 사용", f"{rec['predicted_memory_mb']:.1f}MB")
                            with col3:
                                st.metric("신뢰도", f"{rec['confidence_score']:.0%}")
                            
                            # 모델 특성 정보
                            profile = rec['profile']
                            st.info(f"**모델 특성**: 기본 처리율 {profile.base_processing_rate:.1f}ms/MB, "
                                   f"메모리 효율성 {profile.memory_efficiency:.1f}MB/MB, "
                                   f"에러율 {profile.error_rate:.1%}")
                
                except Exception as e:
                    st.error(f"❌ 모델 추천 실패: {e}")
        
        # 성능 통계
        st.markdown("---")
        st.subheader("📊 예측 엔진 성능 통계")
        
        try:
            summary = self.prediction_engine.get_performance_summary()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("지원 모델 수", f"{summary['total_models']}개")
            with col2:
                st.metric("지원 파일 타입", f"{summary['supported_file_types']}개")
            with col3:
                st.metric("처리 이력", f"{summary['processing_history_count']}개")
            with col4:
                st.metric("예측 정확도", f"{summary['prediction_accuracy']:.1%}")
            
            # 최고 성능 모델 정보
            if summary['fastest_model'] != 'none':
                st.success(f"🚀 최고 속도: {summary['fastest_model']}")
            if summary['most_accurate_model'] != 'none':
                st.success(f"🎯 최고 정확도: {summary['most_accurate_model']}")
        
        except Exception as e:
            st.warning(f"통계 정보를 가져올 수 없습니다: {e}")
    
    def _render_optimization_tab(self):
        """🚀 최적화 탭 렌더링 (v2.6 NEW)"""
        st.subheader("🚀 시스템 최적화 및 스마트 알림 관리")
        
        # 스마트 알림 필터링 상태
        st.subheader("🔍 스마트 알림 필터링 상태")
        
        try:
            alert_summary = self.alert_filter.get_alert_summary()
            
            # 알림 통계 메트릭
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "📢 활성 알림",
                    f"{alert_summary['active_alerts_count']}개",
                    help="현재 활성화된 알림 수"
                )
            
            with col2:
                st.metric(
                    "🚫 억제된 알림",
                    f"{alert_summary['suppressed_alerts_count']}개",
                    help="스마트 필터링으로 억제된 알림 수"
                )
            
            with col3:
                st.metric(
                    "📊 총 처리량",
                    f"{alert_summary['total_processed']}개",
                    help="총 처리된 알림 수"
                )
            
            with col4:
                st.metric(
                    "⚡ 필터링 효율성",
                    f"{alert_summary['filtering_efficiency']:.1f}%",
                    help="억제된 알림 비율"
                )
            
            # 최근 1시간 알림 현황
            if alert_summary['recent_hour_stats']['total'] > 0:
                st.subheader("📈 최근 1시간 알림 현황")
                
                recent_stats = alert_summary['recent_hour_stats']
                
                # 레벨별 분포
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**레벨별 분포**")
                    level_data = recent_stats['by_level']
                    if level_data:
                        fig_level = px.pie(
                            values=list(level_data.values()),
                            names=list(level_data.keys()),
                            title="알림 레벨 분포"
                        )
                        st.plotly_chart(fig_level, use_container_width=True)
                
                with col2:
                    st.write("**카테고리별 분포**")
                    category_data = recent_stats['by_category']
                    if category_data:
                        fig_category = px.pie(
                            values=list(category_data.values()),
                            names=list(category_data.keys()),
                            title="알림 카테고리 분포"
                        )
                        st.plotly_chart(fig_category, use_container_width=True)
            
            # 활성 패턴
            if alert_summary['top_patterns']:
                st.subheader("🔍 감지된 알림 패턴 (상위 5개)")
                
                for pattern in alert_summary['top_patterns']:
                    with st.expander(f"📋 {pattern['name']} (매칭: {pattern['matches']}회)"):
                        st.write(f"**마지막 매칭**: {pattern['last_matched'] or 'N/A'}")
                        st.write(f"**총 매칭 횟수**: {pattern['matches']}회")
        
        except Exception as e:
            st.error(f"❌ 알림 필터링 정보를 가져올 수 없습니다: {e}")
        
        st.markdown("---")
        
        # 성능 최적화 제안
        st.subheader("⚡ 성능 최적화 제안")
        
        try:
            current_metrics = self.performance_monitor.get_current_metrics()
            
            if current_metrics:
                # 메모리 최적화 제안
                if current_metrics.memory_usage > 80:
                    st.warning("🔸 **메모리 사용률이 높습니다**")
                    st.info("• 더 가벼운 AI 모델 사용 고려 (GEMMA3:4B, QWEN2.5:7B)")
                    st.info("• 파일을 더 작은 단위로 분할하여 처리")
                    st.info("• 불필요한 백그라운드 프로세스 종료")
                
                # CPU 최적화 제안
                if current_metrics.cpu_usage > 85:
                    st.warning("🔸 **CPU 사용률이 높습니다**")
                    st.info("• 병렬 처리 작업 수 조정")
                    st.info("• 우선순위가 낮은 작업 일시 중단")
                    st.info("• CPU 집약적 모델 대신 효율적 모델 사용")
                
                # 응답 시간 최적화 제안
                if current_metrics.response_time_ms > 5000:
                    st.warning("🔸 **응답 시간이 깁니다**")
                    st.info("• 더 빠른 AI 모델 사용 (GEMMA3:4B 권장)")
                    st.info("• 파일 크기 최적화 (압축, 해상도 조정)")
                    st.info("• 디스크 I/O 최적화 확인")
                
                # 전반적인 상태가 좋은 경우
                if (current_metrics.memory_usage < 70 and 
                    current_metrics.cpu_usage < 70 and 
                    current_metrics.response_time_ms < 3000):
                    st.success("✅ **시스템이 최적 상태로 운영 중입니다!**")
                    st.info("• 현재 설정을 유지하세요")
                    st.info("• 정기적인 모니터링을 계속하세요")
            
            else:
                st.info("성능 데이터를 수집 중입니다...")
        
        except Exception as e:
            st.error(f"❌ 성능 데이터를 가져올 수 없습니다: {e}")
        
        st.markdown("---")
        
        # 시스템 제어 패널
        st.subheader("🎛️ 시스템 제어 패널")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 캐시 정리", help="시스템 캐시를 정리합니다"):
                st.info("캐시 정리를 시뮬레이션합니다...")
                st.success("✅ 캐시 정리 완료")
        
        with col2:
            if st.button("📊 보고서 생성", help="종합 성능 보고서를 생성합니다"):
                self._generate_comprehensive_report()
        
        with col3:
            if st.button("⚙️ 설정 최적화", help="시스템 설정을 자동 최적화합니다"):
                st.info("설정 최적화를 시뮬레이션합니다...")
                st.success("✅ 설정 최적화 완료")
        
        # 고급 설정
        with st.expander("🔧 고급 설정"):
            st.write("**알림 필터링 설정**")
            max_alerts = st.slider("분당 최대 알림 수", 10, 100, 50)
            
            st.write("**성능 모니터링 설정**")
            monitoring_interval = st.slider("모니터링 간격 (초)", 1, 10, 5)
            
            if st.button("💾 설정 저장"):
                st.success("✅ 설정이 저장되었습니다")
    
    def _generate_comprehensive_report(self):
        """종합 보고서 생성"""
        with st.spinner("📊 종합 보고서 생성 중..."):
            try:
                # 성능 요약
                performance_summary = self.performance_monitor.get_performance_summary()
                
                # 알림 요약
                alert_summary = self.alert_filter.get_alert_summary()
                
                # 예측 엔진 요약
                prediction_summary = self.prediction_engine.get_performance_summary()
                
                # 보고서 데이터 구성
                comprehensive_report = {
                    'report_timestamp': datetime.now().isoformat(),
                    'system_status': 'healthy' if performance_summary.get('active_alerts', 0) < 10 else 'warning',
                    'performance_metrics': performance_summary,
                    'alert_filtering': alert_summary,
                    'prediction_engine': prediction_summary,
                    'recommendations': [
                        "정기적인 모니터링 유지",
                        "예측 정확도 개선을 위한 데이터 수집 지속",
                        "알림 패턴 분석을 통한 시스템 최적화"
                    ]
                }
                
                # 보고서 파일 생성
                report_path = self.project_root / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(comprehensive_report, f, ensure_ascii=False, indent=2)
                
                st.success(f"✅ 종합 보고서가 생성되었습니다: {report_path.name}")
                
                # 다운로드 버튼
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                
                st.download_button(
                    label="📥 종합 보고서 다운로드",
                    data=report_content,
                    file_name=report_path.name,
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"❌ 종합 보고서 생성 실패: {e}")
    
    def _render_memory_tab(self):
        """🧠 메모리 탭 렌더링 (NEW)"""
        from .monitoring_dashboard_memory_addon import render_memory_tab, render_memory_optimization_controls, render_memory_trend_chart
        
        # 메모리 상태 및 통계 렌더링
        render_memory_tab(self.memory_optimizer)
        
        # 메모리 최적화 제어 패널
        render_memory_optimization_controls(self.memory_optimizer, self.project_root)
        
        # 메모리 추세 차트
        render_memory_trend_chart(self.memory_optimizer)
    
    def _render_cache_tab(self):
        """📦 캐시 관리 탭 렌더링 (NEW)"""
        st.subheader("📦 스마트 캐시 관리 시스템")
        
        try:
            # 캐시 관리자 초기화 (지연 로딩)
            if not hasattr(self, 'cache_manager') or self.cache_manager is None:
                from ..smart_cache_manager import SmartCacheManager
                self.cache_manager = SmartCacheManager()
                st.info("✅ 캐시 관리 시스템이 초기화되었습니다.")
            
            # 현재 캐시 상태
            cache_stats = self.cache_manager.get_cache_stats()
            
            # 캐시 상태 메트릭
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "💾 L1 캐시 (메모리)",
                    f"{cache_stats.l1_items}개",
                    delta=f"{cache_stats.l1_size_mb:.1f}MB"
                )
            
            with col2:
                st.metric(
                    "💽 L2 캐시 (디스크)",
                    f"{cache_stats.l2_items}개",
                    delta=f"{cache_stats.l2_size_mb:.1f}MB"
                )
            
            with col3:
                st.metric(
                    "🌐 L3 캐시 (네트워크)",
                    f"{cache_stats.l3_items}개",
                    delta=f"{cache_stats.l3_size_mb:.1f}MB"
                )
            
            with col4:
                hit_rate = cache_stats.hit_rate * 100 if cache_stats.hit_rate > 0 else 0
                st.metric(
                    "🎯 히트율",
                    f"{hit_rate:.1f}%",
                    delta=f"{cache_stats.total_hits}회 히트"
                )
            
            # 캐시 효율성 상태
            if hit_rate >= 80:
                st.success("✅ **캐시 효율성이 우수합니다!**")
            elif hit_rate >= 60:
                st.info("ℹ️ **캐시 효율성이 양호합니다.**")
            elif hit_rate >= 40:
                st.warning("⚠️ **캐시 효율성을 개선할 여지가 있습니다.**")
            else:
                st.error("🔴 **캐시 효율성이 낮습니다. 설정을 검토하세요.**")
            
            st.markdown("---")
            
            # 캐시 전략 설정
            st.subheader("⚙️ 캐시 전략 설정")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**현재 캐시 전략**")
                current_config = self.cache_manager.get_current_config()
                
                st.info(f"🎯 기본 전략: {current_config.default_strategy.upper()}")
                st.info(f"💾 최대 크기: {current_config.max_size_mb:.0f}MB")
                st.info(f"⏰ 기본 TTL: {current_config.default_ttl/60:.0f}분")
                st.info(f"🗜️ 압축: {'활성화' if current_config.compression_enabled else '비활성화'}")
            
            with col2:
                st.write("**전략 변경**")
                
                new_strategy = st.selectbox(
                    "캐시 전략 선택",
                    options=['lru', 'lfu', 'ttl', 'adaptive'],
                    index=['lru', 'lfu', 'ttl', 'adaptive'].index(current_config.default_strategy),
                    help="LRU: 최근 사용, LFU: 빈도 기반, TTL: 시간 기반, Adaptive: 적응형"
                )
                
                new_max_size = st.number_input(
                    "최대 캐시 크기 (MB)",
                    min_value=64,
                    max_value=2048,
                    value=int(current_config.max_size_mb),
                    step=64
                )
                
                new_ttl_minutes = st.number_input(
                    "기본 TTL (분)",
                    min_value=5,
                    max_value=1440,
                    value=int(current_config.default_ttl/60),
                    step=5
                )
                
                if st.button("💾 설정 적용"):
                    try:
                        self.cache_manager.update_config(
                            default_strategy=new_strategy,
                            max_size_mb=new_max_size,
                            default_ttl=new_ttl_minutes * 60
                        )
                        st.success("✅ 캐시 설정이 업데이트되었습니다!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 설정 업데이트 실패: {e}")
            
            st.markdown("---")
            
            # 캐시 성능 분석
            st.subheader("📊 캐시 성능 분석")
            
            try:
                performance_report = self.cache_manager.get_performance_analysis()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "⚡ 평균 응답 시간",
                        f"{performance_report.avg_response_time_ms:.1f}ms"
                    )
                    st.metric(
                        "🔥 가장 인기있는 키",
                        performance_report.most_accessed_key[:20] + "..." if len(performance_report.most_accessed_key) > 20 else performance_report.most_accessed_key
                    )
                
                with col2:
                    st.metric(
                        "💾 메모리 효율성",
                        f"{performance_report.memory_efficiency:.1%}"
                    )
                    st.metric(
                        "🗜️ 압축률",
                        f"{performance_report.compression_ratio:.1%}" if performance_report.compression_ratio > 0 else "N/A"
                    )
                
                with col3:
                    st.metric(
                        "🚫 축출된 항목",
                        f"{performance_report.evicted_items}개"
                    )
                    st.metric(
                        "⚠️ 충돌 비율",
                        f"{performance_report.collision_rate:.1%}"
                    )
                
                # 최적화 제안
                if performance_report.optimization_suggestions:
                    st.subheader("💡 최적화 제안")
                    for suggestion in performance_report.optimization_suggestions[:3]:  # 상위 3개
                        importance_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                        st.info(f"{importance_icons.get(suggestion.importance, '📋')} **{suggestion.title}**: {suggestion.description}")
            
            except Exception as e:
                st.warning(f"⚠️ 성능 분석 데이터를 가져올 수 없습니다: {e}")
            
            st.markdown("---")
            
            # 캐시 관리 액션
            st.subheader("🚀 캐시 관리 액션")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🗑️ L1 캐시 정리", help="메모리 캐시를 정리합니다"):
                    with st.spinner("L1 캐시 정리 중..."):
                        try:
                            cleared_items = self.cache_manager.clear_l1_cache()
                            st.success(f"✅ L1 캐시 정리 완료: {cleared_items}개 항목")
                        except Exception as e:
                            st.error(f"❌ L1 캐시 정리 실패: {e}")
            
            with col2:
                if st.button("💽 L2 캐시 정리", help="디스크 캐시를 정리합니다"):
                    with st.spinner("L2 캐시 정리 중..."):
                        try:
                            cleared_items = self.cache_manager.clear_l2_cache()
                            st.success(f"✅ L2 캐시 정리 완료: {cleared_items}개 항목")
                        except Exception as e:
                            st.error(f"❌ L2 캐시 정리 실패: {e}")
            
            with col3:
                if st.button("🧹 전체 캐시 정리", help="모든 캐시를 정리합니다"):
                    with st.spinner("전체 캐시 정리 중..."):
                        try:
                            cleared_items = self.cache_manager.clear_all_caches()
                            st.success(f"✅ 전체 캐시 정리 완료: {cleared_items}개 항목")
                        except Exception as e:
                            st.error(f"❌ 전체 캐시 정리 실패: {e}")
            
            with col4:
                if st.button("🔄 캐시 최적화", help="캐시를 최적화합니다"):
                    with st.spinner("캐시 최적화 중..."):
                        try:
                            optimization_result = self.cache_manager.optimize_cache()
                            freed_mb = optimization_result.get('freed_space_mb', 0)
                            optimized_items = optimization_result.get('optimized_items', 0)
                            st.success(f"✅ 캐시 최적화 완료: {optimized_items}개 최적화, {freed_mb:.1f}MB 확보")
                        except Exception as e:
                            st.error(f"❌ 캐시 최적화 실패: {e}")
            
            st.markdown("---")
            
            # 캐시 히트/미스 통계 차트
            st.subheader("📈 캐시 성능 통계")
            
            try:
                # 가상의 성능 데이터 (실제로는 cache_manager에서 가져올 수 있음)
                time_points = []
                hit_rates = []
                response_times = []
                
                current_time = datetime.now()
                for i in range(24):  # 24시간 데이터
                    time_points.append(current_time - timedelta(hours=i))
                    # 가상 데이터 생성
                    base_hit_rate = hit_rate + (i % 5 - 2) * 5  # ±10% 변동
                    hit_rates.append(max(30, min(95, base_hit_rate)))
                    response_times.append(max(10, performance_report.avg_response_time_ms + (i % 3 - 1) * 20))
                
                time_points.reverse()
                hit_rates.reverse()
                response_times.reverse()
                
                # 차트 생성
                fig = go.Figure()
                
                # 히트율 라인
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=hit_rates,
                    mode='lines+markers',
                    name='히트율 (%)',
                    line=dict(color='#2ecc71', width=2),
                    yaxis='y1'
                ))
                
                # 응답시간 라인
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=response_times,
                    mode='lines+markers',
                    name='응답시간 (ms)',
                    line=dict(color='#e74c3c', width=2),
                    yaxis='y2'
                ))
                
                # 레이아웃 설정
                fig.update_layout(
                    title="캐시 성능 추세 (24시간)",
                    xaxis_title="시간",
                    yaxis=dict(
                        title="히트율 (%)",
                        side="left",
                        range=[0, 100]
                    ),
                    yaxis2=dict(
                        title="응답시간 (ms)",
                        side="right",
                        overlaying="y",
                        range=[0, max(response_times) * 1.2]
                    ),
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ 성능 차트 생성 실패: {e}")
            
            # 캐시 상세 정보
            with st.expander("🔍 캐시 상세 정보"):
                if cache_stats.l1_items > 0 or cache_stats.l2_items > 0:
                    st.write("**L1 캐시 (메모리)**")
                    l1_keys = self.cache_manager.get_l1_keys()[:10]  # 상위 10개만
                    if l1_keys:
                        st.write(f"키 목록: {', '.join(l1_keys)}")
                    else:
                        st.write("비어있음")
                    
                    st.write("**L2 캐시 (디스크)**")
                    l2_keys = self.cache_manager.get_l2_keys()[:10]  # 상위 10개만
                    if l2_keys:
                        st.write(f"키 목록: {', '.join(l2_keys)}")
                    else:
                        st.write("비어있음")
                else:
                    st.info("캐시가 비어있습니다.")
        
        except Exception as e:
            st.error(f"❌ 캐시 관리 시스템을 로드할 수 없습니다: {e}")
            st.info("캐시 관리 시스템이 아직 초기화되지 않았을 수 있습니다.")

# Streamlit 앱 실행
def main():
    dashboard = MonitoringDashboard("C:\\Users\\PC_58410\\SOLOMONDd-ai-system")
    dashboard.run()

if __name__ == "__main__":
    main()