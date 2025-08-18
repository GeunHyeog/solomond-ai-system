#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 SOLOMOND AI Serena 에이전트 대시보드
Streamlit 기반 코드 분석 및 최적화 인터페이스

주요 기능:
1. 실시간 코드 분석 및 최적화 제안
2. Symbol-level 코드 검색 및 편집
3. 프로젝트 건강도 모니터링
4. SOLOMOND AI 특화 이슈 탐지
5. 자동 최적화 계획 생성
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime
import time

# Serena 에이전트 임포트
try:
    from solomond_serena_agent import SerenaIntegrationEngine, SerenaCodeAnalyzer
    SERENA_AVAILABLE = True
except ImportError as e:
    SERENA_AVAILABLE = False
    st.error(f"Serena 에이전트 로드 실패: {e}")

# 페이지 설정
st.set_page_config(
    page_title="SOLOMOND AI Serena Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SerenaDashboard:
    """Serena 에이전트 대시보드"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        if SERENA_AVAILABLE:
            self.engine = SerenaIntegrationEngine(str(self.project_root))
            self.analyzer = self.engine.analyzer
        else:
            self.engine = None
            self.analyzer = None
        
        self.init_session_state()
    
    def init_session_state(self):
        """세션 상태 초기화"""
        if 'health_report' not in st.session_state:
            st.session_state.health_report = None
        if 'optimization_plan' not in st.session_state:
            st.session_state.optimization_plan = None
        if 'selected_file' not in st.session_state:
            st.session_state.selected_file = None
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []

    def render_header(self):
        """헤더 렌더링"""
        st.title("🧠 SOLOMOND AI Serena 코딩 에이전트")
        st.markdown("**Symbol-level 코드 분석 및 지능형 최적화 시스템**")
        
        if not SERENA_AVAILABLE:
            st.error("❌ Serena 에이전트가 로드되지 않았습니다. 시스템을 확인해주세요.")
            return False
        
        # 상태 표시
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("프로젝트", "SOLOMOND AI", "활성")
        with col2:
            st.metric("분석 엔진", "Serena v1.0", "준비")
        with col3:
            total_files = len(list(self.project_root.glob("**/*.py")))
            st.metric("Python 파일", total_files, "개")
        with col4:
            if st.session_state.health_report:
                score = st.session_state.health_report.get('overall_score', 0)
                st.metric("건강도", f"{score:.1f}/100", "점")
            else:
                st.metric("건강도", "미분석", "-")
        
        return True

    def render_sidebar(self):
        """사이드바 렌더링"""
        st.sidebar.title("🔧 Serena 제어판")
        
        # 빠른 작업
        st.sidebar.subheader("빠른 분석")
        if st.sidebar.button("🚀 전체 프로젝트 분석", type="primary"):
            self.run_full_analysis()
        
        if st.sidebar.button("📊 건강도 체크"):
            self.run_health_check()
        
        if st.sidebar.button("🎯 최적화 계획 생성"):
            self.generate_optimization_plan()
        
        # 파일 선택
        st.sidebar.subheader("📁 파일 분석")
        python_files = list(self.project_root.glob("**/*.py"))
        python_files = [f for f in python_files if not any(skip in str(f) for skip in ['venv', '__pycache__', '.git'])]
        
        if python_files:
            file_names = [f.name for f in python_files]
            selected_idx = st.sidebar.selectbox(
                "분석할 파일 선택:",
                range(len(file_names)),
                format_func=lambda x: file_names[x]
            )
            st.session_state.selected_file = python_files[selected_idx]
            
            if st.sidebar.button("🔍 선택된 파일 분석"):
                self.analyze_selected_file()
        
        # Serena 설정
        st.sidebar.subheader("⚙️ Serena 설정")
        max_tokens = st.sidebar.slider("최대 토큰 수", 1000, 5000, 2000)
        show_low_priority = st.sidebar.checkbox("낮은 우선순위 이슈 표시", False)
        auto_save_memory = st.sidebar.checkbox("자동 메모리 저장", True)
        
        return {
            'max_tokens': max_tokens,
            'show_low_priority': show_low_priority,
            'auto_save_memory': auto_save_memory
        }

    def run_full_analysis(self):
        """전체 프로젝트 분석 실행"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. 건강도 분석
            status_text.text("1/3 프로젝트 건강도 분석 중...")
            progress_bar.progress(33)
            health_report = self.engine.analyze_project_health()
            st.session_state.health_report = health_report
            
            # 2. 최적화 계획 생성
            status_text.text("2/3 최적화 계획 생성 중...")
            progress_bar.progress(66)
            optimization_plan = self.engine.generate_optimization_plan()
            st.session_state.optimization_plan = optimization_plan
            
            # 3. 메모리 업데이트
            status_text.text("3/3 프로젝트 메모리 업데이트 중...")
            progress_bar.progress(100)
            
            # 주요 파일들 메모리 업데이트
            important_files = [
                'conference_analysis_COMPLETE_WORKING.py',
                'hybrid_compute_manager.py',
                'solomond_ai_main_dashboard.py'
            ]
            
            for file_name in important_files:
                file_path = self.project_root / file_name
                if file_path.exists():
                    self.analyzer.update_project_memory(str(file_path))
            
            # 분석 기록 저장
            analysis_record = {
                'timestamp': datetime.now().isoformat(),
                'type': 'full_analysis',
                'health_score': health_report.get('overall_score', 0),
                'critical_issues': health_report.get('critical_issues', 0),
                'files_analyzed': health_report.get('files_analyzed', 0)
            }
            st.session_state.analysis_history.append(analysis_record)
            
            status_text.text("✅ 전체 분석 완료!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            
            st.success("🎉 전체 프로젝트 분석이 완료되었습니다!")
            st.rerun()
            
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
            status_text.empty()
            progress_bar.empty()

    def run_health_check(self):
        """건강도 체크 실행"""
        with st.spinner("건강도 분석 중..."):
            health_report = self.engine.analyze_project_health()
            st.session_state.health_report = health_report
        
        st.success("건강도 분석 완료!")
        st.rerun()

    def generate_optimization_plan(self):
        """최적화 계획 생성"""
        with st.spinner("최적화 계획 생성 중..."):
            optimization_plan = self.engine.generate_optimization_plan()
            st.session_state.optimization_plan = optimization_plan
        
        st.success("최적화 계획 생성 완료!")
        st.rerun()

    def analyze_selected_file(self):
        """선택된 파일 분석"""
        if not st.session_state.selected_file:
            st.warning("파일이 선택되지 않았습니다.")
            return
        
        file_path = st.session_state.selected_file
        
        with st.spinner(f"'{file_path.name}' 분석 중..."):
            # 심볼 분석
            symbols = self.analyzer.analyze_file_symbols(str(file_path))
            
            # 이슈 탐지
            issues = self.analyzer.detect_solomond_issues(str(file_path))
            
            # 최적화 제안
            optimizations = self.analyzer.suggest_optimizations(str(file_path))
            
            # 코드 블록 추출
            code_blocks = self.analyzer.extract_efficient_code_blocks(str(file_path))
            
            # 세션 상태에 저장
            st.session_state.file_analysis = {
                'file_path': str(file_path),
                'symbols': symbols,
                'issues': issues,
                'optimizations': optimizations,
                'code_blocks': code_blocks,
                'timestamp': datetime.now().isoformat()
            }
        
        st.success(f"'{file_path.name}' 분석 완료!")
        st.rerun()

    def render_health_dashboard(self):
        """건강도 대시보드 렌더링"""
        if not st.session_state.health_report:
            st.info("프로젝트 건강도 분석을 실행해주세요.")
            return
        
        health = st.session_state.health_report
        
        st.subheader("📊 프로젝트 건강도 대시보드")
        
        # 메인 메트릭
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            score = health.get('overall_score', 0)
            color = "green" if score >= 80 else "orange" if score >= 60 else "red"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {color}20; border: 2px solid {color};">
                <h2 style="color: {color}; margin: 0;">{score:.1f}/100</h2>
                <p style="margin: 5px 0;">전체 건강도</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("분석된 파일", health.get('files_analyzed', 0), "개")
        
        with col3:
            st.metric("발견된 심볼", health.get('total_symbols', 0), "개")
        
        with col4:
            critical = health.get('critical_issues', 0)
            st.metric("크리티컬 이슈", critical, "개", delta_color="inverse")
        
        # 상세 정보
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🐌 성능 병목점")
            bottlenecks = health.get('performance_bottlenecks', [])
            if bottlenecks:
                for i, bottleneck in enumerate(bottlenecks[:10]):
                    st.write(f"{i+1}. {bottleneck}")
            else:
                st.info("성능 병목점이 발견되지 않았습니다.")
        
        with col2:
            st.subheader("💡 추천사항")
            recommendations = health.get('recommendations', [])
            if recommendations:
                for rec in recommendations:
                    st.write(f"• {rec}")
            else:
                st.info("추가 추천사항이 없습니다.")
        
        # 시간별 트렌드 (분석 기록이 있는 경우)
        if len(st.session_state.analysis_history) > 1:
            st.subheader("📈 건강도 트렌드")
            
            df = pd.DataFrame(st.session_state.analysis_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig = px.line(df, x='timestamp', y='health_score', 
                         title='프로젝트 건강도 변화',
                         labels={'health_score': '건강도 점수', 'timestamp': '시간'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    def render_optimization_dashboard(self):
        """최적화 대시보드 렌더링"""
        if not st.session_state.optimization_plan:
            st.info("최적화 계획을 생성해주세요.")
            return
        
        plan = st.session_state.optimization_plan
        
        st.subheader("🎯 최적화 계획 대시보드")
        
        # 요약 정보
        col1, col2, col3 = st.columns(3)
        
        with col1:
            priority_count = len(plan.get('priority_fixes', []))
            st.metric("우선순위 수정", priority_count, "개")
        
        with col2:
            performance_count = len(plan.get('performance_improvements', []))
            st.metric("성능 개선", performance_count, "개")
        
        with col3:
            total_items = priority_count + performance_count
            st.metric("총 개선 항목", total_items, "개")
        
        # 우선순위 수정사항
        if plan.get('priority_fixes'):
            st.subheader("🚨 우선순위 수정사항")
            
            priority_df = pd.DataFrame([
                {
                    '파일': Path(fix['file']).name,
                    '이슈 유형': fix['issue']['pattern'],
                    '라인': fix['issue']['line'],
                    '예상 시간': fix['estimated_time'],
                    '영향도': fix['impact']
                }
                for fix in plan['priority_fixes']
            ])
            
            st.dataframe(priority_df, use_container_width=True)
        
        # 성능 개선사항
        if plan.get('performance_improvements'):
            st.subheader("📈 성능 개선사항")
            
            performance_df = pd.DataFrame([
                {
                    '파일': Path(imp['file']).name,
                    '함수': imp['function'],
                    '복잡도': imp['complexity'],
                    '제안': imp['suggestion'][:50] + "..." if len(imp['suggestion']) > 50 else imp['suggestion'],
                    '예상 시간': imp['estimated_time']
                }
                for imp in plan['performance_improvements']
            ])
            
            st.dataframe(performance_df, use_container_width=True)
        
        # 예상 효과
        if plan.get('estimated_impact'):
            st.subheader("📊 예상 효과")
            for impact_type, impact_desc in plan['estimated_impact'].items():
                st.write(f"**{impact_type}**: {impact_desc}")

    def render_file_analysis(self):
        """파일 분석 결과 렌더링"""
        if 'file_analysis' not in st.session_state:
            st.info("사이드바에서 파일을 선택하고 분석을 실행해주세요.")
            return
        
        analysis = st.session_state.file_analysis
        file_name = Path(analysis['file_path']).name
        
        st.subheader(f"🔍 파일 분석: {file_name}")
        
        # 탭으로 구분
        tab1, tab2, tab3, tab4 = st.tabs(["심볼", "이슈", "최적화", "코드 블록"])
        
        with tab1:
            st.subheader("🔤 심볼 분석")
            symbols = analysis.get('symbols', [])
            
            if symbols:
                symbol_data = []
                for symbol in symbols:
                    symbol_data.append({
                        '이름': symbol.name,
                        '유형': symbol.type,
                        '시작 라인': symbol.line_start,
                        '종료 라인': symbol.line_end,
                        '복잡도': symbol.complexity,
                        '인덴테이션': symbol.indentation
                    })
                
                df = pd.DataFrame(symbol_data)
                st.dataframe(df, use_container_width=True)
                
                # 심볼 유형별 분포
                type_counts = df['유형'].value_counts()
                fig = px.pie(values=type_counts.values, names=type_counts.index, 
                           title="심볼 유형 분포")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("발견된 심볼이 없습니다.")
        
        with tab2:
            st.subheader("⚠️ 이슈 분석")
            issues = analysis.get('issues', {})
            
            if issues:
                for issue_type, issue_list in issues.items():
                    if issue_list:
                        st.write(f"**{issue_type}** ({len(issue_list)}개)")
                        for issue in issue_list:
                            st.write(f"  • 라인 {issue['line']}: {issue['code']}")
                            st.write(f"    💡 {issue['suggestion']}")
                        st.write("")
            else:
                st.success("발견된 이슈가 없습니다!")
        
        with tab3:
            st.subheader("🚀 최적화 제안")
            optimizations = analysis.get('optimizations', {})
            
            # 고복잡도 함수
            high_complexity = optimizations.get('high_complexity_functions', [])
            if high_complexity:
                st.write("**고복잡도 함수들:**")
                for func in high_complexity:
                    st.write(f"• {func['function']} (복잡도: {func['complexity']}, 라인: {func['line']})")
                    st.write(f"  💡 {func['suggestion']}")
            
            # SOLOMOND 특화 이슈
            solomond_issues = optimizations.get('solomond_specific', [])
            if solomond_issues:
                st.write("**SOLOMOND AI 특화 이슈들:**")
                for issue in solomond_issues:
                    priority_icon = "🔴" if issue['priority'] == 'high' else "🟡"
                    st.write(f"{priority_icon} {issue['type']}: {issue['details']['suggestion']}")
        
        with tab4:
            st.subheader("📝 코드 블록")
            code_blocks = analysis.get('code_blocks', [])
            
            if code_blocks:
                for i, block in enumerate(code_blocks):
                    with st.expander(f"블록 {i+1} (토큰: {block.token_count}, 중요도: {block.importance_score:.2f})"):
                        st.code(block.content, language='python')
                        
                        # 블록 정보
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("토큰 수", block.token_count)
                        with col2:
                            st.metric("중요도", f"{block.importance_score:.2f}")
                        with col3:
                            st.metric("관련성", f"{block.context_relevance:.2f}")
            else:
                st.info("추출된 코드 블록이 없습니다.")

    def render_symbol_search(self):
        """심볼 검색 인터페이스"""
        st.subheader("🔍 심볼 검색")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_term = st.text_input("심볼 이름 검색:", placeholder="예: analyze_file, CompleteWorkingAnalyzer")
        
        with col2:
            symbol_type = st.selectbox("유형 필터:", ["전체", "function", "class", "import"])
        
        if search_term and st.button("🔍 검색"):
            with st.spinner("심볼 검색 중..."):
                type_filter = None if symbol_type == "전체" else symbol_type
                results = self.analyzer.find_symbol(search_term, type_filter)
            
            if results:
                st.success(f"{len(results)}개의 심볼을 찾았습니다!")
                
                for result in results:
                    with st.expander(f"{result.type}: {result.name} ({Path(result.file_path).name})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**파일:** {result.file_path}")
                            st.write(f"**라인:** {result.line_start} - {result.line_end}")
                            st.write(f"**복잡도:** {result.complexity}")
                        
                        with col2:
                            if result.arguments:
                                st.write(f"**인수:** {', '.join(result.arguments)}")
                            if result.docstring:
                                st.write(f"**문서:** {result.docstring[:100]}...")
            else:
                st.warning("검색 결과가 없습니다.")

    def run(self):
        """대시보드 실행"""
        if not self.render_header():
            return
        
        # 사이드바
        settings = self.render_sidebar()
        
        # 메인 컨텐츠
        tab1, tab2, tab3, tab4 = st.tabs(["건강도", "최적화", "파일 분석", "심볼 검색"])
        
        with tab1:
            self.render_health_dashboard()
        
        with tab2:
            self.render_optimization_dashboard()
        
        with tab3:
            self.render_file_analysis()
        
        with tab4:
            self.render_symbol_search()

def main():
    """메인 실행 함수"""
    dashboard = SerenaDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()