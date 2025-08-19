#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 홀리스틱 컨퍼런스 마스터 시스템 (Supabase 지원)
Holistic Conference Master System with Supabase Support - SOLOMOND AI v7.0
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time
import os

# 데이터베이스 어댑터 임포트
from database_adapter import DatabaseFactory, DatabaseInterface

# 모든 하위 시스템 임포트
try:
    from holistic_conference_analyzer_supabase import HolisticConferenceAnalyzerSupabase
    from semantic_connection_engine import SemanticConnectionEngine
    from conference_story_generator import ConferenceStoryGenerator
    from actionable_insights_extractor import ActionableInsightsExtractor
    SUBSYSTEMS_AVAILABLE = True
except ImportError as e:
    st.error(f"하위 시스템 임포트 실패: {e}")
    SUBSYSTEMS_AVAILABLE = False

class HolisticConferenceMasterSupabase:
    """홀리스틱 컨퍼런스 마스터 시스템 (Supabase 지원)"""
    
    def __init__(self, conference_name: str = "default", db_type: str = "auto"):
        self.conference_name = conference_name
        self.db_type = db_type
        
        # 데이터베이스 초기화
        self.db: DatabaseInterface = DatabaseFactory.create_database(db_type, conference_name)
        
        # 하위 시스템 초기화
        if SUBSYSTEMS_AVAILABLE:
            self.analyzer = HolisticConferenceAnalyzerSupabase(conference_name, db_type)
            # 다른 시스템들은 SQLite를 사용하되, 필요시 Supabase 어댑터로 연결
            self.connector = SemanticConnectionEngine(conference_name)
            self.story_generator = ConferenceStoryGenerator(conference_name)
            self.insights_extractor = ActionableInsightsExtractor(conference_name)
        
        # 분석 결과 저장
        self.analysis_results = {}
        self.full_report = {}
    
    def check_data_availability(self) -> Dict[str, Any]:
        """데이터 가용성 확인 (Supabase 지원)"""
        try:
            db_status = self.analyzer.check_database_connection()
            
            if not db_status["connected"]:
                return {
                    "available": False, 
                    "message": f"{db_status['database_type']} 연결 실패",
                    "database_type": db_status['database_type']
                }
            
            fragment_count = db_status["fragment_count"]
            
            if fragment_count == 0:
                return {
                    "available": False, 
                    "message": "분석된 조각이 없습니다. 샘플 데이터를 생성합니다.",
                    "database_type": db_status['database_type']
                }
            
            return {
                "available": True,
                "fragment_count": fragment_count,
                "message": f"{fragment_count}개의 분석 조각이 준비되어 있습니다 ({db_status['database_type']})",
                "database_type": db_status['database_type']
            }
            
        except Exception as e:
            return {
                "available": False, 
                "message": f"데이터 확인 중 오류: {e}",
                "database_type": "unknown"
            }
    
    def create_sample_data_if_needed(self) -> bool:
        """필요시 샘플 데이터 생성"""
        try:
            return self.analyzer.create_sample_data_if_empty()
        except Exception as e:
            st.error(f"샘플 데이터 생성 실패: {e}")
            return False
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """완전한 홀리스틱 분석 실행 (Supabase 지원)"""
        if not SUBSYSTEMS_AVAILABLE:
            return {"error": "하위 시스템을 사용할 수 없습니다."}
        
        results = {
            "conference_name": self.conference_name,
            "database_type": self.db_type,
            "analysis_timestamp": datetime.now().isoformat(),
            "stages": {}
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1단계: 멀티모달 데이터베이스 분석 (Supabase 지원)
            status_text.text("1/4 멀티모달 데이터베이스 분석 중... (Supabase 연동)")
            progress_bar.progress(0.25)
            
            holistic_result = self.analyzer.analyze_conference_holistically()
            if "error" in holistic_result:
                return {"error": f"1단계 실패: {holistic_result['error']}"}
            
            results["stages"]["holistic_analysis"] = holistic_result
            time.sleep(1)
            
            # 2단계: 의미적 연결 분석
            status_text.text("2/4 의미적 연결 분석 중...")
            progress_bar.progress(0.50)
            
            try:
                semantic_result = self.connector.analyze_semantic_connections()
                results["stages"]["semantic_connections"] = semantic_result
            except Exception as e:
                st.warning(f"의미적 연결 분석 건너뜀: {e}")
                results["stages"]["semantic_connections"] = {"warning": "의미적 연결 분석 실패", "error": str(e)}
            
            time.sleep(1)
            
            # 3단계: 컨퍼런스 스토리 생성
            status_text.text("3/4 컨퍼런스 스토리 생성 중...")
            progress_bar.progress(0.75)
            
            try:
                story_narrative = self.story_generator.generate_conference_story()
                results["stages"]["conference_story"] = story_narrative.__dict__ if hasattr(story_narrative, '__dict__') else story_narrative
            except Exception as e:
                st.warning(f"스토리 생성 건너뜀: {e}")
                results["stages"]["conference_story"] = {"warning": "스토리 생성 실패", "error": str(e)}
            
            time.sleep(1)
            
            # 4단계: 실행 가능한 인사이트 추출
            status_text.text("4/4 실행 가능한 인사이트 추출 중...")
            progress_bar.progress(1.0)
            
            try:
                actionable_insights = self.insights_extractor.extract_actionable_insights()
                results["stages"]["actionable_insights"] = actionable_insights.__dict__ if hasattr(actionable_insights, '__dict__') else actionable_insights
            except Exception as e:
                st.warning(f"인사이트 추출 건너뜀: {e}")
                results["stages"]["actionable_insights"] = {"warning": "인사이트 추출 실패", "error": str(e)}
            
            # 완료
            status_text.text("✅ 홀리스틱 분석 완료!")
            self.analysis_results = results
            
            return results
            
        except Exception as e:
            status_text.text(f"❌ 분석 중 오류 발생: {e}")
            return {"error": str(e)}
    
    def generate_executive_summary(self) -> str:
        """경영진 요약 보고서 생성 (Supabase 지원)"""
        if not self.analysis_results:
            return "분석 결과가 없습니다."
        
        summary_parts = []
        
        # 헤더
        db_type = self.analysis_results.get("database_type", "unknown")
        summary_parts.append(f"# {self.conference_name} 홀리스틱 분석 보고서")
        summary_parts.append(f"**분석 일시:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        summary_parts.append(f"**데이터베이스:** {db_type}")
        summary_parts.append("")
        
        # 핵심 요약 (3줄) - 안전한 접근
        if "actionable_insights" in self.analysis_results.get("stages", {}):
            insights = self.analysis_results["stages"]["actionable_insights"]
            three_line = insights.get("three_line_summary", {})
            
            summary_parts.append("## 핵심 요약 (3줄)")
            
            # 객체와 딕셔너리 모두 처리
            if hasattr(three_line, 'line1_what'):
                summary_parts.append(f"**1. 논의 내용:** {three_line.line1_what}")
                summary_parts.append(f"**2. 중요성:** {three_line.line2_why}")
                summary_parts.append(f"**3. 결론:** {three_line.line3_outcome}")
            elif isinstance(three_line, dict):
                summary_parts.append(f"**1. 논의 내용:** {three_line.get('line1_what', '정보 없음')}")
                summary_parts.append(f"**2. 중요성:** {three_line.get('line2_why', '정보 없음')}")
                summary_parts.append(f"**3. 결론:** {three_line.get('line3_outcome', '정보 없음')}")
            else:
                summary_parts.append("**1. 논의 내용:** 정보 없음")
                summary_parts.append("**2. 중요성:** 정보 없음")
                summary_parts.append("**3. 결론:** 정보 없음")
                
            summary_parts.append("")
        
        # 핵심 메트릭 (Supabase 데이터 포함)
        if "holistic_analysis" in self.analysis_results.get("stages", {}):
            holistic = self.analysis_results["stages"]["holistic_analysis"]
            
            summary_parts.append("## 핵심 지표")
            summary_parts.append(f"- **데이터베이스**: {db_type}")
            summary_parts.append(f"- **분석 자료**: {holistic.get('total_fragments', 0)}개")
            summary_parts.append(f"- **주요 개체**: {holistic.get('total_entities', 0)}개")
            summary_parts.append(f"- **논의 주제**: {holistic.get('total_topics', 0)}개")
            summary_parts.append(f"- **평균 신뢰도**: {holistic.get('average_confidence', 0):.1%}")
            summary_parts.append("")
        
        # 5가지 액션 아이템 (안전한 접근)
        if "actionable_insights" in self.analysis_results.get("stages", {}):
            insights = self.analysis_results["stages"]["actionable_insights"]
            action_items = insights.get("action_items", [])
            
            summary_parts.append("## 즉시 실행 아이템 (Top 5)")
            for i, action in enumerate(action_items[:5], 1):
                # 객체와 딕셔너리 모두 처리
                if hasattr(action, 'title'):
                    priority = getattr(action, 'priority', 'medium')
                    title = getattr(action, 'title', '제목 없음')
                    owner = getattr(action, 'owner', '미정')
                    deadline = getattr(action, 'deadline', '미정')
                    description = getattr(action, 'description', '설명 없음')
                else:
                    priority = action.get("priority", "medium")
                    title = action.get('title', '제목 없음')
                    owner = action.get('owner', '미정')
                    deadline = action.get('deadline', '미정')
                    description = action.get('description', '설명 없음')
                
                priority_icon = {"high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}.get(priority, "[NORM]")
                
                summary_parts.append(f"**{i}. {priority_icon} {title}**")
                summary_parts.append(f"   - 담당: {owner}")
                summary_parts.append(f"   - 기한: {deadline}")
                summary_parts.append(f"   - 내용: {description}")
                summary_parts.append("")
        
        # 데이터베이스 정보
        summary_parts.append("## 시스템 정보")
        summary_parts.append(f"- **데이터베이스**: {db_type}")
        summary_parts.append(f"- **컨퍼런스**: {self.conference_name}")
        summary_parts.append("- **분석 엔진**: SOLOMOND AI v7.0 (Supabase 지원)")
        summary_parts.append("")
        
        # 결론
        summary_parts.append("## 결론 및 제안")
        summary_parts.append("본 홀리스틱 분석을 통해 컨퍼런스의 전체적인 맥락과 핵심 사안들이 명확히 파악되었습니다.")
        summary_parts.append("Supabase 클라우드 데이터베이스를 통해 안정적이고 확장 가능한 분석이 수행되었습니다.")
        summary_parts.append("")
        summary_parts.append("---")
        summary_parts.append("*본 보고서는 SOLOMOND AI 홀리스틱 분석 시스템 v7.0으로 자동 생성되었습니다.*")
        
        return "\\n".join(summary_parts)
    
    def export_full_report(self) -> str:
        """전체 상세 보고서 내보내기"""
        if not self.analysis_results:
            return "분석 결과가 없습니다."
        
        # JSON 형태로 전체 결과 저장
        db_type = self.analysis_results.get("database_type", "unknown")
        report_filename = f"holistic_analysis_{db_type}_{self.conference_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = Path(report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, ensure_ascii=False, indent=2, default=str)
        
        return str(report_path)

def main():
    st.set_page_config(
        page_title="홀리스틱 컨퍼런스 마스터 (Supabase)",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 홀리스틱 컨퍼런스 마스터 시스템")
    st.markdown("**SOLOMOND AI v7.0 - Supabase 클라우드 데이터베이스 지원**")
    
    # 사이드바
    st.sidebar.header("⚙️ 시스템 설정")
    conference_name = st.sidebar.text_input("컨퍼런스 이름", "my_conference")
    db_type = st.sidebar.selectbox("데이터베이스 타입", ["auto", "sqlite", "supabase"], 
                                   help="auto: 자동 선택, sqlite: 로컬 파일, supabase: 클라우드 DB")
    
    # 시스템 상태
    st.sidebar.markdown("### 🔧 시스템 상태")
    if SUBSYSTEMS_AVAILABLE:
        st.sidebar.success("✅ 모든 하위 시스템 정상")
    else:
        st.sidebar.error("❌ 하위 시스템 오류")
        return
    
    # 마스터 시스템 초기화
    master = HolisticConferenceMasterSupabase(conference_name, db_type)
    
    # 데이터 가용성 확인
    data_status = master.check_data_availability()
    
    # 데이터베이스 정보 표시
    st.sidebar.markdown(f"**데이터베이스**: {data_status.get('database_type', 'unknown')}")
    
    if not data_status["available"]:
        st.warning(f"⚠️ {data_status['message']}")
        
        # Supabase 설정 안내
        if db_type == "supabase" and data_status.get('database_type') == 'SupabaseAdapter':
            st.info("""
            💡 **Supabase 설정 방법**:
            1. https://supabase.com 에서 프로젝트 생성
            2. 환경변수 설정:
               - SUPABASE_URL: 프로젝트 URL
               - SUPABASE_ANON_KEY: anon/public 키
            3. fragments 테이블 생성 (SQL 실행)
            """)
        
        # 샘플 데이터 생성 버튼
        if st.button("🎲 샘플 데이터 생성"):
            with st.spinner("샘플 데이터를 생성하고 있습니다..."):
                if master.create_sample_data_if_needed():
                    st.success("✅ 샘플 데이터 생성 완료!")
                    st.rerun()
                else:
                    st.error("❌ 샘플 데이터 생성 실패")
        
        # 기본 분석 시스템 바로가기
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 기본 분석 시스템으로 이동"):
                st.markdown("👉 [컨퍼런스 분석 시스템](http://localhost:8501)")
        
        with col2:
            if st.button("🔄 데이터 상태 다시 확인"):
                st.rerun()
        
        return
    
    # 데이터 상태 표시
    st.success(f"✅ {data_status['message']}")
    
    # 메인 분석 인터페이스
    st.markdown("## 🎯 홀리스틱 분석 실행")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        **홀리스틱 분석 과정 (데이터베이스: {data_status.get('database_type', 'unknown')})**:
        1. 🗃️ **멀티모달 데이터베이스** - 모든 파일 내용 통합 분석
        2. 🧠 **의미적 연결 분석** - 파일 간 관계 및 연관성 탐지
        3. 📖 **전체 스토리 생성** - 일관된 컨퍼런스 내러티브 구성
        4. 🎯 **실행 가능한 인사이트** - 3줄 요약 + 5가지 액션 아이템
        """)
    
    with col2:
        st.metric("분석 준비 상태", "완료", "100%")
        st.metric("데이터베이스", data_status.get('database_type', 'unknown'))
    
    # 분석 실행 버튼
    if st.button("🚀 홀리스틱 분석 시작", type="primary", use_container_width=True):
        st.markdown("---")
        
        # 분석 실행
        results = master.run_complete_analysis()
        
        if "error" in results:
            st.error(f"❌ 분석 실패: {results['error']}")
            return
        
        st.success("🎉 홀리스틱 분석 완료!")
        
        # 결과 표시
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 경영진 요약", "🎯 3줄 요약", "✅ 액션 아이템", "📖 컨퍼런스 스토리", "📊 상세 분석"
        ])
        
        with tab1:
            st.markdown("## 📋 경영진 요약 보고서")
            executive_summary = master.generate_executive_summary()
            st.markdown(executive_summary)
            
            # 보고서 다운로드
            st.download_button(
                label="📥 경영진 보고서 다운로드",
                data=executive_summary,
                file_name=f"executive_summary_{results.get('database_type', 'unknown')}_{conference_name}.md",
                mime="text/markdown"
            )
        
        with tab2:
            st.markdown("## 🎯 3줄 핵심 요약")
            
            if "actionable_insights" in results.get("stages", {}):
                insights = results["stages"]["actionable_insights"]
                three_line = insights.get("three_line_summary", {})
                
                if hasattr(three_line, 'line1_what'):
                    st.info(f"**1. 무엇을:** {three_line.line1_what}")
                    st.info(f"**2. 왜 중요:** {three_line.line2_why}")
                    st.info(f"**3. 결론:** {three_line.line3_outcome}")
                    confidence = getattr(three_line, 'confidence', 0)
                    st.metric("요약 신뢰도", f"{confidence:.1%}")
                elif isinstance(three_line, dict):
                    st.info(f"**1. 무엇을:** {three_line.get('line1_what', '정보 없음')}")
                    st.info(f"**2. 왜 중요:** {three_line.get('line2_why', '정보 없음')}")
                    st.info(f"**3. 결론:** {three_line.get('line3_outcome', '정보 없음')}")
                    confidence = three_line.get('confidence', 0)
                    st.metric("요약 신뢰도", f"{confidence:.1%}")
                else:
                    st.warning("3줄 요약 데이터를 찾을 수 없습니다.")
        
        with tab3:
            st.markdown("## ✅ 5가지 핵심 액션 아이템")
            
            if "actionable_insights" in results.get("stages", {}):
                insights = results["stages"]["actionable_insights"]
                action_items = insights.get("action_items", [])
                
                for i, action in enumerate(action_items, 1):
                    # 객체와 딕셔너리 모두 처리
                    if hasattr(action, 'title'):
                        priority = getattr(action, 'priority', 'medium')
                        title = getattr(action, 'title', '제목 없음')
                        owner = getattr(action, 'owner', '미정')
                        deadline = getattr(action, 'deadline', '미정')
                        description = getattr(action, 'description', '설명 없음')
                        success_criteria = getattr(action, 'success_criteria', '기준 없음')
                        dependencies = getattr(action, 'dependencies', [])
                        evidence_source = getattr(action, 'evidence_source', [])
                    else:
                        priority = action.get("priority", "medium")
                        title = action.get('title', '제목 없음')
                        owner = action.get('owner', '미정')
                        deadline = action.get('deadline', '미정')
                        description = action.get('description', '설명 없음')
                        success_criteria = action.get('success_criteria', '기준 없음')
                        dependencies = action.get('dependencies', [])
                        evidence_source = action.get('evidence_source', [])
                    
                    priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                    priority_icon = priority_color.get(priority, "⚪")
                    
                    with st.expander(f"{priority_icon} {i}. {title} ({priority} 우선순위)"):
                        st.markdown(f"**설명:** {description}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**담당자:** {owner}")
                            st.markdown(f"**마감일:** {deadline}")
                        
                        with col2:
                            deps_text = ', '.join(dependencies) if dependencies else '없음'
                            st.markdown(f"**의존성:** {deps_text}")
                            st.markdown(f"**근거 자료:** {len(evidence_source)}개")
                        
                        st.markdown(f"**성공 기준:** {success_criteria}")
        
        with tab4:
            st.markdown("## 📖 컨퍼런스 전체 스토리")
            
            if "conference_story" in results.get("stages", {}):
                story = results["stages"]["conference_story"]
                
                if not isinstance(story, dict) or "warning" in story:
                    st.warning("컨퍼런스 스토리 생성에 실패했습니다.")
                    if "error" in story:
                        st.error(f"오류: {story['error']}")
                else:
                    # 전체 내러티브
                    st.markdown("### 📚 전체 내러티브")
                    st.markdown(story.get("narrative_summary", "스토리를 생성할 수 없습니다."))
                    
                    # 핵심 결과
                    st.markdown("### 🎯 핵심 결과")
                    key_takeaways = story.get("key_takeaways", [])
                    for takeaway in key_takeaways:
                        st.markdown(f"- {takeaway}")
        
        with tab5:
            st.markdown("## 📊 상세 분석 결과")
            
            # 데이터베이스 정보
            st.markdown(f"### 🗃️ 데이터베이스 정보")
            st.markdown(f"**타입**: {results.get('database_type', 'unknown')}")
            st.markdown(f"**분석 시간**: {results.get('analysis_timestamp', 'unknown')}")
            
            # 각 단계별 결과
            stages = results.get("stages", {})
            
            for stage_name, stage_data in stages.items():
                stage_title = {
                    "holistic_analysis": "🗃️ 멀티모달 데이터베이스 분석",
                    "semantic_connections": "🧠 의미적 연결 분석",
                    "conference_story": "📖 컨퍼런스 스토리",
                    "actionable_insights": "🎯 실행 가능한 인사이트"
                }.get(stage_name, stage_name)
                
                with st.expander(stage_title):
                    if isinstance(stage_data, dict):
                        if "error" in stage_data:
                            st.error(f"오류: {stage_data['error']}")
                        elif "warning" in stage_data:
                            st.warning(f"경고: {stage_data['warning']}")
                        else:
                            st.json(stage_data)
                    else:
                        st.write(stage_data)
        
        # 전체 보고서 내보내기
        st.markdown("---")
        if st.button("📥 전체 상세 보고서 내보내기"):
            report_path = master.export_full_report()
            st.success(f"✅ 상세 보고서가 저장되었습니다: {report_path}")
    
    # 사용법 안내
    st.markdown("---")
    st.markdown("### 💡 사용법")
    st.markdown(f"""
    1. **데이터베이스 선택**: {data_status.get('database_type', 'unknown')} 사용 중
    2. **데이터 준비**: 기본 컨퍼런스 분석(포트 8501)에서 파일들을 분석하거나 샘플 데이터 생성
    3. **홀리스틱 분석**: 위의 "홀리스틱 분석 시작" 버튼을 클릭
    4. **결과 확인**: 5개 탭에서 다양한 관점의 분석 결과를 확인
    5. **보고서 활용**: 경영진 요약이나 상세 보고서를 다운로드하여 활용
    """)

if __name__ == "__main__":
    main()