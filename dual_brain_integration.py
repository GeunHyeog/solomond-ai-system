#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠🧠 듀얼 브레인 통합 시스템
Dual Brain Integration System for SOLOMOND AI

핵심 기능:
1. 분석 → 캘린더 자동 연동
2. 캘린더 → AI 인사이트 생성
3. AI 인사이트 → 미래 계획 제안
4. 전체 시스템 통합 관리

워크플로우:
컨퍼런스 분석 완료 → 자동으로 구글 캘린더에 이벤트 생성 → AI가 패턴 분석 → 미래 계획 제안
"""

import os
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import streamlit as st

# 모듈 임포트
sys.path.append(os.path.dirname(__file__))
try:
    from google_calendar_connector import GoogleCalendarConnector
    from ai_insights_engine import AIInsightsEngine
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"❌ 필수 모듈 로드 실패: {e}")

class DualBrainSystem:
    """솔로몬드 AI 듀얼 브레인 통합 시스템"""
    
    def __init__(self):
        """시스템 초기화"""
        self.calendar_connector = GoogleCalendarConnector() if MODULES_AVAILABLE else None
        self.insights_engine = AIInsightsEngine() if MODULES_AVAILABLE else None
        self.integration_log = []
        
    def process_analysis_completion(self, analysis_data: Dict[str, Any]) -> bool:
        """분석 완료 시 전체 워크플로우 실행"""
        try:
            st.info("🧠 듀얼 브레인 시스템 활성화...")
            
            # 1단계: 분석 결과를 캘린더에 저장
            calendar_success = self._sync_to_calendar(analysis_data)
            
            # 2단계: AI 인사이트 생성
            insights_success = self._generate_insights()
            
            # 3단계: 미래 계획 제안
            planning_success = self._suggest_future_plans()
            
            # 통합 결과 기록
            integration_result = {
                "timestamp": datetime.now().isoformat(),
                "analysis_id": analysis_data.get("analysis_id"),
                "calendar_sync": calendar_success,
                "insights_generated": insights_success,
                "planning_completed": planning_success,
                "overall_success": all([calendar_success, insights_success, planning_success])
            }
            
            self.integration_log.append(integration_result)
            self._save_integration_log()
            
            if integration_result["overall_success"]:
                st.success("🎉 듀얼 브레인 시스템 완전 활성화!")
                st.balloons()
            else:
                st.warning("⚠️ 일부 단계에서 문제가 발생했습니다.")
                
            return integration_result["overall_success"]
            
        except Exception as e:
            st.error(f"❌ 듀얼 브레인 통합 오류: {e}")
            return False
    
    def _sync_to_calendar(self, analysis_data: Dict[str, Any]) -> bool:
        """1단계: 분석 결과를 구글 캘린더에 동기화 (사용자 확인 후)"""
        if not self.calendar_connector:
            st.info("📅 캘린더 연동을 위해서는 별도 설정이 필요합니다")
            return False
            
        try:
            # 사용자에게 캘린더 동기화 확인
            st.subheader("📅 구글 캘린더 동기화")
            
            conference_name = analysis_data.get("pre_info", {}).get("conference_name", "Unknown")
            success_rate = f"{analysis_data['success_count']}/{analysis_data['total_files']}"
            
            st.info(f"""
            **분석 완료**: {conference_name}
            **성공률**: {success_rate} ({analysis_data['success_count']/analysis_data['total_files']*100:.1f}%)
            """)
            
            # 사용자 선택 확인
            if st.button("📅 이 분석을 구글 캘린더에 저장하시겠습니까?", key="calendar_confirm"):
                with st.spinner("구글 캘린더에 이벤트 생성 중..."):
                    # 캘린더 연결 및 인증
                    if not self.calendar_connector.setup_credentials():
                        st.warning("⚠️ 구글 API 자격 증명이 필요합니다")
                        return False
                        
                    if not self.calendar_connector.authenticate():
                        st.warning("⚠️ 구글 계정 인증이 필요합니다")
                        return False
                    
                    # 캘린더 이벤트 생성
                    success = self.calendar_connector.create_analysis_event(analysis_data)
                    
                    if success:
                        st.success("✅ 구글 캘린더에 이벤트가 성공적으로 생성되었습니다!")
                        return True
                    else:
                        st.error("❌ 캘린더 이벤트 생성에 실패했습니다")
                        return False
            else:
                st.info("💡 캘린더 동기화는 선택사항입니다. 나중에 메인 대시보드에서도 설정할 수 있습니다.")
                return False  # 사용자가 선택하지 않음
                
        except Exception as e:
            st.error(f"캘린더 동기화 오류: {e}")
            return False
    
    def _generate_insights(self) -> bool:
        """2단계: AI 인사이트 생성"""
        if not self.insights_engine:
            st.warning("⚠️ AI 인사이트 엔진이 필요합니다")
            return False
            
        try:
            st.info("🧠 AI 인사이트 생성 중...")
            
            # 종합 인사이트 생성
            insights = self.insights_engine.generate_comprehensive_insights()
            
            if insights and insights["metadata"]["total_analyses"] > 0:
                st.success("✅ AI 인사이트 생성 완료!")
                
                # 주요 인사이트 미리보기
                with st.expander("🔍 생성된 인사이트 미리보기"):
                    st.write(f"**인사이트 성숙도**: {insights['metadata']['insight_maturity']*100:.0f}%")
                    st.write(f"**종합 점수**: {insights['metadata']['overall_score']:.0f}/100")
                    st.write(f"**요약**: {insights['summary']}")
                
                return True
            else:
                st.info("📊 더 많은 분석 데이터가 축적되면 고급 인사이트를 제공합니다")
                return True
                
        except Exception as e:
            st.error(f"AI 인사이트 생성 오류: {e}")
            return False
    
    def _suggest_future_plans(self) -> bool:
        """3단계: 미래 계획 제안"""
        try:
            st.info("🚀 미래 계획 생성 중...")
            
            if not self.insights_engine:
                return False
            
            # 미래 트렌드 예측
            predictions = self.insights_engine.predict_future_trends()
            recommendations = self.insights_engine.generate_personalized_recommendations()
            
            if predictions or recommendations:
                st.success("✅ 미래 계획 제안 완료!")
                
                # 미래 계획 미리보기
                with st.expander("🔮 미래 계획 미리보기"):
                    if predictions:
                        st.write("**예측 항목**:")
                        for pred in predictions[:2]:  # 상위 2개만 표시
                            st.write(f"• {pred['title']}: {pred['description']}")
                    
                    if recommendations:
                        st.write("**추천 사항**:")
                        for rec in recommendations[:2]:  # 상위 2개만 표시
                            st.write(f"• {rec['title']}: {rec['description']}")
                
                return True
            else:
                st.info("📈 더 많은 데이터 축적 후 정확한 미래 계획을 제공합니다")
                return True
                
        except Exception as e:
            st.error(f"미래 계획 생성 오류: {e}")
            return False
    
    def _save_integration_log(self):
        """통합 로그 저장"""
        try:
            log_dir = Path("analysis_history")
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / "dual_brain_integration.json"
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "integration_history": self.integration_log,
                    "last_update": datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            st.warning(f"통합 로그 저장 실패: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """듀얼 브레인 시스템 상태 확인"""
        status = {
            "modules_available": MODULES_AVAILABLE,
            "calendar_ready": self.calendar_connector is not None,
            "insights_ready": self.insights_engine is not None,
            "integration_count": len(self.integration_log),
            "last_integration": None
        }
        
        if self.integration_log:
            status["last_integration"] = self.integration_log[-1]["timestamp"]
        
        return status
    
    def render_system_dashboard(self):
        """시스템 대시보드 렌더링"""
        st.header("🧠🧠 듀얼 브레인 시스템 대시보드")
        
        # 시스템 상태
        status = self.get_system_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_icon = "🟢" if status["modules_available"] else "🔴"
            st.metric("모듈 상태", f"{status_icon} {'정상' if status['modules_available'] else '오류'}")
        
        with col2:
            calendar_icon = "🟢" if status["calendar_ready"] else "🟡"
            st.metric("캘린더 연동", f"{calendar_icon} {'준비됨' if status['calendar_ready'] else '설정필요'}")
        
        with col3:
            insights_icon = "🟢" if status["insights_ready"] else "🟡"
            st.metric("AI 인사이트", f"{insights_icon} {'활성화' if status['insights_ready'] else '대기중'}")
        
        with col4:
            st.metric("통합 실행 횟수", status["integration_count"])
        
        # 최근 통합 이력
        if self.integration_log:
            st.subheader("📋 최근 통합 이력")
            
            for log_entry in self.integration_log[-5:]:  # 최근 5개
                with st.expander(f"🔄 {log_entry['timestamp']} - {log_entry['analysis_id']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        calendar_status = "✅" if log_entry["calendar_sync"] else "❌"
                        st.write(f"캘린더 동기화: {calendar_status}")
                    
                    with col2:
                        insights_status = "✅" if log_entry["insights_generated"] else "❌"
                        st.write(f"인사이트 생성: {insights_status}")
                    
                    with col3:
                        planning_status = "✅" if log_entry["planning_completed"] else "❌"
                        st.write(f"계획 제안: {planning_status}")

# 스트림릿 인터페이스
def main():
    """메인 인터페이스"""
    st.set_page_config(
        page_title="듀얼 브레인 통합",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠🧠 솔로몬드 AI 듀얼 브레인 통합 시스템")
    st.markdown("**분석 → 캘린더 → AI 인사이트 → 미래 계획의 완전 자동화**")
    
    if not MODULES_AVAILABLE:
        st.error("❌ 필수 모듈이 로드되지 않았습니다. 시스템을 다시 시작해주세요.")
        return
    
    # 듀얼 브레인 시스템 초기화
    if 'dual_brain' not in st.session_state:
        st.session_state.dual_brain = DualBrainSystem()
    
    dual_brain = st.session_state.dual_brain
    
    # 대시보드 렌더링
    dual_brain.render_system_dashboard()
    
    # 테스트 실행
    st.subheader("🧪 시스템 테스트")
    
    if st.button("🔄 듀얼 브레인 시스템 테스트 실행"):
        # 테스트용 분석 데이터
        test_analysis = {
            "analysis_id": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "pre_info": {
                "conference_name": "듀얼 브레인 시스템 테스트",
                "conference_location": "테스트 환경",
                "industry_field": "AI/기술"
            },
            "total_files": 3,
            "success_count": 3,
            "file_types": ["test", "integration"]
        }
        
        # 전체 워크플로우 실행
        success = dual_brain.process_analysis_completion(test_analysis)
        
        if success:
            st.success("🎉 듀얼 브레인 시스템이 성공적으로 작동했습니다!")
        else:
            st.error("❌ 일부 단계에서 문제가 발생했습니다.")

if __name__ == "__main__":
    main()