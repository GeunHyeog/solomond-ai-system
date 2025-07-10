#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
솔로몬드 AI v2.1.1 - 베타 테스트 피드백 수집 및 분석 시스템
한국보석협회 회원사 대상 구조화된 피드백 수집, 실시간 분석, 개선 워크플로우

작성자: 전근혁 (솔로몬드 대표, 한국보석협회 사무국장)
생성일: 2025.07.11
목적: 현장 피드백 기반 지속적 개선

실행 방법:
streamlit run test_environment/feedback_collection_system_v211.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
import logging
from typing import Dict, List, Optional
import uuid

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeedbackCollectionSystem:
    """베타 테스트 피드백 수집 및 분석 시스템"""
    
    def __init__(self, db_path="feedback_v211.db"):
        self.db_path = db_path
        self.init_database()
        
        # 회원사 정보
        self.company_categories = {
            "large_enterprise": "대기업",
            "medium_enterprise": "중견기업", 
            "small_specialist": "소규모 전문업체"
        }
        
        # 평가 영역
        self.evaluation_areas = {
            "technical": "기술적 성능",
            "usability": "사용성",
            "business_value": "비즈니스 가치",
            "satisfaction": "전반적 만족도"
        }
        
        logger.info("🔄 피드백 수집 시스템 초기화 완료")
    
    def init_database(self):
        """SQLite 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 피드백 테이블 생성
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                company_category TEXT NOT NULL,
                user_name TEXT NOT NULL,
                user_role TEXT NOT NULL,
                submission_date TEXT NOT NULL,
                test_day INTEGER NOT NULL,
                evaluation_area TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT,
                improvement_suggestion TEXT,
                priority_level TEXT NOT NULL
            )
        """)
        
        # 버그 리포트 테이블 생성
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bug_reports (
                id TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                user_name TEXT NOT NULL,
                report_date TEXT NOT NULL,
                bug_category TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                steps_to_reproduce TEXT,
                expected_behavior TEXT,
                actual_behavior TEXT,
                status TEXT DEFAULT 'open',
                resolution TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("📊 피드백 데이터베이스 초기화 완료")
    
    def create_feedback_form(self):
        """Streamlit 피드백 폼 생성"""
        st.title("🎯 솔로몬드 AI v2.1.1 베타 테스트 피드백")
        st.markdown("---")
        
        # 기본 정보 입력
        with st.expander("🏢 기본 정보", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                company_name = st.text_input("회사명", placeholder="예: 대원주얼리그룹")
                company_category = st.selectbox(
                    "회사 규모", 
                    options=list(self.company_categories.keys()),
                    format_func=lambda x: self.company_categories[x]
                )
            
            with col2:
                user_name = st.text_input("이름", placeholder="홍길동")
                user_role = st.selectbox(
                    "직책", 
                    ["CEO/대표", "임원", "팀장", "실무진", "기타"]
                )
            
            test_day = st.slider("테스트 진행 일차", 1, 14, 1)
        
        # 기능별 평가
        st.markdown("## 📊 기능별 상세 평가")
        
        feedback_data = []
        
        # 주요 기능 목록
        features = {
            "technical": {
                "음성 인식 정확도": "STT 정확도 및 주얼리 전문용어 인식",
                "다국어 지원": "한/영/중/일 언어 자동 감지 및 번역",
                "품질 분석": "음성/이미지/문서 품질 실시간 검증",
                "문서 OCR": "PDF, 이미지 텍스트 추출 정확도",
                "처리 속도": "파일 분석 및 결과 생성 속도"
            },
            "usability": {
                "인터페이스 직관성": "UI/UX 사용 편의성",
                "학습 용이성": "처음 사용시 학습 시간",
                "기능 접근성": "원하는 기능 찾기 용이성",
                "오류 처리": "문제 발생시 해결 가이드",
                "도움말 품질": "매뉴얼 및 도움말 유용성"
            },
            "business_value": {
                "업무 효율성": "기존 대비 업무 시간 단축",
                "정보 정확도": "분석 결과의 신뢰성",
                "ROI 기대치": "투자 대비 기대 효과",
                "경쟁력 향상": "업계 내 차별화 효과",
                "확장 가능성": "추가 업무 적용 가능성"
            }
        }
        
        for area, area_features in features.items():
            st.markdown(f"### {self.evaluation_areas[area]}")
            
            for feature_key, feature_desc in area_features.items():
                with st.expander(f"🔍 {feature_key}", expanded=False):
                    st.markdown(f"**평가 항목**: {feature_desc}")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        rating = st.select_slider(
                            f"평점",
                            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            value=5,
                            key=f"rating_{area}_{feature_key}"
                        )
                    
                    with col2:
                        priority = st.selectbox(
                            "개선 우선순위",
                            ["낮음", "보통", "높음", "매우 높음"],
                            key=f"priority_{area}_{feature_key}"
                        )
                    
                    comment = st.text_area(
                        "상세 의견",
                        placeholder="좋았던 점, 아쉬운 점, 구체적인 경험을 자유롭게 작성해주세요.",
                        key=f"comment_{area}_{feature_key}"
                    )
                    
                    improvement = st.text_area(
                        "개선 제안사항",
                        placeholder="어떻게 개선되면 더 좋을지 구체적인 제안을 작성해주세요.",
                        key=f"improvement_{area}_{feature_key}"
                    )
                    
                    if rating or comment or improvement:
                        feedback_data.append({
                            'evaluation_area': area,
                            'feature_name': feature_key,
                            'rating': rating,
                            'comment': comment,
                            'improvement_suggestion': improvement,
                            'priority_level': priority
                        })
        
        # 전반적 평가
        st.markdown("## 🎯 전반적 평가")
        
        col1, col2 = st.columns(2)
        
        with col1:
            overall_satisfaction = st.select_slider(
                "전체 만족도 (10점 만점)",
                options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                value=7
            )
            
            recommendation_score = st.select_slider(
                "추천 의향 (10점 만점)",
                options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                value=7
            )
        
        with col2:
            purchase_intention = st.selectbox(
                "구매 의향",
                ["매우 낮음", "낮음", "보통", "높음", "매우 높음"]
            )
            
            expected_roi = st.selectbox(
                "기대 ROI",
                ["손실 예상", "손익분기", "소폭 이익", "상당한 이익", "매우 높은 이익"]
            )
        
        # 종합 의견
        overall_comment = st.text_area(
            "종합 의견 및 제안사항",
            placeholder="전반적인 사용 경험, 개선사항, 추가 기능 요청 등을 자유롭게 작성해주세요.",
            height=150
        )
        
        # 제출 버튼
        if st.button("📤 피드백 제출", type="primary"):
            if company_name and user_name:
                # 전반적 평가 추가
                feedback_data.append({
                    'evaluation_area': 'satisfaction',
                    'feature_name': 'overall_satisfaction',
                    'rating': overall_satisfaction,
                    'comment': overall_comment,
                    'improvement_suggestion': f"추천의향: {recommendation_score}, 구매의향: {purchase_intention}, 기대ROI: {expected_roi}",
                    'priority_level': '높음'
                })
                
                # 데이터베이스 저장
                success_count = 0
                for feedback in feedback_data:
                    if self.save_feedback(
                        company_name, company_category, user_name, user_role,
                        test_day, feedback['evaluation_area'], feedback['feature_name'],
                        feedback['rating'], feedback['comment'], 
                        feedback['improvement_suggestion'], feedback['priority_level']
                    ):
                        success_count += 1
                
                st.success(f"✅ 피드백 {success_count}개 항목이 성공적으로 저장되었습니다!")
                st.balloons()
                
                # 즉시 분석 결과 표시
                self.show_immediate_analysis(company_name, test_day)
                
            else:
                st.error("❌ 회사명과 이름은 필수 입력 항목입니다.")

def main():
    """메인 애플리케이션"""
    st.set_page_config(
        page_title="솔로몬드 AI v2.1.1 피드백 시스템",
        page_icon="💎",
        layout="wide"
    )
    
    # 사이드바 메뉴
    st.sidebar.title("💎 솔로몬드 AI v2.1.1")
    st.sidebar.markdown("---")
    
    menu = st.sidebar.selectbox(
        "메뉴 선택",
        [
            "📝 피드백 제출",
            "🐛 버그 리포트", 
            "📊 분석 대시보드",
            "🔄 개선 추적"
        ]
    )
    
    # 피드백 시스템 초기화
    feedback_system = FeedbackCollectionSystem()
    
    # 메뉴별 페이지 표시
    if menu == "📝 피드백 제출":
        feedback_system.create_feedback_form()
    
    # 푸터
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📞 지원 연락처
    **전근혁 대표**  
    📧 solomond.jgh@gmail.com  
    📱 010-2983-0338
    
    **한국보석협회**  
    📍 서울 종로구
    """)

if __name__ == "__main__":
    main()
