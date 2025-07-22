#!/usr/bin/env python3
"""
간단한 메시지 추출 테스트 UI
대용량 파일 없이 텍스트만으로 빠른 검증
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="빠른 메시지 추출 테스트",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 빠른 메시지 추출 테스트")
st.markdown("대용량 파일 업로드 없이 텍스트만으로 종합 메시지 추출 기능을 테스트합니다.")

# 사이드바에 테스트 시나리오
st.sidebar.title("📋 테스트 시나리오")

test_scenarios = {
    "주얼리 구매 상담": """안녕하세요. 다이아몬드 반지를 찾고 있어요.
약혼반지로 쓸 건데 1캐럿 정도로 생각하고 있습니다.
GIA 인증서 있는 걸로요. 가격이 얼마나 할까요?
할인도 가능한지 궁금해요.""",
    
    "고객 고민 상담": """목걸이를 사고 싶은데 어떤 걸 선택해야 할지 모르겠어요.
선물용인데 상대방이 좋아할지 걱정이에요.
예산은 50만원 정도 생각하고 있어요.
추천해주실 수 있나요?""",
    
    "비교 검토 상담": """다른 매장에서 본 반지와 비교해보고 싶어요.
그쪽은 18K 골드에 0.5캐럿 다이아몬드였는데
여기서는 어떤 옵션이 있나요?
가격 차이도 알고 싶어요.""",
    
    "가격 협상": """이 반지 마음에 드는데 가격이 좀 부담스러워요.
할인 가능한 폭이 어느 정도인지 궁금해요.
현금 결제하면 더 할인 받을 수 있나요?"""
}

selected_scenario = st.sidebar.selectbox(
    "시나리오 선택:",
    list(test_scenarios.keys())
)

if st.sidebar.button("시나리오 불러오기"):
    st.session_state.test_text = test_scenarios[selected_scenario]

# 메인 영역
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 테스트 텍스트")
    test_text = st.text_area(
        "분석할 텍스트를 입력하세요:",
        value=st.session_state.get('test_text', test_scenarios[selected_scenario]),
        height=200,
        key="text_input"
    )
    
    if st.button("🎯 메시지 추출 테스트", type="primary"):
        if test_text.strip():
            with st.spinner("분석 중..."):
                try:
                    from core.comprehensive_message_extractor import extract_comprehensive_messages
                    
                    result = extract_comprehensive_messages(test_text)
                    st.session_state.analysis_result = result
                    
                except Exception as e:
                    st.error(f"분석 실패: {e}")

with col2:
    st.subheader("📊 분석 결과")
    
    if 'analysis_result' in st.session_state:
        result = st.session_state.analysis_result
        
        if result.get('status') == 'success':
            st.success("✅ 분석 성공!")
            
            main_summary = result.get('main_summary', {})
            
            # 핵심 메시지
            if main_summary.get('one_line_summary'):
                st.markdown("### 📢 핵심 메시지")
                st.info(main_summary['one_line_summary'])
            
            # 고객 상태 및 긴급도
            col_status, col_urgency = st.columns(2)
            
            with col_status:
                if main_summary.get('customer_status'):
                    st.markdown("**👤 고객 상태**")
                    st.write(main_summary['customer_status'])
            
            with col_urgency:
                if main_summary.get('urgency_indicator'):
                    st.markdown("**⚡ 긴급도**")
                    urgency_colors = {'높음': '🔴', '보통': '🟡', '낮음': '🟢'}
                    urgency_emoji = urgency_colors.get(main_summary['urgency_indicator'], '⚪')
                    st.write(f"{urgency_emoji} {main_summary['urgency_indicator']}")
            
            # 주요 포인트
            if main_summary.get('key_points'):
                st.markdown("### 🔍 주요 포인트")
                for point in main_summary['key_points']:
                    st.markdown(f"• {point}")
            
            # 추천 액션
            if main_summary.get('recommended_actions'):
                st.markdown("### 💼 추천 액션")
                for action in main_summary['recommended_actions']:
                    st.markdown(f"{action}")
            
            # 신뢰도
            if main_summary.get('confidence_score'):
                st.markdown("### 📊 신뢰도")
                confidence = main_summary['confidence_score']
                st.progress(confidence)
                st.write(f"{confidence*100:.0f}%")
            
            # 상세 분석 (접을 수 있게)
            with st.expander("🔬 상세 분석 결과"):
                st.json(result)
        
        else:
            st.error(f"❌ 분석 실패: {result.get('error', 'Unknown error')}")
    
    else:
        st.info("👈 왼쪽에서 텍스트를 입력하고 분석 버튼을 눌러주세요.")

# 하단 정보
st.markdown("---")
st.markdown("💡 **사용법**: 왼쪽에서 테스트 시나리오를 선택하거나 직접 텍스트를 입력한 후 분석 버튼을 클릭하세요.")
st.markdown("🎯 **목적**: 대용량 파일 업로드 없이 '이 사람들이 무엇을 말하는지' 분석 기능을 빠르게 검증할 수 있습니다.")