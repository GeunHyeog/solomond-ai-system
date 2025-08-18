#!/usr/bin/env python3
"""
🎯 솔로몬드 AI 메인 대시보드 - 모듈 네비게이션
4개 모듈 접근을 위한 간단한 네비게이션 허브

모듈 1: 컨퍼런스 분석 (최신 통합 시스템 - 3가지 모드)
모듈 2: 웹 크롤러
모듈 3: 보석 분석  
모듈 4: 3D CAD 변환
"""

import streamlit as st
import requests
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# v2 Ollama 인터페이스 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))
try:
    from ollama_interface_v2 import advanced_ollama, get_system_info, benchmark_all, ModelTier
except ImportError:
    advanced_ollama = None
    print("⚠️ v2 Ollama 인터페이스를 불러올 수 없습니다.")

# 페이지 설정
st.set_page_config(
    page_title="🎯 솔로몬드 AI 메인 대시보드",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 모듈 정보
MODULES = {
    1: {
        "name": "통합 컨퍼런스 분석 🏆",
        "icon": "🏆",
        "color": "#FFD700", 
        "description": "하나의 시스템, 3가지 모드! 궁극/균형/안전 모드 선택 가능",
        "port": 8550,
        "key_features": ["3가지 모드 통합", "UI 모드 선택", "터보 업로드", "네트워크 안정", "스마트 캐시", "GPU 가속", "10GB 지원", "화자 분리"],
        "is_main": True,
        "is_unified": True
    },
    2: {
        "name": "웹 크롤러",
        "icon": "🕷️",
        "color": "#4ECDC4",
        "description": "뉴스 수집 및 자동 블로그 발행",
        "port": 8502,
        "key_features": ["RSS 피드", "HTML 크롤링", "AI 요약", "블로그 발행"]
    },
    3: {
        "name": "보석 분석",
        "icon": "💎",
        "color": "#45B7D1",
        "description": "보석 이미지 분석 및 산지 감정",
        "port": 8503,
        "key_features": ["색상 분석", "텍스처 분석", "산지 판별", "감정서 생성"]
    },
    4: {
        "name": "3D CAD 변환",
        "icon": "🏗️",
        "color": "#96CEB4",
        "description": "이미지를 3D CAD 파일로 변환",
        "port": 8504,
        "key_features": ["형상 분석", "3D 모델링", "CAD 생성", "제작 가능"]
    }
}

def check_module_status(port):
    """모듈이 실행 중인지 확인"""
    try:
        response = requests.get(f"http://localhost:{port}", timeout=3)
        return response.status_code == 200
    except:
        return False

def render_header():
    """헤더 렌더링"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    ">
        <h1 style="margin: 0; font-size: 2.5rem;">🎯 솔로몬드 AI 메인 대시보드</h1>
        <h3 style="margin: 0.5rem 0; opacity: 0.9;">4개 모듈 통합 플랫폼</h3>
        <p style="margin: 0; font-size: 1.1rem; opacity: 0.8;">
            원하는 모듈을 선택해서 접속하세요
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_module_card(module_id, module_info):
    """모듈 카드 렌더링"""
    is_running = check_module_status(module_info["port"])
    status_icon = "🟢" if is_running else "🔴"
    status_text = "실행 중" if is_running else "중지됨"
    
    # 모듈 타입별 표시
    is_main = module_info.get("is_main", False)
    is_legacy = module_info.get("is_legacy", False)
    is_unified = module_info.get("is_unified", False)
    
    if is_unified:
        badge = "🏆 UNIFIED"
        border_style = "border: 4px solid #FFD700; box-shadow: 0 0 30px rgba(255,215,0,0.8); animation: pulse 2s infinite;"
    elif is_main:
        badge = "⭐ MAIN"
        border_style = "border: 3px solid #FFD700; box-shadow: 0 0 15px rgba(255,215,0,0.4);"
    elif is_legacy:
        badge = "🔧 LEGACY"
        border_style = f"border: 2px dashed {module_info['color']}; box-shadow: 0 2px 4px rgba(0,0,0,0.1); opacity: 0.8;"
    else:
        badge = ""
        border_style = f"border: 2px solid {module_info['color']}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
    
    # 기능 목록
    features_html = "<br>".join([f"• {feature}" for feature in module_info["key_features"]])
    
    st.markdown(f"""
    <div style="
        {border_style}
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        background: linear-gradient(135deg, {module_info['color']}20, {module_info['color']}05);
        position: relative;
    ">
        {f'<div style="position: absolute; top: 10px; right: 15px; background: linear-gradient(135deg, #FFD700, #FFA500); color: #fff; padding: 5px 10px; border-radius: 15px; font-size: 0.8em; font-weight: bold;">{badge}</div>' if badge else ''}
        <h3 style="margin-top:0; color:{module_info['color']};">
            {module_info['icon']} {module_info['name']}
        </h3>
        <p><strong>상태:</strong> {status_icon} {status_text}</p>
        <p style="margin-bottom:15px;">{module_info['description']}</p>
        <div style="font-size:0.9em; color:#666; margin-bottom:15px;">
            {features_html}
        </div>
        <p style="margin-bottom:0;"><small style="color:#666;">포트: {module_info['port']}</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    # 버튼 컬럼
    col1, col2 = st.columns(2)
    
    with col1:
        if is_running:
            # 링크 방식으로 변경
            st.markdown(f"""
            <a href="http://localhost:{module_info['port']}" target="_blank" style="
                display: inline-block;
                background: linear-gradient(135deg, {module_info['color']}, {module_info['color']}90);
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 5px;
                font-weight: bold;
                text-align: center;
                width: 100%;
                box-sizing: border-box;
            ">
                🌐 {module_info['name']} 열기
            </a>
            """, unsafe_allow_html=True)
        else:
            if st.button(f"🚀 {module_info['name']} 시작", key=f"start_{module_id}", type="primary"):
                start_module(module_id, module_info)
    
    with col2:
        if st.button(f"📖 사용법", key=f"info_{module_id}"):
            show_module_info(module_id, module_info)

def start_module(module_id, module_info):
    """모듈 시작"""
    try:
        # 모듈 파일 경로 구성
        if module_id == 1:
            file_path = "modules/module1_conference/conference_analysis_unified.py"
        elif module_id == 2:
            file_path = "modules/module2_crawler/web_crawler_main.py"
        elif module_id == 3:
            file_path = "modules/module3_gemstone/gemstone_analyzer.py"
        elif module_id == 4:
            file_path = "modules/module4_3d_cad/image_to_cad.py"
        
        st.info(f"🚀 {module_info['name']} 시작 중...")
        
        # Streamlit 명령 실행
        command = [
            "python", "-m", "streamlit", "run", 
            file_path,
            "--server.port", str(module_info["port"]),
            "--server.headless", "true"
        ]
        
        subprocess.Popen(command)
        
        st.success(f"✅ {module_info['name']} 시작됨!")
        st.info(f"🌐 http://localhost:{module_info['port']} 에서 접속하세요")
        
        # 3초 후 자동 새로고침
        st.markdown("""
        <script>
        setTimeout(function(){
            window.location.reload();
        }, 3000);
        </script>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ 시작 실패: {str(e)}")
        st.info("수동으로 터미널에서 실행해주세요.")

def show_module_info(module_id, module_info):
    """모듈 사용법 표시"""
    st.markdown(f"### 📖 {module_info['name']} 사용법")
    
    # 파일 경로
    if module_id == 1:
        file_path = "modules/module1_conference/conference_analysis_unified.py"
    elif module_id == 2:
        file_path = "modules/module2_crawler/web_crawler_main.py"  
    elif module_id == 3:
        file_path = "modules/module3_gemstone/gemstone_analyzer.py"
    elif module_id == 4:
        file_path = "modules/module4_3d_cad/image_to_cad.py"
    
    st.markdown(f"""
    #### 🔧 수동 실행 방법
    ```bash
    cd C:\\Users\\PC_58410\\solomond-ai-system
    python -m streamlit run {file_path} --server.port {module_info['port']}
    ```
    
    #### 🌐 접속 URL
    **http://localhost:{module_info['port']}**
    """)
    
    # 모듈별 상세 설명
    if module_id == 1:
        st.markdown("""
        #### 🏆 통합 컨퍼런스 분석 주요 기능
        - **🎯 3가지 모드 통합**: 궁극/균형/안전 모드 UI에서 선택
        - **🏆 궁극 모드**: 모든 기능 + 최고 성능 (10MB 청크, 8스레드)
        - **⚖️ 균형 모드**: 핵심 기능 + 안정성 (5MB 청크, 4스레드)
        - **🛡️ 안전 모드**: 기본 기능 + 최대 안정 (1MB 청크, 2스레드)
        - **🚀 터보 업로드**: 3가지 속도 모드 자동 최적화
        - **🌐 URL 다운로드**: YouTube+웹페이지+문서 지원
        - **💾 스마트 캐시**: 중복 분석 완전 방지
        - **🔥 GPU/CPU 자동**: 환경별 최적화
        - **🎭 고품질 화자 분리**: 29차원 특징 + 실루엣 스코어
        
        #### 🏆 통합 분석 과정 (하나로 통합!)
        1. 분석 모드 선택 (궁극/균형/안전)
        2. 4가지 업로드 방식 (파일/URL/폴더/텍스트)
        3. 통합 분석 엔진 자동 실행
        4. 모드별 최적화된 결과 표시
        
        **🏆 하나의 시스템으로 모든 분석 모드를 경험하세요!**
        """)
    elif module_id == 2:
        st.markdown("""
        #### 🕷️ 웹 크롤러 주요 기능
        - **RSS 피드**: 자동 뉴스 수집
        - **HTML 크롤링**: 웹사이트 컨텐츠 추출
        - **AI 요약**: Ollama AI로 핵심 내용 요약
        - **블로그 발행**: Notion 연동 자동 발행
        """)
    elif module_id == 3:
        st.markdown("""
        #### 💎 보석 분석 주요 기능
        - **색상 분석**: AI 기반 색상 특성 분석
        - **텍스처 분석**: 표면 질감 및 투명도 측정
        - **산지 판별**: Ollama AI로 원산지 추정
        - **감정서 생성**: 전문가 수준 분석 리포트
        """)
    elif module_id == 4:
        st.markdown("""
        #### 🏗️ 3D CAD 변환 주요 기능
        - **형상 분석**: 이미지에서 3D 구조 인식
        - **3D 모델링**: AI 기반 3차원 모델 생성
        - **CAD 생성**: 라이노 호환 스크립트 생성
        - **제작 가능**: 실제 주얼리 제작용 파일 출력
        """)

def render_model_status():
    """🏆 v2 모델 라인업 상태 표시"""
    st.markdown("## 🤖 AI 모델 라인업 (v2.0)")
    
    if advanced_ollama is None:
        st.warning("⚠️ v2 Ollama 인터페이스가 로드되지 않았습니다.")
        return
    
    try:
        system_info = get_system_info()
        
        # 서버 상태
        if system_info['server_status']:
            st.success("🟢 Ollama 서버 연결됨")
        else:
            st.error("🔴 Ollama 서버 연결 실패")
            return
        
        # 모델 라인업 상태
        st.markdown("### 🏆 최적화된 5개 모델 라인업")
        
        model_lineup = system_info['model_lineup']
        cols = st.columns(5)
        
        tier_info = {
            'ultimate': {'emoji': '👑', 'name': 'ULTIMATE', 'color': '#FFD700'},
            'premium': {'emoji': '🔥', 'name': 'PREMIUM', 'color': '#FF6B6B'}, 
            'standard': {'emoji': '⚡', 'name': 'STANDARD', 'color': '#4ECDC4'},
            'stable': {'emoji': '🛡️', 'name': 'STABLE', 'color': '#45B7D1'},
            'fast': {'emoji': '🚀', 'name': 'FAST', 'color': '#96CEB4'}
        }
        
        for i, (tier, info) in enumerate(model_lineup.items()):
            with cols[i]:
                tier_data = tier_info.get(tier, {'emoji': '🤖', 'name': tier.upper(), 'color': '#666'})
                
                status_color = "#28a745" if info['available'] else "#dc3545"
                
                st.markdown(f"""
                <div style="
                    border: 2px solid {tier_data['color']};
                    border-radius: 10px;
                    padding: 15px;
                    text-align: center;
                    background: linear-gradient(135deg, {tier_data['color']}15, {tier_data['color']}05);
                ">
                    <h4 style="margin: 0; color: {tier_data['color']};">
                        {tier_data['emoji']}<br>{tier_data['name']}
                    </h4>
                    <p style="margin: 5px 0; font-size: 0.8em;">{info['model']}</p>
                    <div style="
                        background: {status_color};
                        color: white;
                        padding: 3px 8px;
                        border-radius: 15px;
                        font-size: 0.7em;
                        font-weight: bold;
                    ">{info['status']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # 추천 사항
        if 'recommendations' in system_info:
            if 'install' in system_info['recommendations']:
                st.warning(f"📦 {system_info['recommendations']['install']}")
            elif 'status' in system_info['recommendations']:
                st.success(f"✅ {system_info['recommendations']['status']}")
        
        # 벤치마크 버튼
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏁 모델 성능 벤치마크"):
                with st.spinner("모델 성능 테스트 중..."):
                    benchmark_results = benchmark_all()
                    
                    st.markdown("#### 🏁 벤치마크 결과")
                    
                    for tier, result in benchmark_results.items():
                        tier_data = tier_info.get(tier, {'emoji': '🤖', 'name': tier.upper()})
                        
                        if result['status'] == '성공':
                            st.success(f"{tier_data['emoji']} {tier_data['name']}: {result['time']}초 | {result['response_length']}자")
                            with st.expander(f"미리보기 - {result['model']}"):
                                st.write(result['preview'])
                        else:
                            st.error(f"{tier_data['emoji']} {tier_data['name']}: {result['status']}")
        
        with col2:
            if st.button("🔄 모델 상태 새로고침"):
                st.rerun()
        
    except Exception as e:
        st.error(f"모델 상태 확인 오류: {str(e)}")

def render_system_status():
    """시스템 상태 표시"""
    st.markdown("## 📊 모듈 실행 상태")
    
    # 메인 시스템 먼저 표시
    st.markdown("### ⭐ 메인 분석 시스템")
    col1 = st.columns(1)[0]
    with col1:
        is_running = check_module_status(MODULES[1]["port"])
        status_icon = "🟢" if is_running else "🔴"
        status_text = "실행 중" if is_running else "중지됨"
        st.metric(
            f"{MODULES[1]['icon']} {MODULES[1]['name']}", 
            status_text,
            f"포트 {MODULES[1]['port']}"
        )
    
    st.markdown("### 📋 추가 모듈들")
    cols = st.columns(3)
    
    running_count = 0
    if is_running:
        running_count += 1
        
    for i, module_id in enumerate([2, 3, 4]):
        module_info = MODULES[module_id]
        with cols[i]:
            is_running = check_module_status(module_info["port"])
            if is_running:
                running_count += 1
                
            status_icon = "🟢" if is_running else "🔴"
            status_text = "실행 중" if is_running else "중지됨"
            
            # 레거시는 회색으로 표시
            if module_info.get("is_legacy", False):
                status_icon = f"<span style='opacity: 0.6'>{status_icon}</span>"
            
            st.metric(
                f"{module_info['icon']} {module_info['name']}", 
                status_text,
                f"포트 {module_info['port']}"
            )
    
    # 전체 요약
    st.markdown(f"### 📈 전체 요약")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("실행 중인 모듈", f"{running_count}/4")
    
    with col2:
        st.metric("마지막 업데이트", datetime.now().strftime("%H:%M:%S"))
    
    with col3:
        health_score = (running_count / 4) * 100
        st.metric("시스템 건강도", f"{health_score:.0f}%")

def load_analysis_history():
    """분석 이력 로드"""
    history_file = Path("analysis_history/analysis_metadata.json")
    if history_file.exists():
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"분석 이력 로드 실패: {e}")
    return {"analyses": [], "total_count": 0}

def render_calendar_widget():
    """캘린더 위젯 렌더링"""
    st.subheader("📅 분석 이력 캘린더")
    
    # 분석 이력 로드
    history = load_analysis_history()
    
    if not history["analyses"]:
        st.info("📊 아직 분석 이력이 없습니다. 모듈1에서 컨퍼런스를 분석해보세요!")
        return
    
    # 날짜별 분석 데이터 준비
    analysis_data = []
    
    for analysis in history["analyses"]:
        try:
            timestamp = datetime.fromisoformat(analysis["timestamp"])
            analysis_data.append({
                "date": timestamp.strftime("%Y-%m-%d"),
                "time": timestamp.strftime("%H:%M"),
                "conference": analysis["conference_name"],
                "files": analysis["file_count"],
                "success_rate": analysis["success_rate"],
                "id": analysis["id"]
            })
        except Exception as e:
            continue
    
    if analysis_data:
        # 날짜별 분석 횟수 차트
        df = pd.DataFrame(analysis_data)
        date_counts = df['date'].value_counts().reset_index()
        date_counts.columns = ['날짜', '분석횟수']
        
        fig = px.bar(
            date_counts.head(10), 
            x='날짜', 
            y='분석횟수',
            title="📊 최근 10일 분석 활동",
            labels={'날짜': '날짜', '분석횟수': '분석 횟수'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # 상세 분석 이력 테이블
        st.subheader("📋 최근 분석 이력")
        for data in analysis_data[-5:]:  # 최근 5개만
            with st.expander(f"🎯 {data['conference']} ({data['date']} {data['time']})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("분석 파일 수", data['files'])
                with col2:
                    st.metric("성공률", data['success_rate'])
                with col3:
                    if st.button(f"결과 보기", key=f"view_{data['id']}"):
                        st.info(f"분석 ID: {data['id']}")

def render_quick_insights():
    """빠른 인사이트 카드"""
    st.subheader("💡 AI 듀얼 브레인 인사이트")
    
    # AI 인사이트 엔진 버튼 추가
    if st.button("🧠 고급 AI 인사이트 보기"):
        st.info("🚀 AI 인사이트 엔진을 별도 창에서 실행하세요:")
        st.code("streamlit run ai_insights_engine.py --server.port 8580")
        st.markdown("[🧠 AI 인사이트 엔진 열기](http://localhost:8580)")
    
    history = load_analysis_history()
    
    if history["total_count"] > 0:
        # 통계 계산
        total_analyses = history["total_count"]
        recent_analyses = len([a for a in history["analyses"] 
                              if datetime.fromisoformat(a["timestamp"]) > datetime.now() - timedelta(days=7)])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("총 분석 수", total_analyses, f"+{recent_analyses} (주간)")
        
        with col2:
            # 가장 활발한 요일 계산
            weekdays = []
            for a in history["analyses"]:
                try:
                    timestamp = datetime.fromisoformat(a["timestamp"])
                    weekdays.append(timestamp.strftime("%A"))
                except:
                    continue
            
            most_active_day = max(set(weekdays), key=weekdays.count) if weekdays else "데이터 없음"
            st.metric("가장 활발한 요일", most_active_day)
        
        with col3:
            avg_success = "계산 중"
            if history["analyses"]:
                try:
                    success_rates = []
                    for a in history["analyses"]:
                        rate_parts = a["success_rate"].split('/')
                        if len(rate_parts) == 2:
                            success_rates.append(int(rate_parts[0]) / int(rate_parts[1]))
                    if success_rates:
                        avg_success = f"{sum(success_rates)/len(success_rates)*100:.1f}%"
                except:
                    avg_success = "계산 중"
            
            st.metric("평균 성공률", avg_success)
        
        # AI 패턴 분석 및 추천사항
        st.markdown("### 🧠 AI 패턴 분석")
        
        if recent_analyses > 2:
            st.success("🎯 **AI 분석**: 활발한 분석 패턴을 보이고 있습니다! 이제 캘린더 연동으로 더 체계적인 관리가 가능합니다.")
        elif recent_analyses > 0:
            st.info("📈 **AI 제안**: 꾸준한 분석 활동 중입니다. 구글 캘린더 연동으로 스케줄 최적화를 해보세요.")
        else:
            st.warning("💭 **AI 알림**: 이번 주 분석 활동이 없습니다. 새로운 컨퍼런스를 분석해보세요!")
        
        # 다음 단계 제안
        if total_analyses >= 5:
            st.info("🚀 **다음 단계**: 충분한 데이터가 축적되었습니다. AI 인사이트 엔진 활성화를 준비 중입니다!")
    
    else:
        st.info("🌟 **시작하기**: 아직 분석 이력이 없습니다. 첫 번째 컨퍼런스를 분석해서 AI 듀얼 브레인을 활성화해보세요!")

def main():
    """듀얼 브레인 메인 대시보드"""
    # 헤더 렌더링
    render_header()
    
    # 듀얼 브레인 시스템 소개
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    ">
        <h2 style="margin: 0; font-size: 1.5rem;">🧠 솔로몬드 AI 듀얼 브레인 시스템</h2>
        <h3 style="margin: 0.5rem 0; font-size: 1.2rem;">당신의 세컨드 브레인이 되어 더 나은 인사이트를 제공합니다</h3>
        <p style="margin: 0; font-size: 1rem; opacity: 0.9;">
            📊 분석 → 📅 캘린더 → 🧠 AI 인사이트 → 🚀 미래 계획
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 메인 영역을 2개 칼럼으로 분할
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 캘린더 위젯
        render_calendar_widget()
    
    with col2:
        # AI 인사이트
        render_quick_insights()
    
    # 모듈 카드 그리드 
    st.markdown("## 📋 모듈 선택")
    
    # 메인 통합 분석 시스템 표시
    st.markdown("### 🏆 통합 분석 시스템 (최강 추천)")
    render_module_card(1, MODULES[1])  # 통합 컨퍼런스 분석
    
    st.markdown("### 📋 추가 모듈들")
    col1, col2 = st.columns(2)
    
    with col1:
        render_module_card(2, MODULES[2])  # 웹 크롤러
        render_module_card(4, MODULES[4])  # 3D CAD
    
    with col2:
        render_module_card(3, MODULES[3])  # 보석 분석
    
    st.markdown("---")
    
    # AI 모델 상태 (v2.0)
    render_model_status()
    
    # 모듈 실행 상태
    render_system_status()
    
    # 빠른 새로고침 버튼
    if st.button("🔄 상태 새로고침", help="모든 모듈의 상태를 다시 확인합니다"):
        st.rerun()
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        🎯 솔로몬드 AI 메인 대시보드<br>
        최신 통합 분석 시스템 | 4개 모듈 독립 실행
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()