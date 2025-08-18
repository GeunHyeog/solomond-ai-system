#!/usr/bin/env python3
"""
🎯 Module 1: 컨퍼런스 분석 Streamlit UI
- FastAPI 백엔드와 연동
- 사용자 친화적인 파일 업로드 인터페이스
- 실시간 분석 결과 표시
"""

import streamlit as st
import requests
import json
import time
from pathlib import Path
import io
import base64

# Streamlit 페이지 설정
st.set_page_config(
    page_title="솔로몬드 AI - 컨퍼런스 분석",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API 엔드포인트
API_BASE = "http://localhost:8001"

def check_api_status():
    """API 서버 상태 확인"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def get_supported_formats():
    """지원 형식 조회"""
    try:
        response = requests.get(f"{API_BASE}/supported-formats", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def analyze_file(file_data, filename, analysis_type="comprehensive"):
    """파일 분석 요청"""
    try:
        files = {'file': (filename, file_data)}
        data = {'analysis_type': analysis_type}
        
        response = requests.post(f"{API_BASE}/analyze", files=files, data=data, timeout=300)
        return response.status_code == 200, response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return False, str(e)

def analyze_large_file(file_data, filename, analysis_type="comprehensive"):
    """대용량 파일 분석 요청"""
    try:
        files = {'file': (filename, file_data)}
        data = {'analysis_type': analysis_type}
        
        response = requests.post(f"{API_BASE}/analyze/large-file", files=files, data=data, timeout=600)
        return response.status_code == 200, response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return False, str(e)

def main():
    # 헤더
    st.title("🎯 솔로몬드 AI - 컨퍼런스 분석")
    st.markdown("### 음성, 이미지, 비디오 통합 분석 시스템")
    
    # API 상태 확인
    api_status, health_data = check_api_status()
    
    if not api_status:
        st.error("❌ API 서버에 연결할 수 없습니다. Module 1 서비스가 실행 중인지 확인하세요.")
        st.info("서비스 시작: `python microservices/module1_service.py`")
        st.stop()
    
    # 상태 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("서비스 상태", "✅ 정상")
    
    with col2:
        memory_percent = health_data.get('memory', {}).get('memory_percent', 0)
        st.metric("메모리 사용률", f"{memory_percent:.1f}%")
    
    with col3:
        st.metric("서비스 버전", health_data.get('version', '4.0.0'))
    
    with col4:
        st.metric("로드된 모델", health_data.get('memory', {}).get('loaded_models', 0))
    
    st.divider()
    
    # 사이드바 - 설정
    with st.sidebar:
        st.header("⚙️ 분석 설정")
        
        analysis_type = st.selectbox(
            "분석 유형",
            ["comprehensive", "quick", "detailed"],
            index=0,
            help="comprehensive: 전체 분석, quick: 빠른 분석, detailed: 상세 분석"
        )
        
        use_large_file = st.checkbox(
            "대용량 파일 모드",
            help="10MB 이상 파일이나 처리가 오래 걸리는 파일에 사용"
        )
        
        st.divider()
        
        # 지원 형식 표시
        formats_data = get_supported_formats()
        if formats_data:
            st.header("📋 지원 형식")
            
            with st.expander("🎵 오디오 형식"):
                audio_formats = formats_data.get('supported_formats', {}).get('audio', [])
                st.write(", ".join(audio_formats))
                st.info("m4a → wav 자동 변환 지원")
            
            with st.expander("🖼️ 이미지 형식"):
                image_formats = formats_data.get('supported_formats', {}).get('image', [])
                st.write(", ".join(image_formats))
                st.info("EasyOCR 텍스트 추출")
            
            with st.expander("🎬 비디오 형식"):
                video_formats = formats_data.get('supported_formats', {}).get('video', [])
                st.write(", ".join(video_formats))
                st.info("기본 정보 추출")
    
    # 메인 영역 - 파일 업로드
    st.header("📁 파일 업로드 및 분석")
    
    uploaded_file = st.file_uploader(
        "분석할 파일을 선택하세요",
        type=['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac', 
              'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif',
              'mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="음성, 이미지, 비디오 파일을 업로드하세요"
    )
    
    if uploaded_file is not None:
        # 파일 정보 표시
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**파일명:** {uploaded_file.name}")
        with col2:
            st.info(f"**크기:** {file_size_mb:.2f} MB")
        with col3:
            st.info(f"**형식:** {Path(uploaded_file.name).suffix}")
        
        # 분석 시작 버튼
        if st.button("🚀 분석 시작", type="primary", use_container_width=True):
            with st.spinner("분석 중... 잠시만 기다려주세요."):
                start_time = time.time()
                
                # 파일 데이터 준비
                file_data = uploaded_file.getvalue()
                
                # 분석 함수 선택
                if use_large_file or file_size_mb > 10:
                    success, result = analyze_large_file(file_data, uploaded_file.name, analysis_type)
                else:
                    success, result = analyze_file(file_data, uploaded_file.name, analysis_type)
                
                processing_time = time.time() - start_time
            
            if success:
                st.success(f"✅ 분석 완료! (소요시간: {processing_time:.2f}초)")
                
                # 결과 표시
                st.header("📊 분석 결과")
                
                # 세션 정보
                session_id = result.get('session_id', 'N/A')
                st.info(f"**세션 ID:** {session_id}")
                
                results = result.get('results', {})
                
                # 탭으로 결과 구분
                tabs = st.tabs(["📝 요약", "🎵 음성 분석", "🖼️ 이미지 분석", "🎬 비디오 분석", "📊 파일 정보"])
                
                with tabs[0]:  # 요약
                    if 'summary' in results:
                        st.json(results['summary'])
                    else:
                        st.write("요약 정보가 없습니다.")
                
                with tabs[1]:  # 음성 분석
                    if 'audio_analysis' in results and results['audio_analysis']:
                        for filename, analysis in results['audio_analysis'].items():
                            st.subheader(f"🎵 {filename}")
                            
                            if 'analysis' in analysis and 'transcript' in analysis['analysis']:
                                st.write("**음성 인식 결과:**")
                                st.write(analysis['analysis']['transcript'])
                                
                                if 'language' in analysis['analysis']:
                                    st.write(f"**감지된 언어:** {analysis['analysis']['language']}")
                                
                                if 'confidence' in analysis['analysis']:
                                    st.write(f"**신뢰도:** {analysis['analysis']['confidence']:.2f}")
                            else:
                                st.json(analysis)
                    else:
                        st.write("음성 분석 결과가 없습니다.")
                
                with tabs[2]:  # 이미지 분석
                    if 'image_analysis' in results and results['image_analysis']:
                        for filename, analysis in results['image_analysis'].items():
                            st.subheader(f"🖼️ {filename}")
                            
                            if 'analysis' in analysis:
                                img_analysis = analysis['analysis']
                                
                                if 'total_text' in img_analysis:
                                    st.write("**추출된 텍스트:**")
                                    st.write(img_analysis['total_text'])
                                
                                if 'text_blocks' in img_analysis:
                                    st.write(f"**텍스트 블록 수:** {len(img_analysis['text_blocks'])}")
                                
                                if 'confidence_avg' in img_analysis:
                                    st.write(f"**평균 신뢰도:** {img_analysis['confidence_avg']:.2f}")
                            else:
                                st.json(analysis)
                    else:
                        st.write("이미지 분석 결과가 없습니다.")
                
                with tabs[3]:  # 비디오 분석
                    if 'video_analysis' in results and results['video_analysis']:
                        for filename, analysis in results['video_analysis'].items():
                            st.subheader(f"🎬 {filename}")
                            st.json(analysis)
                    else:
                        st.write("비디오 분석 결과가 없습니다.")
                
                with tabs[4]:  # 파일 정보
                    if 'file_info' in results:
                        st.json(results['file_info'])
                    
                    st.subheader("처리 정보")
                    st.write(f"**처리 시간:** {result.get('processing_time', 0):.2f}초")
                    st.write(f"**타임스탬프:** {result.get('timestamp', 'N/A')}")
                
                # 전체 결과 JSON (접을 수 있는 형태)
                with st.expander("🔍 전체 결과 (JSON)"):
                    st.json(result)
            
            else:
                st.error(f"❌ 분석 실패: {result}")
    
    # 하단 정보
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**API 서버:** http://localhost:8001")
    
    with col2:
        st.info("**문서:** http://localhost:8001/docs")
    
    with col3:
        if st.button("🔄 새로고침"):
            st.rerun()

if __name__ == "__main__":
    main()