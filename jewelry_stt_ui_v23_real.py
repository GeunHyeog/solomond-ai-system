#!/usr/bin/env python3
"""
솔로몬드 AI v2.3 - 실제 분석 통합 버전
가짜 분석을 실제 분석으로 완전 교체
"""

import streamlit as st
import sys
import os
import asyncio
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 실제 분석 엔진 import
try:
    from core.real_analysis_engine import global_analysis_engine, analyze_file_real
    REAL_ANALYSIS_AVAILABLE = True
    print("✅ 실제 분석 엔진 로드 완료")
except ImportError as e:
    REAL_ANALYSIS_AVAILABLE = False
    print(f"❌ 실제 분석 엔진 로드 실패: {e}")

# 기존 모듈들
try:
    from core.hybrid_llm_manager_v23 import HybridLLMManager
    HYBRID_LLM_AVAILABLE = True
except ImportError:
    HYBRID_LLM_AVAILABLE = False
    
# Streamlit 설정
st.set_page_config(
    page_title="솔로몬드 AI v2.3 - 실제 분석",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SolomondRealAnalysisUI:
    """솔로몬드 AI v2.3 실제 분석 UI"""
    
    def __init__(self):
        self.setup_logging()
        self.analysis_engine = global_analysis_engine if REAL_ANALYSIS_AVAILABLE else None
        self.session_stats = {
            "files_analyzed": 0,
            "total_processing_time": 0,
            "successful_analyses": 0,
            "session_start": datetime.now()
        }
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """메인 실행"""
        
        # 헤더
        st.markdown("""
        # 💎 솔로몬드 AI v2.3 - 실제 분석 시스템
        
        **🚀 실제 AI 분석:** Whisper STT + EasyOCR + 무료 AI 모델 통합
        """)
        
        # 시스템 상태 표시
        self.display_system_status()
        
        # 탭 구성
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎤 음성 분석", 
            "🖼️ 이미지 분석", 
            "📊 분석 결과", 
            "⚙️ 시스템 설정"
        ])
        
        with tab1:
            self.render_audio_analysis_tab()
        
        with tab2:
            self.render_image_analysis_tab()
        
        with tab3:
            self.render_results_tab()
        
        with tab4:
            self.render_settings_tab()
    
    def display_system_status(self):
        """시스템 상태 표시"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if REAL_ANALYSIS_AVAILABLE:
                st.success("✅ 실제 분석 엔진")
            else:
                st.error("❌ 실제 분석 엔진")
        
        with col2:
            try:
                import whisper
                st.success("✅ Whisper STT")
            except ImportError:
                st.error("❌ Whisper STT")
        
        with col3:
            try:
                import easyocr
                st.success("✅ EasyOCR")
            except ImportError:
                st.error("❌ EasyOCR")
        
        with col4:
            try:
                from transformers import pipeline
                st.success("✅ Transformers")
            except ImportError:
                st.warning("⚠️ Transformers")
    
    def render_audio_analysis_tab(self):
        """음성 분석 탭"""
        
        st.markdown("## 🎤 실제 음성 분석 (Whisper STT)")
        
        if not REAL_ANALYSIS_AVAILABLE:
            st.error("❌ 실제 분석 엔진이 로드되지 않았습니다.")
            return
        
        # 파일 업로드
        uploaded_file = st.file_uploader(
            "음성 파일 업로드",
            type=['wav', 'mp3', 'flac', 'm4a', 'mp4'],
            help="지원 형식: WAV, MP3, FLAC, M4A, MP4"
        )
        
        if uploaded_file is not None:
            # 파일 정보 표시
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📁 파일: {uploaded_file.name} ({file_size:.2f} MB)")
            
            # 분석 설정
            col1, col2 = st.columns(2)
            with col1:
                language = st.selectbox(
                    "언어 선택",
                    ["ko", "en", "auto"],
                    help="ko: 한국어, en: 영어, auto: 자동 감지"
                )
            
            with col2:
                whisper_model = st.selectbox(
                    "Whisper 모델",
                    ["tiny", "base", "small", "medium"],
                    index=1,
                    help="tiny: 빠름, medium: 정확"
                )
            
            # 분석 시작
            if st.button("🎯 실제 음성 분석 시작", type="primary"):
                self.process_audio_analysis(uploaded_file, language, whisper_model)
    
    def process_audio_analysis(self, uploaded_file, language: str, model_size: str):
        """실제 음성 분석 처리"""
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            with st.spinner("🎤 Whisper STT 실제 분석 중..."):
                
                # 실제 분석 실행
                start_time = time.time()
                result = self.analysis_engine.analyze_audio_file(tmp_file_path, language)
                processing_time = time.time() - start_time
                
                # 결과 표시
                if result.get("status") == "success":
                    st.success(f"✅ 음성 분석 완료! ({result['processing_time']}초)")
                    
                    # 분석 결과 표시
                    self.display_audio_results(result)
                    
                    # 세션 통계 업데이트
                    self.update_session_stats(processing_time, True)
                    
                    # 결과를 세션에 저장
                    if 'analysis_results' not in st.session_state:
                        st.session_state.analysis_results = []
                    st.session_state.analysis_results.append(result)
                    
                else:
                    st.error(f"❌ 음성 분석 실패: {result.get('error', 'Unknown error')}")
                    self.update_session_stats(processing_time, False)
                    
        except Exception as e:
            st.error(f"❌ 분석 중 오류: {str(e)}")
            self.logger.error(f"음성 분석 오류: {e}")
        
        finally:
            # 임시 파일 정리
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    def display_audio_results(self, result: Dict[str, Any]):
        """음성 분석 결과 표시"""
        
        # 기본 정보
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("처리 시간", f"{result['processing_time']}초")
        
        with col2:
            st.metric("감지 언어", result['detected_language'])
        
        with col3:
            st.metric("텍스트 길이", f"{result['text_length']}자")
        
        with col4:
            st.metric("세그먼트", f"{result['segments_count']}개")
        
        # 추출된 텍스트
        st.markdown("### 📄 추출된 텍스트")
        st.text_area(
            "전체 텍스트",
            value=result['full_text'],
            height=200,
            disabled=True
        )
        
        # 요약
        if result.get('summary'):
            st.markdown("### 📋 AI 요약")
            st.info(result['summary'])
        
        # 주얼리 키워드
        if result.get('jewelry_keywords'):
            st.markdown("### 💎 주얼리 키워드")
            for keyword in result['jewelry_keywords']:
                st.badge(keyword)
        
        # 상세 세그먼트 (확장 가능)
        with st.expander("🔍 상세 세그먼트 정보"):
            for i, segment in enumerate(result.get('segments', []), 1):
                st.write(f"**{i}. [{segment['start']:.1f}s - {segment['end']:.1f}s]**")
                st.write(segment['text'])
                st.write("---")
    
    def render_image_analysis_tab(self):
        """이미지 분석 탭"""
        
        st.markdown("## 🖼️ 실제 이미지 분석 (EasyOCR)")
        
        if not REAL_ANALYSIS_AVAILABLE:
            st.error("❌ 실제 분석 엔진이 로드되지 않았습니다.")
            return
        
        # GPU 메모리 경고
        st.warning("⚠️ GPU 메모리 부족시 CPU 모드로 실행됩니다. 처리 시간이 길어질 수 있습니다.")
        
        # 파일 업로드
        uploaded_file = st.file_uploader(
            "이미지 파일 업로드",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="지원 형식: JPG, PNG, BMP, TIFF"
        )
        
        if uploaded_file is not None:
            # 이미지 미리보기
            st.image(uploaded_file, caption="업로드된 이미지", use_container_width=True)
            
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📁 파일: {uploaded_file.name} ({file_size:.2f} MB)")
            
            # 분석 시작
            if st.button("🎯 실제 OCR 분석 시작", type="primary"):
                self.process_image_analysis(uploaded_file)
    
    def process_image_analysis(self, uploaded_file):
        """실제 이미지 분석 처리"""
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            with st.spinner("🖼️ EasyOCR 실제 분석 중..."):
                
                # CPU 모드 강제 설정 (GPU 메모리 부족 방지)
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                
                # 실제 분석 실행
                start_time = time.time()
                result = self.analysis_engine.analyze_image_file(tmp_file_path)
                processing_time = time.time() - start_time
                
                # 결과 표시
                if result.get("status") == "success":
                    st.success(f"✅ 이미지 분석 완료! ({result['processing_time']}초)")
                    
                    # 분석 결과 표시
                    self.display_image_results(result)
                    
                    # 세션 통계 업데이트
                    self.update_session_stats(processing_time, True)
                    
                    # 결과를 세션에 저장
                    if 'analysis_results' not in st.session_state:
                        st.session_state.analysis_results = []
                    st.session_state.analysis_results.append(result)
                    
                else:
                    st.error(f"❌ 이미지 분석 실패: {result.get('error', 'Unknown error')}")
                    self.update_session_stats(processing_time, False)
                    
        except Exception as e:
            st.error(f"❌ 분석 중 오류: {str(e)}")
            self.logger.error(f"이미지 분석 오류: {e}")
        
        finally:
            # 임시 파일 정리
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    def display_image_results(self, result: Dict[str, Any]):
        """이미지 분석 결과 표시"""
        
        # 기본 정보
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("처리 시간", f"{result['processing_time']}초")
        
        with col2:
            st.metric("텍스트 블록", f"{result['blocks_detected']}개")
        
        with col3:
            st.metric("평균 신뢰도", f"{result['average_confidence']:.3f}")
        
        with col4:
            st.metric("파일 크기", f"{result['file_size_mb']} MB")
        
        # 추출된 텍스트
        st.markdown("### 📄 추출된 텍스트")
        st.text_area(
            "OCR 결과",
            value=result['full_text'],
            height=150,
            disabled=True
        )
        
        # 요약
        if result.get('summary'):
            st.markdown("### 📋 AI 요약")
            st.info(result['summary'])
        
        # 주얼리 키워드
        if result.get('jewelry_keywords'):
            st.markdown("### 💎 주얼리 키워드")
            for keyword in result['jewelry_keywords']:
                st.badge(keyword)
        
        # 상세 결과 (확장 가능)
        with st.expander("🔍 상세 OCR 결과"):
            for i, item in enumerate(result.get('detailed_results', []), 1):
                st.write(f"**{i}. 신뢰도: {item['confidence']:.3f}**")
                st.write(f"텍스트: {item['text']}")
                st.write("---")
    
    def render_results_tab(self):
        """분석 결과 탭"""
        
        st.markdown("## 📊 분석 결과 모음")
        
        # 세션 통계
        self.display_session_stats()
        
        # 저장된 결과들
        if 'analysis_results' in st.session_state and st.session_state.analysis_results:
            
            st.markdown("### 📋 분석 기록")
            
            for i, result in enumerate(st.session_state.analysis_results):
                with st.expander(f"🔍 {result['file_name']} - {result['analysis_type']}"):
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**파일:** {result['file_name']}")
                        st.write(f"**타입:** {result['analysis_type']}")
                        st.write(f"**시간:** {result['timestamp']}")
                        st.write(f"**처리 시간:** {result['processing_time']}초")
                        
                        if result.get('full_text'):
                            st.text_area(
                                "추출 텍스트",
                                value=result['full_text'][:500] + ("..." if len(result['full_text']) > 500 else ""),
                                height=100,
                                disabled=True,
                                key=f"text_{i}"
                            )
                    
                    with col2:
                        if st.button(f"📥 결과 다운로드", key=f"download_{i}"):
                            json_str = json.dumps(result, indent=2, ensure_ascii=False)
                            st.download_button(
                                "JSON 다운로드",
                                data=json_str,
                                file_name=f"analysis_{result['file_name']}_{i}.json",
                                mime="application/json"
                            )
            
            # 전체 결과 초기화
            if st.button("🗑️ 모든 결과 초기화"):
                st.session_state.analysis_results = []
                st.rerun()
        
        else:
            st.info("📝 아직 분석 결과가 없습니다. 음성 또는 이미지 분석을 실행해보세요.")
    
    def display_session_stats(self):
        """세션 통계 표시"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("분석한 파일", f"{self.session_stats['files_analyzed']}개")
        
        with col2:
            st.metric("성공한 분석", f"{self.session_stats['successful_analyses']}개")
        
        with col3:
            success_rate = 0
            if self.session_stats['files_analyzed'] > 0:
                success_rate = (self.session_stats['successful_analyses'] / self.session_stats['files_analyzed']) * 100
            st.metric("성공률", f"{success_rate:.1f}%")
        
        with col4:
            st.metric("총 처리 시간", f"{self.session_stats['total_processing_time']:.1f}초")
    
    def render_settings_tab(self):
        """설정 탭"""
        
        st.markdown("## ⚙️ 시스템 설정")
        
        # 분석 엔진 통계
        if REAL_ANALYSIS_AVAILABLE and self.analysis_engine:
            st.markdown("### 📊 분석 엔진 통계")
            
            try:
                stats = self.analysis_engine.get_analysis_stats()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.json({
                        "전체 분석 파일": stats.get('total_files', 0),
                        "성공한 분석": stats.get('successful_analyses', 0),
                        "성공률": f"{stats.get('success_rate', 0):.1f}%"
                    })
                
                with col2:
                    st.json({
                        "총 처리 시간": f"{stats.get('total_processing_time', 0):.1f}초",
                        "평균 처리 시간": f"{stats.get('average_processing_time', 0):.1f}초",
                        "마지막 분석": stats.get('last_analysis_time', 'N/A')
                    })
            
            except Exception as e:
                st.error(f"통계 로드 실패: {e}")
        
        # 시스템 정보
        st.markdown("### 🖥️ 시스템 정보")
        
        system_info = {
            "실제 분석 엔진": "✅ 사용 가능" if REAL_ANALYSIS_AVAILABLE else "❌ 사용 불가",
            "Whisper STT": "✅ 설치됨" if self._check_module('whisper') else "❌ 미설치",
            "EasyOCR": "✅ 설치됨" if self._check_module('easyocr') else "❌ 미설치",
            "Transformers": "✅ 설치됨" if self._check_module('transformers') else "❌ 미설치",
            "Google Gemini": "✅ 설치됨" if self._check_module('google.generativeai') else "❌ 미설치"
        }
        
        st.json(system_info)
        
        # 메모리 최적화 옵션
        st.markdown("### 🔧 성능 최적화")
        
        if st.button("🗑️ GPU 메모리 정리"):
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    st.success("✅ GPU 메모리가 정리되었습니다.")
                else:
                    st.info("ℹ️ CUDA가 사용 불가능합니다.")
            except ImportError:
                st.warning("⚠️ PyTorch가 설치되지 않았습니다.")
    
    def _check_module(self, module_name: str) -> bool:
        """모듈 설치 여부 확인"""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def update_session_stats(self, processing_time: float, success: bool):
        """세션 통계 업데이트"""
        self.session_stats['files_analyzed'] += 1
        self.session_stats['total_processing_time'] += processing_time
        if success:
            self.session_stats['successful_analyses'] += 1

def main():
    """메인 실행 함수"""
    
    # UI 인스턴스 생성 및 실행
    ui = SolomondRealAnalysisUI()
    ui.run()

if __name__ == "__main__":
    main()