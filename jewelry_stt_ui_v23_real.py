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
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📁 멀티파일 분석",
            "🎤 음성 분석", 
            "🖼️ 이미지 분석", 
            "📊 분석 결과", 
            "⚙️ 시스템 설정"
        ])
        
        with tab1:
            self.render_multifile_analysis_tab()
        
        with tab2:
            self.render_audio_analysis_tab()
        
        with tab3:
            self.render_image_analysis_tab()
        
        with tab4:
            self.render_results_tab()
        
        with tab5:
            self.render_settings_tab()
    
    def render_multifile_analysis_tab(self):
        """멀티파일 분석 탭"""
        
        st.markdown("## 📁 멀티파일 배치 분석")
        st.markdown("**🚀 모든 지원 형식을 한번에 업로드하여 배치 분석**")
        
        if not REAL_ANALYSIS_AVAILABLE:
            st.error("❌ 실제 분석 엔진이 로드되지 않았습니다.")
            return
        
        # 지원 형식 안내
        with st.expander("📋 지원하는 파일 형식"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🎤 음성 파일:**")
                st.markdown("- WAV, MP3, FLAC, M4A, MP4")
                st.markdown("- Whisper STT로 실제 변환")
            
            with col2:
                st.markdown("**🖼️ 이미지 파일:**")
                st.markdown("- JPG, JPEG, PNG, BMP, TIFF")
                st.markdown("- EasyOCR로 실제 텍스트 추출")
        
        # 멀티파일 업로드
        uploaded_files = st.file_uploader(
            "파일들을 선택하세요 (여러 개 동시 선택 가능)",
            type=['wav', 'mp3', 'flac', 'm4a', 'mp4', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Ctrl/Cmd + 클릭으로 여러 파일 선택 가능"
        )
        
        if uploaded_files:
            # 업로드된 파일 정보 표시
            st.markdown("### 📋 업로드된 파일 목록")
            
            audio_files = []
            image_files = []
            total_size = 0
            
            # 파일 분류
            for file in uploaded_files:
                file_size = len(file.getvalue()) / (1024 * 1024)
                total_size += file_size
                
                file_ext = file.name.split('.')[-1].lower()
                
                if file_ext in ['wav', 'mp3', 'flac', 'm4a', 'mp4']:
                    audio_files.append(file)
                elif file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
                    image_files.append(file)
            
            # 파일 정보 표시
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎤 음성 파일", f"{len(audio_files)}개")
            
            with col2:
                st.metric("🖼️ 이미지 파일", f"{len(image_files)}개")
            
            with col3:
                st.metric("📦 총 크기", f"{total_size:.2f} MB")
            
            # 파일 목록 상세 표시
            if audio_files or image_files:
                with st.expander("🔍 파일 상세 정보"):
                    
                    if audio_files:
                        st.markdown("**🎤 음성 파일들:**")
                        for i, file in enumerate(audio_files, 1):
                            file_size = len(file.getvalue()) / (1024 * 1024)
                            st.write(f"{i}. {file.name} ({file_size:.2f} MB)")
                    
                    if image_files:
                        st.markdown("**🖼️ 이미지 파일들:**")
                        for i, file in enumerate(image_files, 1):
                            file_size = len(file.getvalue()) / (1024 * 1024)
                            st.write(f"{i}. {file.name} ({file_size:.2f} MB)")
            
            # 분석 설정
            st.markdown("### ⚙️ 배치 분석 설정")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                audio_language = st.selectbox(
                    "음성 언어",
                    ["ko", "en", "auto"],
                    help="모든 음성 파일에 적용"
                )
            
            with col2:
                whisper_model = st.selectbox(
                    "Whisper 모델",
                    ["tiny", "base", "small", "medium"],
                    index=1,
                    help="정확도 vs 속도"
                )
            
            with col3:
                cpu_mode = st.checkbox(
                    "CPU 모드 강제",
                    value=True,
                    help="GPU 메모리 부족 방지"
                )
            
            # 배치 분석 시작
            if st.button("🚀 멀티파일 배치 분석 시작", type="primary"):
                self.process_multifile_analysis(
                    audio_files, image_files, 
                    audio_language, whisper_model, cpu_mode
                )
        
        else:
            st.info("📁 여러 파일을 선택하여 배치 분석을 시작하세요.")
            st.markdown("**💡 사용법:**")
            st.markdown("1. 위의 파일 업로드 버튼 클릭")
            st.markdown("2. Ctrl/Cmd + 클릭으로 여러 파일 선택")
            st.markdown("3. 음성과 이미지 파일을 함께 선택 가능")
            st.markdown("4. 설정 확인 후 배치 분석 시작")
    
    def process_multifile_analysis(self, audio_files: List, image_files: List, 
                                 language: str, model_size: str, cpu_mode: bool):
        """멀티파일 배치 분석 처리"""
        
        total_files = len(audio_files) + len(image_files)
        
        if total_files == 0:
            st.warning("⚠️ 분석할 파일이 없습니다.")
            return
        
        # CPU 모드 설정
        if cpu_mode:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # 진행 상황 표시
        st.markdown("### 🔄 배치 분석 진행 상황")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        batch_results = []
        processed_count = 0
        
        # 배치 분석 시작 시간
        batch_start_time = time.time()
        
        try:
            # 음성 파일 분석
            for i, audio_file in enumerate(audio_files):
                
                status_text.text(f"🎤 음성 분석 중: {audio_file.name} ({i+1}/{len(audio_files)})")
                
                # 임시 파일 생성
                with tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=f".{audio_file.name.split('.')[-1]}"
                ) as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # 실제 분석 실행
                    result = self.analysis_engine.analyze_audio_file(tmp_file_path, language)
                    result['batch_index'] = processed_count + 1
                    result['file_type'] = 'audio'
                    batch_results.append(result)
                    
                    # 결과 실시간 표시
                    with results_container:
                        if result.get('status') == 'success':
                            st.success(f"✅ {audio_file.name}: {result['text_length']}글자 추출 ({result['processing_time']}초)")
                        else:
                            st.error(f"❌ {audio_file.name}: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    error_result = {
                        'status': 'error',
                        'error': str(e),
                        'file_name': audio_file.name,
                        'batch_index': processed_count + 1,
                        'file_type': 'audio'
                    }
                    batch_results.append(error_result)
                    
                    with results_container:
                        st.error(f"❌ {audio_file.name}: {str(e)}")
                
                finally:
                    # 임시 파일 정리
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
                
                # 진행률 업데이트
                processed_count += 1
                progress_bar.progress(processed_count / total_files)
            
            # 이미지 파일 분석
            for i, image_file in enumerate(image_files):
                
                status_text.text(f"🖼️ 이미지 분석 중: {image_file.name} ({i+1}/{len(image_files)})")
                
                # 임시 파일 생성
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f".{image_file.name.split('.')[-1]}"
                ) as tmp_file:
                    tmp_file.write(image_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # 실제 분석 실행
                    result = self.analysis_engine.analyze_image_file(tmp_file_path)
                    result['batch_index'] = processed_count + 1
                    result['file_type'] = 'image'
                    batch_results.append(result)
                    
                    # 결과 실시간 표시
                    with results_container:
                        if result.get('status') == 'success':
                            st.success(f"✅ {image_file.name}: {result['blocks_detected']}개 블록 추출 ({result['processing_time']}초)")
                        else:
                            st.error(f"❌ {image_file.name}: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    error_result = {
                        'status': 'error',
                        'error': str(e),
                        'file_name': image_file.name,
                        'batch_index': processed_count + 1,
                        'file_type': 'image'
                    }
                    batch_results.append(error_result)
                    
                    with results_container:
                        st.error(f"❌ {image_file.name}: {str(e)}")
                
                finally:
                    # 임시 파일 정리
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
                
                # 진행률 업데이트
                processed_count += 1
                progress_bar.progress(processed_count / total_files)
            
            # 배치 분석 완료
            batch_end_time = time.time()
            total_batch_time = batch_end_time - batch_start_time
            
            # 최종 결과 요약
            self.display_batch_results_summary(batch_results, total_batch_time)
            
            # 세션에 결과 저장
            if 'analysis_results' not in st.session_state:
                st.session_state.analysis_results = []
            st.session_state.analysis_results.extend(batch_results)
            
            status_text.text("✅ 멀티파일 배치 분석 완료!")
            
        except Exception as e:
            st.error(f"❌ 배치 분석 중 오류: {str(e)}")
            self.logger.error(f"배치 분석 오류: {e}")
    
    def display_batch_results_summary(self, batch_results: List[Dict], total_time: float):
        """배치 분석 결과 요약 표시"""
        
        st.markdown("### 📊 배치 분석 완료 요약")
        
        # 통계 계산
        total_files = len(batch_results)
        successful_files = len([r for r in batch_results if r.get('status') == 'success'])
        audio_files = len([r for r in batch_results if r.get('file_type') == 'audio'])
        image_files = len([r for r in batch_results if r.get('file_type') == 'image'])
        
        # 메트릭 표시
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 파일", f"{total_files}개")
        
        with col2:
            st.metric("성공", f"{successful_files}개")
        
        with col3:
            success_rate = (successful_files / total_files * 100) if total_files > 0 else 0
            st.metric("성공률", f"{success_rate:.1f}%")
        
        with col4:
            st.metric("총 처리시간", f"{total_time:.1f}초")
        
        # 파일 타입별 통계
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎤 음성 파일 결과:**")
            audio_success = len([r for r in batch_results 
                               if r.get('file_type') == 'audio' and r.get('status') == 'success'])
            st.write(f"- 처리: {audio_files}개")
            st.write(f"- 성공: {audio_success}개")
            
            if audio_success > 0:
                total_text = sum(r.get('text_length', 0) for r in batch_results 
                               if r.get('file_type') == 'audio' and r.get('status') == 'success')
                st.write(f"- 총 추출 텍스트: {total_text}글자")
        
        with col2:
            st.markdown("**🖼️ 이미지 파일 결과:**")
            image_success = len([r for r in batch_results 
                               if r.get('file_type') == 'image' and r.get('status') == 'success'])
            st.write(f"- 처리: {image_files}개")
            st.write(f"- 성공: {image_success}개")
            
            if image_success > 0:
                total_blocks = sum(r.get('blocks_detected', 0) for r in batch_results 
                                 if r.get('file_type') == 'image' and r.get('status') == 'success')
                st.write(f"- 총 텍스트 블록: {total_blocks}개")
        
        # 전체 결과 다운로드
        if st.button("📥 배치 분석 결과 전체 다운로드"):
            batch_json = json.dumps({
                'batch_summary': {
                    'total_files': total_files,
                    'successful_files': successful_files,
                    'success_rate': success_rate,
                    'total_processing_time': total_time,
                    'audio_files': audio_files,
                    'image_files': image_files
                },
                'individual_results': batch_results
            }, indent=2, ensure_ascii=False)
            
            st.download_button(
                "JSON 파일 다운로드",
                data=batch_json,
                file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

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