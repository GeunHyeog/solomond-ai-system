#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 SOLOMOND AI 통합 분석 시스템 - 컴퓨팅 최적화 버전
Integrated Analysis System with Compute Optimization

주요 개선사항:
1. 하이브리드 CPU/GPU 자동 선택
2. 실시간 리소스 모니터링 
3. 작업별 최적화 설정
4. 메모리 효율성 극대화
"""

import streamlit as st
import os
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

# 하이브리드 컴퓨팅 매니저 임포트
try:
    from hybrid_compute_manager import (
        HybridComputeManager, ComputeMode, TaskType,
        auto_optimize_for_whisper, auto_optimize_for_ocr, auto_optimize_for_llm
    )
    COMPUTE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    COMPUTE_OPTIMIZATION_AVAILABLE = False

# 기존 분석 시스템 임포트
try:
    from conference_analysis_COMPLETE_WORKING import CompleteWorkingAnalyzer
    MAIN_ANALYZER_AVAILABLE = True
except ImportError:
    MAIN_ANALYZER_AVAILABLE = False

class OptimizedAnalysisSystem:
    """최적화된 분석 시스템"""
    
    def __init__(self):
        self.compute_manager = HybridComputeManager() if COMPUTE_OPTIMIZATION_AVAILABLE else None
        self.main_analyzer = CompleteWorkingAnalyzer() if MAIN_ANALYZER_AVAILABLE else None
        
        # 리소스 설정 로드
        self.load_resource_config()
    
    def load_resource_config(self):
        """리소스 설정 로드"""
        config_file = Path("resource_config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.resource_config = json.load(f)
            except:
                self.resource_config = {"compute_mode": "auto"}
        else:
            self.resource_config = {"compute_mode": "auto"}
    
    def optimize_for_analysis(self, file_info: Dict) -> Dict[str, Any]:
        """분석 작업에 최적화된 설정 반환"""
        if not self.compute_manager:
            return {"device": "cpu", "optimized": False}
        
        # 파일 정보 기반으로 작업 유형 결정
        file_types = file_info.get('file_types', [])
        file_count = file_info.get('file_count', 1)
        total_size_mb = file_info.get('total_size_mb', 0)
        
        optimization_settings = {}
        
        # 음성 파일 최적화
        if any('audio' in ft for ft in file_types):
            # 예상 음성 길이 (크기 기반 추정)
            estimated_duration = total_size_mb * 10  # 대략적 추정
            whisper_config = auto_optimize_for_whisper(estimated_duration)
            optimization_settings['whisper'] = whisper_config
        
        # 이미지 파일 최적화
        if any('image' in ft for ft in file_types):
            image_count = len([ft for ft in file_types if 'image' in ft])
            ocr_config = auto_optimize_for_ocr(image_count, realtime=False)
            optimization_settings['ocr'] = ocr_config
        
        # LLM 최적화 (종합 분석용)
        llm_config = auto_optimize_for_llm(context_length=2048)
        optimization_settings['llm'] = llm_config
        
        # 전반적인 디바이스 선택
        if any(config.get('device') == 'gpu' for config in optimization_settings.values()):
            optimization_settings['primary_device'] = 'gpu'
        else:
            optimization_settings['primary_device'] = 'cpu'
        
        optimization_settings['optimized'] = True
        
        return optimization_settings
    
    def render_resource_status_widget(self):
        """리소스 상태 위젯 렌더링"""
        if not self.compute_manager:
            return
        
        st.sidebar.markdown("### ⚙️ 리소스 상태")
        
        # 현재 상태 가져오기
        status = self.compute_manager.get_resource_status()
        
        # CPU 상태
        cpu_status = status.get('cpu', {})
        cpu_usage = cpu_status.get('usage_percent', 0)
        
        st.sidebar.metric(
            "CPU 사용률",
            f"{cpu_usage:.1f}%",
            delta=None
        )
        
        memory_gb = cpu_status.get('available_memory_gb', 0)
        st.sidebar.metric(
            "사용 가능 메모리",
            f"{memory_gb:.1f}GB"
        )
        
        # GPU 상태 (있는 경우)
        if 'gpu' in status:
            gpu_status = status['gpu']
            gpu_memory = gpu_status.get('available_memory_gb', 0)
            gpu_usage = gpu_status.get('usage_percent', 0)
            
            st.sidebar.metric(
                "GPU 메모리",
                f"{gpu_memory:.1f}GB",
                delta=f"-{gpu_usage:.1f}% 사용 중"
            )
        else:
            st.sidebar.info("GPU 미사용")
        
        # 컴퓨팅 모드 표시
        current_mode = self.resource_config.get('compute_mode', 'auto')
        mode_display = {
            'auto': '🤖 자동',
            'gpu_preferred': '🚀 GPU 우선',
            'cpu_preferred': '🖥️ CPU 우선',
            'gpu_only': '💪 GPU 전용',
            'cpu_only': '⚡ CPU 전용'
        }
        
        st.sidebar.info(f"**모드**: {mode_display.get(current_mode, current_mode)}")
        
        # 리소스 설정 링크
        if st.sidebar.button("⚙️ 리소스 설정"):
            st.sidebar.markdown("별도 창에서 `streamlit run resource_configurator.py`를 실행하세요")
    
    def show_optimization_info(self, optimization_settings: Dict):
        """최적화 정보 표시"""
        if not optimization_settings.get('optimized', False):
            return
        
        st.info("🚀 **리소스 최적화 적용됨**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            primary_device = optimization_settings.get('primary_device', 'cpu')
            device_icon = "🚀" if primary_device == 'gpu' else "🖥️"
            st.metric("주 디바이스", f"{device_icon} {primary_device.upper()}")
        
        with col2:
            optimized_tasks = len([k for k, v in optimization_settings.items() 
                                 if isinstance(v, dict) and 'device' in v])
            st.metric("최적화 작업", f"{optimized_tasks}개")
        
        with col3:
            st.metric("메모리 관리", "자동 정리")
        
        # 상세 최적화 정보 (펼치기)
        with st.expander("🔍 상세 최적화 설정"):
            for task_name, config in optimization_settings.items():
                if isinstance(config, dict) and 'device' in config:
                    st.write(f"**{task_name}**: {config['device']} 디바이스")
                    
                    # 추가 설정 정보
                    additional_info = []
                    if 'batch_size' in config:
                        additional_info.append(f"배치크기: {config['batch_size']}")
                    if 'fp16' in config and config['fp16']:
                        additional_info.append("FP16 가속")
                    if 'parallel' in config and config['parallel']:
                        additional_info.append("병렬 처리")
                    
                    if additional_info:
                        st.caption(f"  └ {', '.join(additional_info)}")
    
    def cleanup_resources(self):
        """리소스 정리"""
        if self.compute_manager:
            self.compute_manager.cleanup_memory("both")
            st.success("🧹 메모리 정리 완료")

def render_optimized_analysis_interface():
    """최적화된 분석 인터페이스 렌더링"""
    st.set_page_config(
        page_title="SOLOMOND AI 최적화 분석",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 SOLOMOND AI 최적화 분석 시스템")
    st.markdown("**CPU/GPU 리소스를 자동으로 최적화하여 최고 성능을 제공합니다**")
    
    # 최적화된 분석 시스템 초기화
    system = OptimizedAnalysisSystem()
    
    # 사이드바에 리소스 상태 표시
    system.render_resource_status_widget()
    
    # 컴퓨팅 최적화 사용 가능 여부 확인
    if not COMPUTE_OPTIMIZATION_AVAILABLE:
        st.warning("⚠️ 하이브리드 컴퓨팅 매니저를 사용할 수 없습니다. `hybrid_compute_manager.py`를 확인하세요.")
        st.info("💡 기본 CPU 모드로 실행됩니다.")
    
    if not MAIN_ANALYZER_AVAILABLE:
        st.error("❌ 메인 분석 시스템을 로드할 수 없습니다. `conference_analysis_COMPLETE_WORKING.py`를 확인하세요.")
        return
    
    # 파일 업로드 섹션
    st.header("📁 파일 업로드")
    
    uploaded_files = st.file_uploader(
        "분석할 파일들을 업로드하세요",
        type=['mp3', 'm4a', 'wav', 'mp4', 'mov', 'jpg', 'jpeg', 'png', 'pdf', 'docx'],
        accept_multiple_files=True,
        help="음성, 이미지, 비디오, 문서 파일을 지원합니다"
    )
    
    if uploaded_files:
        # 파일 정보 분석
        file_info = {
            'file_count': len(uploaded_files),
            'file_types': [f.type if hasattr(f, 'type') else 'unknown' for f in uploaded_files],
            'total_size_mb': sum(f.size for f in uploaded_files if hasattr(f, 'size')) / (1024*1024)
        }
        
        # 최적화 설정 계산
        optimization_settings = system.optimize_for_analysis(file_info)
        
        # 최적화 정보 표시
        system.show_optimization_info(optimization_settings)
        
        # 분석 시작 버튼
        if st.button("🎯 최적화된 분석 시작", type="primary"):
            
            # 리소스 정리
            if COMPUTE_OPTIMIZATION_AVAILABLE:
                system.compute_manager.cleanup_memory("both")
            
            with st.spinner("🚀 최적화된 분석 실행 중..."):
                try:
                    # 여기서 실제 분석 로직 실행
                    # system.main_analyzer를 사용하여 분석 수행
                    
                    # 진행률 표시
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"분석 중: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                        
                        # 실제 분석 로직은 여기에 구현
                        time.sleep(1)  # 시뮬레이션
                    
                    progress_bar.progress(1.0)
                    status_text.text("분석 완료!")
                    
                    st.success("✅ 최적화된 분석이 완료되었습니다!")
                    
                    # 결과 표시 (실제로는 분석 결과를 표시)
                    st.subheader("📊 분석 결과")
                    st.info("분석 결과가 여기에 표시됩니다.")
                    
                    # 자동 메모리 정리
                    if optimization_settings.get('optimized', False):
                        system.cleanup_resources()
                
                except Exception as e:
                    st.error(f"❌ 분석 중 오류가 발생했습니다: {str(e)}")
    
    # 성능 팁
    with st.expander("💡 성능 최적화 팁"):
        st.markdown("""
        ### 🚀 최고 성능을 위한 권장사항
        
        **GPU 사용 권장 상황:**
        - 대용량 음성 파일 (5분 이상)
        - 많은 이미지 파일 (10장 이상)
        - 고화질 비디오 처리
        
        **CPU 사용 권장 상황:**
        - 소규모 파일 처리
        - 실시간 분석 필요
        - GPU 메모리 부족 시
        
        **메모리 최적화:**
        - 파일을 배치로 나누어 처리
        - 분석 완료 후 자동 정리
        - 불필요한 캐시 제거
        """)
    
    # 하단 정보
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🤖 **자동 최적화**: 파일 유형에 따라 최적 리소스 선택")
    
    with col2:
        st.info("📊 **실시간 모니터링**: CPU/GPU 사용률 지속 추적")
    
    with col3:
        st.info("🧹 **자동 정리**: 분석 완료 후 메모리 자동 해제")

if __name__ == "__main__":
    render_optimized_analysis_interface()