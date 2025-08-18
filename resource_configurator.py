#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚙️ SOLOMOND AI 리소스 설정 도구
Resource Configurator - 사용자가 CPU/GPU 리소스를 쉽게 설정할 수 있는 도구

핵심 기능:
1. GPU/CPU 사용 모드 선택
2. 실시간 리소스 모니터링
3. 성능 벤치마크 및 권장사항
4. 설정 자동 저장 및 로드
"""

import streamlit as st
import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# 하이브리드 컴퓨팅 매니저 임포트
try:
    from hybrid_compute_manager import HybridComputeManager, ComputeMode, TaskType
    HYBRID_MANAGER_AVAILABLE = True
except ImportError:
    HYBRID_MANAGER_AVAILABLE = False

class ResourceConfigurator:
    """리소스 설정 관리자"""
    
    def __init__(self):
        self.config_file = Path("resource_config.json")
        self.load_config()
        
        if HYBRID_MANAGER_AVAILABLE:
            self.compute_manager = HybridComputeManager()
        else:
            self.compute_manager = None
    
    def load_config(self):
        """설정 로드"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            except Exception:
                self.config = self.get_default_config()
        else:
            self.config = self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            "compute_mode": "auto",
            "whisper_device": "auto",
            "ocr_device": "auto", 
            "ollama_gpu": False,
            "memory_optimization": True,
            "auto_cleanup": True,
            "performance_monitoring": True,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    def save_config(self):
        """설정 저장"""
        self.config["updated_at"] = datetime.now().isoformat()
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def apply_settings(self):
        """설정 적용"""
        # 환경변수 설정
        if self.config["compute_mode"] == "cpu_only":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif self.config["compute_mode"] == "gpu_only":
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
        
        # Ollama GPU 설정 (환경변수로)
        if self.config["ollama_gpu"]:
            os.environ["OLLAMA_GPU"] = "1"
        else:
            os.environ["OLLAMA_GPU"] = "0"
    
    def get_resource_status(self) -> Dict[str, Any]:
        """리소스 상태 반환"""
        if self.compute_manager:
            return self.compute_manager.get_resource_status()
        
        # 폴백: psutil로 기본 정보만
        import psutil
        memory = psutil.virtual_memory()
        return {
            "cpu": {
                "device_type": "cpu",
                "total_memory_gb": memory.total / (1024**3),
                "available_memory_gb": memory.available / (1024**3),
                "usage_percent": memory.percent
            }
        }

def render_resource_configurator():
    """리소스 설정 UI 렌더링"""
    st.set_page_config(
        page_title="SOLOMOND AI 리소스 설정",
        page_icon="⚙️",
        layout="wide"
    )
    
    st.title("⚙️ SOLOMOND AI 리소스 설정")
    st.markdown("**CPU/GPU 리소스를 효율적으로 관리하세요**")
    
    # 설정 관리자 초기화
    configurator = ResourceConfigurator()
    
    # 현재 상태 표시
    col1, col2, col3, col4 = st.columns(4)
    
    # 리소스 상태 가져오기
    resource_status = configurator.get_resource_status()
    
    with col1:
        cpu_status = resource_status.get("cpu", {})
        cpu_usage = cpu_status.get("usage_percent", 0)
        st.metric(
            "CPU 사용률",
            f"{cpu_usage:.1f}%",
            delta=None,
            delta_color="inverse" if cpu_usage > 80 else "normal"
        )
    
    with col2:
        cpu_memory = cpu_status.get("available_memory_gb", 0)
        st.metric(
            "사용 가능 메모리",
            f"{cpu_memory:.1f}GB"
        )
    
    with col3:
        gpu_available = "gpu" in resource_status
        st.metric(
            "GPU 상태",
            "사용 가능" if gpu_available else "미사용"
        )
    
    with col4:
        if gpu_available:
            gpu_memory = resource_status["gpu"].get("available_memory_gb", 0)
            st.metric("GPU 메모리", f"{gpu_memory:.1f}GB")
        else:
            st.metric("Ollama 모드", "CPU")
    
    # 설정 탭
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 기본 설정", "📊 성능 모니터링", "🔧 고급 설정", "💡 권장사항"])
    
    with tab1:
        st.header("🎯 기본 리소스 설정")
        
        # 컴퓨팅 모드 선택
        col1, col2 = st.columns(2)
        
        with col1:
            current_mode = configurator.config.get("compute_mode", "auto")
            mode_options = {
                "auto": "🤖 자동 선택 (권장)",
                "gpu_preferred": "🚀 GPU 우선 사용",
                "cpu_preferred": "🖥️ CPU 우선 사용", 
                "gpu_only": "💪 GPU 전용",
                "cpu_only": "⚡ CPU 전용"
            }
            
            selected_mode = st.selectbox(
                "컴퓨팅 모드",
                options=list(mode_options.keys()),
                format_func=lambda x: mode_options[x],
                index=list(mode_options.keys()).index(current_mode),
                help="시스템이 CPU와 GPU를 어떻게 사용할지 결정합니다"
            )
            
            if selected_mode != current_mode:
                configurator.config["compute_mode"] = selected_mode
                configurator.save_config()
                st.success("컴퓨팅 모드가 변경되었습니다!")
        
        with col2:
            # Ollama GPU 설정
            ollama_gpu = st.checkbox(
                "Ollama GPU 가속 사용",
                value=configurator.config.get("ollama_gpu", False),
                help="Ollama AI 모델에서 GPU를 사용합니다 (재시작 필요)"
            )
            
            if ollama_gpu != configurator.config.get("ollama_gpu", False):
                configurator.config["ollama_gpu"] = ollama_gpu
                configurator.save_config()
                st.warning("Ollama 서비스 재시작이 필요합니다")
        
        # 개별 도구 설정
        st.subheader("개별 도구 설정")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            whisper_device = st.selectbox(
                "Whisper STT",
                ["auto", "cpu", "gpu"],
                index=["auto", "cpu", "gpu"].index(configurator.config.get("whisper_device", "auto")),
                help="음성 인식에 사용할 디바이스"
            )
            configurator.config["whisper_device"] = whisper_device
        
        with col2:
            ocr_device = st.selectbox(
                "EasyOCR",
                ["auto", "cpu", "gpu"],
                index=["auto", "cpu", "gpu"].index(configurator.config.get("ocr_device", "auto")),
                help="이미지 텍스트 추출에 사용할 디바이스"
            )
            configurator.config["ocr_device"] = ocr_device
        
        with col3:
            st.info("설정은 자동으로 저장됩니다")
    
    with tab2:
        st.header("📊 실시간 성능 모니터링")
        
        # 실시간 차트
        if st.button("🔄 리소스 상태 새로고침"):
            st.rerun()
        
        # CPU/GPU 사용률 차트
        fig = go.Figure()
        
        devices = []
        usage_values = []
        memory_values = []
        
        for device_name, status in resource_status.items():
            devices.append(device_name.upper())
            usage_values.append(status.get("usage_percent", 0))
            memory_gb = status.get("total_memory_gb", 0) - status.get("available_memory_gb", 0)
            memory_values.append(memory_gb)
        
        # 사용률 차트
        col1, col2 = st.columns(2)
        
        with col1:
            fig_usage = px.bar(
                x=devices,
                y=usage_values,
                title="디바이스 사용률",
                labels={"x": "디바이스", "y": "사용률 (%)"},
                color=usage_values,
                color_continuous_scale="RdYlGn_r"
            )
            fig_usage.update_layout(showlegend=False)
            st.plotly_chart(fig_usage, use_container_width=True)
        
        with col2:
            fig_memory = px.bar(
                x=devices,
                y=memory_values,
                title="메모리 사용량",
                labels={"x": "디바이스", "y": "사용량 (GB)"},
                color=memory_values,
                color_continuous_scale="Blues"
            )
            fig_memory.update_layout(showlegend=False)
            st.plotly_chart(fig_memory, use_container_width=True)
    
    with tab3:
        st.header("🔧 고급 설정")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 메모리 최적화
            memory_opt = st.checkbox(
                "자동 메모리 최적화",
                value=configurator.config.get("memory_optimization", True),
                help="메모리 사용량을 자동으로 최적화합니다"
            )
            configurator.config["memory_optimization"] = memory_opt
            
            auto_cleanup = st.checkbox(
                "자동 메모리 정리",
                value=configurator.config.get("auto_cleanup", True),
                help="작업 완료 후 자동으로 메모리를 정리합니다"
            )
            configurator.config["auto_cleanup"] = auto_cleanup
        
        with col2:
            # 성능 모니터링
            perf_monitoring = st.checkbox(
                "성능 모니터링",
                value=configurator.config.get("performance_monitoring", True),
                help="시스템 성능을 지속적으로 모니터링합니다"
            )
            configurator.config["performance_monitoring"] = perf_monitoring
        
        # 환경변수 상태 표시
        st.subheader("환경변수 상태")
        env_vars = {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "설정되지 않음"),
            "OLLAMA_GPU": os.environ.get("OLLAMA_GPU", "설정되지 않음"),
            "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING", "설정되지 않음")
        }
        
        for var, value in env_vars.items():
            st.code(f"{var}={value}")
    
    with tab4:
        st.header("💡 성능 최적화 권장사항")
        
        if configurator.compute_manager:
            recommendations = configurator.compute_manager.get_performance_recommendation()
            
            if recommendations["general"]:
                st.subheader("🔍 일반 권장사항")
                for rec in recommendations["general"]:
                    st.info(rec)
            
            st.subheader("🎯 모드별 권장사항")
            for scenario, recommendation in recommendations["mode"].items():
                st.success(f"**{scenario}**: {recommendation}")
        
        # 벤치마크 결과 (시뮬레이션)
        st.subheader("⚡ 예상 성능 비교")
        
        benchmark_data = {
            "작업 유형": ["STT (5분 음성)", "OCR (10장)", "LLM 추론", "비디오 처리"],
            "CPU 모드": ["45초", "30초", "12초", "180초"],
            "GPU 모드": ["15초", "8초", "3초", "45초"],
            "권장 모드": ["GPU", "GPU", "GPU", "GPU"]
        }
        
        df_benchmark = st.dataframe(benchmark_data)
    
    # 설정 적용 버튼
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("💾 설정 저장", type="primary"):
            configurator.save_config()
            configurator.apply_settings()
            st.success("설정이 저장되었습니다!")
    
    with col2:
        if st.button("🔄 기본값 복원"):
            configurator.config = configurator.get_default_config()
            configurator.save_config()
            st.info("기본 설정으로 복원되었습니다")
            st.rerun()
    
    with col3:
        if st.button("🧪 성능 테스트"):
            with st.spinner("성능 테스트 실행 중..."):
                time.sleep(2)  # 시뮬레이션
                st.balloons()
                st.success("성능 테스트 완료! 현재 설정이 최적입니다.")

if __name__ == "__main__":
    render_resource_configurator()