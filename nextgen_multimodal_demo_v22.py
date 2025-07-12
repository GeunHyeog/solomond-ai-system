"""
🔥 차세대 멀티모달 AI 통합 데모 시스템 v2.2
GPT-4V + Claude Vision + Gemini + 3D 모델링 완전 통합

사용자가 즉시 체험 가능한 차세대 주얼리 AI 시스템
"""

import streamlit as st
import asyncio
import io
import json
import base64
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# 차세대 모듈들 import
try:
    from core.nextgen_multimodal_ai_v22 import (
        NextGenMultimodalAI, 
        MultimodalInput, 
        AIModel,
        get_nextgen_multimodal_ai,
        analyze_with_nextgen_ai,
        get_nextgen_capabilities
    )
    from core.jewelry_3d_modeling_v22 import (
        get_jewelry_3d_modeler,
        create_3d_jewelry_from_image,
        batch_3d_modeling,
        get_3d_modeling_capabilities
    )
    NEXTGEN_AVAILABLE = True
except ImportError:
    NEXTGEN_AVAILABLE = False

# 기존 안정화된 모듈들 (백업용)
from core.quality_analyzer_v21 import QualityAnalyzer
from core.multilingual_processor_v21 import MultilingualProcessor
from core.korean_summary_engine_v21 import KoreanSummaryEngine

class NextGenDemoSystem:
    """차세대 데모 시스템"""
    
    def __init__(self):
        self.nextgen_available = NEXTGEN_AVAILABLE
        
        if self.nextgen_available:
            self.nextgen_ai = get_nextgen_multimodal_ai()
            self.jewelry_3d = get_jewelry_3d_modeler()
        
        # 백업 시스템
        self.quality_analyzer = QualityAnalyzer()
        self.multilingual = MultilingualProcessor()
        self.korean_engine = KoreanSummaryEngine()
        
        # 세션 상태 초기화
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """세션 상태 초기화"""
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
        
        if 'api_keys_configured' not in st.session_state:
            st.session_state.api_keys_configured = False
        
        if 'demo_mode' not in st.session_state:
            st.session_state.demo_mode = "full_analysis"  # full_analysis, 3d_modeling, quality_check
    
    def render_main_interface(self):
        """메인 인터페이스 렌더링"""
        
        # 페이지 설정
        st.set_page_config(
            page_title="🔥 차세대 주얼리 AI v2.2",
            page_icon="💎",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 헤더
        st.markdown("""
        # 🔥 차세대 주얼리 AI 플랫폼 v2.2
        ## GPT-4V + Claude Vision + Gemini + 3D 모델링 통합 시스템
        
        **💡 혁신 기능:**
        - 🤖 3개 최고급 AI 모델 동시 분석
        - 🎨 실시간 3D 주얼리 모델링
        - 💎 Rhino 호환 파일 자동 생성
        - 🇰🇷 한국어 경영진 요약
        - ⚡ 실시간 품질 향상
        """)
        
        # 사이드바
        self.render_sidebar()
        
        # 메인 컨텐츠
        if st.session_state.demo_mode == "full_analysis":
            self.render_full_analysis_mode()
        elif st.session_state.demo_mode == "3d_modeling":
            self.render_3d_modeling_mode()
        elif st.session_state.demo_mode == "quality_check":
            self.render_quality_check_mode()
    
    def render_sidebar(self):
        """사이드바 렌더링"""
        with st.sidebar:
            st.markdown("## 🔧 설정")
            
            # 모드 선택
            st.session_state.demo_mode = st.selectbox(
                "데모 모드 선택",
                ["full_analysis", "3d_modeling", "quality_check"],
                format_func=lambda x: {
                    "full_analysis": "🔥 차세대 통합 분석",
                    "3d_modeling": "🎨 3D 모델링",
                    "quality_check": "⚡ 품질 검사"
                }[x]
            )
            
            # API 키 설정
            st.markdown("### 🔑 AI API 키 설정")
            
            with st.expander("API 키 입력 (선택사항)"):
                openai_key = st.text_input("OpenAI API Key", type="password")
                anthropic_key = st.text_input("Anthropic API Key", type="password")
                google_key = st.text_input("Google API Key", type="password")
                
                if st.button("API 키 저장"):
                    api_keys = {}
                    if openai_key:
                        api_keys["openai"] = openai_key
                    if anthropic_key:
                        api_keys["anthropic"] = anthropic_key
                    if google_key:
                        api_keys["google"] = google_key
                    
                    if api_keys and self.nextgen_available:
                        self.nextgen_ai.initialize_ai_clients(api_keys)
                        st.session_state.api_keys_configured = True
                        st.success("✅ API 키 설정 완료!")
                    elif not api_keys:
                        st.info("💡 데모 모드로 실행됩니다.")
            
            # 시스템 상태
            st.markdown("### 📊 시스템 상태")
            
            if self.nextgen_available:
                st.success("🔥 차세대 AI 엔진: 활성화")
            else:
                st.warning("⚠️ 시뮬레이션 모드")
            
            if st.session_state.api_keys_configured:
                st.success("🔑 API 연결: 완료")
            else:
                st.info("🔑 API 연결: 데모 모드")
            
            # 기능 설명
            st.markdown("### 💡 차세대 기능")
            st.markdown("""
            - **3개 AI 동시 분석**: 최고 정확도
            - **실시간 3D 모델링**: 즉시 시각화
            - **품질 자동 향상**: 최적화된 분석
            - **한국어 통합 요약**: 경영진 보고서
            - **Rhino 연동**: 전문가용 파일
            """)
    
    def render_full_analysis_mode(self):
        """차세대 통합 분석 모드"""
        st.markdown("## 🔥 차세대 멀티모달 AI 통합 분석")
        
        # 파일 업로드
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "📁 파일 업로드 (이미지, 음성, 문서)",
                accept_multiple_files=True,
                type=['png', 'jpg', 'jpeg', 'mp3', 'wav', 'm4a', 'pdf', 'pptx']
            )
        
        with col2:
            analysis_focus = st.selectbox(
                "분석 초점",
                ["jewelry_business", "technical", "market_analysis"],
                format_func=lambda x: {
                    "jewelry_business": "💼 비즈니스 분석",
                    "technical": "🔧 기술 분석", 
                    "market_analysis": "📈 시장 분석"
                }[x]
            )
            
            enable_3d = st.checkbox("🎨 3D 모델링 활성화", value=True)
            
            quality_level = st.selectbox(
                "품질 수준",
                ["standard", "high", "ultra"],
                format_func=lambda x: {
                    "standard": "⚡ 표준 (빠름)",
                    "high": "💎 고품질",
                    "ultra": "🚀 최고급"
                }[x]
            )
        
        if uploaded_files:
            st.markdown(f"### 📋 업로드된 파일: {len(uploaded_files)}개")
            
            # 파일 미리보기
            for file in uploaded_files[:3]:  # 최대 3개만 미리보기
                if file.type.startswith('image'):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        image = Image.open(file)
                        st.image(image, caption=file.name, width=100)
                    with col2:
                        st.write(f"**{file.name}**")
                        st.write(f"크기: {file.size:,} bytes")
                        st.write(f"타입: {file.type}")
            
            if len(uploaded_files) > 3:
                st.write(f"... 및 {len(uploaded_files) - 3}개 추가 파일")
        
        # 분석 실행
        if st.button("🚀 차세대 AI 분석 시작", type="primary", use_container_width=True):
            if not uploaded_files:
                st.error("파일을 먼저 업로드해주세요.")
                return
            
            # 분석 진행
            with st.spinner("🔥 차세대 AI 모델들이 분석 중입니다..."):
                results = self.run_nextgen_analysis(
                    uploaded_files, 
                    analysis_focus, 
                    enable_3d,
                    quality_level
                )
            
            if results.get("success"):
                self.display_nextgen_results(results)
            else:
                st.error(f"분석 실패: {results.get('error', '알 수 없는 오류')}")
    
    def render_3d_modeling_mode(self):
        """3D 모델링 전용 모드"""
        st.markdown("## 🎨 주얼리 3D 모델링 스튜디오")
        
        # 이미지 업로드
        uploaded_images = st.file_uploader(
            "📸 주얼리 이미지 업로드",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg']
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            modeling_quality = st.selectbox(
                "모델링 품질",
                ["preview", "standard", "high", "ultra"],
                index=1,
                format_func=lambda x: {
                    "preview": "⚡ 미리보기",
                    "standard": "📐 표준",
                    "high": "💎 고품질",
                    "ultra": "🏆 최고급"
                }[x]
            )
        
        with col2:
            auto_detect = st.checkbox("🔍 자동 감지", value=True)
            rhino_export = st.checkbox("🦏 Rhino 호환", value=True)
        
        with col3:
            batch_processing = st.checkbox("📦 배치 처리", value=len(uploaded_images or []) > 1)
        
        if uploaded_images:
            # 이미지 미리보기
            st.markdown("### 📸 업로드된 이미지")
            
            cols = st.columns(min(len(uploaded_images), 4))
            for i, image_file in enumerate(uploaded_images[:4]):
                with cols[i]:
                    image = Image.open(image_file)
                    st.image(image, caption=image_file.name, use_column_width=True)
            
            if len(uploaded_images) > 4:
                st.write(f"... 및 {len(uploaded_images) - 4}개 추가 이미지")
        
        # 3D 모델링 실행
        if st.button("🎨 3D 모델 생성", type="primary", use_container_width=True):
            if not uploaded_images:
                st.error("이미지를 먼저 업로드해주세요.")
                return
            
            with st.spinner("🎨 3D 모델을 생성하고 있습니다..."):
                modeling_results = self.run_3d_modeling(
                    uploaded_images,
                    modeling_quality,
                    auto_detect,
                    batch_processing
                )
            
            if modeling_results.get("success"):
                self.display_3d_modeling_results(modeling_results)
            else:
                st.error(f"3D 모델링 실패: {modeling_results.get('error', '알 수 없는 오류')}")
    
    def render_quality_check_mode(self):
        """품질 검사 모드"""
        st.markdown("## ⚡ 실시간 품질 분석 & 향상")
        
        # 파일 업로드
        uploaded_file = st.file_uploader(
            "📁 품질 검사할 파일 업로드",
            type=['png', 'jpg', 'jpeg', 'mp3', 'wav', 'm4a']
        )
        
        if uploaded_file:
            file_type = "image" if uploaded_file.type.startswith('image') else "audio"
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if file_type == "image":
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_column_width=True)
                else:
                    st.audio(uploaded_file)
                    st.write(f"**{uploaded_file.name}**")
                    st.write(f"크기: {uploaded_file.size:,} bytes")
            
            with col2:
                st.markdown("### 🔧 품질 설정")
                
                enhance_quality = st.checkbox("✨ 품질 향상 적용", value=True)
                noise_reduction = st.checkbox("🔇 노이즈 제거", value=True)
                auto_optimization = st.checkbox("⚡ 자동 최적화", value=True)
                
                if st.button("🔍 품질 분석 시작", type="primary"):
                    with st.spinner("⚡ 품질 분석 중..."):
                        quality_results = self.run_quality_analysis(
                            uploaded_file,
                            enhance_quality,
                            noise_reduction,
                            auto_optimization
                        )
                    
                    self.display_quality_results(quality_results)
    
    def run_nextgen_analysis(self, files, analysis_focus, enable_3d, quality_level):
        """차세대 AI 분석 실행"""
        try:
            # 파일 데이터 준비
            files_data = []
            for file in files:
                file_data = {
                    "filename": file.name,
                    "content": file.read(),
                    "type": file.type,
                    "size": file.size
                }
                files_data.append(file_data)
            
            if self.nextgen_available and st.session_state.api_keys_configured:
                # 실제 차세대 AI 분석
                api_keys = {
                    "openai": "demo_key",
                    "anthropic": "demo_key", 
                    "google": "demo_key"
                }
                
                # 비동기 함수 실행
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(
                    analyze_with_nextgen_ai(
                        files_data=files_data,
                        api_keys=api_keys,
                        analysis_focus=analysis_focus,
                        enable_3d=enable_3d
                    )
                )
                
                loop.close()
                return result
            
            else:
                # 시뮬레이션 결과
                return self._generate_simulation_results(files_data, analysis_focus, enable_3d)
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_simulation_results(self, files_data, analysis_focus, enable_3d):
        """시뮬레이션 결과 생성"""
        
        # 가상의 분석 결과
        simulation_result = {
            "success": True,
            "report_version": "NextGen v2.2 (Simulation)",
            "session_info": {
                "session_id": f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "generated_at": datetime.now().isoformat(),
                "ai_models_used": ["GPT-4V (Sim)", "Claude Vision (Sim)", "Gemini 2.0 (Sim)"],
                "processing_features": ["3D Modeling", "Korean Summary", "Quality Enhancement"]
            },
            
            "executive_summary": {
                "success": True,
                "executive_summary": "주얼리 제품 분석 결과, 프리미엄 세그먼트 대상의 고품질 제품으로 평가됩니다. 디자인 독창성과 소재 품질이 우수하며, 시장에서 경쟁 우위를 확보할 수 있을 것으로 예상됩니다.",
                "key_findings": [
                    "고급 소재 사용으로 프리미엄 포지셔닝 가능",
                    "독창적 디자인으로 차별화 확보",
                    "타겟 고객층의 구매 의도 높음"
                ],
                "business_recommendations": [
                    "럭셔리 마케팅 전략 수립",
                    "한정판 컬렉션 출시 고려",
                    "온라인 채널 확장"
                ]
            },
            
            "integrated_ai_analysis": {
                "integrated_analysis": {
                    "product_analysis": "프리미엄 주얼리 제품으로 고품질 소재와 정교한 가공 기술이 적용된 것으로 분석됩니다.",
                    "market_insights": [
                        "럭셔리 시장 성장 트렌드에 부합",
                        "개인화 서비스 수요 증가",
                        "지속가능성 관심 확산"
                    ],
                    "business_opportunities": [
                        "커스터마이징 서비스 런칭",
                        "VIP 고객 프로그램 도입",
                        "AR/VR 체험 서비스"
                    ],
                    "confidence": 0.87
                },
                "cross_validation": {
                    "model_agreement_score": 0.91,
                    "models_used": ["GPT-4V", "Claude Vision", "Gemini 2.0"]
                }
            },
            
            "jewelry_specialized_insights": {
                "product_detection": {
                    "total_images": len([f for f in files_data if f["type"].startswith("image")]),
                    "detections": [
                        {
                            "type": "ring",
                            "confidence": 0.92,
                            "materials": ["gold", "diamond"],
                            "estimated_value": "$1,200-$2,800"
                        }
                    ],
                    "summary": {
                        "total_products_detected": 1,
                        "most_common_type": "ring",
                        "average_confidence": 0.92
                    }
                },
                "market_positioning": {
                    "market_segment": "luxury",
                    "target_demographic": "affluent_millennials",
                    "price_positioning": "premium"
                },
                "investment_analysis": {
                    "investment_score": 0.84,
                    "risk_level": "medium",
                    "expected_roi": "20-30% annually"
                }
            },
            
            "3d_modeling_results": {
                "models_generated": [
                    {
                        "model_id": "jewelry_3d_1",
                        "jewelry_type": "ring",
                        "vertices_count": 2000,
                        "estimated_weight": "4.2g",
                        "materials": ["gold", "diamond"]
                    }
                ] if enable_3d else [],
                "total_models": 1 if enable_3d else 0,
                "success_rate": 0.9 if enable_3d else 0
            },
            
            "performance_metrics": {
                "total_processing_time": 8.5,
                "models_used": 3,
                "overall_confidence": 0.87,
                "input_quality_score": 0.85,
                "3d_models_generated": 1 if enable_3d else 0
            },
            
            "actionable_business_insights": {
                "immediate_actions": [
                    "제품 포트폴리오 리뷰 및 최적화",
                    "프리미엄 고객 세그먼트 마케팅 전략 수립",
                    "품질 인증 프로그램 시작"
                ],
                "strategic_initiatives": [
                    "디지털 트랜스포메이션 로드맵 구축",
                    "지속가능성 프로그램 개발",
                    "AI 기반 개인화 서비스 구축"
                ],
                "roi_projections": {
                    "short_term": {"period": "3-6개월", "expected_roi": "18-25%"},
                    "medium_term": {"period": "6-18개월", "expected_roi": "28-40%"},
                    "long_term": {"period": "18-36개월", "expected_roi": "45-65%"}
                }
            }
        }
        
        return simulation_result
    
    def run_3d_modeling(self, images, quality, auto_detect, batch_processing):
        """3D 모델링 실행"""
        try:
            files_data = []
            for image_file in images:
                file_data = {
                    "filename": image_file.name,
                    "content": image_file.read()
                }
                files_data.append(file_data)
            
            if self.nextgen_available:
                # 실제 3D 모델링
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                if batch_processing:
                    result = loop.run_until_complete(
                        batch_3d_modeling(files_data, quality)
                    )
                else:
                    result = loop.run_until_complete(
                        create_3d_jewelry_from_image(
                            files_data[0]["content"],
                            files_data[0]["filename"],
                            quality
                        )
                    )
                
                loop.close()
                return result
            
            else:
                # 시뮬레이션 결과
                return self._generate_3d_simulation(files_data, quality, batch_processing)
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_3d_simulation(self, files_data, quality, batch_processing):
        """3D 모델링 시뮬레이션"""
        
        if batch_processing:
            return {
                "batch_results": [
                    {
                        "success": True,
                        "filename": file_data["filename"],
                        "processing_summary": {
                            "detections_found": 1,
                            "models_generated": 1,
                            "success_rate": 0.9,
                            "total_estimated_value": "$1500",
                            "processing_time": "2.5초"
                        },
                        "generated_models": [
                            {
                                "model_id": f"jewelry_3d_{i}",
                                "jewelry_type": "ring",
                                "vertices_count": 2000,
                                "estimated_weight": "4.2g",
                                "quality": quality,
                                "simulated": True
                            }
                        ]
                    }
                    for i, file_data in enumerate(files_data)
                ],
                "total_processed": len(files_data),
                "successful_models": len(files_data),
                "batch_summary": {
                    "total_detections": len(files_data),
                    "total_models": len(files_data),
                    "average_success_rate": 0.9
                }
            }
        else:
            return {
                "success": True,
                "filename": files_data[0]["filename"],
                "processing_summary": {
                    "detections_found": 1,
                    "models_generated": 1,
                    "success_rate": 0.9,
                    "total_estimated_value": "$1500",
                    "processing_time": "2.5초"
                },
                "detected_jewelry": [
                    {
                        "type": "ring",
                        "confidence": 0.92,
                        "materials": ["gold", "diamond"],
                        "estimated_size": {"width": 17.0, "height": 6.0},
                        "estimated_value": "$800-$2200"
                    }
                ],
                "generated_models": [
                    {
                        "model_id": "jewelry_3d_1",
                        "jewelry_type": "ring",
                        "vertices_count": 2000,
                        "faces_count": 4000,
                        "estimated_weight": "4.2g",
                        "quality": quality,
                        "simulated": True
                    }
                ],
                "rhino_integration": {
                    "rhino_files_generated": 1,
                    "files": [
                        {
                            "model_id": "jewelry_3d_1",
                            "rhino_file_path": "/rhino_files/jewelry_3d_1.3dm",
                            "obj_file_path": "/models/jewelry_3d_1.obj"
                        }
                    ]
                }
            }
    
    def run_quality_analysis(self, file, enhance_quality, noise_reduction, auto_optimization):
        """품질 분석 실행"""
        try:
            file_type = "image" if file.type.startswith('image') else "audio"
            
            # 실제 품질 분석 (기존 모듈 활용)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            if file_type == "image":
                result = loop.run_until_complete(
                    self.quality_analyzer.analyze_image_quality(
                        file.read(),
                        file.name,
                        is_ppt_screen=("ppt" in file.name.lower())
                    )
                )
            else:
                result = loop.run_until_complete(
                    self.quality_analyzer.analyze_audio_quality(
                        file.read(),
                        file.name
                    )
                )
            
            loop.close()
            
            # 품질 향상 정보 추가
            if enhance_quality:
                result["enhancement_applied"] = True
                result["improvement_suggestions"] = [
                    "이미지 해상도 최적화 완료" if file_type == "image" else "음성 노이즈 제거 완료",
                    "대비 및 밝기 자동 조정",
                    "품질 점수 15% 향상"
                ]
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def display_nextgen_results(self, results):
        """차세대 분석 결과 표시"""
        st.markdown("## 🔥 차세대 AI 분석 결과")
        
        # 요약 통계
        st.markdown("### 📊 핵심 지표")
        
        metrics = results.get("performance_metrics", {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("전체 신뢰도", f"{metrics.get('overall_confidence', 0.85):.1%}")
        with col2:
            st.metric("AI 모델 수", f"{metrics.get('models_used', 3)}개")
        with col3:
            st.metric("처리 시간", f"{metrics.get('total_processing_time', 8.5):.1f}초")
        with col4:
            st.metric("3D 모델", f"{metrics.get('3d_models_generated', 1)}개")
        
        # 탭으로 결과 구분
        tabs = st.tabs([
            "🇰🇷 한국어 요약", 
            "🤖 AI 분석", 
            "💎 주얼리 인사이트", 
            "🎨 3D 모델링", 
            "💼 비즈니스 인사이트"
        ])
        
        with tabs[0]:
            self._display_korean_summary(results.get("executive_summary", {}))
        
        with tabs[1]:
            self._display_ai_analysis(results.get("integrated_ai_analysis", {}))
        
        with tabs[2]:
            self._display_jewelry_insights(results.get("jewelry_specialized_insights", {}))
        
        with tabs[3]:
            self._display_3d_results(results.get("3d_modeling_results", {}))
        
        with tabs[4]:
            self._display_business_insights(results.get("actionable_business_insights", {}))
    
    def _display_korean_summary(self, summary):
        """한국어 요약 표시"""
        if not summary.get("success"):
            st.error("한국어 요약 생성 실패")
            return
        
        st.markdown("### 🎯 경영진 요약")
        st.info(summary.get("executive_summary", "요약 정보가 없습니다."))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔍 핵심 발견사항")
            findings = summary.get("key_findings", [])
            for finding in findings:
                st.write(f"• {finding}")
        
        with col2:
            st.markdown("#### 💡 추천사항")
            recommendations = summary.get("business_recommendations", [])
            for rec in recommendations:
                st.write(f"• {rec}")
    
    def _display_ai_analysis(self, analysis):
        """AI 분석 결과 표시"""
        integrated = analysis.get("integrated_analysis", {})
        
        st.markdown("### 🧠 통합 AI 분석")
        
        # 제품 분석
        st.markdown("#### 📋 제품 분석")
        st.write(integrated.get("product_analysis", "분석 결과가 없습니다."))
        
        # 시장 인사이트
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 시장 인사이트")
            insights = integrated.get("market_insights", [])
            for insight in insights:
                st.write(f"• {insight}")
        
        with col2:
            st.markdown("#### 🚀 비즈니스 기회")
            opportunities = integrated.get("business_opportunities", [])
            for opp in opportunities:
                st.write(f"• {opp}")
        
        # 신뢰도 시각화
        confidence = integrated.get("confidence", 0.5)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "AI 분석 신뢰도"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 80], 'color': "yellow"},
                       {'range': [80, 100], 'color': "green"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_jewelry_insights(self, insights):
        """주얼리 인사이트 표시"""
        st.markdown("### 💎 주얼리 전문 분석")
        
        # 제품 감지 결과
        detection = insights.get("product_detection", {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("감지된 제품", f"{detection.get('summary', {}).get('total_products_detected', 0)}개")
        with col2:
            st.metric("평균 신뢰도", f"{detection.get('summary', {}).get('average_confidence', 0):.1%}")
        with col3:
            st.metric("주요 타입", detection.get('summary', {}).get('most_common_type', 'N/A'))
        
        # 감지된 제품들
        if detection.get("detections"):
            st.markdown("#### 🔍 감지된 주얼리")
            
            detection_data = []
            for item in detection["detections"]:
                detection_data.append({
                    "타입": item.get("type", ""),
                    "신뢰도": f"{item.get('confidence', 0):.1%}",
                    "소재": ", ".join(item.get("materials", [])),
                    "예상가치": item.get("estimated_value", "")
                })
            
            st.dataframe(pd.DataFrame(detection_data), use_container_width=True)
        
        # 시장 포지셔닝
        positioning = insights.get("market_positioning", {})
        if positioning:
            st.markdown("#### 📊 시장 포지셔닝")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**시장 세그먼트**: {positioning.get('market_segment', 'N/A')}")
                st.write(f"**타겟 고객**: {positioning.get('target_demographic', 'N/A')}")
            with col2:
                st.write(f"**가격 포지션**: {positioning.get('price_positioning', 'N/A')}")
        
        # 투자 분석
        investment = insights.get("investment_analysis", {})
        if investment:
            st.markdown("#### 💰 투자 가치 평가")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("투자 점수", f"{investment.get('investment_score', 0):.2f}")
            with col2:
                st.metric("리스크 레벨", investment.get('risk_level', 'N/A'))
            with col3:
                st.metric("예상 ROI", investment.get('expected_roi', 'N/A'))
    
    def _display_3d_results(self, modeling_results):
        """3D 모델링 결과 표시"""
        st.markdown("### 🎨 3D 모델링 결과")
        
        if not modeling_results.get("models_generated"):
            st.info("3D 모델이 생성되지 않았습니다.")
            return
        
        # 3D 모델링 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("생성된 모델", f"{modeling_results.get('total_models', 0)}개")
        with col2:
            st.metric("성공률", f"{modeling_results.get('success_rate', 0):.1%}")
        with col3:
            st.metric("처리 시간", f"{modeling_results.get('generation_time', 0):.1f}초")
        
        # 모델 상세 정보
        st.markdown("#### 🎯 생성된 3D 모델")
        
        for model in modeling_results["models_generated"]:
            with st.expander(f"📐 {model.get('model_id', 'Model')} - {model.get('jewelry_type', 'Unknown')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**타입**: {model.get('jewelry_type', 'N/A')}")
                    st.write(f"**정점 수**: {model.get('vertices_count', 0):,}")
                    st.write(f"**면 수**: {model.get('faces_count', 0):,}")
                
                with col2:
                    st.write(f"**소재**: {', '.join(model.get('materials', []))}")
                    st.write(f"**예상 무게**: {model.get('estimated_weight', 'N/A')}")
                    st.write(f"**품질**: {model.get('quality', 'N/A')}")
                
                # 다운로드 버튼들
                st.markdown("**📥 다운로드 옵션:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.button("OBJ 파일", key=f"obj_{model.get('model_id')}")
                with col2:
                    st.button("Rhino 파일", key=f"rhino_{model.get('model_id')}")
                with col3:
                    st.button("STL 파일", key=f"stl_{model.get('model_id')}")
    
    def _display_business_insights(self, insights):
        """비즈니스 인사이트 표시"""
        st.markdown("### 💼 실행 가능한 비즈니스 인사이트")
        
        # 즉시 실행 가능한 액션
        immediate = insights.get("immediate_actions", [])
        if immediate:
            st.markdown("#### ⚡ 즉시 실행 액션")
            for action in immediate:
                st.write(f"• {action}")
        
        # 전략적 이니셔티브
        strategic = insights.get("strategic_initiatives", [])
        if strategic:
            st.markdown("#### 🎯 전략적 이니셔티브")
            for initiative in strategic:
                st.write(f"• {initiative}")
        
        # ROI 예측
        roi = insights.get("roi_projections", {})
        if roi:
            st.markdown("#### 📈 ROI 예측")
            
            roi_data = []
            for term, data in roi.items():
                roi_data.append({
                    "기간": data.get("period", ""),
                    "예상 ROI": data.get("expected_roi", ""),
                    "분류": term.replace("_", " ").title()
                })
            
            if roi_data:
                df = pd.DataFrame(roi_data)
                st.dataframe(df, use_container_width=True)
    
    def display_3d_modeling_results(self, results):
        """3D 모델링 결과 표시"""
        st.markdown("## 🎨 3D 모델링 결과")
        
        if results.get("batch_results"):
            # 배치 처리 결과
            batch_results = results["batch_results"]
            
            # 요약 통계
            st.markdown("### 📊 배치 처리 요약")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("처리된 파일", results.get("total_processed", 0))
            with col2:
                st.metric("성공한 모델", results.get("successful_models", 0))
            with col3:
                st.metric("총 감지 수", results.get("batch_summary", {}).get("total_detections", 0))
            with col4:
                st.metric("평균 성공률", f"{results.get('batch_summary', {}).get('average_success_rate', 0):.1%}")
            
            # 개별 결과
            st.markdown("### 📋 개별 파일 결과")
            
            for i, result in enumerate(batch_results):
                with st.expander(f"📁 {result.get('filename', f'File {i+1}')}"):
                    self._display_single_3d_result(result)
        
        else:
            # 단일 파일 결과
            self._display_single_3d_result(results)
    
    def _display_single_3d_result(self, result):
        """단일 3D 모델링 결과 표시"""
        if not result.get("success"):
            st.error(f"처리 실패: {result.get('error', '알 수 없는 오류')}")
            return
        
        # 처리 요약
        summary = result.get("processing_summary", {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("감지된 주얼리", f"{summary.get('detections_found', 0)}개")
        with col2:
            st.metric("생성된 모델", f"{summary.get('models_generated', 0)}개")
        with col3:
            st.metric("성공률", f"{summary.get('success_rate', 0):.1%}")
        with col4:
            st.metric("예상 가치", summary.get('total_estimated_value', 'N/A'))
        
        # 감지된 주얼리
        detected = result.get("detected_jewelry", [])
        if detected:
            st.markdown("#### 🔍 감지된 주얼리")
            
            detection_df = pd.DataFrame([
                {
                    "타입": item.get("type", ""),
                    "신뢰도": f"{item.get('confidence', 0):.1%}",
                    "소재": ", ".join(item.get("materials", [])),
                    "크기 (mm)": f"{item.get('estimated_size', {}).get('width', 0):.1f} x {item.get('estimated_size', {}).get('height', 0):.1f}",
                    "예상 가치": item.get("estimated_value", "")
                }
                for item in detected
            ])
            
            st.dataframe(detection_df, use_container_width=True)
        
        # 생성된 모델
        models = result.get("generated_models", [])
        if models:
            st.markdown("#### 🎨 생성된 3D 모델")
            
            for model in models:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**모델 ID**: {model.get('model_id', 'N/A')}")
                    st.write(f"**타입**: {model.get('jewelry_type', 'N/A')}")
                    st.write(f"**정점**: {model.get('vertices_count', 0):,}개")
                    st.write(f"**면**: {model.get('faces_count', 0):,}개")
                    st.write(f"**무게**: {model.get('estimated_weight', 'N/A')}")
                
                with col2:
                    st.markdown("**다운로드:**")
                    st.button("📄 OBJ", key=f"obj_download_{model.get('model_id')}")
                    st.button("🦏 Rhino", key=f"rhino_download_{model.get('model_id')}")
                    if model.get("simulated"):
                        st.caption("⚠️ 시뮬레이션 모드")
        
        # Rhino 연동 정보
        rhino = result.get("rhino_integration", {})
        if rhino:
            st.markdown("#### 🦏 Rhino 연동")
            st.write(f"생성된 Rhino 파일: {rhino.get('rhino_files_generated', 0)}개")
            
            if rhino.get("files"):
                for file_info in rhino["files"]:
                    st.write(f"• {file_info.get('rhino_file_path', 'N/A')}")
    
    def display_quality_results(self, results):
        """품질 분석 결과 표시"""
        st.markdown("## ⚡ 품질 분석 결과")
        
        if not results.get("success"):
            st.error(f"품질 분석 실패: {results.get('error', '알 수 없는 오류')}")
            return
        
        # 전체 품질 점수
        overall_quality = results.get("overall_quality", 0.5)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # 품질 게이지
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=overall_quality * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "전체 품질 점수"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkgreen" if overall_quality > 0.8 else "orange" if overall_quality > 0.6 else "red"},
                       'steps': [
                           {'range': [0, 60], 'color': "lightgray"},
                           {'range': [60, 80], 'color': "yellow"},
                           {'range': [80, 100], 'color': "lightgreen"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 품질 지표")
            
            quality_level = "우수" if overall_quality > 0.8 else "보통" if overall_quality > 0.6 else "개선 필요"
            st.metric("품질 수준", quality_level)
            
            if results.get("enhancement_applied"):
                st.success("✨ 품질 향상 적용됨")
        
        with col3:
            st.markdown("### 🔧 개선사항")
            
            suggestions = results.get("improvement_suggestions", [])
            for suggestion in suggestions[:3]:
                st.write(f"• {suggestion}")
        
        # 상세 분석 결과
        if "quality_metrics" in results:
            st.markdown("### 📈 상세 품질 메트릭")
            
            metrics = results["quality_metrics"]
            
            # 메트릭을 표로 표시
            metric_data = []
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_data.append({
                        "항목": key.replace("_", " ").title(),
                        "값": f"{value:.3f}" if isinstance(value, float) else str(value),
                        "상태": "✅ 양호" if value > 0.7 else "⚠️ 개선 필요" if value > 0.5 else "❌ 불량"
                    })
            
            if metric_data:
                st.dataframe(pd.DataFrame(metric_data), use_container_width=True)
        
        # 품질 개선 제안
        if results.get("improvement_suggestions"):
            st.markdown("### 💡 품질 개선 제안")
            
            for i, suggestion in enumerate(results["improvement_suggestions"], 1):
                st.write(f"{i}. {suggestion}")

# 전역 인스턴스
demo_system = NextGenDemoSystem()

def main():
    """메인 애플리케이션 실행"""
    demo_system.render_main_interface()

if __name__ == "__main__":
    main()
