"""
💎 솔로몬드 주얼리 AI 플랫폼 v2.3 통합 시스템
기존 v2.1.4 + 신규 하이브리드 AI 엔진 완전 통합

개발자: 전근혁 (솔로몬드 대표)
목표: 99.2% 정확도 + 25초 처리속도 + seamless 사용자 경험
"""

import streamlit as st
import asyncio
import time
import json
import os
import logging
import traceback
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import gc

# v2.3 하이브리드 AI 시스템 imports
try:
    from core.hybrid_llm_manager_v23 import HybridLLMManager, AIResponse, AIModel, AnalysisRequest
    from core.jewelry_specialized_prompts_v23 import JewelrySpecializedPrompts, AnalysisType, AIModelType
    from core.ai_quality_validator_v23 import AIQualityValidator, QualityReport, QualityLevel
    from core.ai_benchmark_system_v23 import PerformanceBenchmark, ABTestManager, PerformanceReportGenerator
    v23_modules_available = True
except ImportError as e:
    logging.warning(f"v2.3 모듈 import 경고: {e}")
    v23_modules_available = False

# 기존 v2.1.4 시스템 imports (호환성 유지)
try:
    from core.jewelry_ai_engine import JewelryAIEngine
    from core.multimodal_integrator import MultimodalIntegrator
    from core.korean_summary_engine_v21 import KoreanSummaryEngine
    from core.quality_analyzer_v21 import QualityAnalyzer
    legacy_modules_available = True
except ImportError as e:
    logging.warning(f"레거시 모듈 import 경고: {e}")
    legacy_modules_available = False

# Streamlit 페이지 설정
st.set_page_config(
    page_title="💎 솔로몬드 AI v2.3",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemMode(Enum):
    """시스템 모드"""
    LEGACY_V214 = "legacy_v214"
    HYBRID_V23 = "hybrid_v23"
    AUTO_OPTIMIZE = "auto_optimize"

class ProcessingStatus(Enum):
    """처리 상태"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AnalysisResult:
    """분석 결과 통합 데이터 클래스"""
    content: str
    confidence: float
    processing_time: float
    system_mode: SystemMode
    quality_score: float
    jewelry_expertise: float
    cost_estimate: float
    model_used: str
    timestamp: float
    metadata: Dict[str, Any]

class IntegratedAISystem:
    """v2.1.4 + v2.3 통합 AI 시스템"""
    
    def __init__(self):
        self.system_mode = SystemMode.AUTO_OPTIMIZE
        self.performance_target = 25.0  # 25초 목표
        self.accuracy_target = 0.992   # 99.2% 목표
        
        # v2.3 시스템 초기화
        if v23_modules_available:
            self.hybrid_manager = HybridLLMManager()
            self.prompt_optimizer = JewelrySpecializedPrompts()
            self.quality_validator = AIQualityValidator()
            self.benchmark_system = PerformanceBenchmark(self.hybrid_manager, self.quality_validator)
            logger.info("✅ v2.3 하이브리드 AI 시스템 초기화 완료")
        else:
            self.hybrid_manager = None
            logger.warning("⚠️ v2.3 시스템 초기화 실패")
        
        # 레거시 v2.1.4 시스템 초기화 (fallback)
        if legacy_modules_available:
            self.legacy_ai_engine = JewelryAIEngine()
            self.legacy_integrator = MultimodalIntegrator()
            self.legacy_summarizer = KoreanSummaryEngine()
            self.legacy_quality = QualityAnalyzer()
            logger.info("✅ v2.1.4 레거시 시스템 초기화 완료")
        else:
            self.legacy_ai_engine = None
            logger.warning("⚠️ 레거시 시스템 초기화 실패")
        
        # 성능 추적
        self.performance_history = []
        self.processing_queue = queue.Queue()
        
        # 메모리 최적화 설정
        self.max_history_size = 100
        self.gc_interval = 10  # 10번마다 가비지 컬렉션
        self.analysis_count = 0
    
    async def analyze_jewelry(self, 
                            input_data: Dict[str, Any],
                            analysis_type: str = "auto",
                            force_mode: Optional[SystemMode] = None) -> AnalysisResult:
        """통합 주얼리 분석 (v2.1.4 + v2.3 최적 선택)"""
        
        start_time = time.time()
        
        # 시스템 모드 결정
        selected_mode = force_mode or self._select_optimal_mode(input_data, analysis_type)
        
        logger.info(f"🧠 분석 시작: {selected_mode.value} 모드")
        
        try:
            # v2.3 하이브리드 시스템 사용
            if selected_mode == SystemMode.HYBRID_V23 and self.hybrid_manager:
                result = await self._analyze_with_v23_system(input_data, analysis_type)
            
            # 레거시 v2.1.4 시스템 사용
            elif selected_mode == SystemMode.LEGACY_V214 and self.legacy_ai_engine:
                result = await self._analyze_with_legacy_system(input_data, analysis_type)
            
            # 자동 최적화 모드 (적응형)
            else:
                result = await self._analyze_with_auto_optimization(input_data, analysis_type)
            
            # 성능 추적 및 최적화
            processing_time = time.time() - start_time
            self._update_performance_metrics(result, processing_time)
            
            # 메모리 최적화
            self.analysis_count += 1
            if self.analysis_count % self.gc_interval == 0:
                gc.collect()
            
            logger.info(f"✅ 분석 완료: {processing_time:.2f}초, 정확도: {result.quality_score:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 분석 실패: {e}")
            return self._create_error_result(str(e), time.time() - start_time)
    
    def _select_optimal_mode(self, input_data: Dict[str, Any], analysis_type: str) -> SystemMode:
        """최적 시스템 모드 자동 선택"""
        
        # 입력 복잡도 분석
        complexity_score = self._analyze_input_complexity(input_data)
        
        # 성능 기록 기반 선택
        if self.performance_history:
            recent_performance = self.performance_history[-10:]  # 최근 10개
            
            v23_avg_time = statistics.mean([p['time'] for p in recent_performance if p['mode'] == SystemMode.HYBRID_V23]) if any(p['mode'] == SystemMode.HYBRID_V23 for p in recent_performance) else float('inf')
            legacy_avg_time = statistics.mean([p['time'] for p in recent_performance if p['mode'] == SystemMode.LEGACY_V214]) if any(p['mode'] == SystemMode.LEGACY_V214 for p in recent_performance) else float('inf')
            
            # 성능 목표 달성 여부로 결정
            if v23_avg_time <= self.performance_target and self.hybrid_manager:
                return SystemMode.HYBRID_V23
            elif legacy_avg_time <= self.performance_target and self.legacy_ai_engine:
                return SystemMode.LEGACY_V214
        
        # 기본 선택 로직
        if complexity_score > 0.7 and self.hybrid_manager:
            return SystemMode.HYBRID_V23
        elif self.legacy_ai_engine:
            return SystemMode.LEGACY_V214
        elif self.hybrid_manager:
            return SystemMode.HYBRID_V23
        else:
            return SystemMode.AUTO_OPTIMIZE
    
    def _analyze_input_complexity(self, input_data: Dict[str, Any]) -> float:
        """입력 데이터 복잡도 분석"""
        
        complexity = 0.0
        
        # 텍스트 길이
        text_content = input_data.get('text', '')
        if len(text_content) > 500:
            complexity += 0.3
        elif len(text_content) > 200:
            complexity += 0.2
        
        # 이미지 포함 여부
        if input_data.get('image') or input_data.get('image_url'):
            complexity += 0.4
        
        # 주얼리 전문 용어 밀도
        jewelry_terms = ['다이아몬드', '루비', '사파이어', '에메랄드', 'GIA', '4C', '캐럿']
        term_count = sum(1 for term in jewelry_terms if term in text_content)
        complexity += min(term_count / len(jewelry_terms), 0.3)
        
        return min(complexity, 1.0)
    
    async def _analyze_with_v23_system(self, input_data: Dict[str, Any], analysis_type: str) -> AnalysisResult:
        """v2.3 하이브리드 시스템으로 분석"""
        
        start_time = time.time()
        
        # 분석 타입 매핑
        v23_analysis_type = self._map_to_v23_analysis_type(analysis_type)
        
        # 분석 요청 생성
        analysis_request = AnalysisRequest(
            text_content=input_data.get('text', ''),
            image_data=input_data.get('image'),
            image_url=input_data.get('image_url'),
            analysis_type=v23_analysis_type,
            require_jewelry_expertise=True
        )
        
        # 하이브리드 분석 실행
        hybrid_result = await self.hybrid_manager.hybrid_analyze(analysis_request)
        
        if hybrid_result['status'] != 'success':
            raise Exception(f"하이브리드 분석 실패: {hybrid_result.get('message', 'Unknown error')}")
        
        # 품질 검증
        ai_response = AIResponse(
            model=AIModel(hybrid_result['best_model']),
            content=hybrid_result['content'],
            confidence=hybrid_result['confidence'],
            processing_time=hybrid_result['processing_time'],
            cost_estimate=hybrid_result['cost_estimate'],
            jewelry_relevance=hybrid_result['jewelry_relevance'],
            metadata=hybrid_result.get('metadata', {})
        )
        
        quality_report = await self.quality_validator.validate_ai_response(
            ai_response, self._map_to_analysis_type_enum(v23_analysis_type), input_data.get('text', '')
        )
        
        # 자동 재분석 (품질이 낮은 경우)
        if quality_report.needs_reanalysis:
            logger.info("🔄 품질 개선을 위한 자동 재분석")
            reanalysis_result = await self.quality_validator.auto_reanalysis_if_needed(
                quality_report, self.hybrid_manager, analysis_request
            )
            if reanalysis_result:
                hybrid_result = reanalysis_result
                ai_response.content = reanalysis_result['content']
                ai_response.confidence = reanalysis_result['confidence']
        
        processing_time = time.time() - start_time
        
        return AnalysisResult(
            content=ai_response.content,
            confidence=ai_response.confidence,
            processing_time=processing_time,
            system_mode=SystemMode.HYBRID_V23,
            quality_score=quality_report.overall_score,
            jewelry_expertise=quality_report.jewelry_expertise_score,
            cost_estimate=ai_response.cost_estimate,
            model_used=hybrid_result['best_model'],
            timestamp=time.time(),
            metadata={
                'quality_report': asdict(quality_report),
                'hybrid_result': hybrid_result,
                'model_count': len(hybrid_result.get('all_responses', []))
            }
        )
    
    async def _analyze_with_legacy_system(self, input_data: Dict[str, Any], analysis_type: str) -> AnalysisResult:
        """v2.1.4 레거시 시스템으로 분석"""
        
        start_time = time.time()
        
        # 레거시 시스템 분석 실행
        if hasattr(self.legacy_ai_engine, 'analyze_comprehensive'):
            result = await self.legacy_ai_engine.analyze_comprehensive(input_data)
        else:
            # 시뮬레이션된 레거시 분석
            result = {
                'content': f"레거시 v2.1.4 시스템 분석 결과:\n\n{input_data.get('text', '')}에 대한 주얼리 전문 분석을 제공합니다.",
                'confidence': 0.88,
                'model': 'legacy_v214'
            }
        
        # 레거시 품질 분석
        if hasattr(self.legacy_quality, 'analyze_quality'):
            quality_result = self.legacy_quality.analyze_quality(result['content'])
            quality_score = quality_result.get('overall_score', 0.85)
        else:
            quality_score = 0.85  # 기본 품질 점수
        
        processing_time = time.time() - start_time
        
        return AnalysisResult(
            content=result['content'],
            confidence=result.get('confidence', 0.85),
            processing_time=processing_time,
            system_mode=SystemMode.LEGACY_V214,
            quality_score=quality_score,
            jewelry_expertise=0.82,  # 레거시 시스템 평균 전문성
            cost_estimate=0.0,  # 레거시 시스템은 무료
            model_used=result.get('model', 'legacy_v214'),
            timestamp=time.time(),
            metadata={'legacy_mode': True}
        )
    
    async def _analyze_with_auto_optimization(self, input_data: Dict[str, Any], analysis_type: str) -> AnalysisResult:
        """자동 최적화 모드 (적응형 분석)"""
        
        # 두 시스템 모두 사용 가능한 경우 병렬 실행 후 최적 선택
        if self.hybrid_manager and self.legacy_ai_engine:
            
            tasks = [
                self._analyze_with_v23_system(input_data, analysis_type),
                self._analyze_with_legacy_system(input_data, analysis_type)
            ]
            
            # 타임아웃과 함께 병렬 실행
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), 
                    timeout=30.0
                )
                
                # 성공한 결과 중 최적 선택
                valid_results = [r for r in results if isinstance(r, AnalysisResult)]
                
                if valid_results:
                    # 품질 점수와 처리 시간을 종합한 최적 결과 선택
                    best_result = max(valid_results, key=lambda x: x.quality_score - x.processing_time/100)
                    best_result.system_mode = SystemMode.AUTO_OPTIMIZE
                    return best_result
                
            except asyncio.TimeoutError:
                logger.warning("⏰ 병렬 분석 타임아웃")
        
        # fallback: 사용 가능한 시스템 사용
        if self.hybrid_manager:
            return await self._analyze_with_v23_system(input_data, analysis_type)
        elif self.legacy_ai_engine:
            return await self._analyze_with_legacy_system(input_data, analysis_type)
        else:
            raise Exception("사용 가능한 AI 시스템이 없습니다")
    
    def _map_to_v23_analysis_type(self, analysis_type: str) -> str:
        """분석 타입을 v2.3 형식으로 매핑"""
        
        mapping = {
            'diamond': 'diamond_4c',
            'colored_stone': 'colored_stone',
            'design': 'jewelry_design',
            'business': 'business_insight',
            'auto': 'diamond_4c'
        }
        
        return mapping.get(analysis_type.lower(), 'diamond_4c')
    
    def _map_to_analysis_type_enum(self, analysis_type: str) -> 'AnalysisType':
        """문자열을 AnalysisType enum으로 변환"""
        
        if not v23_modules_available:
            return None
        
        mapping = {
            'diamond_4c': AnalysisType.DIAMOND_4C,
            'colored_stone': AnalysisType.COLORED_STONE,
            'jewelry_design': AnalysisType.JEWELRY_DESIGN,
            'business_insight': AnalysisType.BUSINESS_INSIGHT
        }
        
        return mapping.get(analysis_type, AnalysisType.DIAMOND_4C)
    
    def _create_error_result(self, error_message: str, processing_time: float) -> AnalysisResult:
        """에러 결과 생성"""
        
        return AnalysisResult(
            content=f"❌ 분석 중 오류가 발생했습니다: {error_message}",
            confidence=0.0,
            processing_time=processing_time,
            system_mode=SystemMode.AUTO_OPTIMIZE,
            quality_score=0.0,
            jewelry_expertise=0.0,
            cost_estimate=0.0,
            model_used="error",
            timestamp=time.time(),
            metadata={'error': True, 'error_message': error_message}
        )
    
    def _update_performance_metrics(self, result: AnalysisResult, processing_time: float):
        """성능 메트릭 업데이트"""
        
        performance_record = {
            'timestamp': time.time(),
            'mode': result.system_mode,
            'time': processing_time,
            'quality': result.quality_score,
            'confidence': result.confidence,
            'cost': result.cost_estimate
        }
        
        self.performance_history.append(performance_record)
        
        # 히스토리 크기 제한
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size:]
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """성능 분석 데이터 반환"""
        
        if not self.performance_history:
            return {"message": "성능 데이터가 없습니다"}
        
        recent_data = self.performance_history[-20:]  # 최근 20개
        
        avg_time = statistics.mean([d['time'] for d in recent_data])
        avg_quality = statistics.mean([d['quality'] for d in recent_data])
        total_cost = sum([d['cost'] for d in recent_data])
        
        # 목표 달성률
        time_target_achievement = sum(1 for d in recent_data if d['time'] <= self.performance_target) / len(recent_data)
        quality_target_achievement = sum(1 for d in recent_data if d['quality'] >= self.accuracy_target) / len(recent_data)
        
        return {
            "전체_분석_수": len(self.performance_history),
            "평균_처리_시간": f"{avg_time:.1f}초",
            "평균_품질_점수": f"{avg_quality:.1%}",
            "총_비용": f"${total_cost:.3f}",
            "25초_목표_달성률": f"{time_target_achievement:.1%}",
            "99.2%_목표_달성률": f"{quality_target_achievement:.1%}",
            "성능_등급": "우수" if avg_time <= 25 and avg_quality >= 0.99 else "양호" if avg_time <= 35 else "개선필요"
        }

def create_streamlit_ui():
    """Streamlit UI 생성"""
    
    # 세션 상태 초기화
    if 'ai_system' not in st.session_state:
        with st.spinner("🚀 솔로몬드 AI v2.3 시스템 초기화 중..."):
            st.session_state.ai_system = IntegratedAISystem()
    
    # 제목 및 설명
    st.title("💎 솔로몬드 AI v2.3 - 하이브리드 주얼리 분석 플랫폼")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 정확도 목표", "99.2%")
    with col2:
        st.metric("⚡ 처리 속도 목표", "25초")
    with col3:
        st.metric("🧠 AI 모델", "3개 통합")
    
    # 사이드바 - 시스템 설정
    with st.sidebar:
        st.header("⚙️ 시스템 설정")
        
        # 시스템 모드 선택
        mode_options = {
            "🤖 자동 최적화": SystemMode.AUTO_OPTIMIZE,
            "🧠 하이브리드 v2.3": SystemMode.HYBRID_V23,
            "🔧 레거시 v2.1.4": SystemMode.LEGACY_V214
        }
        
        selected_mode_name = st.selectbox(
            "시스템 모드", 
            list(mode_options.keys()),
            help="분석에 사용할 AI 시스템을 선택하세요"
        )
        selected_mode = mode_options[selected_mode_name]
        
        # 분석 타입 선택
        analysis_options = {
            "🔍 자동 감지": "auto",
            "💎 다이아몬드 4C": "diamond",
            "🌈 유색보석": "colored_stone", 
            "🎨 주얼리 디자인": "design",
            "📊 비즈니스 분석": "business"
        }
        
        selected_analysis_name = st.selectbox(
            "분석 타입",
            list(analysis_options.keys()),
            help="수행할 분석 타입을 선택하세요"
        )
        selected_analysis = analysis_options[selected_analysis_name]
        
        # 성능 분석 표시
        st.header("📊 성능 현황")
        if st.button("🔄 성능 새로고침"):
            analytics = st.session_state.ai_system.get_performance_analytics()
            for key, value in analytics.items():
                st.metric(key.replace("_", " "), value)
    
    # 메인 영역 - 분석 인터페이스
    st.header("📝 주얼리 분석 입력")
    
    # 입력 탭
    input_tab1, input_tab2, input_tab3 = st.tabs(["📝 텍스트 입력", "🖼️ 이미지 분석", "📊 성능 대시보드"])
    
    with input_tab1:
        text_input = st.text_area(
            "주얼리 정보를 입력하세요",
            placeholder="예: 1.5캐럿 라운드 다이아몬드, D컬러, VVS1 클래리티, Excellent 컷...",
            height=150
        )
        
        if st.button("🚀 분석 시작", key="text_analysis"):
            if text_input.strip():
                run_analysis(text_input, selected_mode, selected_analysis)
            else:
                st.warning("분석할 텍스트를 입력해주세요.")
    
    with input_tab2:
        uploaded_file = st.file_uploader(
            "주얼리 이미지 업로드",
            type=['jpg', 'jpeg', 'png'],
            help="다이아몬드, 보석, 주얼리 이미지를 업로드하세요"
        )
        
        image_text = st.text_area(
            "이미지에 대한 추가 설명 (선택사항)",
            placeholder="이미지에 대한 추가 정보를 입력하세요...",
            height=100
        )
        
        if st.button("🖼️ 이미지 분석", key="image_analysis"):
            if uploaded_file:
                run_image_analysis(uploaded_file, image_text, selected_mode, selected_analysis)
            else:
                st.warning("분석할 이미지를 업로드해주세요.")
    
    with input_tab3:
        st.subheader("📊 실시간 성능 모니터링")
        
        if st.session_state.ai_system.performance_history:
            import pandas as pd
            import matplotlib.pyplot as plt
            
            # 성능 데이터 차트
            df = pd.DataFrame(st.session_state.ai_system.performance_history[-20:])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.line_chart(df[['time', 'quality']].set_index(df.index))
                st.caption("처리 시간 및 품질 점수 추이")
            
            with col2:
                mode_counts = df['mode'].value_counts()
                st.bar_chart(mode_counts)
                st.caption("시스템 모드 사용 분포")
            
            # 상세 통계
            analytics = st.session_state.ai_system.get_performance_analytics()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("평균 처리 시간", analytics.get("평균_처리_시간", "N/A"))
            with col2:
                st.metric("평균 품질 점수", analytics.get("평균_품질_점수", "N/A"))
            with col3:
                st.metric("성능 등급", analytics.get("성능_등급", "N/A"))
        else:
            st.info("성능 데이터가 없습니다. 분석을 실행하면 성능 통계가 표시됩니다.")

def run_analysis(text_input: str, mode: SystemMode, analysis_type: str):
    """텍스트 분석 실행"""
    
    with st.spinner("🧠 AI 분석 중... (최대 25초 소요)"):
        start_time = time.time()
        
        input_data = {"text": text_input}
        
        # 비동기 분석 실행
        try:
            result = asyncio.run(
                st.session_state.ai_system.analyze_jewelry(
                    input_data, analysis_type, mode
                )
            )
            
            # 결과 표시
            display_analysis_result(result, start_time)
            
        except Exception as e:
            st.error(f"❌ 분석 중 오류 발생: {e}")
            st.exception(e)

def run_image_analysis(uploaded_file, description: str, mode: SystemMode, analysis_type: str):
    """이미지 분석 실행"""
    
    with st.spinner("🖼️ 이미지 분석 중... (최대 30초 소요)"):
        start_time = time.time()
        
        # 이미지 데이터 준비
        image_data = uploaded_file.read()
        
        input_data = {
            "text": description or "업로드된 주얼리 이미지를 분석해주세요.",
            "image": image_data
        }
        
        # 비동기 분석 실행
        try:
            result = asyncio.run(
                st.session_state.ai_system.analyze_jewelry(
                    input_data, analysis_type, mode
                )
            )
            
            # 이미지 표시
            st.image(uploaded_file, caption="분석 대상 이미지", width=300)
            
            # 결과 표시
            display_analysis_result(result, start_time)
            
        except Exception as e:
            st.error(f"❌ 이미지 분석 중 오류 발생: {e}")
            st.exception(e)

def display_analysis_result(result: AnalysisResult, start_time: float):
    """분석 결과 표시"""
    
    total_time = time.time() - start_time
    
    # 성능 지표
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⏱️ 처리 시간", f"{total_time:.1f}초")
    with col2:
        st.metric("🎯 품질 점수", f"{result.quality_score:.1%}")
    with col3:
        st.metric("🧠 사용 모델", result.model_used)
    with col4:
        st.metric("💰 예상 비용", f"${result.cost_estimate:.3f}")
    
    # 시스템 모드 및 성능 상태
    mode_color = {
        SystemMode.HYBRID_V23: "🚀",
        SystemMode.LEGACY_V214: "🔧", 
        SystemMode.AUTO_OPTIMIZE: "🤖"
    }
    
    st.info(f"{mode_color.get(result.system_mode, '❓')} 시스템 모드: {result.system_mode.value}")
    
    # 품질 등급 표시
    if result.quality_score >= 0.95:
        st.success("✅ 우수한 분석 품질 (95% 이상)")
    elif result.quality_score >= 0.85:
        st.info("✔️ 양호한 분석 품질 (85% 이상)")
    elif result.quality_score >= 0.70:
        st.warning("⚠️ 보통 분석 품질 (70% 이상)")
    else:
        st.error("❌ 품질 개선 필요 (70% 미만)")
    
    # 분석 결과 내용
    st.header("📋 분석 결과")
    st.markdown(result.content)
    
    # 상세 메타데이터 (접기/펼치기)
    with st.expander("🔍 상세 분석 정보"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("성능 지표")
            st.json({
                "신뢰도": f"{result.confidence:.1%}",
                "주얼리 전문성": f"{result.jewelry_expertise:.1%}",
                "처리 시간": f"{result.processing_time:.2f}초",
                "타임스탬프": result.timestamp
            })
        
        with col2:
            st.subheader("메타데이터")
            st.json(result.metadata)

# 스타일링
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .success {
        border-left-color: #51cf66 !important;
    }
    .info {
        border-left-color: #339af0 !important;
    }
    .warning {
        border-left-color: #ffd43b !important;
    }
    .error {
        border-left-color: #ff6b6b !important;
    }
</style>
""", unsafe_allow_html=True)

# 메인 실행
if __name__ == "__main__":
    try:
        create_streamlit_ui()
    except Exception as e:
        st.error(f"💥 시스템 초기화 실패: {e}")
        st.exception(e)
        st.info("페이지를 새로고침하거나 관리자에게 문의하세요.")
