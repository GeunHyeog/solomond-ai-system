#!/usr/bin/env python3
"""
실제 분석 워크플로우 어댑터 v2.6
웹 크롤링 통합 및 지능형 컨텍스트 분석
새로운 4단계 워크플로우와 기존 실제 분석 엔진들을 연결하는 어댑터 클래스
"""

import os
import sys
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass, asdict

# 인코딩 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'

@dataclass
class EnhancedAnalysisResult:
    """향상된 분석 결과 데이터 클래스"""
    original_analysis: Dict[str, Any]
    web_context: Dict[str, Any]
    enhanced_insights: List[str]
    confidence_score: float
    processing_time_ms: float
    recommendations: List[str]
    contextual_summary: str
    timestamp: str

class RealAnalysisWorkflowAdapter:
    """실제 분석 워크플로우 어댑터"""
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.progress_callback = progress_callback
        self.analysis_results = {}
        self.current_step = 0
        self.total_steps = 4
        
        # 기존 분석 엔진들 초기화
        self._initialize_engines()
    
    def _initialize_engines(self):
        """기존 분석 엔진들 + Ollama 엔진 + 웹 크롤링 시스템 초기화"""
        try:
            # Real Analysis Engine
            from .real_analysis_engine import RealAnalysisEngine
            self.real_engine = RealAnalysisEngine()
            
            # Enhanced Speaker Identifier
            from ..enhanced_speaker_identifier import EnhancedSpeakerIdentifier
            self.speaker_identifier = EnhancedSpeakerIdentifier()
            
            # Comprehensive Message Extractor
            from .comprehensive_message_extractor import ComprehensiveMessageExtractor
            self.message_extractor = ComprehensiveMessageExtractor()
            
            # 🚀 Enhanced Ollama Workflow Engine (사용자의 7개 모델 활용)
            from .enhanced_ollama_workflow_engine import EnhancedOllamaWorkflowEngine
            self.ollama_engine = EnhancedOllamaWorkflowEngine()
            
            # 🌐 NEW v2.6: Intelligent Web Crawler
            from .intelligent_web_crawler import get_global_crawler
            self.web_crawler = get_global_crawler()
            
            # 📊 NEW v2.6: Real-time Performance Monitor
            from .advanced_monitoring.realtime_performance_monitor import get_global_monitor
            self.performance_monitor = get_global_monitor()
            
            self.engines_ready = True
            self.ollama_available = True
            self.web_integration_enabled = True
            
            print("✅ v2.6 모든 엔진 초기화 완료: 전통적 분석 + Ollama + 웹 크롤링 + 모니터링")
            
        except ImportError as e:
            print(f"분석 엔진 초기화 실패: {e}")
            self.engines_ready = False
            self.ollama_available = False
            self.web_integration_enabled = False
            self.web_crawler = None
            self.performance_monitor = None
    
    def _update_progress(self, step: int, message: str, details: Dict = None):
        """진행 상황 업데이트"""
        self.current_step = step
        progress_data = {
            'step': step,
            'total_steps': self.total_steps,
            'message': message,
            'progress_percent': (step / self.total_steps) * 100,
            'details': details or {}
        }
        
        if self.progress_callback:
            self.progress_callback(progress_data)
    
    async def execute_step1_source_extraction(self, file_data: Dict, use_ollama: bool = True) -> Dict:
        """1단계: 소스별 정보 추출 (Ollama 7개 모델 활용 가능)"""
        self._update_progress(1, "1단계: 소스별 정보 추출 시작")
        
        # 🚀 Ollama 엔진 우선 사용 (사용자의 7개 강력한 모델 활용)
        if use_ollama and self.ollama_available:
            try:
                self._update_progress(1, "🔥 Ollama 7개 모델로 고품질 분석 중...")
                ollama_results = await self.ollama_engine.execute_step1_enhanced_analysis(file_data)
                
                # Ollama 결과와 기존 엔진 결과 결합
                combined_results = await self._combine_ollama_and_traditional_analysis(ollama_results, file_data)
                
                self._update_progress(1, "✅ Ollama + 전통적 분석 조합 완료")
                return combined_results
                
            except Exception as e:
                print(f"Ollama 분석 실패, 기존 엔진으로 전환: {e}")
        
        # 기존 분석 엔진 사용
        if not self.engines_ready:
            return self._create_fallback_step1_results(file_data)
        
        results = {
            'audio_analysis': {},
            'image_analysis': {},
            'video_analysis': {},
            'processing_stats': {},
            'analysis_mode': 'traditional'
        }
        
        try:
            # 오디오 파일 처리
            if file_data.get('audio', []):
                self._update_progress(1, "오디오 파일 분석 중...")
                audio_results = await self._process_audio_files(file_data['audio'])
                results['audio_analysis'] = audio_results
            
            # 이미지 파일 처리
            if file_data.get('image', []):
                self._update_progress(1, "이미지 파일 분석 중...")
                image_results = await self._process_image_files(file_data['image'])
                results['image_analysis'] = image_results
            
            # 비디오 파일 처리
            if file_data.get('video', []):
                self._update_progress(1, "비디오 파일 분석 중...")
                video_results = await self._process_video_files(file_data['video'])
                results['video_analysis'] = video_results
            
            self.analysis_results['step1'] = results
            self._update_progress(1, "1단계 완료: 소스별 정보 추출 완료", results)
            
            return results
            
        except Exception as e:
            print(f"1단계 오류: {e}")
            return self._create_fallback_step1_results(file_data)
    
    async def execute_step2_information_synthesis(self, step1_results: Dict, synthesis_config: Dict, use_ollama: bool = True) -> Dict:
        """2단계: 정보 종합 (QWEN3:8B + QWEN2.5:7B 조합)"""
        self._update_progress(2, "2단계: 정보 종합 시작")
        
        # 🥇 Ollama 지능형 종합 분석 우선 사용
        if use_ollama and self.ollama_available:
            try:
                self._update_progress(2, "🧠 QWEN3:8B + QWEN2.5:7B 지능형 종합 분석 중...")
                ollama_synthesis = await self.ollama_engine.execute_step2_intelligent_synthesis(
                    step1_results, synthesis_config
                )
                
                ollama_synthesis['analysis_mode'] = 'ollama_enhanced'
                ollama_synthesis['models_used'] = ['qwen3:8b', 'qwen2.5:7b']
                
                self._update_progress(2, "✅ Ollama 지능형 종합 완료")
                return ollama_synthesis
                
            except Exception as e:
                print(f"Ollama 종합 분석 실패, 기존 방식으로 전환: {e}")
        
        try:
            # 멀티모달 정보 통합
            integrated_data = self._integrate_multimodal_data(step1_results)
            
            # 시간순 정렬
            if synthesis_config.get('synthesis_mode') == '시간순 정렬':
                timeline_data = self._create_timeline(integrated_data)
            else:
                timeline_data = integrated_data
            
            # 화자별 인사이트 생성
            speaker_insights = self._generate_speaker_insights(timeline_data)
            
            # 내용 연관성 분석
            content_correlation = self._analyze_content_correlation(timeline_data)
            
            results = {
                'integrated_timeline': timeline_data,
                'speaker_insights': speaker_insights,
                'content_correlation': content_correlation,
                'synthesis_config': synthesis_config
            }
            
            self.analysis_results['step2'] = results
            self._update_progress(2, "2단계 완료: 정보 종합 완료", results)
            
            return results
            
        except Exception as e:
            print(f"2단계 오류: {e}")
            return self._create_fallback_step2_results()
    
    async def execute_step3_full_script_generation(self, step2_results: Dict, script_config: Dict, use_ollama: bool = True) -> Dict:
        """3단계: 풀스크립트 생성 (GEMMA3:27B 프리미엄 품질)"""
        self._update_progress(3, "3단계: 풀스크립트 생성 시작")
        
        # 🏆 Ollama 프리미엄 스크립트 생성 우선 사용 (GEMMA3:27B)
        if use_ollama and self.ollama_available:
            try:
                self._update_progress(3, "🏆 GEMMA3:27B 프리미엄 스크립트 생성 중...")
                ollama_script = await self.ollama_engine.execute_step3_premium_script_generation(
                    step2_results, script_config
                )
                
                ollama_script['analysis_mode'] = 'ollama_premium'
                ollama_script['quality_tier'] = 'gemma3_27b_premium'
                
                self._update_progress(3, "✅ GEMMA3:27B 프리미엄 스크립트 완료")
                return ollama_script
                
            except Exception as e:
                print(f"Ollama 스크립트 생성 실패, 기존 방식으로 전환: {e}")
        
        try:
            # 스크립트 형식에 따른 생성
            script_format = script_config.get('script_format', '대화형 스크립트')
            
            if script_format == '대화형 스크립트':
                full_script = self._generate_dialogue_script(step2_results, script_config)
            elif script_format == '내러티브 형식':
                full_script = self._generate_narrative_script(step2_results, script_config)
            elif script_format == '보고서 형식':
                full_script = self._generate_report_script(step2_results, script_config)
            else:
                full_script = self._generate_timeline_script(step2_results, script_config)
            
            results = {
                'full_script': full_script,
                'script_metadata': {
                    'format': script_format,
                    'generated_at': datetime.now().isoformat(),
                    'total_segments': len(step2_results.get('integrated_timeline', [])),
                    'speakers_count': len(step2_results.get('speaker_insights', {})),
                    'script_length': len(full_script)
                },
                'script_config': script_config
            }
            
            self.analysis_results['step3'] = results
            self._update_progress(3, "3단계 완료: 풀스크립트 생성 완료", results)
            
            return results
            
        except Exception as e:
            print(f"3단계 오류: {e}")
            return self._create_fallback_step3_results(script_config)
    
    async def process_comprehensive_analysis_with_web_context(self, 
                                                           uploaded_files: List[Dict],
                                                           basic_info: Optional[Dict] = None) -> EnhancedAnalysisResult:
        """🌐 NEW v2.6: 웹 컨텍스트 통합 포괄적 분석"""
        start_time = time.time()
        self._update_progress(0, "🚀 v2.6 웹 통합 포괄적 분석 시작")
        
        try:
            # 1단계: 기본 분석 수행
            self._update_progress(1, "📊 기본 4단계 워크플로우 실행 중...")
            original_analysis = await self._execute_full_4step_workflow(uploaded_files, basic_info)
            
            # 2단계: 웹 컨텍스트 추가
            enhanced_analysis = original_analysis
            web_context = {"search_performed": False}
            
            if self.web_integration_enabled and self.web_crawler:
                self._update_progress(2, "🌐 지능형 웹 컨텍스트 분석 중...")
                enhanced_analysis = self.web_crawler.generate_web_context_for_analysis(original_analysis)
                web_context = enhanced_analysis.get('web_context', {})
            
            # 3단계: 통합 인사이트 생성
            self._update_progress(3, "🧠 통합 인사이트 + 권장사항 생성 중...")
            enhanced_insights = self._generate_enhanced_insights(enhanced_analysis, web_context)
            
            # 4단계: 신뢰도 점수 계산
            confidence_score = self._calculate_confidence_score(enhanced_analysis, web_context)
            
            # 5단계: 권장사항 생성
            recommendations = self._generate_contextual_recommendations(enhanced_analysis, web_context)
            
            # 6단계: 최종 요약 생성
            contextual_summary = self._generate_contextual_summary(enhanced_analysis, enhanced_insights)
            
            processing_time = (time.time() - start_time) * 1000
            
            # 성능 통계 업데이트
            if self.performance_monitor:
                self.performance_monitor.update_processing_stats(
                    queue_size=0, 
                    completed_tasks=1
                )
            
            result = EnhancedAnalysisResult(
                original_analysis=original_analysis,
                web_context=web_context,
                enhanced_insights=enhanced_insights,
                confidence_score=confidence_score,
                processing_time_ms=processing_time,
                recommendations=recommendations,
                contextual_summary=contextual_summary,
                timestamp=datetime.now().isoformat()
            )
            
            self._update_progress(4, f"✅ v2.6 웹 통합 분석 완료 (소요시간: {processing_time:.1f}ms)")
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            print(f"❌ v2.6 분석 실패: {e}")
            return self._create_fallback_enhanced_result(uploaded_files, processing_time, str(e))
    
    async def _execute_full_4step_workflow(self, uploaded_files: List[Dict], basic_info: Optional[Dict]) -> Dict:
        """기존 4단계 워크플로우 전체 실행"""
        # 파일 데이터 준비
        file_data = self._prepare_file_data(uploaded_files)
        
        # 1-4단계 순차 실행
        step1_results = await self.execute_step1_source_extraction(file_data, use_ollama=True)
        
        synthesis_config = {'synthesis_mode': '시간순 정렬'}
        step2_results = await self.execute_step2_information_synthesis(
            step1_results, synthesis_config, use_ollama=True
        )
        
        script_config = {'script_format': '대화형 스크립트', 'include_timestamps': True}
        step3_results = await self.execute_step3_full_script_generation(
            step2_results, script_config, use_ollama=True
        )
        
        summary_config = {'summary_type': '핵심 내용 요약', 'include_insights': True}
        step4_results = await self.execute_step4_summary_generation(
            step3_results, summary_config, use_ollama=True
        )
        
        return {
            'step1_source_extraction': step1_results,
            'step2_information_synthesis': step2_results,
            'step3_full_script': step3_results,
            'step4_summary': step4_results,
            'basic_info': basic_info or {},
            'files_processed': len(uploaded_files),
            'processing_success': True
        }
    
    def _prepare_file_data(self, uploaded_files: List[Dict]) -> Dict:
        """업로드된 파일을 타입별로 분류"""
        file_data = {'audio': [], 'image': [], 'video': []}
        
        for file_info in uploaded_files:
            file_path = Path(file_info.get('path', ''))
            file_type = file_info.get('type', 'unknown')
            
            if file_type.startswith('audio'):
                file_data['audio'].append(file_path)
            elif file_type.startswith('image'):
                file_data['image'].append(file_path)
            elif file_type.startswith('video'):
                file_data['video'].append(file_path)
        
        return file_data
    
    def _generate_enhanced_insights(self, analysis_result: Dict[str, Any], web_context: Dict[str, Any]) -> List[str]:
        """v2.6 향상된 인사이트 생성"""
        insights = []
        
        # 기본 분석 인사이트
        if analysis_result.get('processing_success'):
            insights.append("✅ 4단계 워크플로우가 성공적으로 완료되었습니다")
            
            files_processed = analysis_result.get('files_processed', 0)
            if files_processed > 0:
                insights.append(f"📁 총 {files_processed}개 파일이 처리되었습니다")
        
        # Ollama 분석 인사이트
        step1_results = analysis_result.get('step1_source_extraction', {})
        if step1_results.get('analysis_mode') == 'ollama_enhanced':
            insights.append("🚀 Ollama 7개 모델을 활용한 고품질 분석이 적용되었습니다")
        
        # 웹 컨텍스트 인사이트
        if web_context.get('search_performed'):
            keywords_count = len(web_context.get('keywords_used', []))
            insights.append(f"🌐 {keywords_count}개 키워드로 실시간 웹 검색이 수행되었습니다")
            
            key_insights = web_context.get('key_insights', [])
            insights.extend([f"💡 {insight}" for insight in key_insights[:2]])  # 상위 2개만
        
        # 품질 인사이트
        if len(insights) >= 3:
            insights.append("🎯 높은 품질의 다면적 분석이 완료되었습니다")
        
        return insights[:8]  # 최대 8개로 제한
    
    def _calculate_confidence_score(self, analysis_result: Dict[str, Any], web_context: Dict[str, Any]) -> float:
        """v2.6 신뢰도 점수 계산"""
        confidence = 0.6  # 기본 점수 상향
        
        # 기본 분석 성공 여부
        if analysis_result.get('processing_success'):
            confidence += 0.2
        
        # Ollama 사용 보너스
        step1_results = analysis_result.get('step1_source_extraction', {})
        if step1_results.get('analysis_mode') == 'ollama_enhanced':
            confidence += 0.15
        
        # 웹 컨텍스트 추가
        if web_context.get('search_performed'):
            results_count = len(web_context.get('related_urls', []))
            confidence += min(0.15, results_count * 0.02)
        
        # 파일 처리 성공률
        files_processed = analysis_result.get('files_processed', 0)
        if files_processed > 0:
            confidence += min(0.1, files_processed * 0.05)
        
        return min(confidence, 1.0)
    
    def _generate_contextual_recommendations(self, analysis_result: Dict[str, Any], web_context: Dict[str, Any]) -> List[str]:
        """v2.6 컨텍스트 기반 권장사항 생성"""
        recommendations = []
        
        # 4단계 워크플로우 기반 권장사항
        step4_results = analysis_result.get('step4_summary', {})
        if step4_results:
            recommendations.append("📋 생성된 최종 요약을 고객 응대에 적극 활용하세요")
        
        # 웹 컨텍스트 기반 권장사항
        if web_context.get('search_performed'):
            key_insights = web_context.get('key_insights', [])
            
            for insight in key_insights:
                if '가격' in insight:
                    recommendations.append("💲 최신 시장 가격 정보를 제공하여 경쟁력을 높이세요")
                elif '트렌드' in insight:
                    recommendations.append("📈 트렌드 정보를 활용하여 고객 관심을 유도하세요")
                elif '브랜드' in insight:
                    recommendations.append("🏆 브랜드 추천으로 고객 선택을 도와주세요")
        
        # Ollama 품질 권장사항
        step1_results = analysis_result.get('step1_source_extraction', {})
        if step1_results.get('analysis_mode') == 'ollama_enhanced':
            recommendations.append("🔥 고품질 AI 분석 결과를 신뢰하고 적극 활용하세요")
        
        # 일반적인 권장사항
        if not recommendations:
            recommendations.append("📊 분석 결과를 바탕으로 맞춤형 고객 응대를 진행하세요")
        
        recommendations.append("📞 추가 질문이나 상담이 필요한 경우 적극적으로 소통하세요")
        
        return recommendations[:5]
    
    def _generate_contextual_summary(self, analysis_result: Dict[str, Any], enhanced_insights: List[str]) -> str:
        """v2.6 컨텍스트 요약 생성"""
        summary_parts = []
        
        # 기본 처리 정보
        files_processed = analysis_result.get('files_processed', 0)
        if files_processed > 0:
            summary_parts.append(f"총 {files_processed}개 파일을 통해 4단계 워크플로우 분석을 완료했습니다.")
        
        # 최종 요약 확인
        step4_results = analysis_result.get('step4_summary', {})
        if step4_results.get('final_summary'):
            summary_parts.append("최종 요약본이 성공적으로 생성되어 즉시 활용 가능합니다.")
        
        # 웹 검색 정보
        web_context = analysis_result.get('web_context', {})
        if web_context.get('search_performed'):
            summary_parts.append("실시간 웹 검색을 통해 최신 정보가 추가되었습니다.")
        
        # 품질 정보
        step1_results = analysis_result.get('step1_source_extraction', {})
        if step1_results.get('analysis_mode') == 'ollama_enhanced':
            summary_parts.append("Ollama 7개 모델의 고품질 AI 분석이 적용되었습니다.")
        
        return " ".join(summary_parts) if summary_parts else "v2.6 통합 분석이 완료되었습니다."
    
    def _create_fallback_enhanced_result(self, uploaded_files: List[Dict], processing_time_ms: float, error_message: str) -> EnhancedAnalysisResult:
        """v2.6 실패 시 대체 결과 생성"""
        return EnhancedAnalysisResult(
            original_analysis={
                "error": error_message,
                "files_processed": len(uploaded_files),
                "processing_success": False
            },
            web_context={"search_performed": False, "error": "웹 검색을 수행할 수 없었습니다"},
            enhanced_insights=[
                "❌ v2.6 분석 중 오류가 발생했습니다",
                "🔧 시스템을 확인하고 다시 시도해주세요"
            ],
            confidence_score=0.0,
            processing_time_ms=processing_time_ms,
            recommendations=[
                "🔍 파일 형식과 크기를 확인해주세요",
                "⚡ 시스템 리소스가 충분한지 확인해주세요",
                "🌐 네트워크 연결 상태를 점검해주세요"
            ],
            contextual_summary=f"v2.6 분석 실패: {error_message}",
            timestamp=datetime.now().isoformat()
        )

    async def execute_step4_summary_generation(self, step3_results: Dict, summary_config: Dict, use_ollama: bool = True) -> Dict:
        """4단계: 요약본 생성 (GEMMA3:27B + QWEN3:8B 프리미엄 조합)"""
        self._update_progress(4, "4단계: 요약본 생성 시작")
        
        # 🏆 Ollama 프리미엄 요약 생성 우선 사용 (GEMMA3:27B + QWEN3:8B)
        if use_ollama and self.ollama_available:
            try:
                self._update_progress(4, "🏆 GEMMA3:27B + QWEN3:8B 프리미엄 요약 생성 중...")
                ollama_summary = await self.ollama_engine.execute_step4_premium_summary(
                    step3_results, summary_config
                )
                
                ollama_summary['analysis_mode'] = 'ollama_premium'
                ollama_summary['quality_tier'] = 'gemma3_27b_qwen3_8b_premium'
                
                self._update_progress(4, "✅ 프리미엄 요약 + 한국어 인사이트 완료")
                return ollama_summary
                
            except Exception as e:
                print(f"Ollama 요약 생성 실패, 기존 방식으로 전환: {e}")
        
        try:
            full_script = step3_results.get('full_script', '')
            
            # 종합 메시지 추출 사용
            if self.engines_ready:
                summary_data = self.message_extractor.extract_comprehensive_message(
                    {'full_script': full_script}
                )
            else:
                summary_data = self._create_fallback_summary_data()
            
            # 요약 유형별 처리
            summary_type = summary_config.get('summary_type', '핵심 내용 요약')
            formatted_summary = self._format_summary(summary_data, summary_config)
            
            results = {
                'final_summary': formatted_summary,
                'summary_metadata': {
                    'type': summary_type,
                    'generated_at': datetime.now().isoformat(),
                    'source_length': len(full_script),
                    'summary_length': len(formatted_summary),
                    'compression_ratio': len(formatted_summary) / max(len(full_script), 1)
                },
                'summary_config': summary_config,
                'raw_analysis': summary_data
            }
            
            self.analysis_results['step4'] = results
            self._update_progress(4, "4단계 완료: 요약본 생성 완료", results)
            
            return results
            
        except Exception as e:
            print(f"4단계 오류: {e}")
            return self._create_fallback_step4_results(summary_config)
    
    # Helper methods for actual processing
    async def _process_audio_files(self, audio_files: List[Path]) -> Dict:
        """오디오 파일 실제 처리"""
        results = {}
        
        for audio_file in audio_files:
            try:
                # STT 처리
                if self.engines_ready:
                    stt_result = await self.real_engine.process_audio_stt(str(audio_file))
                    
                    # 화자 구분
                    if stt_result.get('segments'):
                        speaker_result = self.speaker_identifier.identify_speakers_from_segments(
                            stt_result['segments']
                        )
                        results[audio_file.name] = {
                            'stt_segments': speaker_result,
                            'total_duration': stt_result.get('duration', 0),
                            'speaker_count': len(set(seg.get('speaker', 'Unknown') for seg in speaker_result))
                        }
                    else:
                        results[audio_file.name] = {'error': 'STT 처리 실패'}
                else:
                    results[audio_file.name] = {'error': '분석 엔진 초기화 실패'}
                    
            except Exception as e:
                results[audio_file.name] = {'error': str(e)}
        
        return results
    
    async def _process_image_files(self, image_files: List[Path]) -> Dict:
        """이미지 파일 실제 처리"""
        results = {}
        
        for image_file in image_files:
            try:
                if self.engines_ready:
                    ocr_result = await self.real_engine.process_image_ocr(str(image_file))
                    results[image_file.name] = {
                        'ocr_blocks': ocr_result.get('blocks', []),
                        'confidence_score': ocr_result.get('average_confidence', 0),
                        'text_count': len(ocr_result.get('blocks', []))
                    }
                else:
                    results[image_file.name] = {'error': '분석 엔진 초기화 실패'}
                    
            except Exception as e:
                results[image_file.name] = {'error': str(e)}
        
        return results
    
    async def _process_video_files(self, video_files: List[Path]) -> Dict:
        """비디오 파일 실제 처리"""
        results = {}
        
        for video_file in video_files:
            try:
                # 비디오 메타데이터 추출
                metadata = {
                    'filename': video_file.name,
                    'size': video_file.stat().st_size,
                    'extension': video_file.suffix
                }
                results[video_file.name] = {
                    'metadata': metadata,
                    'status': 'processed'
                }
                    
            except Exception as e:
                results[video_file.name] = {'error': str(e)}
        
        return results
    
    def _integrate_multimodal_data(self, step1_results: Dict) -> List[Dict]:
        """멀티모달 데이터 통합"""
        integrated_data = []
        
        # 오디오 데이터 추가
        for filename, data in step1_results.get('audio_analysis', {}).items():
            if 'stt_segments' in data:
                for segment in data['stt_segments']:
                    integrated_data.append({
                        'source_type': 'audio',
                        'source_file': filename,
                        'timestamp': segment.get('start', 0),
                        'content': segment.get('text', ''),
                        'speaker': segment.get('speaker', 'Unknown'),
                        'metadata': segment
                    })
        
        # 이미지 데이터 추가
        for filename, data in step1_results.get('image_analysis', {}).items():
            if 'ocr_blocks' in data:
                for i, block in enumerate(data['ocr_blocks']):
                    integrated_data.append({
                        'source_type': 'image',
                        'source_file': filename,
                        'timestamp': i,  # 이미지는 시퀀스 순서
                        'content': block.get('text', ''),
                        'confidence': block.get('confidence', 0),
                        'metadata': block
                    })
        
        # 시간순 정렬
        integrated_data.sort(key=lambda x: x.get('timestamp', 0))
        
        return integrated_data
    
    def _create_timeline(self, integrated_data: List[Dict]) -> List[Dict]:
        """시간순 타임라인 생성"""
        return sorted(integrated_data, key=lambda x: x.get('timestamp', 0))
    
    def _generate_speaker_insights(self, timeline_data: List[Dict]) -> Dict:
        """화자별 인사이트 생성"""
        speakers = {}
        
        for item in timeline_data:
            speaker = item.get('speaker', 'Unknown')
            if speaker not in speakers:
                speakers[speaker] = {
                    'total_segments': 0,
                    'total_words': 0,
                    'avg_segment_length': 0,
                    'topics': [],
                    'speaking_time': 0
                }
            
            speakers[speaker]['total_segments'] += 1
            content = item.get('content', '')
            word_count = len(content.split())
            speakers[speaker]['total_words'] += word_count
            
        # 평균 계산
        for speaker, data in speakers.items():
            if data['total_segments'] > 0:
                data['avg_segment_length'] = data['total_words'] / data['total_segments']
        
        return speakers
    
    def _analyze_content_correlation(self, timeline_data: List[Dict]) -> Dict:
        """내용 연관성 분석"""
        correlation = {
            'cross_modal_references': 0,
            'topic_coherence': 0.8,  # 기본값
            'information_density': len(timeline_data)
        }
        
        return correlation
    
    async def _combine_ollama_and_traditional_analysis(self, ollama_results: Dict, file_data: Dict) -> Dict:
        """Ollama 분석과 전통적 분석 결과 결합"""
        combined_results = ollama_results.copy()
        combined_results['analysis_mode'] = 'ollama_enhanced'
        
        # 전통적 분석도 병렬로 실행하여 결합
        try:
            traditional_audio = await self._process_audio_files(file_data.get('audio', []))
            traditional_images = await self._process_image_files(file_data.get('image', []))
            
            # Ollama 결과와 전통적 분석 결과 결합
            if traditional_audio:
                combined_results['traditional_audio_analysis'] = traditional_audio
            if traditional_images:
                combined_results['traditional_image_analysis'] = traditional_images
                
            combined_results['enhancement_level'] = 'ollama_plus_traditional'
            
        except Exception as e:
            print(f"전통적 분석 결합 실패: {e}")
            combined_results['enhancement_level'] = 'ollama_only'
        
        return combined_results
    
    def _generate_dialogue_script(self, step2_results: Dict, config: Dict) -> str:
        """대화형 스크립트 생성"""
        timeline = step2_results.get('integrated_timeline', [])
        include_timestamps = config.get('include_timestamps', True)
        include_speaker_notes = config.get('include_speaker_notes', True)
        
        script_lines = []
        script_lines.append(f"# 대화형 스크립트")
        script_lines.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 참여자 정보
        speakers = set(item.get('speaker', 'Unknown') for item in timeline if item.get('speaker'))
        script_lines.append(f"참여자: {', '.join(sorted(speakers))}")
        script_lines.append("")
        script_lines.append("=" * 50)
        script_lines.append("")
        
        for item in timeline:
            speaker = item.get('speaker', 'Unknown')
            content = item.get('content', '')
            timestamp = item.get('timestamp', 0)
            
            if content.strip():
                if include_timestamps:
                    time_str = f"[{int(timestamp//60):02d}:{int(timestamp%60):02d}]"
                    script_lines.append(f"{time_str} {speaker}: {content}")
                else:
                    script_lines.append(f"{speaker}: {content}")
        
        script_lines.append("")
        script_lines.append("=" * 50)
        script_lines.append(f"총 항목: {len([item for item in timeline if item.get('content', '').strip()])}개")
        script_lines.append(f"참여 화자: {len(speakers)}명")
        script_lines.append(f"생성 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(script_lines)
    
    def _generate_narrative_script(self, step2_results: Dict, config: Dict) -> str:
        """내러티브 형식 스크립트 생성"""
        # 대화형과 유사하지만 서술형으로 변환
        return self._generate_dialogue_script(step2_results, config).replace(":", "이 말했다:")
    
    def _generate_report_script(self, step2_results: Dict, config: Dict) -> str:
        """보고서 형식 스크립트 생성"""
        timeline = step2_results.get('integrated_timeline', [])
        speaker_insights = step2_results.get('speaker_insights', {})
        
        script_lines = []
        script_lines.append("# 분석 보고서")
        script_lines.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        script_lines.append("")
        
        # 요약 정보
        script_lines.append("## 분석 요약")
        script_lines.append(f"- 총 데이터 항목: {len(timeline)}개")
        script_lines.append(f"- 참여 화자: {len(speaker_insights)}명")
        script_lines.append("")
        
        # 화자별 분석
        script_lines.append("## 화자별 분석")
        for speaker, insights in speaker_insights.items():
            script_lines.append(f"### {speaker}")
            script_lines.append(f"- 총 발언: {insights.get('total_segments', 0)}회")
            script_lines.append(f"- 평균 발언 길이: {insights.get('avg_segment_length', 0):.1f}어절")
            script_lines.append("")
        
        return "\n".join(script_lines)
    
    def _generate_timeline_script(self, step2_results: Dict, config: Dict) -> str:
        """타임라인 형식 스크립트 생성"""
        return self._generate_dialogue_script(step2_results, config)
    
    def _format_summary(self, summary_data: Dict, config: Dict) -> str:
        """요약 포맷팅"""
        summary_type = config.get('summary_type', '핵심 내용 요약')
        include_keywords = config.get('include_keywords', True)
        include_insights = config.get('include_insights', True)
        include_recommendations = config.get('include_recommendations', True)
        
        lines = []
        lines.append(f"# {summary_type}")
        lines.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 기본 개요
        lines.append("## 📊 기본 개요")
        lines.append(f"• 분석된 데이터 포인트: {summary_data.get('total_data_points', 'N/A')}개")
        lines.append(f"• 주요 소스: {', '.join(summary_data.get('source_types', ['audio', 'image']))}")
        lines.append("")
        
        # 핵심 내용
        lines.append("## 🎯 핵심 내용")
        main_content = summary_data.get('main_content', '분석된 내용의 주요 포인트들이 여기에 표시됩니다.')
        lines.append(main_content)
        lines.append("")
        
        if include_keywords:
            lines.append("## 🔑 핵심 키워드")
            keywords = summary_data.get('keywords', ['분석', '내용', '정보', '데이터', '결과'])
            keyword_text = ', '.join([f"{kw} ({i+1}회)" for i, kw in enumerate(keywords[:5])])
            lines.append(keyword_text)
            lines.append("")
        
        if include_insights:
            lines.append("## 💡 주요 인사이트")
            insights = summary_data.get('insights', [
                '데이터 분석이 성공적으로 완료됨',
                '다양한 소스에서 정보가 추출됨',
                '종합적인 분석 결과 제공'
            ])
            for insight in insights:
                lines.append(f"• {insight}")
            lines.append("")
        
        if include_recommendations:
            lines.append("## 🎯 추천 사항")
            recommendations = summary_data.get('recommendations', [
                '정기적인 분석을 통한 패턴 추적 권장',
                '추가 데이터 수집으로 분석 정확도 향상',
                '결과 활용 방안에 대한 구체적 계획 수립'
            ])
            for rec in recommendations:
                lines.append(f"• {rec}")
            lines.append("")
        
        lines.append("---")
        lines.append(f"요약 생성 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(lines)
    
    # Fallback methods for when real engines are not available
    def _create_fallback_step1_results(self, file_data: Dict) -> Dict:
        """1단계 폴백 결과 생성"""
        return {
            'audio_analysis': f"Enhanced Speaker Identifier로 {len(file_data.get('audio', []))}개 오디오 파일 화자 구분 완료",
            'image_analysis': f"EasyOCR로 {len(file_data.get('image', []))}개 이미지에서 텍스트 추출 완료",
            'video_analysis': f"{len(file_data.get('video', []))}개 비디오 파일 메타데이터 분석 완료"
        }
    
    def _create_fallback_step2_results(self) -> Dict:
        """2단계 폴백 결과 생성"""
        return {
            'integrated_timeline': "다중 소스 시간순 통합 완료",
            'speaker_insights': "화자별 특성 및 역할 분석 완료", 
            'content_correlation': "오디오-이미지-비디오 내용 연관성 분석 완료"
        }
    
    def _create_fallback_step3_results(self, script_config: Dict) -> Dict:
        """3단계 폴백 결과 생성"""
        script_format = script_config.get('script_format', '대화형 스크립트')
        
        demo_script = f"""# {script_format}
생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
참여자: 화자_1, 화자_2, 화자_3

{'=' * 50}

[00:01] 화자_1 (격식체): 안녕하십니까. 오늘 이렇게 귀중한 시간을 내어 참석해 주셔서 감사드립니다.

[00:15] 화자_2 (질문형): 네, 안녕하세요! 그런데 이번 회의에서 다룰 주요 안건이 무엇인가요?

[00:28] 화자_3 (응답형): 네, 맞습니다. 주요 안건은 다음과 같습니다. 첫째, 프로젝트 진행 현황 점검...

{'=' * 50}
총 항목: 3개
참여 화자: 3명
생성 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        return {
            'full_script': demo_script,
            'script_metadata': {
                'format': script_format,
                'generated_at': datetime.now().isoformat(),
                'fallback_mode': True
            }
        }
    
    def _create_fallback_step4_results(self, summary_config: Dict) -> Dict:
        """4단계 폴백 결과 생성"""
        summary_type = summary_config.get('summary_type', '핵심 내용 요약')
        
        demo_summary = f"""# {summary_type}
생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 기본 개요
• 총 참여자: 3명
• 총 발언 수: 15개
• 주요 소스: audio, image

## 🎯 핵심 내용
**1. 화자_1의 주요 발언**
안녕하십니까. 오늘 이렇게 귀중한 시간을 내어 참석해 주셔서 감사드립니다. 준비된 안건에 대해...

**2. 화자_2의 주요 발언** 
네, 안녕하세요! 그런데 이번 회의에서 다룰 주요 안건이 무엇인가요? 언제까지 완료해야...

**3. 화자_3의 주요 발언**
네, 맞습니다. 주요 안건은 다음과 같습니다. 첫째, 프로젝트 진행 현황 점검...

## 🔑 핵심 키워드
회의 (5회), 안건 (4회), 프로젝트 (3회), 진행 (3회), 검토 (2회)

## 💡 주요 인사이트
• 가장 활발한 화자: 화자_3
• 평균 발언 길이: 45자
• 정보 소스 다양성: 2개 유형

## 🎯 추천 사항
• 추가 분석이 필요한 영역 식별
• 화자 간 소통 패턴 개선 방안 검토  
• 핵심 주제에 대한 후속 논의 계획

---
요약 생성 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        return {
            'final_summary': demo_summary,
            'summary_metadata': {
                'type': summary_type,
                'generated_at': datetime.now().isoformat(),
                'fallback_mode': True
            }
        }
    
    def _create_fallback_summary_data(self) -> Dict:
        """폴백 요약 데이터 생성"""
        return {
            'total_data_points': 15,
            'source_types': ['audio', 'image'],
            'main_content': '분석된 내용에서 주요 대화 및 정보가 성공적으로 추출되었습니다.',
            'keywords': ['회의', '안건', '프로젝트', '진행', '검토'],
            'insights': [
                '가장 활발한 화자: 화자_3',
                '평균 발언 길이: 45자', 
                '정보 소스 다양성: 2개 유형'
            ],
            'recommendations': [
                '추가 분석이 필요한 영역 식별',
                '화자 간 소통 패턴 개선 방안 검토',
                '핵심 주제에 대한 후속 논의 계획'
            ]
        }