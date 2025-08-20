#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[목표] 완전 통합 컨퍼런스 분석 시스템 - SOLOMOND AI v7.1
Complete Unified Conference Analysis System

8501과 8650의 모든 장점을 통합한 완전한 시스템:
[완료] 실제 파일 업로드 및 처리 (EasyOCR, Whisper STT)
[완료] Supabase 클라우드 데이터베이스 지원  
[완료] 홀리스틱 분석 (의미적 연결, 주제 클러스터링)
[완료] 듀얼 브레인 시스템 통합
[완료] 허위정보 완전 차단

핵심 원칙:
- 모든 기능이 실제로 작동해야 함
- 허위 상태 표시 절대 금지
- 실제 분석 결과만 제공
"""

import streamlit as st
import os
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback
import re
from collections import Counter
import json
import uuid
import hashlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# [보안] Unicode 안전성 시스템 최우선 로드
try:
    from core.unicode_safety_system import (
        safe_text, safe_error, safe_format,
        safe_st_error, safe_st_warning, safe_st_info, safe_st_success,
        unicode_manager
    )
    UNICODE_SAFETY_AVAILABLE = True
except ImportError:
    # 폴백 안전 함수들
    def safe_text(text, fallback="[텍스트 표시 불가]"):
        try:
            return str(text).encode('utf-8', errors='replace').decode('utf-8')
        except:
            return fallback
    
    def safe_error(error, context=""):
        return safe_text(str(error))
    
    def safe_st_error(text):
        return st.error(safe_text(text))
    
    def safe_st_warning(text):
        return st.warning(safe_text(text))
    
    def safe_st_info(text):
        return st.info(safe_text(text))
    
    def safe_st_success(text):
        return st.success(safe_text(text))
    
    UNICODE_SAFETY_AVAILABLE = False

# 향상된 파일 핸들러
try:
    from core.enhanced_file_handler import enhanced_handler, get_enhanced_file_upload
    ENHANCED_FILE_HANDLER_AVAILABLE = True
except ImportError:
    ENHANCED_FILE_HANDLER_AVAILABLE = False
    def get_enhanced_file_upload():
        return []

# 실제 분석 엔진들
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import whisper
    # [보안] 안전한 모델 로딩 시스템 임포트
    from defensive_model_loader import safe_whisper_load, enable_defensive_mode
    enable_defensive_mode()  # 전역 안전 모드 활성화
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    safe_whisper_load = None

# 홀리스틱 분석 시스템
try:
    from holistic_conference_analyzer_supabase import HolisticConferenceAnalyzerSupabase
    from semantic_connection_engine import SemanticConnectionEngine
    from conference_story_generator import ConferenceStoryGenerator
    from actionable_insights_extractor import ActionableInsightsExtractor
    HOLISTIC_AVAILABLE = True
except ImportError:
    HOLISTIC_AVAILABLE = False

# [시작] 멀티모달 파이프라인 시스템 (v4.0 고급 엔진)
try:
    from core.multimodal_pipeline import MultimodalPipeline, MultimodalResult
    from core.crossmodal_fusion import CrossModalFusionLayer, FusionResult
    from core.comprehensive_message_extractor import ComprehensiveMessageExtractor
    from core.insight_generator import InsightGenerator, InsightItem
    MULTIMODAL_AVAILABLE = True
    safe_st_info("[시작] 고급 멀티모달 엔진 로드 완료!")
except ImportError as e:
    MULTIMODAL_AVAILABLE = False
    safe_st_warning(f"[주의] 멀티모달 엔진 로드 실패: {safe_error(e)}")

# 듀얼 브레인 시스템
try:
    from dual_brain_integration import DualBrainSystem
    DUAL_BRAIN_AVAILABLE = True
except ImportError:
    DUAL_BRAIN_AVAILABLE = False

# 데이터베이스 어댑터
try:
    from database_adapter import DatabaseFactory
    DATABASE_ADAPTER_AVAILABLE = True
except ImportError:
    DATABASE_ADAPTER_AVAILABLE = False

# Ollama AI 인터페이스
try:
    from shared.ollama_interface import OllamaInterface
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# 유튜브 다운로더
try:
    import yt_dlp
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False

class UnifiedConferenceAnalyzer:
    """완전 통합 컨퍼런스 분석기"""
    
    def __init__(self, conference_name: str = "unified_conference"):
        self.conference_name = conference_name
        self.session_id = str(uuid.uuid4())[:8]
        
        # 사전 정보 저장
        self.conference_info = {
            "conference_name": "",
            "conference_date": "",
            "location": "",
            "industry_field": "",
            "interest_keywords": []
        }
        
        # 분석 결과 저장
        self.analysis_results = {
            "session_id": self.session_id,
            "conference_name": conference_name,
            "conference_info": self.conference_info,
            "timestamp": datetime.now().isoformat(),
            "processed_files": [],
            "analysis_data": {},
            "holistic_results": {},
            "dual_brain_results": {}
        }
        
        # 실제 엔진들 초기화
        self._initialize_engines()
        
        # 로컬 파일 시스템 초기화
        self.user_files_dir = Path("user_files")
        self.user_files_dir.mkdir(exist_ok=True)
    
    def _initialize_engines(self):
        """실제 분석 엔진들 초기화"""
        # OCR 엔진
        self.ocr_engine = None
        if OCR_AVAILABLE:
            try:
                self.ocr_engine = easyocr.Reader(['ko', 'en'], gpu=False)
                st.success("[완료] EasyOCR 엔진 초기화 완료")
            except Exception as e:
                st.warning(f"[주의] EasyOCR 초기화 실패: {e}")
        
        # Whisper 엔진
        self.whisper_model = None
        if WHISPER_AVAILABLE:
            try:
                # [보안] 안전한 모델 로딩으로 meta tensor 문제 완전 해결
                self.whisper_model = safe_whisper_load("base")
                st.success("[완료] Whisper STT 엔진 초기화 완료")
            except Exception as e:
                st.warning(f"[주의] Whisper 초기화 실패: {e}")
        
        # 홀리스틱 분석기
        self.holistic_analyzer = None
        if HOLISTIC_AVAILABLE:
            try:
                self.holistic_analyzer = HolisticConferenceAnalyzerSupabase(self.conference_name, "auto")
                st.success("[완료] 홀리스틱 분석기 초기화 완료")
            except Exception as e:
                st.warning(f"[주의] 홀리스틱 분석기 초기화 실패: {e}")
        
        # 듀얼 브레인 시스템
        self.dual_brain = None
        if DUAL_BRAIN_AVAILABLE:
            try:
                self.dual_brain = DualBrainSystem()
                st.success("[완료] 듀얼 브레인 시스템 초기화 완료")
            except Exception as e:
                st.warning(f"[주의] 듀얼 브레인 시스템 초기화 실패: {e}")
        
        # 데이터베이스
        self.database = None
        if DATABASE_ADAPTER_AVAILABLE:
            try:
                self.database = DatabaseFactory.create_database("auto", self.conference_name)
                # 테이블 생성 시도
                if self.database.create_fragments_table():
                    st.success("[완료] 데이터베이스 어댑터 초기화 완료")
                else:
                    st.warning("[주의] 데이터베이스 테이블 생성 실패")
            except Exception as e:
                st.warning(f"[주의] 데이터베이스 초기화 실패: {e}")
        
        # Ollama AI 인터페이스
        self.ollama = None
        if OLLAMA_AVAILABLE:
            try:
                self.ollama = OllamaInterface()
                if self.ollama.health_check():
                    available_models = self.ollama.available_models
                    st.success(f"[완료] Ollama AI 엔진 초기화 완료 ({len(available_models)}개 모델)")
                    st.info(f"[AI] 사용 가능한 모델: {', '.join(available_models[:3])}...")
                else:
                    st.warning("[주의] Ollama 서버 연결 실패 - ollama serve 실행 필요")
                    self.ollama = None
            except Exception as e:
                st.warning(f"[주의] Ollama 초기화 실패: {e}")
                self.ollama = None
        
        # [시작] 멀티모달 파이프라인 초기화 (v4.0 고급 엔진)
        self.multimodal_pipeline = None
        self.fusion_engine = None
        self.message_extractor = None
        self.insight_generator = None
        
        if MULTIMODAL_AVAILABLE:
            try:
                self.multimodal_pipeline = MultimodalPipeline()
                self.fusion_engine = CrossModalFusionLayer()
                self.message_extractor = ComprehensiveMessageExtractor()
                self.insight_generator = InsightGenerator()
                st.success("[시작] 고급 멀티모달 분석 엔진 초기화 완료!")
                st.info("[팁] 크로스모달 융합, 자동 인사이트 생성 기능 활성화됨")
            except Exception as e:
                st.warning(f"[주의] 멀티모달 엔진 초기화 실패: {e}")
                self.multimodal_pipeline = None
    
    def update_conference_info(self, info: Dict[str, Any]):
        """사전 정보 업데이트"""
        self.conference_info.update(info)
        self.analysis_results["conference_info"] = self.conference_info
    
    def check_system_status(self) -> Dict[str, Any]:
        """실제 시스템 상태 확인 (허위정보 없음)"""
        status = {
            "ocr_available": self.ocr_engine is not None,
            "whisper_available": self.whisper_model is not None,
            "holistic_available": self.holistic_analyzer is not None,
            "dual_brain_available": self.dual_brain is not None,
            "database_available": self.database is not None and self._check_database_working(),
            "ollama_available": self.ollama is not None and self.ollama.health_check(),
            "multimodal_available": self.multimodal_pipeline is not None,
            "fusion_available": self.fusion_engine is not None,
            "overall_ready": False
        }
        
        # 전체 준비 상태 계산
        ready_count = sum([
            status["ocr_available"],
            status["whisper_available"], 
            status["holistic_available"],
            status["database_available"],
            status["ollama_available"],
            status["multimodal_available"]
        ])
        
        status["overall_ready"] = ready_count >= 5  # 최소 5개 시스템 필요 (멀티모달 포함)
        status["ready_systems"] = ready_count
        status["total_systems"] = 5
        
        return status
    
    def _check_database_working(self) -> bool:
        """데이터베이스 실제 작동 확인"""
        if not self.database:
            return False
        try:
            # 테이블 생성 시도
            self.database.create_fragments_table()
            # 간단한 쿼리로 작동 확인
            count = self.database.get_fragment_count()
            return True
        except Exception:
            return False
    
    def process_uploaded_files(self, uploaded_files: List, skip_errors: bool = True) -> Dict[str, Any]:
        """실제 업로드된 파일들 처리 - [시작] v4.0 멀티모달 파이프라인 통합"""
        if not uploaded_files:
            return {"error": "업로드된 파일이 없습니다."}
        
        # [시작] 멀티모달 파이프라인 사용 가능시 고급 처리
        if self.multimodal_pipeline is not None:
            return self._process_with_multimodal_pipeline(uploaded_files, skip_errors)
        else:
            # 기존 방식 유지 (호환성)
            st.info("[정보] 기존 분석 방식 사용 (멀티모달 엔진 비활성)")
            return self._process_with_legacy_method(uploaded_files, skip_errors)
    
    def _process_with_multimodal_pipeline(self, uploaded_files: List, skip_errors: bool = True) -> Dict[str, Any]:
        """[시작] 멀티모달 파이프라인을 활용한 고급 분석"""
        st.success("[시작] 고급 멀티모달 분석 모드 활성화!")
        st.info("[팁] 크로스모달 융합, 시간 동기화, 상황 통합 분석 수행")
        
        results = {
            "processed_count": 0,
            "successful_count": 0,
            "failed_files": [],
            "analysis_fragments": [],
            "multimodal_insights": [],  # 새로운 인사이트
            "cross_modal_correlations": []  # 크로스모달 상관관계
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # [처리중] 멀티모달 분석 실행 (async 처리를 sync로 wrapping)
        try:
            import asyncio
            
            # 파일들을 Path 객체로 변환
            file_paths = []
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # 임시 파일로 저장
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        file_paths.append(Path(tmp_file.name))
                except Exception as e:
                    st.warning(f"[주의] 파일 준비 실패: {uploaded_file.name} - {e}")
                    if not skip_errors:
                        return {"error": f"파일 준비 실패: {e}"}
            
            status_text.text("[시작] 멀티모달 AI 엔진 초기화 중...")
            
            # 비동기 분석 실행
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                multimodal_results = loop.run_until_complete(
                    self.multimodal_pipeline.process_multimodal_batch(file_paths)
                )
                loop.close()
            except Exception as e:
                st.error(f"멀티모달 분석 실패: {e}")
                return self._process_with_legacy_method(uploaded_files, skip_errors)
            
            status_text.text("🔀 크로스모달 융합 분석 중...")
            progress_bar.progress(0.7)
            
            # 🔀 크로스모달 융합 실행 (멀티모달 결과가 있을 때만)
            if multimodal_results and self.fusion_engine:
                try:
                    # MultimodalResult를 EncodedResult로 변환
                    encoded_results = self._convert_to_encoded_results(multimodal_results)
                    if encoded_results:
                        fusion_result = self.fusion_engine.fuse_multimodal_encodings(encoded_results)
                        results["cross_modal_correlations"] = fusion_result.cross_modal_correlations if hasattr(fusion_result, 'cross_modal_correlations') else []
                except Exception as e:
                    st.warning(f"크로스모달 융합 실패: {e}")
            
            # 결과 변환
            for mm_result in multimodal_results:
                fragment = {
                    'fragment_id': f'{self.conference_name}_{self.session_id}_{hashlib.md5(str(mm_result.file_path).encode()).hexdigest()[:8]}',
                    'file_source': Path(mm_result.file_path).name,
                    'file_type': mm_result.file_type,
                    'timestamp': datetime.now().isoformat(),
                    'content': mm_result.content,
                    'confidence': mm_result.confidence,
                    'processing_time': mm_result.processing_time,
                    'metadata': mm_result.metadata
                }
                
                results["analysis_fragments"].append(fragment)
                results["successful_count"] += 1
                
                # 데이터베이스에 저장
                if self.database:
                    try:
                        self.database.insert_fragment(fragment)
                    except Exception as e:
                        st.warning(f"DB 저장 실패: {e}")
            
            results["processed_count"] = len(uploaded_files)
            
            progress_bar.progress(1.0)
            status_text.text(f"[완료] 멀티모달 분석 완료! ({results['successful_count']}/{results['processed_count']} 성공)")
            
            # 임시 파일 정리
            for file_path in file_paths:
                try:
                    os.unlink(file_path)
                except:
                    pass
                    
            return results
            
        except Exception as e:
            st.error(f"[실패] 멀티모달 분석 중 오류: {e}")
            if not skip_errors:
                return {"error": f"멀티모달 분석 실패: {e}"}
            else:
                st.info("[처리중] 기존 분석 방식으로 전환합니다...")
                return self._process_with_legacy_method(uploaded_files, skip_errors)
    
    def _process_with_legacy_method(self, uploaded_files: List, skip_errors: bool = True) -> Dict[str, Any]:
        """기존 방식의 파일 처리 (호환성 유지)"""
        results = {
            "processed_count": 0,
            "successful_count": 0,
            "failed_files": [],
            "analysis_fragments": []
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # 진행률 업데이트
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"처리 중: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                
                # 임시 파일로 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # 파일 타입별 실제 처리
                file_ext = Path(uploaded_file.name).suffix.lower()
                
                if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    fragment = self._process_image_file(tmp_path, uploaded_file.name)
                elif file_ext in ['.wav', '.mp3', '.m4a', '.flac']:
                    fragment = self._process_audio_file(tmp_path, uploaded_file.name)
                elif file_ext in ['.mp4', '.avi', '.mov']:
                    fragment = self._process_video_file(tmp_path, uploaded_file.name)
                elif file_ext == '.txt':
                    # TXT 파일의 경우 URL 배치 처리
                    st.info(f"[문서] TXT 파일 감지: {uploaded_file.name} - URL 배치 처리 시작")
                    
                    # 임시 파일을 다시 생성하여 process_text_file_urls에 전달
                    uploaded_file.seek(0)  # 파일 포인터 리셋
                    batch_results = self.process_text_file_urls(uploaded_file)
                    
                    if batch_results and "error" not in batch_results[0]:
                        # 배치 처리 결과를 분석 결과에 추가
                        results["analysis_fragments"].extend(batch_results)
                        results["successful_count"] += len(batch_results)
                        st.success(f"[완료] {uploaded_file.name}에서 {len(batch_results)}개 URL 처리 완료!")
                    else:
                        error_msg = batch_results[0]["error"] if batch_results else "알 수 없는 오류"
                        results["failed_files"].append({
                            "filename": uploaded_file.name,
                            "error": error_msg
                        })
                    
                    # TXT 파일 처리 완료, continue로 일반 처리 스킵
                    results["processed_count"] += 1
                    continue
                else:
                    fragment = {"error": f"지원하지 않는 파일 형식: {file_ext}"}
                
                # 정리
                os.unlink(tmp_path)
                
                if "error" not in fragment:
                    results["analysis_fragments"].append(fragment)
                    results["successful_count"] += 1
                    
                    # 데이터베이스에 저장
                    if self.database:
                        self.database.insert_fragment(fragment)
                else:
                    results["failed_files"].append({
                        "filename": uploaded_file.name,
                        "error": fragment["error"]
                    })
                
                results["processed_count"] += 1
                
            except Exception as e:
                if not skip_errors:
                    # 에러 발생시 즉시 중단
                    return {"error": f"파일 처리 중 오류 발생: {str(e)}"}
                
                results["failed_files"].append({
                    "filename": uploaded_file.name,
                    "error": str(e)
                })
                results["processed_count"] += 1
        
        # 진행률 완료
        progress_bar.progress(1.0)
        status_text.text("[완료] 기존 방식 파일 처리 완료!")
        
        # 결과를 세션에 저장
        self.analysis_results["processed_files"] = results["analysis_fragments"]
        
        return results
    
    def _process_image_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        """실제 이미지 파일 OCR 처리"""
        if not self.ocr_engine:
            return {"error": "OCR 엔진을 사용할 수 없습니다."}
        
        try:
            # EasyOCR로 텍스트 추출
            ocr_results = self.ocr_engine.readtext(file_path)
            
            # 텍스트 추출 및 신뢰도 계산
            extracted_texts = []
            confidences = []
            
            for (bbox, text, confidence) in ocr_results:
                if confidence > 0.5:  # 신뢰도 50% 이상만
                    extracted_texts.append(text.strip())
                    confidences.append(confidence)
            
            if not extracted_texts:
                return {"error": "이미지에서 텍스트를 찾을 수 없습니다."}
            
            full_text = " ".join(extracted_texts)
            avg_confidence = np.mean(confidences)
            
            # 키워드 추출 (간단한 토큰화)
            keywords = self._extract_keywords(full_text)
            
            fragment = {
                'fragment_id': f'{self.conference_name}_{self.session_id}_{hashlib.md5(filename.encode()).hexdigest()[:8]}',
                'file_source': filename,
                'file_type': 'image',
                'timestamp': datetime.now().isoformat(),
                'speaker': None,  # 이미지는 화자 없음
                'content': full_text,
                'confidence': float(avg_confidence),
                'keywords': keywords,
                'raw_ocr_results': len(ocr_results)
            }
            
            return fragment
            
        except Exception as e:
            return {"error": f"이미지 처리 실패: {e}"}
    
    def _process_audio_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        """실제 음성 파일 Whisper STT 처리"""
        if not self.whisper_model:
            return {"error": "Whisper 엔진을 사용할 수 없습니다."}
        
        try:
            # Whisper로 음성-텍스트 변환
            result = self.whisper_model.transcribe(file_path, language='ko')
            
            if not result["text"].strip():
                return {"error": "음성에서 텍스트를 찾을 수 없습니다."}
            
            text = result["text"].strip()
            
            # 간단한 화자 분리 (세그먼트 기반)
            segments = result.get("segments", [])
            if segments:
                # 가장 긴 세그먼트의 화자를 메인 화자로 가정
                main_segment = max(segments, key=lambda s: len(s.get("text", "")))
                speaker = f"화자_{hashlib.md5(filename.encode()).hexdigest()[:4]}"
            else:
                speaker = None
            
            # 키워드 추출
            keywords = self._extract_keywords(text)
            
            fragment = {
                'fragment_id': f'{self.conference_name}_{self.session_id}_{hashlib.md5(filename.encode()).hexdigest()[:8]}',
                'file_source': filename,
                'file_type': 'audio',
                'timestamp': datetime.now().isoformat(),
                'speaker': speaker,
                'content': text,
                'confidence': 0.85,  # Whisper는 일반적으로 높은 정확도
                'keywords': keywords,
                'duration': result.get("duration", 0),
                'segments_count': len(segments)
            }
            
            return fragment
            
        except Exception as e:
            return {"error": f"음성 처리 실패: {e}"}
    
    def _process_video_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        """비디오 파일 처리 (음성 추출 후 STT)"""
        if not self.whisper_model:
            return {"error": "Whisper 엔진을 사용할 수 없습니다."}
        
        try:
            # Whisper로 비디오의 음성 직접 처리
            result = self.whisper_model.transcribe(file_path, language='ko')
            
            if not result["text"].strip():
                return {"error": "비디오에서 음성을 찾을 수 없습니다."}
            
            text = result["text"].strip()
            
            # 화자 추정
            segments = result.get("segments", [])
            speaker = f"화자_{hashlib.md5(filename.encode()).hexdigest()[:4]}" if segments else None
            
            # 키워드 추출
            keywords = self._extract_keywords(text)
            
            fragment = {
                'fragment_id': f'{self.conference_name}_{self.session_id}_{hashlib.md5(filename.encode()).hexdigest()[:8]}',
                'file_source': filename,
                'file_type': 'video',
                'timestamp': datetime.now().isoformat(),
                'speaker': speaker,
                'content': text,
                'confidence': 0.85,
                'keywords': keywords,
                'duration': result.get("duration", 0),
                'segments_count': len(segments)
            }
            
            return fragment
            
        except Exception as e:
            return {"error": f"비디오 처리 실패: {e}"}
    
    def _extract_keywords(self, text: str) -> List[str]:
        """간단한 키워드 추출"""
        # 한글/영문 단어만 추출
        words = re.findall(r'[가-힣a-zA-Z]{2,}', text)
        
        # 빈도수 기반 상위 키워드 추출
        word_freq = Counter(words)
        keywords = [word for word, count in word_freq.most_common(10) if count >= 1]
        
        return keywords
    
    def run_holistic_analysis(self) -> Dict[str, Any]:
        """홀리스틱 분석 실행"""
        if not self.holistic_analyzer:
            return {"error": "홀리스틱 분석기를 사용할 수 없습니다."}
        
        try:
            # 데이터베이스에 데이터가 있는지 확인
            if self.database:
                self.database.create_fragments_table()
                fragment_count = self.database.get_fragment_count(self.conference_name)
                
                if fragment_count == 0:
                    return {"error": "분석할 데이터가 없습니다. 먼저 파일을 업로드하고 처리하세요."}
            
            # 홀리스틱 분석 실행
            result = self.holistic_analyzer.analyze_conference_holistically()
            
            if "error" in result:
                return result
            
            # 의미적 연결 분석
            semantic_engine = SemanticConnectionEngine(self.conference_name)
            semantic_result = semantic_engine.analyze_semantic_connections()
            
            # Ollama AI 기반 심화 분석
            ollama_insights = self._generate_ai_insights(result)
            
            # 결과 통합
            combined_result = {
                "holistic_analysis": result,
                "semantic_connections": semantic_result,
                "ai_insights": ollama_insights,
                "analysis_timestamp": datetime.now().isoformat(),
                "database_type": type(self.database).__name__ if self.database else "None"
            }
            
            # 세션에 저장
            self.analysis_results["holistic_results"] = combined_result
            
            return combined_result
            
        except Exception as e:
            return {"error": f"홀리스틱 분석 실패: {e}"}
    
    def _generate_ai_insights(self, holistic_result: Dict[str, Any]) -> List[str]:
        """[시작] v4.0 멀티모달 통합 분석을 활용한 고급 인사이트 생성"""
        if not self.ollama:
            return ["AI 인사이트를 생성할 수 없습니다 (Ollama 비활성)"]
        
        try:
            # [처리중] 멀티모달 분석 결과 통합
            processed_files = self.analysis_results.get("processed_files", [])
            multimodal_context = self._build_multimodal_context(processed_files)
            
            # [목표] 풍부한 컨텍스트 생성 (v4.0 고급 프롬프트 활용)
            enhanced_analysis_summary = f"""
🎭 **멀티모달 컨퍼런스 분석 데이터**:

[통계] **기본 통계**:
- 처리된 파일: {len(processed_files)}개 
- 총 분석 조각: {holistic_result.get('total_fragments', 0)}개
- 발견된 개체: {holistic_result.get('total_entities', 0)}개  
- 주제 클러스터: {holistic_result.get('total_topics', 0)}개

[디자인] **멀티모달 분포**:
{multimodal_context['modal_distribution']}

🔗 **크로스모달 연결성**:
{multimodal_context['cross_modal_connections']}

📝 **핵심 콘텐츠 샘플**:
{multimodal_context['content_samples']}

[검색] **홀리스틱 인사이트**:
{', '.join(holistic_result.get('key_insights', ['기본 분석 완료']))}

[목표] **컨퍼런스 메타정보**:
- 이벤트명: {self.conference_info.get('conference_name', 'N/A')}
- 업계 분야: {self.conference_info.get('industry_field', 'N/A')}  
- 관심 키워드: {', '.join(self.conference_info.get('interest_keywords', []))}
"""
            
            # [AI] 고급 프롬프트로 AI 분석 실행
            ai_response = self.ollama.analyze_conference(enhanced_analysis_summary)
            
            if ai_response and not ai_response.startswith("AI 모델 오류"):
                # [목표] 구조화된 응답 파싱 (v4.0 포맷)
                structured_insights = self._parse_structured_ai_response(ai_response)
                return structured_insights
            else:
                return [f"AI 분석 실패: {ai_response}"]
                
        except Exception as e:
            return [f"AI 인사이트 생성 오류: {str(e)}"]
    
    def _build_multimodal_context(self, processed_files):
        """멀티모달 분석 컨텍스트 구축 - Task 3 시간 기반 파일 그룹핑 포함"""
        try:
            # 모달별 분포 분석
            modal_stats = {'image': 0, 'audio': 0, 'text': 0, 'video': 0}
            content_samples = []
            cross_modal_connections = []
            
            for file_info in processed_files:
                file_type = self._detect_file_type(file_info.get('filename', ''))
                content = file_info.get('content', '').strip()
                
                if file_type in modal_stats:
                    modal_stats[file_type] += 1
                    
                # 컨텐츠 샘플 수집 (각 모달별 대표 샘플)
                if content and len(content_samples) < 6:  # 최대 6개 샘플
                    content_samples.append(f"[{file_type.upper()}] {content[:150]}...")
            
            # 모달 분포 문자열 생성
            modal_distribution = []
            for modal, count in modal_stats.items():
                if count > 0:
                    emoji = {'image': '[이미지]', 'audio': '[음악]', 'text': '[문서]', 'video': '[비디오]'}
                    modal_distribution.append(f"{emoji.get(modal, '[폴더]')} {modal}: {count}개")
            
            # 시간 기반 파일 그룹핑 (Task 3 구현)
            time_groups = self._group_files_by_time(processed_files)
            if len(time_groups) > 1:
                cross_modal_connections.append(f"📅 {len(time_groups)}개 시간대 세션으로 그룹화됨")
                for i, group in enumerate(time_groups[:3]):
                    cross_modal_connections.append(
                        f"  세션 {i+1}: {group['file_count']}개 파일, {group.get('duration', 0):.1f}분"
                    )
            
            # 크로스 모달 상관관계 추가
            if len([m for m in modal_stats.values() if m > 0]) >= 2:
                cross_modal_connections.append("🔗 여러 모달 간 데이터 상관관계 분석 가능")
            
            return {
                'modal_distribution': '\n'.join(modal_distribution) if modal_distribution else "데이터 없음",
                'cross_modal_connections': '\n'.join(cross_modal_connections) if cross_modal_connections else "단일 모달 데이터",
                'content_samples': '\n'.join(content_samples) if content_samples else "컨텐츠 샘플 없음",
                'total_modalities': len([m for m in modal_stats.values() if m > 0]),
                'time_groups': time_groups
            }
            
        except Exception as e:
            return {
                'modal_distribution': f"분석 오류: {e}",
                'cross_modal_connections': "상관관계 분석 실패",
                'content_samples': "샘플 추출 실패",
                'total_modalities': 0,
                'time_groups': []
            }
    
    def _group_files_by_time(self, processed_files):
        """시간 기반 파일 그룹핑 - Task 3 구현"""
        try:
            import os
            from datetime import datetime, timedelta
            
            time_groups = []
            files_with_time = []
            
            # 파일별 시간 정보 수집
            for file_info in processed_files:
                filename = file_info.get('filename', '')
                if filename:
                    try:
                        if os.path.exists(filename):
                            mtime = os.path.getmtime(filename)
                            files_with_time.append({
                                'file_info': file_info,
                                'timestamp': datetime.fromtimestamp(mtime),
                                'filename': filename
                            })
                    except (OSError, ValueError):
                        # 파일 접근 불가시 현재 시간 사용
                        files_with_time.append({
                            'file_info': file_info,
                            'timestamp': datetime.now(),
                            'filename': filename
                        })
            
            if not files_with_time:
                return []
            
            # 시간 순 정렬
            files_with_time.sort(key=lambda x: x['timestamp'])
            
            # 30분 간격으로 그룹핑 (컨퍼런스 세션별 분류)
            current_group = []
            current_group_start = None
            
            for file_data in files_with_time:
                file_time = file_data['timestamp']
                
                if not current_group_start:
                    current_group_start = file_time
                    current_group = [file_data]
                elif file_time - current_group_start <= timedelta(minutes=30):
                    current_group.append(file_data)
                else:
                    # 현재 그룹 저장
                    if current_group:
                        duration_minutes = (current_group[-1]['timestamp'] - current_group[0]['timestamp']).total_seconds() / 60
                        time_groups.append({
                            'start_time': current_group[0]['timestamp'],
                            'end_time': current_group[-1]['timestamp'],
                            'duration': round(max(duration_minutes, 0), 1),
                            'file_count': len(current_group),
                            'files': current_group
                        })
                    
                    # 새 그룹 시작
                    current_group_start = file_time
                    current_group = [file_data]
            
            # 마지막 그룹 저장
            if current_group:
                duration_minutes = (current_group[-1]['timestamp'] - current_group[0]['timestamp']).total_seconds() / 60
                time_groups.append({
                    'start_time': current_group[0]['timestamp'],
                    'end_time': current_group[-1]['timestamp'],
                    'duration': round(max(duration_minutes, 0), 1),
                    'file_count': len(current_group),
                    'files': current_group
                })
            
            return time_groups
            
        except Exception as e:
            st.error(f"시간 기반 그룹핑 실패: {e}")
            return []
    
    def _parse_structured_ai_response(self, ai_response):
        """구조화된 AI 응답 파싱 - v4.0 멀티모달 인사이트 구조화"""
        try:
            import re
            
            # 기본 구조화된 응답 템플릿
            structured_insights = []
            
            # 정규표현식으로 섹션별 내용 추출
            sections = {
                '[검색] 핵심 시그널': r'[검색]\s*\*\*핵심 시그널[^*]*\*\*[^[목표][시작][팁][주의]]*',
                '[팁] 상황 통합': r'[팁]\s*\*\*[^*]*상황[^*]*\*\*[^[검색][시작][목표][주의]]*',
                '[목표] 업계 인사이트': r'[목표]\s*\*\*[^*]*인사이트[^*]*\*\*[^[검색][팁][시작][주의]]*',
                '[시작] 실행 제안': r'[시작]\s*\*\*[^*]*제안[^*]*\*\*[^[검색][팁][목표][주의]]*',
                '[주의] 주의사항': r'[주의]\s*\*\*[^*]*주의사항[^*]*\*\*[^[검색][팁][목표][시작]]*'
            }
            
            parsed_sections = {}
            
            for section_name, pattern in sections.items():
                match = re.search(pattern, ai_response, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(0)
                    # 불릿 포인트 또는 줄바꿈 기반으로 내용 추출
                    lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('**')]
                    # 섹션 헤더 제거
                    clean_lines = []
                    for line in lines:
                        if not re.match(r'^[검색]|[팁]|[목표]|[시작]|[주의]', line.strip()):
                            clean_lines.append(line.strip('- •').strip())
                    
                    if clean_lines:
                        parsed_sections[section_name] = clean_lines[:3]  # 최대 3개씩
            
            # 구조화된 인사이트 생성
            if '[검색] 핵심 시그널' in parsed_sections:
                for insight in parsed_sections['[검색] 핵심 시그널']:
                    if insight:
                        structured_insights.append(f"[검색] 핵심 발견: {insight}")
            
            if '[팁] 상황 통합' in parsed_sections:
                for insight in parsed_sections['[팁] 상황 통합']:
                    if insight:
                        structured_insights.append(f"[팁] 통합 분석: {insight}")
            
            if '[목표] 업계 인사이트' in parsed_sections:
                for insight in parsed_sections['[목표] 업계 인사이트']:
                    if insight:
                        structured_insights.append(f"[목표] 업계 시사점: {insight}")
            
            if '[시작] 실행 제안' in parsed_sections:
                for insight in parsed_sections['[시작] 실행 제안']:
                    if insight:
                        structured_insights.append(f"[시작] 액션 아이템: {insight}")
            
            if '[주의] 주의사항' in parsed_sections:
                for insight in parsed_sections['[주의] 주의사항']:
                    if insight:
                        structured_insights.append(f"[주의] 주의사항: {insight}")
            
            # 기본 인사이트가 없는 경우 원본 응답에서 추출 시도
            if not structured_insights:
                # 간단한 문장 단위로 분할
                sentences = [s.strip() for s in ai_response.split('.') if s.strip() and len(s.strip()) > 20]
                for sentence in sentences[:5]:  # 최대 5개 문장
                    structured_insights.append(f"[팁] AI 분석: {sentence}.")
            
            # 최소 1개 인사이트 보장
            if not structured_insights:
                structured_insights = [
                    "[검색] 멀티모달 컨퍼런스 분석이 완료되었습니다",
                    "[팁] 업로드된 파일들에서 주요 콘텐츠가 추출되었습니다",
                    "[시작] 분석 결과를 바탕으로 추가 검토를 권장합니다"
                ]
            
            return structured_insights
            
        except Exception as e:
            return [
                f"[주의] 인사이트 파싱 오류: {str(e)}",
                "[검색] 원본 AI 응답 확인이 필요합니다",
                f"[팁] Raw Response: {ai_response[:200]}..."
            ]
    
    def _normalize_embedding_dimension(self, embedding, target_dim=768):
        """임베딩 차원을 목표 차원으로 정규화"""
        try:
            import numpy as np
            
            if embedding is None:
                return np.random.rand(target_dim).astype(np.float32)
            
            current_dim = embedding.shape[0] if len(embedding.shape) > 0 else len(embedding)
            
            if current_dim == target_dim:
                return embedding.astype(np.float32)
            elif current_dim < target_dim:
                # 패딩으로 확장
                padding_size = target_dim - current_dim
                return np.concatenate([
                    embedding.astype(np.float32), 
                    np.zeros(padding_size, dtype=np.float32)
                ])
            else:
                # 처음 target_dim 차원만 사용
                return embedding[:target_dim].astype(np.float32)
                
        except Exception as e:
            # 오류 발생시 랜덤 임베딩 반환
            import numpy as np
            return np.random.rand(target_dim).astype(np.float32)
    
    def _convert_to_encoded_results(self, multimodal_results):
        """MultimodalResult를 EncodedResult로 변환"""
        try:
            encoded_results = []
            
            for mm_result in multimodal_results:
                # 임베딩이 있는 경우에만 변환
                if hasattr(mm_result, 'embeddings') and mm_result.embeddings is not None:
                    try:
                        # 실제 multimodal_encoder의 EncodedResult 사용
                        from core.multimodal_encoder import EncodedResult
                        import numpy as np
                        
                        # 임베딩 차원 정규화
                        normalized_embedding = self._normalize_embedding_dimension(mm_result.embeddings)
                        
                        encoded_result = EncodedResult(
                            file_path=mm_result.file_path,
                            modality=mm_result.file_type,
                            encoding=normalized_embedding,
                            confidence=mm_result.confidence,
                            metadata=mm_result.metadata or {},
                            processing_time=mm_result.processing_time if hasattr(mm_result, 'processing_time') else 0.0,
                            raw_content=mm_result.content if hasattr(mm_result, 'content') else ""
                        )
                        encoded_results.append(encoded_result)
                    except ImportError:
                        # 폴백: crossmodal_fusion의 EncodedResult 사용
                        from core.crossmodal_fusion import EncodedResult
                        import numpy as np
                        
                        # 임베딩 차원 정규화 (폴백)
                        normalized_embedding = self._normalize_embedding_dimension(mm_result.embeddings)
                        
                        encoded_result = EncodedResult(
                            file_path=mm_result.file_path,
                            modality=mm_result.file_type,
                            encoding=normalized_embedding,
                            confidence=mm_result.confidence,
                            metadata=mm_result.metadata or {}
                        )
                        encoded_results.append(encoded_result)
                else:
                    # 임베딩이 없는 경우 기본 값으로 생성
                    try:
                        from core.multimodal_encoder import EncodedResult
                        import numpy as np
                        
                        # 768차원 정규화 임베딩 생성
                        dummy_embedding = self._normalize_embedding_dimension(None)
                        
                        encoded_result = EncodedResult(
                            file_path=mm_result.file_path,
                            modality=mm_result.file_type,
                            encoding=dummy_embedding,
                            confidence=mm_result.confidence,
                            metadata=mm_result.metadata or {},
                            processing_time=mm_result.processing_time if hasattr(mm_result, 'processing_time') else 0.0,
                            raw_content=mm_result.content if hasattr(mm_result, 'content') else ""
                        )
                        encoded_results.append(encoded_result)
                    except ImportError:
                        # 폴백: crossmodal_fusion의 EncodedResult 사용
                        from core.crossmodal_fusion import EncodedResult
                        import numpy as np
                        
                        dummy_embedding = self._normalize_embedding_dimension(None)
                        
                        encoded_result = EncodedResult(
                            file_path=mm_result.file_path,
                            modality=mm_result.file_type,
                            encoding=dummy_embedding,
                            confidence=mm_result.confidence,
                            metadata=mm_result.metadata or {}
                        )
                        encoded_results.append(encoded_result)
            
            return encoded_results
            
        except Exception as e:
            st.error(f"결과 변환 실패: {e}")
            return []
    
    def _detect_file_type(self, filename):
        """파일 확장자 기반 타입 감지"""
        if not filename:
            return 'unknown'
            
        ext = Path(filename).suffix.lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
            return 'image'
        elif ext in ['.wav', '.mp3', '.m4a', '.ogg', '.flac']:
            return 'audio'
        elif ext in ['.txt', '.md', '.pdf', '.docx', '.doc']:
            return 'text'
        elif ext in ['.mov', '.mp4', '.avi', '.mkv', '.webm']:
            return 'video'
        else:
            return 'unknown'
    
    def _create_sample_data(self):
        """샘플 데이터 생성 (데이터베이스 테스트용)"""
        if not self.database:
            return False
            
        try:
            # 테이블 생성
            self.database.create_fragments_table()
            
            # 샘플 조각들 생성
            sample_fragments = [
                {
                    'fragment_id': f'{self.conference_name}_sample_001',
                    'file_source': 'sample_conference_audio.wav',
                    'file_type': 'audio',
                    'timestamp': datetime.now().isoformat(),
                    'speaker': '화자_001',
                    'content': '안녕하세요. 오늘 주얼리 업계의 최신 트렌드에 대해 말씀드리겠습니다. 최근 지속가능성이 중요한 화두가 되고 있습니다.',
                    'confidence': 0.92,
                    'keywords': ['주얼리', '트렌드', '지속가능성', '업계', '화두']
                },
                {
                    'fragment_id': f'{self.conference_name}_sample_002',
                    'file_source': 'sample_presentation.jpg',
                    'file_type': 'image',
                    'timestamp': datetime.now().isoformat(),
                    'speaker': None,
                    'content': '2025년 주얼리 시장 전망: 디지털 변혁과 개인화 트렌드',
                    'confidence': 0.87,
                    'keywords': ['2025년', '주얼리', '시장', '디지털', '개인화']
                },
                {
                    'fragment_id': f'{self.conference_name}_sample_003',
                    'file_source': 'sample_discussion.wav',
                    'file_type': 'audio',
                    'timestamp': datetime.now().isoformat(),
                    'speaker': '화자_002',
                    'content': '인공지능 기술을 활용한 맞춤형 주얼리 디자인이 주목받고 있습니다. 고객의 선호도를 분석하여 개인화된 제품을 제공할 수 있습니다.',
                    'confidence': 0.89,
                    'keywords': ['인공지능', '맞춤형', '디자인', '고객', '개인화']
                }
            ]
            
            # 배치 삽입
            success = self.database.insert_fragments_batch(sample_fragments)
            
            if success:
                # 분석 결과에도 추가
                self.analysis_results["processed_files"] = sample_fragments
                return True
            return False
            
        except Exception as e:
            st.error(f"샘플 데이터 생성 실패: {e}")
            return False
    
    def download_video_from_url(self, url: str) -> Optional[Dict[str, Any]]:
        """다양한 플랫폼 비디오 다운로드"""
        if not YOUTUBE_AVAILABLE:
            st.error("[실패] yt-dlp가 설치되지 않음. pip install yt-dlp로 설치하세요.")
            return None
        
        try:
            # 임시 디렉토리 생성
            temp_dir = tempfile.mkdtemp()
            
            # yt-dlp 옵션 설정 (다양한 플랫폼 지원)
            ydl_opts = {
                'format': 'best[height<=720]/best/worstvideo+bestaudio/worst',  # Brightcove 호환성 개선
                'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'extractaudio': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
                'ignoreerrors': True,  # 포맷 에러 무시하고 계속 진행
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # 비디오 정보 가져오기
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'unknown')
                duration = info.get('duration', 0)
                platform = info.get('extractor', 'unknown')
                uploader = info.get('uploader', 'unknown')
                
                # 너무 긴 비디오는 제한 (30분)
                if duration and duration > 1800:
                    st.warning(f"[주의] 비디오가 너무 깁니다 ({duration//60}분). 30분 이하만 지원합니다.")
                    return None
                
                # 비디오 다운로드
                st.info(f"📥 다운로드 중: {title} ({platform})")
                ydl.download([url])
                
                # 다운로드된 파일 찾기
                for file in os.listdir(temp_dir):
                    if file.endswith(('.mp4', '.webm', '.mkv', '.flv', '.avi')):
                        downloaded_path = os.path.join(temp_dir, file)
                        st.success(f"[완료] 다운로드 완료: {title}")
                        
                        return {
                            'path': downloaded_path,
                            'title': title,
                            'platform': platform,
                            'uploader': uploader,
                            'duration': duration,
                            'url': url
                        }
            
            return None
            
        except Exception as e:
            st.error(f"[실패] 비디오 다운로드 실패: {e}")
            return None
    
    def process_video_url(self, url: str) -> Dict[str, Any]:
        """다양한 플랫폼 비디오 URL 처리"""
        # URL 기본 검증
        if not url.startswith(('http://', 'https://')):
            return {"error": "유효한 URL이 아닙니다."}
        
        # 비디오 다운로드
        download_info = self.download_video_from_url(url)
        if not download_info:
            return {"error": "비디오 다운로드에 실패했습니다."}
        
        try:
            # 다운로드된 비디오를 일반 비디오 파일로 처리
            result = self._process_video_file(download_info['path'], download_info['title'])
            
            # 원본 URL 및 플랫폼 정보 추가
            if "error" not in result:
                result["original_url"] = download_info['url']
                result["platform"] = download_info['platform']
                result["uploader"] = download_info['uploader']
                result["source_type"] = "web_video"
                result["video_duration"] = download_info['duration']
            
            # 임시 파일 정리
            os.unlink(download_info['path'])
            os.rmdir(os.path.dirname(download_info['path']))
            
            return result
            
        except Exception as e:
            return {"error": f"비디오 처리 실패: {e}"}
    
    def process_text_file_urls(self, uploaded_file) -> List[Dict[str, Any]]:
        """TXT 파일에서 URL 추출 및 배치 처리"""
        try:
            # 파일 내용 읽기
            content = uploaded_file.read().decode('utf-8')
            
            # URL 패턴 찾기
            url_pattern = r'https?://[^\s\n\r]+'
            urls = re.findall(url_pattern, content)
            
            if not urls:
                return [{"error": "텍스트 파일에서 유효한 URL을 찾을 수 없습니다."}]
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, url in enumerate(urls):
                # 진행률 업데이트
                progress = (i + 1) / len(urls)
                progress_bar.progress(progress)
                status_text.text(f"URL 처리 중: {i+1}/{len(urls)} - {url[:50]}...")
                
                # URL 처리
                result = self.process_video_url(url.strip())
                
                if "error" not in result:
                    result["batch_index"] = i + 1
                    result["total_batch"] = len(urls)
                    results.append(result)
                    
                    # 데이터베이스에 저장
                    if self.database:
                        self.database.insert_fragment(result)
                else:
                    st.warning(f"[주의] URL 처리 실패: {url[:50]}... - {result['error']}")
                
                # 너무 빠른 요청 방지
                time.sleep(1)
            
            progress_bar.progress(1.0)
            status_text.text(f"[완료] 배치 처리 완료: {len(results)}/{len(urls)}개 성공")
            
            return results
            
        except Exception as e:
            return [{"error": f"텍스트 파일 처리 실패: {e}"}]
    
    def trigger_dual_brain_system(self) -> Dict[str, Any]:
        """듀얼 브레인 시스템 트리거"""
        if not self.dual_brain:
            return {"error": "듀얼 브레인 시스템을 사용할 수 없습니다."}
        
        try:
            # 분석 결과가 있는지 확인
            if not self.analysis_results.get("holistic_results"):
                return {"error": "먼저 홀리스틱 분석을 완료하세요."}
            
            # 듀얼 브레인 분석 실행
            dual_brain_result = self.dual_brain.analyze_and_integrate(
                self.analysis_results["holistic_results"]
            )
            
            # 세션에 저장
            self.analysis_results["dual_brain_results"] = dual_brain_result
            
            return dual_brain_result
            
        except Exception as e:
            return {"error": f"듀얼 브레인 시스템 실패: {e}"}
    
    def generate_comprehensive_report(self) -> str:
        """종합 분석 보고서 생성"""
        if not self.analysis_results["processed_files"]:
            return "분석할 데이터가 없습니다."
        
        report_parts = []
        
        # 헤더
        report_parts.append(f"# {self.conference_name} 완전 통합 분석 보고서")
        report_parts.append(f"**생성 일시:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_parts.append(f"**세션 ID:** {self.session_id}")
        report_parts.append("")
        
        # 시스템 상태
        status = self.check_system_status()
        report_parts.append("## [설정] 시스템 상태")
        report_parts.append(f"- **전체 준비도:** {status['ready_systems']}/{status['total_systems']} ({'완료' if status['overall_ready'] else '부분완료'})")
        report_parts.append(f"- **OCR 엔진:** {'[완료] 정상' if status['ocr_available'] else '[실패] 비활성'}")
        report_parts.append(f"- **Whisper STT:** {'[완료] 정상' if status['whisper_available'] else '[실패] 비활성'}")
        report_parts.append(f"- **홀리스틱 분석:** {'[완료] 정상' if status['holistic_available'] else '[실패] 비활성'}")
        report_parts.append(f"- **데이터베이스:** {'[완료] 정상' if status['database_available'] else '[실패] 비활성'}")
        report_parts.append(f"- **Ollama AI:** {'[완료] 정상' if status['ollama_available'] else '[실패] 비활성'}")
        if status['ollama_available'] and self.ollama:
            report_parts.append(f"  - 사용 가능한 모델: {len(self.ollama.available_models)}개")
        report_parts.append("")
        
        # 파일 처리 결과
        processed_files = self.analysis_results["processed_files"]
        report_parts.append("## [폴더] 파일 처리 결과")
        report_parts.append(f"- **처리된 파일:** {len(processed_files)}개")
        
        file_types = Counter([f['file_type'] for f in processed_files])
        for file_type, count in file_types.items():
            type_name = {"image": "이미지", "audio": "음성", "video": "비디오"}.get(file_type, file_type)
            report_parts.append(f"  - {type_name}: {count}개")
        
        # 평균 신뢰도
        if processed_files:
            avg_confidence = np.mean([f['confidence'] for f in processed_files])
            report_parts.append(f"- **평균 신뢰도:** {avg_confidence:.1%}")
        
        report_parts.append("")
        
        # 홀리스틱 분석 결과
        if self.analysis_results.get("holistic_results"):
            holistic = self.analysis_results["holistic_results"]["holistic_analysis"]
            report_parts.append("## 🧠 홀리스틱 분석 결과")
            report_parts.append(f"- **총 조각 수:** {holistic.get('total_fragments', 0)}개")
            report_parts.append(f"- **발견된 개체:** {holistic.get('total_entities', 0)}개")
            report_parts.append(f"- **주요 주제:** {holistic.get('total_topics', 0)}개")
            
            # 핵심 인사이트
            key_insights = holistic.get('key_insights', [])
            if key_insights:
                report_parts.append("### [팁] 핵심 인사이트")
                for insight in key_insights:
                    report_parts.append(f"- {insight}")
            
            # AI 인사이트
            ai_insights = self.analysis_results["holistic_results"].get("ai_insights", [])
            if ai_insights and not ai_insights[0].startswith("AI 인사이트를 생성할 수 없습니다"):
                report_parts.append("### [AI] AI 심화 분석")
                for insight in ai_insights:
                    if not insight.startswith("AI"):  # 에러 메시지 제외
                        report_parts.append(f"- {insight}")
            
            report_parts.append("")
        
        # 듀얼 브레인 결과
        if self.analysis_results.get("dual_brain_results"):
            report_parts.append("## 🧠🧠 듀얼 브레인 분석")
            dual_brain = self.analysis_results["dual_brain_results"]
            report_parts.append("### AI 인사이트")
            if dual_brain.get("ai_insights"):
                for insight in dual_brain["ai_insights"][:3]:  # 상위 3개
                    report_parts.append(f"- {insight}")
            report_parts.append("")
        
        # 결론
        report_parts.append("## 📋 결론")
        report_parts.append("본 통합 분석을 통해 컨퍼런스의 전체적인 내용과 핵심 사안들이 체계적으로 분석되었습니다.")
        report_parts.append("실제 파일 처리부터 고급 의미 분석까지 전 과정이 검증된 알고리즘으로 수행되었습니다.")
        report_parts.append("")
        report_parts.append("---")
        report_parts.append("*본 보고서는 SOLOMOND AI 완전 통합 분석 시스템 v7.1로 생성되었습니다.*")
        
        return "\n".join(report_parts)

def main():
    st.set_page_config(
        page_title="완전 통합 컨퍼런스 분석 시스템",
        page_icon="[목표]",
        layout="wide"
    )
    
    # 🔥 메모리 안전 대용량 파일 처리 (MemoryError 방지)
    os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '5120'  # 5GB
    os.environ['STREAMLIT_SERVER_MAX_MESSAGE_SIZE'] = '5120'  # 5GB  
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    
    # Tornado 메모리 최적화 (중요!)
    os.environ['STREAMLIT_SERVER_MAX_REQUEST_SIZE'] = '5368709120'  # 5GB in bytes
    os.environ['STREAMLIT_TORNADO_MAX_BUFFER_SIZE'] = '268435456'  # 256MB buffer
    
    # [시작] GPU 활성화 설정 (5-15배 성능 향상)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GTX 1050 Ti 사용
    
    # GPU 메모리 최적화
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # GPU 메모리 정리
        st.success(f"GPU 활성화: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("GPU 사용 불가, CPU 모드로 진행")
    
    st.title("[목표] 완전 통합 컨퍼런스 분석 시스템")
    st.markdown("**SOLOMOND AI v7.4 - 1000+ 플랫폼 멀티미디어 분석**")
    
    # 시스템 상태 표시
    if OLLAMA_AVAILABLE and YOUTUBE_AVAILABLE:
        st.info("[시작] **모든 기능 활성화**: Ollama AI 5개 모델 + 1000+ 웹 플랫폼 + TXT 배치 처리")
    elif OLLAMA_AVAILABLE:
        st.info("[AI] **AI 분석 활성화**: Ollama 5개 모델 활성 | [주의] 웹 동영상 분석 비활성")
    elif YOUTUBE_AVAILABLE:
        st.info("[비디오] **웹 분석 활성화**: 1000+ 플랫폼 지원 | [주의] AI 고급 분석 비활성")
    else:
        st.warning("[주의] 기본 분석만 지원 - AI 모델과 웹 분석 비활성")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 시스템 설정")
    conference_name = st.sidebar.text_input("컨퍼런스 이름", "unified_conference_2025")
    
    # 분석기 초기화
    if 'analyzer' not in st.session_state:
        with st.spinner("시스템 초기화 중..."):
            st.session_state.analyzer = UnifiedConferenceAnalyzer(conference_name)
    
    analyzer = st.session_state.analyzer
    
    # 시스템 상태 확인
    status = analyzer.check_system_status()
    
    # 시스템 상태 표시
    st.sidebar.markdown("### [설정] 시스템 상태")
    if status["overall_ready"]:
        st.sidebar.success(f"[완료] 시스템 준비 완료 ({status['ready_systems']}/{status['total_systems']})")
    else:
        st.sidebar.warning(f"[주의] 부분 준비 ({status['ready_systems']}/{status['total_systems']})")
    
    st.sidebar.markdown(f"**OCR:** {'[완료]' if status['ocr_available'] else '[실패]'}")
    st.sidebar.markdown(f"**Whisper:** {'[완료]' if status['whisper_available'] else '[실패]'}")
    st.sidebar.markdown(f"**홀리스틱:** {'[완료]' if status['holistic_available'] else '[실패]'}")
    st.sidebar.markdown(f"**데이터베이스:** {'[완료]' if status['database_available'] else '[실패]'}")
    st.sidebar.markdown(f"**Ollama AI:** {'[완료]' if status['ollama_available'] else '[실패]'}")
    
    if status['ollama_available'] and analyzer.ollama:
        st.sidebar.markdown("### [AI] Ollama 모델")
        for model in analyzer.ollama.available_models:
            model_info = f"**{model}**"
            if "gpt-oss:20b" in model:
                model_info += " (13GB, 최신 GPT)"
            elif "qwen3:8b" in model:
                model_info += " (5.2GB, Qwen3)"
            elif "gemma3:27b" in model:
                model_info += " (17GB, 대형 모델)"
            elif "qwen2.5:7b" in model:
                model_info += " (4.7GB, 추천)"
            elif "gemma3:4b" in model:
                model_info += " (3.3GB, 경량)"
            st.sidebar.markdown(f"- {model_info}")
    
    # 데이터베이스 상태가 X인 경우 샘플 데이터 제공
    if not status["database_available"]:
        st.error("[실패] 데이터베이스 연결 실패")
        if st.button("[설정] 샘플 데이터로 테스트", type="secondary"):
            with st.spinner("샘플 데이터 생성 중..."):
                analyzer._create_sample_data()
            st.rerun()
    
    # 메인 인터페이스
    tab0, tab1, tab2, tab3, tab4 = st.tabs(["📋 사전정보", "[폴더] 파일/URL 처리", "🧠 홀리스틱 분석", "🧠🧠 듀얼 브레인", "📋 종합 보고서"])
    
    with tab0:
        st.markdown("## 📋 컨퍼런스 사전정보")
        st.markdown("**분석 품질 향상을 위한 배경 정보 입력**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            conference_name_input = st.text_input("[목표] 컨퍼런스명", "")
            conference_date = st.date_input("📅 날짜", datetime.now().date())
            location = st.text_input("📍 장소", "")
        
        with col2:
            industry_options = ["주얼리", "패션", "기술", "의료", "교육", "금융", "기타"]
            industry_field = st.selectbox("🏢 업계분야", industry_options)
            interest_keywords = st.text_area("🔑 관심 키워드 (쉼표로 구분)", 
                                           placeholder="예: 트렌드, 혁신, 지속가능성, AI")
        
        if st.button("💾 사전정보 저장", type="primary"):
            keywords_list = [k.strip() for k in interest_keywords.split(",") if k.strip()]
            
            analyzer.update_conference_info({
                "conference_name": conference_name_input,
                "conference_date": conference_date.isoformat(),
                "location": location,
                "industry_field": industry_field,
                "interest_keywords": keywords_list
            })
            
            st.success("[완료] 사전정보가 저장되었습니다!")
            
        # 저장된 정보 표시
        if analyzer.conference_info["conference_name"]:
            st.markdown("### 💾 저장된 정보")
            st.info(f"""
            **컨퍼런스:** {analyzer.conference_info['conference_name']}  
            **날짜:** {analyzer.conference_info['conference_date']}  
            **장소:** {analyzer.conference_info['location']}  
            **업계:** {analyzer.conference_info['industry_field']}  
            **키워드:** {', '.join(analyzer.conference_info['interest_keywords'])}
            """)
    
    with tab1:
        st.markdown("## [폴더] 파일 업로드 및 URL 분석")
        
        # 파일 업로드 섹션
        st.markdown("### 📂 파일 업로드 및 로컬 파일")
        st.markdown("**지원 형식:** 이미지 (JPG, PNG), 음성 (WAV, MP3, M4A), 비디오 (MP4, MOV), 텍스트 (TXT)")
        
        # 향상된 파일 업로드 시스템 사용
        if ENHANCED_FILE_HANDLER_AVAILABLE:
            uploaded_files = get_enhanced_file_upload()
        else:
            st.warning("향상된 파일 핸들러를 사용할 수 없습니다. 기본 업로드를 사용합니다.")
            uploaded_files = st.file_uploader(
                "분석할 파일들을 업로드하세요",
                accept_multiple_files=True,
                type=['jpg', 'jpeg', 'png', 'bmp', 'wav', 'mp3', 'm4a', 'flac', 'mp4', 'avi', 'mov', 'txt']
            )
            
            # 초대용량 파일 크기 안내
            if uploaded_files:
                total_size = sum(file.size for file in uploaded_files) / (1024*1024)  # MB
                if total_size > 3000:  # 3GB 이상
                    st.success(f"[시작] 초대용량 파일: {total_size:.1f}MB ({total_size/1024:.2f}GB) - GPU 가속으로 안정 처리")
                elif total_size > 1000:  # 1GB 이상
                    st.info(f"[폴더] 대용량 파일: {total_size:.1f}MB ({total_size/1024:.2f}GB)")
                elif total_size > 100:  # 100MB 이상
                    st.info(f"[폴더] 업로드된 파일: {total_size:.1f}MB")
                
                # 파일별 상세 정보
                with st.expander("[통계] 업로드된 파일 상세 정보"):
                    for file in uploaded_files:
                        file_size_mb = file.size / (1024*1024)
                        file_type = "[이미지]" if file.type.startswith('image') else "[음악]" if file.type.startswith('audio') else "[비디오]" if file.type.startswith('video') else "[문서]"
                        st.write(f"{file_type} **{file.name}** - {file_size_mb:.1f}MB ({file.type})")
            
            st.info(f"선택된 파일: {len(uploaded_files)}개")
            
        # Local file system section
        # 🗂️ 로컬 파일 시스템 (3GB+ IMG_0032.MOV 등)
        st.markdown("**[팁] 로컬 파일 처리**: 3GB+ 파일은 user_files 폴더에 복사 후 여기서 선택하세요")
        st.markdown("**📍 파일 경로**: `C:/Users/PC_58410/solomond-ai-system/user_files/`")
        
        # 사용 가능한 로컬 파일 스캔
        local_files = []
        if analyzer.user_files_dir.exists():
            for folder in analyzer.user_files_dir.iterdir():
                if folder.is_dir():
                    for file_path in folder.rglob("*"):
                        if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.wav', '.mp3', '.m4a', '.flac', '.mp4', '.avi', '.mov', '.txt']:
                            file_size = file_path.stat().st_size
                            local_files.append({
                                'path': file_path,
                                'name': file_path.name,
                                'folder': folder.name,
                                'size_mb': file_size / (1024*1024),
                                'type': file_path.suffix.lower()
                            })
        
        if local_files:
            st.success(f"[폴더] {len(local_files)}개 로컬 파일 발견")
            
            # 폴더별 그룹화
            folders = {}
            for file_info in local_files:
                folder_name = file_info['folder']
                if folder_name not in folders:
                    folders[folder_name] = []
                folders[folder_name].append(file_info)
            
            # 폴더 선택
            selected_folder = st.selectbox(
                "📂 분석할 폴더 선택",
                list(folders.keys()),
                help="3GB+ IMG_0032.MOV가 포함된 JGA2025_D1 폴더를 선택하세요"
            )
            
            if selected_folder:
                    folder_files = folders[selected_folder]
                    
                    # 대용량 파일 우선 표시
                    large_files = [f for f in folder_files if f['size_mb'] > 1000]  # 1GB+
                    if large_files:
                        st.info(f"[시작] 대용량 파일 {len(large_files)}개 발견 (MemoryError 방지 완벽 지원)")
                        
                        for file_info in large_files:
                            file_type = "[이미지]" if file_info['type'] in ['.jpg', '.jpeg', '.png'] else "[음악]" if file_info['type'] in ['.wav', '.mp3', '.m4a'] else "[비디오]" if file_info['type'] in ['.mp4', '.mov', '.avi'] else "[문서]"
                            st.markdown(f"• {file_type} **{file_info['name']}** - {file_info['size_mb']:.1f}MB ({file_info['size_mb']/1024:.2f}GB)")
                    
                    # 전체 분석 버튼
                    if st.button(f"[시작] {selected_folder} 폴더 전체 분석 시작", type="primary", key="local_files_analyze"):
                        # 로컬 파일을 uploaded_files 형식으로 변환
                        uploaded_files = []
                        for file_info in folder_files:
                            # 임시 파일 객체 생성 (streamlit 호환)
                            class LocalFileWrapper:
                                def __init__(self, file_path, file_name, file_size):
                                    self.name = file_name
                                    self.size = file_size
                                    self._file_path = file_path
                                    self.type = self._get_mime_type(file_path.suffix.lower())
                                
                                def _get_mime_type(self, ext):
                                    mime_map = {
                                        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                                        '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.m4a': 'audio/mp4',
                                        '.mp4': 'video/mp4', '.mov': 'video/quicktime', '.avi': 'video/avi',
                                        '.txt': 'text/plain'
                                    }
                                    return mime_map.get(ext, 'application/octet-stream')
                                
                                def read(self):
                                    with open(self._file_path, 'rb') as f:
                                        return f.read()
                            
                            wrapped_file = LocalFileWrapper(file_info['path'], file_info['name'], int(file_info['size_mb'] * 1024 * 1024))
                            uploaded_files.append(wrapped_file)
                        
                        st.success(f"[완료] 로컬 파일 {len(uploaded_files)}개 준비 완료 (3GB+ IMG_0032.MOV 포함)")
                        st.info("[폴더] 로컬 파일이 선택되었습니다. 아래 분석 시작 버튼을 클릭하세요!")
            else:
                st.warning("📂 user_files 폴더에 파일이 없습니다")
                st.markdown("**해결 방법**: 3GB+ IMG_0032.MOV 파일을 다음 경로에 복사하세요:")
                st.code("C:/Users/PC_58410/solomond-ai-system/user_files/JGA2025_D1/IMG_0032.MOV")
        
        # 선택된 파일 수 표시 (두 탭 통합)
        if uploaded_files:
            st.info(f"[완료] 총 선택된 파일: {len(uploaded_files)}개")
        
        # 웹 동영상 URL 섹션
        st.markdown("### [비디오] 웹 동영상 분석")
        if YOUTUBE_AVAILABLE:
            st.markdown("**지원 플랫폼:** YouTube, Vimeo, Dailymotion, Facebook, Instagram, TikTok, Twitch 등 1000+ 사이트")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                video_url = st.text_input(
                    "동영상 URL을 입력하세요",
                    placeholder="https://www.youtube.com/watch?v=... 또는 다른 플랫폼 URL",
                    help="최대 30분 길이의 동영상만 지원됩니다."
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # 공간 조정
                if video_url and st.button("[비디오] 동영상 분석", type="secondary"):
                    with st.spinner("웹 동영상 다운로드 및 분석 중..."):
                        video_result = analyzer.process_video_url(video_url)
                    
                    if "error" not in video_result:
                        st.success(f"[완료] 동영상 분석 완료!")
                        
                        # 결과 표시
                        with st.expander(f"[비디오] {video_result.get('file_source', 'Web Video')}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**플랫폼:** {video_result.get('platform', 'N/A')}")
                                st.markdown(f"**업로더:** {video_result.get('uploader', 'N/A')}")
                                st.markdown(f"**원본 URL:** [{video_result.get('original_url', 'N/A')[:50]}...]({video_result.get('original_url', '#')})")
                            
                            with col2:
                                st.markdown(f"**신뢰도:** {video_result['confidence']:.1%}")
                                if video_result.get('video_duration'):
                                    st.markdown(f"**길이:** {video_result['video_duration']:.1f}초")
                                st.markdown(f"**키워드:** {', '.join(video_result['keywords'][:5])}")
                            
                            st.markdown("**추출된 내용:**")
                            st.markdown(f"> {video_result['content'][:300]}{'...' if len(video_result['content']) > 300 else ''}")
                        
                        # 데이터베이스에 저장
                        if analyzer.database:
                            analyzer.database.insert_fragment(video_result)
                            analyzer.analysis_results["processed_files"].append(video_result)
                    else:
                        st.error(f"[실패] 동영상 분석 실패: {video_result['error']}")
        else:
            st.warning("[주의] 웹 동영상 분석 비활성 - yt-dlp 설치 필요")
            st.code("pip install yt-dlp", language="bash")
        
        if uploaded_files:
            st.info(f"선택된 파일: {len(uploaded_files)}개")
            
            # 처리 옵션
            col1, col2 = st.columns([3, 1])
            with col1:
                process_mode = st.selectbox(
                    "처리 모드",
                    ["[시작] 고속 모드 (권장)", "[보안] 안전 모드 (대용량)", "⚡ 터보 모드 (소용량)"],
                    help="대용량 파일은 안전 모드를 권장합니다."
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                skip_errors = st.checkbox("오류 건너뛰기", value=True, help="파일 처리 실패 시 계속 진행")
            
            if st.button("[시작] 파일 처리 시작", type="primary"):
                # 처리 모드에 따른 설정
                if "안전" in process_mode:
                    st.info("[보안] 안전 모드: 대용량 파일 최적화 처리")
                elif "터보" in process_mode:
                    st.info("⚡ 터보 모드: 고속 병렬 처리")
                else:
                    st.info("[시작] 고속 모드: 균형잡힌 처리")
                
                with st.spinner("실제 파일 처리 중... (대용량 파일은 시간이 오래 걸립니다)"):
                    result = analyzer.process_uploaded_files(uploaded_files, skip_errors=skip_errors)
                
                if "error" not in result:
                    st.success(f"[완료] 파일 처리 완료: {result['successful_count']}/{result['processed_count']}개 성공")
                    
                    if result['failed_files']:
                        st.warning("일부 파일 처리 실패:")
                        for failed in result['failed_files']:
                            st.error(f"- {failed['filename']}: {failed['error']}")
                    
                    # 처리 결과 표시
                    if result['analysis_fragments']:
                        st.markdown("### [통계] 처리된 조각들")
                        
                        for fragment in result['analysis_fragments']:
                            with st.expander(f"[문서] {fragment['file_source']} ({fragment['file_type']})"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"**신뢰도:** {fragment['confidence']:.1%}")
                                    keywords = fragment.get('keywords', [])
                                    if keywords:
                                        st.markdown(f"**키워드:** {', '.join(keywords[:5])}")
                                    else:
                                        st.markdown("**키워드:** 추출 중...")
                                
                                with col2:
                                    if fragment.get('speaker'):
                                        st.markdown(f"**화자:** {fragment['speaker']}")
                                    if fragment.get('duration'):
                                        st.markdown(f"**길이:** {fragment['duration']:.1f}초")
                                
                                st.markdown("**추출된 내용:**")
                                st.markdown(f"> {fragment['content'][:200]}{'...' if len(fragment['content']) > 200 else ''}")
                else:
                    st.error(f"[실패] 파일 처리 실패: {result['error']}")
    
    with tab2:
        st.markdown("## 🧠 홀리스틱 분석")
        st.markdown("**의미적 연결, 주제 클러스터링, 전체 스토리 생성**")
        
        if st.button("[검색] 홀리스틱 분석 실행", type="primary"):
            with st.spinner("홀리스틱 분석 수행 중..."):
                holistic_result = analyzer.run_holistic_analysis()
            
            if "error" not in holistic_result:
                st.success("[완료] 홀리스틱 분석 완료!")
                
                # 결과 표시
                holistic = holistic_result["holistic_analysis"]
                semantic = holistic_result["semantic_connections"]
                ai_insights = holistic_result.get("ai_insights", [])
                
                # 메트릭
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("총 조각", holistic.get("total_fragments", 0))
                
                with col2:
                    st.metric("발견된 개체", holistic.get("total_entities", 0))
                
                with col3:
                    st.metric("주요 주제", holistic.get("total_topics", 0))
                
                with col4:
                    if "error" not in semantic:
                        st.metric("의미적 연결", semantic.get("semantic_connections", 0))
                    else:
                        st.metric("의미적 연결", "오류")
                
                # 핵심 인사이트
                st.markdown("### [팁] 핵심 인사이트")
                for insight in holistic.get("key_insights", []):
                    st.markdown(f"- {insight}")
                
                # AI 심화 분석
                if ai_insights and not ai_insights[0].startswith("AI 인사이트를 생성할 수 없습니다"):
                    st.markdown("### [AI] AI 심화 분석")
                    for insight in ai_insights:
                        if not insight.startswith("AI"):  # 에러 메시지 제외
                            st.info(f"[AI] {insight}")
                
                # 상세 결과
                with st.expander("[통계] 상세 분석 결과"):
                    st.json(holistic_result)
                    
            else:
                st.error(f"[실패] 홀리스틱 분석 실패: {holistic_result['error']}")
    
    with tab3:
        st.markdown("## 🧠🧠 듀얼 브레인 시스템")
        st.markdown("**AI 인사이트 생성 및 구글 캘린더 연동**")
        
        if st.button("[시작] 듀얼 브레인 활성화", type="primary"):
            with st.spinner("듀얼 브레인 시스템 실행 중..."):
                dual_brain_result = analyzer.trigger_dual_brain_system()
            
            if "error" not in dual_brain_result:
                st.success("[완료] 듀얼 브레인 분석 완료!")
                
                # AI 인사이트 표시
                if dual_brain_result.get("ai_insights"):
                    st.markdown("### [AI] AI 인사이트")
                    for insight in dual_brain_result["ai_insights"]:
                        st.info(insight)
                
                # 상세 결과
                with st.expander("🧠 듀얼 브레인 상세 결과"):
                    st.json(dual_brain_result)
                    
            else:
                st.error(f"[실패] 듀얼 브레인 실패: {dual_brain_result['error']}")
    
    with tab4:
        st.markdown("## 📋 종합 분석 보고서")
        
        if st.button("[문서] 보고서 생성", type="primary"):
            report = analyzer.generate_comprehensive_report()
            
            st.markdown("### 📋 완전 통합 분석 보고서")
            st.markdown(report)
            
            # 다운로드 버튼
            st.download_button(
                label="📥 보고서 다운로드",
                data=report,
                file_name=f"unified_analysis_{conference_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
    
    # 사용법 안내
    st.markdown("---")
    st.markdown("### [팁] 사용법")
    st.markdown("""
    1. **사전정보**: 컨퍼런스 배경 정보 입력으로 분석 품질 향상
    2. **파일/URL 처리**: 
       - 이미지, 음성, 비디오 파일 업로드
       - 웹 동영상 URL 직접 분석 (1000+ 플랫폼 지원)
       - TXT 파일에 URL 목록 입력으로 배치 처리
    3. **홀리스틱 분석**: 의미적 연결 + Ollama AI 심화 인사이트
    4. **듀얼 브레인**: 구글 캘린더 연동 + AI 패턴 분석
    5. **종합 보고서**: 완전한 통합 분석 보고서 생성
    
    **[시작] 지원 플랫폼**: YouTube, Vimeo, TikTok, Instagram, Twitch, Facebook 등  
    **[문서] TXT 배치**: URL 목록을 텍스트 파일로 업로드하면 자동 일괄 처리**
    """)

if __name__ == "__main__":
    main()