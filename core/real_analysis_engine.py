#!/usr/bin/env python3
"""
실제 분석 엔진 - 가짜 분석을 실제 분석으로 교체
Whisper STT + EasyOCR + 무료 AI 모델 통합
"""

import os
import time
import logging
import re
from typing import Dict, List, Any, Optional, Union
from utils.logger import get_logger
from pathlib import Path
import json
from datetime import datetime

# 대용량 파일 스트리밍 최적화 시스템
try:
    from .large_file_streaming_optimizer import get_global_streaming_optimizer, StreamingConfig
    from .memory_aware_file_processor import get_global_file_processor, ProcessingConfig
    STREAMING_OPTIMIZER_AVAILABLE = True
except ImportError:
    STREAMING_OPTIMIZER_AVAILABLE = False

# AI 모델 메모리 관리 시스템
try:
    from .smart_ai_model_loader import get_global_smart_loader, smart_load_whisper, smart_load_easyocr, smart_load_transformers
    from .ai_model_memory_manager import get_global_ai_memory_manager, ModelPriority
    AI_MEMORY_MANAGEMENT_AVAILABLE = True
except ImportError:
    AI_MEMORY_MANAGEMENT_AVAILABLE = False

# 사용자 설정 관리 시스템
try:
    from .user_settings_manager import get_global_settings_manager, SettingType, SettingScope
    USER_SETTINGS_AVAILABLE = True
except ImportError:
    USER_SETTINGS_AVAILABLE = False

# GPU 메모리 관리 시스템 초기화
try:
    from .gpu_memory_manager import global_gpu_manager, ComputeMode
    GPU_MANAGER_AVAILABLE = True
    
    # 권장 모드로 설정
    recommended_mode = global_gpu_manager.get_recommended_mode(required_memory_mb=2000)  # 2GB 필요
    global_gpu_manager.switch_mode(recommended_mode)
    
except ImportError:
    GPU_MANAGER_AVAILABLE = False
    # 폴백: 기본 CPU 모드 설정
    from config.compute_config import force_cpu_mode
    force_cpu_mode()

# PyTorch 설정 최적화
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')

# Unicode 인코딩 문제 해결 (Windows)
os.environ['PYTHONIOENCODING'] = 'utf-8'
import sys
if sys.platform == 'win32':
    import codecs
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except (AttributeError, OSError):
        # Streamlit에서는 이미 인코딩이 설정되어 있을 수 있음
        pass

# 경고 메시지 억제
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', message='.*pin_memory.*')

# 실제 분석 라이브러리들
import whisper
import easyocr
# 향상된 OCR 핸들러 import
try:
    from .enhanced_ocr_handler import EnhancedOCRHandler
    ENHANCED_OCR_AVAILABLE = True
except ImportError:
    ENHANCED_OCR_AVAILABLE = False
import subprocess
import tempfile

# 고급 스토리텔링 엔진
try:
    from .advanced_storytelling_engine import global_storytelling_engine, create_comprehensive_korean_story
    storytelling_available = True
except ImportError:
    storytelling_available = False

try:
    import librosa
    import numpy as np
    librosa_available = True
except ImportError:
    librosa_available = False

try:
    from transformers import pipeline
    transformers_available = True
except ImportError:
    transformers_available = False

try:
    import google.generativeai as genai
    gemini_available = True
except ImportError:
    gemini_available = False

try:
    from .document_processor import document_processor
    document_processor_available = True
except ImportError:
    document_processor_available = False

try:
    from .youtube_processor import youtube_processor
    youtube_processor_available = True
except ImportError:
    youtube_processor_available = False

try:
    from .large_video_processor import large_video_processor
    large_video_processor_available = True
except ImportError:
    large_video_processor_available = False

try:
    from .error_recovery_analyzer import error_recovery_analyzer
    error_recovery_available = True
except ImportError:
    error_recovery_available = False

try:
    from .analysis_quality_enhancer import global_quality_enhancer, enhance_analysis_quality
    quality_enhancer_available = True
except ImportError:
    quality_enhancer_available = False

try:
    from .comprehensive_message_extractor import global_message_extractor, extract_speaker_message
    message_extractor_available = True
except ImportError:
    message_extractor_available = False

try:
    from .ppt_intelligence_engine import global_ppt_engine, analyze_ppt_slide
    ppt_intelligence_available = True
except ImportError:
    ppt_intelligence_available = False

try:
    from .jewelry_domain_enhancer import global_jewelry_enhancer, enhance_with_jewelry_domain
    jewelry_enhancer_available = True
except (ImportError, SyntaxError):
    jewelry_enhancer_available = False

try:
    from .audio_converter import global_audio_converter, convert_audio_to_wav, get_audio_info
    audio_converter_available = True
except ImportError:
    audio_converter_available = False

try:
    from .performance_monitor import global_performance_monitor, record_analysis_result
    performance_monitor_available = True
except ImportError:
    performance_monitor_available = False

try:
    from .comprehensive_message_extractor import ComprehensiveMessageExtractor
    
    # 전역 인스턴스 생성
    global_message_extractor = ComprehensiveMessageExtractor()
    
    def extract_comprehensive_messages(text: str, context: dict = None):
        """종합 메시지 추출 함수"""
        return global_message_extractor.extract_key_messages(text, context)
    
    message_extractor_available = True
except ImportError as e:
    print(f"종합 메시지 추출 엔진 로드 실패: {e}")
    message_extractor_available = False
    
    def extract_comprehensive_messages(text: str, context: dict = None):
        """폴백 함수"""
        return {"status": "unavailable", "error": "메시지 추출 엔진을 사용할 수 없습니다"}

# 실시간 화자 분리 시스템 (v2.3 추가)
try:
    from .realtime_speaker_diarization import global_speaker_diarization, analyze_speakers_realtime
    
    def analyze_speakers_in_audio(audio_file: str, transcript: str = "", progress_container=None):
        """실시간 화자 분리 분석 함수"""
        return global_speaker_diarization.analyze_speakers_in_audio(audio_file, transcript, progress_container)
    
    speaker_diarization_available = True
    print("🎤 실시간 화자 분리 시스템 로드 완료")
except ImportError as e:
    speaker_diarization_available = False
    print(f"실시간 화자 분리 시스템 로드 실패: {e}")
    
    def analyze_speakers_in_audio(audio_file: str, transcript: str = "", progress_container=None):
        """폴백 함수"""
        return {"status": "unavailable", "error": "화자 분리 시스템을 사용할 수 없습니다"}

class RealAnalysisEngine:
    """실제 파일 분석 엔진"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 분석 모델들 초기화
        self.whisper_model = None
        self.ocr_reader = None
        self.nlp_pipeline = None
        
        # 향상된 OCR 핸들러 초기화
        if ENHANCED_OCR_AVAILABLE:
            try:
                self.enhanced_ocr = EnhancedOCRHandler(logger=self.logger)
                self.logger.info("✅ 향상된 OCR 핸들러 초기화됨")
            except Exception as e:
                self.logger.warning(f"⚠️ 향상된 OCR 핸들러 초기화 실패: {e}")
                self.enhanced_ocr = None
        else:
            self.enhanced_ocr = None
        
        # 스트리밍 최적화 시스템 초기화
        if STREAMING_OPTIMIZER_AVAILABLE:
            try:
                # 기본 설정으로 파일 프로세서 초기화
                config = ProcessingConfig(
                    max_memory_usage_mb=512.0,
                    memory_optimization_level="balanced",
                    auto_gc_enabled=True
                )
                self.file_processor = get_global_file_processor(config)
                self.logger.info("✅ 대용량 파일 스트리밍 최적화 시스템 활성화")
            except Exception as e:
                self.logger.warning(f"⚠️ 스트리밍 최적화 시스템 초기화 실패: {e}")
                self.file_processor = None
        else:
            self.file_processor = None
        
        # AI 모델 메모리 관리 시스템 초기화
        if AI_MEMORY_MANAGEMENT_AVAILABLE:
            try:
                self.smart_loader = get_global_smart_loader()
                self.ai_memory_manager = get_global_ai_memory_manager(max_memory_mb=2048.0)
                self.logger.info("✅ AI 모델 메모리 관리 시스템 활성화")
                
                # 초기 메모리 상태 확인
                memory_status = self.smart_loader.get_memory_status()
                pressure_level = memory_status['memory_profile']['memory_pressure_level']
                if pressure_level in ['high', 'critical']:
                    self.logger.warning(f"⚠️ 초기 메모리 압박 감지: {pressure_level}")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ AI 모델 메모리 관리 시스템 초기화 실패: {e}")
                self.smart_loader = None
                self.ai_memory_manager = None
        else:
            self.smart_loader = None
            self.ai_memory_manager = None
        
        # 사용자 설정 관리 시스템 초기화
        if USER_SETTINGS_AVAILABLE:
            try:
                self.settings_manager = get_global_settings_manager()
                self.logger.info("✅ 사용자 설정 관리 시스템 활성화")
                
                # 분석 설정 로드
                self._load_analysis_settings()
                
            except Exception as e:
                self.logger.warning(f"⚠️ 사용자 설정 관리 시스템 초기화 실패: {e}")
                self.settings_manager = None
        else:
            self.settings_manager = None
        
        # 성능 추적
        self.analysis_stats = {
            "total_files": 0,
            "successful_analyses": 0,
            "partial_successes": 0,
            "failed_analyses": 0,
            "total_processing_time": 0,
            "last_analysis_time": None
        }
        
        self.logger.info("[INFO] 실제 분석 엔진 초기화 완료")
    
    def _load_analysis_settings(self) -> None:
        """분석 설정 로드"""
        if not self.settings_manager:
            return
        
        try:
            # AI 모델 설정 로드
            whisper_model = self.settings_manager.get_setting("ai.whisper_model_size", "base")
            easyocr_languages = self.settings_manager.get_setting("ai.easyocr_languages", ["ko", "en"])
            transformers_model = self.settings_manager.get_setting("ai.transformers_model", "facebook/bart-large-cnn")
            
            # 분석 설정 로드
            max_file_size = self.settings_manager.get_setting("analysis.max_file_size_mb", 500.0)
            batch_size = self.settings_manager.get_setting("analysis.batch_size", 5)
            enable_streaming = self.settings_manager.get_setting("analysis.enable_streaming", True)
            
            # 성능 설정 로드
            max_memory = self.settings_manager.get_setting("performance.max_memory_mb", 2048.0)
            enable_gpu = self.settings_manager.get_setting("performance.enable_gpu", False)
            
            # 설정 적용
            self.config = {
                'whisper_model_size': whisper_model,
                'easyocr_languages': easyocr_languages,
                'transformers_model': transformers_model,
                'max_file_size_mb': max_file_size,
                'batch_size': batch_size,
                'enable_streaming': enable_streaming,
                'max_memory_mb': max_memory,
                'enable_gpu': enable_gpu
            }
            
            # AI 모델 메모리 관리자 설정 업데이트
            if self.ai_memory_manager:
                self.ai_memory_manager.max_memory_mb = max_memory
            
            self.logger.info(f"📋 분석 설정 로드: Whisper={whisper_model}, 메모리={max_memory}MB, 스트리밍={enable_streaming}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 분석 설정 로드 실패: {e}")
            # 기본 설정 사용
            self.config = {
                'whisper_model_size': 'base',
                'easyocr_languages': ['ko', 'en'],
                'transformers_model': 'facebook/bart-large-cnn',
                'max_file_size_mb': 500.0,
                'batch_size': 5,
                'enable_streaming': True,
                'max_memory_mb': 2048.0,
                'enable_gpu': False
            }
    
    def _save_analysis_settings(self) -> None:
        """분석 설정 저장"""
        if not self.settings_manager or not hasattr(self, 'config'):
            return
        
        try:
            # 현재 설정을 사용자 설정으로 저장
            self.settings_manager.set_setting(
                "ai.whisper_model_size", 
                self.config.get('whisper_model_size', 'base'),
                SettingType.AI_MODEL,
                SettingScope.GLOBAL,
                "Whisper 모델 크기"
            )
            
            self.settings_manager.set_setting(
                "ai.easyocr_languages",
                self.config.get('easyocr_languages', ['ko', 'en']),
                SettingType.AI_MODEL,
                SettingScope.GLOBAL,
                "EasyOCR 지원 언어"
            )
            
            self.settings_manager.set_setting(
                "analysis.max_file_size_mb",
                self.config.get('max_file_size_mb', 500.0),
                SettingType.ANALYSIS,
                SettingScope.GLOBAL,
                "최대 파일 크기 (MB)"
            )
            
            self.settings_manager.set_setting(
                "analysis.enable_streaming",
                self.config.get('enable_streaming', True),
                SettingType.ANALYSIS,
                SettingScope.GLOBAL,
                "스트리밍 분석 활성화"
            )
            
            self.settings_manager.set_setting(
                "performance.max_memory_mb",
                self.config.get('max_memory_mb', 2048.0),
                SettingType.PERFORMANCE,
                SettingScope.GLOBAL,
                "최대 메모리 사용량 (MB)"
            )
            
            self.logger.debug("💾 분석 설정 저장 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 분석 설정 저장 실패: {e}")
    
    def update_analysis_config(self, **config_updates) -> bool:
        """분석 설정 업데이트"""
        try:
            if not hasattr(self, 'config'):
                self._load_analysis_settings()
            
            # 설정 업데이트
            for key, value in config_updates.items():
                if key in self.config:
                    self.config[key] = value
                    self.logger.info(f"⚙️ 설정 업데이트: {key} = {value}")
            
            # 설정 저장
            self._save_analysis_settings()
            
            # AI 모델 메모리 관리자 설정 업데이트
            if self.ai_memory_manager and 'max_memory_mb' in config_updates:
                self.ai_memory_manager.max_memory_mb = config_updates['max_memory_mb']
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 분석 설정 업데이트 실패: {e}")
            return False

    def _should_use_streaming(self, file_path: str) -> bool:
        """파일이 스트리밍 최적화를 사용해야 하는지 판단"""
        if not self.file_processor:
            return False
        
        # 사용자 설정 확인
        if hasattr(self, 'config') and not self.config.get('enable_streaming', True):
            return False
        
        try:
            file_path = Path(file_path)
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # 사용자 설정 최대 파일 크기 확인
            max_file_size = getattr(self, 'config', {}).get('max_file_size_mb', 500.0)
            if file_size_mb > max_file_size:
                self.logger.warning(f"⚠️ 파일 크기 제한 초과: {file_size_mb:.1f}MB > {max_file_size}MB")
                return False
            
            # 50MB 이상의 파일은 스트리밍 사용
            if file_size_mb >= 50:
                return True
            
            # 특정 파일 타입은 크기가 작아도 스트리밍 사용 (메모리 효율성)
            file_ext = file_path.suffix.lower()
            streaming_preferred_types = ['.mp4', '.avi', '.mov', '.mkv', '.wav', '.flac']
            
            if file_ext in streaming_preferred_types and file_size_mb >= 10:
                return True
            
            return False
        
        except Exception:
            return False
    
    def _enhance_with_context(self, extracted_text: str, context: Dict[str, Any] = None) -> str:
        """컨텍스트 정보를 활용한 텍스트 후처리"""
        if not context or not extracted_text:
            return extracted_text
        
        enhanced_text = extracted_text
        
        # 주제 키워드가 있으면 관련 용어 보정
        if context.get('topic_keywords'):
            # topic_keywords가 문자열인지 리스트인지 확인
            topic_keywords = context['topic_keywords']
            if isinstance(topic_keywords, str):
                keywords = [k.strip() for k in topic_keywords.split(',')]
            elif isinstance(topic_keywords, list):
                keywords = [str(k).strip() for k in topic_keywords]
            else:
                keywords = []
            
            for keyword in keywords:
                if keyword and len(keyword) > 2:
                    # 유사한 단어 찾아서 보정 (간단한 레벤슈타인 거리 기반)
                    import difflib
                    words = enhanced_text.split()
                    for i, word in enumerate(words):
                        if difflib.SequenceMatcher(None, word.lower(), keyword.lower()).ratio() > 0.7:
                            words[i] = keyword  # 정확한 키워드로 교체
                    enhanced_text = ' '.join(words)
        
        # 참석자/발표자 정보로 인명 보정
        if context.get('speakers') or context.get('participants'):
            names = []
            if context.get('speakers'):
                # speakers가 문자열인지 리스트인지 확인
                speakers = context['speakers']
                if isinstance(speakers, str):
                    names.extend([n.strip() for n in speakers.split(',')])
                elif isinstance(speakers, list):
                    names.extend([str(n).strip() for n in speakers])
            
            if context.get('participants'):
                # 괄호 안 내용 제거하고 이름만 추출
                import re
                participants = context['participants']
                if isinstance(participants, str):
                    participant_text = participants
                    participant_names = re.findall(r'([가-힣a-zA-Z\s]+?)(?:\s*\([^)]*\))?(?:,|$)', participant_text)
                    names.extend([n.strip() for n in participant_names if n.strip()])
                elif isinstance(participants, list):
                    names.extend([str(p).strip() for p in participants])
            
            # 인명 보정
            for name in names:
                if name and len(name) > 1:
                    import difflib
                    words = enhanced_text.split()
                    for i, word in enumerate(words):
                        if difflib.SequenceMatcher(None, word, name).ratio() > 0.6:
                            words[i] = name
                    enhanced_text = ' '.join(words)
        
        return enhanced_text
    
    def _generate_context_aware_summary(self, text: str, context: Dict[str, Any] = None) -> str:
        """컨텍스트를 고려한 요약 생성"""
        if not context:
            return self._generate_summary(text)
        
        # 기본 요약 생성
        base_summary = self._generate_summary(text)
        
        # 컨텍스트 정보 추가
        if context.get('event_context'):
            context_prefix = f"[{context['event_context']}] "
        else:
            context_prefix = ""
        
        if context.get('objective'):
            objective_suffix = f" (목적: {context['objective']})"
        else:
            objective_suffix = ""
        
        return f"{context_prefix}{base_summary}{objective_suffix}"
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정 (Unicode 인코딩 문제 해결)"""
        logger = get_logger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            # UTF-8 인코딩 강제 설정
            if hasattr(handler.stream, 'reconfigure'):
                handler.stream.reconfigure(encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _lazy_load_whisper(self, model_size: str = None) -> whisper.Whisper:
        """Whisper 모델 지연 로딩 (사용자 설정 기반)"""
        if self.whisper_model is None:
            # 사용자 설정에서 모델 크기 가져오기
            if model_size is None:
                model_size = getattr(self, 'config', {}).get('whisper_model_size', 'base')
            
            # 🧠 스마트 AI 모델 로더 사용
            if self.smart_loader:
                self.logger.info(f"🧠 스마트 Whisper {model_size} 모델 로딩... (메모리 최적화)")
                
                # GPU 설정 확인
                device = "gpu" if getattr(self, 'config', {}).get('enable_gpu', False) else "cpu"
                
                result = self.smart_loader.load_whisper_model(model_size, device=device)
                
                if result.success:
                    self.whisper_model = result.model
                    self.logger.info(f"✅ 스마트 Whisper {model_size} 로드 완료 ({result.load_time_seconds:.1f}초, {result.memory_usage_mb:.1f}MB)")
                else:
                    # 폴백: tiny 모델로 재시도
                    self.logger.warning(f"⚠️ {model_size} 모델 로드 실패: {result.error_message}")
                    fallback_result = self.smart_loader.load_whisper_model("tiny", device="cpu")
                    
                    if fallback_result.success:
                        self.whisper_model = fallback_result.model
                        self.logger.info(f"✅ 폴백 Whisper tiny 로드 완료 ({fallback_result.load_time_seconds:.1f}초)")
                    else:
                        raise RuntimeError(f"Whisper 모델을 로드할 수 없습니다: {fallback_result.error_message}")
            else:
                # 기존 방식 폴백
                self.logger.info(f"🎤 Whisper {model_size} 모델 로딩... (기존 방식)")
                start_time = time.time()
                
                try:
                    import whisper
                    self.whisper_model = whisper.load_model(model_size, device="cpu")
                    load_time = time.time() - start_time
                    self.logger.info(f"✅ Whisper {model_size} 로드 완료 ({load_time:.1f}초)")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ {model_size} 모델 로드 실패: {e}")
                    
                    # 폴백: tiny 모델로 재시도
                    try:
                        fallback_model = "tiny"
                        self.logger.info(f"🔄 폴백: Whisper {fallback_model} 모델로 재시도...")
                        self.whisper_model = whisper.load_model(fallback_model, device="cpu")
                        load_time = time.time() - start_time
                        self.logger.info(f"✅ Whisper {fallback_model} 로드 완료 ({load_time:.1f}초)")
                        
                    except Exception as e2:
                        self.logger.error(f"❌ Whisper 모델 로드 완전 실패: {e2}")
                        raise RuntimeError(f"Whisper 모델을 로드할 수 없습니다: {e2}")
                    
        return self.whisper_model
    
    def _lazy_load_ocr(self, languages: List[str] = None) -> easyocr.Reader:
        """EasyOCR 모델 지연 로딩 (사용자 설정 기반)"""
        if self.ocr_reader is None:
            # 사용자 설정에서 언어 가져오기
            if languages is None:
                languages = getattr(self, 'config', {}).get('easyocr_languages', ['ko', 'en'])
            
            # 🧠 스마트 AI 모델 로더 사용
            if self.smart_loader:
                lang_str = ', '.join(languages)
                self.logger.info(f"🧠 스마트 EasyOCR {lang_str} 모델 로딩... (메모리 최적화)")
                
                # GPU 설정 확인
                device = "gpu" if getattr(self, 'config', {}).get('enable_gpu', False) else "cpu"
                
                result = self.smart_loader.load_easyocr_model(languages, device=device)
                
                if result.success:
                    self.ocr_reader = result.model
                    self.logger.info(f"✅ 스마트 EasyOCR {lang_str} 로드 완료 ({result.load_time_seconds:.1f}초, {result.memory_usage_mb:.1f}MB)")
                else:
                    # 폴백: 기본 한/영 모델로 재시도
                    self.logger.warning(f"⚠️ {lang_str} 모델 로드 실패: {result.error_message}")
                    fallback_result = self.smart_loader.load_easyocr_model(['ko', 'en'], device="cpu")
                    
                    if fallback_result.success:
                        self.ocr_reader = fallback_result.model
                        self.logger.info(f"✅ 폴백 EasyOCR 한/영 로드 완료 ({fallback_result.load_time_seconds:.1f}초)")
                    else:
                        raise RuntimeError(f"EasyOCR 모델을 로드할 수 없습니다: {fallback_result.error_message}")
                    
            else:
                # 기존 방식 폴백
                self.logger.info("🖼️ EasyOCR 한/영 모델 로딩... (기존 방식)")
                start_time = time.time()
                
                # CPU 모드와 성능 최적화 설정
                import torch
                # PyTorch DataLoader pin_memory 경고 방지
                if not torch.cuda.is_available():
                    torch.backends.cudnn.enabled = False
                    # CPU 모드에서 스레드 수 최적화
                    torch.set_num_threads(2)  # CPU 코어에 맞게 조정
                
                # 메모리 정리 (기존 모델이 있는 경우)
                if hasattr(self, 'ocr_reader') and self.ocr_reader is not None:
                    del self.ocr_reader
                    import gc
                    gc.collect()
                
                import easyocr
                self.ocr_reader = easyocr.Reader(
                    ['ko', 'en'],
                    gpu=False,  # CPU 강제 사용
                    model_storage_directory=None,  # 기본 모델 디렉토리 사용
                    user_network_directory=None,
                    recog_network='CRNN',  # 기본 recognition network
                detector=True,
                recognizer=True,
                verbose=False,  # 로그 최소화
                download_enabled=True
            )
            
            load_time = time.time() - start_time
            self.logger.info(f"✅ EasyOCR 로드 완료 ({load_time:.1f}초)")
        return self.ocr_reader
    
    def _lazy_load_nlp(self, model_name: str = None, task: str = "summarization") -> Optional[any]:
        """NLP 파이프라인 지연 로딩 (사용자 설정 기반)"""
        if not transformers_available:
            return None
            
        if self.nlp_pipeline is None:
            # 사용자 설정에서 모델 이름 가져오기
            if model_name is None:
                model_name = getattr(self, 'config', {}).get('transformers_model', 'facebook/bart-base')
            
            # 🧠 스마트 AI 모델 로더 사용
            if self.smart_loader:
                self.logger.info(f"🧠 스마트 NLP {model_name} 모델 로딩... (메모리 최적화)")
                
                # GPU 설정 확인
                device = "gpu" if getattr(self, 'config', {}).get('enable_gpu', False) else "cpu"
                
                result = self.smart_loader.load_transformers_model(
                    model_name=model_name,
                    task=task,
                    device=device
                )
                
                if result.success:
                    self.nlp_pipeline = result.model
                    self.logger.info(f"✅ 스마트 NLP {model_name} 로드 완료 ({result.load_time_seconds:.1f}초, {result.memory_usage_mb:.1f}MB)")
                else:
                    # 폴백: 더 작은 모델로 재시도
                    self.logger.warning(f"⚠️ {model_name} 모델 로드 실패: {result.error_message}")
                    fallback_result = self.smart_loader.load_transformers_model(
                        model_name="facebook/bart-base",
                        task=task,
                        device="cpu"
                    )
                    
                    if fallback_result.success:
                        self.nlp_pipeline = fallback_result.model
                        self.logger.info(f"✅ 폴백 NLP 모델 로드 완료 ({fallback_result.load_time_seconds:.1f}초)")
                    else:
                        self.logger.warning(f"NLP 모델 로드 완전 실패: {fallback_result.error_message}")
                        return None
            else:
                # 기존 방식 폴백
                try:
                    self.logger.info("🧠 NLP 모델 로딩... (기존 방식)")
                    start_time = time.time()
                    
                    from transformers import pipeline
                    self.nlp_pipeline = pipeline("summarization", 
                                               model="facebook/bart-base",
                                               device=-1)  # CPU 사용
                    load_time = time.time() - start_time
                    self.logger.info(f"✅ NLP 로드 완료 ({load_time:.1f}초)")
                except Exception as e:
                    self.logger.warning(f"NLP 모델 로드 실패: {e}")
                    return None
                    
        return self.nlp_pipeline
    
    def _validate_whisper_language(self, language: str) -> Optional[str]:
        """Whisper 언어 설정 검증 및 변환"""
        if language == "auto":
            return None  # Whisper 자동 감지
        
        # Whisper에서 지원하는 주요 언어 코드
        whisper_languages = {
            "ko": "ko",  # 한국어
            "en": "en",  # 영어
            "ja": "ja",  # 일본어
            "zh": "zh",  # 중국어
            "es": "es",  # 스페인어
            "fr": "fr",  # 프랑스어
            "de": "de",  # 독일어
            "it": "it",  # 이탈리아어
            "pt": "pt",  # 포르투갈어
            "ru": "ru",  # 러시아어
            "ar": "ar",  # 아랍어
            "hi": "hi",  # 힌디어
        }
        
        # 언어 코드 정규화
        lang_code = language.lower().strip()
        
        if lang_code in whisper_languages:
            return whisper_languages[lang_code]
        else:
            self.logger.warning(f"⚠️ 지원하지 않는 언어 코드: {language}, 자동 감지로 대체")
            return None  # 지원하지 않는 언어는 자동 감지로 대체
    
    def _validate_audio_data(self, file_path: str) -> bool:
        """오디오 파일 데이터 검증"""
        try:
            if librosa_available:
                # librosa를 사용한 정확한 오디오 데이터 검증
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                
                # 오디오 데이터가 비어있는지 확인
                if len(audio_data) == 0:
                    self.logger.error("❌ 오디오 데이터가 비어있습니다")
                    return False
                
                # 오디오 길이 확인 (최소 0.1초)
                duration = len(audio_data) / sample_rate
                if duration < 0.1:
                    self.logger.warning(f"⚠️ 오디오가 너무 짧습니다: {duration:.2f}초")
                    return False
                
                # NaN 또는 무한대 값 확인
                if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                    self.logger.error("❌ 오디오 데이터에 유효하지 않은 값이 포함되어 있습니다")
                    return False
                
                self.logger.info(f"✅ 오디오 검증 성공: {duration:.2f}초, {sample_rate}Hz")
                return True
            else:
                # librosa가 없는 경우 기본 파일 검증
                self.logger.info("🔧 librosa 없음, 기본 파일 검증 사용")
                
                # 파일 존재 및 크기 확인
                if not os.path.exists(file_path):
                    self.logger.error("❌ 파일이 존재하지 않습니다")
                    return False
                
                file_size = os.path.getsize(file_path)
                if file_size < 1024:  # 1KB 미만
                    self.logger.error(f"❌ 파일이 너무 작습니다: {file_size} bytes")
                    return False
                
                # FFmpeg/ffprobe를 사용한 파일 정보 확인
                try:
                    # ffprobe가 있는지 먼저 확인
                    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', file_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        import json
                        info = json.loads(result.stdout)
                        duration = float(info.get('format', {}).get('duration', 0))
                        
                        if duration < 0.1:
                            self.logger.warning(f"⚠️ 오디오가 너무 짧습니다: {duration:.2f}초")
                            return False
                        
                        self.logger.info(f"✅ 기본 오디오 검증 성공: {duration:.2f}초")
                        return True
                    else:
                        raise Exception("ffprobe failed")
                        
                except Exception:
                    # ffprobe 실패시 ffmpeg으로 대체 시도
                    try:
                        cmd = ['ffmpeg', '-i', file_path, '-f', 'null', '-', '-v', 'quiet']
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                        
                        # ffmpeg가 성공적으로 파일을 읽을 수 있으면 유효한 파일로 간주
                        if result.returncode == 0:
                            self.logger.info("✅ ffmpeg 기본 검증 성공")
                            return True
                        else:
                            self.logger.warning("⚠️ ffmpeg 검증 실패, 파일 형식을 신뢰하고 진행")
                            return True
                            
                    except (subprocess.TimeoutExpired, Exception) as e:
                        self.logger.warning(f"⚠️ 기본 검증 실패: {e}, 파일 형식을 신뢰하고 진행")
                        return True
            
        except Exception as e:
            self.logger.error(f"❌ 오디오 검증 실패: {e}")
            return False
    
    def _convert_m4a_to_wav(self, m4a_path: str) -> str:
        """M4A 파일을 WAV로 변환 (강화된 처리)"""
        try:
            # 강화된 M4A 프로세서 사용
            from .enhanced_m4a_processor import global_m4a_processor
            
            self.logger.info("🔄 강화된 M4A → WAV 변환 시작...")
            
            # 심층 분석 후 변환
            result_path = global_m4a_processor.process_m4a_to_wav(m4a_path, target_sample_rate=16000)
            
            if result_path and os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                self.logger.info("✅ 강화된 M4A 변환 완료")
                return result_path
            else:
                self.logger.warning("⚠️ 강화된 M4A 변환 실패, 기본 FFmpeg 시도...")
                return self._fallback_m4a_conversion(m4a_path)
                
        except ImportError:
            self.logger.warning("⚠️ 강화된 M4A 프로세서 없음, 기본 FFmpeg 사용")
            return self._fallback_m4a_conversion(m4a_path)
        except Exception as e:
            self.logger.warning(f"⚠️ 강화된 M4A 변환 예외: {e}, 기본 방법 시도")
            return self._fallback_m4a_conversion(m4a_path)
    
    def _fallback_m4a_conversion(self, m4a_path: str) -> str:
        """M4A 변환 폴백 메서드"""
        try:
            # 임시 WAV 파일 생성
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            
            # FFmpeg 명령어로 M4A를 WAV로 변환
            cmd = [
                'ffmpeg', '-i', m4a_path,
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',          # 16kHz 샘플링 레이트
                '-ac', '1',              # 모노 채널
                '-y',                    # 덮어쓰기 허용
                temp_wav_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.logger.info("✅ 폴백 M4A 변환 완료")
                return temp_wav_path
            else:
                self.logger.error(f"❌ 폴백 FFmpeg 변환 실패: {result.stderr}")
                # 실패시 임시 파일 정리
                if os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error("❌ FFmpeg 변환 시간 초과")
            return None
        except Exception as e:
            self.logger.error(f"❌ M4A 변환 중 오류: {e}")
            return None
    
    def _preprocess_audio_file(self, file_path: str) -> str:
        """오디오 파일 전처리 (M4A 포함 모든 포맷 지원)"""
        file_ext = Path(file_path).suffix.lower()
        self.logger.info(f"🎵 오디오 파일 전처리 시작: {file_ext}")
        
        # 1. 오디오 파일 정보 확인
        if audio_converter_available:
            audio_info = get_audio_info(file_path)
            self.logger.info(f"📊 오디오 정보: {audio_info['duration_seconds']:.1f}초, "
                           f"{audio_info['file_size_mb']:.1f}MB, {audio_info['sample_rate']}Hz")
            
            # 유효하지 않은 오디오 파일이거나 M4A인 경우 변환
            if not audio_info['is_valid'] or file_ext in ['.m4a', '.aac']:
                self.logger.info("🔧 오디오 변환 시도...")
                converted_path = convert_audio_to_wav(file_path, target_sample_rate=16000)
                
                if converted_path and self._validate_audio_data(converted_path):
                    self.logger.info("✅ 오디오 변환 성공")
                    return converted_path
                else:
                    self.logger.warning("⚠️ 오디오 변환 실패, 원본 파일로 시도")
                    # 변환 실패시 원본으로 시도
                    if self._validate_audio_data(file_path):
                        return file_path
                    return None
            else:
                # 유효한 오디오 파일이면 검증 후 사용
                if self._validate_audio_data(file_path):
                    return file_path
                else:
                    # 검증 실패시 변환 시도
                    converted_path = convert_audio_to_wav(file_path)
                    return converted_path if converted_path and self._validate_audio_data(converted_path) else None
        
        # 오디오 컨버터 없으면 기존 M4A 변환 로직 사용
        else:
            if file_ext == ".m4a":
                if not self._validate_audio_data(file_path):
                    self.logger.info("🔧 FFmpeg 변환으로 재시도")
                    converted_path = self._convert_m4a_to_wav(file_path)
                    if converted_path and self._validate_audio_data(converted_path):
                        return converted_path
                    else:
                        if converted_path and os.path.exists(converted_path):
                            os.unlink(converted_path)
                        return None
            
            # 원본 파일이 정상이면 그대로 사용
            return file_path if self._validate_audio_data(file_path) else None
    
    def analyze_audio_file(self, file_path: str, language: str = "ko", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """실제 음성 파일 분석"""
        self.logger.info(f"🎤 실제 음성 분석 시작: {os.path.basename(file_path)}")
        
        start_time = time.time()
        processed_file_path = None
        temp_file_created = False
        
        try:
            # Whisper 모델 로드
            model = self._lazy_load_whisper()
            
            # 언어 설정 처리 - "auto"인 경우 None으로 변환하여 자동 감지 활성화
            whisper_language = self._validate_whisper_language(language)
            self.logger.info(f"🔤 언어 설정: {language} -> Whisper: {whisper_language}")
            
            # 파일 형식 확인 및 특별 처리
            file_ext = Path(file_path).suffix.lower()
            self.logger.info(f"📁 파일 형식: {file_ext}")
            
            # 오디오 파일 전처리 (모든 포맷 지원)
            processed_file_path = self._preprocess_audio_file(file_path)
            if processed_file_path is None:
                raise Exception("오디오 파일 전처리 실패: 오디오 데이터를 읽을 수 없습니다")
            
            # 변환된 파일인지 확인 (임시 파일 정리를 위해)
            temp_file_created = (processed_file_path != file_path)
            
            # 음성-텍스트 변환
            self.logger.info("🔄 음성-텍스트 변환 중...")
            transcribe_options = {
                "language": whisper_language,
                "fp16": False,  # 안정성을 위해 fp16 비활성화
                "verbose": False
            }
            
            # M4A 파일의 경우 추가 옵션 설정
            if file_ext == ".m4a":
                self.logger.info("🎵 M4A 파일 특별 처리 모드")
                transcribe_options.update({
                    "condition_on_previous_text": False,
                    "beam_size": 1,           # 안정성을 위해 빔 사이즈 축소
                    "best_of": 1,            # 최상 후보만 사용
                    "temperature": 0.0,      # 온도를 0으로 설정하여 일관성 향상
                    "compression_ratio_threshold": 2.4,  # 압축비 임계값 설정
                    "logprob_threshold": -1.0,           # 로그 확률 임계값 설정
                    "no_speech_threshold": 0.6           # 무음 감지 임계값 설정
                })
            
            # Whisper에 오디오 전달하기 전 마지막 안전 체크
            if not os.path.exists(processed_file_path):
                raise Exception(f"처리된 오디오 파일을 찾을 수 없습니다: {processed_file_path}")
            
            result = model.transcribe(processed_file_path, **transcribe_options)
            
            processing_time = time.time() - start_time
            
            # 결과 분석
            text = result["text"]
            segments = result["segments"]
            detected_language = result["language"]
            
            # 컨텍스트 기반 텍스트 향상
            enhanced_text = self._enhance_with_context(text, context)
            
            # 컨텍스트를 고려한 요약 생성
            summary = self._generate_context_aware_summary(enhanced_text, context)
            
            # 실시간 화자 분리 분석 (v2.3 향상된 기능)
            speaker_analysis = {}
            if speaker_diarization_available:
                try:
                    self.logger.info("🎤 실시간 화자 분리 분석 시작...")
                    speaker_analysis = analyze_speakers_in_audio(processed_file_path or file_path, text)
                    self.logger.info(f"✅ 화자 분리 완료: {speaker_analysis.get('speaker_count', 0)}명 감지")
                except Exception as e:
                    self.logger.warning(f"⚠️ 화자 분리 분석 실패: {e}")
                    speaker_analysis = {"status": "error", "error": str(e)}
            else:
                # 텍스트 기반 화자 식별 시스템 폴백
                if message_extractor_available:
                    try:
                        from .speaker_identification import analyze_speakers_in_text
                        speaker_analysis = analyze_speakers_in_text(text, segments)
                    except Exception as e:
                        speaker_analysis = {"status": "fallback_unavailable", "error": str(e)}
                else:
                    speaker_analysis = {"status": "unavailable", "message": "화자 분리 시스템을 사용할 수 없습니다"}
            
            # 핵심 발언 추출 (사용자 요구사항 반영)
            key_statements = self._extract_key_statements_from_text(enhanced_text, context)
            
            # 주얼리 키워드 분석 (향상된 텍스트 기반)
            jewelry_keywords = self._extract_jewelry_keywords(enhanced_text)
            
            # 종합 메시지 추출 (핵심 개선: "이 사람들이 무엇을 말하는지" 명확히)
            comprehensive_messages = {}
            if enhanced_text and len(enhanced_text.strip()) > 10:
                try:
                    self.logger.info("🧠 종합 메시지 추출 중...")
                    comprehensive_messages = extract_comprehensive_messages(enhanced_text, context)
                    self.logger.info("✅ 종합 메시지 추출 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ 메시지 추출 실패: {e}")
                    comprehensive_messages = {"status": "error", "error": str(e)}
            
            analysis_result = {
                "status": "success",
                "file_name": os.path.basename(file_path),
                "file_size_mb": float(round(os.path.getsize(file_path) / (1024 * 1024), 2)),
                "processing_time": float(round(processing_time, 1)),
                "detected_language": detected_language,
                "segments_count": int(len(segments)),
                "text_length": int(len(text)),
                "full_text": enhanced_text,
                "original_text": text,
                "summary": summary,
                "jewelry_keywords": jewelry_keywords,
                "segments": segments,
                "comprehensive_messages": comprehensive_messages,  # 핵심 추가
                "speaker_analysis": speaker_analysis,  # 화자 구분 추가
                "key_statements": key_statements,  # 핵심 발언 추가
                "analysis_type": "real_whisper_stt",
                "timestamp": datetime.now().isoformat()
            }
            
            # 🎯 MCP 자동 통합 시스템 적용 (모든 상황 대응)
            try:
                import asyncio
                from ..mcp_auto_integration_wrapper import enhance_result_with_mcp
                
                # 사용자 요청 컨텍스트 구성
                user_request = f"음성 파일 {os.path.basename(file_path)} 분석"
                mcp_context = {
                    "files": [{"name": os.path.basename(file_path), "size_mb": analysis_result["file_size_mb"]}],
                    "text_content": enhanced_text,
                    "analysis_type": "audio_analysis",
                    "detected_language": detected_language,
                    "processing_time": processing_time,
                    "context": context or {}
                }
                
                # MCP 자동 향상 적용 (비동기 함수를 동기 컨텍스트에서 실행)
                enhanced_analysis = asyncio.run(enhance_result_with_mcp(user_request, analysis_result, mcp_context))
                if enhanced_analysis != analysis_result:  # MCP 향상이 적용된 경우
                    analysis_result = enhanced_analysis
                    self.logger.info("✅ MCP 자동 통합으로 분석 품질 향상 완료")
                else:
                    self.logger.info("📋 MCP 향상 조건 미달, 기본 분석 결과 유지")
                
            except Exception as mcp_error:
                self.logger.warning(f"⚠️ MCP 자동 통합 중 오류 (분석은 정상 진행): {mcp_error}")
                # MCP 오류가 있어도 기본 분석 결과는 그대로 반환
            
            self._update_stats(processing_time, True)
            self.logger.info(f"✅ 음성 분석 완료 ({processing_time:.1f}초)")
            
            return analysis_result
            
        except Exception as e:
            # M4A 관련 특별 에러 처리
            error_str = str(e)
            if file_ext == ".m4a" and ("reshape" in error_str or "tensor" in error_str or "0 elements" in error_str):
                error_msg = f"M4A 파일 처리 실패: 오디오 데이터가 손상되었거나 빈 파일입니다. FFmpeg나 librosa 설치를 확인하세요. 원본 오류: {error_str}"
                self.logger.error("❌ M4A 텐서 리셰이프 오류 감지")
            else:
                error_msg = f"음성 분석 실패: {error_str}"
            
            self.logger.error(error_msg)
            
            # 에러 복구 분석 시도
            recovery_result = self._try_recovery_analysis(file_path, "audio", error_msg)
            if recovery_result:
                # 복구 성공시 부분 성공으로 처리
                recovery_result.update({
                    "file_name": os.path.basename(file_path),
                    "file_extension": file_ext,
                    "librosa_available": librosa_available,
                    "analysis_type": "real_whisper_stt_recovery",
                    "timestamp": datetime.now().isoformat(),
                    "file_size_mb": float(round(os.path.getsize(file_path) / (1024 * 1024), 2))
                })
                
                # 부분 성공으로 통계 업데이트
                self._update_stats(time.time() - start_time, True, partial=True)
                return recovery_result
            
            # 복구 실패시 원래 에러 반환
            self._update_stats(time.time() - start_time, False)
            
            return {
                "status": "error",
                "error": error_msg,
                "file_name": os.path.basename(file_path),
                "file_extension": file_ext,
                "librosa_available": librosa_available,
                "analysis_type": "real_whisper_stt",
                "timestamp": datetime.now().isoformat()
            }
        
        finally:
            # 임시 파일 정리
            if temp_file_created and processed_file_path and os.path.exists(processed_file_path):
                try:
                    os.unlink(processed_file_path)
                    self.logger.info("🗑️ 임시 변환 파일 정리 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ 임시 파일 정리 실패: {e}")
    
    def analyze_image_file(self, file_path: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """실제 이미지 파일 OCR 분석"""
        self.logger.info(f"🖼️ 실제 이미지 분석 시작: {os.path.basename(file_path)}")
        
        start_time = time.time()
        
        try:
            # 파일 크기 확인 및 전처리
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            self.logger.info(f"📁 파일 크기: {file_size_mb:.1f}MB")
            
            # 큰 파일의 경우 추가 최적화
            if file_size_mb > 5:  # 5MB 이상
                canvas_size = 960
                mag_ratio = 0.8
                text_threshold = 0.6
                self.logger.info("📏 대용량 파일 감지 - 추가 속도 최적화 적용")
            else:
                canvas_size = 1280
                mag_ratio = 1.0
                text_threshold = 0.5
            
            # 향상된 OCR 핸들러 사용 (에러 복구 및 대안 지원)
            if self.enhanced_ocr is not None:
                self.logger.info("🔄 향상된 OCR 핸들러로 이미지 분석...")
                ocr_result = self.enhanced_ocr.process_image(file_path)
                
                if ocr_result["success"]:
                    # 향상된 OCR 결과를 기존 형식으로 변환
                    results = []
                    for block in ocr_result["results"]:
                        results.append((block["bbox"], block["text"], block["confidence"]))
                    
                    self.logger.info(f"✅ 향상된 OCR 분석 완료 ({ocr_result['analysis_type']})")
                else:
                    # 향상된 OCR 실패시 기존 방식 사용
                    self.logger.warning("⚠️ 향상된 OCR 실패, 기존 방식으로 복구 시도")
                    reader = self._lazy_load_ocr()
                    results = reader.readtext(file_path)
            else:
                # 기존 OCR 방식 사용
                reader = self._lazy_load_ocr()
                
                self.logger.info("🔄 이미지 텍스트 추출 중... (속도 최적화 모드)")
                results = reader.readtext(
                    file_path,
                    width_ths=0.7,     # 텍스트 폭 임계값 (속도 향상)
                    height_ths=0.7,    # 텍스트 높이 임계값 (속도 향상)
                    paragraph=False,   # 단락 모드 비활성화 (속도 향상)
                    detail=1,          # 상세 정보 포함
                    batch_size=1,      # CPU 모드에서 배치 크기 최적화
                    workers=0,         # CPU 모드에서 멀티프로세싱 비활성화
                    text_threshold=text_threshold,   # 동적 임계값
                    low_text=0.4,      # 낮은 텍스트 신뢰도 임계값 (속도 향상)
                    link_threshold=0.4, # 링크 임계값 (속도 향상)
                    canvas_size=canvas_size,  # 동적 캔버스 크기
                    mag_ratio=mag_ratio      # 동적 확대 비율
                )
            
            # 메모리 정리
            import gc
            gc.collect()
            
            processing_time = time.time() - start_time
            
            # 성능 모니터링 기록
            if performance_monitor_available:
                record_analysis_result(
                    file_name=os.path.basename(file_path),
                    file_type="image",
                    processing_time=processing_time,
                    status="success",
                    additional_info={
                        "file_size_mb": file_size_mb,
                        "canvas_size": canvas_size,
                        "detected_blocks": len(results)
                    }
                )
            
            # 결과 처리
            detected_texts = []
            total_confidence = 0
            
            for bbox, text, confidence in results:
                detected_texts.append({
                    "text": text,
                    "confidence": float(round(confidence, 3)),  # NumPy float를 Python float로 변환
                    "bbox": bbox
                })
                total_confidence += float(confidence)  # NumPy float를 Python float로 변환
            
            avg_confidence = float(total_confidence / len(results)) if results else 0.0
            full_text = ' '.join([item["text"] for item in detected_texts])
            
            # 컨텍스트 기반 텍스트 향상
            enhanced_text = self._enhance_with_context(full_text, context)
            
            # 컨텍스트를 고려한 요약 생성
            summary = self._generate_context_aware_summary(enhanced_text, context)
            
            # 주얼리 키워드 분석 (향상된 텍스트 기반)
            jewelry_keywords = self._extract_jewelry_keywords(enhanced_text)
            
            # 종합 메시지 추출 (핵심 개선: "이 사람들이 무엇을 말하는지" 명확히)
            comprehensive_messages = {}
            if enhanced_text and len(enhanced_text.strip()) > 10:
                try:
                    self.logger.info("🧠 종합 메시지 추출 중...")
                    comprehensive_messages = extract_comprehensive_messages(enhanced_text, context)
                    self.logger.info("✅ 종합 메시지 추출 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ 메시지 추출 실패: {e}")
                    comprehensive_messages = {"status": "error", "error": str(e)}
            
            analysis_result = {
                "status": "success",
                "file_name": os.path.basename(file_path),
                "file_size_mb": float(round(os.path.getsize(file_path) / (1024 * 1024), 2)),
                "processing_time": float(round(processing_time, 1)),
                "blocks_detected": int(len(results)),
                "average_confidence": float(round(avg_confidence, 3)),
                "full_text": enhanced_text,
                "original_text": full_text,
                "summary": summary,
                "jewelry_keywords": jewelry_keywords,
                "detailed_results": detected_texts,
                "comprehensive_messages": comprehensive_messages,  # 핵심 추가
                "analysis_type": "real_easyocr",
                "timestamp": datetime.now().isoformat()
            }
            
            self._update_stats(processing_time, True)
            self.logger.info(f"✅ 이미지 분석 완료 ({processing_time:.1f}초)")
            
            return analysis_result
            
        except Exception as e:
            error_msg = f"이미지 분석 실패: {str(e)}"
            self.logger.error(error_msg)
            
            # 성능 모니터링 기록 (실패)
            if performance_monitor_available:
                record_analysis_result(
                    file_name=os.path.basename(file_path),
                    file_type="image",
                    processing_time=time.time() - start_time,
                    status="failed",
                    error_msg=error_msg,
                    additional_info={
                        "file_size_mb": float(round(os.path.getsize(file_path) / (1024 * 1024), 2)) if os.path.exists(file_path) else 0
                    }
                )
            
            # 에러 복구 분석 시도
            recovery_result = self._try_recovery_analysis(file_path, "image", error_msg)
            if recovery_result:
                # 복구 성공시 부분 성공으로 처리
                recovery_result.update({
                    "file_name": os.path.basename(file_path),
                    "analysis_type": "real_easyocr_recovery",
                    "timestamp": datetime.now().isoformat(),
                    "file_size_mb": float(round(os.path.getsize(file_path) / (1024 * 1024), 2))
                })
                
                # 부분 성공으로 통계 업데이트
                self._update_stats(time.time() - start_time, True, partial=True)
                
                # 성능 모니터링 기록 (부분 성공)
                if performance_monitor_available:
                    record_analysis_result(
                        file_name=os.path.basename(file_path),
                        file_type="image",
                        processing_time=time.time() - start_time,
                        status="partial",
                        error_msg=f"복구됨: {error_msg}",
                        additional_info={"recovery_used": True}
                    )
                return recovery_result
            
            # 복구 실패시 원래 에러 반환
            self._update_stats(time.time() - start_time, False)
            
            return {
                "status": "error", 
                "error": error_msg,
                "file_name": os.path.basename(file_path),
                "analysis_type": "real_easyocr",
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_summary(self, text: str) -> str:
        """텍스트 요약 생성"""
        if not text or len(text.strip()) < 50:
            return "텍스트가 너무 짧아 요약을 생성할 수 없습니다."
        
        # NLP 모델 사용 가능시
        nlp = self._lazy_load_nlp()
        if nlp and len(text) > 100:
            try:
                # 긴 텍스트는 자르기
                if len(text) > 1024:
                    text = text[:1024]
                
                summary_result = nlp(text, max_length=100, min_length=30, do_sample=False)
                return summary_result[0]['summary_text']
            except Exception as e:
                self.logger.debug(f"NLP 요약 실패: {e}")
        
        # 기본 요약 (첫 100자)
        return text[:100] + "..." if len(text) > 100 else text
    
    def _extract_key_statements_from_text(self, text: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """텍스트에서 핵심 발언 추출"""
        if not text or len(text.strip()) < 20:
            return []
        
        try:
            # 문장 단위로 분할
            sentences = re.split(r'[.!?]\s+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            key_statements = []
            
            for sentence in sentences:
                # 중요도 점수 계산
                importance_score = self._calculate_sentence_importance(sentence)
                
                if importance_score > 0.3:  # 임계값 이상인 발언만
                    key_statements.append({
                        "content": sentence,
                        "importance_score": importance_score,
                        "length": len(sentence),
                        "keywords": self._extract_keywords_from_sentence(sentence)
                    })
            
            # 중요도 순으로 정렬하고 상위 10개만 반환
            key_statements.sort(key=lambda x: x["importance_score"], reverse=True)
            return key_statements[:10]
            
        except Exception as e:
            self.logger.warning(f"핵심 발언 추출 실패: {e}")
            return []
    
    def _calculate_sentence_importance(self, sentence: str) -> float:
        """문장 중요도 계산"""
        score = 0.0
        
        # 주얼리 관련 키워드 보너스
        jewelry_keywords = ["다이아몬드", "금", "은", "반지", "목걸이", "가격", "품질", "디자인"]
        for keyword in jewelry_keywords:
            if keyword in sentence:
                score += 0.2
        
        # 감정 표현 보너스
        emotion_words = ["좋다", "훌륭하다", "만족", "추천", "최고", "완벽"]
        for word in emotion_words:
            if word in sentence:
                score += 0.1
        
        # 구체적 수치 보너스
        if re.search(r'\d+', sentence):
            score += 0.1
        
        # 문장 길이 고려 (너무 짧거나 긴 것 페널티)
        length = len(sentence)
        if 20 <= length <= 200:
            score += 0.1
        
        return min(score, 1.0)  # 최대 1.0으로 제한
    
    def _extract_keywords_from_sentence(self, sentence: str) -> List[str]:
        """문장에서 키워드 추출"""
        keywords = []
        
        # 주얼리 관련 키워드 추출
        jewelry_terms = ["다이아몬드", "금", "은", "백금", "루비", "사파이어", "에메랄드", 
                        "반지", "목걸이", "귀걸이", "팔찌", "브로치", "시계"]
        
        for term in jewelry_terms:
            if term in sentence:
                keywords.append(term)
        
        return keywords

    def _extract_jewelry_keywords(self, text: str) -> List[str]:
        """주얼리 관련 키워드 추출"""
        if not text:
            return []
        
        jewelry_terms = [
            # 영어 주얼리 용어
            "diamond", "gold", "silver", "platinum", "jewelry", "jewellery", 
            "ring", "necklace", "bracelet", "earring", "pendant", "gemstone",
            "ruby", "sapphire", "emerald", "pearl", "crystal", "luxury",
            "carat", "cut", "clarity", "color", "certificate", "GIA",
            
            # 한국어 주얼리 용어  
            "다이아몬드", "금", "은", "백금", "주얼리", "반지", "목걸이", 
            "팔찌", "귀걸이", "펜던트", "보석", "루비", "사파이어", 
            "에메랄드", "진주", "크리스탈", "럭셔리", "캐럿", "커팅",
            "투명도", "색상", "인증서", "지아"
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for term in jewelry_terms:
            if term.lower() in text_lower:
                found_keywords.append(term)
        
        return list(set(found_keywords))  # 중복 제거
    
    def analyze_document_file(self, file_path: str) -> Dict[str, Any]:
        """문서 파일 분석 (PDF, DOCX, DOC)"""
        start_time = time.time()
        file_name = os.path.basename(file_path)
        
        try:
            self.logger.info(f"[INFO] 문서 파일 분석 시작: {file_name}")
            
            if not document_processor_available:
                raise Exception("문서 처리 모듈을 사용할 수 없습니다. document_processor를 확인하세요.")
            
            # 문서 텍스트 추출
            doc_result = document_processor.process_document(file_path)
            
            if doc_result['status'] != 'success':
                if doc_result['status'] == 'partial_success':
                    self.logger.warning(f"[WARNING] 문서 부분 처리: {doc_result.get('warning', '')}")
                else:
                    raise Exception(doc_result.get('error', '문서 처리 실패'))
            
            extracted_text = doc_result['extracted_text']
            
            if not extracted_text or len(extracted_text.strip()) < 10:
                raise Exception("추출된 텍스트가 너무 짧거나 비어있습니다")
            
            # AI 요약 생성
            summary = self._generate_summary(extracted_text)
            
            # 주얼리 키워드 추출
            jewelry_keywords = self._extract_jewelry_keywords(extracted_text)
            
            # 텍스트 품질 평가
            quality_score = min(100, len(extracted_text) / 10)  # 간단한 품질 점수
            
            processing_time = time.time() - start_time
            
            result = {
                "status": "success",
                "file_name": file_name,
                "file_path": file_path,
                "file_type": doc_result.get('file_type', 'unknown'),
                "extracted_text": extracted_text,
                "text_length": len(extracted_text),
                "summary": summary,
                "jewelry_keywords": jewelry_keywords,
                "quality_score": round(quality_score, 1),
                "processing_time": round(processing_time, 2),
                "timestamp": datetime.now().isoformat(),
                "document_metadata": doc_result.get('metadata', {}),
                "document_info": {
                    "total_characters": doc_result.get('total_characters', 0),
                    "page_count": doc_result.get('page_count'),
                    "paragraph_count": doc_result.get('paragraph_count')
                }
            }
            
            # 통계 업데이트
            self._update_stats(processing_time, True)
            self.logger.info(f"[SUCCESS] 문서 분석 완료 ({processing_time:.1f}초)")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"문서 분석 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            
            # 에러 복구 분석 시도
            recovery_result = self._try_recovery_analysis(file_path, "document", error_msg)
            if recovery_result:
                # 복구 성공시 부분 성공으로 처리
                recovery_result.update({
                    "file_name": file_name,
                    "file_path": file_path,
                    "processing_time": round(processing_time, 2),
                    "timestamp": datetime.now().isoformat(),
                    "analysis_type": "document_recovery"
                })
                
                # 부분 성공으로 통계 업데이트
                self._update_stats(processing_time, True, partial=True)
                return recovery_result
            
            # 복구 실패시 원래 에러 반환
            self._update_stats(processing_time, False)
            
            return {
                "status": "error",
                "error": error_msg,
                "file_name": file_name,
                "file_path": file_path,
                "processing_time": round(processing_time, 2),
                "timestamp": datetime.now().isoformat()
            }
    
    def analyze_youtube_video(self, url: str, language: str = "ko") -> Dict[str, Any]:
        """YouTube 영상 분석"""
        start_time = time.time()
        
        try:
            self.logger.info(f"[INFO] YouTube 영상 분석 시작: {url}")
            
            if not youtube_processor_available:
                raise Exception("YouTube 처리 모듈을 사용할 수 없습니다. youtube_processor를 확인하세요.")
            
            if not youtube_processor.is_youtube_url(url):
                raise Exception("유효하지 않은 YouTube URL입니다.")
            
            # 1. 영상 정보 가져오기
            video_info = youtube_processor.get_video_info(url)
            if video_info['status'] != 'success':
                raise Exception(f"영상 정보 조회 실패: {video_info.get('error', 'Unknown')}")
            
            # 2. 오디오 다운로드
            self.logger.info("[INFO] YouTube 오디오 다운로드 중...")
            download_result = youtube_processor.download_audio(url)
            
            if download_result['status'] != 'success':
                raise Exception(f"오디오 다운로드 실패: {download_result.get('error', 'Unknown')}")
            
            audio_file = download_result['audio_file']
            
            # 3. 오디오 분석 수행
            self.logger.info("[INFO] 다운로드된 오디오 분석 중...")
            audio_analysis = self.analyze_audio_file(audio_file, language=language)
            
            if audio_analysis['status'] != 'success':
                # 오디오 분석 실패해도 기본 정보는 반환
                self.logger.warning(f"[WARNING] 오디오 분석 실패: {audio_analysis.get('error', 'Unknown')}")
            
            processing_time = time.time() - start_time
            
            # 결과 통합
            result = {
                "status": "success",
                "source_type": "youtube",
                "url": url,
                "video_info": video_info,
                "download_info": {
                    "audio_file": download_result['audio_file'],
                    "file_size_mb": download_result.get('file_size_mb', 0),
                    "download_time": download_result.get('processing_time', 0)
                },
                "audio_analysis": audio_analysis,
                "processing_time": round(processing_time, 2),
                "timestamp": datetime.now().isoformat()
            }
            
            # STT 결과가 있으면 YouTube 특화 분석 추가
            if audio_analysis.get('status') == 'success' and audio_analysis.get('transcription'):
                # YouTube 영상 + STT 결과 통합 분석
                combined_text = f"영상 제목: {video_info['title']}\n"
                combined_text += f"설명: {video_info.get('description', '')[:500]}...\n"
                combined_text += f"음성 내용: {audio_analysis['transcription']}"
                
                # 통합 요약 생성
                combined_summary = self._generate_summary(combined_text)
                combined_keywords = self._extract_jewelry_keywords(combined_text)
                
                result["combined_analysis"] = {
                    "integrated_summary": combined_summary,
                    "jewelry_keywords": combined_keywords,
                    "content_type": self._analyze_content_type(video_info, audio_analysis),
                    "engagement_metrics": {
                        "view_count": video_info.get('view_count', 0),
                        "like_count": video_info.get('like_count', 0),
                        "duration": video_info.get('duration_formatted', 'N/A')
                    }
                }
            
            # 통계 업데이트
            self._update_stats(processing_time, True)
            self.logger.info(f"[SUCCESS] YouTube 영상 분석 완료 ({processing_time:.1f}초)")
            
            # 임시 파일 정리 (선택적)
            try:
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
                    self.logger.info("[INFO] 임시 오디오 파일 정리됨")
            except:
                pass  # 정리 실패해도 무시
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"YouTube 영상 분석 오류: {str(e)}"
            
            # 통계 업데이트
            self._update_stats(processing_time, False)
            self.logger.error(f"[ERROR] {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg,
                "url": url,
                "processing_time": round(processing_time, 2),
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_content_type(self, video_info: Dict, audio_analysis: Dict) -> str:
        """콘텐츠 타입 분석"""
        title = video_info.get('title', '').lower()
        description = video_info.get('description', '').lower()
        transcription = audio_analysis.get('transcription', '').lower()
        
        # 주얼리 관련 콘텐츠 판별
        jewelry_indicators = [
            'jewelry', 'diamond', 'gold', 'silver', 'ring', 'necklace',
            '주얼리', '다이아몬드', '금', '은', '반지', '목걸이', '보석'
        ]
        
        combined_text = f"{title} {description} {transcription}"
        
        jewelry_score = sum(1 for indicator in jewelry_indicators if indicator in combined_text)
        
        if jewelry_score >= 3:
            return "jewelry_focused"
        elif jewelry_score >= 1:
            return "jewelry_related"
        else:
            return "general"
    
    def analyze_video_file(self, video_path: str, language: str = "ko") -> Dict[str, Any]:
        """비디오 파일 분석 (MOV, MP4, AVI 등)"""
        start_time = time.time()
        file_name = os.path.basename(video_path)
        
        try:
            self.logger.info(f"[INFO] 비디오 파일 분석 시작: {file_name}")
            
            if not large_video_processor_available:
                raise Exception("대용량 비디오 처리 모듈을 사용할 수 없습니다. large_video_processor를 확인하세요.")
            
            # 1. 향상된 비디오 정보 조회 (MoviePy 기능 포함)
            video_info = large_video_processor.get_enhanced_video_info_moviepy(video_path)
            if video_info['status'] not in ['success', 'partial_success']:
                raise Exception(f"비디오 정보 조회 실패: {video_info.get('error', 'Unknown')}")
            
            # 1.5. 키프레임 추출 및 OCR 분석 (MoviePy 사용 가능한 경우)
            keyframes_info = None
            visual_analysis = None
            if video_info.get('moviepy_duration') and video_info['file_size_mb'] <= 500:  # 500MB 이하만
                try:
                    self.logger.info("[INFO] 키프레임 추출 중...")
                    keyframes_result = large_video_processor.extract_keyframes_moviepy(video_path, num_frames=5)
                    if keyframes_result['status'] == 'success':
                        keyframes_info = keyframes_result
                        self.logger.info(f"[SUCCESS] {len(keyframes_result['keyframes'])}개 키프레임 추출 완료")
                        
                        # 🆕 키프레임별 OCR 분석 추가
                        self.logger.info("🔍 키프레임 OCR 분석 중...")
                        visual_analysis = self._analyze_keyframes_ocr(keyframes_result['keyframes'], context)
                        
                except Exception as e:
                    self.logger.warning(f"[WARNING] 키프레임 추출 실패: {e}")
                    keyframes_info = {"error": str(e)}
            
            # 2. 오디오 추출 (오디오 트랙이 있는 경우)
            audio_analysis = None
            audio_file = None
            
            if video_info.get('has_audio', False):
                self.logger.info("[INFO] 비디오에서 오디오 추출 중...")
                
                audio_extract_result = large_video_processor.extract_audio_from_video(video_path)
                
                if audio_extract_result['status'] == 'success':
                    audio_file = audio_extract_result['audio_file']
                    
                    # 3. 추출된 오디오 STT 분석
                    self.logger.info("[INFO] 추출된 오디오 STT 분석 중...")
                    audio_analysis = self.analyze_audio_file(audio_file, language=language)
                    
                    # 임시 오디오 파일 정리
                    try:
                        if os.path.exists(audio_file):
                            os.unlink(audio_file)
                            self.logger.info("[INFO] 임시 오디오 파일 정리됨")
                    except:
                        pass
                
                else:
                    self.logger.warning(f"[WARNING] 오디오 추출 실패: {audio_extract_result.get('error', 'Unknown')}")
                    audio_extract_result = None
            
            else:
                self.logger.info("[INFO] 비디오에 오디오 트랙이 없음 - STT 분석 생략")
                audio_extract_result = None
            
            processing_time = time.time() - start_time
            
            # 결과 통합
            result = {
                "status": "success",
                "file_name": file_name,
                "file_path": video_path,
                "file_type": "video",
                "video_info": video_info,
                "keyframes_info": keyframes_info,
                "visual_analysis": visual_analysis,  # 🆕 시각적 OCR 분석 결과
                "audio_extraction": audio_extract_result,
                "audio_analysis": audio_analysis,
                "processing_time": round(processing_time, 2),
                "timestamp": datetime.now().isoformat(),
                "enhanced_features": {
                    "moviepy_analysis": video_info.get('moviepy_duration') is not None,
                    "quality_analysis": video_info.get('video_quality_analysis') is not None,
                    "keyframe_extraction": keyframes_info is not None,
                    "visual_ocr_analysis": visual_analysis is not None,  # 🆕 시각 분석 여부
                    "frame_analysis": video_info.get('frame_analysis') is not None
                }
            }
            
            # 🚀 진정한 다각도 분석: 음성 + 시각 정보 통합
            audio_available = audio_analysis and audio_analysis.get('status') == 'success'
            visual_available = visual_analysis and visual_analysis.get('status') == 'success'
            
            if audio_available or visual_available:
                # 멀티모달 정보 통합
                combined_content = {
                    "metadata": f"비디오 파일: {file_name}",
                    "audio_content": "",
                    "visual_content": "",
                    "temporal_mapping": []
                }
                
                # 기본 메타데이터 추가
                if video_info.get('format_name'):
                    combined_content["metadata"] += f" | 형식: {video_info['format_name']}"
                if video_info.get('duration_formatted'):
                    combined_content["metadata"] += f" | 길이: {video_info['duration_formatted']}"
                
                # 음성 정보 추가
                if audio_available:
                    transcription = audio_analysis.get('transcription', '')
                    combined_content["audio_content"] = f"음성 내용: {transcription}"
                    
                # 🆕 시각 정보 추가 (새로 구현된 기능)
                if visual_available:
                    visual_text = visual_analysis.get('combined_visual_text', '')
                    combined_content["visual_content"] = f"화면 텍스트: {visual_text}"
                    
                    # 🆕 시간대별 매핑 (음성 + 시각 정보 동기화)
                    if visual_analysis.get('frame_details'):
                        for frame in visual_analysis['frame_details']:
                            if frame.get('enhanced_text', '').strip():
                                combined_content["temporal_mapping"].append({
                                    "timestamp": frame['timestamp_formatted'],
                                    "timestamp_seconds": frame['timestamp_seconds'], 
                                    "visual_info": frame['enhanced_text'],
                                    "confidence": frame.get('average_confidence', 0)
                                })
                
                # 🎯 멀티모달 통합 텍스트 생성
                integrated_text = f"{combined_content['metadata']}\n"
                if combined_content["audio_content"]:
                    integrated_text += f"{combined_content['audio_content']}\n"
                if combined_content["visual_content"]:
                    integrated_text += f"{combined_content['visual_content']}\n"
                
                # 🧠 통합 분석 수행
                integrated_summary = self._generate_context_aware_summary(integrated_text, context)
                integrated_keywords = self._extract_jewelry_keywords(integrated_text)
                
                result["integrated_analysis"] = {
                    "summary": integrated_summary,
                    "jewelry_keywords": integrated_keywords,
                    "multimodal_insights": {
                        "has_audio": audio_available,
                        "has_visual_text": visual_available and len(combined_content["visual_content"]) > 20,
                        "temporal_mappings": len(combined_content["temporal_mapping"]),
                        "analysis_depth": "multimodal" if (audio_available and visual_available) else "single_modal"
                    },
                    "content_analysis": {
                        "has_speech": audio_available,
                        "has_visual_info": visual_available,
                        "speech_duration": video_info.get('duration', 0),
                        "video_quality": video_info.get('video_info', {}).get('quality', 'Unknown'),
                        "file_size_category": self._categorize_file_size(video_info.get('file_size_mb', 0))
                    },
                    "temporal_synchronization": combined_content["temporal_mapping"][:10]  # 상위 10개 시간대
                }
            
            else:
                # 오디오 없거나 STT 실패 시 비디오 정보만으로 분석
                video_text = f"비디오 파일: {file_name} ({video_info.get('format_name', 'Unknown')})"
                
                result["integrated_analysis"] = {
                    "summary": f"비디오 파일 분석 완료. 길이: {video_info.get('duration_formatted', 'Unknown')}",
                    "jewelry_keywords": self._extract_jewelry_keywords(video_text),
                    "content_analysis": {
                        "has_speech": False,
                        "video_quality": video_info.get('video_info', {}).get('quality', 'Unknown'),
                        "file_size_category": self._categorize_file_size(video_info.get('file_size_mb', 0))
                    }
                }
            
            # 통계 업데이트
            self._update_stats(processing_time, True)
            self.logger.info(f"[SUCCESS] 비디오 분석 완료 ({processing_time:.1f}초)")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"비디오 분석 오류: {str(e)}"
            
            # 임시 파일 정리 (오류 시에도)
            if 'audio_file' in locals() and audio_file and os.path.exists(audio_file):
                try:
                    os.unlink(audio_file)
                except:
                    pass
            
            self.logger.error(f"[ERROR] {error_msg}")
            
            # 에러 복구 분석 시도
            recovery_result = self._try_recovery_analysis(video_path, "video", error_msg)
            if recovery_result:
                # 복구 성공시 부분 성공으로 처리
                recovery_result.update({
                    "file_name": file_name,
                    "file_path": video_path,
                    "processing_time": round(processing_time, 2),
                    "timestamp": datetime.now().isoformat(),
                    "analysis_type": "video_recovery"
                })
                
                # 부분 성공으로 통계 업데이트
                self._update_stats(processing_time, True, partial=True)
                return recovery_result
            
            # 복구 실패시 원래 에러 반환
            self._update_stats(processing_time, False)
            
            return {
                "status": "error",
                "error": error_msg,
                "file_name": file_name,
                "file_path": video_path,
                "processing_time": round(processing_time, 2),
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_keyframes_ocr(self, keyframes_list: List[Dict], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """키프레임별 OCR 분석 - 영상의 시각적 정보 추출"""
        start_time = time.time()
        
        try:
            # OCR 모델 로드 (이미지 분석과 동일)
            reader = self._lazy_load_ocr()
            
            frame_analyses = []
            all_extracted_texts = []
            total_confidence = 0
            
            for i, frame_info in enumerate(keyframes_list):
                frame_path = frame_info.get('frame_path')
                timestamp_seconds = frame_info.get('timestamp', i * 10)  # 기본값으로 10초 간격
                
                if not frame_path or not os.path.exists(frame_path):
                    continue
                    
                try:
                    self.logger.info(f"🖼️ 프레임 {i+1} OCR 분석: {timestamp_seconds}초")
                    
                    # OCR 수행 (프레임별 속도 최적화)
                    results = reader.readtext(
                        frame_path,
                        width_ths=0.8, height_ths=0.8, paragraph=False, detail=1,
                        batch_size=1, workers=0,  # 프레임 분석 고속화
                        text_threshold=0.6, low_text=0.5, link_threshold=0.5,
                        canvas_size=960, mag_ratio=1.0  # 더 작은 크기로 속도 향상
                    )
                    
                    # 결과 처리
                    detected_texts = []
                    frame_confidence = 0
                    
                    for bbox, text, confidence in results:
                        if confidence > 0.3:  # 최소 신뢰도 필터링
                            detected_texts.append({
                                "text": text.strip(),
                                "confidence": float(round(confidence, 3)),
                                "bbox": bbox
                            })
                            frame_confidence += float(confidence)
                    
                    avg_frame_confidence = float(frame_confidence / len(results)) if results else 0.0
                    frame_text = ' '.join([item["text"] for item in detected_texts])
                    
                    # 컨텍스트 기반 텍스트 향상 (참석자/키워드 보정)
                    enhanced_frame_text = self._enhance_with_context(frame_text, context) if frame_text.strip() else ""
                    
                    frame_analysis = {
                        "frame_index": i + 1,
                        "timestamp_seconds": timestamp_seconds,
                        "timestamp_formatted": self._format_timestamp(timestamp_seconds),
                        "frame_path": frame_path,
                        "texts_detected": len(detected_texts),
                        "average_confidence": avg_frame_confidence,
                        "raw_text": frame_text,
                        "enhanced_text": enhanced_frame_text,
                        "detected_elements": detected_texts[:5]  # 상위 5개만 저장
                    }
                    
                    frame_analyses.append(frame_analysis)
                    all_extracted_texts.append(enhanced_frame_text)
                    total_confidence += avg_frame_confidence
                    
                except Exception as frame_error:
                    self.logger.warning(f"❌ 프레임 {i+1} OCR 실패: {frame_error}")
                    continue
            
            processing_time = time.time() - start_time
            
            # 전체 텍스트 통합 및 분석
            combined_visual_text = ' '.join(filter(None, all_extracted_texts))
            visual_summary = self._generate_context_aware_summary(combined_visual_text, context) if combined_visual_text.strip() else "시각적 텍스트 정보 없음"
            visual_keywords = self._extract_jewelry_keywords(combined_visual_text) if combined_visual_text.strip() else []
            
            return {
                "status": "success",
                "processing_time": round(processing_time, 1),
                "frames_analyzed": len(frame_analyses),
                "total_texts_found": len(all_extracted_texts),
                "average_confidence": round(total_confidence / len(frame_analyses), 3) if frame_analyses else 0,
                "combined_visual_text": combined_visual_text,
                "visual_summary": visual_summary,
                "visual_keywords": visual_keywords,
                "frame_details": frame_analyses,
                "analysis_type": "keyframe_ocr"
            }
            
        except Exception as e:
            self.logger.error(f"❌ 키프레임 OCR 분석 실패: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": round(time.time() - start_time, 1)
            }
    
    def _format_timestamp(self, seconds: float) -> str:
        """초를 mm:ss 형식으로 변환"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _categorize_file_size(self, size_mb: float) -> str:
        """파일 크기 카테고리 분류"""
        if size_mb < 10:
            return "small"
        elif size_mb < 100:
            return "medium"
        elif size_mb < 1000:
            return "large"
        else:
            return "very_large"
    
    def _update_stats(self, processing_time: float, success: bool, partial: bool = False):
        """통계 업데이트"""
        self.analysis_stats["total_files"] += 1
        self.analysis_stats["total_processing_time"] += processing_time
        if success:
            if partial:
                self.analysis_stats["partial_successes"] += 1
            else:
                self.analysis_stats["successful_analyses"] += 1
        else:
            self.analysis_stats["failed_analyses"] += 1
        self.analysis_stats["last_analysis_time"] = datetime.now().isoformat()
    
    def _try_recovery_analysis(self, file_path: str, file_type: str, original_error: str) -> Dict[str, Any]:
        """실패한 분석에 대한 복구 시도"""
        if not error_recovery_available:
            return None
        
        try:
            self.logger.info(f"[RECOVERY] 복구 분석 시도: {os.path.basename(file_path)}")
            recovery_result = error_recovery_analyzer.recover_failed_analysis(file_path, file_type, original_error)
            
            if recovery_result.get("status") == "partial_success":
                self.logger.info(f"[RECOVERY] 부분 복구 성공: {recovery_result.get('recovery_method', 'unknown')}")
                return recovery_result
            else:
                self.logger.warning(f"[RECOVERY] 복구 실패: {recovery_result.get('recovery_error', 'unknown')}")
                return None
                
        except Exception as e:
            self.logger.error(f"[RECOVERY] 복구 분석 중 오류: {e}")
            return None
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """분석 통계 반환"""
        total_files = self.analysis_stats["total_files"]
        if total_files == 0:
            return self.analysis_stats
        
        stats = self.analysis_stats.copy()
        
        # 전체 성공률 (완전 성공 + 부분 성공)
        total_successes = stats["successful_analyses"] + stats["partial_successes"]
        stats["overall_success_rate"] = round((total_successes / total_files) * 100, 1)
        
        # 완전 성공률만
        stats["full_success_rate"] = round((stats["successful_analyses"] / total_files) * 100, 1)
        
        # 부분 성공률
        stats["partial_success_rate"] = round((stats["partial_successes"] / total_files) * 100, 1)
        
        # 실패율
        stats["failure_rate"] = round((stats["failed_analyses"] / total_files) * 100, 1)
        
        # 평균 처리 시간
        stats["average_processing_time"] = round(stats["total_processing_time"] / total_files, 1)
        
        # 복구 효과 (부분 성공이 있으면 복구 시스템이 작동했다는 의미)
        stats["recovery_effectiveness"] = stats["partial_successes"] > 0
        
        return stats

# 전역 분석 엔진 인스턴스
global_analysis_engine = RealAnalysisEngine()

def analyze_file_real(file_path: str, file_type: str, language: str = "auto", context: Dict[str, Any] = None, timeout_minutes: int = 5) -> Dict[str, Any]:
    """파일 실제 분석 (메모리 누수 및 무한루프 방지)"""
    import signal
    import gc
    import time
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"분석이 {timeout_minutes}분 내에 완료되지 않아 중단됨")
    
    start_time = time.time()
    analysis_start_memory = None
    
    try:
        # 메모리 사용량 확인 (Linux/Mac만)
        try:
            import psutil
            process = psutil.Process()
            analysis_start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        except:
            analysis_start_memory = 0
        
        # 타임아웃 설정 (Unix 시스템만)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_minutes * 60)
        
        # 기본 분석 수행
        if file_type == "audio":
            result = global_analysis_engine.analyze_audio_file(file_path, language=language, context=context)
        elif file_type == "image":
            result = global_analysis_engine.analyze_image_file(file_path, context=context)
        elif file_type == "document":
            result = global_analysis_engine.analyze_document_file(file_path)
        elif file_type == "youtube":
            result = global_analysis_engine.analyze_youtube_video(file_path, language=language)
        elif file_type == "video":
            result = global_analysis_engine.analyze_video_file(file_path, language=language)
        else:
            return {
                "status": "error",
                "error": f"지원하지 않는 파일 타입: {file_type}",
                "file_name": os.path.basename(file_path) if os.path.exists(file_path) else file_path,
                "timestamp": datetime.now().isoformat()
            }
        
        # 처리 시간 및 메모리 사용량 기록
        processing_time = time.time() - start_time
        
        try:
            analysis_end_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_delta = analysis_end_memory - analysis_start_memory
        except:
            memory_delta = 0
        
        # 메모리 사용량이 과도한 경우 즉시 중단 (메모리 누수 방지)
        if memory_delta > 2000:  # 2GB 이상 증가시 즉시 중단
            get_logger(__name__).error(f"심각한 메모리 누수 감지: {memory_delta:.1f}MB 증가 - 분석 중단")
            gc.collect()
            # 메모리 누수로 인한 강제 종료
            raise MemoryError(f"메모리 사용량이 {memory_delta:.1f}MB로 증가하여 분석을 중단합니다")
        elif memory_delta > 500:  # 500MB 이상 증가시 경고
            get_logger(__name__).warning(f"높은 메모리 사용 감지: {memory_delta:.1f}MB 증가")
            gc.collect()  # 가비지 컬렉션 강제 실행
        
        # 처리 시간이 과도한 경우 경고
        if processing_time > 300:  # 5분 이상
            get_logger(__name__).warning(f"긴 처리 시간 감지: {processing_time:.1f}초")
        
        # 메타데이터 추가
        if result and isinstance(result, dict):
            result['processing_time_seconds'] = processing_time
            result['memory_usage_mb'] = memory_delta
            result['analysis_timestamp'] = datetime.now().isoformat()
            
    except TimeoutError as e:
        return {
            "status": "timeout",
            "error": str(e),
            "file_name": os.path.basename(file_path) if os.path.exists(file_path) else file_path,
            "processing_time_seconds": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": f"분석 중 오류 발생: {str(e)}",
            "file_name": os.path.basename(file_path) if os.path.exists(file_path) else file_path,
            "processing_time_seconds": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
    finally:
        # 타임아웃 해제
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        # 메모리 정리
        gc.collect()
    
    # 🚀 종합 분석 엔진 적용 (클로바 노트 + ChatGPT 수준)
    if result.get('status') == 'success':
        try:
            # 1. 품질 향상 엔진 적용
            if quality_enhancer_available:
                result = enhance_analysis_quality(result, context)
                result['quality_enhancement_applied'] = True
            
            # 2. PPT 지능형 분석 (이미지 파일인 경우)
            if file_type == "image" and ppt_intelligence_available:
                ppt_analysis = analyze_ppt_slide(file_path, context)
                if ppt_analysis.get('status') == 'success':
                    result['ppt_intelligence'] = ppt_analysis['ppt_intelligence']
                    result['ppt_enhanced_understanding'] = ppt_analysis['enhanced_understanding']
            
            # 3. 종합 메시지 추출 (다중 모달 데이터가 있는 경우)
            if message_extractor_available:
                # 다중 모달 데이터 준비
                multimodal_data = {}
                if file_type == "audio":
                    multimodal_data['audio_analysis'] = result
                elif file_type == "image":
                    multimodal_data['image_analysis'] = [result]
                elif file_type == "video":
                    multimodal_data['video_analysis'] = result
                
                # 종합 메시지 추출 (안전 래퍼)
                if multimodal_data:
                    try:
                        # 메모리 및 시간 제한 설정
                        import signal
                        def timeout_message_extraction(signum, frame):
                            raise TimeoutError("메시지 추출 시간 초과")
                        
                        # 30초 제한
                        if hasattr(signal, 'SIGALRM'):
                            signal.signal(signal.SIGALRM, timeout_message_extraction)
                            signal.alarm(30)
                        
                        message_analysis = extract_speaker_message(multimodal_data, context)
                        
                        if hasattr(signal, 'SIGALRM'):
                            signal.alarm(0)  # 타임아웃 해제
                        
                        if message_analysis.get('status') == 'success':
                            result['comprehensive_message'] = message_analysis['comprehensive_analysis']
                            if message_analysis['comprehensive_analysis'].get('main_summary'):
                                result['clova_style_summary'] = message_analysis['comprehensive_analysis']['main_summary']
                        else:
                            result['comprehensive_message'] = {'status': 'extraction_failed', 'error': message_analysis.get('error', 'Unknown error')}
                            
                    except (TimeoutError, Exception) as e:
                        result['comprehensive_message'] = {'status': 'extraction_timeout', 'error': f'메시지 추출 실패: {str(e)}'}
            
            # 4. 주얼리 도메인 특화 분석 (해당하는 경우)
            if jewelry_enhancer_available and result.get('enhanced_text'):
                result = enhance_with_jewelry_domain(result, result['enhanced_text'])
            
            # 종합 분석 완료 마킹
            result['comprehensive_analysis_applied'] = True
            result['analysis_engines_used'] = {
                'quality_enhancer': quality_enhancer_available,
                'message_extractor': message_extractor_available,
                'ppt_intelligence': ppt_intelligence_available and file_type == "image",
                'jewelry_domain': jewelry_enhancer_available
            }
            
        except Exception as e:
            # 종합 분석 실패시 원본 결과 반환 (로그 기록)
            get_logger(__name__).warning(f"종합 분석 실패: {e}")
            result['comprehensive_analysis_error'] = str(e)
            result['comprehensive_analysis_applied'] = False
    
    return result

if __name__ == "__main__":
    # 테스트 실행
    print("🚀 실제 분석 엔진 테스트")
    print("=" * 50)
    
    engine = RealAnalysisEngine()
    
    # 테스트 파일들
    test_files = [
        ("/mnt/c/Users/PC_58410/Desktop/근혁/세미나/202506홍콩쇼/D1/새로운 녹음 2.m4a", "audio"),
        ("/mnt/c/Users/PC_58410/Desktop/근혁/세미나/202506홍콩쇼/D1/IMG_2160.JPG", "image")
    ]
    
    for file_path, file_type in test_files:
        if os.path.exists(file_path):
            print(f"\n🧪 테스트: {os.path.basename(file_path)}")
            result = analyze_file_real(file_path, file_type)
            print(f"결과: {result.get('status', 'unknown')}")
            if result.get('status') == 'success':
                print(f"처리시간: {result.get('processing_time', 0)}초")
                if 'full_text' in result:
                    text = result['full_text']
                    print(f"추출 텍스트: {text[:100]}{'...' if len(text) > 100 else ''}")
        else:
            print(f"⚠️ 파일 없음: {file_path}")
    
    # 통계 출력
    print(f"\n📊 분석 통계:")
    stats = engine.get_analysis_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ 실제 분석 엔진 테스트 완료!")

# 고급 스토리텔링 함수 추가
def create_comprehensive_story_from_sources(sources: List[Dict[str, Any]], 
                                           story_type: str = "general") -> Dict[str, Any]:
    """
    다중 소스 분석 결과를 하나의 일관된 한국어 스토리로 변환
    사용자의 궁극적인 목표: "어떤 이야기를 하고 있는지" 파악
    
    Args:
        sources: 분석된 소스들의 리스트
        story_type: 스토리 유형 (general, consultation, meeting, multimedia)
    
    Returns:
        생성된 한국어 스토리
    """
    
    if not storytelling_available:
        return {
            "status": "error",
            "error": "고급 스토리텔링 엔진이 사용할 수 없습니다",
            "fallback": "OpenAI API Key 또는 Anthropic API Key를 설정하면 더 나은 스토리 생성이 가능합니다"
        }
    
    try:
        # 분석 결과 구조화
        analysis_results = {
            "sources": sources,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_sources": len(sources)
            }
        }
        
        # 고급 스토리텔링 엔진 활용
        story_result = create_comprehensive_korean_story(analysis_results, story_type)
        
        return story_result
        
    except Exception as e:
        return {
            "status": "error", 
            "error": f"스토리 생성 중 오류: {str(e)}",
            "fallback_story": _create_simple_fallback_story(sources)
        }

def _create_simple_fallback_story(sources: List[Dict[str, Any]]) -> str:
    """간단한 폴백 스토리 생성"""
    
    story_parts = []
    story_parts.append("## 분석 결과 종합")
    story_parts.append("")
    
    audio_count = sum(1 for s in sources if s.get("type") == "audio")
    image_count = sum(1 for s in sources if s.get("type") == "image") 
    document_count = sum(1 for s in sources if s.get("type") == "document")
    
    story_parts.append(f"총 {len(sources)}개의 파일이 분석되었습니다:")
    if audio_count > 0:
        story_parts.append(f"- 음성/동영상 파일: {audio_count}개")
    if image_count > 0:
        story_parts.append(f"- 이미지 파일: {image_count}개")  
    if document_count > 0:
        story_parts.append(f"- 문서 파일: {document_count}개")
    
    story_parts.append("")
    story_parts.append("### 주요 내용")
    
    for i, source in enumerate(sources[:5], 1):  # 최대 5개까지
        analysis = source.get("analysis_result", {})
        if analysis.get("status") == "success":
            text = analysis.get("transcription") or analysis.get("text") or analysis.get("full_text", "")
            if text:
                story_parts.append(f"{i}. {source.get('name', 'Unknown')}: {text[:150]}...")
    
    story_parts.append("")
    story_parts.append("**참고**: 더 상세하고 일관된 스토리 생성을 위해 OpenAI API Key를 설정하시기 바랍니다.")
    
    return "\n".join(story_parts)