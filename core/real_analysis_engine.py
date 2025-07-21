#!/usr/bin/env python3
"""
실제 분석 엔진 - 가짜 분석을 실제 분석으로 교체
Whisper STT + EasyOCR + 무료 AI 모델 통합
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime

# GPU 메모리 문제 해결을 위한 CPU 모드 강제 설정
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# 실제 분석 라이브러리들
import whisper
import easyocr
import subprocess
import tempfile

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

class RealAnalysisEngine:
    """실제 파일 분석 엔진"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 분석 모델들 초기화
        self.whisper_model = None
        self.ocr_reader = None
        self.nlp_pipeline = None
        
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
        self.logger.info(f"[INFO] 에러 복구 분석기: {'활성화' if error_recovery_available else '비활성화'}")
    
    def _enhance_with_context(self, extracted_text: str, context: Dict[str, Any] = None) -> str:
        """컨텍스트 정보를 활용한 텍스트 후처리"""
        if not context or not extracted_text:
            return extracted_text
        
        enhanced_text = extracted_text
        
        # 주제 키워드가 있으면 관련 용어 보정
        if context.get('topic_keywords'):
            keywords = [k.strip() for k in context['topic_keywords'].split(',')]
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
                names.extend([n.strip() for n in context['speakers'].split(',')])
            if context.get('participants'):
                # 괄호 안 내용 제거하고 이름만 추출
                import re
                participant_text = context['participants']
                participant_names = re.findall(r'([가-힣a-zA-Z\s]+?)(?:\s*\([^)]*\))?(?:,|$)', participant_text)
                names.extend([n.strip() for n in participant_names if n.strip()])
            
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
        """로깅 설정"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _lazy_load_whisper(self, model_size: str = "base") -> whisper.Whisper:
        """Whisper 모델 지연 로딩"""
        if self.whisper_model is None:
            self.logger.info(f"🎤 Whisper {model_size} 모델 로딩...")
            start_time = time.time()
            self.whisper_model = whisper.load_model(model_size)
            load_time = time.time() - start_time
            self.logger.info(f"✅ Whisper 로드 완료 ({load_time:.1f}초)")
        return self.whisper_model
    
    def _lazy_load_ocr(self) -> easyocr.Reader:
        """EasyOCR 모델 지연 로딩 (성능 최적화)"""
        if self.ocr_reader is None:
            self.logger.info("🖼️ EasyOCR 한/영 모델 로딩... (CPU 최적화)")
            start_time = time.time()
            
            # CPU 모드와 성능 최적화 설정
            self.ocr_reader = easyocr.Reader(
                ['ko', 'en'],
                gpu=False,  # CPU 강제 사용
                model_storage_directory=None,  # 기본 모델 디렉토리 사용
                user_network_directory=None,
                recog_network='CRNN',  # 기본 recognition network
                detector=True,
                recognizer=True,
                verbose=False  # 로그 최소화
            )
            
            load_time = time.time() - start_time
            self.logger.info(f"✅ EasyOCR 로드 완료 ({load_time:.1f}초)")
        return self.ocr_reader
    
    def _lazy_load_nlp(self) -> Optional[any]:
        """NLP 파이프라인 지연 로딩"""
        if not transformers_available:
            return None
            
        if self.nlp_pipeline is None:
            try:
                self.logger.info("🧠 NLP 모델 로딩...")
                start_time = time.time()
                self.nlp_pipeline = pipeline("summarization", 
                                           model="facebook/bart-large-cnn")
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
        """M4A 파일을 WAV로 변환 (FFmpeg 사용)"""
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
            
            self.logger.info("🔄 M4A → WAV 변환 중...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.logger.info("✅ M4A → WAV 변환 완료")
                return temp_wav_path
            else:
                self.logger.error(f"❌ FFmpeg 변환 실패: {result.stderr}")
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
    
    def _preprocess_m4a_file(self, file_path: str) -> str:
        """M4A 파일 전처리"""
        self.logger.info("🎵 M4A 파일 전처리 시작")
        
        # 1. 원본 파일 검증
        if not self._validate_audio_data(file_path):
            # 검증 실패시 FFmpeg 변환 시도
            self.logger.info("🔧 FFmpeg 변환으로 재시도")
            converted_path = self._convert_m4a_to_wav(file_path)
            if converted_path and self._validate_audio_data(converted_path):
                return converted_path
            else:
                # 변환도 실패시 정리하고 None 반환
                if converted_path and os.path.exists(converted_path):
                    os.unlink(converted_path)
                return None
        
        # 원본 파일이 정상이면 그대로 사용
        return file_path
    
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
            
            # M4A 파일 전처리
            if file_ext == ".m4a":
                processed_file_path = self._preprocess_m4a_file(file_path)
                if processed_file_path is None:
                    raise Exception("M4A 파일 전처리 실패: 오디오 데이터를 읽을 수 없습니다")
                
                # 변환된 파일인지 확인 (임시 파일 정리를 위해)
                temp_file_created = (processed_file_path != file_path)
            else:
                processed_file_path = file_path
            
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
            
            # 주얼리 키워드 분석 (향상된 텍스트 기반)
            jewelry_keywords = self._extract_jewelry_keywords(enhanced_text)
            
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
                "analysis_type": "real_whisper_stt",
                "timestamp": datetime.now().isoformat()
            }
            
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
            # OCR 모델 로드
            reader = self._lazy_load_ocr()
            
            # OCR 텍스트 추출 (품질과 성능 균형)
            self.logger.info("🔄 이미지 텍스트 추출 중... (품질 우선 모드)")
            results = reader.readtext(
                file_path,
                width_ths=0.5,     # 텍스트 폭 임계값 (더 민감하게)
                height_ths=0.5,    # 텍스트 높이 임계값 (더 민감하게)
                paragraph=False,   # 단락 모드 비활성화 (속도 향상)
                detail=1,          # 상세 정보 포함 (bbox, text, confidence)
                batch_size=1,      # 배치 크기 최소화
                workers=1,         # 워커 수 최소화
                text_threshold=0.4,   # 텍스트 감지 임계값 낮춤 (더 많은 텍스트 감지)
                low_text=0.3,      # 낮은 텍스트 신뢰도 임계값 낮춤
                link_threshold=0.3, # 링크 임계값 낮춤
                canvas_size=4096,  # 캔버스 크기 증가 (고해상도 지원)
                mag_ratio=2.0      # 확대 비율 증가 (작은 텍스트 감지)
            )
            
            processing_time = time.time() - start_time
            
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
                "analysis_type": "real_easyocr",
                "timestamp": datetime.now().isoformat()
            }
            
            self._update_stats(processing_time, True)
            self.logger.info(f"✅ 이미지 분석 완료 ({processing_time:.1f}초)")
            
            return analysis_result
            
        except Exception as e:
            error_msg = f"이미지 분석 실패: {str(e)}"
            self.logger.error(error_msg)
            
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
            
            # 1.5. 키프레임 추출 (MoviePy 사용 가능한 경우)
            keyframes_info = None
            if video_info.get('moviepy_duration') and video_info['file_size_mb'] <= 500:  # 500MB 이하만
                try:
                    self.logger.info("[INFO] 키프레임 추출 중...")
                    keyframes_result = large_video_processor.extract_keyframes_moviepy(video_path, num_frames=3)
                    if keyframes_result['status'] == 'success':
                        keyframes_info = keyframes_result
                        self.logger.info(f"[SUCCESS] {len(keyframes_result['keyframes'])}개 키프레임 추출 완료")
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
                "audio_extraction": audio_extract_result,
                "audio_analysis": audio_analysis,
                "processing_time": round(processing_time, 2),
                "timestamp": datetime.now().isoformat(),
                "enhanced_features": {
                    "moviepy_analysis": video_info.get('moviepy_duration') is not None,
                    "quality_analysis": video_info.get('video_quality_analysis') is not None,
                    "keyframe_extraction": keyframes_info is not None,
                    "frame_analysis": video_info.get('frame_analysis') is not None
                }
            }
            
            # 비디오 + 오디오 통합 분석
            if audio_analysis and audio_analysis.get('status') == 'success':
                transcription = audio_analysis.get('transcription', '')
                
                if transcription:
                    # 비디오 메타데이터 + STT 결과 통합
                    combined_text = f"비디오 파일: {file_name}\n"
                    
                    if video_info.get('format_name'):
                        combined_text += f"형식: {video_info['format_name']}\n"
                    
                    if video_info.get('duration_formatted'):
                        combined_text += f"길이: {video_info['duration_formatted']}\n"
                    
                    if video_info.get('video_info', {}).get('quality'):
                        combined_text += f"화질: {video_info['video_info']['quality']}\n"
                    
                    combined_text += f"음성 내용: {transcription}"
                    
                    # 통합 요약 및 키워드 추출
                    summary = self._generate_summary(combined_text)
                    keywords = self._extract_jewelry_keywords(combined_text)
                    
                    result["integrated_analysis"] = {
                        "summary": summary,
                        "jewelry_keywords": keywords,
                        "content_analysis": {
                            "has_speech": True,
                            "speech_duration": video_info.get('duration', 0),
                            "video_quality": video_info.get('video_info', {}).get('quality', 'Unknown'),
                            "file_size_category": self._categorize_file_size(video_info.get('file_size_mb', 0))
                        }
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

def analyze_file_real(file_path: str, file_type: str, language: str = "auto", context: Dict[str, Any] = None) -> Dict[str, Any]:
    """파일 실제 분석 (간편 사용, 컨텍스트 지원)"""
    if file_type == "audio":
        return global_analysis_engine.analyze_audio_file(file_path, language=language, context=context)
    elif file_type == "image":
        return global_analysis_engine.analyze_image_file(file_path, context=context)
    elif file_type == "document":
        return global_analysis_engine.analyze_document_file(file_path)
    elif file_type == "youtube":
        return global_analysis_engine.analyze_youtube_video(file_path, language=language)
    elif file_type == "video":
        return global_analysis_engine.analyze_video_file(file_path, language=language)
    else:
        return {
            "status": "error",
            "error": f"지원하지 않는 파일 타입: {file_type}",
            "file_name": os.path.basename(file_path) if os.path.exists(file_path) else file_path,
            "timestamp": datetime.now().isoformat()
        }

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