"""
솔로몬드 AI 시스템 - 통합 STT 분석 엔진 (주얼리 특화 v2.0)
OpenAI Whisper 기반 음성 인식 모듈 + 주얼리 업계 특화 후처리 + 다국어 번역
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union, List
import asyncio

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# 주얼리 특화 모듈 임포트
try:
    from .jewelry_enhancer import get_jewelry_enhancer, enhance_jewelry_transcription
    JEWELRY_ENHANCER_AVAILABLE = True
except ImportError:
    JEWELRY_ENHANCER_AVAILABLE = False
    print("[WARNING] 기존 주얼리 특화 모듈 로드 실패 - 기본 STT 기능만 사용")

# 새로운 확장 모듈 임포트
try:
    from .multilingual_translator import JewelryMultilingualTranslator
    MULTILINGUAL_AVAILABLE = True
except ImportError:
    MULTILINGUAL_AVAILABLE = False
    print("[WARNING] 다국어 번역 모듈 로드 실패")

try:
    from .jewelry_database import JewelryTerminologyDB
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    print("[WARNING] 주얼리 데이터베이스 모듈 로드 실패")

try:
    from .audio_processor import JewelryAudioProcessor
    AUDIO_PROCESSOR_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSOR_AVAILABLE = False
    print("[WARNING] 고급 오디오 처리 모듈 로드 실패")

class EnhancedAudioAnalyzer:
    """통합 음성 분석 엔진 클래스 (주얼리 특화 + 다국어 + 고급 오디오 처리)"""
    
    def __init__(self, 
                 model_size: str = "base", 
                 enable_jewelry_enhancement: bool = True,
                 enable_multilingual: bool = True,
                 enable_audio_preprocessing: bool = True,
                 enable_database: bool = True):
        """
        초기화
        
        Args:
            model_size: Whisper 모델 크기 (tiny, base, small, medium, large)
            enable_jewelry_enhancement: 주얼리 특화 기능 활성화 여부
            enable_multilingual: 다국어 번역 기능 활성화 여부
            enable_audio_preprocessing: 고급 오디오 전처리 활성화 여부
            enable_database: 주얼리 데이터베이스 활성화 여부
        """
        self.model_size = model_size
        self.model = None
        self.supported_formats = ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac']
        
        # 기능 활성화 플래그
        self.enable_jewelry_enhancement = enable_jewelry_enhancement and JEWELRY_ENHANCER_AVAILABLE
        self.enable_multilingual = enable_multilingual and MULTILINGUAL_AVAILABLE
        self.enable_audio_preprocessing = enable_audio_preprocessing and AUDIO_PROCESSOR_AVAILABLE
        self.enable_database = enable_database and DATABASE_AVAILABLE
        
        # 🌍 지원하는 언어 목록 (확장됨)
        self.supported_languages = {
            "auto": {"name": "자동 감지", "code": None, "flag": "🌐"},
            "ko": {"name": "한국어", "code": "ko", "flag": "🇰🇷"},
            "en": {"name": "English", "code": "en", "flag": "🇺🇸"},
            "zh": {"name": "中文", "code": "zh", "flag": "🇨🇳"},
            "ja": {"name": "日本語", "code": "ja", "flag": "🇯🇵"},
            "th": {"name": "ไทย", "code": "th", "flag": "🇹🇭"},
            "es": {"name": "Español", "code": "es", "flag": "🇪🇸"},
            "fr": {"name": "Français", "code": "fr", "flag": "🇫🇷"},
            "de": {"name": "Deutsch", "code": "de", "flag": "🇩🇪"},
            "ru": {"name": "Русский", "code": "ru", "flag": "🇷🇺"},
            "pt": {"name": "Português", "code": "pt", "flag": "🇵🇹"},
            "it": {"name": "Italiano", "code": "it", "flag": "🇮🇹"}
        }
        
        # 확장 모듈 초기화
        self._init_enhanced_modules()
        
    def _init_enhanced_modules(self):
        """확장 모듈들 초기화"""
        # 💎 기존 주얼리 특화 기능
        if self.enable_jewelry_enhancement:
            try:
                self.jewelry_enhancer = get_jewelry_enhancer()
                print("[JEWELRY] 주얼리 특화 기능 활성화")
            except Exception as e:
                print(f"[WARNING] 주얼리 특화 기능 비활성화: {e}")
                self.enable_jewelry_enhancement = False
                self.jewelry_enhancer = None
        else:
            self.jewelry_enhancer = None
        
        # 🌍 다국어 번역 모듈
        if self.enable_multilingual:
            try:
                self.translator = JewelryMultilingualTranslator()
                print("🌍 다국어 번역 모듈 활성화")
            except Exception as e:
                print(f"⚠️ 다국어 번역 모듈 비활성화: {e}")
                self.enable_multilingual = False
                self.translator = None
        else:
            self.translator = None
        
        # 🎵 고급 오디오 전처리 모듈
        if self.enable_audio_preprocessing:
            try:
                self.audio_processor = JewelryAudioProcessor()
                print("🎵 고급 오디오 전처리 모듈 활성화")
            except Exception as e:
                print(f"⚠️ 오디오 전처리 모듈 비활성화: {e}")
                self.enable_audio_preprocessing = False
                self.audio_processor = None
        else:
            self.audio_processor = None
        
        # 💾 주얼리 데이터베이스 모듈
        if self.enable_database:
            try:
                self.jewelry_db = JewelryTerminologyDB()
                print("💾 주얼리 데이터베이스 모듈 활성화")
            except Exception as e:
                print(f"⚠️ 데이터베이스 모듈 비활성화: {e}")
                self.enable_database = False
                self.jewelry_db = None
        else:
            self.jewelry_db = None
        
    def load_model(self) -> bool:
        """Whisper 모델 로드"""
        if not WHISPER_AVAILABLE:
            return False
            
        try:
            print(f"🎤 Whisper 모델 로딩... ({self.model_size})")
            self.model = whisper.load_model(self.model_size)
            print(f"✅ 모델 로드 성공: {self.model_size}")
            
            # 활성화된 기능들 출력
            enabled_features = []
            if self.enable_jewelry_enhancement: enabled_features.append("💎 주얼리 특화")
            if self.enable_multilingual: enabled_features.append("🌍 다국어 번역")
            if self.enable_audio_preprocessing: enabled_features.append("🎵 오디오 전처리")
            if self.enable_database: enabled_features.append("💾 데이터베이스")
            
            if enabled_features:
                print(f"🚀 활성화된 기능: {', '.join(enabled_features)}")
            
            return True
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    def is_supported_format(self, filename: str) -> bool:
        """지원하는 파일 형식인지 확인"""
        file_ext = Path(filename).suffix.lower()
        return file_ext in self.supported_formats
    
    def get_supported_languages(self) -> Dict:
        """지원하는 언어 목록 반환"""
        return self.supported_languages
    
    async def preprocess_audio_if_enabled(self, audio_path: str) -> str:
        """오디오 전처리 (활성화된 경우)"""
        if not self.enable_audio_preprocessing or not self.audio_processor:
            return audio_path
        
        try:
            print("🎵 고급 오디오 전처리 시작...")
            
            # 오디오 환경 분석
            analysis = self.audio_processor.analyze_jewelry_audio_environment(audio_path)
            env_type = analysis.get('jewelry_environment', {}).get('environment_type', 'auto')
            
            print(f"🏢 감지된 환경: {env_type}")
            
            # 환경에 맞는 전처리 적용
            processed_path = self.audio_processor.preprocess_jewelry_audio(
                audio_path, 
                environment_type=env_type,
                enhancement_level='medium'
            )
            
            print("✅ 오디오 전처리 완료")
            return processed_path
            
        except Exception as e:
            print(f"⚠️ 오디오 전처리 오류: {e}")
            return audio_path  # 실패 시 원본 사용
    
    async def analyze_audio_file(self, 
                                file_path: str, 
                                language: str = "auto",
                                enable_jewelry_features: bool = None,
                                enable_translation: bool = None,
                                target_languages: List[str] = None) -> Dict:
        """
        음성 파일 분석 (통합 버전)
        
        Args:
            file_path: 분석할 음성 파일 경로
            language: 인식할 언어 코드 (auto, ko, en, zh, ja 등)
            enable_jewelry_features: 주얼리 특화 기능 사용 여부
            enable_translation: 번역 기능 사용 여부
            target_languages: 번역할 언어 목록 (예: ['ko', 'en', 'zh'])
            
        Returns:
            분석 결과 딕셔너리 (주얼리 특화 정보 + 번역 + 오디오 분석 포함)
        """
        start_time = time.time()
        
        # 기능 사용 여부 결정
        use_jewelry_features = (enable_jewelry_features if enable_jewelry_features is not None 
                               else self.enable_jewelry_enhancement)
        use_translation = (enable_translation if enable_translation is not None 
                          else self.enable_multilingual)
        
        # 기본 번역 대상 언어 설정
        if target_languages is None and use_translation:
            target_languages = ['ko', 'en', 'zh']  # 기본값
        
        try:
            # 모델이 로드되지 않았으면 로드
            if self.model is None:
                if not self.load_model():
                    return {
                        "success": False,
                        "error": "Whisper 모델을 로드할 수 없습니다.",
                        "whisper_available": WHISPER_AVAILABLE
                    }
            
            # 파일 존재 확인
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"파일이 존재하지 않습니다: {file_path}"
                }
            
            # 파일 정보 수집
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
            
            print(f"🔍 음성 인식 시작: {Path(file_path).name}")
            
            # 1️⃣ 오디오 전처리 (필요한 경우)
            processed_file_path = await self.preprocess_audio_if_enabled(file_path)
            temp_files = [processed_file_path] if processed_file_path != file_path else []
            
            # 2️⃣ 언어 감지 (auto인 경우)
            target_language = language
            if language == "auto":
                detection_result = self.detect_language(processed_file_path)
                if detection_result["success"]:
                    target_language = detection_result["detected_language"]
                    print(f"🌐 자동 감지된 언어: {target_language}")
                else:
                    target_language = "ko"  # 기본값
                    print("⚠️ 언어 감지 실패, 한국어로 설정")
            
            # 3️⃣ Whisper STT 실행
            whisper_options = {
                "verbose": False,
                "task": "transcribe"
            }
            
            if target_language != "auto" and target_language in self.supported_languages:
                lang_code = self.supported_languages[target_language]["code"]
                if lang_code:
                    whisper_options["language"] = lang_code
            
            result = self.model.transcribe(processed_file_path, **whisper_options)
            
            transcribed_text = result["text"].strip()
            processing_time = round(time.time() - start_time, 2)
            detected_language = result.get("language", target_language)
            
            print(f"✅ 기본 인식 완료: {processing_time}초")
            print(f"📝 원본 결과: {transcribed_text[:100]}...")
            
            # 언어 정보 추가
            lang_info = self.supported_languages.get(detected_language, {
                "name": f"Unknown ({detected_language})", 
                "code": detected_language, 
                "flag": "❓"
            })
            
            # 기본 결과 구성
            result_data = {
                "success": True,
                "transcribed_text": transcribed_text,
                "processing_time": processing_time,
                "file_size_mb": file_size_mb,
                "detected_language": detected_language,
                "language_info": lang_info,
                "requested_language": language,
                "confidence": result.get("confidence", 0.0),
                "segments": result.get("segments", [])
            }
            
            # 4️⃣ 주얼리 특화 처리
            if use_jewelry_features and transcribed_text.strip():
                print("[JEWELRY] 주얼리 특화 후처리 시작...")
                jewelry_start_time = time.time()
                
                try:
                    # 기존 주얼리 enhancer 사용
                    if self.jewelry_enhancer:
                        jewelry_result = enhance_jewelry_transcription(
                            transcribed_text, 
                            detected_language,
                            include_analysis=True
                        )
                        
                        result_data.update({
                            "enhanced_text": jewelry_result.get("enhanced_text", transcribed_text),
                            "jewelry_corrections": jewelry_result.get("corrections", []),
                            "detected_jewelry_terms": jewelry_result.get("detected_terms", []),
                            "jewelry_analysis": jewelry_result.get("analysis", {}),
                            "jewelry_summary": jewelry_result.get("summary", "")
                        })
                    
                    # 데이터베이스 검색 (새로운 기능)
                    if self.jewelry_db and transcribed_text.strip():
                        # 주요 단어들로 용어 검색
                        words = transcribed_text.split()[:10]  # 처음 10개 단어만
                        db_terms = []
                        for word in words:
                            if len(word) > 2:  # 2글자 이상만
                                terms = self.jewelry_db.search_terms(word, detected_language, limit=3)
                                db_terms.extend(terms)
                        
                        result_data["database_terms"] = db_terms[:10]  # 최대 10개
                        
                        # 사용 통계 업데이트
                        for term in db_terms:
                            self.jewelry_db.update_usage_stats(term['term_key'])
                    
                    jewelry_processing_time = round(time.time() - jewelry_start_time, 2)
                    result_data["jewelry_processing_time"] = jewelry_processing_time
                    
                    print(f"[JEWELRY] 주얼리 특화 처리 완료: {jewelry_processing_time}초")
                    
                except Exception as e:
                    print(f"⚠️ 주얼리 특화 처리 오류: {e}")
                    result_data["jewelry_enhancement_error"] = str(e)
            
            # 5️⃣ 다국어 번역 처리
            if use_translation and self.translator and target_languages:
                print("🌍 다국어 번역 시작...")
                translation_start_time = time.time()
                
                try:
                    # 향상된 텍스트가 있으면 그것을 번역, 없으면 원본 번역
                    text_to_translate = result_data.get("enhanced_text", transcribed_text)
                    
                    translations = self.translator.translate_multiple(
                        text_to_translate,
                        target_languages,
                        detected_language
                    )
                    
                    result_data["translations"] = translations
                    result_data["translation_count"] = len(translations)
                    
                    translation_processing_time = round(time.time() - translation_start_time, 2)
                    result_data["translation_processing_time"] = translation_processing_time
                    
                    print(f"🌍 번역 완료: {len(translations)}개 언어, {translation_processing_time}초")
                    
                except Exception as e:
                    print(f"⚠️ 번역 오류: {e}")
                    result_data["translation_error"] = str(e)
            
            # 6️⃣ 최종 정리
            result_data["total_processing_time"] = round(time.time() - start_time, 2)
            result_data["enabled_features"] = {
                "jewelry_enhancement": use_jewelry_features,
                "multilingual_translation": use_translation,
                "audio_preprocessing": self.enable_audio_preprocessing,
                "database_lookup": self.enable_database
            }
            
            # 임시 파일 정리
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
            
            return result_data
            
        except Exception as e:
            processing_time = round(time.time() - start_time, 2)
            error_msg = str(e)
            
            print(f"❌ 분석 오류: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "processing_time": processing_time,
                "requested_language": language
            }
    
    def detect_language(self, audio_path: str) -> Dict:
        """자동 언어 감지 기능"""
        try:
            if self.model is None:
                if not self.load_model():
                    return {"success": False, "error": "모델 로드 실패"}
            
            # Whisper의 언어 감지 기능 사용
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # 언어 감지 실행
            _, probs = self.model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            confidence = probs[detected_lang]
            
            # 지원하는 언어인지 확인
            lang_info = self.supported_languages.get(detected_lang, {
                "name": f"Unknown ({detected_lang})", 
                "code": detected_lang, 
                "flag": "❓"
            })
            
            print(f"🌐 언어 감지: {lang_info['name']} (신뢰도: {confidence:.2f})")
            
            return {
                "success": True,
                "detected_language": detected_lang,
                "confidence": confidence,
                "language_info": lang_info,
                "all_probabilities": dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5])
            }
            
        except Exception as e:
            print(f"❌ 언어 감지 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "detected_language": "ko",  # 기본값
                "confidence": 0.0
            }
    
    async def analyze_uploaded_file(self, 
                                   file_content: bytes,
                                   filename: str,
                                   language: str = "auto",
                                   enable_jewelry_features: bool = None,
                                   enable_translation: bool = None,
                                   target_languages: List[str] = None) -> Dict:
        """
        업로드된 파일 분석 (통합 버전)
        """
        if not self.is_supported_format(filename):
            return {
                "success": False,
                "error": f"지원하지 않는 파일 형식: {Path(filename).suffix}. {', '.join(self.supported_formats)}만 지원합니다."
            }
        
        # 임시 파일 생성
        file_ext = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        try:
            # 분석 실행
            result = await self.analyze_audio_file(
                temp_path, 
                language, 
                enable_jewelry_features,
                enable_translation,
                target_languages
            )
            
            # 성공한 경우 파일 정보 추가
            if result["success"]:
                result["filename"] = filename
                result["file_size"] = f"{result['file_size_mb']} MB"
            
            return result
            
        finally:
            # 임시 파일 정리
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환 (확장된 정보 포함)"""
        info = {
            "model_size": self.model_size,
            "model_loaded": self.model is not None,
            "whisper_available": WHISPER_AVAILABLE,
            "supported_formats": self.supported_formats,
            "supported_languages": self.supported_languages,
            "default_language": "auto",
            "version": "2.0 - Enhanced with Multilingual & Advanced Audio Processing",
            "enabled_features": {
                "jewelry_enhancement": self.enable_jewelry_enhancement,
                "multilingual_translation": self.enable_multilingual,
                "audio_preprocessing": self.enable_audio_preprocessing,
                "database_lookup": self.enable_database
            }
        }
        
        # 각 모듈별 상세 정보
        if self.enable_multilingual and self.translator:
            info["translation_languages"] = self.translator.get_supported_languages()
        
        if self.enable_database and self.jewelry_db:
            info["database_stats"] = self.jewelry_db.get_stats()
        
        return info
    
    def get_jewelry_terminology_suggestions(self, query: str, language: str = "ko") -> List[Dict]:
        """주얼리 용어 제안 (자동완성용)"""
        if not self.enable_database or not self.jewelry_db:
            return []
        
        try:
            return self.jewelry_db.search_terms(query, language, limit=10)
        except Exception as e:
            print(f"⚠️ 용어 검색 오류: {e}")
            return []

# 하위 호환성을 위한 AudioAnalyzer 클래스 (기존 클래스 별칭)
class AudioAnalyzer(EnhancedAudioAnalyzer):
    """하위 호환성을 위한 기존 클래스 별칭"""
    def __init__(self, model_size: str = "base", enable_jewelry_enhancement: bool = True):
        super().__init__(model_size, enable_jewelry_enhancement)

# 전역 분석기 인스턴스 (싱글톤 패턴)
_analyzer_instance = None

def get_analyzer(model_size: str = "base", 
                enable_jewelry_enhancement: bool = True,
                enable_all_features: bool = True) -> EnhancedAudioAnalyzer:
    """전역 분석기 인스턴스 반환 (싱글톤)"""
    global _analyzer_instance
    if _analyzer_instance is None:
        if enable_all_features:
            _analyzer_instance = EnhancedAudioAnalyzer(
                model_size, 
                enable_jewelry_enhancement,
                enable_multilingual=True,
                enable_audio_preprocessing=True,
                enable_database=True
            )
        else:
            _analyzer_instance = EnhancedAudioAnalyzer(model_size, enable_jewelry_enhancement)
    return _analyzer_instance

# 편의 함수들
async def quick_analyze(file_path: str, 
                       language: str = "auto",
                       enable_jewelry_features: bool = True,
                       enable_translation: bool = True) -> Dict:
    """빠른 분석 함수 (통합 버전)"""
    analyzer = get_analyzer(enable_jewelry_enhancement=enable_jewelry_features)
    return await analyzer.analyze_audio_file(
        file_path, 
        language, 
        enable_jewelry_features,
        enable_translation
    )

def check_system_status() -> Dict:
    """시스템 전체 상태 확인"""
    return {
        "whisper_available": WHISPER_AVAILABLE,
        "jewelry_enhancement_available": JEWELRY_ENHANCER_AVAILABLE,
        "multilingual_available": MULTILINGUAL_AVAILABLE,
        "database_available": DATABASE_AVAILABLE,
        "audio_processor_available": AUDIO_PROCESSOR_AVAILABLE,
        "version": "2.0",
        "ready": WHISPER_AVAILABLE
    }

def get_language_support() -> Dict:
    """지원 언어 정보 반환"""
    analyzer = get_analyzer()
    return {
        "supported_languages": analyzer.get_supported_languages(),
        "auto_detection": True,
        "default_language": "auto",
        "translation_support": analyzer.enable_multilingual,
        "version": "2.0 - Enhanced Multilingual Support"
    }

def get_enhanced_features_info() -> Dict:
    """확장된 기능 정보 반환"""
    return {
        "jewelry_enhancement": {
            "available": JEWELRY_ENHANCER_AVAILABLE,
            "description": "주얼리 업계 전문 용어 인식 및 분석"
        },
        "multilingual_translation": {
            "available": MULTILINGUAL_AVAILABLE,
            "description": "실시간 다국어 번역 (한/영/중/일/태 등)"
        },
        "advanced_audio_processing": {
            "available": AUDIO_PROCESSOR_AVAILABLE,
            "description": "주얼리 업계 환경 특화 오디오 전처리"
        },
        "terminology_database": {
            "available": DATABASE_AVAILABLE,
            "description": "SQLite 기반 주얼리 전문 용어 데이터베이스"
        }
    }

# 호환성을 위한 별칭들
STTAnalyzer = EnhancedAudioAnalyzer
JewelrySTTAnalyzer = EnhancedAudioAnalyzer

def get_stt_analyzer():
    """STT 분석기 인스턴스 반환 (호환성 함수)"""
    return EnhancedAudioAnalyzer()

def get_jewelry_stt_analyzer():
    """주얼리 특화 STT 분석기 인스턴스 반환 (호환성 함수)"""
    return EnhancedAudioAnalyzer(enable_jewelry_enhancement=True)
