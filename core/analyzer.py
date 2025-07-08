"""
솔로몬드 AI 시스템 - STT 분석 엔진 (주얼리 특화 v1.1)
OpenAI Whisper 기반 음성 인식 모듈 + 주얼리 업계 특화 후처리
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
    print("⚠️ 주얼리 특화 모듈 로드 실패 - 기본 STT 기능만 사용")

class AudioAnalyzer:
    """음성 분석 엔진 클래스 (주얼리 특화 지원)"""
    
    def __init__(self, model_size: str = "base", enable_jewelry_enhancement: bool = True):
        """
        초기화
        
        Args:
            model_size: Whisper 모델 크기 (tiny, base, small, medium, large)
            enable_jewelry_enhancement: 주얼리 특화 기능 활성화 여부
        """
        self.model_size = model_size
        self.model = None
        self.supported_formats = ['.mp3', '.wav', '.m4a']
        self.enable_jewelry_enhancement = enable_jewelry_enhancement and JEWELRY_ENHANCER_AVAILABLE
        
        # 🌍 지원하는 언어 목록
        self.supported_languages = {
            "auto": {"name": "자동 감지", "code": None, "flag": "🌐"},
            "ko": {"name": "한국어", "code": "ko", "flag": "🇰🇷"},
            "en": {"name": "English", "code": "en", "flag": "🇺🇸"},
            "zh": {"name": "中文", "code": "zh", "flag": "🇨🇳"},
            "ja": {"name": "日本語", "code": "ja", "flag": "🇯🇵"},
            "es": {"name": "Español", "code": "es", "flag": "🇪🇸"},
            "fr": {"name": "Français", "code": "fr", "flag": "🇫🇷"},
            "de": {"name": "Deutsch", "code": "de", "flag": "🇩🇪"},
            "ru": {"name": "Русский", "code": "ru", "flag": "🇷🇺"},
            "pt": {"name": "Português", "code": "pt", "flag": "🇵🇹"},
            "it": {"name": "Italiano", "code": "it", "flag": "🇮🇹"}
        }
        
        # 💎 주얼리 특화 기능 초기화
        if self.enable_jewelry_enhancement:
            try:
                self.jewelry_enhancer = get_jewelry_enhancer()
                print("💎 주얼리 특화 기능 활성화")
            except Exception as e:
                print(f"⚠️ 주얼리 특화 기능 비활성화: {e}")
                self.enable_jewelry_enhancement = False
        else:
            self.jewelry_enhancer = None
        
    def load_model(self) -> bool:
        """Whisper 모델 로드"""
        if not WHISPER_AVAILABLE:
            return False
            
        try:
            print(f"🎤 Whisper 모델 로딩... ({self.model_size})")
            self.model = whisper.load_model(self.model_size)
            print(f"✅ 모델 로드 성공: {self.model_size}")
            
            if self.enable_jewelry_enhancement:
                print("💎 주얼리 특화 모드 활성화됨")
            
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
    
    async def analyze_audio_file(self, 
                                file_path: str, 
                                language: str = "auto",
                                enable_jewelry_features: bool = None) -> Dict:
        """
        음성 파일 분석 (주얼리 특화 지원)
        
        Args:
            file_path: 분석할 음성 파일 경로
            language: 인식할 언어 코드 (auto, ko, en, zh, ja 등)
            enable_jewelry_features: 주얼리 특화 기능 사용 여부 (None이면 기본값 사용)
            
        Returns:
            분석 결과 딕셔너리 (주얼리 특화 정보 포함)
        """
        start_time = time.time()
        
        # 주얼리 특화 기능 사용 여부 결정
        use_jewelry_features = (enable_jewelry_features if enable_jewelry_features is not None 
                               else self.enable_jewelry_enhancement)
        
        try:
            # 모델이 로드되지 않았으면 로드
            if self.model is None:
                if not self.load_model():
                    return {
                        "success": False,
                        "error": "Whisper 모델을 로드할 수 없습니다.",
                        "whisper_available": WHISPER_AVAILABLE,
                        "jewelry_enhancement": False
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
            if use_jewelry_features:
                print("💎 주얼리 특화 분석 모드")
            
            # 자동 언어 감지
            target_language = language
            if language == "auto":
                detection_result = self.detect_language(file_path)
                if detection_result["success"]:
                    target_language = detection_result["detected_language"]
                    print(f"🌐 자동 감지된 언어: {target_language}")
                else:
                    target_language = "ko"  # 기본값
                    print("⚠️ 언어 감지 실패, 한국어로 설정")
            
            # Whisper로 음성 인식 실행
            whisper_options = {
                "verbose": False,
                "task": "transcribe"
            }
            
            # 자동 감지가 아닌 경우에만 언어 지정
            if target_language != "auto" and target_language in self.supported_languages:
                lang_code = self.supported_languages[target_language]["code"]
                if lang_code:
                    whisper_options["language"] = lang_code
            
            result = self.model.transcribe(file_path, **whisper_options)
            
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
            basic_result = {
                "success": True,
                "transcribed_text": transcribed_text,
                "processing_time": processing_time,
                "file_size_mb": file_size_mb,
                "detected_language": detected_language,
                "language_info": lang_info,
                "requested_language": language,
                "confidence": result.get("confidence", 0.0),
                "segments": result.get("segments", []),
                "jewelry_enhancement": use_jewelry_features
            }
            
            # 💎 주얼리 특화 처리
            if use_jewelry_features and transcribed_text.strip():
                try:
                    print("💎 주얼리 특화 후처리 시작...")
                    jewelry_start_time = time.time()
                    
                    # 주얼리 특화 분석 실행
                    jewelry_result = enhance_jewelry_transcription(
                        transcribed_text, 
                        detected_language,
                        include_analysis=True
                    )
                    
                    jewelry_processing_time = round(time.time() - jewelry_start_time, 2)
                    
                    # 결과 통합
                    basic_result.update({
                        "enhanced_text": jewelry_result.get("enhanced_text", transcribed_text),
                        "jewelry_corrections": jewelry_result.get("corrections", []),
                        "detected_jewelry_terms": jewelry_result.get("detected_terms", []),
                        "jewelry_analysis": jewelry_result.get("analysis", {}),
                        "jewelry_summary": jewelry_result.get("summary", ""),
                        "enhancement_confidence": jewelry_result.get("confidence", 1.0),
                        "jewelry_processing_time": jewelry_processing_time
                    })
                    
                    print(f"💎 주얼리 특화 처리 완료: {jewelry_processing_time}초")
                    print(f"🔧 {len(jewelry_result.get('corrections', []))}개 수정사항")
                    print(f"📚 {len(jewelry_result.get('detected_terms', []))}개 주얼리 용어 발견")
                    
                    if jewelry_result.get("enhanced_text") != transcribed_text:
                        print(f"✨ 개선된 결과: {jewelry_result.get('enhanced_text', '')[:100]}...")
                    
                except Exception as e:
                    print(f"⚠️ 주얼리 특화 처리 오류: {e}")
                    # 오류가 발생해도 기본 STT 결과는 반환
                    basic_result["jewelry_enhancement_error"] = str(e)
            
            # 총 처리 시간 업데이트
            basic_result["total_processing_time"] = round(time.time() - start_time, 2)
            
            return basic_result
            
        except Exception as e:
            processing_time = round(time.time() - start_time, 2)
            error_msg = str(e)
            
            print(f"❌ 분석 오류: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "processing_time": processing_time,
                "requested_language": language,
                "jewelry_enhancement": use_jewelry_features
            }
    
    async def analyze_uploaded_file(self, 
                                   file_content: bytes,
                                   filename: str,
                                   language: str = "auto",
                                   enable_jewelry_features: bool = None) -> Dict:
        """
        업로드된 파일 분석 (주얼리 특화 지원)
        
        Args:
            file_content: 파일 바이너리 데이터
            filename: 원본 파일명
            language: 인식할 언어 코드
            enable_jewelry_features: 주얼리 특화 기능 사용 여부
            
        Returns:
            분석 결과 딕셔너리
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
                enable_jewelry_features
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
        """모델 정보 반환 (주얼리 특화 정보 포함)"""
        info = {
            "model_size": self.model_size,
            "model_loaded": self.model is not None,
            "whisper_available": WHISPER_AVAILABLE,
            "supported_formats": self.supported_formats,
            "supported_languages": self.supported_languages,
            "default_language": "auto",
            "phase": "3.3 - Jewelry Industry Specialized",
            "jewelry_enhancement": self.enable_jewelry_enhancement
        }
        
        # 주얼리 특화 기능 정보 추가
        if self.enable_jewelry_enhancement and self.jewelry_enhancer:
            try:
                jewelry_stats = self.jewelry_enhancer.get_enhancement_stats()
                info["jewelry_features"] = jewelry_stats
            except:
                info["jewelry_features"] = {"status": "available", "details": "loaded"}
        
        return info
    
    def translate_to_korean(self, text: str, source_lang: str = "en") -> str:
        """다른 언어 텍스트를 한국어로 번역 (확장된 구현)"""
        # 임시 구현: 확장된 키워드 번역
        translations = {
            "en": {
                "hello": "안녕하세요",
                "thank you": "감사합니다",
                "yes": "네",
                "no": "아니오",
                "good morning": "좋은 아침",
                "good evening": "좋은 저녁",
                "how are you": "안녕하세요",
                "nice to meet you": "만나서 반갑습니다",
                # 주얼리 관련 번역 추가
                "diamond": "다이아몬드",
                "ruby": "루비", 
                "sapphire": "사파이어",
                "emerald": "에메랄드",
                "carat": "캐럿",
                "cut": "컷",
                "color": "컬러",
                "clarity": "클래리티",
                "certification": "감정서"
            },
            "zh": {
                "你好": "안녕하세요",
                "谢谢": "감사합니다",
                "是": "네",
                "不是": "아니오",
                # 주얼리 관련 번역 추가
                "钻石": "다이아몬드",
                "红宝石": "루비",
                "蓝宝石": "사파이어",
                "祖母绿": "에메랄드",
                "克拉": "캐럿"
            },
            "ja": {
                "こんにちは": "안녕하세요",
                "ありがとう": "감사합니다",
                "はい": "네",
                "いいえ": "아니오",
                # 주얼리 관련 번역 추가
                "ダイヤモンド": "다이아몬드",
                "ルビー": "루비",
                "サファイア": "사파이어",
                "エメラルド": "에메랄드"
            }
        }
        
        if source_lang in translations:
            result = text
            for foreign, korean in translations[source_lang].items():
                result = result.replace(foreign, korean)
            return result
        
        return text  # 번역 사전에 없으면 원문 반환

# 전역 분석기 인스턴스 (싱글톤 패턴)
_analyzer_instance = None

def get_analyzer(model_size: str = "base", enable_jewelry_enhancement: bool = True) -> AudioAnalyzer:
    """전역 분석기 인스턴스 반환 (싱글톤)"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = AudioAnalyzer(model_size, enable_jewelry_enhancement)
    return _analyzer_instance

# 편의 함수들
async def quick_analyze(file_path: str, 
                       language: str = "auto",
                       enable_jewelry_features: bool = True) -> Dict:
    """빠른 분석 함수 (주얼리 특화 지원)"""
    analyzer = get_analyzer(enable_jewelry_enhancement=enable_jewelry_features)
    return await analyzer.analyze_audio_file(file_path, language, enable_jewelry_features)

def check_whisper_status() -> Dict:
    """Whisper 및 주얼리 특화 기능 상태 확인"""
    return {
        "whisper_available": WHISPER_AVAILABLE,
        "jewelry_enhancement_available": JEWELRY_ENHANCER_AVAILABLE,
        "import_error": None if WHISPER_AVAILABLE else "openai-whisper 패키지를 설치해주세요: pip install openai-whisper",
        "jewelry_enhancement_error": None if JEWELRY_ENHANCER_AVAILABLE else "주얼리 특화 모듈 로드 실패"
    }

def get_language_support() -> Dict:
    """지원 언어 정보 반환"""
    analyzer = get_analyzer()
    return {
        "supported_languages": analyzer.get_supported_languages(),
        "auto_detection": True,
        "default_language": "auto",
        "jewelry_enhancement": analyzer.enable_jewelry_enhancement,
        "phase": "3.3 - Jewelry Industry Specialized"
    }

def get_jewelry_features_info() -> Dict:
    """주얼리 특화 기능 정보 반환"""
    if not JEWELRY_ENHANCER_AVAILABLE:
        return {
            "available": False,
            "error": "주얼리 특화 모듈을 로드할 수 없습니다"
        }
    
    try:
        enhancer = get_jewelry_enhancer()
        return {
            "available": True,
            "features": enhancer.get_enhancement_stats(),
            "description": "주얼리 업계 전문 용어 인식 및 분석 기능"
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }
