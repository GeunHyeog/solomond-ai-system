"""
솔로몬드 AI 시스템 - STT 분석 엔진
OpenAI Whisper 기반 음성 인식 모듈
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union
import asyncio

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

class AudioAnalyzer:
    """음성 분석 엔진 클래스"""
    
    def __init__(self, model_size: str = "base"):
        """
        초기화
        
        Args:
            model_size: Whisper 모델 크기 (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.model = None
        self.supported_formats = ['.mp3', '.wav', '.m4a']
        
    def load_model(self) -> bool:
        """Whisper 모델 로드"""
        if not WHISPER_AVAILABLE:
            return False
            
        try:
            print(f"🎤 Whisper 모델 로딩... ({self.model_size})")
            self.model = whisper.load_model(self.model_size)
            print(f"✅ 모델 로드 성공: {self.model_size}")
            return True
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    def is_supported_format(self, filename: str) -> bool:
        """지원하는 파일 형식인지 확인"""
        file_ext = Path(filename).suffix.lower()
        return file_ext in self.supported_formats
    
    async def analyze_audio_file(self, 
                                file_path: str, 
                                language: str = "ko") -> Dict:
        """
        음성 파일 분석
        
        Args:
            file_path: 분석할 음성 파일 경로
            language: 인식할 언어 코드 (기본: 한국어)
            
        Returns:
            분석 결과 딕셔너리
        """
        start_time = time.time()
        
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
            
            # Whisper로 음성 인식 실행
            result = self.model.transcribe(file_path, language=language)
            
            transcribed_text = result["text"].strip()
            processing_time = round(time.time() - start_time, 2)
            
            print(f"✅ 인식 완료: {processing_time}초")
            print(f"📝 결과: {transcribed_text[:100]}...")
            
            return {
                "success": True,
                "transcribed_text": transcribed_text,
                "processing_time": processing_time,
                "file_size_mb": file_size_mb,
                "detected_language": result.get("language", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "segments": result.get("segments", [])
            }
            
        except Exception as e:
            processing_time = round(time.time() - start_time, 2)
            error_msg = str(e)
            
            print(f"❌ 분석 오류: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "processing_time": processing_time
            }
    
    async def analyze_uploaded_file(self, 
                                   file_content: bytes,
                                   filename: str,
                                   language: str = "ko") -> Dict:
        """
        업로드된 파일 분석 (임시 파일 생성 후 분석)
        
        Args:
            file_content: 파일 바이너리 데이터
            filename: 원본 파일명
            language: 인식할 언어 코드
            
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
            result = await self.analyze_audio_file(temp_path, language)
            
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
        """모델 정보 반환"""
        return {
            "model_size": self.model_size,
            "model_loaded": self.model is not None,
            "whisper_available": WHISPER_AVAILABLE,
            "supported_formats": self.supported_formats
        }
    
    def translate_to_korean(self, text: str) -> str:
        """
        영어 텍스트를 한국어로 번역 (기본 구현)
        추후 번역 API나 모델로 대체 예정
        """
        # 임시 구현: 간단한 키워드 번역
        translations = {
            "hello": "안녕하세요",
            "thank you": "감사합니다",
            "yes": "네",
            "no": "아니오"
        }
        
        result = text
        for eng, kor in translations.items():
            result = result.replace(eng, kor)
        
        return result

# 전역 분석기 인스턴스 (싱글톤 패턴)
_analyzer_instance = None

def get_analyzer(model_size: str = "base") -> AudioAnalyzer:
    """전역 분석기 인스턴스 반환 (싱글톤)"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = AudioAnalyzer(model_size)
    return _analyzer_instance

# 편의 함수들
async def quick_analyze(file_path: str, language: str = "ko") -> Dict:
    """빠른 분석 함수"""
    analyzer = get_analyzer()
    return await analyzer.analyze_audio_file(file_path, language)

def check_whisper_status() -> Dict:
    """Whisper 상태 확인"""
    return {
        "whisper_available": WHISPER_AVAILABLE,
        "import_error": None if WHISPER_AVAILABLE else "openai-whisper 패키지를 설치해주세요: pip install openai-whisper"
    }
