"""
음성 분석 엔진
Whisper STT 기반 음성-텍스트 변환 및 분석
"""

import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from .base_engine import BaseEngine

class AudioEngine(BaseEngine):
    """음성 분석 엔진 - Whisper STT 기반"""
    
    def __init__(self, model_size: str = "base", language: str = "ko"):
        super().__init__("audio")
        self.model_size = model_size
        self.language = language
        self.whisper_model = None
        
    def initialize(self) -> bool:
        """Whisper 모델 초기화"""
        try:
            import whisper
            
            # GPU 메모리 제한을 위해 CPU 모드 강제
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
            self.whisper_model = whisper.load_model(self.model_size)
            self.is_initialized = True
            
            logging.info(f"AudioEngine initialized with model: {self.model_size}")
            return True
            
        except Exception as e:
            logging.error(f"AudioEngine initialization failed: {e}")
            return False
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """음성 파일 분석"""
        if not self.is_initialized:
            raise RuntimeError("AudioEngine not initialized")
        
        if not self.is_supported_file(file_path):
            raise ValueError(f"Unsupported audio format: {Path(file_path).suffix}")
        
        try:
            # Whisper로 음성-텍스트 변환
            result = self.whisper_model.transcribe(
                file_path,
                language=self.language,
                verbose=False
            )
            
            # 세그먼트별 정보 추출
            segments = []
            for segment in result.get("segments", []):
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"], 
                    "text": segment["text"].strip(),
                    "confidence": segment.get("avg_logprob", 0)
                })
            
            # 화자 구분 및 키워드 추출 (간단 버전)
            full_text = result["text"]
            keywords = self._extract_keywords(full_text)
            
            return {
                "success": True,
                "full_text": full_text,
                "segments": segments,
                "keywords": keywords,
                "language": result.get("language", "ko"),
                "duration": segments[-1]["end"] if segments else 0,
                "segment_count": len(segments)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """간단한 키워드 추출"""
        # 기본적인 키워드 추출 (추후 확장 가능)
        import re
        
        # 한국어 단어 추출 (2글자 이상)
        korean_words = re.findall(r'[가-힣]{2,}', text)
        
        # 빈도수 기반 상위 키워드
        word_count = {}
        for word in korean_words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # 상위 10개 키워드
        top_keywords = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:10]
        return [word for word, count in top_keywords]
    
    def get_supported_formats(self) -> List[str]:
        """지원하는 오디오 형식"""
        return [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_size": self.model_size,
            "language": self.language,
            "initialized": self.is_initialized,
            "supported_formats": self.get_supported_formats()
        }