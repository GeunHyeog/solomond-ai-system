"""
이미지 분석 엔진
EasyOCR 기반 텍스트 추출 및 이미지 분석
"""

import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from .base_engine import BaseEngine

class ImageEngine(BaseEngine):
    """이미지 분석 엔진 - EasyOCR 기반"""
    
    def __init__(self, languages: List[str] = None):
        super().__init__("image")
        self.languages = languages or ['ko', 'en']
        self.ocr_reader = None
        
    def initialize(self) -> bool:
        """EasyOCR 초기화"""
        try:
            import easyocr
            
            # GPU 메모리 제한을 위해 CPU 모드 강제
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
            self.ocr_reader = easyocr.Reader(
                self.languages,
                gpu=False  # CPU 모드 강제
            )
            self.is_initialized = True
            
            logging.info(f"ImageEngine initialized with languages: {self.languages}")
            return True
            
        except Exception as e:
            logging.error(f"ImageEngine initialization failed: {e}")
            return False
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """이미지 파일 분석"""
        if not self.is_initialized:
            raise RuntimeError("ImageEngine not initialized")
        
        if not self.is_supported_file(file_path):
            raise ValueError(f"Unsupported image format: {Path(file_path).suffix}")
        
        try:
            # EasyOCR로 텍스트 추출
            results = self.ocr_reader.readtext(file_path)
            
            # 텍스트 블록들 정리
            text_blocks = []
            all_text = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # 신뢰도 임계값
                    text_blocks.append({
                        "text": text.strip(),
                        "confidence": float(confidence),
                        "bbox": {
                            "top_left": bbox[0],
                            "top_right": bbox[1], 
                            "bottom_right": bbox[2],
                            "bottom_left": bbox[3]
                        }
                    })
                    all_text.append(text.strip())
            
            # 전체 텍스트 결합
            full_text = " ".join(all_text)
            
            # 간단한 분석
            analysis = self._analyze_text_content(full_text)
            
            return {
                "success": True,
                "text_blocks": text_blocks,
                "full_text": full_text,
                "total_blocks": len(text_blocks),
                "average_confidence": sum(block["confidence"] for block in text_blocks) / len(text_blocks) if text_blocks else 0,
                "analysis": analysis
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    def _analyze_text_content(self, text: str) -> Dict[str, Any]:
        """텍스트 내용 간단 분석"""
        import re
        
        # 기본 통계
        word_count = len(text.split())
        char_count = len(text)
        
        # 숫자, 이메일, URL 등 패턴 감지
        numbers = re.findall(r'\d+', text)
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        
        # 언어 감지 (간단 버전)
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        language = "korean" if korean_chars > english_chars else "english" if english_chars > 0 else "mixed"
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "detected_language": language,
            "contains_numbers": len(numbers) > 0,
            "contains_emails": len(emails) > 0,
            "contains_urls": len(urls) > 0,
            "number_count": len(numbers),
            "email_count": len(emails),
            "url_count": len(urls)
        }
    
    def get_supported_formats(self) -> List[str]:
        """지원하는 이미지 형식"""
        return [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "languages": self.languages,
            "initialized": self.is_initialized,
            "supported_formats": self.get_supported_formats()
        }