"""
텍스트 분석 엔진
자연어 처리 및 텍스트 분석을 위한 엔진
"""

import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import re

from .base_engine import BaseEngine

class TextEngine(BaseEngine):
    """텍스트 분석 엔진"""
    
    def __init__(self, language: str = "ko"):
        super().__init__("text")
        self.language = language
        self.transformers_available = False
        
    def initialize(self) -> bool:
        """텍스트 처리 라이브러리 초기화"""
        try:
            import transformers
            self.transformers_available = True
            logging.info("Transformers library available")
        except ImportError:
            logging.warning("Transformers library not available - using basic text processing")
        
        self.is_initialized = True
        return True
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """텍스트 파일 분석"""
        if not self.is_initialized:
            raise RuntimeError("TextEngine not initialized")
        
        if not self.is_supported_file(file_path):
            raise ValueError(f"Unsupported text format: {Path(file_path).suffix}")
        
        try:
            # 파일 읽기
            content = self._read_text_file(file_path)
            
            # 기본 텍스트 분석
            basic_analysis = self._analyze_basic_stats(content)
            
            # 언어 감지
            language_info = self._detect_language(content)
            
            # 키워드 추출
            keywords = self._extract_keywords(content)
            
            # 요약 (가능한 경우)
            summary = self._generate_summary(content) if self.transformers_available else ""
            
            return {
                "success": True,
                "content": content[:1000] + "..." if len(content) > 1000 else content,  # 미리보기
                "full_content_length": len(content),
                "basic_analysis": basic_analysis,
                "language_info": language_info,
                "keywords": keywords,
                "summary": summary,
                "has_transformers": self.transformers_available
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    def _read_text_file(self, file_path: str) -> str:
        """텍스트 파일 읽기 (다양한 인코딩 시도)"""
        encodings = ['utf-8', 'utf-8-sig', 'euc-kr', 'cp949', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # PDF 처리
        if Path(file_path).suffix.lower() == '.pdf':
            return self._extract_pdf_text(file_path)
        
        # DOCX 처리  
        if Path(file_path).suffix.lower() == '.docx':
            return self._extract_docx_text(file_path)
        
        raise ValueError(f"Could not decode text file: {file_path}")
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """PDF 텍스트 추출"""
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except ImportError:
            return "PDF text extraction requires PyPDF2 library"
        except Exception as e:
            return f"PDF extraction error: {str(e)}"
    
    def _extract_docx_text(self, file_path: str) -> str:
        """DOCX 텍스트 추출"""
        try:
            import docx
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except ImportError:
            return "DOCX text extraction requires python-docx library"
        except Exception as e:
            return f"DOCX extraction error: {str(e)}"
    
    def _analyze_basic_stats(self, text: str) -> Dict[str, Any]:
        """기본 텍스트 통계 분석"""
        lines = text.split('\n')
        words = text.split()
        
        # 문장 수 (간단 추정)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            "character_count": len(text),
            "word_count": len(words),
            "line_count": len(lines),
            "sentence_count": len(sentences),
            "paragraph_count": len([line for line in lines if line.strip()]),
            "average_words_per_sentence": len(words) / len(sentences) if sentences else 0,
            "average_chars_per_word": len(text) / len(words) if words else 0
        }
    
    def _detect_language(self, text: str) -> Dict[str, Any]:
        """언어 감지"""
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        chinese_chars = len(re.findall(r'[一-龯]', text)) 
        japanese_chars = len(re.findall(r'[ひらがなカタカナ]', text))
        numbers = len(re.findall(r'\d', text))
        
        total_chars = korean_chars + english_chars + chinese_chars + japanese_chars
        
        if total_chars == 0:
            return {"primary_language": "unknown", "confidence": 0}
        
        # 비율 계산
        ratios = {
            "korean": korean_chars / total_chars,
            "english": english_chars / total_chars,
            "chinese": chinese_chars / total_chars,
            "japanese": japanese_chars / total_chars
        }
        
        primary = max(ratios, key=ratios.get)
        confidence = ratios[primary]
        
        return {
            "primary_language": primary,
            "confidence": confidence,
            "language_ratios": ratios,
            "number_ratio": numbers / len(text) if text else 0
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출 (기본 버전)"""
        # 불용어 제거 (간단 버전)
        stopwords = {
            '이', '가', '을', '를', '에', '에서', '로', '으로', '와', '과', '의', '는', '은',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'
        }
        
        # 단어 추출
        words = re.findall(r'[가-힣a-zA-Z]{2,}', text.lower())
        words = [word for word in words if word not in stopwords]
        
        # 빈도수 계산
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # 상위 키워드 반환
        top_keywords = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:20]
        return [word for word, count in top_keywords]
    
    def _generate_summary(self, text: str) -> str:
        """텍스트 요약 생성 (Transformers 사용)"""
        if not self.transformers_available or len(text) < 100:
            return ""
        
        try:
            from transformers import pipeline
            
            # 요약 파이프라인 생성 (첫 번째 실행 시 모델 다운로드)
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            
            # 텍스트 길이 제한 (BART 모델 제한)
            max_length = 1024
            if len(text) > max_length:
                text = text[:max_length]
            
            summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
            return summary[0]['summary_text']
            
        except Exception as e:
            logging.warning(f"Summary generation failed: {e}")
            return f"Summary generation error: {str(e)}"
    
    def get_supported_formats(self) -> List[str]:
        """지원하는 텍스트 형식"""
        return [".txt", ".md", ".json", ".csv", ".log", ".py", ".js", ".html", ".xml", ".pdf", ".docx"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "language": self.language,
            "transformers_available": self.transformers_available,
            "initialized": self.is_initialized,
            "supported_formats": self.get_supported_formats()
        }