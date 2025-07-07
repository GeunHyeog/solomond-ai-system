"""
솔로몬드 AI 시스템 - 파일 처리기
다양한 파일 형식 처리 및 전처리 모듈
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

class FileProcessor:
    """파일 처리 클래스"""
    
    def __init__(self):
        self.supported_audio_formats = ['.mp3', '.wav', '.m4a']
        self.supported_document_formats = ['.pdf', '.docx', '.txt']
        self.supported_image_formats = ['.jpg', '.jpeg', '.png']
        
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """지원하는 파일 형식 반환"""
        return {
            "audio": self.supported_audio_formats,
            "document": self.supported_document_formats, 
            "image": self.supported_image_formats
        }
    
    def detect_file_type(self, filename: str) -> str:
        """파일 타입 감지"""
        ext = Path(filename).suffix.lower()
        
        if ext in self.supported_audio_formats:
            return "audio"
        elif ext in self.supported_document_formats:
            return "document"
        elif ext in self.supported_image_formats:
            return "image"
        else:
            return "unknown"
    
    def validate_file(self, filename: str, max_size_mb: int = 100) -> Dict:
        """파일 유효성 검사"""
        file_type = self.detect_file_type(filename)
        
        if file_type == "unknown":
            return {
                "valid": False,
                "error": f"지원하지 않는 파일 형식: {Path(filename).suffix}"
            }
        
        return {
            "valid": True,
            "file_type": file_type,
            "extension": Path(filename).suffix.lower()
        }
    
    async def process_file(self, file_content: bytes, filename: str) -> Dict:
        """파일 처리 (미래 확장용)"""
        validation = self.validate_file(filename)
        
        if not validation["valid"]:
            return validation
        
        # 미래에 파일 타입별 처리 로직 추가 예정
        return {
            "success": True,
            "file_info": {
                "filename": filename,
                "size_bytes": len(file_content),
                "size_mb": round(len(file_content) / (1024 * 1024), 2),
                "type": validation["file_type"]
            }
        }

# 전역 인스턴스
_processor_instance = None

def get_file_processor() -> FileProcessor:
    """전역 파일 처리기 인스턴스 반환"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = FileProcessor()
    return _processor_instance
