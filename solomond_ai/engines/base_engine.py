"""
기본 분석 엔진 클래스
모든 특화 엔진들이 상속받는 베이스 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
import time

class BaseEngine(ABC):
    """모든 분석 엔진의 기본 클래스"""
    
    def __init__(self, engine_name: str):
        self.engine_name = engine_name
        self.is_initialized = False
        self.model = None
        self.config = {}
    
    @abstractmethod
    def initialize(self) -> bool:
        """엔진 초기화 (모델 로딩 등)"""
        pass
    
    @abstractmethod
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """단일 파일 분석"""
        pass
    
    def analyze_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """다중 파일 분석"""
        if not self.is_initialized:
            self.initialize()
        
        results = []
        for file_path in file_paths:
            try:
                result = self.analyze_file(file_path)
                result["file_path"] = file_path
                result["timestamp"] = time.time()
                result["engine"] = self.engine_name
                results.append(result)
            except Exception as e:
                results.append({
                    "file_path": file_path,
                    "error": str(e),
                    "engine": self.engine_name,
                    "timestamp": time.time()
                })
        
        return results
    
    def get_supported_formats(self) -> List[str]:
        """지원하는 파일 형식 반환"""
        return []
    
    def is_supported_file(self, file_path: str) -> bool:
        """파일이 지원되는 형식인지 확인"""
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.get_supported_formats()
    
    def set_config(self, config: Dict[str, Any]):
        """엔진 설정"""
        self.config.update(config)
    
    def get_status(self) -> Dict[str, Any]:
        """엔진 상태 정보"""
        return {
            "name": self.engine_name,
            "initialized": self.is_initialized,
            "supported_formats": self.get_supported_formats(),
            "config": self.config
        }