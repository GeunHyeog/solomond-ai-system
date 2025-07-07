"""
솔로몬드 AI 시스템 - 로깅 시스템
시스템 로깅 및 모니터링 모듈
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

class SystemLogger:
    """시스템 로거 클래스"""
    
    def __init__(self, name: str = "solomond_ai", log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 핸들러가 이미 있는지 확인
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """로그 핸들러 설정"""
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """INFO 레벨 로그"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """WARNING 레벨 로그"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """ERROR 레벨 로그"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """DEBUG 레벨 로그"""
        self.logger.debug(message)

# 전역 로거 인스턴스
_logger_instance = None

def get_logger(name: Optional[str] = None) -> SystemLogger:
    """전역 로거 인스턴스 반환"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = SystemLogger(name or "solomond_ai")
    return _logger_instance

# 편의 함수들
def log_info(message: str):
    """INFO 로그 편의 함수"""
    get_logger().info(message)

def log_error(message: str):
    """ERROR 로그 편의 함수"""
    get_logger().error(message)

def log_warning(message: str):
    """WARNING 로그 편의 함수"""
    get_logger().warning(message)
