"""
로깅 유틸리티
통합 로깅 시스템 설정 및 관리
"""

import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "solomond_ai", 
                level: str = "INFO",
                log_file: Optional[str] = None,
                console_output: bool = True) -> logging.Logger:
    """로거 설정"""
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 기존 핸들러 제거 (중복 방지)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포매터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 출력
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 파일 출력
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = "solomond_ai") -> logging.Logger:
    """로거 인스턴스 가져오기"""
    return logging.getLogger(name)

class AnalysisLogger:
    """분석 과정 전용 로거"""
    
    def __init__(self, session_id: str, output_dir: str = "logs"):
        self.session_id = session_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 세션별 로그 파일
        log_file = self.output_dir / f"analysis_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        self.logger = setup_logger(
            name=f"analysis_{session_id}",
            log_file=str(log_file),
            console_output=False
        )
    
    def log_start(self, files: list, engines: list):
        """분석 시작 로그"""
        self.logger.info(f"=== Analysis Session {self.session_id} Started ===")
        self.logger.info(f"Files to process: {len(files)}")
        self.logger.info(f"Engines enabled: {engines}")
        for i, file_path in enumerate(files, 1):
            self.logger.info(f"  {i}. {file_path}")
    
    def log_engine_start(self, engine_name: str, file_count: int):
        """엔진 시작 로그"""
        self.logger.info(f"--- {engine_name.upper()} Engine Started ---")
        self.logger.info(f"Processing {file_count} files")
    
    def log_file_processed(self, engine_name: str, file_path: str, success: bool, duration: float, details: dict = None):
        """파일 처리 완료 로그"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"{engine_name}: {status} - {file_path} ({duration:.2f}s)")
        
        if details:
            for key, value in details.items():
                self.logger.info(f"  {key}: {value}")
    
    def log_engine_complete(self, engine_name: str, success_count: int, total_count: int, total_duration: float):
        """엔진 완료 로그"""
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        self.logger.info(f"--- {engine_name.upper()} Engine Completed ---")
        self.logger.info(f"Success: {success_count}/{total_count} ({success_rate:.1f}%)")
        self.logger.info(f"Total duration: {total_duration:.2f}s")
    
    def log_integration_start(self):
        """통합 분석 시작 로그"""
        self.logger.info("--- Integration Analysis Started ---")
    
    def log_integration_complete(self, consistency_score: float, insights: dict):
        """통합 분석 완료 로그"""
        self.logger.info("--- Integration Analysis Completed ---")
        self.logger.info(f"Consistency Score: {consistency_score:.1f}/100")
        
        if insights:
            self.logger.info("Key Insights:")
            for key, value in insights.items():
                self.logger.info(f"  {key}: {value}")
    
    def log_error(self, error_msg: str, exception: Exception = None):
        """에러 로그"""
        self.logger.error(f"ERROR: {error_msg}")
        if exception:
            self.logger.error(f"Exception: {str(exception)}")
    
    def log_warning(self, warning_msg: str):
        """경고 로그"""
        self.logger.warning(f"WARNING: {warning_msg}")
    
    def log_finish(self, total_duration: float, summary: dict):
        """분석 완료 로그"""
        self.logger.info(f"=== Analysis Session {self.session_id} Completed ===")
        self.logger.info(f"Total Duration: {total_duration:.2f}s")
        
        if summary:
            self.logger.info("Final Summary:")
            for key, value in summary.items():
                self.logger.info(f"  {key}: {value}")
    
    def get_log_file_path(self) -> str:
        """로그 파일 경로 반환"""
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return handler.baseFilename
        return ""