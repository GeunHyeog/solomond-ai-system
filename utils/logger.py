"""
🔧 로거 설정 통합 모듈
솔로몬드 AI 시스템 - 중복 코드 제거 (1/3단계)

목적: 20개 파일에서 반복되는 logger 설정을 중앙화
효과: 로깅 설정 표준화 및 60줄 코드 감소
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import os

class SolomondLogger:
    """솔로몬드 AI 시스템 전용 로거"""
    
    _loggers: Dict[str, logging.Logger] = {}
    _initialized = False
    
    @classmethod
    def initialize(cls, 
                   log_level: str = "INFO",
                   log_dir: str = None,
                   enable_file_logging: bool = True):
        """로깅 시스템 초기화"""
        
        if cls._initialized:
            return
        
        # 로그 디렉토리 설정
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs"
        
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)
        
        # 기본 포맷 설정
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 루트 로거 설정
        root_logger = logging.getLogger('solomond')
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # 파일 핸들러 (선택적)
        if enable_file_logging:
            log_file = log_dir / f"solomond_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        cls._initialized = True

    @classmethod
    def get_logger(cls, name: str = None) -> logging.Logger:
        """표준 로거 인스턴스 반환
        
        Args:
            name: 로거 이름 (보통 __name__ 사용)
            
        Returns:
            logging.Logger: 설정된 로거 인스턴스
        """
        
        # 자동 초기화
        if not cls._initialized:
            cls.initialize()
        
        # 이름 처리
        if name is None:
            name = 'solomond.main'
        elif not name.startswith('solomond'):
            name = f'solomond.{name}'
        
        # 캐시된 로거 반환
        if name in cls._loggers:
            return cls._loggers[name]
        
        # 새 로거 생성
        logger = logging.getLogger(name)
        cls._loggers[name] = logger
        
        return logger

def get_logger(name: str = None) -> logging.Logger:
    """간편한 로거 생성 함수
    
    기존 코드에서 이렇게 사용:
    ```python
    from utils.logger import get_logger
    logger = get_logger(__name__)
    ```
    
    Args:
        name: 로거 이름 (보통 __name__ 사용)
        
    Returns:
        logging.Logger: 설정된 로거 인스턴스
    """
    return SolomondLogger.get_logger(name)

def set_log_level(level: str):
    """로그 레벨 동적 변경
    
    Args:
        level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    logging.getLogger('solomond').setLevel(getattr(logging, level.upper()))

def create_module_logger(module_name: str, 
                        prefix: str = "",
                        extra_context: Dict = None) -> logging.Logger:
    """모듈별 특화 로거 생성
    
    Args:
        module_name: 모듈 이름
        prefix: 로그 메시지 앞에 붙을 접두사
        extra_context: 추가 컨텍스트 정보
        
    Returns:
        logging.Logger: 특화된 로거
    """
    
    logger = get_logger(module_name)
    
    if prefix or extra_context:
        # 커스텀 어댑터 사용
        logger = logging.LoggerAdapter(logger, extra_context or {})
        if prefix:
            original_process = logger.process
            def process(msg, kwargs):
                return f"[{prefix}] {msg}", kwargs
            logger.process = process
    
    return logger

# 주요 모듈별 전용 로거들
def get_stt_logger() -> logging.Logger:
    """STT 모듈 전용 로거"""
    return create_module_logger('stt', '🎙️STT')

def get_ocr_logger() -> logging.Logger:
    """OCR 모듈 전용 로거"""
    return create_module_logger('ocr', '📸OCR')

def get_ai_logger() -> logging.Logger:
    """AI 분석 모듈 전용 로거"""
    return create_module_logger('ai_analysis', '🧠AI')

def get_ui_logger() -> logging.Logger:
    """UI 모듈 전용 로거"""
    return create_module_logger('ui', '🖥️UI')

def get_server_logger() -> logging.Logger:
    """서버 모듈 전용 로거"""
    return create_module_logger('server', '🌐SERVER')

# 레거시 호환성을 위한 함수들
def setup_logging(level: str = "INFO") -> logging.Logger:
    """레거시 호환용 - 기존 코드와의 호환성 유지"""
    SolomondLogger.initialize(log_level=level)
    return get_logger('legacy')

def logger_with_context(context: str) -> logging.Logger:
    """컨텍스트가 있는 로거 - 레거시 호환용"""
    return create_module_logger('context', context)

# 환경 변수를 통한 로그 레벨 설정 지원
if os.getenv('SOLOMOND_LOG_LEVEL'):
    SolomondLogger.initialize(log_level=os.getenv('SOLOMOND_LOG_LEVEL'))

# 사용 예시 (주석으로 남겨둠)
"""
사용 예시:

# 기본 사용법 (가장 일반적)
from utils.logger import get_logger
logger = get_logger(__name__)
logger.info("분석 시작")

# 모듈별 특화 로거
from utils.logger import get_stt_logger, get_ai_logger
stt_logger = get_stt_logger()
ai_logger = get_ai_logger()

# 레거시 코드 호환성
from utils.logger import setup_logging
logger = setup_logging("DEBUG")

# 환경 변수로 로그 레벨 제어
# 환경 변수: SOLOMOND_LOG_LEVEL=DEBUG
"""