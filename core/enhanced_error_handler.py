#!/usr/bin/env python3
"""
강화된 에러 처리 시스템
사용자 친화적 메시지, 자동 복구, 해결방안 제시
"""

import os
import traceback
from utils.logger import get_logger
import sys
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from pathlib import Path
import json

class ErrorSeverity:
    """에러 심각도 레벨"""
    INFO = "info"
    WARNING = "warning"  
    ERROR = "error"
    CRITICAL = "critical"

class ErrorCode:
    """에러 코드 상수"""
    # 파일 관련
    FILE_NOT_FOUND = "FILE_001"
    FILE_PERMISSION = "FILE_002"
    FILE_CORRUPTED = "FILE_003"
    FILE_TOO_LARGE = "FILE_004"
    FILE_EMPTY = "FILE_005"
    
    # 오디오 관련
    AUDIO_CONVERSION_FAILED = "AUDIO_001"
    AUDIO_FORMAT_UNSUPPORTED = "AUDIO_002"
    AUDIO_CODEC_ERROR = "AUDIO_003"
    AUDIO_DURATION_INVALID = "AUDIO_004"
    
    # AI 모델 관련
    MODEL_LOAD_FAILED = "MODEL_001"
    MODEL_INFERENCE_FAILED = "MODEL_002"
    MODEL_OUT_OF_MEMORY = "MODEL_003"
    MODEL_TIMEOUT = "MODEL_004"
    
    # 시스템 관련
    DEPENDENCY_MISSING = "SYSTEM_001"
    MEMORY_INSUFFICIENT = "SYSTEM_002"
    DISK_SPACE_LOW = "SYSTEM_003"
    NETWORK_ERROR = "SYSTEM_004"

class EnhancedError(Exception):
    """강화된 에러 클래스"""
    
    def __init__(self, 
                 code: str, 
                 message: str, 
                 severity: str = ErrorSeverity.ERROR,
                 context: Dict[str, Any] = None,
                 solutions: List[str] = None,
                 auto_recovery: Callable = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.severity = severity
        self.context = context or {}
        self.solutions = solutions or []
        self.auto_recovery = auto_recovery
        self.timestamp = datetime.now()

class EnhancedErrorHandler:
    """강화된 에러 처리기"""
    
    def __init__(self, log_file: str = None):
        self.logger = self._setup_logging(log_file)
        self.error_history = []
        self.auto_recovery_enabled = True
        
        # 에러별 해결방안 데이터베이스
        self.solution_database = self._build_solution_database()
    
    def _setup_logging(self, log_file: str = None):
        """로깅 설정"""
        return get_logger(f'{__name__}.EnhancedErrorHandler')
            logger.addHandler(console_handler)
            
            # 파일 핸들러 (선택적)
            if log_file:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s - Context: %(context)s'
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _build_solution_database(self) -> Dict[str, Dict[str, Any]]:
        """에러별 해결방안 데이터베이스 구축"""
        return {
            ErrorCode.FILE_NOT_FOUND: {
                "user_message": "📁 파일을 찾을 수 없습니다",
                "solutions": [
                    "파일 경로가 정확한지 확인해주세요",
                    "파일이 다른 위치로 이동되었는지 확인해주세요", 
                    "파일명에 특수문자가 있는지 확인해주세요",
                    "다른 파일을 선택하여 다시 시도해주세요"
                ],
                "auto_recovery": self._recover_file_not_found
            },
            
            ErrorCode.AUDIO_CONVERSION_FAILED: {
                "user_message": "🎵 오디오 변환에 실패했습니다",
                "solutions": [
                    "파일이 손상되지 않았는지 확인해주세요",
                    "지원하는 오디오 형식(WAV, MP3, M4A)인지 확인해주세요",
                    "파일 크기가 너무 크지 않은지 확인해주세요 (권장: 100MB 이하)",
                    "다른 오디오 파일로 테스트해주세요"
                ],
                "auto_recovery": self._recover_audio_conversion
            },
            
            ErrorCode.MODEL_LOAD_FAILED: {
                "user_message": "🧠 AI 모델 로딩에 실패했습니다",
                "solutions": [
                    "인터넷 연결을 확인해주세요 (모델 다운로드 필요)",
                    "디스크 여유 공간을 확인해주세요 (최소 5GB 필요)",
                    "메모리 여유 공간을 확인해주세요 (최소 4GB RAM 필요)",
                    "애플리케이션을 재시작해주세요"
                ],
                "auto_recovery": self._recover_model_load
            },
            
            ErrorCode.MODEL_OUT_OF_MEMORY: {
                "user_message": "💾 메모리가 부족합니다",
                "solutions": [
                    "다른 애플리케이션을 종료하여 메모리를 확보해주세요",
                    "파일을 더 작은 크기로 분할해보세요",
                    "시스템을 재부팅하여 메모리를 초기화해주세요",
                    "더 작은 AI 모델을 사용해보세요"
                ],
                "auto_recovery": self._recover_out_of_memory
            },
            
            ErrorCode.DEPENDENCY_MISSING: {
                "user_message": "🔧 필요한 소프트웨어가 설치되지 않았습니다",
                "solutions": [
                    "requirements.txt 파일을 사용하여 의존성을 설치해주세요",
                    "FFmpeg가 설치되었는지 확인해주세요",
                    "Python 패키지를 업데이트해주세요: pip install --upgrade -r requirements.txt"
                ],
                "auto_recovery": self._recover_missing_dependency
            }
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """에러 처리 메인 메서드"""
        
        # 에러 분류
        if isinstance(error, EnhancedError):
            enhanced_error = error
        else:
            enhanced_error = self._classify_error(error, context)
        
        # 에러 기록
        error_record = self._record_error(enhanced_error, context)
        
        # 사용자 친화적 메시지 생성
        user_message = self._generate_user_message(enhanced_error)
        
        # 자동 복구 시도
        recovery_result = None
        if self.auto_recovery_enabled and enhanced_error.auto_recovery:
            recovery_result = self._attempt_auto_recovery(enhanced_error)
        
        # 결과 반환
        result = {
            "success": False,
            "error_code": enhanced_error.code,
            "severity": enhanced_error.severity,
            "user_message": user_message,
            "solutions": self._get_solutions(enhanced_error.code),
            "technical_details": str(enhanced_error),
            "context": enhanced_error.context,
            "timestamp": enhanced_error.timestamp.isoformat(),
            "recovery_attempted": recovery_result is not None,
            "recovery_success": recovery_result.get("success", False) if recovery_result else False
        }
        
        # 로깅
        self._log_error(enhanced_error, result)
        
        return result
    
    def _classify_error(self, error: Exception, context: Dict[str, Any] = None) -> EnhancedError:
        """일반 에러를 EnhancedError로 분류"""
        error_str = str(error)
        error_type = type(error).__name__
        
        # 파일 관련 에러
        if "FileNotFoundError" in error_type or "No such file" in error_str:
            return EnhancedError(
                ErrorCode.FILE_NOT_FOUND,
                f"파일을 찾을 수 없습니다: {error_str}",
                ErrorSeverity.ERROR,
                context
            )
        
        elif "PermissionError" in error_type:
            return EnhancedError(
                ErrorCode.FILE_PERMISSION,
                f"파일 접근 권한이 없습니다: {error_str}",
                ErrorSeverity.ERROR,
                context
            )
        
        # 메모리 관련 에러
        elif "OutOfMemoryError" in error_type or "out of memory" in error_str.lower():
            return EnhancedError(
                ErrorCode.MODEL_OUT_OF_MEMORY,
                f"메모리가 부족합니다: {error_str}",
                ErrorSeverity.CRITICAL,
                context
            )
        
        # 모델 관련 에러
        elif "model" in error_str.lower() and ("load" in error_str.lower() or "download" in error_str.lower()):
            return EnhancedError(
                ErrorCode.MODEL_LOAD_FAILED,
                f"모델 로딩 실패: {error_str}",
                ErrorSeverity.ERROR,
                context
            )
        
        # 오디오 관련 에러  
        elif "ffmpeg" in error_str.lower() or "audio" in error_str.lower():
            return EnhancedError(
                ErrorCode.AUDIO_CONVERSION_FAILED,
                f"오디오 처리 실패: {error_str}",
                ErrorSeverity.ERROR,
                context
            )
        
        # 기타 에러
        else:
            return EnhancedError(
                "UNKNOWN_001",
                f"알 수 없는 에러: {error_str}",
                ErrorSeverity.ERROR,
                context
            )
    
    def _record_error(self, error: EnhancedError, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """에러 기록"""
        record = {
            "timestamp": error.timestamp.isoformat(),
            "code": error.code,
            "message": error.message,
            "severity": error.severity,
            "context": error.context,
            "traceback": traceback.format_exc()
        }
        
        self.error_history.append(record)
        
        # 최근 100개 에러만 유지
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
        
        return record
    
    def _generate_user_message(self, error: EnhancedError) -> str:
        """사용자 친화적 메시지 생성"""
        solution_info = self.solution_database.get(error.code)
        
        if solution_info:
            return solution_info["user_message"]
        else:
            return f"⚠️ 처리 중 문제가 발생했습니다 ({error.code})"
    
    def _get_solutions(self, error_code: str) -> List[str]:
        """해결방안 목록 반환"""
        solution_info = self.solution_database.get(error_code)
        return solution_info.get("solutions", []) if solution_info else []
    
    def _attempt_auto_recovery(self, error: EnhancedError) -> Dict[str, Any]:
        """자동 복구 시도"""
        try:
            if error.auto_recovery:
                return error.auto_recovery(error)
            else:
                solution_info = self.solution_database.get(error.code)
                if solution_info and solution_info.get("auto_recovery"):
                    return solution_info["auto_recovery"](error)
        except Exception as e:
            self.logger.warning(f"자동 복구 실패: {e}")
        
        return {"success": False, "message": "자동 복구 불가능"}
    
    def _log_error(self, error: EnhancedError, result: Dict[str, Any]):
        """에러 로깅"""
        log_level = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error.severity, logging.ERROR)
        
        self.logger.log(
            log_level,
            f"[{error.code}] {error.message}",
            extra={"context": json.dumps(error.context, ensure_ascii=False)}
        )
    
    # 자동 복구 메서드들
    def _recover_file_not_found(self, error: EnhancedError) -> Dict[str, Any]:
        """파일 없음 에러 자동 복구"""
        # 유사한 파일명 찾기 시도
        context = error.context
        if "file_path" in context:
            file_path = Path(context["file_path"])
            parent_dir = file_path.parent
            
            if parent_dir.exists():
                similar_files = []
                for file in parent_dir.iterdir():
                    if file.stem.lower() in file_path.stem.lower():
                        similar_files.append(str(file))
                
                if similar_files:
                    return {
                        "success": True,
                        "message": "유사한 파일을 찾았습니다",
                        "suggested_files": similar_files[:5]
                    }
        
        return {"success": False, "message": "대체 파일을 찾을 수 없습니다"}
    
    def _recover_audio_conversion(self, error: EnhancedError) -> Dict[str, Any]:
        """오디오 변환 에러 자동 복구"""
        # 다른 변환 방법 시도
        return {"success": False, "message": "다른 변환 방법을 시도하세요"}
    
    def _recover_model_load(self, error: EnhancedError) -> Dict[str, Any]:
        """모델 로드 에러 자동 복구"""
        # CPU 모드로 전환 시도
        return {"success": False, "message": "CPU 모드로 전환을 시도하세요"}
    
    def _recover_out_of_memory(self, error: EnhancedError) -> Dict[str, Any]:
        """메모리 부족 에러 자동 복구"""
        # 가비지 컬렉션 강제 실행
        import gc
        gc.collect()
        
        return {"success": True, "message": "메모리 정리를 수행했습니다"}
    
    def _recover_missing_dependency(self, error: EnhancedError) -> Dict[str, Any]:
        """의존성 누락 에러 자동 복구"""
        return {"success": False, "message": "수동으로 의존성을 설치하세요"}
    
    def get_error_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """에러 히스토리 반환"""
        return self.error_history[-limit:] if self.error_history else []
    
    def clear_error_history(self):
        """에러 히스토리 초기화"""
        self.error_history.clear()

# 전역 에러 핸들러 인스턴스
global_error_handler = EnhancedErrorHandler()

def handle_error(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """전역 에러 처리 함수"""
    return global_error_handler.handle_error(error, context)

def create_enhanced_error(code: str, message: str, severity: str = ErrorSeverity.ERROR,
                         context: Dict[str, Any] = None, solutions: List[str] = None) -> EnhancedError:
    """EnhancedError 생성 헬퍼 함수"""
    return EnhancedError(code, message, severity, context, solutions)

if __name__ == "__main__":
    # 테스트 코드
    handler = EnhancedErrorHandler()
    
    # 테스트 에러들
    test_errors = [
        FileNotFoundError("test.txt not found"),
        PermissionError("Permission denied"),
        Exception("Out of memory"),
        Exception("Model loading failed")
    ]
    
    for error in test_errors:
        result = handler.handle_error(error, {"test": True})
        print(f"Error: {result['error_code']}")
        print(f"Message: {result['user_message']}")
        print(f"Solutions: {result['solutions']}")
        print("-" * 50)