"""
🛡️ 솔로몬드 AI v2.1.2 - 에러 복구 시스템
자동 복구, 데이터 보호 및 시스템 안정성 보장

주요 기능:
- 자동 에러 감지 및 복구
- 진행 상태 저장 및 복원
- 안전한 데이터 백업
- 재시도 로직 및 회로 차단기
- 시스템 건강 상태 모니터링
"""

import os
import sys
import json
import time
import pickle
import logging
import threading
import traceback
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile
import shutil
import hashlib
from enum import Enum
from collections import defaultdict, deque
import asyncio
import signal

class ErrorLevel(Enum):
    """에러 심각도 레벨"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"

class RecoveryAction(Enum):
    """복구 액션 타입"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    RESTART = "restart"

@dataclass
class ErrorEvent:
    """에러 이벤트"""
    timestamp: str
    error_id: str
    level: ErrorLevel
    module: str
    function: str
    message: str
    exception_type: str
    stack_trace: str
    context: Dict[str, Any]
    recovery_action: Optional[RecoveryAction] = None
    resolved: bool = False

@dataclass
class ProgressCheckpoint:
    """진행 상태 체크포인트"""
    checkpoint_id: str
    timestamp: str
    module: str
    operation: str
    progress_percent: float
    data: Dict[str, Any]
    file_paths: List[str]

class CircuitBreaker:
    """회로 차단기 패턴"""
    
    def __init__(self, failure_threshold: int = 5, 
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def call(self, func, *args, **kwargs):
        """함수 호출 with 회로 차단기"""
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise
    
    def _on_success(self):
        """성공 시 처리"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """실패 시 처리"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"회로 차단기 OPEN: {self.failure_count}회 연속 실패")
    
    def _should_attempt_reset(self) -> bool:
        """리셋 시도 가능 여부"""
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

class RetryManager:
    """재시도 관리자"""
    
    def __init__(self, max_retries: int = 3, 
                 backoff_factor: float = 2.0,
                 max_delay: float = 60.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.logger = logging.getLogger(__name__)
    
    def retry_with_backoff(self, func: Callable, 
                          exceptions: tuple = (Exception,),
                          *args, **kwargs) -> Any:
        """지수 백오프로 재시도"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except exceptions as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    self.logger.error(f"최대 재시도 횟수 초과: {func.__name__}")
                    break
                
                delay = min(self.backoff_factor ** attempt, self.max_delay)
                self.logger.warning(f"재시도 {attempt + 1}/{self.max_retries} "
                                  f"- {delay:.1f}초 후 재시도: {str(e)}")
                time.sleep(delay)
        
        raise last_exception

class DataBackupManager:
    """데이터 백업 관리자"""
    
    def __init__(self, backup_dir: str = None):
        self.backup_dir = Path(backup_dir or tempfile.gettempdir()) / "solomond_backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.max_backups = 10
        self.logger = logging.getLogger(__name__)
    
    def create_backup(self, data: Any, backup_id: str) -> str:
        """데이터 백업 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{backup_id}_{timestamp}.backup"
        backup_path = self.backup_dir / backup_filename
        
        try:
            with open(backup_path, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'timestamp': timestamp,
                    'backup_id': backup_id,
                    'metadata': {
                        'size': sys.getsizeof(data),
                        'type': type(data).__name__
                    }
                }, f)
            
            # 오래된 백업 정리
            self._cleanup_old_backups(backup_id)
            
            self.logger.info(f"백업 생성 완료: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"백업 생성 실패: {e}")
            raise
    
    def restore_backup(self, backup_path: str) -> Any:
        """백업 복원"""
        try:
            with open(backup_path, 'rb') as f:
                backup_data = pickle.load(f)
            
            self.logger.info(f"백업 복원 완료: {backup_path}")
            return backup_data['data']
            
        except Exception as e:
            self.logger.error(f"백업 복원 실패: {e}")
            raise
    
    def list_backups(self, backup_id: str = None) -> List[Dict[str, Any]]:
        """백업 목록 조회"""
        backups = []
        
        for backup_file in self.backup_dir.glob("*.backup"):
            if backup_id and not backup_file.name.startswith(backup_id):
                continue
            
            try:
                stat = backup_file.stat()
                backups.append({
                    'path': str(backup_file),
                    'name': backup_file.name,
                    'size_mb': stat.st_size / (1024 * 1024),
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception as e:
                self.logger.debug(f"백업 파일 정보 읽기 실패: {backup_file} - {e}")
        
        return sorted(backups, key=lambda x: x['modified'], reverse=True)
    
    def _cleanup_old_backups(self, backup_id: str):
        """오래된 백업 정리"""
        backups = self.list_backups(backup_id)
        
        if len(backups) > self.max_backups:
            for backup in backups[self.max_backups:]:
                try:
                    os.remove(backup['path'])
                    self.logger.debug(f"오래된 백업 삭제: {backup['name']}")
                except Exception as e:
                    self.logger.debug(f"백업 삭제 실패: {backup['name']} - {e}")

class ProgressManager:
    """진행 상태 관리자"""
    
    def __init__(self, checkpoint_dir: str = None):
        self.checkpoint_dir = Path(checkpoint_dir or tempfile.gettempdir()) / "solomond_checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoints = {}
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, checkpoint: ProgressCheckpoint) -> str:
        """체크포인트 저장"""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.checkpoint"
        
        try:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(checkpoint), f, indent=2, ensure_ascii=False)
            
            self.checkpoints[checkpoint.checkpoint_id] = checkpoint
            self.logger.debug(f"체크포인트 저장: {checkpoint.checkpoint_id}")
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"체크포인트 저장 실패: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[ProgressCheckpoint]:
        """체크포인트 로드"""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.checkpoint"
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            checkpoint = ProgressCheckpoint(**data)
            self.checkpoints[checkpoint_id] = checkpoint
            self.logger.debug(f"체크포인트 로드: {checkpoint_id}")
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"체크포인트 로드 실패: {e}")
            return None
    
    def list_checkpoints(self) -> List[str]:
        """체크포인트 목록"""
        return list(self.checkpoints.keys())
    
    def cleanup_checkpoint(self, checkpoint_id: str):
        """체크포인트 정리"""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.checkpoint"
        
        try:
            if checkpoint_path.exists():
                os.remove(checkpoint_path)
            
            if checkpoint_id in self.checkpoints:
                del self.checkpoints[checkpoint_id]
            
            self.logger.debug(f"체크포인트 정리: {checkpoint_id}")
            
        except Exception as e:
            self.logger.error(f"체크포인트 정리 실패: {e}")

class ErrorRecoverySystem:
    """통합 에러 복구 시스템"""
    
    def __init__(self, 
                 backup_dir: str = None,
                 checkpoint_dir: str = None,
                 log_level: str = "INFO"):
        
        # 컴포넌트 초기화
        self.backup_manager = DataBackupManager(backup_dir)
        self.progress_manager = ProgressManager(checkpoint_dir)
        self.retry_manager = RetryManager()
        
        # 에러 추적
        self.error_history = deque(maxlen=1000)
        self.circuit_breakers = {}
        self.error_patterns = defaultdict(int)
        
        # 복구 전략
        self.recovery_strategies = {}
        self.fallback_functions = {}
        
        # 시스템 상태
        self.system_health = "HEALTHY"
        self.active_operations = {}
        
        # 로깅 설정
        self.logger = self._setup_logger(log_level)
        
        # 신호 핸들러 등록
        self._setup_signal_handlers()
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_signal_handlers(self):
        """시스템 신호 핸들러 설정"""
        def signal_handler(signum, frame):
            self.logger.warning(f"시스템 신호 수신: {signum}")
            self._emergency_shutdown()
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except Exception as e:
            self.logger.debug(f"신호 핸들러 설정 실패: {e}")
    
    def register_recovery_strategy(self, error_pattern: str, 
                                  strategy: RecoveryAction,
                                  fallback_func: Callable = None):
        """복구 전략 등록"""
        self.recovery_strategies[error_pattern] = strategy
        if fallback_func:
            self.fallback_functions[error_pattern] = fallback_func
        
        self.logger.info(f"복구 전략 등록: {error_pattern} -> {strategy.value}")
    
    def resilient_execution(self, operation_id: str = None,
                          auto_backup: bool = True,
                          checkpoint_interval: int = 100):
        """복구 가능한 실행 데코레이터"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                op_id = operation_id or f"{func.__module__}.{func.__name__}"
                
                # 진행 중인 작업 등록
                self.active_operations[op_id] = {
                    'start_time': time.time(),
                    'function': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                }
                
                try:
                    # 이전 체크포인트 확인
                    checkpoint = self.progress_manager.load_checkpoint(op_id)
                    if checkpoint:
                        self.logger.info(f"이전 체크포인트 발견: {op_id}")
                        # 체크포인트에서 복원 로직 구현 가능
                    
                    # 백업 생성
                    if auto_backup and args:
                        try:
                            self.backup_manager.create_backup(args[0], op_id)
                        except Exception as e:
                            self.logger.debug(f"백업 생성 실패: {e}")
                    
                    # 함수 실행
                    result = func(*args, **kwargs)
                    
                    # 성공 시 체크포인트 정리
                    self.progress_manager.cleanup_checkpoint(op_id)
                    
                    return result
                    
                except Exception as e:
                    # 에러 처리
                    error_event = self._create_error_event(e, func, op_id)
                    recovery_action = self._determine_recovery_action(error_event)
                    
                    if recovery_action == RecoveryAction.RETRY:
                        return self.retry_manager.retry_with_backoff(
                            func, (type(e),), *args, **kwargs
                        )
                    elif recovery_action == RecoveryAction.FALLBACK:
                        return self._execute_fallback(error_event, *args, **kwargs)
                    else:
                        raise
                
                finally:
                    # 작업 완료 처리
                    if op_id in self.active_operations:
                        del self.active_operations[op_id]
            
            return wrapper
        return decorator
    
    def _create_error_event(self, exception: Exception, func: Callable, 
                          operation_id: str) -> ErrorEvent:
        """에러 이벤트 생성"""
        error_id = hashlib.md5(f"{func.__name__}{str(exception)}{time.time()}".encode()).hexdigest()[:8]
        
        # 심각도 판단
        level = ErrorLevel.ERROR
        if isinstance(exception, (MemoryError, SystemExit)):
            level = ErrorLevel.FATAL
        elif isinstance(exception, (FileNotFoundError, ConnectionError)):
            level = ErrorLevel.WARNING
        
        error_event = ErrorEvent(
            timestamp=datetime.now().isoformat(),
            error_id=error_id,
            level=level,
            module=func.__module__,
            function=func.__name__,
            message=str(exception),
            exception_type=type(exception).__name__,
            stack_trace=traceback.format_exc(),
            context={'operation_id': operation_id}
        )
        
        self.error_history.append(error_event)
        self.error_patterns[type(exception).__name__] += 1
        
        self.logger.error(f"에러 발생: {error_event.error_id} - {error_event.message}")
        
        return error_event
    
    def _determine_recovery_action(self, error_event: ErrorEvent) -> RecoveryAction:
        """복구 액션 결정"""
        exception_type = error_event.exception_type
        
        # 등록된 전략 확인
        for pattern, strategy in self.recovery_strategies.items():
            if pattern in exception_type.lower():
                error_event.recovery_action = strategy
                return strategy
        
        # 기본 전략
        if exception_type in ['ConnectionError', 'TimeoutError', 'HTTPError']:
            return RecoveryAction.RETRY
        elif exception_type in ['FileNotFoundError', 'PermissionError']:
            return RecoveryAction.FALLBACK
        elif exception_type in ['MemoryError', 'SystemExit']:
            return RecoveryAction.ABORT
        else:
            return RecoveryAction.RETRY
    
    def _execute_fallback(self, error_event: ErrorEvent, *args, **kwargs) -> Any:
        """폴백 실행"""
        exception_type = error_event.exception_type
        
        for pattern, fallback_func in self.fallback_functions.items():
            if pattern in exception_type.lower():
                self.logger.info(f"폴백 실행: {pattern}")
                return fallback_func(*args, **kwargs)
        
        # 기본 폴백
        self.logger.warning("기본 폴백 실행: None 반환")
        return None
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """회로 차단기 가져오기/생성"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        recent_errors = [e for e in self.error_history 
                        if datetime.fromisoformat(e.timestamp) > 
                        datetime.now() - timedelta(hours=1)]
        
        error_rate = len(recent_errors) / max(len(self.error_history), 1) * 100
        
        # 시스템 건강도 계산
        if error_rate > 50:
            health_status = "CRITICAL"
        elif error_rate > 20:
            health_status = "WARNING"
        elif len(self.active_operations) > 10:
            health_status = "BUSY"
        else:
            health_status = "HEALTHY"
        
        return {
            "health_status": health_status,
            "error_rate_1h": round(error_rate, 2),
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "active_operations": len(self.active_operations),
            "circuit_breakers": {
                name: breaker.state 
                for name, breaker in self.circuit_breakers.items()
            },
            "top_error_types": dict(list(self.error_patterns.items())[:5]),
            "backups_available": len(self.backup_manager.list_backups()),
            "checkpoints_active": len(self.progress_manager.list_checkpoints())
        }
    
    def _emergency_shutdown(self):
        """긴급 종료 처리"""
        self.logger.critical("긴급 종료 시작")
        
        # 진행 중인 작업의 체크포인트 저장
        for op_id, operation in self.active_operations.items():
            try:
                checkpoint = ProgressCheckpoint(
                    checkpoint_id=op_id,
                    timestamp=datetime.now().isoformat(),
                    module="emergency",
                    operation=operation['function'],
                    progress_percent=50.0,  # 추정값
                    data=operation,
                    file_paths=[]
                )
                self.progress_manager.save_checkpoint(checkpoint)
                self.logger.info(f"긴급 체크포인트 저장: {op_id}")
            except Exception as e:
                self.logger.error(f"긴급 체크포인트 저장 실패: {op_id} - {e}")
        
        self.logger.critical("긴급 종료 완료")
    
    def generate_recovery_report(self) -> Dict[str, Any]:
        """복구 리포트 생성"""
        if not self.error_history:
            return {"message": "에러 기록이 없습니다."}
        
        # 최근 24시간 에러 분석
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_errors = [
            e for e in self.error_history 
            if datetime.fromisoformat(e.timestamp) > recent_cutoff
        ]
        
        # 에러 패턴 분석
        error_by_level = defaultdict(int)
        error_by_module = defaultdict(int)
        resolved_count = 0
        
        for error in recent_errors:
            error_by_level[error.level.value] += 1
            error_by_module[error.module] += 1
            if error.resolved:
                resolved_count += 1
        
        # 복구 성공률
        recovery_rate = (resolved_count / len(recent_errors) * 100) if recent_errors else 0
        
        return {
            "report_generated": datetime.now().isoformat(),
            "summary": {
                "total_errors_24h": len(recent_errors),
                "recovery_success_rate": round(recovery_rate, 2),
                "system_health": self.get_system_status()["health_status"]
            },
            "error_analysis": {
                "by_level": dict(error_by_level),
                "by_module": dict(error_by_module),
                "top_patterns": dict(list(self.error_patterns.items())[:10])
            },
            "recovery_stats": {
                "strategies_registered": len(self.recovery_strategies),
                "fallback_functions": len(self.fallback_functions),
                "circuit_breakers": len(self.circuit_breakers)
            },
            "recommendations": self._generate_recovery_recommendations(recent_errors)
        }
    
    def _generate_recovery_recommendations(self, recent_errors: List[ErrorEvent]) -> List[str]:
        """복구 권장사항 생성"""
        recommendations = []
        
        if len(recent_errors) > 50:
            recommendations.append("🚨 에러 발생률이 높습니다. 시스템 점검이 필요합니다.")
        
        # 반복되는 에러 패턴 확인
        pattern_counts = defaultdict(int)
        for error in recent_errors[-20:]:  # 최근 20개
            pattern_counts[error.exception_type] += 1
        
        for pattern, count in pattern_counts.items():
            if count > 5:
                recommendations.append(f"⚠️ {pattern} 에러가 반복됩니다. 근본 원인 분석이 필요합니다.")
        
        # 회로 차단기 상태 확인
        open_breakers = [name for name, breaker in self.circuit_breakers.items() 
                        if breaker.state == "OPEN"]
        if open_breakers:
            recommendations.append(f"🔧 회로 차단기가 열려있습니다: {', '.join(open_breakers)}")
        
        if not recommendations:
            recommendations.append("✅ 시스템이 안정적으로 운영되고 있습니다.")
        
        return recommendations

# 전역 복구 시스템 인스턴스
global_recovery_system = ErrorRecoverySystem()

def resilient(operation_id: str = None, auto_backup: bool = True):
    """복구 가능한 실행 데코레이터 (간편 사용)"""
    return global_recovery_system.resilient_execution(operation_id, auto_backup)

def with_circuit_breaker(service_name: str):
    """회로 차단기 데코레이터"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            breaker = global_recovery_system.get_circuit_breaker(service_name)
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # 테스트 실행
    print("🛡️ 솔로몬드 AI 에러 복구 시스템 v2.1.2")
    print("=" * 50)
    
    recovery_system = ErrorRecoverySystem()
    
    # 복구 전략 등록
    def file_fallback(*args, **kwargs):
        return "파일을 찾을 수 없어 기본값을 반환합니다."
    
    recovery_system.register_recovery_strategy(
        "filenotfound", 
        RecoveryAction.FALLBACK, 
        file_fallback
    )
    
    # 테스트 함수들
    @recovery_system.resilient_execution("test_operation")
    def test_file_operation():
        """파일 작업 테스트"""
        raise FileNotFoundError("테스트 파일을 찾을 수 없습니다")
    
    @with_circuit_breaker("external_service")
    def test_external_call():
        """외부 서비스 호출 테스트"""
        import random
        if random.random() < 0.7:  # 70% 확률로 실패
            raise ConnectionError("외부 서비스 연결 실패")
        return "외부 서비스 응답"
    
    # 테스트 실행
    print("\n🧪 복구 시스템 테스트...")
    
    # 파일 작업 테스트 (폴백 실행)
    try:
        result = test_file_operation()
        print(f"파일 작업 결과: {result}")
    except Exception as e:
        print(f"파일 작업 실패: {e}")
    
    # 회로 차단기 테스트
    print("\n🔌 회로 차단기 테스트...")
    success_count = 0
    for i in range(10):
        try:
            result = test_external_call()
            success_count += 1
            print(f"외부 호출 {i+1}: 성공")
        except Exception as e:
            print(f"외부 호출 {i+1}: 실패 - {str(e)[:50]}")
    
    print(f"성공률: {success_count}/10")
    
    # 시스템 상태 확인
    print("\n📊 시스템 상태:")
    status = recovery_system.get_system_status()
    print(f"건강 상태: {status['health_status']}")
    print(f"에러율 (1시간): {status['error_rate_1h']}%")
    print(f"총 에러 수: {status['total_errors']}")
    print(f"활성 작업 수: {status['active_operations']}")
    
    # 복구 리포트
    print("\n📋 복구 리포트:")
    report = recovery_system.generate_recovery_report()
    print(f"복구 성공률: {report['summary']['recovery_success_rate']}%")
    print("권장사항:")
    for rec in report['recommendations']:
        print(f"  {rec}")
    
    print("\n✅ 에러 복구 시스템 테스트 완료!")
