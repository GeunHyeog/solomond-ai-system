# Phase 2 Week 3 Day 2: 에러 복구 시스템
# 체크포인트 기반 복구 + 자동 재시도 + 상세 로깅

import asyncio
import json
import pickle
import time
import traceback
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import hashlib
import sqlite3
import logging
from datetime import datetime, timedelta
import psutil

class RecoveryLevel(Enum):
    """복구 레벨"""
    NONE = "none"           # 복구 없음
    BASIC = "basic"         # 기본 재시도
    CHECKPOINT = "checkpoint"  # 체크포인트 복구
    FULL = "full"           # 완전 복구

class ErrorSeverity(Enum):
    """오류 심각도"""
    LOW = "low"             # 경고
    MEDIUM = "medium"       # 중간
    HIGH = "high"           # 높음
    CRITICAL = "critical"   # 치명적

@dataclass
class ErrorInfo:
    """오류 정보"""
    error_id: str
    timestamp: float
    error_type: str
    error_message: str
    stack_trace: str
    severity: ErrorSeverity
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    resolved: bool = False

@dataclass
class CheckpointData:
    """체크포인트 데이터"""
    checkpoint_id: str
    session_id: str
    timestamp: float
    progress_percent: float
    completed_chunks: List[str]
    failed_chunks: List[str]
    processing_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecoveryConfig:
    """복구 설정"""
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    checkpoint_interval: int = 10  # 초
    auto_recovery_enabled: bool = True
    recovery_level: RecoveryLevel = RecoveryLevel.FULL
    log_retention_days: int = 30
    max_error_history: int = 1000

class ErrorLogger:
    """향상된 오류 로깅 시스템"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 로그 파일 설정
        self.error_log_file = self.log_dir / "errors.log"
        self.recovery_log_file = self.log_dir / "recovery.log"
        self.performance_log_file = self.log_dir / "performance.log"
        
        # 로거 설정
        self._setup_loggers()
        
        # 에러 히스토리 (메모리)
        self.error_history: List[ErrorInfo] = []
        self.error_stats = {
            "total_errors": 0,
            "critical_errors": 0,
            "resolved_errors": 0,
            "recovery_success_rate": 0.0
        }
    
    def _setup_loggers(self):
        """로거 설정"""
        # 에러 로거
        self.error_logger = logging.getLogger("error_logger")
        self.error_logger.setLevel(logging.ERROR)
        
        error_handler = logging.FileHandler(self.error_log_file)
        error_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        error_handler.setFormatter(error_formatter)
        self.error_logger.addHandler(error_handler)
        
        # 복구 로거
        self.recovery_logger = logging.getLogger("recovery_logger")
        self.recovery_logger.setLevel(logging.INFO)
        
        recovery_handler = logging.FileHandler(self.recovery_log_file)
        recovery_formatter = logging.Formatter(
            '%(asctime)s - RECOVERY - %(message)s'
        )
        recovery_handler.setFormatter(recovery_formatter)
        self.recovery_logger.addHandler(recovery_handler)
        
        # 성능 로거
        self.performance_logger = logging.getLogger("performance_logger")
        self.performance_logger.setLevel(logging.INFO)
        
        perf_handler = logging.FileHandler(self.performance_log_file)
        perf_formatter = logging.Formatter(
            '%(asctime)s - PERF - %(message)s'
        )
        perf_handler.setFormatter(perf_formatter)
        self.performance_logger.addHandler(perf_handler)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> str:
        """오류 로깅"""
        error_id = hashlib.md5(f"{time.time()}{str(error)}".encode()).hexdigest()[:8]
        
        error_info = ErrorInfo(
            error_id=error_id,
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            severity=self._determine_severity(error),
            context=context or {}
        )
        
        # 히스토리에 추가
        self.error_history.append(error_info)
        self._update_error_stats()
        
        # 로그 파일에 기록
        log_message = json.dumps({
            "error_id": error_id,
            "type": error_info.error_type,
            "message": error_info.error_message,
            "severity": error_info.severity.value,
            "context": error_info.context,
            "timestamp": error_info.timestamp
        })
        
        self.error_logger.error(log_message)
        
        return error_id
    
    def log_recovery_attempt(self, error_id: str, attempt: int, success: bool, details: str = ""):
        """복구 시도 로깅"""
        message = {
            "error_id": error_id,
            "attempt": attempt,
            "success": success,
            "details": details,
            "timestamp": time.time()
        }
        
        self.recovery_logger.info(json.dumps(message))
        
        # 에러 히스토리 업데이트
        for error_info in self.error_history:
            if error_info.error_id == error_id:
                error_info.recovery_attempts = attempt
                error_info.resolved = success
                break
        
        self._update_error_stats()
    
    def log_performance(self, operation: str, duration: float, details: Dict[str, Any] = None):
        """성능 로깅"""
        message = {
            "operation": operation,
            "duration_seconds": duration,
            "details": details or {},
            "timestamp": time.time(),
            "memory_usage_mb": psutil.virtual_memory().used / 1024 / 1024
        }
        
        self.performance_logger.info(json.dumps(message))
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """오류 심각도 판단"""
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (IOError, OSError, ConnectionError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _update_error_stats(self):
        """에러 통계 업데이트"""
        if not self.error_history:
            return
        
        self.error_stats["total_errors"] = len(self.error_history)
        self.error_stats["critical_errors"] = sum(
            1 for e in self.error_history if e.severity == ErrorSeverity.CRITICAL
        )
        self.error_stats["resolved_errors"] = sum(
            1 for e in self.error_history if e.resolved
        )
        
        if self.error_stats["total_errors"] > 0:
            self.error_stats["recovery_success_rate"] = (
                self.error_stats["resolved_errors"] / self.error_stats["total_errors"]
            )
    
    def get_error_report(self) -> Dict[str, Any]:
        """에러 리포트 생성"""
        recent_errors = [
            {
                "error_id": e.error_id,
                "type": e.error_type,
                "message": e.error_message,
                "severity": e.severity.value,
                "timestamp": e.timestamp,
                "resolved": e.resolved
            }
            for e in self.error_history[-10:]  # 최근 10개
        ]
        
        return {
            "statistics": self.error_stats,
            "recent_errors": recent_errors,
            "log_files": {
                "errors": str(self.error_log_file),
                "recovery": str(self.recovery_log_file),
                "performance": str(self.performance_log_file)
            }
        }

class CheckpointManager:
    """체크포인트 관리자"""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # SQLite 데이터베이스로 체크포인트 메타데이터 관리
        self.db_path = self.checkpoint_dir / "checkpoints.db"
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    progress_percent REAL NOT NULL,
                    completed_chunks TEXT NOT NULL,
                    failed_chunks TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    file_path TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_timestamp 
                ON checkpoints(session_id, timestamp)
            """)
    
    async def save_checkpoint(
        self, 
        session_id: str, 
        progress_percent: float,
        completed_chunks: List[str],
        failed_chunks: List[str],
        processing_state: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> str:
        """체크포인트 저장"""
        
        checkpoint_id = f"{session_id}_{int(time.time())}"
        
        checkpoint_data = CheckpointData(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            timestamp=time.time(),
            progress_percent=progress_percent,
            completed_chunks=completed_chunks,
            failed_chunks=failed_chunks,
            processing_state=processing_state,
            metadata=metadata or {}
        )
        
        # 체크포인트 파일 저장
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # 데이터베이스에 메타데이터 저장
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO checkpoints 
                    (checkpoint_id, session_id, timestamp, progress_percent, 
                     completed_chunks, failed_chunks, metadata, file_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    checkpoint_id,
                    session_id,
                    checkpoint_data.timestamp,
                    progress_percent,
                    json.dumps(completed_chunks),
                    json.dumps(failed_chunks),
                    json.dumps(metadata or {}),
                    str(checkpoint_file)
                ))
            
            return checkpoint_id
            
        except Exception as e:
            raise Exception(f"체크포인트 저장 실패: {e}")
    
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointData]:
        """체크포인트 로드"""
        try:
            # 데이터베이스에서 파일 경로 조회
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT file_path FROM checkpoints WHERE checkpoint_id = ?",
                    (checkpoint_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    return None
                
                checkpoint_file = Path(result[0])
            
            # 체크포인트 파일 로드
            if checkpoint_file.exists():
                with open(checkpoint_file, 'rb') as f:
                    return pickle.load(f)
            
            return None
            
        except Exception as e:
            raise Exception(f"체크포인트 로드 실패: {e}")
    
    async def get_latest_checkpoint(self, session_id: str) -> Optional[CheckpointData]:
        """최신 체크포인트 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT checkpoint_id FROM checkpoints 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (session_id,))
                
                result = cursor.fetchone()
                
                if result:
                    return await self.load_checkpoint(result[0])
                
                return None
                
        except Exception as e:
            raise Exception(f"최신 체크포인트 조회 실패: {e}")
    
    async def cleanup_old_checkpoints(self, retention_days: int = 7):
        """오래된 체크포인트 정리"""
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 삭제할 체크포인트 파일 경로 조회
                cursor = conn.execute(
                    "SELECT file_path FROM checkpoints WHERE timestamp < ?",
                    (cutoff_time,)
                )
                
                file_paths = [row[0] for row in cursor.fetchall()]
                
                # 파일 삭제
                for file_path in file_paths:
                    try:
                        Path(file_path).unlink(missing_ok=True)
                    except:
                        pass
                
                # 데이터베이스에서 레코드 삭제
                conn.execute(
                    "DELETE FROM checkpoints WHERE timestamp < ?",
                    (cutoff_time,)
                )
                
                return len(file_paths)
                
        except Exception as e:
            raise Exception(f"체크포인트 정리 실패: {e}")
    
    def get_checkpoint_history(self, session_id: str) -> List[Dict[str, Any]]:
        """체크포인트 히스토리 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT checkpoint_id, timestamp, progress_percent, 
                           completed_chunks, failed_chunks
                    FROM checkpoints 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC
                """, (session_id,))
                
                return [
                    {
                        "checkpoint_id": row[0],
                        "timestamp": row[1],
                        "progress_percent": row[2],
                        "completed_chunks": len(json.loads(row[3])),
                        "failed_chunks": len(json.loads(row[4]))
                    }
                    for row in cursor.fetchall()
                ]
                
        except Exception as e:
            return []

class RecoveryManager:
    """종합 복구 관리자"""
    
    def __init__(self, config: RecoveryConfig, work_dir: Path):
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        
        # 하위 시스템 초기화
        self.error_logger = ErrorLogger(self.work_dir / "logs")
        self.checkpoint_manager = CheckpointManager(self.work_dir / "checkpoints")
        
        # 복구 상태 추적
        self.active_recoveries: Dict[str, Dict[str, Any]] = {}
        self.recovery_callbacks: List[Callable] = []
        
        # 자동 정리 스케줄러 시작
        if self.config.auto_recovery_enabled:
            asyncio.create_task(self._cleanup_scheduler())
    
    async def execute_with_recovery(
        self,
        operation_id: str,
        operation_func: Callable,
        session_id: str = None,
        context: Dict[str, Any] = None,
        checkpoint_data: Dict[str, Any] = None
    ) -> Any:
        """복구 기능이 포함된 작업 실행"""
        
        start_time = time.time()
        attempt = 0
        last_error = None
        
        while attempt < self.config.max_retry_attempts:
            attempt += 1
            
            try:
                # 체크포인트에서 복구 시도
                if attempt > 1 and session_id and self.config.recovery_level == RecoveryLevel.CHECKPOINT:
                    await self._attempt_checkpoint_recovery(session_id, operation_func)
                
                # 작업 실행
                result = await operation_func()
                
                # 성공 시 성능 로깅
                duration = time.time() - start_time
                self.error_logger.log_performance(
                    operation=operation_id,
                    duration=duration,
                    details={"attempts": attempt, "success": True}
                )
                
                # 복구 성공 로깅 (재시도였다면)
                if attempt > 1 and last_error:
                    error_id = self.error_logger.log_error(last_error, context)
                    self.error_logger.log_recovery_attempt(
                        error_id, attempt, True, f"복구 성공: {operation_id}"
                    )
                
                return result
                
            except Exception as e:
                last_error = e
                error_id = self.error_logger.log_error(e, context)
                
                # 복구 시도 로깅
                self.error_logger.log_recovery_attempt(
                    error_id, attempt, False, f"시도 {attempt} 실패"
                )
                
                # 마지막 시도가 아니라면 대기 후 재시도
                if attempt < self.config.max_retry_attempts:
                    await asyncio.sleep(self.config.retry_delay_seconds * attempt)
                    continue
                else:
                    # 모든 시도 실패
                    duration = time.time() - start_time
                    self.error_logger.log_performance(
                        operation=operation_id,
                        duration=duration,
                        details={"attempts": attempt, "success": False, "final_error": str(e)}
                    )
                    
                    raise e
    
    async def _attempt_checkpoint_recovery(
        self, 
        session_id: str, 
        operation_func: Callable
    ):
        """체크포인트 기반 복구 시도"""
        
        try:
            latest_checkpoint = await self.checkpoint_manager.get_latest_checkpoint(session_id)
            
            if latest_checkpoint:
                # 체크포인트 데이터로 상태 복원
                if hasattr(operation_func, 'restore_from_checkpoint'):
                    await operation_func.restore_from_checkpoint(latest_checkpoint)
                
                self.error_logger.recovery_logger.info(
                    f"체크포인트 복구 적용: {latest_checkpoint.checkpoint_id}"
                )
        
        except Exception as e:
            self.error_logger.recovery_logger.warning(
                f"체크포인트 복구 실패: {e}"
            )
    
    async def save_checkpoint_auto(
        self,
        session_id: str,
        progress_percent: float,
        completed_chunks: List[str],
        failed_chunks: List[str],
        processing_state: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> str:
        """자동 체크포인트 저장"""
        
        try:
            checkpoint_id = await self.checkpoint_manager.save_checkpoint(
                session_id=session_id,
                progress_percent=progress_percent,
                completed_chunks=completed_chunks,
                failed_chunks=failed_chunks,
                processing_state=processing_state,
                metadata=metadata
            )
            
            self.error_logger.recovery_logger.info(
                f"자동 체크포인트 저장: {checkpoint_id} (진행률: {progress_percent:.1f}%)"
            )
            
            return checkpoint_id
            
        except Exception as e:
            self.error_logger.log_error(e, {"operation": "auto_checkpoint_save"})
            raise
    
    async def recover_from_failure(
        self,
        session_id: str,
        failure_point: str = None
    ) -> Optional[CheckpointData]:
        """실패 지점부터 복구"""
        
        try:
            # 최신 체크포인트 로드
            checkpoint = await self.checkpoint_manager.get_latest_checkpoint(session_id)
            
            if not checkpoint:
                self.error_logger.recovery_logger.warning(
                    f"복구 불가: 체크포인트 없음 (세션: {session_id})"
                )
                return None
            
            # 복구 정보 로깅
            self.error_logger.recovery_logger.info(
                f"복구 시작: 세션 {session_id}, 진행률 {checkpoint.progress_percent:.1f}%"
            )
            
            # 복구 콜백 실행
            for callback in self.recovery_callbacks:
                try:
                    await callback(checkpoint)
                except Exception as e:
                    self.error_logger.log_error(e, {"operation": "recovery_callback"})
            
            return checkpoint
            
        except Exception as e:
            self.error_logger.log_error(e, {"operation": "recover_from_failure"})
            return None
    
    def add_recovery_callback(self, callback: Callable):
        """복구 콜백 추가"""
        self.recovery_callbacks.append(callback)
    
    async def _cleanup_scheduler(self):
        """정리 스케줄러"""
        while True:
            try:
                # 1시간마다 정리
                await asyncio.sleep(3600)
                
                # 오래된 체크포인트 정리
                cleaned_count = await self.checkpoint_manager.cleanup_old_checkpoints(
                    self.config.log_retention_days
                )
                
                if cleaned_count > 0:
                    self.error_logger.recovery_logger.info(
                        f"자동 정리 완료: {cleaned_count}개 체크포인트 삭제"
                    )
                
            except Exception as e:
                self.error_logger.log_error(e, {"operation": "cleanup_scheduler"})
    
    def get_recovery_report(self) -> Dict[str, Any]:
        """복구 시스템 리포트"""
        error_report = self.error_logger.get_error_report()
        
        return {
            "config": {
                "max_retry_attempts": self.config.max_retry_attempts,
                "recovery_level": self.config.recovery_level.value,
                "auto_recovery_enabled": self.config.auto_recovery_enabled
            },
            "error_statistics": error_report["statistics"],
            "recent_errors": error_report["recent_errors"],
            "active_recoveries": len(self.active_recoveries),
            "log_files": error_report["log_files"]
        }

# 데코레이터: 자동 복구 기능
def with_auto_recovery(
    recovery_manager: RecoveryManager,
    operation_id: str = None,
    session_id: str = None
):
    """자동 복구 데코레이터"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            op_id = operation_id or func.__name__
            
            return await recovery_manager.execute_with_recovery(
                operation_id=op_id,
                operation_func=lambda: func(*args, **kwargs),
                session_id=session_id,
                context={"function": func.__name__, "args": len(args), "kwargs": list(kwargs.keys())}
            )
        return wrapper
    return decorator

# 사용 예시
async def demo_recovery_system():
    """복구 시스템 데모"""
    
    print("🛡️ 에러 복구 시스템 데모 시작")
    print("=" * 60)
    
    # 복구 관리자 설정
    config = RecoveryConfig(
        max_retry_attempts=3,
        retry_delay_seconds=0.5,
        checkpoint_interval=5,
        auto_recovery_enabled=True,
        recovery_level=RecoveryLevel.FULL
    )
    
    recovery_manager = RecoveryManager(config, Path("recovery_demo"))
    
    # 테스트 함수들
    test_counter = {"value": 0}
    
    @with_auto_recovery(recovery_manager, "test_operation", "demo_session")
    async def test_operation_with_failures():
        """실패하는 테스트 작업"""
        test_counter["value"] += 1
        
        if test_counter["value"] < 3:
            raise ValueError(f"의도적 실패 {test_counter['value']}")
        
        return {"result": "성공!", "attempts": test_counter["value"]}
    
    # 체크포인트 콜백 등록
    async def checkpoint_callback(checkpoint: CheckpointData):
        print(f"📍 체크포인트 복구: {checkpoint.checkpoint_id}")
    
    recovery_manager.add_recovery_callback(checkpoint_callback)
    
    try:
        # 1. 자동 복구 테스트
        print("🔄 자동 복구 테스트...")
        result = await test_operation_with_failures()
        print(f"✅ 결과: {result}")
        
        # 2. 체크포인트 저장 테스트
        print("\n📍 체크포인트 저장 테스트...")
        checkpoint_id = await recovery_manager.save_checkpoint_auto(
            session_id="demo_session",
            progress_percent=75.0,
            completed_chunks=["chunk1", "chunk2", "chunk3"],
            failed_chunks=["chunk4"],
            processing_state={"current_stage": "processing", "temp_files": []},
            metadata={"demo": True, "timestamp": time.time()}
        )
        print(f"✅ 체크포인트 저장: {checkpoint_id}")
        
        # 3. 복구 테스트
        print("\n🔧 복구 테스트...")
        recovered_checkpoint = await recovery_manager.recover_from_failure("demo_session")
        if recovered_checkpoint:
            print(f"✅ 복구 성공: 진행률 {recovered_checkpoint.progress_percent:.1f}%")
        
        # 4. 리포트 생성
        print("\n📊 복구 시스템 리포트:")
        report = recovery_manager.get_recovery_report()
        print(f"   총 오류: {report['error_statistics']['total_errors']}")
        print(f"   복구 성공률: {report['error_statistics']['recovery_success_rate']:.1%}")
        print(f"   활성 복구: {report['active_recoveries']}")
        
        # 5. 체크포인트 히스토리
        history = recovery_manager.checkpoint_manager.get_checkpoint_history("demo_session")
        print(f"\n📚 체크포인트 히스토리: {len(history)}개")
        for h in history[:3]:  # 최근 3개
            print(f"   - {h['checkpoint_id']}: {h['progress_percent']:.1f}%")
        
    except Exception as e:
        print(f"❌ 데모 실행 오류: {e}")
    
    finally:
        print("\n🧹 정리 중...")
        # 테스트 파일 정리
        import shutil
        demo_dir = Path("recovery_demo")
        if demo_dir.exists():
            shutil.rmtree(demo_dir, ignore_errors=True)
        
        print("✅ 에러 복구 시스템 데모 완료!")

if __name__ == "__main__":
    # 데모 실행
    asyncio.run(demo_recovery_system())
