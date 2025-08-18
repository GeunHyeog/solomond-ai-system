#!/usr/bin/env python3
"""
🛡️ 모듈1 오류 처리 및 메모리 관리 시스템
안정성 향상을 위한 포괄적 에러 핸들링 및 리소스 관리

업데이트: 2025-01-30 - 안정성 향상 시스템
"""

import streamlit as st
import traceback
import logging
import psutil
import gc
import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import tempfile
import shutil

class ErrorLevel(Enum):
    """오류 수준 정의"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ErrorInfo:
    """오류 정보 클래스"""
    level: ErrorLevel
    message: str
    details: str
    timestamp: float
    function_name: str
    recovery_suggestions: List[str]

class MemoryManager:
    """메모리 관리 시스템"""
    
    def __init__(self, max_memory_gb: float = 4.0):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.process = psutil.Process()
        self.temp_files = []
        self.cleanup_callbacks = []
        
    def get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량 반환"""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def check_memory_threshold(self, threshold_percent: float = 80.0) -> bool:
        """메모리 임계값 확인"""
        current_percent = self.process.memory_percent()
        return current_percent > threshold_percent
    
    def force_garbage_collection(self):
        """강제 가비지 컬렉션"""
        collected = gc.collect()
        st.sidebar.info(f"🗑️ 메모리 정리: {collected}개 객체 해제")
        return collected
    
    def register_temp_file(self, filepath: str):
        """임시 파일 등록"""
        self.temp_files.append(filepath)
    
    def register_cleanup_callback(self, callback: Callable):
        """정리 콜백 등록"""
        self.cleanup_callbacks.append(callback)
    
    def cleanup_resources(self):
        """리소스 정리"""
        # 임시 파일 삭제
        for filepath in self.temp_files:
            try:
                if os.path.exists(filepath):
                    os.unlink(filepath)
            except Exception as e:
                st.sidebar.warning(f"임시 파일 삭제 실패: {e}")
        
        # 콜백 실행
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                st.sidebar.warning(f"정리 콜백 실패: {e}")
        
        # 가비지 컬렉션
        self.force_garbage_collection()
        
        # 리스트 초기화
        self.temp_files.clear()
        self.cleanup_callbacks.clear()
    
    @contextmanager
    def memory_monitor(self, operation_name: str):
        """메모리 모니터링 컨텍스트"""
        start_memory = self.get_memory_usage()
        start_time = time.time()
        
        try:
            yield self
        finally:
            end_memory = self.get_memory_usage()
            end_time = time.time()
            
            memory_diff = end_memory['rss_mb'] - start_memory['rss_mb']
            time_diff = end_time - start_time
            
            # 메모리 증가량이 큰 경우 경고
            if memory_diff > 100:  # 100MB 이상
                st.sidebar.warning(f"⚠️ {operation_name}: 메모리 사용량 증가 {memory_diff:.1f}MB")
            
            st.sidebar.metric(
                f"💾 {operation_name}",
                f"{end_memory['rss_mb']:.1f}MB",
                f"{memory_diff:+.1f}MB"
            )

class SafeErrorHandler:
    """안전한 오류 처리 시스템"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.errors = []
        self.recovery_attempts = {}
        self.setup_logging(log_file)
        
    def setup_logging(self, log_file: Optional[str]):
        """로깅 설정"""
        if log_file:
            logging.basicConfig(
                filename=log_file,
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
        self.logger = logging.getLogger(__name__)
    
    def handle_error(self, error: Exception, function_name: str, 
                    recovery_suggestions: List[str] = None) -> ErrorInfo:
        """오류 처리"""
        error_info = ErrorInfo(
            level=self._determine_error_level(error),
            message=str(error),
            details=traceback.format_exc(),
            timestamp=time.time(),
            function_name=function_name,
            recovery_suggestions=recovery_suggestions or []
        )
        
        self.errors.append(error_info)
        self._log_error(error_info)
        self._display_error(error_info)
        
        return error_info
    
    def _determine_error_level(self, error: Exception) -> ErrorLevel:
        """오류 수준 결정"""
        if isinstance(error, (MemoryError, OSError)):
            return ErrorLevel.CRITICAL
        elif isinstance(error, (FileNotFoundError, PermissionError)):
            return ErrorLevel.ERROR
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorLevel.WARNING
        else:
            return ErrorLevel.INFO
    
    def _log_error(self, error_info: ErrorInfo):
        """오류 로깅"""
        log_message = f"{error_info.function_name}: {error_info.message}"
        
        if error_info.level == ErrorLevel.CRITICAL:
            self.logger.critical(log_message)
        elif error_info.level == ErrorLevel.ERROR:
            self.logger.error(log_message)
        elif error_info.level == ErrorLevel.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _display_error(self, error_info: ErrorInfo):
        """오류 표시"""
        if error_info.level == ErrorLevel.CRITICAL:
            st.error(f"🚨 심각한 오류: {error_info.message}")
            st.stop()
        elif error_info.level == ErrorLevel.ERROR:
            st.error(f"❌ 오류: {error_info.message}")
        elif error_info.level == ErrorLevel.WARNING:
            st.warning(f"⚠️ 경고: {error_info.message}")
        else:
            st.info(f"ℹ️ 정보: {error_info.message}")
        
        # 복구 제안 표시
        if error_info.recovery_suggestions:
            with st.expander("💡 복구 제안"):
                for suggestion in error_info.recovery_suggestions:
                    st.write(f"• {suggestion}")
    
    @contextmanager
    def safe_execution(self, function_name: str, recovery_suggestions: List[str] = None):
        """안전한 실행 컨텍스트"""
        try:
            yield
        except Exception as e:
            self.handle_error(e, function_name, recovery_suggestions)
            raise  # 재발생하여 호출자가 처리할 수 있도록

class RobustFileProcessor:
    """견고한 파일 처리 시스템"""
    
    def __init__(self, memory_manager: MemoryManager, error_handler: SafeErrorHandler):
        self.memory_manager = memory_manager
        self.error_handler = error_handler
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        
    def validate_file(self, file_data: bytes, filename: str) -> bool:
        """파일 유효성 검사"""
        try:
            # 파일 크기 검사
            if len(file_data) > self.max_file_size:
                raise ValueError(f"파일 크기가 너무 큽니다: {len(file_data)/1024/1024:.1f}MB > 100MB")
            
            # 파일 형식 검사 (확장자 기반)
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.mp3', '.wav', '.m4a', '.mp4', '.avi', '.mov']
            file_ext = Path(filename).suffix.lower()
            if file_ext not in allowed_extensions:
                raise ValueError(f"지원하지 않는 파일 형식: {file_ext}")
            
            # 메모리 사용량 검사
            if self.memory_manager.check_memory_threshold(70.0):
                raise MemoryError("메모리 사용량이 70%를 초과했습니다")
            
            return True
            
        except Exception as e:
            self.error_handler.handle_error(
                e, 
                f"validate_file({filename})",
                [
                    "파일 크기를 줄여보세요",
                    "지원되는 파일 형식을 확인해보세요", 
                    "다른 파일들을 먼저 처리해보세요"
                ]
            )
            return False
    
    def create_safe_temp_file(self, file_data: bytes, suffix: str = '.tmp') -> Optional[str]:
        """안전한 임시 파일 생성"""
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(file_data)
                temp_path = temp_file.name
            
            self.memory_manager.register_temp_file(temp_path)
            return temp_path
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                "create_safe_temp_file",
                [
                    "디스크 공간을 확인해보세요",
                    "임시 폴더 권한을 확인해보세요",
                    "파일 크기를 줄여보세요"
                ]
            )
            return None
    
    @contextmanager
    def safe_file_processing(self, filename: str):
        """안전한 파일 처리 컨텍스트"""
        with self.memory_manager.memory_monitor(f"파일처리({filename})"):
            with self.error_handler.safe_execution(
                f"process_file({filename})",
                [
                    "파일을 다시 업로드해보세요",
                    "파일 형식을 확인해보세요",
                    "메모리를 정리한 후 다시 시도해보세요"
                ]
            ):
                yield

class StabilityMonitor:
    """안정성 모니터링 시스템"""
    
    def __init__(self):
        self.start_time = time.time()
        self.operation_count = 0
        self.error_count = 0
        self.memory_snapshots = []
        
    def record_operation(self, success: bool = True):
        """작업 기록"""
        self.operation_count += 1
        if not success:
            self.error_count += 1
    
    def take_memory_snapshot(self, label: str):
        """메모리 스냅샷 기록"""
        memory_info = psutil.Process().memory_info()
        self.memory_snapshots.append({
            'label': label,
            'timestamp': time.time(),
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024
        })
    
    def get_stability_report(self) -> Dict[str, Any]:
        """안정성 보고서 생성"""
        runtime = time.time() - self.start_time
        success_rate = ((self.operation_count - self.error_count) / self.operation_count * 100) if self.operation_count > 0 else 100
        
        memory_trend = "안정적"
        if len(self.memory_snapshots) >= 2:
            start_memory = self.memory_snapshots[0]['rss_mb']
            end_memory = self.memory_snapshots[-1]['rss_mb']
            growth = end_memory - start_memory
            
            if growth > 500:  # 500MB 이상 증가
                memory_trend = "메모리 누수 의심"
            elif growth > 200:  # 200MB 이상 증가  
                memory_trend = "메모리 사용량 증가"
        
        return {
            'runtime_seconds': runtime,
            'total_operations': self.operation_count,
            'error_count': self.error_count,
            'success_rate': success_rate,
            'memory_trend': memory_trend,
            'memory_snapshots': self.memory_snapshots[-5:]  # 최근 5개만
        }
    
    def display_stability_dashboard(self):
        """안정성 대시보드 표시"""
        report = self.get_stability_report()
        
        st.sidebar.markdown("### 🛡️ 시스템 안정성")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("작업 수", report['total_operations'])
            st.metric("성공률", f"{report['success_rate']:.1f}%")
        
        with col2:
            st.metric("오류 수", report['error_count'])
            st.metric("실행 시간", f"{report['runtime_seconds']:.1f}초")
        
        # 메모리 트렌드 표시
        if report['memory_trend'] != "안정적":
            st.sidebar.warning(f"⚠️ {report['memory_trend']}")
        else:
            st.sidebar.success("✅ 메모리 안정적")

# 통합 안정성 관리자
class IntegratedStabilityManager:
    """통합 안정성 관리 시스템"""
    
    def __init__(self, max_memory_gb: float = 4.0, log_file: Optional[str] = None):
        self.memory_manager = MemoryManager(max_memory_gb)
        self.error_handler = SafeErrorHandler(log_file)
        self.file_processor = RobustFileProcessor(self.memory_manager, self.error_handler)
        self.stability_monitor = StabilityMonitor()
        
        # 자동 정리 스케줄러
        self.setup_auto_cleanup()
    
    def setup_auto_cleanup(self):
        """자동 정리 설정"""
        def cleanup_thread():
            while True:
                time.sleep(300)  # 5분마다
                if self.memory_manager.check_memory_threshold(75.0):
                    st.sidebar.info("🧹 자동 메모리 정리 실행 중...")
                    self.memory_manager.cleanup_resources()
        
        cleanup_thread = threading.Thread(target=cleanup_thread, daemon=True)
        cleanup_thread.start()
    
    def get_system_health(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        memory_usage = self.memory_manager.get_memory_usage()
        stability_report = self.stability_monitor.get_stability_report()
        
        health_score = 100
        issues = []
        
        # 메모리 상태 확인
        if memory_usage['percent'] > 80:
            health_score -= 30
            issues.append("메모리 사용량 높음")
        elif memory_usage['percent'] > 60:
            health_score -= 15
            issues.append("메모리 사용량 주의")
        
        # 오류율 확인
        if stability_report['success_rate'] < 90:
            health_score -= 25
            issues.append("높은 오류율")
        elif stability_report['success_rate'] < 95:
            health_score -= 10
            issues.append("오류율 주의")
        
        return {
            'health_score': max(0, health_score),
            'status': 'healthy' if health_score >= 80 else 'warning' if health_score >= 60 else 'critical',
            'issues': issues,
            'memory_usage': memory_usage,
            'stability_report': stability_report
        }
    
    def display_health_dashboard(self):
        """건강 상태 대시보드"""
        health = self.get_system_health()
        
        st.sidebar.markdown("### 🏥 시스템 건강 상태")
        
        # 건강 점수 표시
        score_color = "green" if health['health_score'] >= 80 else "orange" if health['health_score'] >= 60 else "red"
        st.sidebar.markdown(f"**건강 점수**: <span style='color: {score_color}'>{health['health_score']}/100</span>", unsafe_allow_html=True)
        
        # 상태별 아이콘
        if health['status'] == 'healthy':
            st.sidebar.success("✅ 시스템 정상")
        elif health['status'] == 'warning':
            st.sidebar.warning("⚠️ 주의 필요")
        else:
            st.sidebar.error("🚨 긴급 조치 필요")
        
        # 문제점 표시
        if health['issues']:
            with st.sidebar.expander("🔍 발견된 문제", expanded=True):
                for issue in health['issues']:
                    st.write(f"• {issue}")
        
        # 안정성 모니터 표시
        self.stability_monitor.display_stability_dashboard()
    
    @contextmanager
    def stable_operation(self, operation_name: str):
        """안정적 작업 실행 컨텍스트"""
        self.stability_monitor.take_memory_snapshot(f"{operation_name}_start")
        
        try:
            with self.memory_manager.memory_monitor(operation_name):
                yield self
            self.stability_monitor.record_operation(success=True)
            
        except Exception as e:
            self.stability_monitor.record_operation(success=False)
            raise
            
        finally:
            self.stability_monitor.take_memory_snapshot(f"{operation_name}_end")
            
            # 메모리 정리가 필요한 경우
            if self.memory_manager.check_memory_threshold(70.0):
                self.memory_manager.cleanup_resources()