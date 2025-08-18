#!/usr/bin/env python3
"""
스트리밍 진행률 추적 시스템 v2.6
대용량 파일 처리 진행 상황을 실시간 추적
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import streamlit as st
from pathlib import Path

@dataclass
class FileProcessingStatus:
    """파일 처리 상태"""
    file_name: str
    file_size_mb: float
    processed_mb: float = 0.0
    status: str = "pending"  # pending, processing, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_chunk: int = 0
    total_chunks: int = 0
    processing_speed_mbps: float = 0.0
    memory_usage_mb: float = 0.0
    error_message: Optional[str] = None
    optimization_level: str = "balanced"
    streaming_enabled: bool = False

@dataclass
class BatchProcessingStatus:
    """배치 처리 상태"""
    total_files: int
    completed_files: int = 0
    failed_files: int = 0
    current_file_index: int = 0
    overall_progress_percent: float = 0.0
    estimated_completion_time: Optional[datetime] = None
    total_size_mb: float = 0.0
    processed_size_mb: float = 0.0
    files: List[FileProcessingStatus] = field(default_factory=list)

class StreamingProgressTracker:
    """스트리밍 진행률 추적 시스템"""
    
    def __init__(self):
        self.lock = threading.RLock()
        self.logger = self._setup_logging()
        
        # 현재 처리 상태
        self.current_batch: Optional[BatchProcessingStatus] = None
        self.current_file: Optional[FileProcessingStatus] = None
        
        # 진행률 콜백들
        self.progress_callbacks: List[Callable] = []
        self.update_callbacks: List[Callable] = []
        
        # 성능 통계
        self.session_stats = {
            'total_files_processed': 0,
            'total_size_processed_mb': 0.0,
            'total_processing_time_seconds': 0.0,
            'average_speed_mbps': 0.0,
            'session_start_time': datetime.now()
        }
        
        # UI 업데이트 스레드
        self.ui_update_thread = None
        self.stop_ui_updates = False
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def start_batch_processing(self, file_paths: List[str]) -> str:
        """배치 처리 시작"""
        with self.lock:
            batch_id = f"batch_{int(time.time())}"
            
            # 파일 정보 수집
            files = []
            total_size_mb = 0.0
            
            for file_path in file_paths:
                try:
                    path = Path(file_path)
                    if path.exists():
                        size_mb = path.stat().st_size / (1024 * 1024)
                        total_size_mb += size_mb
                        
                        file_status = FileProcessingStatus(
                            file_name=path.name,
                            file_size_mb=size_mb
                        )
                        files.append(file_status)
                    else:
                        self.logger.warning(f"⚠️ 파일을 찾을 수 없음: {file_path}")
                
                except Exception as e:
                    self.logger.error(f"❌ 파일 정보 수집 실패 {file_path}: {e}")
            
            # 배치 상태 초기화
            self.current_batch = BatchProcessingStatus(
                total_files=len(files),
                total_size_mb=total_size_mb,
                files=files
            )
            
            self.logger.info(f"🚀 배치 처리 시작: {len(files)}개 파일, {total_size_mb:.1f}MB")
            
            # UI 업데이트 스레드 시작
            self._start_ui_updates()
            
            return batch_id
    
    def start_file_processing(self, file_index: int, streaming_enabled: bool = False, optimization_level: str = "balanced") -> None:
        """파일 처리 시작"""
        with self.lock:
            if not self.current_batch or file_index >= len(self.current_batch.files):
                return
            
            self.current_batch.current_file_index = file_index
            self.current_file = self.current_batch.files[file_index]
            
            # 파일 상태 업데이트
            self.current_file.status = "processing"
            self.current_file.start_time = datetime.now()
            self.current_file.streaming_enabled = streaming_enabled
            self.current_file.optimization_level = optimization_level
            
            self.logger.info(f"📁 파일 처리 시작: {self.current_file.file_name} ({self.current_file.file_size_mb:.1f}MB)")
            
            # 콜백 호출
            self._trigger_update_callbacks()
    
    def update_file_progress(self, processed_mb: float, chunk_id: int = 0, total_chunks: int = 0, 
                           memory_usage_mb: float = 0.0, speed_mbps: float = 0.0) -> None:
        """파일 처리 진행률 업데이트"""
        with self.lock:
            if not self.current_file:
                return
            
            # 진행률 업데이트
            self.current_file.processed_mb = min(processed_mb, self.current_file.file_size_mb)
            self.current_file.current_chunk = chunk_id
            self.current_file.total_chunks = total_chunks
            self.current_file.memory_usage_mb = memory_usage_mb
            self.current_file.processing_speed_mbps = speed_mbps
            
            # 배치 전체 진행률 계산
            if self.current_batch:
                # 이전 파일들의 크기
                prev_files_size = sum(
                    f.file_size_mb for f in self.current_batch.files[:self.current_batch.current_file_index]
                )
                
                # 현재까지 처리된 총 크기
                total_processed = prev_files_size + self.current_file.processed_mb
                
                # 전체 진행률
                if self.current_batch.total_size_mb > 0:
                    self.current_batch.overall_progress_percent = (total_processed / self.current_batch.total_size_mb) * 100
                    self.current_batch.processed_size_mb = total_processed
                
                # 완료 시간 예측
                if speed_mbps > 0:
                    remaining_mb = self.current_batch.total_size_mb - total_processed
                    remaining_time_seconds = remaining_mb / speed_mbps
                    self.current_batch.estimated_completion_time = datetime.now() + timedelta(seconds=remaining_time_seconds)
            
            # 진행률 콜백 호출
            self._trigger_progress_callbacks()
    
    def complete_file_processing(self, success: bool = True, error_message: Optional[str] = None) -> None:
        """파일 처리 완료"""
        with self.lock:
            if not self.current_file:
                return
            
            # 파일 상태 업데이트
            self.current_file.end_time = datetime.now()
            self.current_file.status = "completed" if success else "failed"
            
            if error_message:
                self.current_file.error_message = error_message
            
            # 세션 통계 업데이트
            if success:
                if self.current_batch:
                    self.current_batch.completed_files += 1
                
                self.session_stats['total_files_processed'] += 1
                self.session_stats['total_size_processed_mb'] += self.current_file.file_size_mb
                
                # 처리 시간 계산
                if self.current_file.start_time:
                    processing_time = (self.current_file.end_time - self.current_file.start_time).total_seconds()
                    self.session_stats['total_processing_time_seconds'] += processing_time
                    
                    # 평균 속도 업데이트
                    if self.session_stats['total_processing_time_seconds'] > 0:
                        self.session_stats['average_speed_mbps'] = (
                            self.session_stats['total_size_processed_mb'] / 
                            self.session_stats['total_processing_time_seconds']
                        )
            else:
                if self.current_batch:
                    self.current_batch.failed_files += 1
            
            status_icon = "✅" if success else "❌"
            self.logger.info(f"{status_icon} 파일 처리 완료: {self.current_file.file_name}")
            
            # 콜백 호출
            self._trigger_update_callbacks()
            
            self.current_file = None
    
    def complete_batch_processing(self) -> None:
        """배치 처리 완료"""
        with self.lock:
            if not self.current_batch:
                return
            
            self.logger.info(f"🎉 배치 처리 완료: {self.current_batch.completed_files}/{self.current_batch.total_files} 성공")
            
            # UI 업데이트 중지
            self._stop_ui_updates()
            
            # 최종 콜백 호출
            self._trigger_update_callbacks()
            
            self.current_batch = None
    
    def add_progress_callback(self, callback: Callable) -> None:
        """진행률 콜백 추가"""
        with self.lock:
            if callback not in self.progress_callbacks:
                self.progress_callbacks.append(callback)
    
    def add_update_callback(self, callback: Callable) -> None:
        """업데이트 콜백 추가"""
        with self.lock:
            if callback not in self.update_callbacks:
                self.update_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable) -> None:
        """진행률 콜백 제거"""
        with self.lock:
            if callback in self.progress_callbacks:
                self.progress_callbacks.remove(callback)
    
    def remove_update_callback(self, callback: Callable) -> None:
        """업데이트 콜백 제거"""
        with self.lock:
            if callback in self.update_callbacks:
                self.update_callbacks.remove(callback)
    
    def _trigger_progress_callbacks(self) -> None:
        """진행률 콜백 트리거"""
        for callback in self.progress_callbacks:
            try:
                callback(self.get_current_status())
            except Exception as e:
                self.logger.error(f"❌ 진행률 콜백 오류: {e}")
    
    def _trigger_update_callbacks(self) -> None:
        """업데이트 콜백 트리거"""
        for callback in self.update_callbacks:
            try:
                callback(self.get_current_status())
            except Exception as e:
                self.logger.error(f"❌ 업데이트 콜백 오류: {e}")
    
    def _start_ui_updates(self) -> None:
        """UI 업데이트 스레드 시작"""
        if self.ui_update_thread and self.ui_update_thread.is_alive():
            return
        
        self.stop_ui_updates = False
        self.ui_update_thread = threading.Thread(target=self._ui_update_loop, daemon=True)
        self.ui_update_thread.start()
    
    def _stop_ui_updates(self) -> None:
        """UI 업데이트 중지"""
        self.stop_ui_updates = True
        if self.ui_update_thread and self.ui_update_thread.is_alive():
            self.ui_update_thread.join(timeout=1.0)
    
    def _ui_update_loop(self) -> None:
        """UI 업데이트 루프"""
        while not self.stop_ui_updates:
            try:
                # 주기적으로 UI 업데이트 트리거
                if self.current_file and self.current_file.status == "processing":
                    self._trigger_progress_callbacks()
                
                time.sleep(0.5)  # 0.5초 간격
            
            except Exception as e:
                self.logger.error(f"❌ UI 업데이트 루프 오류: {e}")
                break
    
    def get_current_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        with self.lock:
            status = {
                'has_batch': self.current_batch is not None,
                'has_current_file': self.current_file is not None,
                'session_stats': self.session_stats.copy()
            }
            
            if self.current_batch:
                status['batch'] = {
                    'total_files': self.current_batch.total_files,
                    'completed_files': self.current_batch.completed_files,
                    'failed_files': self.current_batch.failed_files,
                    'current_file_index': self.current_batch.current_file_index,
                    'overall_progress_percent': self.current_batch.overall_progress_percent,
                    'total_size_mb': self.current_batch.total_size_mb,
                    'processed_size_mb': self.current_batch.processed_size_mb,
                    'estimated_completion_time': self.current_batch.estimated_completion_time.isoformat() if self.current_batch.estimated_completion_time else None
                }
            
            if self.current_file:
                file_progress_percent = (self.current_file.processed_mb / self.current_file.file_size_mb * 100) if self.current_file.file_size_mb > 0 else 0
                
                processing_time_seconds = 0
                if self.current_file.start_time:
                    processing_time_seconds = (datetime.now() - self.current_file.start_time).total_seconds()
                
                status['current_file'] = {
                    'file_name': self.current_file.file_name,
                    'file_size_mb': self.current_file.file_size_mb,
                    'processed_mb': self.current_file.processed_mb,
                    'progress_percent': file_progress_percent,
                    'status': self.current_file.status,
                    'current_chunk': self.current_file.current_chunk,
                    'total_chunks': self.current_file.total_chunks,
                    'processing_speed_mbps': self.current_file.processing_speed_mbps,
                    'memory_usage_mb': self.current_file.memory_usage_mb,
                    'processing_time_seconds': processing_time_seconds,
                    'optimization_level': self.current_file.optimization_level,
                    'streaming_enabled': self.current_file.streaming_enabled,
                    'error_message': self.current_file.error_message
                }
            
            return status
    
    def get_detailed_file_status(self, file_index: int) -> Optional[Dict[str, Any]]:
        """특정 파일의 상세 상태 반환"""
        with self.lock:
            if not self.current_batch or file_index >= len(self.current_batch.files):
                return None
            
            file_status = self.current_batch.files[file_index]
            
            return {
                'file_name': file_status.file_name,
                'file_size_mb': file_status.file_size_mb,
                'processed_mb': file_status.processed_mb,
                'status': file_status.status,
                'start_time': file_status.start_time.isoformat() if file_status.start_time else None,
                'end_time': file_status.end_time.isoformat() if file_status.end_time else None,
                'current_chunk': file_status.current_chunk,
                'total_chunks': file_status.total_chunks,
                'processing_speed_mbps': file_status.processing_speed_mbps,
                'memory_usage_mb': file_status.memory_usage_mb,
                'optimization_level': file_status.optimization_level,
                'streaming_enabled': file_status.streaming_enabled,
                'error_message': file_status.error_message
            }
    
    def reset_session(self) -> None:
        """세션 리셋"""
        with self.lock:
            self._stop_ui_updates()
            
            self.current_batch = None
            self.current_file = None
            
            self.session_stats = {
                'total_files_processed': 0,
                'total_size_processed_mb': 0.0,
                'total_processing_time_seconds': 0.0,
                'average_speed_mbps': 0.0,
                'session_start_time': datetime.now()
            }
            
            self.logger.info("🔄 진행률 추적 세션 리셋")

# 전역 진행률 추적 시스템
_global_progress_tracker = None
_global_tracker_lock = threading.Lock()

def get_global_progress_tracker() -> StreamingProgressTracker:
    """전역 진행률 추적 시스템 가져오기"""
    global _global_progress_tracker
    
    with _global_tracker_lock:
        if _global_progress_tracker is None:
            _global_progress_tracker = StreamingProgressTracker()
        return _global_progress_tracker

# Streamlit 진행률 표시 컴포넌트
def render_streaming_progress(tracker: StreamingProgressTracker) -> None:
    """Streamlit에서 스트리밍 진행률 표시"""
    status = tracker.get_current_status()
    
    if not status['has_batch']:
        st.info("📊 현재 처리 중인 배치가 없습니다.")
        return
    
    batch_info = status['batch']
    
    # 전체 진행률
    st.subheader("📊 전체 진행 상황")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📁 총 파일",
            f"{batch_info['total_files']}개"
        )
    
    with col2:
        st.metric(
            "✅ 완료",
            f"{batch_info['completed_files']}개"
        )
    
    with col3:
        st.metric(
            "❌ 실패",
            f"{batch_info['failed_files']}개"
        )
    
    with col4:
        st.metric(
            "📊 전체 진행률",
            f"{batch_info['overall_progress_percent']:.1f}%"
        )
    
    # 진행률 바
    progress_bar = st.progress(batch_info['overall_progress_percent'] / 100)
    
    # 예상 완료 시간
    if batch_info['estimated_completion_time']:
        completion_time = datetime.fromisoformat(batch_info['estimated_completion_time'])
        remaining_time = completion_time - datetime.now()
        if remaining_time.total_seconds() > 0:
            st.info(f"⏱️ 예상 완료 시간: {remaining_time.total_seconds()/60:.1f}분 후")
    
    # 현재 처리 중인 파일
    if status['has_current_file']:
        current_file = status['current_file']
        
        st.subheader(f"📁 현재 처리 중: {current_file['file_name']}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📈 파일 진행률",
                f"{current_file['progress_percent']:.1f}%"
            )
        
        with col2:
            st.metric(
                "⚡ 처리 속도",
                f"{current_file['processing_speed_mbps']:.1f}MB/s"
            )
        
        with col3:
            st.metric(
                "💾 메모리 사용",
                f"{current_file['memory_usage_mb']:.1f}MB"
            )
        
        # 파일 진행률 바
        file_progress_bar = st.progress(current_file['progress_percent'] / 100)
        
        # 청크 정보 (스트리밍 사용 시)
        if current_file['streaming_enabled'] and current_file['total_chunks'] > 0:
            st.info(f"🔄 청크 진행: {current_file['current_chunk']}/{current_file['total_chunks']} ({current_file['optimization_level']} 최적화)")
        
        # 에러 메시지
        if current_file['error_message']:
            st.error(f"❌ 오류: {current_file['error_message']}")
    
    # 세션 통계
    session_stats = status['session_stats']
    
    with st.expander("📊 세션 통계"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "📁 처리된 파일",
                f"{session_stats['total_files_processed']}개"
            )
            st.metric(
                "💾 처리된 크기",
                f"{session_stats['total_size_processed_mb']:.1f}MB"
            )
        
        with col2:
            st.metric(
                "⏱️ 총 처리 시간",
                f"{session_stats['total_processing_time_seconds']/60:.1f}분"
            )
            st.metric(
                "📊 평균 속도",
                f"{session_stats['average_speed_mbps']:.1f}MB/s"
            )

# 사용 예시
if __name__ == "__main__":
    # 테스트용 진행률 추적
    tracker = StreamingProgressTracker()
    
    # 배치 시작
    test_files = ["test1.mp4", "test2.wav", "test3.jpg"]
    batch_id = tracker.start_batch_processing(test_files)
    
    # 파일 처리 시뮬레이션
    for i, file_path in enumerate(test_files):
        tracker.start_file_processing(i, streaming_enabled=True, optimization_level="balanced")
        
        # 진행률 업데이트 시뮬레이션
        for progress in range(0, 101, 10):
            tracker.update_file_progress(
                processed_mb=progress * 0.1,  # 가상 크기
                chunk_id=progress // 10,
                total_chunks=10,
                memory_usage_mb=50 + progress * 0.5,
                speed_mbps=10.0
            )
            
            time.sleep(0.1)  # 진행률 업데이트 간격
        
        tracker.complete_file_processing(success=True)
    
    # 배치 완료
    tracker.complete_batch_processing()
    
    print("✅ 스트리밍 진행률 추적 테스트 완료!")