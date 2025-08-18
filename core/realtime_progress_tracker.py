#!/usr/bin/env python3
"""
실시간 진행 추적기 - 경과 시간 및 진행 상황 표시
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
import streamlit as st
from utils.logger import get_logger

class RealtimeProgressTracker:
    """실시간 진행 상황 추적 및 표시"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.start_time = None
        self.current_stage = ""
        self.current_file = ""
        self.total_files = 0
        self.processed_files = 0
        self.stage_start_time = None
        self.file_start_time = None
        self.estimated_total_time = None
        self.file_processing_times = []  # 파일별 처리 시간 기록
        self.is_running = False
        self.progress_container = None
        
    def start_analysis(self, total_files: int, progress_container=None):
        """분석 시작"""
        self.start_time = time.time()
        self.total_files = total_files
        self.processed_files = 0
        self.file_processing_times = []
        self.is_running = True
        self.progress_container = progress_container
        
        if progress_container:
            with progress_container.container():
                st.info(f"🚀 분석 시작 - 총 {total_files}개 파일 처리 예정")
                st.write(f"⏰ 시작 시간: {datetime.now().strftime('%H:%M:%S')}")
    
    def start_stage(self, stage_name: str):
        """단계 시작"""
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        
        if self.progress_container:
            with self.progress_container.container():
                elapsed = self._get_elapsed_time()
                st.info(f"📋 {stage_name} 시작 (경과시간: {elapsed})")
    
    def start_file_processing(self, filename: str, file_size_mb: float = 0):
        """파일 처리 시작"""
        self.current_file = filename
        self.file_start_time = time.time()
        
        # 예상 처리 시간 계산
        estimated_file_time = self._estimate_file_processing_time(file_size_mb)
        
        if self.progress_container:
            with self.progress_container.container():
                elapsed = self._get_elapsed_time()
                remaining_files = self.total_files - self.processed_files
                
                # 전체 예상 완료 시간 계산
                if self.file_processing_times:
                    avg_time_per_file = sum(self.file_processing_times) / len(self.file_processing_times)
                    estimated_remaining = avg_time_per_file * remaining_files
                    eta = datetime.now() + timedelta(seconds=estimated_remaining)
                    eta_str = f" | 예상 완료: {eta.strftime('%H:%M:%S')}"
                else:
                    eta_str = ""
                
                st.write(f"🔄 처리 중: **{filename}** ({self.processed_files + 1}/{self.total_files})")
                st.write(f"⏱️ 경과시간: {elapsed} | 파일 크기: {file_size_mb:.1f}MB{eta_str}")
                
                if estimated_file_time > 0:
                    st.write(f"📊 예상 처리 시간: {self._format_time(estimated_file_time)}")
    
    def finish_file_processing(self):
        """파일 처리 완료"""
        if self.file_start_time:
            processing_time = time.time() - self.file_start_time
            self.file_processing_times.append(processing_time)
            self.processed_files += 1
            
            if self.progress_container:
                with self.progress_container.container():
                    elapsed = self._get_elapsed_time()
                    st.success(f"✅ **{self.current_file}** 처리 완료 ({self._format_time(processing_time)})")
                    
                    # 진행률 바 표시
                    progress = self.processed_files / self.total_files if self.total_files > 0 else 0
                    st.progress(progress, text=f"전체 진행률: {progress*100:.1f}% ({self.processed_files}/{self.total_files})")
    
    def finish_stage(self):
        """단계 완료"""
        if self.stage_start_time:
            stage_time = time.time() - self.stage_start_time
            
            if self.progress_container:
                with self.progress_container.container():
                    elapsed = self._get_elapsed_time()
                    st.success(f"✅ **{self.current_stage}** 완료 (소요시간: {self._format_time(stage_time)})")
    
    def finish_analysis(self):
        """분석 완료"""
        if self.start_time:
            total_time = time.time() - self.start_time
            self.is_running = False
            
            if self.progress_container:
                with self.progress_container.container():
                    st.success(f"🎉 **전체 분석 완료!**")
                    st.write(f"📊 총 소요시간: {self._format_time(total_time)}")
                    st.write(f"📁 처리된 파일: {self.processed_files}개")
                    
                    # 통계 정보
                    if self.file_processing_times:
                        avg_time = sum(self.file_processing_times) / len(self.file_processing_times)
                        max_time = max(self.file_processing_times)
                        min_time = min(self.file_processing_times)
                        
                        st.write("📈 **처리 통계**:")
                        st.write(f"   • 평균 파일 처리 시간: {self._format_time(avg_time)}")
                        st.write(f"   • 최대 처리 시간: {self._format_time(max_time)}")
                        st.write(f"   • 최소 처리 시간: {self._format_time(min_time)}")
    
    def update_progress_with_time(self, message: str, detail: str = ""):
        """시간 정보가 포함된 진행 상황 업데이트"""
        if self.progress_container:
            with self.progress_container.container():
                elapsed = self._get_elapsed_time()
                st.info(f"⏱️ {message} (경과시간: {elapsed})")
                if detail:
                    st.write(f"   📝 {detail}")
    
    def _get_elapsed_time(self) -> str:
        """경과 시간 문자열 반환"""
        if not self.start_time:
            return "00:00:00"
        
        elapsed_seconds = time.time() - self.start_time
        return self._format_time(elapsed_seconds)
    
    def _format_time(self, seconds: float) -> str:
        """시간을 읽기 쉬운 형태로 포맷"""
        if seconds < 60:
            return f"{seconds:.1f}초"
        elif seconds < 3600:  # 1시간 미만
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}분 {secs}초"
        else:  # 1시간 이상
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}시간 {minutes}분"
    
    def _estimate_file_processing_time(self, file_size_mb: float) -> float:
        """파일 크기 기반 예상 처리 시간 계산"""
        if not self.file_processing_times or file_size_mb <= 0:
            return 0
        
        # 기본 예상치 (MB당 처리 시간)
        base_time_per_mb = {
            'audio': 8.0,    # 음성 파일: MB당 8초
            'video': 12.0,   # 영상 파일: MB당 12초  
            'image': 2.0,    # 이미지 파일: MB당 2초
            'document': 1.0  # 문서 파일: MB당 1초
        }
        
        # 현재 단계에 따른 예상 시간
        if 'audio' in self.current_stage.lower() or 'Audio' in self.current_stage:
            return file_size_mb * base_time_per_mb['audio']
        elif 'video' in self.current_stage.lower() or 'Video' in self.current_stage:
            return file_size_mb * base_time_per_mb['video']
        elif 'image' in self.current_stage.lower() or 'Image' in self.current_stage:
            return file_size_mb * base_time_per_mb['image']
        else:
            return file_size_mb * base_time_per_mb['document']
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        if not self.file_processing_times:
            return {}
        
        total_time = time.time() - self.start_time if self.start_time else 0
        avg_time = sum(self.file_processing_times) / len(self.file_processing_times)
        
        return {
            'total_elapsed_time': total_time,
            'processed_files': self.processed_files,
            'average_time_per_file': avg_time,
            'estimated_remaining_time': avg_time * (self.total_files - self.processed_files) if self.total_files > self.processed_files else 0,
            'processing_speed': self.processed_files / (total_time / 60) if total_time > 0 else 0  # 파일/분
        }

# 전역 인스턴스
global_progress_tracker = RealtimeProgressTracker()

def track_analysis_progress(func):
    """분석 함수를 위한 데코레이터"""
    def wrapper(*args, **kwargs):
        # 진행 추적 시작
        if hasattr(args[0], 'total_files'):
            global_progress_tracker.start_analysis(args[0].total_files)
        
        try:
            result = func(*args, **kwargs)
            global_progress_tracker.finish_analysis()
            return result
        except Exception as e:
            global_progress_tracker.update_progress_with_time(f"❌ 오류 발생: {str(e)}")
            raise
    
    return wrapper