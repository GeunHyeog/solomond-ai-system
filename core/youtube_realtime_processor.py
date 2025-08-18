#!/usr/bin/env python3
"""
실시간 추적이 통합된 YouTube 처리 모듈
YouTube 영상 다운로드 및 분석에 실시간 진행 상황 표시 적용
"""

import os
import re
import time
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from utils.logger import get_logger

# YouTube 처리를 위한 라이브러리들
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class YouTubeRealtimeProcessor:
    """실시간 추적이 통합된 YouTube 처리 시스템"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.temp_dir = tempfile.gettempdir()
        
        # 실시간 추적 시스템 로드
        try:
            from .realtime_progress_tracker import global_progress_tracker
            from .mcp_auto_problem_solver import global_mcp_solver
            self.progress_tracker = global_progress_tracker
            self.problem_solver = global_mcp_solver
            self.realtime_tracking_available = True
            self.logger.info("실시간 추적 시스템 로드 완료")
        except ImportError as e:
            self.progress_tracker = None
            self.problem_solver = None
            self.realtime_tracking_available = False
            self.logger.warning(f"실시간 추적 시스템 로드 실패: {e}")
        
        # YouTube URL 패턴
        self.youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]+)'
        ]
        
        self._check_dependencies()
    
    def _check_dependencies(self):
        """의존성 확인"""
        if YT_DLP_AVAILABLE:
            self.logger.info("[INFO] yt-dlp 사용 가능")
        else:
            self.logger.warning("[WARNING] yt-dlp 설치 필요: pip install yt-dlp")
        
        if REQUESTS_AVAILABLE:
            self.logger.info("[INFO] requests 사용 가능")
        else:
            self.logger.warning("[WARNING] requests 설치 필요: pip install requests")
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """YouTube URL에서 비디오 ID 추출"""
        for pattern in self.youtube_patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """YouTube 비디오 정보 가져오기 (실시간 추적 포함)"""
        if not YT_DLP_AVAILABLE:
            return {'error': 'yt-dlp not available'}
        
        if self.realtime_tracking_available:
            self.progress_tracker.update_progress_with_time(
                "YouTube 비디오 정보 수집 중...",
                f"URL: {url}"
            )
        
        try:
            video_id = self.extract_video_id(url)
            if not video_id:
                return {'error': 'Invalid YouTube URL'}
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'skip_download': True
            }
            
            start_time = time.time()
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            
            processing_time = time.time() - start_time
            
            # 비디오 정보 정리
            video_info = {
                'id': video_id,
                'title': info.get('title', 'Unknown'),
                'description': info.get('description', ''),
                'duration': info.get('duration', 0),
                'upload_date': info.get('upload_date', ''),
                'uploader': info.get('uploader', 'Unknown'),
                'view_count': info.get('view_count', 0),
                'like_count': info.get('like_count', 0),
                'thumbnail': info.get('thumbnail', ''),
                'formats': len(info.get('formats', [])),
                'extraction_time': processing_time,
                'url': url
            }
            
            # 예상 다운로드 크기 계산
            best_format = None
            for fmt in info.get('formats', []):
                if fmt.get('ext') == 'mp4' and fmt.get('acodec') != 'none':
                    best_format = fmt
                    break
            
            if best_format:
                filesize = best_format.get('filesize') or best_format.get('filesize_approx', 0)
                video_info['estimated_size_mb'] = filesize / (1024 * 1024) if filesize else 0
            else:
                video_info['estimated_size_mb'] = 0
            
            if self.realtime_tracking_available:
                self.progress_tracker.update_progress_with_time(
                    f"비디오 정보 수집 완료: {video_info['title']}",
                    f"길이: {video_info['duration']}초, 예상크기: {video_info['estimated_size_mb']:.1f}MB"
                )
            
            return video_info
            
        except Exception as e:
            error_msg = f"비디오 정보 추출 실패: {str(e)}"
            self.logger.error(error_msg)
            
            # MCP 문제 해결 시스템 활용
            if self.realtime_tracking_available and self.problem_solver:
                problem_result = self.problem_solver.detect_and_solve_problems(
                    memory_usage_mb=100,  # 기본값
                    processing_time=processing_time if 'processing_time' in locals() else 10,
                    file_info={'name': url, 'size_mb': 0},
                    error_message=str(e)
                )
                
                if problem_result['solutions_found']:
                    self.logger.info(f"자동 해결책 {len(problem_result['solutions_found'])}개 발견")
            
            return {'error': error_msg}
    
    def download_audio(self, url: str, output_path: str = None, progress_container=None) -> Dict[str, Any]:
        """YouTube 비디오에서 오디오 추출 (실시간 추적 포함)"""
        if not YT_DLP_AVAILABLE:
            return {'error': 'yt-dlp not available'}
        
        try:
            # 비디오 정보 먼저 가져오기
            video_info = self.get_video_info(url)
            if 'error' in video_info:
                return video_info
            
            # 출력 경로 설정
            if not output_path:
                output_path = os.path.join(
                    self.temp_dir, 
                    f"youtube_audio_{video_info['id']}.%(ext)s"
                )
            
            # 실시간 추적 시작
            if self.realtime_tracking_available:
                # 단일 파일로 추적 시작
                self.progress_tracker.start_analysis(1, progress_container)
                self.progress_tracker.start_file_processing(
                    f"{video_info['title']}.mp3",
                    video_info.get('estimated_size_mb', 0)
                )
            
            # 다운로드 진행률 추적을 위한 훅 함수
            def progress_hook(d):
                if self.realtime_tracking_available and progress_container:
                    if d['status'] == 'downloading':
                        # 다운로드 진행률 표시
                        percent = d.get('_percent_str', '0%').replace('%', '')
                        try:
                            percent_float = float(percent)
                            downloaded = d.get('downloaded_bytes', 0)
                            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                            
                            if total > 0:
                                downloaded_mb = downloaded / (1024 * 1024)
                                total_mb = total / (1024 * 1024)
                                speed = d.get('speed', 0)
                                
                                detail = f"다운로드: {downloaded_mb:.1f}MB / {total_mb:.1f}MB"
                                if speed:
                                    speed_mbps = speed / (1024 * 1024)
                                    detail += f" ({speed_mbps:.1f}MB/s)"
                                
                                with progress_container.container():
                                    import streamlit as st
                                    st.progress(
                                        percent_float / 100,
                                        text=f"YouTube 오디오 다운로드: {percent}%"
                                    )
                                    st.write(detail)
                        except (ValueError, TypeError):
                            pass
                    
                    elif d['status'] == 'finished':
                        if progress_container:
                            with progress_container.container():
                                import streamlit as st
                                st.success(f"다운로드 완료: {d.get('filename', 'unknown')}")
            
            # yt-dlp 옵션 설정
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': output_path,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'progress_hooks': [progress_hook] if self.realtime_tracking_available else [],
                'quiet': False if self.realtime_tracking_available else True,
                'no_warnings': not self.realtime_tracking_available
            }
            
            start_time = time.time()
            
            # 다운로드 실행
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            processing_time = time.time() - start_time
            
            # 출력 파일 찾기
            output_base = output_path.replace('.%(ext)s', '')
            possible_files = [
                f"{output_base}.mp3",
                f"{output_base}.m4a",
                f"{output_base}.webm"
            ]
            
            actual_output_file = None
            for file_path in possible_files:
                if os.path.exists(file_path):
                    actual_output_file = file_path
                    break
            
            if not actual_output_file:
                # 다른 방법으로 파일 찾기
                import glob
                pattern = f"{output_base}.*"
                matches = glob.glob(pattern)
                if matches:
                    actual_output_file = matches[0]
            
            if actual_output_file and os.path.exists(actual_output_file):
                file_size = os.path.getsize(actual_output_file) / (1024 * 1024)  # MB
                
                result = {
                    'success': True,
                    'output_file': actual_output_file,
                    'file_size_mb': file_size,
                    'processing_time': processing_time,
                    'video_info': video_info,
                    'download_speed_mbps': file_size / processing_time if processing_time > 0 else 0
                }
                
                if self.realtime_tracking_available:
                    self.progress_tracker.finish_file_processing()
                    self.progress_tracker.finish_analysis()
                    
                    self.progress_tracker.update_progress_with_time(
                        f"오디오 추출 완료: {file_size:.1f}MB",
                        f"처리시간: {processing_time:.1f}초, 속도: {result['download_speed_mbps']:.1f}MB/s"
                    )
                
                return result
            else:
                error_msg = "다운로드된 파일을 찾을 수 없습니다"
                if self.realtime_tracking_available:
                    self.progress_tracker.update_progress_with_time(f"오류: {error_msg}")
                
                return {'error': error_msg}
                
        except Exception as e:
            error_msg = f"YouTube 오디오 다운로드 실패: {str(e)}"
            self.logger.error(error_msg)
            
            # 실시간 추적 중이면 오류 표시
            if self.realtime_tracking_available:
                self.progress_tracker.update_progress_with_time(f"오류 발생: {error_msg}")
                
                # MCP 문제 해결 시스템 활용
                if self.problem_solver:
                    processing_time = time.time() - start_time if 'start_time' in locals() else 0
                    problem_result = self.problem_solver.detect_and_solve_problems(
                        memory_usage_mb=200,  # YouTube 다운로드는 메모리를 더 사용
                        processing_time=processing_time,
                        file_info={
                            'name': video_info.get('title', url) if 'video_info' in locals() else url,
                            'size_mb': video_info.get('estimated_size_mb', 0) if 'video_info' in locals() else 0
                        },
                        error_message=str(e)
                    )
                    
                    if problem_result['solutions_found']:
                        self.logger.info(f"자동 해결책 {len(problem_result['solutions_found'])}개 발견")
                        
                        # 사용자에게 해결책 표시
                        if progress_container:
                            with progress_container.container():
                                import streamlit as st
                                st.warning("문제가 감지되었습니다. 자동 해결책을 확인하세요:")
                                for i, solution in enumerate(problem_result['solutions_found'][:2], 1):
                                    st.write(f"{i}. {solution.get('title', '해결책')}")
                                    if solution.get('url'):
                                        st.write(f"   참고: {solution['url']}")
            
            return {'error': error_msg}
    
    def batch_process_youtube_urls(self, urls: List[str], progress_container=None) -> List[Dict[str, Any]]:
        """여러 YouTube URL 배치 처리 (실시간 추적 포함)"""
        if not urls:
            return []
        
        results = []
        
        # 실시간 추적 시작
        if self.realtime_tracking_available:
            self.progress_tracker.start_analysis(len(urls), progress_container)
        
        for i, url in enumerate(urls):
            if self.realtime_tracking_available:
                self.progress_tracker.update_progress_with_time(
                    f"YouTube URL 처리 중 ({i+1}/{len(urls)})",
                    f"URL: {url}"
                )
            
            # 각 URL별 오디오 다운로드
            result = self.download_audio(url, progress_container=progress_container)
            result['url'] = url
            result['index'] = i + 1
            results.append(result)
            
            # 처리 간 짧은 대기 (서버 부담 방지)
            time.sleep(1)
        
        if self.realtime_tracking_available:
            self.progress_tracker.finish_analysis()
            
            # 결과 요약
            successful = len([r for r in results if 'success' in r and r['success']])
            failed = len(results) - successful
            
            self.progress_tracker.update_progress_with_time(
                f"배치 처리 완료: {successful}개 성공, {failed}개 실패",
                f"총 {len(urls)}개 URL 처리 완료"
            )
        
        return results
    
    def get_processing_status(self) -> Dict[str, Any]:
        """현재 처리 상태 반환"""
        status = {
            'realtime_tracking_available': self.realtime_tracking_available,
            'yt_dlp_available': YT_DLP_AVAILABLE,
            'requests_available': REQUESTS_AVAILABLE,
            'temp_directory': self.temp_dir,
            'supported_patterns': len(self.youtube_patterns),
            'last_check': datetime.now().isoformat()
        }
        
        if self.realtime_tracking_available and self.progress_tracker:
            try:
                performance_stats = self.progress_tracker.get_performance_stats()
                status['performance_stats'] = performance_stats
            except:
                pass
        
        return status

# 전역 인스턴스
global_youtube_realtime_processor = YouTubeRealtimeProcessor()