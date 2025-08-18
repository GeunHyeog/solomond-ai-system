#!/usr/bin/env python3
"""
YouTube 영상 처리 모듈 - 영상 다운로드 및 오디오 추출
"""

import os
import re
import logging
import tempfile
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import subprocess

# YouTube 다운로드를 위한 라이브러리들
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

class YouTubeProcessor:
    """YouTube 영상 처리 클래스"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.temp_dir = tempfile.gettempdir()
        self.supported_formats = ['mp4', 'webm', 'mkv']
        self.audio_formats = ['mp3', 'wav', 'm4a', 'aac']
        
        # YouTube URL 패턴
        self.youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]+)'
        ]
        
        self._check_dependencies()
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _check_dependencies(self):
        """의존성 확인"""
        if YT_DLP_AVAILABLE:
            self.logger.info("[INFO] yt-dlp 사용 가능")
        else:
            self.logger.warning("[WARNING] yt-dlp 미설치 - YouTube 다운로드 불가")
        
        if REQUESTS_AVAILABLE:
            self.logger.info("[INFO] requests 사용 가능")
        else:
            self.logger.warning("[WARNING] requests 미설치 - HTTP 요청 제한됨")
    
    def is_youtube_url(self, url: str) -> bool:
        """YouTube URL 여부 확인"""
        for pattern in self.youtube_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        return False
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """YouTube URL에서 비디오 ID 추출"""
        for pattern in self.youtube_patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """YouTube 영상 정보 가져오기"""
        if not YT_DLP_AVAILABLE:
            return {
                "success": False,
                "status": "error",
                "error": "yt-dlp가 설치되지 않음",
                "install_command": "pip install yt-dlp"
            }
        
        if not self.is_youtube_url(url):
            return {
                "success": False,
                "status": "error",
                "error": "유효하지 않은 YouTube URL"
            }
        
        try:
            self.logger.info(f"[INFO] YouTube 영상 정보 조회: {url}")
            
            # yt-dlp 옵션 설정
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # 필요한 정보만 추출
                extracted_info = {
                    "video_id": info.get('id', ''),
                    "title": info.get('title', ''),
                    "description": info.get('description', ''),
                    "duration": info.get('duration', 0),
                    "upload_date": info.get('upload_date', ''),
                    "uploader": info.get('uploader', ''),
                    "view_count": info.get('view_count', 0),
                    "like_count": info.get('like_count', 0),
                    "thumbnail": info.get('thumbnail', ''),
                    "webpage_url": info.get('webpage_url', url),
                    "categories": info.get('categories', []),
                    "tags": info.get('tags', []),
                    "automatic_captions": list(info.get('automatic_captions', {}).keys()),
                    "subtitles": list(info.get('subtitles', {}).keys())
                }
                
                video_info = {
                    "success": True,
                    "status": "success", 
                    "info": extracted_info
                }
                
                # 지속 시간을 읽기 쉬운 형태로 변환
                if extracted_info['duration']:
                    duration_sec = extracted_info['duration']
                    hours = duration_sec // 3600
                    minutes = (duration_sec % 3600) // 60
                    seconds = duration_sec % 60
                    
                    if hours > 0:
                        extracted_info['duration_formatted'] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    else:
                        extracted_info['duration_formatted'] = f"{minutes:02d}:{seconds:02d}"
                else:
                    extracted_info['duration_formatted'] = "알 수 없음"
                
                self.logger.info(f"[SUCCESS] 영상 정보 조회 완료: {extracted_info['title']}")
                return video_info
                
        except Exception as e:
            error_msg = f"YouTube 정보 조회 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            return {
                "success": False,
                "status": "error",
                "error": error_msg,
                "url": url
            }
    
    def download_audio(self, url: str, output_dir: str = None) -> Dict[str, Any]:
        """YouTube 영상에서 오디오만 다운로드"""
        if not YT_DLP_AVAILABLE:
            return {
                "status": "error",
                "error": "yt-dlp가 설치되지 않음",
                "install_command": "pip install yt-dlp"
            }
        
        start_time = time.time()
        
        try:
            self.logger.info(f"[INFO] YouTube 오디오 다운로드 시작: {url}")
            
            # 출력 디렉토리 설정
            if output_dir is None:
                output_dir = os.path.join(self.temp_dir, "youtube_audio")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 영상 정보 먼저 가져오기
            video_info = self.get_video_info(url)
            if video_info['status'] != 'success':
                return video_info
            
            # 파일명 안전하게 만들기
            safe_title = re.sub(r'[^\w\s-]', '', video_info['title'])
            safe_title = re.sub(r'[-\s]+', '-', safe_title)
            
            # yt-dlp 다운로드 옵션
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(output_dir, f"{safe_title}.%(ext)s"),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # 다운로드된 파일 찾기
            audio_file = os.path.join(output_dir, f"{safe_title}.wav")
            
            if not os.path.exists(audio_file):
                # 다른 확장자로 시도
                for ext in ['wav', 'mp3', 'm4a', 'webm']:
                    potential_file = os.path.join(output_dir, f"{safe_title}.{ext}")
                    if os.path.exists(potential_file):
                        audio_file = potential_file
                        break
            
            if not os.path.exists(audio_file):
                raise Exception("다운로드된 오디오 파일을 찾을 수 없습니다")
            
            processing_time = time.time() - start_time
            file_size = os.path.getsize(audio_file)
            
            result = {
                "status": "success",
                "audio_file": audio_file,
                "file_size": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "processing_time": round(processing_time, 2),
                "video_info": video_info,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"[SUCCESS] 오디오 다운로드 완료 ({processing_time:.1f}초, {result['file_size_mb']}MB)")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"YouTube 오디오 다운로드 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg,
                "url": url,
                "processing_time": round(processing_time, 2),
                "timestamp": datetime.now().isoformat()
            }
    
    def download_with_subtitles(self, url: str, output_dir: str = None) -> Dict[str, Any]:
        """YouTube 영상을 자막과 함께 다운로드"""
        if not YT_DLP_AVAILABLE:
            return {
                "status": "error", 
                "error": "yt-dlp가 설치되지 않음"
            }
        
        try:
            self.logger.info(f"[INFO] YouTube 자막 포함 다운로드: {url}")
            
            if output_dir is None:
                output_dir = os.path.join(self.temp_dir, "youtube_with_subs")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 영상 정보 가져오기
            video_info = self.get_video_info(url)
            if video_info['status'] != 'success':
                return video_info
            
            safe_title = re.sub(r'[^\w\s-]', '', video_info['title'])
            safe_title = re.sub(r'[-\s]+', '-', safe_title)
            
            # 자막 다운로드 옵션
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(output_dir, f"{safe_title}.%(ext)s"),
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['ko', 'en', 'ja', 'zh'],
                'subtitlesformat': 'vtt',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'quiet': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # 결과 파일들 찾기
            result_files = {
                "audio_file": None,
                "subtitle_files": []
            }
            
            # 다운로드된 파일들 검색
            for file in os.listdir(output_dir):
                if file.startswith(safe_title):
                    full_path = os.path.join(output_dir, file)
                    if file.endswith(('.wav', '.mp3', '.m4a')):
                        result_files["audio_file"] = full_path
                    elif file.endswith('.vtt'):
                        result_files["subtitle_files"].append(full_path)
            
            return {
                "status": "success",
                "files": result_files,
                "video_info": video_info,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"자막 포함 다운로드 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            return {
                "status": "error",
                "error": error_msg
            }
    
    def get_installation_guide(self) -> Dict[str, Any]:
        """설치 가이드"""
        missing_packages = []
        
        if not YT_DLP_AVAILABLE:
            missing_packages.append({
                "package": "yt-dlp",
                "command": "pip install yt-dlp",
                "purpose": "YouTube 영상 다운로드"
            })
        
        if not REQUESTS_AVAILABLE:
            missing_packages.append({
                "package": "requests", 
                "command": "pip install requests",
                "purpose": "HTTP 요청"
            })
        
        return {
            "available": YT_DLP_AVAILABLE and REQUESTS_AVAILABLE,
            "missing_packages": missing_packages,
            "install_all": "pip install yt-dlp requests",
            "ffmpeg_required": "FFmpeg가 설치되어 있어야 오디오 변환이 가능합니다"
        }

# 전역 인스턴스
youtube_processor = YouTubeProcessor()

def download_youtube_audio(url: str, output_dir: str = None) -> Dict[str, Any]:
    """YouTube 오디오 다운로드 (전역 접근용)"""
    return youtube_processor.download_audio(url, output_dir)

def get_youtube_info(url: str) -> Dict[str, Any]:
    """YouTube 영상 정보 가져오기 (전역 접근용)"""
    return youtube_processor.get_video_info(url)