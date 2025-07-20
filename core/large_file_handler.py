#!/usr/bin/env python3
"""
대용량 파일 처리 시스템
청크 기반 업로드 및 메모리 효율적 처리
"""

import os
import tempfile
import shutil
import hashlib
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Iterator
import streamlit as st

class LargeFileHandler:
    """대용량 파일 처리를 위한 핸들러"""
    
    def __init__(self, chunk_size: int = 50 * 1024 * 1024):  # 50MB 청크
        self.chunk_size = chunk_size
        self.logger = self._setup_logging()
        self.temp_dir = Path(tempfile.gettempdir()) / "solomond_large_files"
        self.temp_dir.mkdir(exist_ok=True)
        
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f"{__name__}.LargeFileHandler")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def save_uploaded_file_chunked(self, uploaded_file, progress_callback=None) -> Optional[str]:
        """
        청크 단위로 업로드된 파일을 저장
        
        Args:
            uploaded_file: Streamlit 업로드 파일 객체
            progress_callback: 진행률 콜백 함수
            
        Returns:
            저장된 파일 경로 또는 None (실패 시)
        """
        try:
            # 파일 정보
            file_name = uploaded_file.name
            file_size = uploaded_file.size if hasattr(uploaded_file, 'size') else len(uploaded_file.getvalue())
            
            self.logger.info(f"🔄 대용량 파일 저장 시작: {file_name} ({file_size / (1024*1024*1024):.2f}GB)")
            
            # 임시 파일 생성
            file_extension = Path(file_name).suffix
            temp_file_path = self.temp_dir / f"upload_{int(time.time())}_{hashlib.md5(file_name.encode()).hexdigest()[:8]}{file_extension}"
            
            # 청크 단위로 파일 저장
            total_written = 0
            with open(temp_file_path, 'wb') as temp_file:
                # Streamlit 업로드 파일에서 청크 단위로 읽기
                uploaded_file.seek(0)  # 파일 포인터를 처음으로
                
                while True:
                    chunk = uploaded_file.read(self.chunk_size)
                    if not chunk:
                        break
                    
                    temp_file.write(chunk)
                    total_written += len(chunk)
                    
                    # 진행률 업데이트
                    if progress_callback and file_size > 0:
                        progress = min(total_written / file_size, 1.0)
                        progress_callback(progress)
            
            self.logger.info(f"✅ 파일 저장 완료: {temp_file_path} ({total_written / (1024*1024):.1f}MB)")
            return str(temp_file_path)
            
        except Exception as e:
            self.logger.error(f"❌ 파일 저장 실패: {e}")
            return None
    
    def extract_audio_from_video(self, video_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        동영상에서 오디오 추출 (FFmpeg 사용)
        
        Args:
            video_path: 동영상 파일 경로
            output_path: 출력 오디오 파일 경로 (None이면 자동 생성)
            
        Returns:
            추출된 오디오 파일 경로 또는 None (실패 시)
        """
        try:
            import subprocess
            
            if output_path is None:
                video_name = Path(video_path).stem
                output_path = self.temp_dir / f"{video_name}_audio.wav"
            
            self.logger.info(f"🎵 동영상에서 오디오 추출 시작: {Path(video_path).name}")
            
            # FFmpeg 명령어로 오디오 추출
            cmd = [
                'ffmpeg', 
                '-i', video_path,
                '-vn',  # 비디오 스트림 제외
                '-acodec', 'pcm_s16le',  # PCM 16-bit 오디오 코덱
                '-ar', '16000',  # 16kHz 샘플링 레이트
                '-ac', '1',  # 모노 채널
                '-y',  # 덮어쓰기 허용
                str(output_path)
            ]
            
            # 진행률 표시를 위한 프로세스 실행
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            
            # 프로세스 완료 대기
            stdout, stderr = process.communicate(timeout=3600)  # 1시간 타임아웃
            
            if process.returncode == 0:
                self.logger.info(f"✅ 오디오 추출 완료: {output_path}")
                return str(output_path)
            else:
                self.logger.error(f"❌ FFmpeg 오류: {stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error("❌ 오디오 추출 시간 초과")
            process.kill()
            return None
        except Exception as e:
            self.logger.error(f"❌ 오디오 추출 실패: {e}")
            return None
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        동영상 파일 정보 조회
        
        Args:
            video_path: 동영상 파일 경로
            
        Returns:
            동영상 정보 딕셔너리
        """
        try:
            import subprocess
            import json
            
            cmd = [
                'ffprobe', 
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                
                # 기본 정보 추출
                format_info = info.get('format', {})
                duration = float(format_info.get('duration', 0))
                size = int(format_info.get('size', 0))
                
                # 오디오 스트림 찾기
                has_audio = False
                audio_codec = None
                for stream in info.get('streams', []):
                    if stream.get('codec_type') == 'audio':
                        has_audio = True
                        audio_codec = stream.get('codec_name')
                        break
                
                return {
                    'duration': duration,
                    'size': size,
                    'has_audio': has_audio,
                    'audio_codec': audio_codec,
                    'format_name': format_info.get('format_name', ''),
                    'bit_rate': int(format_info.get('bit_rate', 0))
                }
            else:
                self.logger.error(f"❌ FFprobe 오류: {result.stderr}")
                return {}
                
        except Exception as e:
            self.logger.error(f"❌ 동영상 정보 조회 실패: {e}")
            return {}
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        오래된 임시 파일들 정리
        
        Args:
            max_age_hours: 삭제할 파일의 최대 나이 (시간)
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            cleaned_files = 0
            cleaned_size = 0
            
            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        cleaned_files += 1
                        cleaned_size += file_size
            
            if cleaned_files > 0:
                self.logger.info(f"🗑️ 임시 파일 정리: {cleaned_files}개 파일, {cleaned_size / (1024*1024):.1f}MB")
                
        except Exception as e:
            self.logger.error(f"❌ 임시 파일 정리 실패: {e}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """임시 저장소 정보 조회"""
        try:
            total_size = 0
            file_count = 0
            
            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            # 디스크 사용량 조회
            disk_usage = shutil.disk_usage(self.temp_dir)
            
            return {
                'temp_files_count': file_count,
                'temp_files_size_mb': total_size / (1024 * 1024),
                'disk_free_gb': disk_usage.free / (1024 * 1024 * 1024),
                'disk_total_gb': disk_usage.total / (1024 * 1024 * 1024)
            }
            
        except Exception as e:
            self.logger.error(f"❌ 저장소 정보 조회 실패: {e}")
            return {}

# 전역 핸들러 인스턴스
large_file_handler = LargeFileHandler()