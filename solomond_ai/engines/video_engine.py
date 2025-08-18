"""
비디오 분석 엔진
OpenCV 및 FFmpeg 기반 비디오 메타데이터 추출 및 프레임 샘플링
"""

import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from .base_engine import BaseEngine

class VideoEngine(BaseEngine):
    """비디오 분석 엔진"""
    
    def __init__(self, sample_frames: int = 5):
        super().__init__("video")
        self.sample_frames = sample_frames
        self.cv2_available = False
        self.ffmpeg_available = False
        
    def initialize(self) -> bool:
        """비디오 처리 라이브러리 초기화"""
        try:
            import cv2
            self.cv2_available = True
            logging.info("OpenCV available for video processing")
        except ImportError:
            logging.warning("OpenCV not available")
        
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.ffmpeg_available = True
                logging.info("FFmpeg available for video processing")
        except Exception:
            logging.warning("FFmpeg not available")
        
        self.is_initialized = self.cv2_available or self.ffmpeg_available
        
        if not self.is_initialized:
            logging.error("No video processing libraries available")
        
        return self.is_initialized
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """비디오 파일 분석"""
        if not self.is_initialized:
            raise RuntimeError("VideoEngine not initialized")
        
        if not self.is_supported_file(file_path):
            raise ValueError(f"Unsupported video format: {Path(file_path).suffix}")
        
        try:
            # 기본 메타데이터 추출
            metadata = self._extract_basic_metadata(file_path)
            
            # FFmpeg으로 상세 정보 추출 (가능한 경우)
            if self.ffmpeg_available:
                ffmpeg_info = self._extract_ffmpeg_metadata(file_path)
                metadata.update(ffmpeg_info)
            
            # OpenCV로 프레임 샘플링 (가능한 경우)
            frames_info = []
            if self.cv2_available:
                frames_info = self._extract_sample_frames(file_path)
            
            return {
                "success": True,
                "metadata": metadata,
                "sample_frames": frames_info,
                "frame_count": len(frames_info),
                "processing_libraries": {
                    "opencv": self.cv2_available,
                    "ffmpeg": self.ffmpeg_available
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    def _extract_basic_metadata(self, file_path: str) -> Dict[str, Any]:
        """기본 메타데이터 추출"""
        file_stat = Path(file_path).stat()
        
        return {
            "file_name": Path(file_path).name,
            "file_size": file_stat.st_size,
            "file_size_mb": round(file_stat.st_size / (1024 * 1024), 2),
            "created_time": file_stat.st_ctime,
            "modified_time": file_stat.st_mtime
        }
    
    def _extract_ffmpeg_metadata(self, file_path: str) -> Dict[str, Any]:
        """FFmpeg으로 비디오 메타데이터 추출"""
        try:
            import subprocess
            import json
            
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # 비디오 스트림 정보 추출
                video_stream = None
                audio_stream = None
                
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video' and not video_stream:
                        video_stream = stream
                    elif stream.get('codec_type') == 'audio' and not audio_stream:
                        audio_stream = stream
                
                metadata = {
                    "duration": float(data.get('format', {}).get('duration', 0)),
                    "bit_rate": int(data.get('format', {}).get('bit_rate', 0)),
                    "format_name": data.get('format', {}).get('format_name', ''),
                }
                
                if video_stream:
                    metadata.update({
                        "width": int(video_stream.get('width', 0)),
                        "height": int(video_stream.get('height', 0)),
                        "fps": eval(video_stream.get('r_frame_rate', '0/1')),
                        "video_codec": video_stream.get('codec_name', ''),
                        "video_bitrate": int(video_stream.get('bit_rate', 0))
                    })
                
                if audio_stream:
                    metadata.update({
                        "audio_codec": audio_stream.get('codec_name', ''),
                        "sample_rate": int(audio_stream.get('sample_rate', 0)),
                        "channels": int(audio_stream.get('channels', 0))
                    })
                
                return metadata
            
        except Exception as e:
            logging.warning(f"FFmpeg metadata extraction failed: {e}")
        
        return {}
    
    def _extract_sample_frames(self, file_path: str) -> List[Dict[str, Any]]:
        """샘플 프레임 추출"""
        try:
            import cv2
            
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                return []
            
            # 비디오 정보
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                return []
            
            # 균등하게 프레임 샘플링
            frame_indices = [
                int(i * total_frames / self.sample_frames) 
                for i in range(self.sample_frames)
            ]
            
            frames_info = []
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # 프레임 정보
                    timestamp = frame_idx / fps if fps > 0 else 0
                    
                    frames_info.append({
                        "frame_index": frame_idx,
                        "timestamp": timestamp,
                        "width": frame.shape[1],
                        "height": frame.shape[0],
                        "channels": frame.shape[2],
                        "sample_number": i + 1
                    })
            
            cap.release()
            return frames_info
            
        except Exception as e:
            logging.warning(f"Frame extraction failed: {e}")
            return []
    
    def get_supported_formats(self) -> List[str]:
        """지원하는 비디오 형식"""
        return [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "sample_frames": self.sample_frames,
            "opencv_available": self.cv2_available,
            "ffmpeg_available": self.ffmpeg_available,
            "initialized": self.is_initialized,
            "supported_formats": self.get_supported_formats()
        }