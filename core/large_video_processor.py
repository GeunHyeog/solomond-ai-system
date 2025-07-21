#!/usr/bin/env python3
"""
대용량 비디오 처리 모듈 - MOV, MP4, AVI 등 지원
고용량 파일 스트리밍 처리 및 메모리 최적화
"""

import os
import tempfile
import logging
import time
import subprocess
from typing import Dict, Any, List, Optional, Iterator
from pathlib import Path
from datetime import datetime
import json

# 비디오 처리를 위한 라이브러리들
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class LargeVideoProcessor:
    """대용량 비디오 처리 클래스"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 지원 형식
        self.supported_formats = ['.mov', '.mp4', '.avi', '.mkv', '.wmv', '.flv', '.webm']
        
        # 처리 설정
        self.max_file_size = 5 * 1024 * 1024 * 1024  # 5GB
        self.chunk_duration = 30  # 30초 청크
        self.max_duration = 3600  # 1시간
        
        # 오디오 추출 설정
        self.audio_sample_rate = 16000
        self.audio_channels = 1
        
        # 처리 통계
        self.processing_stats = {
            "total_files": 0,
            "successful_processes": 0,
            "total_processing_time": 0,
            "largest_file_size": 0,
            "last_process_time": None
        }
        
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
        if CV2_AVAILABLE:
            self.logger.info("[INFO] OpenCV 사용 가능 - 비디오 프레임 처리 지원")
        else:
            self.logger.warning("[WARNING] OpenCV 미설치 - 프레임 분석 제한됨")
        
        if MOVIEPY_AVAILABLE:
            self.logger.info("[INFO] MoviePy 사용 가능 - 비디오 편집 지원")
        else:
            self.logger.warning("[WARNING] MoviePy 미설치 - 비디오 편집 제한됨")
        
        if NUMPY_AVAILABLE:
            self.logger.info("[INFO] NumPy 사용 가능 - 수치 연산 지원")
        else:
            self.logger.warning("[WARNING] NumPy 미설치 - 수치 처리 제한됨")
        
        # FFmpeg 확인
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.logger.info("[INFO] FFmpeg 사용 가능 - 비디오 변환 지원")
                self.ffmpeg_available = True
            else:
                self.logger.warning("[WARNING] FFmpeg 실행 실패")
                self.ffmpeg_available = False
        except Exception:
            self.logger.warning("[WARNING] FFmpeg 미설치 - 비디오 변환 제한됨")
            self.ffmpeg_available = False
    
    def is_available(self) -> bool:
        """대용량 비디오 처리 가능 여부"""
        return self.ffmpeg_available and NUMPY_AVAILABLE
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """비디오 파일 정보 추출"""
        if not os.path.exists(video_path):
            return {
                "status": "error",
                "error": f"파일이 존재하지 않음: {video_path}"
            }
        
        try:
            self.logger.info(f"[INFO] 비디오 정보 조회: {os.path.basename(video_path)}")
            
            file_size = os.path.getsize(video_path)
            file_ext = os.path.splitext(video_path)[1].lower()
            
            if file_ext not in self.supported_formats:
                return {
                    "status": "error",
                    "error": f"지원하지 않는 형식: {file_ext}",
                    "supported_formats": self.supported_formats
                }
            
            # FFmpeg로 비디오 정보 추출
            if self.ffmpeg_available:
                cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                    '-show_format', '-show_streams', video_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    probe_data = json.loads(result.stdout)
                    
                    # 비디오 스트림 정보
                    video_stream = None
                    audio_stream = None
                    
                    for stream in probe_data.get('streams', []):
                        if stream.get('codec_type') == 'video' and not video_stream:
                            video_stream = stream
                        elif stream.get('codec_type') == 'audio' and not audio_stream:
                            audio_stream = stream
                    
                    format_info = probe_data.get('format', {})
                    
                    # 결과 구성
                    result = {
                        "status": "success",
                        "file_path": video_path,
                        "file_name": os.path.basename(video_path),
                        "file_size": file_size,
                        "file_size_mb": round(file_size / (1024 * 1024), 2),
                        "file_format": file_ext,
                        "duration": float(format_info.get('duration', 0)),
                        "bitrate": int(format_info.get('bit_rate', 0)),
                        "format_name": format_info.get('format_long_name', 'Unknown'),
                        "video_info": {},
                        "audio_info": {},
                        "has_video": video_stream is not None,
                        "has_audio": audio_stream is not None
                    }
                    
                    # 비디오 스트림 정보
                    if video_stream:
                        result["video_info"] = {
                            "codec": video_stream.get('codec_name', 'Unknown'),
                            "width": video_stream.get('width', 0),
                            "height": video_stream.get('height', 0),
                            "fps": eval(video_stream.get('r_frame_rate', '0/1')),
                            "frames": int(video_stream.get('nb_frames', 0)),
                            "pixel_format": video_stream.get('pix_fmt', 'Unknown')
                        }
                        
                        # 해상도 카테고리
                        width = result["video_info"]["width"]
                        height = result["video_info"]["height"]
                        if width >= 3840:
                            result["video_info"]["quality"] = "4K"
                        elif width >= 1920:
                            result["video_info"]["quality"] = "Full HD"
                        elif width >= 1280:
                            result["video_info"]["quality"] = "HD"
                        else:
                            result["video_info"]["quality"] = "SD"
                    
                    # 오디오 스트림 정보
                    if audio_stream:
                        result["audio_info"] = {
                            "codec": audio_stream.get('codec_name', 'Unknown'),
                            "sample_rate": int(audio_stream.get('sample_rate', 0)),
                            "channels": audio_stream.get('channels', 0),
                            "channel_layout": audio_stream.get('channel_layout', 'Unknown'),
                            "bitrate": int(audio_stream.get('bit_rate', 0))
                        }
                    
                    # 지속시간 포맷
                    duration_sec = result["duration"]
                    hours = int(duration_sec // 3600)
                    minutes = int((duration_sec % 3600) // 60)
                    seconds = int(duration_sec % 60)
                    result["duration_formatted"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    
                    # 크기 체크
                    if file_size > self.max_file_size:
                        result["warning"] = f"파일 크기가 매우 큽니다 ({result['file_size_mb']}MB). 처리 시간이 오래 걸릴 수 있습니다."
                    
                    self.logger.info(f"[SUCCESS] 비디오 정보 조회 완료: {result['duration_formatted']}, {result['file_size_mb']}MB")
                    return result
                
                else:
                    raise Exception(f"FFprobe 실행 실패: {result.stderr}")
            
            else:
                # FFmpeg 없이 기본 정보만
                return {
                    "status": "partial_success",
                    "file_path": video_path,
                    "file_name": os.path.basename(video_path),
                    "file_size": file_size,
                    "file_size_mb": round(file_size / (1024 * 1024), 2),
                    "file_format": file_ext,
                    "warning": "FFmpeg가 없어 제한적인 정보만 제공됩니다"
                }
                
        except Exception as e:
            error_msg = f"비디오 정보 조회 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg,
                "file_path": video_path
            }
    
    def extract_audio_from_video(self, video_path: str, output_dir: str = None) -> Dict[str, Any]:
        """비디오에서 오디오 추출"""
        start_time = time.time()
        
        try:
            self.logger.info(f"[INFO] 비디오 오디오 추출 시작: {os.path.basename(video_path)}")
            
            if not self.ffmpeg_available:
                return {
                    "status": "error",
                    "error": "FFmpeg가 설치되지 않음",
                    "install_guide": "FFmpeg 설치 필요"
                }
            
            # 비디오 정보 먼저 확인
            video_info = self.get_video_info(video_path)
            if video_info['status'] != 'success':
                return video_info
            
            if not video_info.get('has_audio', False):
                return {
                    "status": "error",
                    "error": "비디오에 오디오 트랙이 없습니다"
                }
            
            # 출력 디렉토리 설정
            if output_dir is None:
                output_dir = tempfile.gettempdir()
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 출력 파일명 생성
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            safe_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            output_path = os.path.join(output_dir, f"{safe_name}_audio.wav")
            
            # FFmpeg 명령어 구성
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # 비디오 스트림 제외
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', str(self.audio_sample_rate),  # 샘플링 레이트
                '-ac', str(self.audio_channels),  # 채널 수
                '-y',  # 덮어쓰기 허용
                output_path
            ]
            
            self.logger.info("[INFO] FFmpeg 오디오 추출 실행 중...")
            
            # 프로세스 실행 (시간 제한)
            timeout = max(300, video_info.get('duration', 0) * 2)  # 최소 5분, 또는 영상 길이의 2배
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                if os.path.exists(output_path):
                    processing_time = time.time() - start_time
                    audio_size = os.path.getsize(output_path)
                    
                    # 통계 업데이트
                    self._update_stats(processing_time, True, video_info.get('file_size', 0))
                    
                    result_data = {
                        "status": "success",
                        "audio_file": output_path,
                        "original_video": video_path,
                        "audio_size": audio_size,
                        "audio_size_mb": round(audio_size / (1024 * 1024), 2),
                        "processing_time": round(processing_time, 2),
                        "audio_duration": video_info.get('duration', 0),
                        "audio_info": {
                            "sample_rate": self.audio_sample_rate,
                            "channels": self.audio_channels,
                            "format": "WAV PCM 16-bit"
                        },
                        "video_info": video_info,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    self.logger.info(f"[SUCCESS] 오디오 추출 완료 ({processing_time:.1f}초, {result_data['audio_size_mb']}MB)")
                    return result_data
                
                else:
                    raise Exception("오디오 파일이 생성되지 않았습니다")
            
            else:
                raise Exception(f"FFmpeg 오디오 추출 실패: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, False, 0)
            error_msg = f"오디오 추출 시간 초과 ({timeout}초)"
            self.logger.error(f"[ERROR] {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg,
                "processing_time": round(processing_time, 2)
            }
        
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, False, 0)
            error_msg = f"오디오 추출 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg,
                "processing_time": round(processing_time, 2),
                "file_path": video_path
            }
    
    def process_large_video_streaming(self, video_path: str, chunk_duration: int = None) -> Iterator[Dict[str, Any]]:
        """대용량 비디오 스트리밍 처리"""
        if chunk_duration is None:
            chunk_duration = self.chunk_duration
        
        try:
            self.logger.info(f"[INFO] 스트리밍 처리 시작: {os.path.basename(video_path)}")
            
            # 비디오 정보 확인
            video_info = self.get_video_info(video_path)
            if video_info['status'] != 'success':
                yield video_info
                return
            
            total_duration = video_info.get('duration', 0)
            total_chunks = int(total_duration / chunk_duration) + 1
            
            # 청크별 처리
            for chunk_idx in range(total_chunks):
                start_time = chunk_idx * chunk_duration
                end_time = min((chunk_idx + 1) * chunk_duration, total_duration)
                
                if start_time >= total_duration:
                    break
                
                # 청크 오디오 추출
                chunk_result = self._extract_audio_chunk(
                    video_path, start_time, end_time - start_time, chunk_idx
                )
                
                chunk_result.update({
                    "chunk_info": {
                        "chunk_index": chunk_idx,
                        "total_chunks": total_chunks,
                        "start_time": start_time,
                        "end_time": end_time,
                        "progress": round((chunk_idx + 1) / total_chunks * 100, 1)
                    },
                    "video_info": video_info
                })
                
                yield chunk_result
                
        except Exception as e:
            error_msg = f"스트리밍 처리 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            
            yield {
                "status": "error",
                "error": error_msg,
                "file_path": video_path
            }
    
    def _extract_audio_chunk(self, video_path: str, start_time: float, duration: float, chunk_idx: int) -> Dict[str, Any]:
        """비디오 청크에서 오디오 추출"""
        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # FFmpeg 명령어 (시간 범위 지정)
            cmd = [
                'ffmpeg', '-i', video_path,
                '-ss', str(start_time),  # 시작 시간
                '-t', str(duration),     # 지속 시간
                '-vn',  # 비디오 제외
                '-acodec', 'pcm_s16le',
                '-ar', str(self.audio_sample_rate),
                '-ac', str(self.audio_channels),
                '-y',
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(temp_path):
                return {
                    "status": "success",
                    "audio_chunk": temp_path,
                    "chunk_duration": duration,
                    "start_time": start_time
                }
            else:
                return {
                    "status": "error",
                    "error": f"청크 {chunk_idx} 추출 실패",
                    "ffmpeg_error": result.stderr
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": f"청크 {chunk_idx} 처리 오류: {str(e)}"
            }
    
    def _update_stats(self, processing_time: float, success: bool, file_size: int):
        """처리 통계 업데이트"""
        self.processing_stats["total_files"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        
        if success:
            self.processing_stats["successful_processes"] += 1
        
        if file_size > self.processing_stats["largest_file_size"]:
            self.processing_stats["largest_file_size"] = file_size
        
        self.processing_stats["last_process_time"] = datetime.now().isoformat()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        stats = self.processing_stats.copy()
        
        if stats["total_files"] > 0:
            stats["success_rate"] = round(
                stats["successful_processes"] / stats["total_files"] * 100, 1
            )
            stats["average_processing_time"] = round(
                stats["total_processing_time"] / stats["total_files"], 2
            )
            stats["largest_file_size_mb"] = round(
                stats["largest_file_size"] / (1024 * 1024), 2
            )
        
        return stats
    
    def extract_keyframes_moviepy(self, video_path: str, num_frames: int = 5) -> Dict[str, Any]:
        """MoviePy를 사용한 키프레임 추출"""
        if not MOVIEPY_AVAILABLE:
            return {
                "status": "error",
                "error": "MoviePy가 설치되지 않음"
            }
        
        try:
            self.logger.info(f"[INFO] MoviePy 키프레임 추출: {os.path.basename(video_path)}")
            
            from moviepy.editor import VideoFileClip
            import tempfile
            
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
                interval = duration / (num_frames + 1)
                
                keyframes = []
                temp_dir = tempfile.mkdtemp()
                
                for i in range(1, num_frames + 1):
                    timestamp = interval * i
                    frame_path = os.path.join(temp_dir, f"keyframe_{i:03d}.jpg")
                    
                    frame = clip.get_frame(timestamp)
                    
                    # OpenCV로 저장 (가능한 경우)
                    if CV2_AVAILABLE:
                        import cv2
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(frame_path, frame_bgr)
                    else:
                        # PIL로 저장
                        from PIL import Image
                        Image.fromarray(frame).save(frame_path)
                    
                    keyframes.append({
                        "frame_index": i,
                        "timestamp": round(timestamp, 2),
                        "frame_path": frame_path,
                        "file_size": os.path.getsize(frame_path)
                    })
                
                self.logger.info(f"[SUCCESS] {num_frames}개 키프레임 추출 완료")
                
                return {
                    "status": "success",
                    "keyframes": keyframes,
                    "total_frames": num_frames,
                    "video_duration": duration,
                    "temp_directory": temp_dir
                }
                
        except Exception as e:
            error_msg = f"키프레임 추출 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg
            }
    
    def get_enhanced_video_info_moviepy(self, video_path: str) -> Dict[str, Any]:
        """MoviePy를 사용한 향상된 비디오 정보"""
        if not MOVIEPY_AVAILABLE:
            return self.get_video_info(video_path)
        
        try:
            self.logger.info(f"[INFO] MoviePy 향상된 정보 추출: {os.path.basename(video_path)}")
            
            from moviepy.editor import VideoFileClip
            
            # 기본 정보 먼저 가져오기
            basic_info = self.get_video_info(video_path)
            if basic_info['status'] != 'success':
                return basic_info
            
            with VideoFileClip(video_path) as clip:
                enhanced_info = basic_info.copy()
                
                # MoviePy로 얻을 수 있는 추가 정보
                enhanced_info.update({
                    "moviepy_duration": clip.duration,
                    "has_mask": clip.mask is not None,
                    "fps_exact": clip.fps,
                    "audio_fps": clip.audio.fps if clip.audio else None,
                    "audio_duration": clip.audio.duration if clip.audio else None,
                    "video_quality_analysis": self._analyze_video_quality_moviepy(clip),
                    "frame_analysis": self._analyze_frames_moviepy(clip)
                })
                
                self.logger.info("[SUCCESS] MoviePy 향상된 정보 추출 완료")
                return enhanced_info
                
        except Exception as e:
            error_msg = f"MoviePy 정보 추출 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            
            # 기본 정보라도 반환
            basic_info = self.get_video_info(video_path)
            if basic_info['status'] == 'success':
                basic_info['moviepy_error'] = error_msg
            return basic_info
    
    def _analyze_video_quality_moviepy(self, clip) -> Dict[str, Any]:
        """MoviePy 클립의 품질 분석"""
        try:
            # 중간 프레임 샘플링
            mid_time = clip.duration / 2
            frame = clip.get_frame(mid_time)
            
            if NUMPY_AVAILABLE:
                # 기본 품질 메트릭
                mean_brightness = np.mean(frame)
                std_brightness = np.std(frame)
                
                return {
                    "mean_brightness": round(float(mean_brightness), 2),
                    "brightness_std": round(float(std_brightness), 2),
                    "brightness_quality": "good" if 50 < mean_brightness < 200 else "poor",
                    "contrast_quality": "good" if std_brightness > 30 else "low"
                }
            else:
                return {"analysis": "NumPy 필요"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_frames_moviepy(self, clip) -> Dict[str, Any]:
        """MoviePy 클립의 프레임 분석"""
        try:
            total_frames = int(clip.fps * clip.duration)
            
            # 샘플 프레임들 분석
            sample_times = [clip.duration * i / 4 for i in range(5)]
            frame_sizes = []
            
            for time in sample_times:
                if time < clip.duration:
                    frame = clip.get_frame(time)
                    frame_sizes.append(frame.nbytes if hasattr(frame, 'nbytes') else len(frame.tobytes()))
            
            avg_frame_size = sum(frame_sizes) / len(frame_sizes) if frame_sizes else 0
            
            return {
                "estimated_total_frames": total_frames,
                "average_frame_size_bytes": round(avg_frame_size),
                "sample_frame_count": len(frame_sizes)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_installation_guide(self) -> Dict[str, Any]:
        """설치 가이드"""
        missing_packages = []
        
        if not CV2_AVAILABLE:
            missing_packages.append({
                "package": "opencv-python",
                "command": "pip install opencv-python",
                "purpose": "비디오 프레임 처리"
            })
        
        if not MOVIEPY_AVAILABLE:
            missing_packages.append({
                "package": "moviepy",
                "command": "pip install moviepy==1.0.3",
                "purpose": "비디오 편집 및 처리"
            })
        
        if not NUMPY_AVAILABLE:
            missing_packages.append({
                "package": "numpy",
                "command": "pip install numpy",
                "purpose": "수치 연산"
            })
        
        return {
            "available": self.is_available(),
            "ffmpeg_available": self.ffmpeg_available,
            "moviepy_available": MOVIEPY_AVAILABLE,
            "opencv_available": CV2_AVAILABLE,
            "numpy_available": NUMPY_AVAILABLE,
            "missing_packages": missing_packages,
            "install_all": "pip install opencv-python moviepy==1.0.3 numpy",
            "ffmpeg_install": {
                "windows": "https://ffmpeg.org/download.html#build-windows",
                "note": "FFmpeg는 별도로 설치하고 PATH에 추가해야 합니다"
            },
            "supported_formats": self.supported_formats,
            "max_file_size_gb": round(self.max_file_size / (1024**3), 1),
            "enhanced_features": {
                "keyframe_extraction": MOVIEPY_AVAILABLE,
                "quality_analysis": MOVIEPY_AVAILABLE and NUMPY_AVAILABLE,
                "frame_analysis": MOVIEPY_AVAILABLE,
                "enhanced_info": MOVIEPY_AVAILABLE
            }
        }

# 전역 인스턴스
large_video_processor = LargeVideoProcessor()

def extract_audio_from_video(video_path: str, output_dir: str = None) -> Dict[str, Any]:
    """비디오에서 오디오 추출 (전역 접근용)"""
    return large_video_processor.extract_audio_from_video(video_path, output_dir)

def get_video_info(video_path: str) -> Dict[str, Any]:
    """비디오 정보 조회 (전역 접근용)"""
    return large_video_processor.get_video_info(video_path)

def get_enhanced_video_info(video_path: str) -> Dict[str, Any]:
    """향상된 비디오 정보 조회 (전역 접근용)"""
    return large_video_processor.get_enhanced_video_info_moviepy(video_path)

def extract_keyframes(video_path: str, num_frames: int = 5) -> Dict[str, Any]:
    """키프레임 추출 (전역 접근용)"""
    return large_video_processor.extract_keyframes_moviepy(video_path, num_frames)