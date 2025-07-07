"""
솔로몬드 AI 시스템 - 비디오 처리기
동영상 파일에서 음성 추출 및 STT 처리 모듈
"""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union
import asyncio

class VideoProcessor:
    """동영상 처리 클래스"""
    
    def __init__(self):
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
        self.audio_output_format = 'wav'  # Whisper 호환성 최고
        
    def get_supported_formats(self) -> List[str]:
        """지원하는 비디오 형식 반환"""
        return self.supported_video_formats
    
    def is_video_file(self, filename: str) -> bool:
        """비디오 파일인지 확인"""
        ext = Path(filename).suffix.lower()
        return ext in self.supported_video_formats
    
    def check_ffmpeg_available(self) -> bool:
        """FFmpeg 설치 여부 확인"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    async def extract_audio_from_video(self, 
                                     video_content: bytes, 
                                     original_filename: str) -> Dict:
        """
        동영상에서 음성 추출
        
        Args:
            video_content: 비디오 파일 바이너리 데이터
            original_filename: 원본 파일명
            
        Returns:
            추출 결과 딕셔너리
        """
        try:
            # FFmpeg 확인
            if not self.check_ffmpeg_available():
                # FFmpeg 없이 대체 방법 시도
                return await self._extract_audio_python_only(video_content, original_filename)
            
            # 임시 비디오 파일 생성
            video_ext = Path(original_filename).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=video_ext) as video_temp:
                video_temp.write(video_content)
                video_path = video_temp.name
            
            try:
                # 임시 오디오 파일 경로
                audio_path = video_path.replace(video_ext, f'.{self.audio_output_format}')
                
                # FFmpeg로 음성 추출
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-vn',  # 비디오 스트림 제거
                    '-acodec', 'pcm_s16le',  # WAV 포맷
                    '-ar', '16000',  # 샘플링 레이트 (Whisper 최적화)
                    '-ac', '1',  # 모노 채널
                    '-y',  # 덮어쓰기
                    audio_path
                ]
                
                print(f"🎬 동영상 음성 추출 시작: {original_filename}")
                
                # 비동기 프로세스 실행
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    # 추출된 오디오 파일 읽기
                    with open(audio_path, 'rb') as audio_file:
                        audio_content = audio_file.read()
                    
                    print(f"✅ 음성 추출 성공: {len(audio_content)} bytes")
                    
                    return {
                        "success": True,
                        "audio_content": audio_content,
                        "audio_format": self.audio_output_format,
                        "original_filename": original_filename,
                        "extracted_filename": f"{Path(original_filename).stem}.{self.audio_output_format}",
                        "extraction_method": "ffmpeg"
                    }
                else:
                    error_msg = stderr.decode('utf-8') if stderr else "알 수 없는 오류"
                    print(f"❌ FFmpeg 오류: {error_msg}")
                    
                    return {
                        "success": False,
                        "error": f"음성 추출 실패: {error_msg}",
                        "extraction_method": "ffmpeg"
                    }
                    
            finally:
                # 임시 파일 정리
                try:
                    os.unlink(video_path)
                    if os.path.exists(audio_path):
                        os.unlink(audio_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"❌ 비디오 처리 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "extraction_method": "ffmpeg"
            }
    
    async def _extract_audio_python_only(self, 
                                       video_content: bytes, 
                                       original_filename: str) -> Dict:
        """
        Python 라이브러리만으로 음성 추출 (FFmpeg 대체)
        """
        try:
            # moviepy 사용 시도 (Python 3.13 호환성 문제로 제한적)
            print("⚠️ FFmpeg 없음. Python 라이브러리로 대체 시도...")
            
            # 기본적으로는 실패 반환 (사용자가 FFmpeg 설치 유도)
            return {
                "success": False,
                "error": "동영상 처리를 위해 FFmpeg가 필요합니다. 설치 방법을 안내해드립니다.",
                "extraction_method": "python_fallback",
                "install_guide": {
                    "windows": "https://ffmpeg.org/download.html에서 다운로드 후 PATH 설정",
                    "mac": "brew install ffmpeg",
                    "ubuntu": "sudo apt update && sudo apt install ffmpeg"
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Python 대체 처리 실패: {str(e)}",
                "extraction_method": "python_fallback"
            }
    
    async def get_video_info(self, video_content: bytes, filename: str) -> Dict:
        """비디오 파일 정보 분석"""
        try:
            file_size_mb = round(len(video_content) / (1024 * 1024), 2)
            file_ext = Path(filename).suffix.lower()
            
            info = {
                "filename": filename,
                "size_mb": file_size_mb,
                "format": file_ext,
                "supported": self.is_video_file(filename),
                "ffmpeg_available": self.check_ffmpeg_available()
            }
            
            # FFmpeg로 상세 정보 추출 시도
            if info["ffmpeg_available"]:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                        temp_file.write(video_content)
                        temp_path = temp_file.name
                    
                    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', temp_path]
                    process = await asyncio.create_subprocess_exec(
                        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                    )
                    stdout, _ = await process.communicate()
                    
                    if process.returncode == 0:
                        import json
                        probe_data = json.loads(stdout.decode())
                        format_info = probe_data.get('format', {})
                        
                        info.update({
                            "duration_seconds": float(format_info.get('duration', 0)),
                            "bitrate": int(format_info.get('bit_rate', 0)),
                            "has_audio": any('audio' in stream.get('codec_type', '') 
                                           for stream in probe_data.get('streams', []))
                        })
                    
                    os.unlink(temp_path)
                    
                except Exception as e:
                    print(f"⚠️ 비디오 정보 추출 실패: {e}")
            
            return info
            
        except Exception as e:
            return {
                "filename": filename,
                "error": str(e),
                "supported": False
            }

# 전역 인스턴스
_video_processor_instance = None

def get_video_processor() -> VideoProcessor:
    """전역 비디오 처리기 인스턴스 반환"""
    global _video_processor_instance
    if _video_processor_instance is None:
        _video_processor_instance = VideoProcessor()
    return _video_processor_instance

# 편의 함수들
async def extract_audio_from_video(video_content: bytes, filename: str) -> Dict:
    """비디오에서 음성 추출 편의 함수"""
    processor = get_video_processor()
    return await processor.extract_audio_from_video(video_content, filename)

def check_video_support() -> Dict:
    """비디오 지원 상태 확인"""
    processor = get_video_processor()
    return {
        "supported_formats": processor.get_supported_formats(),
        "ffmpeg_available": processor.check_ffmpeg_available(),
        "python_version": "3.13+"
    }
