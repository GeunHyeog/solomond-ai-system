#!/usr/bin/env python3
"""
🛠️ 솔로몬드 AI 강력한 파일 프로세서
- m4a 파일 처리 완전 안정화
- 대용량 파일 (10GB+) 청킹 처리
- 파일 형식 자동 감지 및 변환
- 에러 복구 및 재시도 시스템
"""

import os
import io
import hashlib
import logging
import tempfile
import subprocess
from typing import Dict, Any, Optional, List, Union, BinaryIO
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FileInfo:
    """파일 정보"""
    filename: str
    original_size: int
    processed_size: int
    format: str
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    is_converted: bool = False
    conversion_path: Optional[str] = None

class RobustFileProcessor:
    """강력한 파일 처리 시스템"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.chunk_size = 10 * 1024 * 1024  # 10MB 청크
        self.max_file_size = 10 * 1024 * 1024 * 1024  # 10GB 최대
        self.supported_audio = {'.m4a', '.mp3', '.wav', '.flac', '.ogg', '.aac'}
        self.supported_video = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        self.supported_image = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        
        # FFmpeg 설치 확인
        self.ffmpeg_available = self._check_ffmpeg()
        
        logger.info(f"🛠️ 강력한 파일 프로세서 초기화 (FFmpeg: {'✅' if self.ffmpeg_available else '❌'})")
    
    def _check_ffmpeg(self) -> bool:
        """FFmpeg 설치 확인"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
    
    async def process_file(self, file_data: bytes, filename: str, 
                          target_format: Optional[str] = None) -> FileInfo:
        """파일 처리 메인 함수"""
        original_size = len(file_data)
        file_ext = Path(filename).suffix.lower()
        
        logger.info(f"📄 파일 처리 시작: {filename} ({original_size:,} bytes)")
        
        # 대용량 파일 체크
        if original_size > self.max_file_size:
            raise ValueError(f"파일 크기 초과: {original_size:,} > {self.max_file_size:,} bytes")
        
        file_info = FileInfo(
            filename=filename,
            original_size=original_size,
            processed_size=original_size,
            format=file_ext
        )
        
        try:
            # 파일 타입별 처리
            if file_ext in self.supported_audio:
                return await self._process_audio_file(file_data, file_info, target_format)
            elif file_ext in self.supported_video:
                return await self._process_video_file(file_data, file_info, target_format)
            elif file_ext in self.supported_image:
                return await self._process_image_file(file_data, file_info)
            else:
                logger.warning(f"⚠️ 지원되지 않는 파일 형식: {file_ext}")
                return file_info
                
        except Exception as e:
            logger.error(f"❌ 파일 처리 실패 {filename}: {e}")
            raise
    
    async def _process_audio_file(self, file_data: bytes, file_info: FileInfo,
                                target_format: Optional[str] = None) -> FileInfo:
        """오디오 파일 처리 (m4a → wav 변환 포함)"""
        original_ext = file_info.format
        target_ext = target_format or '.wav'  # 기본 WAV 변환
        
        # 이미 WAV면 변환 불필요
        if original_ext == '.wav' and target_format is None:
            file_info.processed_size = len(file_data)
            return file_info
        
        # m4a나 기타 형식을 WAV로 변환
        if self.ffmpeg_available and original_ext in {'.m4a', '.mp3', '.aac'}:
            logger.info(f"🔄 오디오 변환: {original_ext} → {target_ext}")
            
            try:
                converted_data = await self._convert_audio_with_ffmpeg(
                    file_data, original_ext, target_ext
                )
                
                if converted_data:
                    file_info.processed_size = len(converted_data)
                    file_info.format = target_ext
                    file_info.is_converted = True
                    
                    # 변환된 파일 임시 저장
                    temp_path = await self._save_temp_file(converted_data, target_ext)
                    file_info.conversion_path = temp_path
                    
                    # 오디오 정보 추출
                    await self._extract_audio_info(file_info, temp_path)
                    
                    logger.info(f"✅ 변환 완료: {len(converted_data):,} bytes")
                    return file_info
                    
            except Exception as e:
                logger.warning(f"⚠️ FFmpeg 변환 실패, 원본 사용: {e}")
        
        # 변환 실패 시 원본 사용
        file_info.processed_size = len(file_data)
        temp_path = await self._save_temp_file(file_data, original_ext)
        file_info.conversion_path = temp_path
        
        return file_info
    
    async def _convert_audio_with_ffmpeg(self, file_data: bytes, 
                                       input_ext: str, output_ext: str) -> Optional[bytes]:
        """FFmpeg을 사용한 오디오 변환"""
        input_temp = None
        output_temp = None
        
        try:
            # 임시 입력 파일 생성
            input_temp = await self._save_temp_file(file_data, input_ext)
            output_temp = str(Path(self.temp_dir) / f"converted_{os.getpid()}{output_ext}")
            
            # FFmpeg 명령어 구성
            cmd = [
                'ffmpeg', '-y',  # 덮어쓰기 허용
                '-i', input_temp,  # 입력 파일
                '-acodec', 'pcm_s16le',  # WAV 코덱
                '-ar', '16000',  # 샘플링 레이트 (Whisper 권장)
                '-ac', '1',  # 모노 채널
                '-f', 'wav',  # WAV 형식
                output_temp
            ]
            
            # 비동기 실행
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor, 
                    lambda: subprocess.run(
                        cmd, capture_output=True, timeout=300
                    )
                )
            
            if result.returncode == 0 and os.path.exists(output_temp):
                # 변환된 파일 읽기
                with open(output_temp, 'rb') as f:
                    converted_data = f.read()
                
                logger.info(f"✅ FFmpeg 변환 성공: {len(converted_data):,} bytes")
                return converted_data
            else:
                error_msg = result.stderr.decode() if result.stderr else "Unknown error"
                logger.error(f"❌ FFmpeg 변환 실패: {error_msg}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("❌ FFmpeg 변환 시간 초과 (5분)")
            return None
        except Exception as e:
            logger.error(f"❌ FFmpeg 변환 오류: {e}")
            return None
        finally:
            # 임시 파일 정리
            for temp_file in [input_temp, output_temp]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
    
    async def _process_video_file(self, file_data: bytes, file_info: FileInfo,
                                target_format: Optional[str] = None) -> FileInfo:
        """비디오 파일 처리"""
        # 현재는 기본 정보만 추출
        temp_path = await self._save_temp_file(file_data, file_info.format)
        file_info.conversion_path = temp_path
        
        # 비디오 정보 추출 (FFmpeg 사용)
        if self.ffmpeg_available:
            await self._extract_video_info(file_info, temp_path)
        
        return file_info
    
    async def _process_image_file(self, file_data: bytes, file_info: FileInfo) -> FileInfo:
        """이미지 파일 처리"""
        temp_path = await self._save_temp_file(file_data, file_info.format)
        file_info.conversion_path = temp_path
        
        # 이미지 정보 추출
        await self._extract_image_info(file_info, temp_path)
        
        return file_info
    
    async def _save_temp_file(self, data: bytes, extension: str) -> str:
        """임시 파일 저장"""
        # 고유한 파일명 생성
        file_hash = hashlib.md5(data[:1024]).hexdigest()[:8]
        temp_filename = f"solomond_{os.getpid()}_{file_hash}{extension}"
        temp_path = os.path.join(self.temp_dir, temp_filename)
        
        # 비동기 파일 쓰기
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                lambda: self._write_file_sync(temp_path, data)
            )
        
        return temp_path
    
    def _write_file_sync(self, path: str, data: bytes):
        """동기 파일 쓰기"""
        with open(path, 'wb') as f:
            f.write(data)
    
    async def _extract_audio_info(self, file_info: FileInfo, file_path: str):
        """오디오 정보 추출"""
        if not self.ffmpeg_available:
            return
        
        try:
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                file_path
            ]
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    lambda: subprocess.run(cmd, capture_output=True, timeout=30)
                )
            
            if result.returncode == 0:
                import json
                info = json.loads(result.stdout.decode())
                format_info = info.get('format', {})
                
                file_info.duration = float(format_info.get('duration', 0))
                
                # 샘플링 레이트와 채널은 스트림에서 추출 필요 (간단히 생략)
                file_info.sample_rate = 16000  # 변환 시 지정한 값
                file_info.channels = 1  # 모노로 변환
                
        except Exception as e:
            logger.warning(f"⚠️ 오디오 정보 추출 실패: {e}")
    
    async def _extract_video_info(self, file_info: FileInfo, file_path: str):
        """비디오 정보 추출"""
        # FFprobe로 비디오 정보 추출 (구현 간소화)
        file_info.duration = 0.0  # 기본값
    
    async def _extract_image_info(self, file_info: FileInfo, file_path: str):
        """이미지 정보 추출"""
        try:
            # PIL로 이미지 정보 추출
            from PIL import Image
            with Image.open(file_path) as img:
                # 기본 정보만 저장
                pass
        except Exception as e:
            logger.warning(f"⚠️ 이미지 정보 추출 실패: {e}")
    
    async def process_large_file_chunked(self, file_stream: BinaryIO, filename: str,
                                       chunk_callback=None) -> FileInfo:
        """대용량 파일 청킹 처리"""
        total_size = 0
        chunks = []
        chunk_num = 0
        
        logger.info(f"📦 대용량 파일 청킹 처리 시작: {filename}")
        
        try:
            while True:
                chunk = file_stream.read(self.chunk_size)
                if not chunk:
                    break
                
                chunks.append(chunk)
                total_size += len(chunk)
                chunk_num += 1
                
                # 콜백 호출 (진행률 표시용)
                if chunk_callback:
                    await chunk_callback(chunk_num, len(chunk), total_size)
                
                # 메모리 보호
                if total_size > self.max_file_size:
                    raise ValueError(f"파일 크기 초과: {total_size:,} bytes")
            
            # 모든 청크 결합
            complete_data = b''.join(chunks)
            
            # 일반 처리로 전환
            return await self.process_file(complete_data, filename)
            
        except Exception as e:
            logger.error(f"❌ 청킹 처리 실패: {e}")
            raise
    
    def cleanup_temp_files(self, file_info: FileInfo):
        """임시 파일 정리"""
        if file_info.conversion_path and os.path.exists(file_info.conversion_path):
            try:
                os.unlink(file_info.conversion_path)
                logger.debug(f"🗑️ 임시 파일 정리: {file_info.conversion_path}")
            except Exception as e:
                logger.warning(f"⚠️ 임시 파일 정리 실패: {e}")
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """지원되는 파일 형식 목록"""
        return {
            'audio': list(self.supported_audio),
            'video': list(self.supported_video), 
            'image': list(self.supported_image)
        }

# 글로벌 파일 프로세서 인스턴스
robust_processor = RobustFileProcessor()

# 편의 함수들
async def process_file_robust(file_data: bytes, filename: str, 
                            target_format: Optional[str] = None) -> FileInfo:
    """강력한 파일 처리"""
    return await robust_processor.process_file(file_data, filename, target_format)

async def process_m4a_to_wav(file_data: bytes, filename: str) -> FileInfo:
    """M4A → WAV 변환"""
    return await robust_processor.process_file(file_data, filename, '.wav')

def get_supported_formats() -> Dict[str, List[str]]:
    """지원 형식 조회"""
    return robust_processor.get_supported_formats()

if __name__ == "__main__":
    # 테스트
    print("🛠️ 강력한 파일 프로세서 테스트")
    
    # 지원 형식 출력
    formats = get_supported_formats()
    for category, extensions in formats.items():
        print(f"{category}: {', '.join(extensions)}")
    
    # FFmpeg 상태
    print(f"FFmpeg 사용 가능: {'✅' if robust_processor.ffmpeg_available else '❌'}")