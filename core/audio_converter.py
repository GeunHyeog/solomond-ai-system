#!/usr/bin/env python3
"""
오디오 파일 변환 전용 모듈
M4A, MP3, FLAC 등 다양한 오디오 포맷을 WAV로 변환
FFmpeg와 pydub을 활용한 강력한 변환 시스템
"""

import os
import tempfile
import subprocess
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import shutil

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

class AudioConverter:
    """오디오 파일 변환 클래스"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.temp_files = []  # 임시 파일 추적
        
        # FFmpeg 가용성 확인
        self.ffmpeg_available = self._check_ffmpeg()
        self.logger.info(f"🔧 FFmpeg: {'사용 가능' if self.ffmpeg_available else '사용 불가'}")
        self.logger.info(f"🔧 pydub: {'사용 가능' if PYDUB_AVAILABLE else '사용 불가'}")
        self.logger.info(f"🔧 librosa: {'사용 가능' if LIBROSA_AVAILABLE else '사용 불가'}")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.AudioConverter')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _check_ffmpeg(self) -> bool:
        """FFmpeg 설치 확인"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                    capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def convert_to_wav(self, input_path: str, target_sample_rate: int = 16000,
                      target_channels: int = 1) -> Optional[str]:
        """
        오디오 파일을 WAV로 변환
        
        Args:
            input_path: 입력 파일 경로
            target_sample_rate: 대상 샘플링 레이트 (기본: 16kHz)
            target_channels: 대상 채널 수 (기본: 1 - 모노)
            
        Returns:
            변환된 WAV 파일 경로 또는 None (실패시)
        """
        if not os.path.exists(input_path):
            self.logger.error(f"❌ 입력 파일이 존재하지 않음: {input_path}")
            return None
        
        input_ext = Path(input_path).suffix.lower()
        self.logger.info(f"🔄 오디오 변환 시작: {input_ext} → WAV")
        
        # 임시 WAV 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            output_path = temp_wav.name
        
        self.temp_files.append(output_path)
        
        # 변환 방법 우선순위: FFmpeg > pydub > librosa
        success = False
        
        # 1. FFmpeg 시도 (가장 안정적)
        if self.ffmpeg_available and not success:
            success = self._convert_with_ffmpeg(input_path, output_path, 
                                                target_sample_rate, target_channels)
        
        # 2. pydub 시도 (Python 기반, 다양한 포맷 지원)
        if PYDUB_AVAILABLE and not success:
            success = self._convert_with_pydub(input_path, output_path, 
                                               target_sample_rate, target_channels)
        
        # 3. librosa 시도 (과학적 오디오 처리)
        if LIBROSA_AVAILABLE and not success:
            success = self._convert_with_librosa(input_path, output_path, target_sample_rate)
        
        if success:
            self.logger.info("✅ 오디오 변환 완료")
            return output_path
        else:
            self.logger.error("❌ 모든 변환 방법 실패")
            self.cleanup_temp_file(output_path)
            return None
    
    def _convert_with_ffmpeg(self, input_path: str, output_path: str,
                            sample_rate: int, channels: int) -> bool:
        """FFmpeg를 사용한 변환"""
        try:
            cmd = [
                'ffmpeg', '-i', input_path,
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', str(sample_rate),  # 샘플링 레이트
                '-ac', str(channels),     # 채널 수
                '-y',                     # 덮어쓰기 허용
                output_path
            ]
            
            self.logger.info("🔧 FFmpeg 변환 시도...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.logger.info("✅ FFmpeg 변환 성공")
                return True
            else:
                self.logger.warning(f"⚠️ FFmpeg 변환 실패: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.warning("⚠️ FFmpeg 변환 시간 초과")
            return False
        except Exception as e:
            self.logger.warning(f"⚠️ FFmpeg 변환 오류: {e}")
            return False
    
    def _convert_with_pydub(self, input_path: str, output_path: str,
                           sample_rate: int, channels: int) -> bool:
        """pydub를 사용한 변환"""
        try:
            self.logger.info("🔧 pydub 변환 시도...")
            
            # 오디오 로드
            audio = AudioSegment.from_file(input_path)
            
            # 채널 수 조정
            if channels == 1 and audio.channels > 1:
                audio = audio.set_channels(1)
            
            # 샘플링 레이트 조정
            if audio.frame_rate != sample_rate:
                audio = audio.set_frame_rate(sample_rate)
            
            # WAV로 내보내기
            audio.export(output_path, format="wav")
            
            self.logger.info("✅ pydub 변환 성공")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ pydub 변환 오류: {e}")
            return False
    
    def _convert_with_librosa(self, input_path: str, output_path: str,
                             sample_rate: int) -> bool:
        """librosa를 사용한 변환"""
        try:
            self.logger.info("🔧 librosa 변환 시도...")
            
            # 오디오 로드
            audio_data, original_sr = librosa.load(input_path, sr=sample_rate, mono=True)
            
            # WAV로 저장
            sf.write(output_path, audio_data, sample_rate)
            
            self.logger.info("✅ librosa 변환 성공")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ librosa 변환 오류: {e}")
            return False
    
    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """오디오 파일 정보 추출"""
        info = {
            "file_size_mb": 0,
            "duration_seconds": 0,
            "sample_rate": 0,
            "channels": 0,
            "format": "unknown",
            "is_valid": False
        }
        
        try:
            # 파일 크기
            info["file_size_mb"] = round(os.path.getsize(file_path) / (1024 * 1024), 2)
            info["format"] = Path(file_path).suffix.lower()
            
            # pydub로 오디오 정보 추출
            if PYDUB_AVAILABLE:
                try:
                    audio = AudioSegment.from_file(file_path)
                    info["duration_seconds"] = len(audio) / 1000.0
                    info["sample_rate"] = audio.frame_rate
                    info["channels"] = audio.channels
                    info["is_valid"] = True
                except:
                    pass
            
            # librosa로 보완 시도
            if not info["is_valid"] and LIBROSA_AVAILABLE:
                try:
                    audio_data, sample_rate = librosa.load(file_path, sr=None)
                    info["duration_seconds"] = len(audio_data) / sample_rate
                    info["sample_rate"] = sample_rate
                    info["channels"] = 1  # librosa는 기본적으로 모노로 로드
                    info["is_valid"] = True
                except:
                    pass
            
        except Exception as e:
            self.logger.warning(f"⚠️ 오디오 정보 추출 실패: {e}")
        
        return info
    
    def is_supported_format(self, file_path: str) -> bool:
        """지원하는 오디오 포맷인지 확인"""
        supported_formats = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma'}
        file_ext = Path(file_path).suffix.lower()
        return file_ext in supported_formats
    
    def cleanup_temp_file(self, file_path: str):
        """임시 파일 정리"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                if file_path in self.temp_files:
                    self.temp_files.remove(file_path)
                self.logger.info(f"🧹 임시 파일 정리됨: {os.path.basename(file_path)}")
        except Exception as e:
            self.logger.warning(f"⚠️ 임시 파일 정리 실패: {e}")
    
    def cleanup_all_temp_files(self):
        """모든 임시 파일 정리"""
        for temp_file in self.temp_files[:]:  # 복사본으로 반복
            self.cleanup_temp_file(temp_file)
    
    def __del__(self):
        """소멸자에서 임시 파일 정리"""
        self.cleanup_all_temp_files()

# 전역 컨버터 인스턴스
global_audio_converter = AudioConverter()

def convert_audio_to_wav(input_path: str, target_sample_rate: int = 16000) -> Optional[str]:
    """간편 오디오 변환 함수"""
    return global_audio_converter.convert_to_wav(input_path, target_sample_rate)

def get_audio_info(file_path: str) -> Dict[str, Any]:
    """간편 오디오 정보 함수"""
    return global_audio_converter.get_audio_info(file_path)

def is_audio_file(file_path: str) -> bool:
    """오디오 파일인지 확인"""
    return global_audio_converter.is_supported_format(file_path)

if __name__ == "__main__":
    # 테스트 코드
    converter = AudioConverter()
    test_file = "/path/to/test.m4a"  # 테스트 파일 경로
    
    if os.path.exists(test_file):
        info = converter.get_audio_info(test_file)
        print(f"오디오 정보: {info}")
        
        converted = converter.convert_to_wav(test_file)
        if converted:
            print(f"변환 성공: {converted}")
            converter.cleanup_temp_file(converted)
        else:
            print("변환 실패")