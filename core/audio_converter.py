#!/usr/bin/env python3
"""
오디오 파일 변환 전용 모듈
M4A, MP3, FLAC 등 다양한 오디오 포맷을 WAV로 변환
FFmpeg와 pydub을 활용한 강력한 변환 시스템
"""

import os
import tempfile
import subprocess
from typing import Optional, Dict, Any
from utils.logger import get_logger
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
    """오디오 파일 변환 클래스 - m4a 파일 처리 최적화"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.temp_files = []  # 임시 파일 추적
        
        # 지원하는 오디오 포맷 정의
        self.supported_formats = {
            '.m4a': 'M4A (AAC)',
            '.mp3': 'MP3',
            '.wav': 'WAV', 
            '.flac': 'FLAC',
            '.ogg': 'OGG',
            '.aac': 'AAC',
            '.wma': 'WMA'
        }
        
        # FFmpeg 가용성 확인
        self.ffmpeg_available = self._check_ffmpeg()
        self.logger.info(f"🔧 FFmpeg: {'사용 가능' if self.ffmpeg_available else '사용 불가'}")
        self.logger.info(f"🔧 pydub: {'사용 가능' if PYDUB_AVAILABLE else '사용 불가'}")
        self.logger.info(f"🔧 librosa: {'사용 가능' if LIBROSA_AVAILABLE else '사용 불가'}")
        
        # m4a 파일 특별 처리 설정
        self.m4a_optimized = True
        self.logger.info("🎵 M4A 파일 최적화 처리 활성화")
    
    def _setup_logging(self):
        """로깅 설정"""
        return get_logger(f'{__name__}.AudioConverter')
    
    def _check_ffmpeg(self) -> bool:
        """FFmpeg 설치 확인"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                    capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def convert_to_wav(self, input_path: str, target_sample_rate: int = 16000,
                      target_channels: int = 1, progress_callback=None) -> Optional[str]:
        """
        오디오 파일을 WAV로 변환 (m4a 파일 최적화)
        
        Args:
            input_path: 입력 파일 경로
            target_sample_rate: 대상 샘플링 레이트 (기본: 16kHz)
            target_channels: 대상 채널 수 (기본: 1 - 모노)
            progress_callback: 진행 상황 콜백 함수
            
        Returns:
            변환된 WAV 파일 경로 또는 None (실패시)
        """
        if not os.path.exists(input_path):
            self.logger.error(f"❌ 입력 파일이 존재하지 않음: {input_path}")
            return None
        
        input_ext = Path(input_path).suffix.lower()
        format_name = self.supported_formats.get(input_ext, input_ext.upper())
        
        self.logger.info(f"🔄 오디오 변환 시작: {format_name} → WAV")
        if progress_callback:
            progress_callback(0, "변환 준비 중...")
        
        # 파일 크기 확인
        file_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
        self.logger.info(f"📏 파일 크기: {file_size:.1f}MB")
        
        # 임시 WAV 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            output_path = temp_wav.name
        
        self.temp_files.append(output_path)
        
        # M4A 파일 특별 처리
        if input_ext == '.m4a' and self.m4a_optimized:
            if progress_callback:
                progress_callback(20, "M4A 파일 특별 처리 중...")
            success = self._convert_m4a_optimized(input_path, output_path, 
                                                 target_sample_rate, target_channels, progress_callback)
        else:
            # 일반 변환 방법 우선순위: FFmpeg > pydub > librosa
            success = False
            
            # 1. FFmpeg 시도 (가장 안정적)
            if self.ffmpeg_available and not success:
                if progress_callback:
                    progress_callback(30, "FFmpeg로 변환 중...")
                success = self._convert_with_ffmpeg(input_path, output_path, 
                                                    target_sample_rate, target_channels)
            
            # 2. pydub 시도 (Python 기반, 다양한 포맷 지원)
            if PYDUB_AVAILABLE and not success:
                if progress_callback:
                    progress_callback(60, "pydub로 변환 중...")
                success = self._convert_with_pydub(input_path, output_path, 
                                                   target_sample_rate, target_channels)
            
            # 3. librosa 시도 (과학적 오디오 처리)
            if LIBROSA_AVAILABLE and not success:
                if progress_callback:
                    progress_callback(80, "librosa로 변환 중...")
                success = self._convert_with_librosa(input_path, output_path, target_sample_rate)
        
        if success:
            if progress_callback:
                progress_callback(100, "변환 완료!")
            
            # 변환된 파일 검증
            if self._validate_wav_file(output_path):
                self.logger.info("✅ 오디오 변환 및 검증 완료")
                return output_path
            else:
                self.logger.error("❌ 변환된 파일 검증 실패")
                self.cleanup_temp_file(output_path)
                return None
        else:
            self.logger.error("❌ 모든 변환 방법 실패")
            if progress_callback:
                progress_callback(0, "변환 실패")
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
    
    def _convert_m4a_optimized(self, input_path: str, output_path: str,
                              sample_rate: int, channels: int, progress_callback=None) -> bool:
        """최적화된 M4A 파일 변환 (실패 문제 해결)"""
        
        self.logger.info("🎵 M4A 파일 최적화 변환 시작")
        
        # 방법1: FFmpeg로 M4A 전용 처리
        if self.ffmpeg_available:
            if progress_callback:
                progress_callback(40, "FFmpeg M4A 변환 중...")
            
            try:
                # M4A 파일에 최적화된 FFmpeg 명령
                cmd = [
                    'ffmpeg', '-i', input_path,
                    '-vn',                    # 비디오 스트림 비활성화
                    '-acodec', 'pcm_s16le',   # PCM 16-bit 인코딩
                    '-ar', str(sample_rate),  # 샘플링 레이트
                    '-ac', str(channels),     # 채널 수
                    '-f', 'wav',              # WAV 형식 강제
                    '-hide_banner',           # 불필요한 정보 숨김
                    '-loglevel', 'error',     # 에러만 표시
                    '-y',                     # 덮어쓰기 허용
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                
                if result.returncode == 0:
                    self.logger.info("✅ FFmpeg M4A 변환 성공")
                    return True
                else:
                    self.logger.warning(f"⚠️ FFmpeg M4A 변환 실패: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                self.logger.warning("⚠️ FFmpeg M4A 변환 시간 초과")
            except Exception as e:
                self.logger.warning(f"⚠️ FFmpeg M4A 변환 오류: {e}")
        
        # 방법2: pydub로 M4A 전용 처리
        if PYDUB_AVAILABLE:
            if progress_callback:
                progress_callback(70, "pydub M4A 변환 중...")
            
            try:
                self.logger.info("🔧 pydub M4A 변환 시도")
                
                # M4A 파일 로드 (명시적 포맷 지정)
                audio = AudioSegment.from_file(input_path, format="m4a")
                
                # 채널 수 조정
                if channels == 1 and audio.channels > 1:
                    audio = audio.set_channels(1)
                
                # 샘플링 레이트 조정
                if audio.frame_rate != sample_rate:
                    audio = audio.set_frame_rate(sample_rate)
                
                # WAV로 내보내기 (명시적 파라미터)
                audio.export(output_path, format="wav", 
                           parameters=["-acodec", "pcm_s16le"])
                
                self.logger.info("✅ pydub M4A 변환 성공")
                return True
                
            except Exception as e:
                self.logger.warning(f"⚠️ pydub M4A 변환 오류: {e}")
        
        # 방법3: librosa로 M4A 전용 처리
        if LIBROSA_AVAILABLE:
            if progress_callback:
                progress_callback(90, "librosa M4A 변환 중...")
            
            try:
                self.logger.info("🔧 librosa M4A 변환 시도")
                
                # M4A 파일 로드
                audio_data, original_sr = librosa.load(input_path, 
                                                      sr=sample_rate, 
                                                      mono=(channels==1))
                
                # WAV로 저장
                sf.write(output_path, audio_data, sample_rate, format='WAV', subtype='PCM_16')
                
                self.logger.info("✅ librosa M4A 변환 성공")
                return True
                
            except Exception as e:
                self.logger.warning(f"⚠️ librosa M4A 변환 오류: {e}")
        
        self.logger.error("❌ M4A 파일 최적화 변환 모든 방법 실패")
        return False
    
    def _validate_wav_file(self, wav_path: str) -> bool:
        """변환된 WAV 파일 검증"""
        try:
            if not os.path.exists(wav_path):
                return False
            
            # 파일 크기 확인
            file_size = os.path.getsize(wav_path)
            if file_size < 1024:  # 1KB 미만은 오류
                self.logger.warning(f"⚠️ WAV 파일이 너무 작음: {file_size} bytes")
                return False
            
            # pydub로 파일 검증
            if PYDUB_AVAILABLE:
                try:
                    audio = AudioSegment.from_wav(wav_path)
                    duration = len(audio)  # 밀리초
                    if duration < 100:  # 0.1초 미만은 오류
                        self.logger.warning(f"⚠️ WAV 파일 재생 시간이 너무 짧음: {duration}ms")
                        return False
                    
                    self.logger.info(f"✅ WAV 파일 검증 성공: {file_size/1024:.1f}KB, {duration/1000:.1f}초")
                    return True
                except Exception as e:
                    self.logger.warning(f"⚠️ pydub WAV 검증 실패: {e}")
            
            # 기본 검증: 파일 헤더 확인
            with open(wav_path, 'rb') as f:
                header = f.read(12)
                if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
                    self.logger.info(f"✅ WAV 파일 기본 검증 성공: {file_size/1024:.1f}KB")
                    return True
            
        except Exception as e:
            self.logger.error(f"❌ WAV 파일 검증 오류: {e}")
        
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
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_formats
    
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

def convert_audio_to_wav(input_path: str, target_sample_rate: int = 16000, 
                        progress_callback=None) -> Optional[str]:
    """간편 오디오 변환 함수 (M4A 최적화 포함)
    
    기존 코드에서 이렇게 사용:
    ```python
    from core.audio_converter import convert_audio_to_wav
    wav_path = convert_audio_to_wav("input.m4a")
    ```
    
    Args:
        input_path: 입력 오디오 파일 경로
        target_sample_rate: 대상 샘플링 레이트 (기본: 16kHz)
        progress_callback: 진행 상황 콜백 함수
        
    Returns:
        변환된 WAV 파일 경로 또는 None
    """
    return global_audio_converter.convert_to_wav(input_path, target_sample_rate, 1, progress_callback)

def get_audio_info(file_path: str) -> Dict[str, Any]:
    """간편 오디오 정보 함수"""
    return global_audio_converter.get_audio_info(file_path)

def is_audio_file(file_path: str) -> bool:
    """오디오 파일인지 확인"""
    return global_audio_converter.is_supported_format(file_path)

def convert_m4a_to_wav(input_path: str, target_sample_rate: int = 16000) -> Optional[str]:
    """M4A 파일 전용 변환 함수 (레거시 호환성)
    
    기존 코드에서 이렇게 사용:
    ```python
    from core.audio_converter import convert_m4a_to_wav
    wav_path = convert_m4a_to_wav("input.m4a")
    ```
    """
    if not input_path.lower().endswith('.m4a'):
        global_audio_converter.logger.warning(f"⚠️ M4A 파일이 아니지만 변환 시도: {input_path}")
    
    return global_audio_converter.convert_to_wav(input_path, target_sample_rate, 1)

def batch_convert_audio_files(file_paths: list, target_sample_rate: int = 16000, 
                             progress_callback=None) -> Dict[str, Optional[str]]:
    """여러 오디오 파일 일괄 변환
    
    Args:
        file_paths: 변환할 파일 경로 목록
        target_sample_rate: 대상 샘플링 레이트
        progress_callback: 진행 상황 콜백 (current, total, message)
        
    Returns:
        {input_path: output_path} 매핑 딕셔너리
    """
    results = {}
    total_files = len(file_paths)
    
    for i, file_path in enumerate(file_paths):
        if progress_callback:
            progress_callback(i, total_files, f"변환 중: {os.path.basename(file_path)}")
        
        try:
            wav_path = global_audio_converter.convert_to_wav(file_path, target_sample_rate)
            results[file_path] = wav_path
            
            if wav_path:
                global_audio_converter.logger.info(f"✅ 변환 성공: {os.path.basename(file_path)}")
            else:
                global_audio_converter.logger.error(f"❌ 변환 실패: {os.path.basename(file_path)}")
                
        except Exception as e:
            global_audio_converter.logger.error(f"❌ 변환 오류 {os.path.basename(file_path)}: {e}")
            results[file_path] = None
    
    if progress_callback:
        progress_callback(total_files, total_files, "모든 변환 완료")
    
    return results

def get_conversion_summary(results: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """변환 결과 요약
    
    Args:
        results: batch_convert_audio_files 결과
        
    Returns:
        변환 통계 데이터
    """
    total = len(results)
    success = sum(1 for v in results.values() if v is not None)
    failed = total - success
    
    success_files = [k for k, v in results.items() if v is not None]
    failed_files = [k for k, v in results.items() if v is None]
    
    return {
        'total_files': total,
        'success_count': success,
        'failed_count': failed,
        'success_rate': round((success / total * 100), 1) if total > 0 else 0,
        'success_files': success_files,
        'failed_files': failed_files
    }

if __name__ == "__main__":
    # 테스트 코드
    print("🎵 오디오 변환기 테스트 - M4A 최적화 버전")
    print("=" * 60)
    
    converter = AudioConverter()
    
    # 지원 포맷 표시
    print("📁 지원 포맷:")
    for ext, name in converter.supported_formats.items():
        print(f"  {ext} - {name}")
    
    # 시스템 상태 표시
    print(f"\n🔧 시스템 상태:")
    print(f"  FFmpeg: {'✅' if converter.ffmpeg_available else '❌'}")
    print(f"  pydub: {'✅' if PYDUB_AVAILABLE else '❌'}")
    print(f"  librosa: {'✅' if LIBROSA_AVAILABLE else '❌'}")
    
    # 테스트 파일 예시
    test_files = [
        "test_audio.m4a",
        "test_audio.mp3", 
        "test_audio.wav"
    ]
    
    print(f"\n🧪 테스트 파일 예시:")
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"  ✅ {test_file} - 사용 가능")
            
            # 파일 정보 표시
            info = converter.get_audio_info(test_file)
            print(f"     크기: {info['file_size_mb']}MB, 재생시간: {info['duration_seconds']}초")
            
            # 변환 테스트
            def progress_cb(percent, message):
                print(f"     {percent}% - {message}")
            
            wav_path = converter.convert_to_wav(test_file, progress_callback=progress_cb)
            if wav_path:
                print(f"     ✅ 변환 성공: {wav_path}")
                # 테스트 후 정리
                converter.cleanup_temp_file(wav_path)
            else:
                print(f"     ❌ 변환 실패")
        else:
            print(f"  ❌ {test_file} - 파일 없음")
    
    print("\n🎉 테스트 완료!")
    
    if os.path.exists(test_file):
        info = converter.get_audio_info(test_file)
        print(f"오디오 정보: {info}")
        
        converted = converter.convert_to_wav(test_file)
        if converted:
            print(f"변환 성공: {converted}")
            converter.cleanup_temp_file(converted)
        else:
            print("변환 실패")