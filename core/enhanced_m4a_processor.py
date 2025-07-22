#!/usr/bin/env python3
"""
M4A 파일 처리 전용 강화 모듈
메타데이터 검증, 다중 변환 시도, 오디오 데이터 검증 등을 통한 M4A 처리 안정성 극대화
"""

import os
import tempfile
import subprocess
import logging
import json
from typing import Optional, Dict, Any, List, Tuple
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

class EnhancedM4AProcessor:
    """M4A 파일 처리 강화 클래스"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.temp_files = []
        
        # 의존성 확인
        self.ffmpeg_available = self._check_ffmpeg()
        self.ffprobe_available = self._check_ffprobe()
        
        self.logger.info(f"🔧 FFmpeg: {'✅' if self.ffmpeg_available else '❌'}")
        self.logger.info(f"🔧 FFprobe: {'✅' if self.ffprobe_available else '❌'}")
        self.logger.info(f"🔧 pydub: {'✅' if PYDUB_AVAILABLE else '❌'}")
        self.logger.info(f"🔧 librosa: {'✅' if LIBROSA_AVAILABLE else '❌'}")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.EnhancedM4AProcessor')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - M4A - %(levelname)s - %(message)s')
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
    
    def _check_ffprobe(self) -> bool:
        """FFprobe 설치 확인"""
        try:
            result = subprocess.run(['ffprobe', '-version'], 
                                    capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def analyze_m4a_file(self, file_path: str) -> Dict[str, Any]:
        """M4A 파일 심층 분석"""
        analysis = {
            "file_exists": False,
            "file_size_mb": 0,
            "is_readable": False,
            "has_metadata": False,
            "has_audio_stream": False,
            "codec": "unknown",
            "duration_seconds": 0,
            "sample_rate": 0,
            "channels": 0,
            "bitrate": 0,
            "corruption_detected": False,
            "recommended_method": "ffmpeg",
            "issues": []
        }
        
        if not os.path.exists(file_path):
            analysis["issues"].append("파일이 존재하지 않음")
            return analysis
        
        analysis["file_exists"] = True
        analysis["file_size_mb"] = round(os.path.getsize(file_path) / (1024 * 1024), 3)
        
        # 빈 파일 확인
        if analysis["file_size_mb"] < 0.001:  # 1KB 미만
            analysis["issues"].append("파일이 너무 작음 (빈 파일 의심)")
            return analysis
        
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)
            analysis["is_readable"] = True
        except Exception as e:
            analysis["issues"].append(f"파일 읽기 실패: {str(e)}")
            return analysis
        
        # FFprobe로 메타데이터 분석
        if self.ffprobe_available:
            metadata = self._analyze_with_ffprobe(file_path)
            analysis.update(metadata)
        
        # pydub으로 보완 분석
        if PYDUB_AVAILABLE and not analysis["has_audio_stream"]:
            pydub_info = self._analyze_with_pydub(file_path)
            if pydub_info["success"]:
                analysis.update(pydub_info)
        
        # 문제점 기반 변환 방법 추천
        analysis["recommended_method"] = self._recommend_conversion_method(analysis)
        
        return analysis
    
    def _analyze_with_ffprobe(self, file_path: str) -> Dict[str, Any]:
        """FFprobe를 사용한 상세 분석"""
        result = {
            "has_metadata": False,
            "has_audio_stream": False,
            "codec": "unknown",
            "duration_seconds": 0,
            "sample_rate": 0,
            "channels": 0,
            "bitrate": 0
        }
        
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', file_path
            ]
            
            probe_result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if probe_result.returncode != 0:
                self.logger.warning(f"FFprobe 분석 실패: {probe_result.stderr}")
                return result
            
            data = json.loads(probe_result.stdout)
            
            # 포맷 정보
            if 'format' in data:
                format_info = data['format']
                result["has_metadata"] = True
                
                if 'duration' in format_info:
                    result["duration_seconds"] = float(format_info['duration'])
                if 'bit_rate' in format_info:
                    result["bitrate"] = int(format_info['bit_rate'])
            
            # 스트림 정보
            if 'streams' in data:
                for stream in data['streams']:
                    if stream.get('codec_type') == 'audio':
                        result["has_audio_stream"] = True
                        result["codec"] = stream.get('codec_name', 'unknown')
                        result["sample_rate"] = int(stream.get('sample_rate', 0))
                        result["channels"] = int(stream.get('channels', 0))
                        break
            
        except json.JSONDecodeError:
            self.logger.warning("FFprobe JSON 파싱 실패")
        except subprocess.TimeoutExpired:
            self.logger.warning("FFprobe 분석 시간 초과")
        except Exception as e:
            self.logger.warning(f"FFprobe 분석 오류: {e}")
        
        return result
    
    def _analyze_with_pydub(self, file_path: str) -> Dict[str, Any]:
        """pydub을 사용한 분석"""
        result = {
            "success": False,
            "has_audio_stream": False,
            "duration_seconds": 0,
            "sample_rate": 0,
            "channels": 0
        }
        
        try:
            audio = AudioSegment.from_file(file_path)
            result["success"] = True
            result["has_audio_stream"] = True
            result["duration_seconds"] = len(audio) / 1000.0
            result["sample_rate"] = audio.frame_rate
            result["channels"] = audio.channels
            
        except Exception as e:
            self.logger.warning(f"pydub 분석 오류: {e}")
        
        return result
    
    def _recommend_conversion_method(self, analysis: Dict[str, Any]) -> str:
        """분석 결과를 바탕으로 변환 방법 추천"""
        issues = analysis.get("issues", [])
        
        # 심각한 문제가 있는 경우
        if not analysis["has_audio_stream"]:
            return "repair_then_convert"
        
        # 메타데이터 문제가 있는 경우
        if not analysis["has_metadata"] or analysis["corruption_detected"]:
            return "repair_then_convert"
        
        # 일반적인 경우
        if self.ffmpeg_available:
            return "ffmpeg"
        elif PYDUB_AVAILABLE:
            return "pydub"
        elif LIBROSA_AVAILABLE:
            return "librosa"
        else:
            return "none_available"
    
    def process_m4a_to_wav(self, input_path: str, 
                          target_sample_rate: int = 16000,
                          target_channels: int = 1) -> Optional[str]:
        """M4A 파일을 WAV로 변환 (강화된 처리)"""
        
        self.logger.info(f"🎵 M4A 처리 시작: {os.path.basename(input_path)}")
        
        # 1. 파일 분석
        analysis = self.analyze_m4a_file(input_path)
        
        # 분석 결과 로깅
        self.logger.info(f"📊 파일 크기: {analysis['file_size_mb']}MB")
        self.logger.info(f"🎤 오디오 스트림: {'✅' if analysis['has_audio_stream'] else '❌'}")
        self.logger.info(f"⚙️ 추천 방법: {analysis['recommended_method']}")
        
        if analysis["issues"]:
            for issue in analysis["issues"]:
                self.logger.warning(f"⚠️ 문제점: {issue}")
        
        # 2. 변환 불가능한 경우 체크
        if not analysis["file_exists"] or not analysis["is_readable"]:
            self.logger.error("❌ 파일 접근 불가")
            return None
        
        if analysis["file_size_mb"] < 0.001:
            self.logger.error("❌ 빈 파일")
            return None
        
        # 3. 변환 시도 (다중 방법)
        conversion_methods = self._get_conversion_methods(analysis["recommended_method"])
        
        for method_name, method_func in conversion_methods:
            self.logger.info(f"🔄 {method_name} 시도...")
            
            try:
                result = method_func(input_path, target_sample_rate, target_channels)
                if result and os.path.exists(result) and os.path.getsize(result) > 0:
                    self.logger.info(f"✅ {method_name} 성공")
                    return result
                else:
                    self.logger.warning(f"⚠️ {method_name} 실패 (결과 파일 문제)")
            
            except Exception as e:
                self.logger.warning(f"⚠️ {method_name} 예외: {e}")
        
        self.logger.error("❌ 모든 변환 방법 실패")
        return None
    
    def _get_conversion_methods(self, recommended: str) -> List[Tuple[str, callable]]:
        """변환 방법 목록 반환 (우선순위 순)"""
        methods = []
        
        if recommended == "repair_then_convert":
            if self.ffmpeg_available:
                methods.append(("FFmpeg 복구 변환", self._repair_and_convert_with_ffmpeg))
            if PYDUB_AVAILABLE:
                methods.append(("pydub 안전 변환", self._safe_convert_with_pydub))
        
        # 기본 변환 방법들
        if self.ffmpeg_available:
            methods.append(("FFmpeg 표준 변환", self._convert_with_ffmpeg))
            methods.append(("FFmpeg 호환 변환", self._convert_with_ffmpeg_compatible))
        
        if PYDUB_AVAILABLE:
            methods.append(("pydub 변환", self._convert_with_pydub))
        
        if LIBROSA_AVAILABLE:
            methods.append(("librosa 변환", self._convert_with_librosa))
        
        return methods
    
    def _repair_and_convert_with_ffmpeg(self, input_path: str, 
                                       sample_rate: int, channels: int) -> Optional[str]:
        """FFmpeg을 사용한 복구 후 변환"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            output_path = temp_wav.name
        self.temp_files.append(output_path)
        
        try:
            cmd = [
                'ffmpeg', '-err_detect', 'ignore_err', '-i', input_path,
                '-c:a', 'pcm_s16le', '-ar', str(sample_rate), '-ac', str(channels),
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts+igndts',
                '-y', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            return output_path if result.returncode == 0 else None
            
        except Exception:
            return None
    
    def _convert_with_ffmpeg_compatible(self, input_path: str, 
                                       sample_rate: int, channels: int) -> Optional[str]:
        """FFmpeg 호환성 모드 변환"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            output_path = temp_wav.name
        self.temp_files.append(output_path)
        
        try:
            cmd = [
                'ffmpeg', '-i', input_path,
                '-acodec', 'pcm_s16le', '-ar', str(sample_rate), '-ac', str(channels),
                '-f', 'wav', '-y', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            return output_path if result.returncode == 0 else None
            
        except Exception:
            return None
    
    def _convert_with_ffmpeg(self, input_path: str, 
                           sample_rate: int, channels: int) -> Optional[str]:
        """표준 FFmpeg 변환"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            output_path = temp_wav.name
        self.temp_files.append(output_path)
        
        try:
            cmd = [
                'ffmpeg', '-i', input_path,
                '-acodec', 'pcm_s16le', '-ar', str(sample_rate), '-ac', str(channels),
                '-y', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return output_path if result.returncode == 0 else None
            
        except Exception:
            return None
    
    def _safe_convert_with_pydub(self, input_path: str, 
                                sample_rate: int, channels: int) -> Optional[str]:
        """pydub 안전 변환"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            output_path = temp_wav.name
        self.temp_files.append(output_path)
        
        try:
            # 다중 포맷 시도
            audio = None
            formats_to_try = ['m4a', 'mp4', 'aac']
            
            for fmt in formats_to_try:
                try:
                    audio = AudioSegment.from_file(input_path, format=fmt)
                    break
                except:
                    continue
            
            if audio is None:
                audio = AudioSegment.from_file(input_path)
            
            # 변환 적용
            if channels == 1 and audio.channels > 1:
                audio = audio.set_channels(1)
            if audio.frame_rate != sample_rate:
                audio = audio.set_frame_rate(sample_rate)
            
            audio.export(output_path, format="wav")
            return output_path
            
        except Exception:
            return None
    
    def _convert_with_pydub(self, input_path: str, 
                           sample_rate: int, channels: int) -> Optional[str]:
        """표준 pydub 변환"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            output_path = temp_wav.name
        self.temp_files.append(output_path)
        
        try:
            audio = AudioSegment.from_file(input_path)
            
            if channels == 1 and audio.channels > 1:
                audio = audio.set_channels(1)
            if audio.frame_rate != sample_rate:
                audio = audio.set_frame_rate(sample_rate)
            
            audio.export(output_path, format="wav")
            return output_path
            
        except Exception:
            return None
    
    def _convert_with_librosa(self, input_path: str, sample_rate: int) -> Optional[str]:
        """librosa 변환"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            output_path = temp_wav.name
        self.temp_files.append(output_path)
        
        try:
            audio_data, _ = librosa.load(input_path, sr=sample_rate, mono=True)
            sf.write(output_path, audio_data, sample_rate)
            return output_path
            
        except Exception:
            return None
    
    def cleanup_temp_files(self):
        """임시 파일 정리"""
        for temp_file in self.temp_files[:]:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    self.temp_files.remove(temp_file)
            except Exception as e:
                self.logger.warning(f"임시 파일 정리 실패: {e}")
    
    def __del__(self):
        """소멸자"""
        self.cleanup_temp_files()

# 전역 프로세서 인스턴스
global_m4a_processor = EnhancedM4AProcessor()

def process_m4a_file(input_path: str, target_sample_rate: int = 16000) -> Optional[str]:
    """M4A 파일 처리 함수"""
    return global_m4a_processor.process_m4a_to_wav(input_path, target_sample_rate)

def analyze_m4a_file(file_path: str) -> Dict[str, Any]:
    """M4A 파일 분석 함수"""
    return global_m4a_processor.analyze_m4a_file(file_path)

if __name__ == "__main__":
    # 테스트 코드
    processor = EnhancedM4AProcessor()
    
    test_file = "test.m4a"
    if os.path.exists(test_file):
        analysis = processor.analyze_m4a_file(test_file)
        print(f"분석 결과: {json.dumps(analysis, indent=2, ensure_ascii=False)}")
        
        converted = processor.process_m4a_to_wav(test_file)
        if converted:
            print(f"변환 성공: {converted}")
            processor.cleanup_temp_files()
        else:
            print("변환 실패")