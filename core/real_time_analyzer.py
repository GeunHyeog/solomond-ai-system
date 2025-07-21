#!/usr/bin/env python3
"""
실시간 스트리밍 분석 모듈 - 마이크 입력 실시간 처리
"""

import os
import threading
import time
import queue
import logging
from typing import Dict, Any, List, Optional, Callable
import tempfile
from datetime import datetime
import json

# 실시간 오디오 처리를 위한 라이브러리들
try:
    import pyaudio
    import wave
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class RealTimeAnalyzer:
    """실시간 분석 클래스"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 오디오 설정
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        self.format = None
        
        if PYAUDIO_AVAILABLE:
            self.format = pyaudio.paInt16
        
        # 버퍼 설정
        self.audio_buffer = queue.Queue()
        self.analysis_buffer = queue.Queue()
        self.max_buffer_size = 10  # 최대 10초 버퍼
        
        # 상태 관리
        self.is_recording = False
        self.is_analyzing = False
        self.recording_thread = None
        self.analysis_thread = None
        
        # VAD (Voice Activity Detection) 설정
        self.vad = None
        if WEBRTC_VAD_AVAILABLE:
            self.vad = webrtcvad.Vad(2)  # 중간 민감도
        
        # 분석 콜백
        self.analysis_callbacks = []
        
        # 통계
        self.session_stats = {
            "start_time": None,
            "total_chunks": 0,
            "speech_chunks": 0,
            "silence_chunks": 0,
            "total_analysis_time": 0
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
        if PYAUDIO_AVAILABLE:
            self.logger.info("[INFO] PyAudio 사용 가능 - 마이크 입력 지원")
        else:
            self.logger.warning("[WARNING] PyAudio 미설치 - 마이크 입력 불가")
        
        if WEBRTC_VAD_AVAILABLE:
            self.logger.info("[INFO] WebRTC VAD 사용 가능 - 음성 감지 지원")
        else:
            self.logger.warning("[WARNING] WebRTC VAD 미설치 - 음성 감지 제한됨")
        
        if NUMPY_AVAILABLE:
            self.logger.info("[INFO] NumPy 사용 가능 - 오디오 처리 지원")
        else:
            self.logger.warning("[WARNING] NumPy 미설치 - 오디오 처리 제한됨")
    
    def is_available(self) -> bool:
        """실시간 분석 시스템 사용 가능 여부"""
        return PYAUDIO_AVAILABLE and NUMPY_AVAILABLE
    
    def get_audio_devices(self) -> Dict[str, Any]:
        """사용 가능한 오디오 장치 목록"""
        if not PYAUDIO_AVAILABLE:
            return {
                "status": "error",
                "error": "PyAudio가 설치되지 않음"
            }
        
        try:
            p = pyaudio.PyAudio()
            devices = []
            
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:  # 입력 장치만
                    devices.append({
                        "index": i,
                        "name": device_info['name'],
                        "channels": device_info['maxInputChannels'],
                        "sample_rate": device_info['defaultSampleRate']
                    })
            
            p.terminate()
            
            return {
                "status": "success",
                "devices": devices,
                "default_device": p.get_default_input_device_info()['index'] if devices else None
            }
            
        except Exception as e:
            self.logger.error(f"[ERROR] 오디오 장치 조회 실패: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def add_analysis_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """분석 결과 콜백 추가"""
        self.analysis_callbacks.append(callback)
    
    def remove_analysis_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """분석 결과 콜백 제거"""
        if callback in self.analysis_callbacks:
            self.analysis_callbacks.remove(callback)
    
    def _audio_recording_thread(self, device_index: Optional[int] = None):
        """오디오 녹음 스레드"""
        if not PYAUDIO_AVAILABLE:
            self.logger.error("[ERROR] PyAudio를 사용할 수 없습니다")
            return
        
        try:
            p = pyaudio.PyAudio()
            
            # 스트림 설정
            stream_params = {
                'format': self.format,
                'channels': self.channels,
                'rate': self.sample_rate,
                'input': True,
                'frames_per_buffer': self.chunk_size
            }
            
            if device_index is not None:
                stream_params['input_device_index'] = device_index
            
            stream = p.open(**stream_params)
            
            self.logger.info("[INFO] 실시간 오디오 녹음 시작")
            self.session_stats["start_time"] = datetime.now()
            
            while self.is_recording:
                try:
                    # 오디오 데이터 읽기
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # 버퍼 크기 관리
                    if self.audio_buffer.qsize() >= self.max_buffer_size:
                        try:
                            self.audio_buffer.get_nowait()  # 오래된 데이터 제거
                        except queue.Empty:
                            pass
                    
                    # 버퍼에 추가
                    timestamp = time.time()
                    self.audio_buffer.put({
                        "data": data,
                        "timestamp": timestamp,
                        "chunk_id": self.session_stats["total_chunks"]
                    })
                    
                    self.session_stats["total_chunks"] += 1
                    
                except Exception as e:
                    self.logger.error(f"[ERROR] 오디오 읽기 오류: {e}")
                    continue
            
            # 정리
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            self.logger.info("[INFO] 실시간 오디오 녹음 종료")
            
        except Exception as e:
            self.logger.error(f"[ERROR] 녹음 스레드 오류: {e}")
            self.is_recording = False
    
    def _audio_analysis_thread(self):
        """오디오 분석 스레드"""
        self.logger.info("[INFO] 실시간 분석 스레드 시작")
        
        accumulated_audio = b""
        analysis_interval = 3.0  # 3초마다 분석
        last_analysis_time = time.time()
        
        while self.is_analyzing:
            try:
                # 오디오 버퍼에서 데이터 가져오기
                try:
                    audio_chunk = self.audio_buffer.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # 음성 감지 (VAD)
                is_speech = self._detect_voice_activity(audio_chunk["data"])
                
                if is_speech:
                    accumulated_audio += audio_chunk["data"]
                    self.session_stats["speech_chunks"] += 1
                else:
                    self.session_stats["silence_chunks"] += 1
                
                # 일정 시간마다 분석 수행
                current_time = time.time()
                if (current_time - last_analysis_time >= analysis_interval and 
                    len(accumulated_audio) > 0):
                    
                    # 임시 파일에 저장
                    temp_file = self._save_audio_to_temp_file(accumulated_audio)
                    
                    if temp_file:
                        # 분석 수행
                        analysis_result = self._perform_analysis(temp_file, audio_chunk["timestamp"])
                        
                        # 콜백 호출
                        for callback in self.analysis_callbacks:
                            try:
                                callback(analysis_result)
                            except Exception as e:
                                self.logger.error(f"[ERROR] 콜백 오류: {e}")
                        
                        # 분석 버퍼에 추가
                        self.analysis_buffer.put(analysis_result)
                        
                        # 임시 파일 정리
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
                    
                    # 리셋
                    accumulated_audio = b""
                    last_analysis_time = current_time
                
            except Exception as e:
                self.logger.error(f"[ERROR] 분석 스레드 오류: {e}")
                continue
        
        self.logger.info("[INFO] 실시간 분석 스레드 종료")
    
    def _detect_voice_activity(self, audio_data: bytes) -> bool:
        """음성 활동 감지"""
        if not WEBRTC_VAD_AVAILABLE or not self.vad:
            return True  # VAD가 없으면 모든 것을 음성으로 처리
        
        try:
            # 웹RTC VAD는 10ms, 20ms, 30ms 프레임만 지원
            frame_duration = 30  # ms
            frame_size = int(self.sample_rate * frame_duration / 1000)
            
            if len(audio_data) < frame_size * 2:  # 16-bit이므로 *2
                return False
            
            # 첫 번째 프레임만 검사
            frame = audio_data[:frame_size * 2]
            return self.vad.is_speech(frame, self.sample_rate)
            
        except Exception as e:
            self.logger.debug(f"[DEBUG] VAD 오류: {e}")
            return True  # 오류 시 음성으로 처리
    
    def _save_audio_to_temp_file(self, audio_data: bytes) -> Optional[str]:
        """오디오 데이터를 임시 WAV 파일로 저장"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # WAV 파일 생성
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)
            
            return temp_path
            
        except Exception as e:
            self.logger.error(f"[ERROR] 임시 파일 저장 실패: {e}")
            return None
    
    def _perform_analysis(self, audio_file: str, timestamp: float) -> Dict[str, Any]:
        """실제 오디오 분석 수행"""
        analysis_start = time.time()
        
        try:
            # 실제 분석 엔진 사용 (import 필요)
            from .real_analysis_engine import global_analysis_engine
            
            # STT 분석 수행
            result = global_analysis_engine.analyze_audio_file(audio_file, language="ko")
            
            processing_time = time.time() - analysis_start
            self.session_stats["total_analysis_time"] += processing_time
            
            # 실시간 분석 메타데이터 추가
            result.update({
                "real_time_metadata": {
                    "timestamp": timestamp,
                    "processing_time": round(processing_time, 2),
                    "chunk_id": self.session_stats["total_chunks"],
                    "analysis_type": "real_time_streaming"
                }
            })
            
            return result
            
        except Exception as e:
            processing_time = time.time() - analysis_start
            error_msg = f"실시간 분석 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg,
                "real_time_metadata": {
                    "timestamp": timestamp,
                    "processing_time": round(processing_time, 2),
                    "analysis_type": "real_time_streaming"
                }
            }
    
    def start_real_time_analysis(self, device_index: Optional[int] = None) -> Dict[str, Any]:
        """실시간 분석 시작"""
        if not self.is_available():
            return {
                "status": "error",
                "error": "실시간 분석을 위한 필요 패키지가 설치되지 않았습니다",
                "missing_packages": self.get_installation_guide()["missing_packages"]
            }
        
        if self.is_recording or self.is_analyzing:
            return {
                "status": "error",
                "error": "이미 실시간 분석이 진행 중입니다"
            }
        
        try:
            # 상태 초기화
            self.is_recording = True
            self.is_analyzing = True
            
            # 스레드 시작
            self.recording_thread = threading.Thread(
                target=self._audio_recording_thread,
                args=(device_index,)
            )
            self.analysis_thread = threading.Thread(
                target=self._audio_analysis_thread
            )
            
            self.recording_thread.start()
            self.analysis_thread.start()
            
            self.logger.info("[SUCCESS] 실시간 분석 시작됨")
            
            return {
                "status": "success",
                "message": "실시간 분석이 시작되었습니다",
                "session_id": int(time.time()),
                "device_index": device_index
            }
            
        except Exception as e:
            self.is_recording = False
            self.is_analyzing = False
            error_msg = f"실시간 분석 시작 실패: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg
            }
    
    def stop_real_time_analysis(self) -> Dict[str, Any]:
        """실시간 분석 중지"""
        try:
            self.logger.info("[INFO] 실시간 분석 중지 요청")
            
            # 플래그 설정
            self.is_recording = False
            self.is_analyzing = False
            
            # 스레드 종료 대기
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=5.0)
            
            if self.analysis_thread and self.analysis_thread.is_alive():
                self.analysis_thread.join(timeout=5.0)
            
            # 통계 계산
            session_duration = 0
            if self.session_stats["start_time"]:
                session_duration = (datetime.now() - self.session_stats["start_time"]).total_seconds()
            
            final_stats = {
                "session_duration": round(session_duration, 2),
                "total_chunks": self.session_stats["total_chunks"],
                "speech_chunks": self.session_stats["speech_chunks"],
                "silence_chunks": self.session_stats["silence_chunks"],
                "speech_ratio": round(
                    self.session_stats["speech_chunks"] / max(self.session_stats["total_chunks"], 1) * 100, 1
                ),
                "total_analysis_time": round(self.session_stats["total_analysis_time"], 2)
            }
            
            self.logger.info("[SUCCESS] 실시간 분석 중지됨")
            
            return {
                "status": "success",
                "message": "실시간 분석이 중지되었습니다",
                "session_stats": final_stats
            }
            
        except Exception as e:
            error_msg = f"실시간 분석 중지 실패: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg
            }
    
    def get_analysis_results(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """최근 분석 결과 가져오기"""
        results = []
        
        # 버퍼에서 결과 수집
        for _ in range(min(max_results, self.analysis_buffer.qsize())):
            try:
                result = self.analysis_buffer.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        
        return results
    
    def get_installation_guide(self) -> Dict[str, Any]:
        """설치 가이드"""
        missing_packages = []
        
        if not PYAUDIO_AVAILABLE:
            missing_packages.append({
                "package": "pyaudio",
                "command": "pip install pyaudio",
                "purpose": "마이크 입력"
            })
        
        if not WEBRTC_VAD_AVAILABLE:
            missing_packages.append({
                "package": "webrtcvad",
                "command": "pip install webrtcvad",
                "purpose": "음성 활동 감지"
            })
        
        if not NUMPY_AVAILABLE:
            missing_packages.append({
                "package": "numpy",
                "command": "pip install numpy",
                "purpose": "오디오 처리"
            })
        
        return {
            "available": self.is_available(),
            "missing_packages": missing_packages,
            "install_all": "pip install pyaudio webrtcvad numpy",
            "additional_notes": [
                "Windows에서 PyAudio 설치 시 오류가 발생할 수 있습니다",
                "해결책: pip install pipwin && pipwin install pyaudio"
            ]
        }

# 전역 인스턴스
real_time_analyzer = RealTimeAnalyzer()

def start_real_time_analysis(device_index: Optional[int] = None) -> Dict[str, Any]:
    """실시간 분석 시작 (전역 접근용)"""
    return real_time_analyzer.start_real_time_analysis(device_index)

def stop_real_time_analysis() -> Dict[str, Any]:
    """실시간 분석 중지 (전역 접근용)"""
    return real_time_analyzer.stop_real_time_analysis()