#!/usr/bin/env python3
"""
실시간 음성 스트리밍 엔진
실시간 음성 입력, 스트리밍 STT, 즉석 분석
"""

import asyncio
import time
import json
import queue
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import numpy as np

try:
    import pyaudio
    import wave
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

from utils.logger import get_logger

class RealtimeAudioStreamingEngine:
    """실시간 음성 스트리밍 엔진"""
    
    def __init__(self):
        self.logger = get_logger(f'{__name__}.RealtimeAudioStreamingEngine')
        
        # 오디오 설정
        self.sample_rate = 16000  # Whisper 최적화 샘플레이트
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16 if AUDIO_AVAILABLE else None
        
        # 스트리밍 설정
        self.stream = None
        self.pyaudio_instance = None
        self.is_streaming = False
        self.audio_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        
        # 실시간 분석 설정
        self.chunk_duration = 3.0  # 3초 단위로 분석
        self.overlap_duration = 1.0  # 1초 겹침
        self.silence_threshold = 500  # 무음 감지 임계값
        self.min_speech_duration = 1.0  # 최소 음성 길이
        
        # 콜백 함수들
        self.on_speech_detected = None
        self.on_text_recognized = None
        self.on_analysis_complete = None
        self.on_error = None
        
        # 상태 관리
        self.current_session = None
        self.streaming_stats = {
            "start_time": None,
            "total_chunks": 0,
            "recognized_text_count": 0,
            "analysis_count": 0,
            "errors": []
        }
        
        self.logger.info("실시간 음성 스트리밍 엔진 초기화 완료")
    
    def set_callbacks(self, 
                     speech_detected: Callable = None,
                     text_recognized: Callable = None, 
                     analysis_complete: Callable = None,
                     error_handler: Callable = None):
        """콜백 함수 설정"""
        
        self.on_speech_detected = speech_detected
        self.on_text_recognized = text_recognized
        self.on_analysis_complete = analysis_complete
        self.on_error = error_handler
        
        self.logger.info("콜백 함수 설정 완료")
    
    async def start_streaming(self, session_info: Dict[str, Any] = None) -> bool:
        """실시간 스트리밍 시작"""
        
        if not AUDIO_AVAILABLE:
            self.logger.error("PyAudio 모듈이 설치되지 않음")
            if self.on_error:
                await self.on_error("오디오 라이브러리가 설치되지 않았습니다")
            return False
        
        try:
            self.logger.info("실시간 음성 스트리밍 시작")
            
            # 세션 정보 저장
            self.current_session = session_info or {
                "session_id": f"stream_{int(time.time())}",
                "start_time": datetime.now().isoformat(),
                "participants": "실시간 입력"
            }
            
            # PyAudio 초기화
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # 마이크 스트림 열기
            self.stream = self.pyaudio_instance.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_streaming = True
            self.streaming_stats["start_time"] = time.time()
            
            # 오디오 처리 스레드 시작
            self.processing_thread = threading.Thread(
                target=self._process_audio_chunks,
                daemon=True
            )
            self.processing_thread.start()
            
            # 분석 처리 스레드 시작
            self.analysis_thread = threading.Thread(
                target=self._process_analysis_queue,
                daemon=True
            )
            self.analysis_thread.start()
            
            self.stream.start_stream()
            
            self.logger.info("스트리밍 시작 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"스트리밍 시작 실패: {str(e)}")
            if self.on_error:
                await self.on_error(f"스트리밍 시작 실패: {str(e)}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """오디오 콜백 함수"""
        
        if self.is_streaming:
            # 오디오 데이터를 큐에 추가
            audio_chunk = {
                "data": in_data,
                "timestamp": time.time(),
                "frame_count": frame_count
            }
            
            try:
                self.audio_queue.put_nowait(audio_chunk)
                self.streaming_stats["total_chunks"] += 1
            except queue.Full:
                self.logger.warning("오디오 큐 가득참 - 오래된 데이터 삭제")
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_chunk)
                except queue.Empty:
                    pass
        
        return (in_data, pyaudio.paContinue)
    
    def _process_audio_chunks(self):
        """오디오 청크 처리 (별도 스레드)"""
        
        audio_buffer = []
        buffer_duration = 0
        last_process_time = time.time()
        
        while self.is_streaming:
            try:
                # 오디오 청크 가져오기
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer.append(chunk)
                
                # 버퍼 지속시간 계산
                chunk_duration = chunk["frame_count"] / self.sample_rate
                buffer_duration += chunk_duration
                
                # 처리 조건 확인
                current_time = time.time()
                should_process = (
                    buffer_duration >= self.chunk_duration or
                    (current_time - last_process_time) >= self.chunk_duration
                )
                
                if should_process and audio_buffer:
                    # 오디오 데이터 합성
                    combined_audio = self._combine_audio_chunks(audio_buffer)
                    
                    # 음성 감지
                    if self._detect_speech(combined_audio):
                        # STT 처리 큐에 추가
                        processing_item = {
                            "audio_data": combined_audio,
                            "timestamp": current_time,
                            "duration": buffer_duration,
                            "session_info": self.current_session
                        }
                        
                        try:
                            self.processing_queue.put_nowait(processing_item)
                            
                            # 음성 감지 콜백
                            if self.on_speech_detected:
                                asyncio.create_task(self.on_speech_detected({
                                    "timestamp": current_time,
                                    "duration": buffer_duration
                                }))
                                
                        except queue.Full:
                            self.logger.warning("처리 큐 가득참")
                    
                    # 버퍼 초기화 (겹침 유지)
                    overlap_chunks = int(len(audio_buffer) * (self.overlap_duration / buffer_duration))
                    audio_buffer = audio_buffer[-overlap_chunks:] if overlap_chunks > 0 else []
                    buffer_duration = overlap_chunks * chunk_duration
                    last_process_time = current_time
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"오디오 처리 오류: {str(e)}")
                self.streaming_stats["errors"].append(str(e))
    
    def _combine_audio_chunks(self, chunks: List[Dict]) -> bytes:
        """오디오 청크들을 하나로 합성"""
        
        combined_data = b''
        for chunk in chunks:
            combined_data += chunk["data"]
        
        return combined_data
    
    def _detect_speech(self, audio_data: bytes) -> bool:
        """음성 감지 (간단한 볼륨 기반)"""
        
        try:
            # bytes를 numpy array로 변환
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # RMS 계산
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            
            # 임계값과 비교
            return rms > self.silence_threshold
            
        except Exception as e:
            self.logger.error(f"음성 감지 오류: {str(e)}")
            return False
    
    def _process_analysis_queue(self):
        """분석 큐 처리 (별도 스레드)"""
        
        while self.is_streaming:
            try:
                # 처리할 항목 가져오기
                item = self.processing_queue.get(timeout=0.5)
                
                # 비동기 분석 실행
                asyncio.create_task(self._analyze_audio_chunk(item))
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"분석 큐 처리 오류: {str(e)}")
    
    async def _analyze_audio_chunk(self, item: Dict[str, Any]):
        """개별 오디오 청크 분석"""
        
        try:
            self.logger.info(f"오디오 청크 분석 시작 (길이: {item['duration']:.1f}초)")
            
            # 1. STT 처리 (시뮬레이션)
            recognized_text = await self._process_stt(item["audio_data"])
            
            if recognized_text:
                self.streaming_stats["recognized_text_count"] += 1
                
                # 텍스트 인식 콜백
                if self.on_text_recognized:
                    await self.on_text_recognized({
                        "text": recognized_text,
                        "timestamp": item["timestamp"],
                        "confidence": 0.85,
                        "duration": item["duration"]
                    })
                
                # 2. 실시간 분석 (간단한 키워드 기반)
                analysis_result = await self._quick_analysis(recognized_text, item)
                
                if analysis_result:
                    self.streaming_stats["analysis_count"] += 1
                    
                    # 분석 완료 콜백
                    if self.on_analysis_complete:
                        await self.on_analysis_complete(analysis_result)
            
        except Exception as e:
            self.logger.error(f"오디오 청크 분석 실패: {str(e)}")
            if self.on_error:
                await self.on_error(f"분석 실패: {str(e)}")
    
    async def _process_stt(self, audio_data: bytes) -> Optional[str]:
        """STT 처리 (시뮬레이션)"""
        
        # 실제 환경에서는 Whisper STT 사용
        await asyncio.sleep(0.5)  # STT 처리 시간 시뮬레이션
        
        # 시뮬레이션 텍스트 (실제로는 Whisper 결과)
        sample_texts = [
            "고객님 안녕하세요 결혼반지 보러 왔습니다",
            "예산은 200만원 정도로 생각하고 있어요",
            "너무 화려하지 않고 심플한 디자인으로 부탁드려요",
            "다이아몬드 등급은 어떻게 되나요",
            "할부 결제도 가능한가요"
        ]
        
        import random
        if random.random() > 0.3:  # 70% 확률로 텍스트 인식
            return random.choice(sample_texts)
        
        return None
    
    async def _quick_analysis(self, text: str, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """빠른 실시간 분석"""
        
        analysis = {
            "timestamp": item["timestamp"],
            "duration": item["duration"],
            "text": text,
            "keywords": [],
            "intent": "unknown",
            "urgency": "normal",
            "sentiment": "neutral",
            "actions": []
        }
        
        # 키워드 감지
        jewelry_keywords = ["결혼반지", "목걸이", "귀걸이", "다이아몬드", "골드", "예물"]
        budget_keywords = ["예산", "가격", "만원", "할부", "결제"]
        urgent_keywords = ["급하게", "빨리", "오늘", "지금"]
        
        for keyword in jewelry_keywords:
            if keyword in text:
                analysis["keywords"].append(keyword)
                analysis["intent"] = "jewelry_inquiry"
        
        for keyword in budget_keywords:
            if keyword in text:
                analysis["keywords"].append(keyword)
                if analysis["intent"] == "jewelry_inquiry":
                    analysis["intent"] = "price_inquiry"
        
        for keyword in urgent_keywords:
            if keyword in text:
                analysis["keywords"].append(keyword)
                analysis["urgency"] = "high"
        
        # 감정 분석 (간단한 키워드 기반)
        positive_keywords = ["좋아요", "마음에", "예뻐요", "괜찮네요"]
        negative_keywords = ["비싸네요", "별로", "아니에요", "부담"]
        
        if any(keyword in text for keyword in positive_keywords):
            analysis["sentiment"] = "positive"
        elif any(keyword in text for keyword in negative_keywords):
            analysis["sentiment"] = "negative"
        
        # 액션 추천
        if analysis["intent"] == "jewelry_inquiry":
            analysis["actions"].append("제품 카탈로그 준비")
            analysis["actions"].append("관련 제품 검색")
        
        if analysis["intent"] == "price_inquiry":
            analysis["actions"].append("가격표 확인")
            analysis["actions"].append("할부 옵션 안내")
        
        if analysis["urgency"] == "high":
            analysis["actions"].append("우선 처리 필요")
        
        return analysis if analysis["keywords"] else None
    
    async def stop_streaming(self) -> Dict[str, Any]:
        """스트리밍 중지"""
        
        self.logger.info("실시간 스트리밍 중지 시작")
        
        try:
            self.is_streaming = False
            
            # 스트림 정지
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            # PyAudio 종료
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
            
            # 최종 통계
            end_time = time.time()
            total_duration = end_time - self.streaming_stats["start_time"]
            
            final_stats = {
                "session_info": self.current_session,
                "total_duration": total_duration,
                "total_chunks": self.streaming_stats["total_chunks"],
                "recognized_text_count": self.streaming_stats["recognized_text_count"],
                "analysis_count": self.streaming_stats["analysis_count"],
                "errors": self.streaming_stats["errors"],
                "avg_chunks_per_second": self.streaming_stats["total_chunks"] / total_duration,
                "text_recognition_rate": self.streaming_stats["recognized_text_count"] / max(1, self.streaming_stats["total_chunks"]),
                "end_time": datetime.now().isoformat()
            }
            
            self.logger.info(f"스트리밍 중지 완료 - 총 {total_duration:.1f}초, {self.streaming_stats['total_chunks']}개 청크 처리")
            return final_stats
            
        except Exception as e:
            self.logger.error(f"스트리밍 중지 실패: {str(e)}")
            return {"error": str(e)}
    
    async def get_streaming_status(self) -> Dict[str, Any]:
        """현재 스트리밍 상태 조회"""
        
        if not self.is_streaming:
            return {"status": "stopped"}
        
        current_time = time.time()
        duration = current_time - self.streaming_stats["start_time"]
        
        return {
            "status": "streaming",
            "session_info": self.current_session,
            "duration": duration,
            "total_chunks": self.streaming_stats["total_chunks"],
            "recognized_text_count": self.streaming_stats["recognized_text_count"],
            "analysis_count": self.streaming_stats["analysis_count"],
            "chunks_per_second": self.streaming_stats["total_chunks"] / duration,
            "audio_queue_size": self.audio_queue.qsize(),
            "processing_queue_size": self.processing_queue.qsize(),
            "errors": len(self.streaming_stats["errors"])
        }

# 사용 예시
async def demo_realtime_streaming():
    """실시간 스트리밍 데모"""
    
    print("=== 실시간 음성 스트리밍 데모 ===")
    
    engine = RealtimeAudioStreamingEngine()
    
    # 콜백 함수 정의
    async def on_speech_detected(info):
        print(f"[음성감지] {info['timestamp']:.1f}초 - 길이: {info['duration']:.1f}초")
    
    async def on_text_recognized(result):
        print(f"[STT] {result['text']} (신뢰도: {result['confidence']:.1%})")
    
    async def on_analysis_complete(analysis):
        print(f"[분석] 의도: {analysis['intent']}, 키워드: {analysis['keywords']}")
        if analysis['actions']:
            print(f"[액션] {', '.join(analysis['actions'])}")
    
    async def on_error(error):
        print(f"[오류] {error}")
    
    # 콜백 설정
    engine.set_callbacks(
        speech_detected=on_speech_detected,
        text_recognized=on_text_recognized,
        analysis_complete=on_analysis_complete,
        error_handler=on_error
    )
    
    # 스트리밍 시작
    session_info = {
        "session_id": "demo_session",
        "participants": "고객, 상담사",
        "context": "주얼리 상담"
    }
    
    success = await engine.start_streaming(session_info)
    
    if success:
        print("스트리밍 시작됨. 10초 후 자동 종료...")
        
        # 10초 대기 (실제로는 사용자가 수동으로 중지)
        await asyncio.sleep(10)
        
        # 스트리밍 중지
        final_stats = await engine.stop_streaming()
        
        print("\n=== 최종 통계 ===")
        print(f"총 처리 시간: {final_stats['total_duration']:.1f}초")
        print(f"처리된 청크: {final_stats['total_chunks']}개")
        print(f"인식된 텍스트: {final_stats['recognized_text_count']}개")
        print(f"완료된 분석: {final_stats['analysis_count']}개")
    
    else:
        print("스트리밍 시작 실패")

if __name__ == "__main__":
    asyncio.run(demo_realtime_streaming())