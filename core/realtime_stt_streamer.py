"""
🎙️ 실시간 STT 스트리밍 시스템 v1.0
Phase 2 Week 3-4: WebSocket 기반 실시간 음성 인식 및 주얼리 분석

작성자: 전근혁 (솔로몬드 대표)
목적: 현장에서 실시간으로 음성을 분석하여 즉시 주얼리 인사이트 제공
통합: 기존 solomond-ai-system과 완전 호환
"""

import asyncio
import websockets
import json
import base64
import tempfile
import os
import wave
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import threading
import queue
from collections import deque

# 기존 시스템 모듈 import
try:
    from core.analyzer import JewelrySTTAnalyzer
    from core.jewelry_ai_engine import JewelryAIEngine, integrate_with_existing_system
    from core.jewelry_enhancer import enhance_jewelry_transcription
except ImportError:
    print("⚠️ 기존 모듈 import 실패 - 개발 모드로 실행")
    # 개발용 Mock 클래스들
    class MockJewelrySTTAnalyzer:
        def analyze_audio_file(self, filepath, language='ko'): 
            return {"transcription": "모의 텍스트", "language": language}
    
    class MockJewelryAIEngine:
        def analyze_text(self, text, context="realtime"):
            return {"insights": [], "summary": "실시간 분석 결과"}
    
    JewelrySTTAnalyzer = MockJewelrySTTAnalyzer
    JewelryAIEngine = MockJewelryAIEngine

class RealtimeSTTStreamer:
    """실시간 STT 스트리밍 서버"""
    
    def __init__(self, 
                 chunk_duration: float = 2.0,
                 overlap_duration: float = 0.5,
                 sample_rate: int = 16000,
                 channels: int = 1):
        """
        Args:
            chunk_duration: 오디오 청크 길이 (초)
            overlap_duration: 청크 간 겹침 시간 (초)
            sample_rate: 샘플링 레이트
            channels: 채널 수
        """
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = sample_rate
        self.channels = channels
        
        # 기존 시스템 연동
        try:
            self.stt_analyzer = JewelrySTTAnalyzer()
            self.ai_engine = JewelryAIEngine()
        except:
            print("🔧 Mock 모드로 실행")
            self.stt_analyzer = MockJewelrySTTAnalyzer()
            self.ai_engine = MockJewelryAIEngine()
        
        # 실시간 처리를 위한 큐 시스템
        self.audio_queue = queue.Queue()
        self.processing_thread = None
        self.is_running = False
        
        # 컨텍스트 관리 (이전 대화 내용 유지)
        self.conversation_context = deque(maxlen=20)  # 최근 20개 문장 유지
        self.session_data = {}
        
        # 실시간 최적화 설정
        self.min_speech_length = 0.5  # 최소 음성 길이 (초)
        self.silence_threshold = 0.01  # 무음 임계값
        self.max_chunk_size = 10 * 1024 * 1024  # 10MB
    
    async def start_server(self, host: str = "localhost", port: int = 8765):
        """WebSocket 서버 시작"""
        print(f"🎙️ 실시간 STT 스트리밍 서버 시작")
        print(f"📡 주소: ws://{host}:{port}")
        print(f"⚙️ 설정: {self.chunk_duration}초 청크, {self.sample_rate}Hz")
        
        async with websockets.serve(self.handle_client, host, port):
            print("🚀 서버 준비 완료 - 클라이언트 연결 대기중...")
            await asyncio.Future()  # 무한 대기
    
    async def handle_client(self, websocket, path):
        """클라이언트 연결 처리"""
        client_id = f"client_{datetime.now().strftime('%H%M%S')}"
        print(f"👤 새 클라이언트 연결: {client_id}")
        
        # 세션 초기화
        self.session_data[client_id] = {
            'start_time': datetime.now(),
            'total_chunks': 0,
            'processed_text': "",
            'insights': [],
            'language': 'ko'
        }
        
        try:
            await self.send_message(websocket, {
                'type': 'connection_established',
                'client_id': client_id,
                'message': '실시간 주얼리 STT 시스템에 연결되었습니다',
                'capabilities': {
                    'realtime_stt': True,
                    'jewelry_analysis': True,
                    'multi_language': True,
                    'chunk_duration': self.chunk_duration
                }
            })
            
            async for message in websocket:
                await self.process_client_message(websocket, client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            print(f"👋 클라이언트 {client_id} 연결 종료")
        except Exception as e:
            print(f"❌ 클라이언트 {client_id} 오류: {e}")
            await self.send_error(websocket, f"서버 오류: {str(e)}")
        finally:
            # 세션 정리
            if client_id in self.session_data:
                del self.session_data[client_id]
    
    async def process_client_message(self, websocket, client_id: str, message):
        """클라이언트 메시지 처리"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'audio_chunk':
                await self.process_audio_chunk(websocket, client_id, data)
            elif msg_type == 'config':
                await self.update_config(websocket, client_id, data)
            elif msg_type == 'get_session_summary':
                await self.send_session_summary(websocket, client_id)
            elif msg_type == 'reset_session':
                await self.reset_session(websocket, client_id)
            else:
                await self.send_error(websocket, f"알 수 없는 메시지 타입: {msg_type}")
                
        except json.JSONDecodeError:
            await self.send_error(websocket, "JSON 파싱 오류")
        except Exception as e:
            await self.send_error(websocket, f"메시지 처리 오류: {str(e)}")
    
    async def process_audio_chunk(self, websocket, client_id: str, data):
        """오디오 청크 실시간 처리"""
        try:
            # 오디오 데이터 추출
            audio_data = base64.b64decode(data['audio'])
            chunk_id = data.get('chunk_id', 0)
            language = data.get('language', 'ko')
            
            # 세션 정보 업데이트
            session = self.session_data[client_id]
            session['total_chunks'] += 1
            session['language'] = language
            
            # 오디오 데이터 검증
            if len(audio_data) < 1000:  # 너무 작은 청크 무시
                return
            
            # 오디오 파일로 저장 (임시)
            temp_file = await self.save_audio_chunk(audio_data, chunk_id)
            
            # STT 처리 (비동기)
            transcription_result = await self.process_stt_async(temp_file, language)
            
            if transcription_result and transcription_result.get('transcription'):
                # 기존 텍스트와 결합
                text = transcription_result['transcription'].strip()
                
                if len(text) > 3:  # 의미있는 텍스트만 처리
                    # 주얼리 특화 후처리
                    enhanced_text = await self.enhance_text_async(text)
                    
                    # AI 분석 (실시간 최적화)
                    ai_insights = await self.analyze_text_async(enhanced_text, client_id)
                    
                    # 결과 전송
                    await self.send_message(websocket, {
                        'type': 'transcription_result',
                        'chunk_id': chunk_id,
                        'original_text': text,
                        'enhanced_text': enhanced_text,
                        'ai_insights': ai_insights,
                        'timestamp': datetime.now().isoformat(),
                        'session_stats': {
                            'total_chunks': session['total_chunks'],
                            'session_duration': (datetime.now() - session['start_time']).total_seconds()
                        }
                    })
                    
                    # 컨텍스트 업데이트
                    self.conversation_context.append({
                        'text': enhanced_text,
                        'timestamp': datetime.now().isoformat(),
                        'client_id': client_id
                    })
            
            # 임시 파일 정리
            try:
                os.unlink(temp_file)
            except:
                pass
                
        except Exception as e:
            print(f"❌ 오디오 청크 처리 오류: {e}")
            await self.send_error(websocket, f"오디오 처리 실패: {str(e)}")
    
    async def save_audio_chunk(self, audio_data: bytes, chunk_id: int) -> str:
        """오디오 청크를 임시 파일로 저장"""
        temp_file = tempfile.NamedTemporaryFile(
            suffix=f'_chunk_{chunk_id}.wav', 
            delete=False
        )
        
        # WAV 형식으로 저장 (기존 analyzer.py와 호환)
        with wave.open(temp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data)
        
        return temp_file.name
    
    async def process_stt_async(self, audio_file: str, language: str) -> Dict:
        """STT 처리 (비동기)"""
        try:
            # 별도 스레드에서 STT 실행 (블로킹 방지)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.stt_analyzer.analyze_audio_file,
                audio_file,
                language
            )
            return result
        except Exception as e:
            print(f"STT 처리 오류: {e}")
            return {'transcription': '', 'error': str(e)}
    
    async def enhance_text_async(self, text: str) -> str:
        """텍스트 후처리 (비동기)"""
        try:
            # 기존 jewelry_enhancer와 연동
            loop = asyncio.get_event_loop()
            enhanced = await loop.run_in_executor(
                None,
                enhance_jewelry_transcription,
                text
            )
            return enhanced.get('enhanced_text', text)
        except Exception as e:
            print(f"텍스트 후처리 오류: {e}")
            return text
    
    async def analyze_text_async(self, text: str, client_id: str) -> Dict:
        """AI 텍스트 분석 (비동기)"""
        try:
            # 컨텍스트 포함하여 분석
            context_text = self.build_context_text(client_id)
            full_text = f"{context_text} {text}" if context_text else text
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.ai_engine.analyze_text,
                full_text,
                "realtime_streaming"
            )
            
            # 실시간 최적화: 핵심 인사이트만 반환
            return self.optimize_insights_for_realtime(result)
            
        except Exception as e:
            print(f"AI 분석 오류: {e}")
            return {'insights': [], 'error': str(e)}
    
    def build_context_text(self, client_id: str) -> str:
        """대화 컨텍스트 구성"""
        recent_texts = [
            item['text'] for item in self.conversation_context 
            if item['client_id'] == client_id
        ]
        return " ".join(recent_texts[-5:])  # 최근 5개 문장
    
    def optimize_insights_for_realtime(self, ai_result: Dict) -> Dict:
        """실시간 처리를 위한 인사이트 최적화"""
        try:
            insights = ai_result.get('insights', [])
            
            # 우선순위 높은 인사이트만 선별
            priority_insights = [
                insight for insight in insights 
                if insight.get('priority') in ['최고', '높음']
            ]
            
            return {
                'priority_insights': priority_insights[:3],  # 상위 3개만
                'summary': ai_result.get('summary', ''),
                'confidence': ai_result.get('confidence_score', 0),
                'keywords': ai_result.get('keywords', {}),
                'sentiment': ai_result.get('sentiment', {}).get('primary', 'neutral')
            }
        except Exception as e:
            return {'error': f'인사이트 최적화 오류: {str(e)}'}
    
    async def update_config(self, websocket, client_id: str, data):
        """실시간 설정 업데이트"""
        config = data.get('config', {})
        
        # 동적 설정 변경 (주의: 일부 설정만 런타임에 변경 가능)
        if 'language' in config:
            self.session_data[client_id]['language'] = config['language']
        
        await self.send_message(websocket, {
            'type': 'config_updated',
            'message': '설정이 업데이트되었습니다',
            'current_config': {
                'language': self.session_data[client_id]['language'],
                'chunk_duration': self.chunk_duration
            }
        })
    
    async def send_session_summary(self, websocket, client_id: str):
        """세션 요약 전송"""
        session = self.session_data.get(client_id, {})
        
        # 전체 텍스트 결합
        full_text = " ".join([
            item['text'] for item in self.conversation_context 
            if item['client_id'] == client_id
        ])
        
        # 종합 분석 수행
        if full_text:
            try:
                final_analysis = self.ai_engine.analyze_text(full_text, "session_summary")
                business_report = self.ai_engine.generate_business_report(final_analysis)
            except:
                final_analysis = {'summary': '분석 중 오류 발생'}
                business_report = '리포트 생성 실패'
        else:
            final_analysis = {'summary': '분석할 텍스트가 없습니다'}
            business_report = '내용이 부족합니다'
        
        await self.send_message(websocket, {
            'type': 'session_summary',
            'session_stats': {
                'duration': (datetime.now() - session['start_time']).total_seconds(),
                'total_chunks': session['total_chunks'],
                'total_text_length': len(full_text)
            },
            'full_text': full_text,
            'final_analysis': final_analysis,
            'business_report': business_report,
            'timestamp': datetime.now().isoformat()
        })
    
    async def reset_session(self, websocket, client_id: str):
        """세션 초기화"""
        if client_id in self.session_data:
            self.session_data[client_id] = {
                'start_time': datetime.now(),
                'total_chunks': 0,
                'processed_text': "",
                'insights': [],
                'language': 'ko'
            }
        
        # 컨텍스트에서 해당 클라이언트 데이터 제거
        self.conversation_context = deque([
            item for item in self.conversation_context 
            if item['client_id'] != client_id
        ], maxlen=20)
        
        await self.send_message(websocket, {
            'type': 'session_reset',
            'message': '세션이 초기화되었습니다',
            'timestamp': datetime.now().isoformat()
        })
    
    async def send_message(self, websocket, data: Dict):
        """클라이언트에게 메시지 전송"""
        try:
            await websocket.send(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            print(f"메시지 전송 오류: {e}")
    
    async def send_error(self, websocket, error_message: str):
        """에러 메시지 전송"""
        await self.send_message(websocket, {
            'type': 'error',
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        })

class RealtimeSTTClient:
    """실시간 STT 클라이언트 (테스트용)"""
    
    def __init__(self, server_url: str = "ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        self.is_connected = False
    
    async def connect(self):
        """서버에 연결"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.is_connected = True
            print(f"🔗 서버 연결 성공: {self.server_url}")
            
            # 연결 확인 메시지 수신
            response = await self.websocket.recv()
            data = json.loads(response)
            print(f"📨 서버 응답: {data['message']}")
            
        except Exception as e:
            print(f"❌ 연결 실패: {e}")
            self.is_connected = False
    
    async def send_audio_file(self, file_path: str, language: str = 'ko'):
        """오디오 파일을 청크로 나누어 전송 (테스트용)"""
        if not self.is_connected:
            print("❌ 서버에 연결되지 않음")
            return
        
        try:
            # 오디오 파일 읽기
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            
            # 청크로 분할 (2초씩)
            chunk_size = 32000  # 대략 2초 (16kHz * 2초)
            chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
            
            print(f"🎵 {len(chunks)}개 청크로 분할하여 전송 시작")
            
            for i, chunk in enumerate(chunks):
                # Base64 인코딩
                encoded_chunk = base64.b64encode(chunk).decode('utf-8')
                
                # 서버로 전송
                message = {
                    'type': 'audio_chunk',
                    'audio': encoded_chunk,
                    'chunk_id': i,
                    'language': language
                }
                
                await self.websocket.send(json.dumps(message))
                print(f"📤 청크 {i+1}/{len(chunks)} 전송")
                
                # 결과 수신
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
                    result = json.loads(response)
                    
                    if result['type'] == 'transcription_result':
                        print(f"📝 인식 결과: {result['enhanced_text']}")
                        if result['ai_insights']['priority_insights']:
                            print(f"💡 인사이트: {result['ai_insights']['priority_insights'][0]['insight']}")
                    
                except asyncio.TimeoutError:
                    print("⏰ 응답 대기 시간 초과")
                
                await asyncio.sleep(0.5)  # 자연스러운 간격
            
            print("✅ 모든 청크 전송 완료")
            
        except Exception as e:
            print(f"❌ 파일 전송 오류: {e}")
    
    async def get_session_summary(self):
        """세션 요약 요청"""
        if not self.is_connected:
            return
        
        await self.websocket.send(json.dumps({'type': 'get_session_summary'}))
        response = await self.websocket.recv()
        data = json.loads(response)
        
        if data['type'] == 'session_summary':
            print("📊 세션 요약:")
            print(f"⏱️ 총 시간: {data['session_stats']['duration']:.1f}초")
            print(f"📦 총 청크: {data['session_stats']['total_chunks']}개")
            print(f"📝 전체 텍스트: {data['full_text'][:200]}...")
            print(f"🧠 최종 분석: {data['final_analysis']['summary']}")
    
    async def disconnect(self):
        """연결 종료"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            print("👋 연결 종료")

# 메인 실행부
async def main():
    """실시간 STT 스트리밍 시스템 데모"""
    print("🎙️ 주얼리 업계 특화 실시간 STT 스트리밍 시스템 v1.0")
    print("=" * 60)
    
    # 서버 모드 vs 클라이언트 모드 선택
    mode = input("실행 모드를 선택하세요 (1: 서버, 2: 클라이언트 테스트): ")
    
    if mode == "1":
        # 서버 모드
        streamer = RealtimeSTTStreamer(
            chunk_duration=2.0,
            overlap_duration=0.5
        )
        await streamer.start_server()
    
    elif mode == "2":
        # 클라이언트 테스트 모드
        client = RealtimeSTTClient()
        await client.connect()
        
        if client.is_connected:
            # 테스트 파일 경로 입력
            file_path = input("테스트할 오디오 파일 경로 (Enter로 건너뛰기): ")
            if file_path and os.path.exists(file_path):
                await client.send_audio_file(file_path)
                await client.get_session_summary()
            else:
                print("🔧 서버 연결만 테스트 완료")
            
            await client.disconnect()
    
    else:
        print("❌ 잘못된 선택")

if __name__ == "__main__":
    print("🚀 실시간 STT 스트리밍 시스템 시작...")
    print("💎 주얼리 업계 특화 기능 포함")
    print("🔧 기존 solomond-ai-system과 완전 통합")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 시스템 종료")
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")