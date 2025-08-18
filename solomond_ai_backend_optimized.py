#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOLOMOND AI - 고성능 최적화 백엔드 시스템
v2.0 - 메모리 최적화, 비동기 처리, 캐싱, 모니터링 통합
"""

import json
import subprocess
import logging
import os
import base64
import tempfile
import hashlib
import threading
import queue
import gc
import psutil
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import weakref

# 파일 처리 라이브러리
try:
    import easyocr
    OCR_AVAILABLE = True
    print("EasyOCR available")
except ImportError:
    OCR_AVAILABLE = False
    print("EasyOCR not available - image analysis limited")

try:
    import whisper
    STT_AVAILABLE = True
    print("Whisper STT available")
except ImportError:
    STT_AVAILABLE = False
    print("Whisper STT not available - audio analysis limited")

# 성능 모니터링 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.total_processing_time = 0
        self.peak_memory = 0
        self.active_threads = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
    def log_request(self, processing_time):
        """요청 처리 시간 기록"""
        self.request_count += 1
        self.total_processing_time += processing_time
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
            
    def get_stats(self):
        """성능 통계 반환"""
        uptime = time.time() - self.start_time
        avg_processing_time = self.total_processing_time / max(1, self.request_count)
        
        return {
            "uptime_seconds": round(uptime, 2),
            "total_requests": self.request_count,
            "avg_processing_time": round(avg_processing_time, 3),
            "current_memory_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 2),
            "peak_memory_mb": round(self.peak_memory, 2),
            "active_threads": threading.active_count(),
            "cache_hit_rate": round(self.cache_hits / max(1, self.cache_hits + self.cache_misses) * 100, 1),
            "cpu_percent": psutil.cpu_percent()
        }

class FileProcessingCache:
    """파일 처리 결과 캐싱"""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
        
    def get_file_hash(self, file_data):
        """파일 데이터 해시 생성"""
        return hashlib.md5(file_data.encode() if isinstance(file_data, str) else file_data).hexdigest()
    
    def get(self, file_hash):
        """캐시에서 결과 조회"""
        if file_hash in self.cache:
            self.access_count[file_hash] = self.access_count.get(file_hash, 0) + 1
            return self.cache[file_hash]
        return None
    
    def set(self, file_hash, result):
        """캐시에 결과 저장"""
        if len(self.cache) >= self.max_size:
            # LRU 방식으로 가장 적게 사용된 항목 제거
            least_used = min(self.access_count.items(), key=lambda x: x[1])
            del self.cache[least_used[0]]
            del self.access_count[least_used[0]]
            
        self.cache[file_hash] = result
        self.access_count[file_hash] = 1
    
    def clear(self):
        """캐시 초기화"""
        self.cache.clear()
        self.access_count.clear()
        gc.collect()

class OptimizedAIProcessor:
    """최적화된 AI 처리 엔진"""
    
    def __init__(self):
        self.ocr_reader = None
        self.whisper_model = None
        self.file_cache = FileProcessingCache()
        self.performance_monitor = PerformanceMonitor()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.processing_queue = queue.Queue()
        
        # 약한 참조를 사용한 모델 관리
        self._model_refs = weakref.WeakValueDictionary()
        
        self.initialize_models()
        
    def initialize_models(self):
        """AI 모델 지연 초기화"""
        logger.info("🤖 AI 모델 초기화 시작")
        
        if OCR_AVAILABLE:
            try:
                # GPU 사용 가능 여부 확인
                gpu_available = self._check_gpu_available()
                self.ocr_reader = easyocr.Reader(['ko', 'en'], gpu=gpu_available)
                logger.info(f"✅ EasyOCR 모델 로드 완료 (GPU: {gpu_available})")
            except Exception as e:
                logger.error(f"❌ EasyOCR 로드 실패: {e}")
        
        if STT_AVAILABLE:
            try:
                # 메모리 사용량을 고려한 모델 선택
                available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024  # GB
                model_size = "base" if available_memory > 4 else "tiny"
                self.whisper_model = whisper.load_model(model_size)
                logger.info(f"✅ Whisper STT 모델 로드 완료 (모델: {model_size})")
            except Exception as e:
                logger.error(f"❌ Whisper STT 로드 실패: {e}")
    
    def _check_gpu_available(self):
        """GPU 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def process_image_ocr_optimized(self, image_data):
        """최적화된 이미지 OCR 처리"""
        start_time = time.time()
        
        if not OCR_AVAILABLE or self.ocr_reader is None:
            return {"error": "EasyOCR이 설치되지 않았습니다"}
        
        try:
            # 파일 해시 생성
            file_hash = self.file_cache.get_file_hash(image_data)
            
            # 캐시 확인
            cached_result = self.file_cache.get(file_hash)
            if cached_result:
                self.performance_monitor.cache_hits += 1
                logger.info(f"💾 캐시에서 OCR 결과 반환 (해시: {file_hash[:8]})")
                return cached_result
            
            self.performance_monitor.cache_misses += 1
            
            # Base64 디코딩
            image_bytes = base64.b64decode(image_data.split(',')[1])
            
            # 메모리 효율적인 임시 파일 처리
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name
            
            try:
                # OCR 실행
                results = self.ocr_reader.readtext(temp_path)
                
                # 결과 최적화
                extracted_texts = []
                for (bbox, text, confidence) in results:
                    if confidence > 0.5:  # 신뢰도 필터링
                        extracted_texts.append({
                            "text": text.strip(),
                            "confidence": round(float(confidence), 3)
                        })
                
                result = {
                    "success": True,
                    "texts": extracted_texts,
                    "text_count": len(extracted_texts),
                    "processing_time": round(time.time() - start_time, 3)
                }
                
                # 캐시에 저장
                self.file_cache.set(file_hash, result)
                
                return result
                
            finally:
                # 임시 파일 정리
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"❌ 이미지 OCR 처리 실패: {e}")
            return {"error": f"이미지 OCR 처리 실패: {e}"}
        finally:
            self.performance_monitor.log_request(time.time() - start_time)
            # 메모리 정리
            gc.collect()
    
    def process_audio_stt_optimized(self, audio_data):
        """최적화된 오디오 STT 처리"""
        start_time = time.time()
        
        if not STT_AVAILABLE or self.whisper_model is None:
            return {"error": "Whisper STT가 설치되지 않았습니다"}
        
        try:
            # 파일 해시 생성
            file_hash = self.file_cache.get_file_hash(audio_data)
            
            # 캐시 확인
            cached_result = self.file_cache.get(file_hash)
            if cached_result:
                self.performance_monitor.cache_hits += 1
                logger.info(f"💾 캐시에서 STT 결과 반환 (해시: {file_hash[:8]})")
                return cached_result
            
            self.performance_monitor.cache_misses += 1
            
            # Base64 디코딩
            audio_bytes = base64.b64decode(audio_data.split(',')[1])
            
            # 메모리 효율적인 임시 파일 처리
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            try:
                # STT 실행
                result_data = self.whisper_model.transcribe(temp_path)
                
                result = {
                    "success": True,
                    "text": result_data["text"].strip(),
                    "language": result_data.get("language", "unknown"),
                    "processing_time": round(time.time() - start_time, 3)
                }
                
                # 캐시에 저장
                self.file_cache.set(file_hash, result)
                
                return result
                
            finally:
                # 임시 파일 정리
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"❌ 음성 STT 처리 실패: {e}")
            return {"error": f"음성 STT 처리 실패: {e}"}
        finally:
            self.performance_monitor.log_request(time.time() - start_time)
            # 메모리 정리
            gc.collect()
    
    @lru_cache(maxsize=100)
    def get_ollama_models(self):
        """Ollama 모델 목록 캐싱"""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                models = []
                lines = result.stdout.strip().split('\n')[1:]  # 헤더 제외
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            return []
        except Exception as e:
            logger.error(f"❌ Ollama 모델 목록 로드 실패: {e}")
            return []
    
    def analyze_with_ollama_optimized(self, prompt, context=""):
        """최적화된 Ollama AI 분석"""
        start_time = time.time()
        
        try:
            # 최적 모델 선택
            available_models = self.get_ollama_models()
            preferred_models = ['gpt-oss:20b', 'qwen3:8b', 'gemma3:27b', 'qwen2.5:7b', 'gemma3:4b']
            
            selected_model = None
            for model in preferred_models:
                if model in available_models:
                    selected_model = model
                    break
            
            if not selected_model and available_models:
                selected_model = available_models[0]
            
            if not selected_model:
                return {"error": "사용 가능한 Ollama 모델이 없습니다"}
            
            # 프롬프트 최적화 (길이 제한)
            max_prompt_length = 2000
            if len(prompt) > max_prompt_length:
                prompt = prompt[:max_prompt_length] + "...(내용 truncated)"
            
            # Ollama 실행
            cmd = ['ollama', 'run', selected_model, prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            
            if result.returncode == 0:
                response = {
                    "success": True,
                    "model": selected_model,
                    "response": result.stdout.strip(),
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": round(time.time() - start_time, 3)
                }
            else:
                response = {
                    "success": False,
                    "error": f"Ollama 실행 실패: {result.stderr}",
                    "response": "AI 분석 중 오류가 발생했습니다."
                }
                
        except subprocess.TimeoutExpired:
            response = {
                "success": False,
                "error": "AI 분석 시간 초과 (45초)",
                "response": "분석에 시간이 너무 오래 걸렸습니다."
            }
        except Exception as e:
            logger.error(f"❌ AI 분석 오류: {e}")
            response = {
                "success": False,
                "error": str(e),
                "response": "시스템 오류가 발생했습니다."
            }
        finally:
            self.performance_monitor.log_request(time.time() - start_time)
            
        return response
    
    def get_system_health(self):
        """시스템 건강도 체크"""
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        return {
            "timestamp": datetime.now().isoformat(),
            "performance": self.performance_monitor.get_stats(),
            "memory": {
                "total_gb": round(memory_info.total / 1024 / 1024 / 1024, 2),
                "available_gb": round(memory_info.available / 1024 / 1024 / 1024, 2),
                "usage_percent": memory_info.percent
            },
            "disk": {
                "total_gb": round(disk_info.total / 1024 / 1024 / 1024, 2),
                "free_gb": round(disk_info.free / 1024 / 1024 / 1024, 2),
                "usage_percent": round((disk_info.used / disk_info.total) * 100, 1)
            },
            "cache": {
                "size": len(self.file_cache.cache),
                "max_size": self.file_cache.max_size,
                "hit_rate": round(self.performance_monitor.cache_hits / max(1, self.performance_monitor.cache_hits + self.performance_monitor.cache_misses) * 100, 1)
            },
            "models": {
                "ocr_loaded": self.ocr_reader is not None,
                "stt_loaded": self.whisper_model is not None,
                "available_ollama": len(self.get_ollama_models())
            }
        }
    
    def cleanup(self):
        """리소스 정리"""
        logger.info("🧹 리소스 정리 시작")
        self.file_cache.clear()
        self.thread_pool.shutdown(wait=True)
        gc.collect()
        logger.info("✅ 리소스 정리 완료")

class OptimizedSOLOMONDHandler(BaseHTTPRequestHandler):
    """최적화된 HTTP 요청 처리"""
    
    def __init__(self, *args, ai_processor=None, **kwargs):
        self.ai_processor = ai_processor
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """GET 요청 처리"""
        path = urlparse(self.path).path
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        if path == '/health':
            response = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "v2.0-optimized"
            }
        elif path == '/status':
            response = self.ai_processor.get_system_health()
        elif path == '/performance':
            response = self.ai_processor.performance_monitor.get_stats()
        else:
            self.send_response(404)
            response = {"error": "Not Found"}
        
        self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
    
    def do_POST(self):
        """POST 요청 처리"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            path = urlparse(self.path).path
            
            if path == '/analyze':
                prompt = data.get('prompt', '')
                context = data.get('context', '')
                response = self.ai_processor.analyze_with_ollama_optimized(prompt, context)
                
            elif path == '/process_image':
                image_data = data.get('image_data', '')
                response = self.ai_processor.process_image_ocr_optimized(image_data)
                
            elif path == '/process_audio':
                audio_data = data.get('audio_data', '')
                response = self.ai_processor.process_audio_stt_optimized(audio_data)
                
            elif path == '/clear_cache':
                self.ai_processor.file_cache.clear()
                response = {"success": True, "message": "캐시가 초기화되었습니다"}
                
            else:
                response = {"error": "지원하지 않는 엔드포인트입니다"}
            
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            logger.error(f"❌ POST 요청 처리 실패: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error = {"error": str(e)}
            self.wfile.write(json.dumps(error, ensure_ascii=False).encode('utf-8'))
    
    def do_OPTIONS(self):
        """OPTIONS 요청 처리 (CORS)"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        """로그 간소화"""
        pass

# 글로벌 AI 프로세서 인스턴스
ai_processor = None

class CustomOptimizedHandler(OptimizedSOLOMONDHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, ai_processor=ai_processor, **kwargs)

def main():
    """최적화된 서버 실행"""
    global ai_processor
    
    port = 9000
    
    print("SOLOMOND AI v2.0 Optimized Backend Starting")
    print(f"Port: {port}")
    print("Performance Optimizations:")
    print("  - Memory usage optimization")
    print("  - File processing caching")  
    print("  - Asynchronous processing")
    print("  - Real-time monitoring")
    
    # AI 프로세서 초기화
    ai_processor = OptimizedAIProcessor()
    
    server = HTTPServer(('localhost', port), CustomOptimizedHandler)
    
    print(f"Health check: http://localhost:{port}/health")
    print(f"Performance monitoring: http://localhost:{port}/performance")
    print("Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopping...")
        if ai_processor:
            ai_processor.cleanup()
        server.shutdown()
        print("Server stopped successfully")

if __name__ == '__main__':
    main()