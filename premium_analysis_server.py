#!/usr/bin/env python3
"""
Premium Analysis Server - 최고 성과 우선 시스템
안정성을 유지하면서 분석 품질을 최대한 끌어올리는 시스템
"""

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import requests
import webbrowser
import threading
import time
import os
import re
import yt_dlp
from urllib.parse import urlparse
import gc
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, 
     origins="*",
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"])

# 성능과 품질의 균형을 맞춘 스레드풀
executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="PREMIUM")

class PremiumOllamaInterface:
    """최고 품질 우선 Ollama 인터페이스"""
    
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.available_models = []
        self.smart_cache = {}  # 스마트 캐싱 (중요한 것만)
        self.max_cache_size = 20  # 캐시 크기 줄이고 품질 우선
        
    def check_connection(self):
        try:
            # 빠른 연결 체크는 유지 (사용자 경험)
            cache_key = "connection_check"
            if cache_key in self.smart_cache:
                cache_time = self.smart_cache[cache_key].get('timestamp', 0)
                if time.time() - cache_time < 30:
                    return self.smart_cache[cache_key]['result']
            
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if response.status_code == 200:
                models_data = response.json()
                self.available_models = [model['name'] for model in models_data.get('models', [])]
                
                self.smart_cache[cache_key] = {
                    'result': True,
                    'timestamp': time.time()
                }
                return True
        except:
            pass
        return False
    
    def get_best_model_for_analysis(self):
        """분석 품질 우선 모델 선택"""
        # 품질 우선 순서: 큰 모델 → 작은 모델
        quality_priority = [
            'qwen2.5:7b',    # 최고 품질
            'gemma3:27b',    # 초고품질 (있을 경우)
            'qwen3:8b',      # 고품질
            'gemma2:9b',     # 고품질
            'gemma3:4b',     # 중상급 품질
            'gemma2:2b'      # 기본 품질 (최후 수단)
        ]
        
        for model in quality_priority:
            if model in self.available_models:
                logger.info(f"Selected premium model: {model}")
                return model
                
        # fallback
        return self.available_models[0] if self.available_models else 'qwen2.5:7b'
    
    def generate_premium_response(self, prompt, context="", force_model=None):
        """최고 품질 우선 AI 응답 생성"""
        try:
            # 모델 선택: 품질 우선
            model = force_model or self.get_best_model_for_analysis()
            
            # 캐싱은 완전히 동일한 요청에만 적용 (품질 보장)
            cache_key = f"{model}:{hash(prompt + context)}"
            if cache_key in self.smart_cache:
                cache_time = self.smart_cache[cache_key].get('timestamp', 0)
                if time.time() - cache_time < 120:  # 2분만 캐시 (최신성 우선)
                    logger.info("Premium cache hit")
                    return self.smart_cache[cache_key]['result']
            
            # 품질 우선 프롬프트 (길이 제한 없음)
            enhanced_prompt = f"""
{context}

다음 내용을 최고 품질로 종합 분석해주세요:

{prompt}

## 💡 "이 사람들이 무엇을 말하는지" 핵심 파악
구체적이고 명확하게 분석해주세요.

## 📊 종합 메시지 추출
- 명시적 메시지: 직접적으로 언급된 내용
- 암시적 메시지: 숨겨진 의도나 감정
- 맥락적 메시지: 상황과 배경을 고려한 해석

## 🎯 상세 분석 포인트
- 핵심 키워드와 반복되는 주제
- 감정과 어조 변화
- 중요한 결정사항이나 합의점
- 우려사항이나 문제점
- 기회요소나 긍정적 신호

## 💼 비즈니스 인사이트
- 시장 반응 및 고객 니즈
- 경쟁 상황 및 포지셔닝
- 개선 기회 및 위험 요소
- 전략적 방향성 제안

## 🚀 구체적 추천 액션
우선순위별로 실행 가능한 액션 플랜을 제시해주세요.

상세하고 심도있게 한국어로 분석해주세요. 내용을 축약하지 말고 완전한 분석을 제공해주세요.
            """.strip()
            
            # 품질 우선 설정
            payload = {
                "model": model,
                "prompt": enhanced_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,      # 더 정확한 분석을 위해 낮춤
                    "num_predict": 1000,     # 충분한 길이의 응답
                    "top_p": 0.8,           # 품질 우선
                    "repeat_penalty": 1.1,   # 반복 방지
                    "top_k": 40             # 품질 있는 선택지
                }
            }
            
            logger.info(f"Premium analysis starting with {model}")
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/api/generate", 
                json=payload, 
                timeout=120,  # 충분한 시간 제공 (품질 우선)
                headers={'Connection': 'close'}
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result_data = response.json()
                result = {
                    "success": True, 
                    "response": result_data.get("response", ""), 
                    "model": model,
                    "processing_time": processing_time,
                    "quality_mode": "PREMIUM"
                }
                
                # 중요한 분석만 캐싱
                if len(self.smart_cache) >= self.max_cache_size:
                    oldest_key = min(self.smart_cache.keys(), 
                                   key=lambda k: self.smart_cache[k].get('timestamp', 0))
                    del self.smart_cache[oldest_key]
                
                self.smart_cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }
                
                logger.info(f"Premium analysis completed in {processing_time:.2f}s")
                return result
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Premium analysis error: {e}")
            return {"success": False, "error": str(e)}
        finally:
            gc.collect()

ollama = PremiumOllamaInterface()

@app.route('/')
def main_page():
    return '''
    <html>
    <head><title>SOLOMOND AI - Premium Analysis</title></head>
    <body style="font-family: Arial; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; min-height: 100vh;">
        <h1>SOLOMOND AI - Premium Quality System</h1>
        <div style="background: rgba(255,255,255,0.1); padding: 25px; border-radius: 15px; margin: 20px 0; backdrop-filter: blur(10px);">
            <h2>최고 성과 우선 시스템</h2>
            <p><strong>핵심 원칙:</strong></p>
            <ul>
                <li>결과물 품질 절대 타협 없음</li>
                <li>"이 사람들이 무엇을 말하는지" 완벽 파악</li>
                <li>심도 있는 종합 분석</li>
                <li>실행 가능한 비즈니스 인사이트</li>
            </ul>
            <a href="/module1" style="background: linear-gradient(45deg, #ef4444, #dc2626); color: white; padding: 20px 40px; border-radius: 10px; text-decoration: none; font-size: 18px; margin: 15px 0; display: inline-block; box-shadow: 0 10px 25px rgba(239, 68, 68, 0.3);">
                Premium Analysis 시작
            </a>
        </div>
        <div id="status" style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 10px;">
            <p><strong>시스템 상태:</strong> <span id="systemStatus">확인 중...</span></p>
        </div>
        <script>
            async function checkStatus() {
                try {
                    const response = await fetch('/api/health');
                    const data = await response.json();
                    document.getElementById('systemStatus').innerHTML = 
                        data.ollama_connected ? 
                        `Premium Ready (${data.best_model || 'Premium model'} 활성화)` : 
                        'Ollama 연결 필요';
                } catch(error) {
                    document.getElementById('systemStatus').textContent = '확인 실패';
                }
            }
            checkStatus();
        </script>
    </body>
    </html>
    '''

@app.route('/api/health')
def health():
    connection_ok = ollama.check_connection()
    best_model = ollama.get_best_model_for_analysis() if connection_ok else None
    
    return jsonify({
        "ollama_connected": connection_ok,
        "available_models": ollama.available_models,
        "best_model": best_model,
        "quality_mode": "PREMIUM",
        "cache_size": len(ollama.smart_cache)
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """최고 품질 우선 분석 엔드포인트"""
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
            
        # 품질 우선: 데이터 제한 없음
        model = data.get('model')  # 사용자 지정 모델 우선
        context = data.get('context', 'Premium Analysis')
        image_texts = data.get('image_texts', [])  # 제한 없음
        audio_texts = data.get('audio_texts', [])   # 제한 없음
        
        # 모든 데이터를 보존 (품질 우선)
        all_content = []
        
        if image_texts:
            all_content.append("=== 이미지 분석 데이터 ===")
            for i, text in enumerate(image_texts, 1):
                all_content.append(f"이미지 {i}: {text}")
        
        if audio_texts:
            all_content.append("\n=== 음성 분석 데이터 ===")
            for i, text in enumerate(audio_texts, 1):
                all_content.append(f"음성 {i}: {text}")
        
        combined_content = "\n".join(all_content)
        
        # 컨텍스트 정보 추가
        context_info = f"분석 컨텍스트: {context}\n처리 파일 수: {len(image_texts + audio_texts)}개"
        
        # 비동기 프리미엄 분석
        def premium_analysis():
            return ollama.generate_premium_response(
                prompt=combined_content,
                context=context_info,
                force_model=model
            )
        
        try:
            future = executor.submit(premium_analysis)
            result = future.result(timeout=150)  # 충분한 시간 (품질 우선)
            
            if result.get('success'):
                logger.info(f"Premium analysis delivered: {result.get('processing_time', 0):.2f}s")
            
            return jsonify(result)
            
        except Exception as timeout_error:
            logger.error(f"Premium analysis timeout: {timeout_error}")
            return jsonify({
                "success": False, 
                "error": "Premium analysis taking longer than expected",
                "suggestion": "High-quality analysis in progress, please wait or try again",
                "quality_mode": "PREMIUM"
            }), 408
        
    except Exception as e:
        logger.error(f"Premium analysis error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        gc.collect()

@app.route('/api/download_url', methods=['POST'])
def download_url():
    """품질 우선 URL 다운로드"""
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
            
        url = data.get('url', '').strip()
        duration_limit = data.get('duration_limit', 0)  # 기본값: 제한 없음 (품질 우선)
        
        if not url:
            return jsonify({"success": False, "error": "URL is required"}), 400
            
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return jsonify({"success": False, "error": "Invalid URL format"}), 400
            
        download_dir = os.path.join(os.getcwd(), 'downloads')
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            
        # 품질 우선 다운로드 설정
        ydl_opts = {
            'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),
            'format': 'best[height<=1080]/best',  # 고화질 우선
            'extractaudio': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'no_warnings': True,
            'socket_timeout': 60,  # 안정성을 위해 증가
            'retries': 3          # 재시도 증가
        }
        
        def premium_download():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                
                # 사용자가 제한을 설정한 경우만 체크
                if duration_limit > 0 and duration > duration_limit:
                    limit_minutes = duration_limit // 60
                    video_minutes = duration // 60
                    raise Exception(f"Video length ({video_minutes} min) exceeds limit ({limit_minutes} min)")
                
                if duration > 3600:
                    logger.info(f"Processing long video: {duration//60} minutes - Premium quality maintained")
                    
                ydl.download([url])
                
                downloaded_files = []
                for file in os.listdir(download_dir):
                    if any(ext in file.lower() for ext in ['.mp4', '.webm', '.mkv', '.m4a', '.mp3']):
                        if title[:30].replace(' ', '_') in file:
                            file_path = os.path.join(download_dir, file)
                            file_size = os.path.getsize(file_path)
                            downloaded_files.append({
                                'filename': file,
                                'path': file_path,
                                'size': file_size,
                                'title': title
                            })
                            break
                
                return {
                    "success": True,
                    "message": f"Premium download completed: {title}",
                    "files": downloaded_files,
                    "duration": duration,
                    "quality": "PREMIUM"
                }
        
        future = executor.submit(premium_download)
        try:
            result = future.result(timeout=600)  # 10분 타임아웃 (품질 우선)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Premium download error: {e}")
            raise e
            
    except yt_dlp.DownloadError as e:
        return jsonify({"success": False, "error": f"Download failed: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Error: {str(e)}"}), 500
    finally:
        gc.collect()

@app.route('/api/extract_urls', methods=['POST'])
def extract_urls():
    """품질 보장 URL 추출"""
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
            
        text = data.get('text', '').strip()  # 길이 제한 없음
        if not text:
            return jsonify({"success": False, "error": "Text is required"}), 400
            
        url_pattern = r'https?://[^\s<>"\{\}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        
        valid_urls = []
        seen_urls = set()
        
        # 모든 URL 처리 (제한 없음)
        for url in urls:
            if url not in seen_urls:
                parsed = urlparse(url)
                if parsed.scheme and parsed.netloc:
                    valid_urls.append({
                        'url': url,
                        'domain': parsed.netloc
                    })
                    seen_urls.add(url)
                    
        return jsonify({
            "success": True,
            "urls": valid_urls,
            "count": len(valid_urls),
            "quality": "PREMIUM"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/module1')
def module1():
    """프리미엄 모듈1 페이지"""
    try:
        with open('MODULE1_ULTIMATE_AI_ANALYSIS.html', 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return """
        <html>
        <head><title>Premium Module 1</title></head>
        <body style="font-family: Arial; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center; min-height: 100vh;">
            <h1>Premium Analysis Module</h1>
            <div style="background: rgba(255,255,255,0.1); padding: 30px; border-radius: 15px; margin: 20px 0; backdrop-filter: blur(10px);">
                <h2>최고 품질 분석 시스템 준비 완료</h2>
                <p><strong>특징:</strong></p>
                <ul style="text-align: left; max-width: 500px; margin: 20px auto;">
                    <li>데이터 손실 없는 완전한 분석</li>
                    <li>최고 성능 AI 모델 자동 선택</li>
                    <li>심도 있는 종합 분석 보고서</li>
                    <li>실행 가능한 비즈니스 인사이트</li>
                </ul>
            </div>
            <a href="/" style="background: linear-gradient(45deg, #10b981, #059669); color: white; padding: 15px 30px; border-radius: 10px; text-decoration: none;">메인으로 돌아가기</a>
        </body>
        </html>
        """

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://localhost:8001')  # 다른 포트 사용

if __name__ == '__main__':
    threading.Thread(target=open_browser, daemon=True).start()
    print("SOLOMOND AI Premium Server Starting...")
    print("Premium Features:")
    print("  • No data loss - complete analysis")
    print("  • Best AI model auto-selection")
    print("  • Comprehensive insights")
    print("  • Quality over speed priority")
    print("URL: http://localhost:8001")
    
    app.run(
        host='0.0.0.0', 
        port=8001, 
        debug=False,
        threaded=True
    )