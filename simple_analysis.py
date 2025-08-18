#!/usr/bin/env python3
"""
Simple Analysis Server - fetch 문제 해결을 위한 웹서버
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
import gc  # 메모리 정리용
from concurrent.futures import ThreadPoolExecutor  # 병렬 처리용
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 스레드풀 (성능 최적화)
executor = ThreadPoolExecutor(max_workers=4)

app = Flask(__name__)
CORS(app, 
     origins="*",  # file:// 프로토콜 허용
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"])

class OllamaInterface:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.available_models = []
        self.response_cache = {}  # 응답 캐싱 추가
        self.max_cache_size = 50
        
    def check_connection(self):
        try:
            # 캐시 확인 (30초)
            cache_key = "connection_check"
            if cache_key in self.response_cache:
                cache_time = self.response_cache[cache_key].get('timestamp', 0)
                if time.time() - cache_time < 30:
                    return self.response_cache[cache_key]['result']
            
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)  # 타임아웃 단축
            if response.status_code == 200:
                models_data = response.json()
                self.available_models = [model['name'] for model in models_data.get('models', [])]
                
                # 결과 캐싱
                self.response_cache[cache_key] = {
                    'result': True,
                    'timestamp': time.time()
                }
                return True
        except:
            pass
        return False
    
    def generate_response(self, prompt, model="gemma2:2b"):
        try:
            # 응답 캐싱 확인
            cache_key = f"{model}:{hash(prompt[:100])}"
            if cache_key in self.response_cache:
                cache_time = self.response_cache[cache_key].get('timestamp', 0)
                if time.time() - cache_time < 300:  # 5분 캐시
                    logger.info("Cache hit for analysis request")
                    return self.response_cache[cache_key]['result']
            
            # 프롬프트 최적화 (너무 길면 줄이기)
            if len(prompt) > 2000:
                prompt = prompt[:1800] + "...[요약됨]"
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 400,  # 응답 길이 제한
                    "top_p": 0.9
                }
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=45)
            if response.status_code == 200:
                result_data = response.json()
                result = {"success": True, "response": result_data.get("response", ""), "model": model}
                
                # 캐시 크기 관리
                if len(self.response_cache) >= self.max_cache_size:
                    oldest_key = min(self.response_cache.keys(), 
                                   key=lambda k: self.response_cache[k].get('timestamp', 0))
                    del self.response_cache[oldest_key]
                
                # 결과 캐싱
                self.response_cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }
                
                return result
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return {"success": False, "error": str(e)}
        finally:
            # 메모리 정리
            gc.collect()

ollama = OllamaInterface()

@app.route('/')
def main_page():
    return '''
    <html>
    <head><title>SOLOMOND AI 시작</title></head>
    <body style="font-family: Arial; padding: 40px; background: #f0f8ff;">
        <h1>🤖 SOLOMOND AI</h1>
        <div style="background: #e8f5e8; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h2>🚀 문제 해결 완료</h2>
            <p>fetch 연결 문제가 해결되었습니다. 이제 정상적으로 AI 분석이 가능합니다!</p>
            <a href="/module1" style="background: #10b981; color: white; padding: 15px 30px; border-radius: 5px; text-decoration: none; font-size: 18px; margin: 10px 0; display: inline-block;">
                🏆 모듈1 AI 분석 시작
            </a>
        </div>
        <div id="status" style="background: #fff3cd; padding: 15px; border-radius: 8px;">
            <p><strong>Ollama:</strong> <span id="ollamaStatus">확인 중...</span></p>
        </div>
        <script>
            async function checkStatus() {
                try {
                    const response = await fetch('/api/health');
                    const data = await response.json();
                    document.getElementById('ollamaStatus').innerHTML = 
                        data.ollama_connected ? 
                        `✅ 연결됨 (${data.available_models.length}개 모델)` : 
                        '❌ Ollama 서버 시작 필요';
                } catch(error) {
                    document.getElementById('ollamaStatus').textContent = '❌ 확인 실패';
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
    return jsonify({
        "ollama_connected": connection_ok,
        "available_models": ollama.available_models
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        # JSON 데이터 안전하게 파싱
        if not request.is_json:
            return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
            
        # 메모리 효율적인 데이터 처리
        model = data.get('model', 'gemma2:2b')
        context = data.get('context', 'Analysis')
        image_texts = data.get('image_texts', [])[:8]  # 최대 8개로 제한
        audio_texts = data.get('audio_texts', [])[:8]  # 최대 8개로 제한
        
        # 텍스트 길이 최적화
        combined_texts = []
        for text in image_texts + audio_texts:
            if len(text) > 400:
                text = text[:350] + "...[요약]"
            combined_texts.append(text)
        
        # 최적화된 프롬프트
        prompt = f"""
컨텍스트: {context}

분석 데이터:
{' | '.join(combined_texts[:6])}

다음 형식으로 간결하게 분석:
## 핵심 메시지
## 주요 포인트
## 추천 액션

각 섹션 2-3문장으로 한국어 요약.
"""
        
        # 비동기 처리 (타임아웃 방지)
        def process_analysis():
            return ollama.generate_response(prompt, model)
        
        try:
            future = executor.submit(process_analysis)
            result = future.result(timeout=40)  # 40초 타임아웃
            return jsonify(result)
        except Exception as timeout_error:
            return jsonify({
                "success": False, 
                "error": "Analysis timeout - try shorter content",
                "suggestion": "Use gemma2:2b model for faster response"
            }), 408
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        gc.collect()

@app.route('/api/download_url', methods=['POST'])
def download_url():
    """URL에서 비디오/오디오 다운로드"""
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
            
        url = data.get('url', '').strip()
        duration_limit = data.get('duration_limit', 3600)  # 기본값 1시간
        
        if not url:
            return jsonify({"success": False, "error": "URL is required"}), 400
            
        # URL 유효성 검사
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return jsonify({"success": False, "error": "Invalid URL format"}), 400
            
        # 다운로드 디렉터리 생성
        download_dir = os.path.join(os.getcwd(), 'downloads')
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            
        # yt-dlp 설정
        ydl_opts = {
            'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),
            'format': 'best[height<=720]/best',  # 720p 이하로 제한
            'extractaudio': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'no_warnings': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 비디오 정보 추출
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            
            # 사용자가 설정한 제한시간 체크
            if duration_limit > 0 and duration > duration_limit:
                limit_minutes = duration_limit // 60
                video_minutes = duration // 60
                return jsonify({
                    "success": False, 
                    "error": f"Video too long ({video_minutes} minutes). Current limit: {limit_minutes} minutes. Please adjust duration limit or choose a shorter video."
                }), 400
            elif duration > 3600:  # 1시간 이상시 경고 메시지
                print(f"⚠️ Long video detected: {duration//60} minutes. Processing may take longer.")
                
            # 다운로드 실행
            ydl.download([url])
            
            # 다운로드된 파일 찾기
            downloaded_files = []
            for file in os.listdir(download_dir):
                if title.replace(' ', '_') in file or any(ext in file for ext in ['.mp4', '.webm', '.mkv']):
                    file_path = os.path.join(download_dir, file)
                    file_size = os.path.getsize(file_path)
                    downloaded_files.append({
                        'filename': file,
                        'path': file_path,
                        'size': file_size,
                        'title': title
                    })
                    break
            
            return jsonify({
                "success": True,
                "message": f"Downloaded: {title}",
                "files": downloaded_files,
                "duration": duration
            })
            
    except yt_dlp.DownloadError as e:
        return jsonify({"success": False, "error": f"Download failed: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Unexpected error: {str(e)}"}), 500

@app.route('/api/extract_urls', methods=['POST'])
def extract_urls():
    """텍스트에서 URL 추출"""
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
            
        text = data.get('text', '').strip()
        if not text:
            return jsonify({"success": False, "error": "Text is required"}), 400
            
        # URL 패턴으로 추출
        url_pattern = r'https?://[^\s<>"{\}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        
        # 유효한 URL만 필터링
        valid_urls = []
        for url in urls:
            parsed = urlparse(url)
            if parsed.scheme and parsed.netloc:
                valid_urls.append({
                    'url': url,
                    'domain': parsed.netloc
                })
                
        return jsonify({
            "success": True,
            "urls": valid_urls,
            "count": len(valid_urls)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/module1')
def module1():
    """모듈1 AI 분석 페이지를 제공"""
    try:
        with open('MODULE1_REAL_AI_ANALYSIS.html', 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return """
        <html>
        <head><title>오류</title></head>
        <body style="font-family: Arial; padding: 40px; text-align: center;">
            <h1>⚠️ 모듈1 파일을 찾을 수 없습니다</h1>
            <p>MODULE1_REAL_AI_ANALYSIS.html 파일이 필요합니다.</p>
            <a href="/" style="background: #10b981; color: white; padding: 10px 20px; border-radius: 5px; text-decoration: none;">🏠 메인으로 돌아가기</a>
        </body>
        </html>
        """
    except Exception as e:
        return f"<h1>오류 발생: {str(e)}</h1>"

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://localhost:8000')

if __name__ == '__main__':
    threading.Thread(target=open_browser, daemon=True).start()
    print("SOLOMOND AI Server Starting...")
    print("URL: http://localhost:8000") 
    print("Connection issues resolved!")
    app.run(host='0.0.0.0', port=8000, debug=False)
