#!/usr/bin/env python3
"""
🤖 Ollama Integration Module for SOLOMOND AI
HTML 환경에서 Ollama API 호출을 위한 Python 백엔드 서버
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import json
import os
import logging
from pathlib import Path
import threading
import time

app = Flask(__name__)
CORS(app)  # HTML에서 AJAX 호출 허용

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaInterface:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.available_models = []
        self.check_connection()
    
    def check_connection(self):
        """Ollama 연결 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                self.available_models = [model['name'] for model in models_data.get('models', [])]
                logger.info(f"Ollama 연결 성공! 사용가능 모델: {self.available_models}")
                return True
        except Exception as e:
            logger.error(f"Ollama 연결 실패: {e}")
            return False
    
    def generate_response(self, prompt, model="gemma2:2b", stream=False):
        """Ollama 모델로 응답 생성"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2000
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # 2분으로 증가
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "model": model,
                    "done": result.get("done", True)
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            logger.error(f"Ollama 생성 오류: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_extracted_content(self, image_texts, audio_texts, context="주얼리 컨퍼런스"):
        """추출된 텍스트들을 종합 분석"""
        
        # 프롬프트 구성
        prompt = f"""
You are a professional analyst specializing in {context}. Please analyze the following extracted texts:

=== Texts from Images ===
{chr(10).join([f"Image {i+1}: {text}" for i, text in enumerate(image_texts)])}

=== Texts from Audio ===
{chr(10).join([f"Audio {i+1}: {text}" for i, text in enumerate(audio_texts)])}

Please provide comprehensive analysis in the following format:

## Core Message Summary
(What are these people talking about? Clear summary)

## Key Content Analysis
1. Main Topics:
2. Important Keywords:
3. Speaker Intentions:

## Business/Jewelry Industry Insights
(Analysis from industry perspective)

## Action Items
(Future tasks or important points to note)

Please provide detailed and professional analysis in Korean language.
"""
        
        return self.generate_response(prompt)

# Ollama 인터페이스 인스턴스
ollama = OllamaInterface()

@app.route('/')
def index():
    """기본 상태 페이지"""
    return jsonify({
        "service": "SOLOMOND AI Ollama Integration",
        "status": "running",
        "available_models": ollama.available_models,
        "endpoints": [
            "/health - 상태 확인",
            "/models - 사용가능 모델 목록", 
            "/generate - 텍스트 생성",
            "/analyze - 종합 분석"
        ]
    })

@app.route('/health')
def health():
    """서비스 상태 확인"""
    connection_ok = ollama.check_connection()
    return jsonify({
        "ollama_connected": connection_ok,
        "available_models": ollama.available_models,
        "status": "healthy" if connection_ok else "ollama_disconnected"
    })

@app.route('/models')
def get_models():
    """사용가능한 모델 목록"""
    return jsonify({
        "models": ollama.available_models
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Ollama 텍스트 생성"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        model = data.get('model', 'qwen2.5:7b')
        
        if not prompt:
            return jsonify({"success": False, "error": "prompt가 필요합니다"}), 400
        
        result = ollama.generate_response(prompt, model)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """종합 분석 수행"""
    try:
        data = request.json
        image_texts = data.get('image_texts', [])
        audio_texts = data.get('audio_texts', [])
        context = data.get('context', '주얼리 컨퍼런스')
        
        if not image_texts and not audio_texts:
            return jsonify({
                "success": False, 
                "error": "분석할 텍스트가 필요합니다"
            }), 400
        
        result = ollama.analyze_extracted_content(image_texts, audio_texts, context)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/test_analysis')
def test_analysis():
    """테스트용 분석 실행"""
    # 샘플 데이터로 테스트
    sample_image_texts = [
        "JGA 2025 주얼리 박람회",
        "다이아몬드 트렌드 발표",
        "2025년 시장 전망"
    ]
    
    sample_audio_texts = [
        "안녕하세요. 오늘은 주얼리 시장의 미래에 대해 말씀드리겠습니다.",
        "다이아몬드 가격이 작년 대비 15% 상승했습니다.",
        "새로운 디자인 트렌드가 주목받고 있습니다."
    ]
    
    result = ollama.analyze_extracted_content(sample_image_texts, sample_audio_texts)
    return jsonify(result)

def run_server():
    """서버 실행"""
    port = 8888
    print(f"Ollama Integration Server Starting...")
    print(f"URL: http://localhost:{port}")
    print(f"Available Models: {ollama.available_models}")
    
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    run_server()