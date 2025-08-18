#!/usr/bin/env python3
"""
간단한 Flask 기반 AI 분석 웹앱
Streamlit 문제를 우회하여 AI 기능 구현
"""

from flask import Flask, request, render_template_string, jsonify
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from shared.ollama_interface import OllamaInterface
    ollama = OllamaInterface()
    AI_AVAILABLE = True
except Exception as e:
    AI_AVAILABLE = False
    print(f"Ollama not available: {e}")

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🤖 AI 컨퍼런스 분석기</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        textarea { width: 100%; height: 200px; }
        button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        .result { background: #f9f9f9; padding: 20px; margin-top: 20px; border-radius: 5px; }
        .status { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI 컨퍼런스 분석기</h1>
        
        {% if ai_available %}
            <p class="status">✅ AI 연결됨 ({{ model_count }}개 모델 사용 가능)</p>
        {% else %}
            <p class="status">❌ AI 연결 실패</p>
        {% endif %}
        
        <form method="post">
            <h3>📝 회의 내용 입력:</h3>
            <textarea name="content" placeholder="화자1: 안녕하세요. 오늘 회의를 시작하겠습니다.
화자2: 네, 프로젝트 진행 상황을 점검해보겠습니다.">{{ content or '' }}</textarea>
            
            <br><br>
            <button type="submit">🔍 AI 분석 실행</button>
        </form>
        
        {% if analysis %}
        <div class="result">
            <h3>🎯 AI 분석 결과:</h3>
            
            <h4>🔍 전체 상황 분석:</h4>
            <p>{{ analysis.situation_analysis }}</p>
            
            <h4>🎭 화자별 의미 분석:</h4>
            {% for speaker, meaning in analysis.speaker_meanings.items() %}
                <p><strong>{{ speaker }}:</strong> {{ meaning }}</p>
            {% endfor %}
            
            <h4>📋 회의 맥락 분석:</h4>
            <p>{{ analysis.context_analysis }}</p>
            
            <h4>🎯 AI 종합 결론:</h4>
            <p><strong>{{ analysis.conclusion }}</strong></p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    content = ""
    analysis = None
    
    if request.method == 'POST':
        content = request.form.get('content', '')
        
        if content and AI_AVAILABLE:
            # 화자 구분
            speaker_contents = {}
            lines = content.split('\n')
            
            for line in lines:
                if '화자' in line and ':' in line:
                    if '화자1' in line:
                        speaker_id = '화자_1'
                    elif '화자2' in line:
                        speaker_id = '화자_2'
                    else:
                        continue
                    
                    if speaker_id not in speaker_contents:
                        speaker_contents[speaker_id] = []
                    
                    speaker_text = line.split(':', 1)[-1].strip()
                    if speaker_text:
                        speaker_contents[speaker_id].append(speaker_text)
            
            # AI 분석 실행
            try:
                model = "qwen2.5:7b"
                
                # 전체 상황 분석
                situation_prompt = f"""다음 회의 내용을 분석해주세요:

{content}

분석 결과 (간단히):
1. 회의 주제:
2. 주요 이슈:
3. 분위기:"""
                
                situation_analysis = ollama.generate_response(situation_prompt, model=model)
                
                # 화자별 의미 분석
                speaker_meanings = {}
                for speaker_id, contents in speaker_contents.items():
                    if contents:
                        combined_speech = " ".join(contents)
                        speaker_prompt = f"{speaker_id} 발언 의미 분석: {combined_speech}"
                        speaker_meanings[speaker_id] = ollama.generate_response(speaker_prompt, model=model)[:200] + "..."
                
                # 회의 맥락 분석
                context_prompt = f"회의 결론과 다음 단계: {content[:500]}"
                context_analysis = ollama.generate_response(context_prompt, model=model)
                
                # 종합 결론
                conclusion_prompt = f"한 문장으로 이 회의를 요약하면: {content[:300]}"
                conclusion = ollama.generate_response(conclusion_prompt, model=model)
                
                analysis = {
                    'situation_analysis': situation_analysis[:500] + "...",
                    'speaker_meanings': speaker_meanings,
                    'context_analysis': context_analysis[:300] + "...",
                    'conclusion': conclusion[:200] + "..."
                }
                
            except Exception as e:
                analysis = {'error': str(e)}
    
    return render_template_string(HTML_TEMPLATE, 
                                content=content, 
                                analysis=analysis,
                                ai_available=AI_AVAILABLE,
                                model_count=len(ollama.available_models) if AI_AVAILABLE else 0)

if __name__ == '__main__':
    print("AI 분석 웹앱 시작...")
    print("접속 주소: http://localhost:8555")
    app.run(host='127.0.0.1', port=8555, debug=False)