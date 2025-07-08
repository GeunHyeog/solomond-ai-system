#!/usr/bin/env python3
"""
솔로몬드 주얼리 특화 STT 시스템 - 웹 UI
Jewelry Industry Specialized Speech-to-Text System

개발자: 전근혁 (솔로몬드 대표, 주얼리 전문가)
목적: 주얼리 업계 회의, 강의, 세미나 음성을 정확하게 분석
"""

import os
import sys
import tempfile
import traceback
import time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 시스템 정보 출력
print(f"💎 솔로몬드 주얼리 특화 STT 시스템")
print(f"🐍 Python 버전: {sys.version}")
print(f"📁 현재 디렉토리: {os.getcwd()}")

# Whisper 및 주얼리 모듈 확인
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ Whisper 라이브러리 로드 성공")
except ImportError:
    WHISPER_AVAILABLE = False
    print("❌ Whisper 라이브러리 없음")

try:
    from core.analyzer import get_analyzer, check_whisper_status, get_jewelry_features_info
    from core.jewelry_enhancer import get_jewelry_enhancer
    JEWELRY_ENHANCEMENT_AVAILABLE = True
    print("💎 주얼리 특화 모듈 로드 성공")
except ImportError as e:
    JEWELRY_ENHANCEMENT_AVAILABLE = False
    print(f"⚠️ 주얼리 특화 모듈 로드 실패: {e}")

# FastAPI 앱 생성
app = FastAPI(
    title="솔로몬드 주얼리 특화 STT 시스템",
    description="주얼리 업계 전문가를 위한 AI 음성 분석 플랫폼",
    version="1.0"
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 파일 크기 제한 설정 (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

# 주얼리 특화 HTML 템플릿
HTML_TEMPLATE = f"""
<!DOCTYPE html>
<html>
<head>
    <title>💎 솔로몬드 주얼리 AI 시스템</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 0; 
            padding: 20px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #333;
            min-height: 100vh;
        }}
        
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            margin-top: 10px;
            opacity: 0.9;
        }}
        
        .container {{ 
            max-width: 900px; 
            margin: 0 auto; 
            background: white; 
            padding: 40px; 
            border-radius: 20px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }}
        
        .system-status {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .status-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #2980b9;
            text-align: center;
        }}
        
        .status-card.active {{
            border-left-color: #27ae60;
            background: #e8f5e8;
        }}
        
        .status-card.warning {{
            border-left-color: #f39c12;
            background: #fff3cd;
        }}
        
        .status-card h3 {{
            margin: 0 0 10px 0;
            font-size: 1.1em;
        }}
        
        .upload-section {{
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
        }}
        
        .upload-area {{ 
            border: 3px dashed rgba(255,255,255,0.5); 
            padding: 40px; 
            text-align: center; 
            border-radius: 15px;
            background: rgba(255,255,255,0.1);
            transition: all 0.3s ease;
        }}
        
        .upload-area:hover {{ 
            border-color: rgba(255,255,255,0.8); 
            background: rgba(255,255,255,0.2); 
        }}
        
        input[type="file"] {{ 
            padding: 15px; 
            font-size: 16px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 8px;
            width: 100%;
            margin: 10px 0;
            background: rgba(255,255,255,0.9);
            color: #333;
        }}
        
        .options {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .option-group {{
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
        }}
        
        .option-group label {{
            display: block;
            margin-bottom: 10px;
            font-weight: bold;
        }}
        
        select, input[type="checkbox"] {{
            width: 100%;
            padding: 10px;
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 5px;
            background: rgba(255,255,255,0.9);
            color: #333;
        }}
        
        button {{ 
            background: linear-gradient(45deg, #27ae60, #2ecc71); 
            color: white; 
            padding: 18px 40px; 
            border: none; 
            border-radius: 10px; 
            cursor: pointer; 
            font-size: 18px;
            font-weight: bold;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 20px;
        }}
        
        button:hover {{ 
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(39, 174, 96, 0.4);
        }}
        
        button:disabled {{
            background: #95a5a6;
            cursor: not-allowed;
            transform: none;
        }}
        
        .progress {{
            width: 100%;
            background-color: rgba(255,255,255,0.3);
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
            display: none;
            height: 40px;
        }}
        
        .progress-bar {{
            width: 0%;
            height: 100%;
            background: linear-gradient(45deg, #27ae60, #2ecc71);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        
        .result {{ 
            margin-top: 30px; 
            padding: 0;
            background: transparent;
            display: none;
        }}
        
        .result-section {{
            background: white;
            margin: 20px 0;
            padding: 25px;
            border-radius: 12px;
            border-left: 5px solid #3498db;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .jewelry-terms {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        
        .term-category {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 3px solid #e74c3c;
        }}
        
        .term-category.gemstone {{ border-left-color: #e74c3c; }}
        .term-category.grading {{ border-left-color: #9b59b6; }}
        .term-category.business {{ border-left-color: #f39c12; }}
        .term-category.technical {{ border-left-color: #27ae60; }}
        
        .term-category h4 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
        }}
        
        .term-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }}
        
        .term-badge {{
            background: #ecf0f1;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.9em;
            color: #2c3e50;
        }}
        
        .status {{ 
            padding: 15px; 
            margin: 15px 0; 
            border-radius: 8px; 
            font-weight: bold;
        }}
        
        .success {{ background: #d4edda; color: #155724; border-left: 5px solid #28a745; }}
        .error {{ background: #f8d7da; color: #721c24; border-left: 5px solid #dc3545; }}
        .info {{ background: #d1ecf1; color: #0c5460; border-left: 5px solid #17a2b8; }}
        .warning {{ background: #fff3cd; color: #856404; border-left: 5px solid #ffc107; }}
        
        .enhancement-stats {{
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .stat-item {{
            text-align: center;
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
        }}
        
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            display: block;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-top: 5px;
        }}
        
        @media (max-width: 768px) {{
            .container {{ padding: 20px; margin: 10px; }}
            .header h1 {{ font-size: 2em; }}
            .options {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>💎 솔로몬드 주얼리 AI 시스템</h1>
        <div class="subtitle">주얼리 업계 전문가를 위한 음성 분석 플랫폼</div>
        <div style="margin-top: 15px; font-size: 0.9em;">
            개발: 전근혁 대표 (솔로몬드, 한국보석협회 사무국장)
        </div>
    </div>
    
    <div class="container">
        <div class="system-status">
            <div class="status-card {'active' if WHISPER_AVAILABLE else 'warning'}">
                <h3>🎤 음성 인식</h3>
                <div>{"✅ Whisper 준비완료" if WHISPER_AVAILABLE else "❌ 설치 필요"}</div>
            </div>
            <div class="status-card {'active' if JEWELRY_ENHANCEMENT_AVAILABLE else 'warning'}">
                <h3>💎 주얼리 특화</h3>
                <div>{"✅ 활성화" if JEWELRY_ENHANCEMENT_AVAILABLE else "❌ 비활성화"}</div>
            </div>
            <div class="status-card active">
                <h3>🌍 다국어 지원</h3>
                <div>✅ 한/영/중</div>
            </div>
            <div class="status-card active">
                <h3>📁 파일 지원</h3>
                <div>✅ MP3/WAV/M4A</div>
            </div>
        </div>
        
        {"" if WHISPER_AVAILABLE else '''
        <div class="status warning">
            <strong>⚠️ Whisper 설치 필요</strong><br>
            음성 인식을 위해 다음 명령어를 실행하세요:<br>
            <code style="background: #000; color: #0f0; padding: 5px; border-radius: 3px;">pip install openai-whisper</code>
        </div>
        '''}
        
        <div class="upload-section">
            <h2 style="margin-top: 0; text-align: center;">🎯 주얼리 세미나/회의 음성 분석</h2>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-area">
                    <div style="font-size: 48px; margin-bottom: 20px;">🎤💎</div>
                    <h3>주얼리 관련 음성 파일을 선택하세요</h3>
                    <p>세미나, 강의, 고객 상담, 무역 협상 등</p>
                    <input type="file" name="audio_file" accept=".mp3,.wav,.m4a,.aac,.flac" required>
                </div>
                
                <div class="options">
                    <div class="option-group">
                        <label>🌍 언어 설정</label>
                        <select id="languageSelect">
                            <option value="auto">🌐 자동 감지</option>
                            <option value="ko" selected>🇰🇷 한국어</option>
                            <option value="en">🇺🇸 English</option>
                            <option value="zh">🇨🇳 中文</option>
                            <option value="ja">🇯🇵 日本語</option>
                        </select>
                    </div>
                    
                    <div class="option-group">
                        <label>💎 주얼리 특화 기능</label>
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <input type="checkbox" id="jewelryEnhancement" 
                                   {"checked" if JEWELRY_ENHANCEMENT_AVAILABLE else "disabled"}
                                   style="width: auto;">
                            <span>용어 자동 수정 & 분석</span>
                        </div>
                    </div>
                </div>
                
                <button type="submit" id="submitBtn">
                    🚀 주얼리 특화 음성 분석 시작
                </button>
                
                <div class="progress" id="progressContainer">
                    <div class="progress-bar" id="progressBar">0%</div>
                </div>
            </form>
        </div>
        
        <div id="result" class="result">
            <div id="resultContent"></div>
        </div>
    </div>

    <script>
        // 진행률 시뮬레이션
        function startProgress() {{
            const progressContainer = document.getElementById('progressContainer');
            const progressBar = document.getElementById('progressBar');
            
            progressContainer.style.display = 'block';
            let progress = 0;
            
            const interval = setInterval(() => {{
                progress += Math.random() * 10;
                if (progress > 85) progress = 85;
                
                progressBar.style.width = progress + '%';
                progressBar.textContent = Math.round(progress) + '%';
            }}, 800);
            
            return interval;
        }}
        
        // 주얼리 용어 카테고리별 표시
        function displayJewelryTerms(terms) {{
            if (!terms || terms.length === 0) return '';
            
            const categories = {{}};
            terms.forEach(term => {{
                const cat = term.category || '기타';
                if (!categories[cat]) categories[cat] = [];
                categories[cat].push(term.term);
            }});
            
            let html = '<div class="jewelry-terms">';
            
            Object.entries(categories).forEach(([category, termList]) => {{
                const categoryClass = {{
                    '보석': 'gemstone',
                    '등급': 'grading', 
                    '비즈니스': 'business',
                    '기술': 'technical'
                }}[category] || 'gemstone';
                
                html += `
                    <div class="term-category ${{categoryClass}}">
                        <h4>${{category}}</h4>
                        <div class="term-list">
                            ${{[...new Set(termList)].map(term => 
                                `<span class="term-badge">${{term}}</span>`
                            ).join('')}}
                        </div>
                    </div>
                `;
            }});
            
            html += '</div>';
            return html;
        }}
        
        // 주얼리 분석 결과 표시
        function displayJewelryAnalysis(analysis) {{
            if (!analysis) return '';
            
            let html = '';
            
            if (analysis.identified_topics && analysis.identified_topics.length > 0) {{
                html += `
                    <div class="result-section">
                        <h3>🎯 식별된 주제</h3>
                        <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                            ${{analysis.identified_topics.map(topic => 
                                `<span style="background: #3498db; color: white; padding: 8px 15px; border-radius: 20px; font-size: 0.9em;">${{topic}}</span>`
                            ).join('')}}
                        </div>
                    </div>
                `;
            }}
            
            if (analysis.business_insights && analysis.business_insights.length > 0) {{
                html += `
                    <div class="result-section">
                        <h3>💡 비즈니스 인사이트</h3>
                        <ul>
                            ${{analysis.business_insights.map(insight => 
                                `<li>${{insight}}</li>`
                            ).join('')}}
                        </ul>
                    </div>
                `;
            }}
            
            if (analysis.technical_level || analysis.language_complexity) {{
                html += `
                    <div class="result-section">
                        <h3>📊 콘텐츠 분석</h3>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                            ${{analysis.technical_level ? `
                                <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                                    <strong>기술적 복잡도</strong><br>
                                    <span style="font-size: 1.5em; color: #2c3e50;">${{analysis.technical_level}}</span>
                                </div>
                            ` : ''}}
                            ${{analysis.language_complexity ? `
                                <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                                    <strong>언어 복잡도</strong><br>
                                    <span style="font-size: 1.5em; color: #2c3e50;">${{analysis.language_complexity}}</span>
                                </div>
                            ` : ''}}
                        </div>
                    </div>
                `;
            }}
            
            return html;
        }}
        
        // 수정사항 표시
        function displayCorrections(corrections) {{
            if (!corrections || corrections.length === 0) return '';
            
            return `
                <div class="result-section">
                    <h3>🔧 주얼리 용어 수정사항 (${{corrections.length}}개)</h3>
                    <div style="max-height: 200px; overflow-y: auto;">
                        ${{corrections.map(correction => `
                            <div style="padding: 10px; margin: 5px 0; background: #f8f9fa; border-radius: 5px; border-left: 3px solid #e74c3c;">
                                <strong>'${{correction.original}}'</strong> → <strong>'${{correction.corrected}}'</strong>
                                <span style="color: #666; font-size: 0.9em;">(${{correction.type}})</span>
                            </div>
                        `).join('')}}
                    </div>
                </div>
            `;
        }}
        
        // 파일 업로드 처리
        document.getElementById('uploadForm').onsubmit = async function(e) {{
            e.preventDefault();
            
            const fileInput = document.querySelector('input[type="file"]');
            const submitBtn = document.getElementById('submitBtn');
            const resultDiv = document.getElementById('result');
            const resultContent = document.getElementById('resultContent');
            const languageSelect = document.getElementById('languageSelect');
            const jewelryEnhancement = document.getElementById('jewelryEnhancement');
            
            // 파일 선택 확인
            if (!fileInput.files[0]) {{
                alert('파일을 선택해주세요');
                return;
            }}
            
            const file = fileInput.files[0];
            const fileSize = (file.size / (1024 * 1024)).toFixed(2);
            
            // 파일 크기 검사
            if (file.size > 100 * 1024 * 1024) {{
                alert('파일 크기가 100MB를 초과합니다.');
                return;
            }}
            
            // UI 상태 변경
            submitBtn.disabled = true;
            submitBtn.textContent = '🔄 주얼리 특화 분석 중...';
            resultDiv.style.display = 'block';
            
            // 진행률 시작
            const progressInterval = startProgress();
            
            resultContent.innerHTML = `
                <div class="result-section">
                    <div class="status info">
                        <strong>📁 파일 정보</strong><br>
                        파일명: ${{file.name}}<br>
                        크기: ${{fileSize}} MB<br>
                        언어: ${{languageSelect.options[languageSelect.selectedIndex].text}}<br>
                        주얼리 특화: ${{jewelryEnhancement.checked ? '✅ 활성화' : '❌ 비활성화'}}
                    </div>
                    <div class="status info">
                        🔄 서버로 업로드 중... 주얼리 용어 분석 준비 중입니다.
                    </div>
                </div>
            `;
            
            try {{
                // FormData 생성
                const formData = new FormData();
                formData.append('audio_file', file);
                formData.append('language', languageSelect.value);
                formData.append('enable_jewelry', jewelryEnhancement.checked);
                
                console.log('📤 주얼리 특화 분석 시작:', file.name);
                
                // 서버로 전송
                const response = await fetch('/jewelry_analyze', {{
                    method: 'POST',
                    body: formData
                }});
                
                console.log('📡 서버 응답 수신:', response.status);
                
                if (!response.ok) {{
                    throw new Error(`HTTP ${{response.status}}: ${{response.statusText}}`);
                }}
                
                const result = await response.json();
                console.log('📋 분석 결과:', result);
                
                // 진행률 완료
                clearInterval(progressInterval);
                document.getElementById('progressBar').style.width = '100%';
                document.getElementById('progressBar').textContent = '완료!';
                
                if (result.success) {{
                    let resultHtml = `
                        <div class="result-section">
                            <div class="status success">
                                <strong>✅ 주얼리 특화 음성 분석 완료!</strong><br>
                                처리 시간: ${{result.total_processing_time || result.processing_time}}초
                            </div>
                        </div>
                    `;
                    
                    // 기본 텍스트 결과
                    const displayText = result.enhanced_text || result.transcribed_text;
                    resultHtml += `
                        <div class="result-section">
                            <h3>📝 분석된 텍스트</h3>
                            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 5px solid #27ae60; font-size: 16px; line-height: 1.6;">
                                ${{displayText || '(텍스트가 감지되지 않았습니다)'}}
                            </div>
                        </div>
                    `;
                    
                    // 주얼리 특화 결과
                    if (result.jewelry_enhancement && result.detected_jewelry_terms) {{
                        resultHtml += `
                            <div class="result-section">
                                <h3>💎 발견된 주얼리 용어 (${{result.detected_jewelry_terms.length}}개)</h3>
                                ${{displayJewelryTerms(result.detected_jewelry_terms)}}
                            </div>
                        `;
                    }}
                    
                    // 수정사항
                    if (result.jewelry_corrections && result.jewelry_corrections.length > 0) {{
                        resultHtml += displayCorrections(result.jewelry_corrections);
                    }}
                    
                    // 주얼리 분석
                    if (result.jewelry_analysis) {{
                        resultHtml += displayJewelryAnalysis(result.jewelry_analysis);
                    }}
                    
                    // 요약
                    if (result.jewelry_summary) {{
                        resultHtml += `
                            <div class="result-section">
                                <h3>📄 주얼리 업계 맞춤 요약</h3>
                                <div style="background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 10px;">
                                    ${{result.jewelry_summary.replace(/\\n/g, '<br>')}}
                                </div>
                            </div>
                        `;
                    }}
                    
                    // 시스템 정보
                    resultHtml += `
                        <div class="result-section">
                            <div class="status info">
                                <strong>📊 분석 상세 정보</strong><br>
                                감지 언어: ${{result.language_info?.name || result.detected_language || '자동 감지'}}<br>
                                파일 크기: ${{result.file_size || fileSize + ' MB'}}<br>
                                주얼리 특화: ${{result.jewelry_enhancement ? '✅ 적용됨' : '❌ 미적용'}}<br>
                                처리 방식: Whisper + 주얼리 도메인 AI
                            </div>
                        </div>
                    `;
                    
                    resultContent.innerHTML = resultHtml;
                    
                }} else {{
                    resultContent.innerHTML = `
                        <div class="result-section">
                            <div class="status error">
                                <strong>❌ 분석 실패</strong><br>
                                오류: ${{result.error}}<br>
                                처리 시간: ${{result.processing_time || 0}}초
                            </div>
                            <div class="status warning">
                                <strong>💡 해결 방법:</strong><br>
                                • 파일 형식 확인 (MP3, WAV, M4A)<br>
                                • 파일 크기 확인 (100MB 이하)<br>
                                • Whisper 설치 확인<br>
                                • 주얼리 특화 모듈 상태 확인
                            </div>
                        </div>
                    `;
                }}
                
            }} catch (error) {{
                console.error('❌ 업로드 오류:', error);
                
                clearInterval(progressInterval);
                
                resultContent.innerHTML = `
                    <div class="result-section">
                        <div class="status error">
                            <strong>❌ 네트워크 오류</strong><br>
                            ${{error.message}}
                        </div>
                        <div class="status warning">
                            <strong>🔧 진단 단계:</strong><br>
                            1. 서버가 실행 중인지 확인<br>
                            2. 방화벽 설정 확인<br>
                            3. 주얼리 특화 모듈 상태 확인<br>
                            4. 브라우저 콘솔(F12) 확인
                        </div>
                    </div>
                `;
            }} finally {{
                // UI 복원
                submitBtn.disabled = false;
                submitBtn.textContent = '🚀 주얼리 특화 음성 분석 시작';
                
                setTimeout(() => {{
                    document.getElementById('progressContainer').style.display = 'none';
                }}, 3000);
            }}
        }};
        
        // 페이지 로드 시 주얼리 특화 기능 상태 확인
        window.onload = function() {{
            if (!{str(JEWELRY_ENHANCEMENT_AVAILABLE).lower()}) {{
                document.getElementById('jewelryEnhancement').disabled = true;
                const label = document.querySelector('label[for="jewelryEnhancement"]');
                if (label) {{
                    label.style.opacity = '0.5';
                    label.title = '주얼리 특화 모듈이 로드되지 않았습니다';
                }}
            }}
        }};
        
        // 파일 선택 시 정보 표시
        document.querySelector('input[type="file"]').onchange = function(e) {{
            const file = e.target.files[0];
            if (file) {{
                const fileSize = (file.size / (1024 * 1024)).toFixed(2);
                console.log(`📁 주얼리 분석용 파일 선택: ${{file.name}} (${{fileSize}} MB)`);
            }}
        }};
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지"""
    return HTML_TEMPLATE

@app.post("/jewelry_analyze") 
async def jewelry_analyze(
    audio_file: UploadFile = File(...),
    language: str = "ko",
    enable_jewelry: bool = True
):
    """주얼리 특화 음성 분석"""
    import time
    start_time = time.time()
    
    try:
        # 1. 파일 기본 정보 확인
        filename = audio_file.filename or "unknown_file"
        print(f"💎 주얼리 특화 분석 시작: {filename}")
        
        # 2. 파일 내용 읽기
        try:
            content = await audio_file.read()
            file_size_mb = len(content) / (1024 * 1024)
            file_size_str = f"{file_size_mb:.2f} MB"
            
            print(f"📊 파일 크기: {file_size_str}")
            
            if len(content) > MAX_FILE_SIZE:
                return JSONResponse({
                    "success": False,
                    "error": f"파일 크기가 {MAX_FILE_SIZE/(1024*1024):.0f}MB를 초과합니다. (현재: {file_size_str})",
                    "processing_time": round(time.time() - start_time, 2)
                })
                
        except Exception as read_error:
            print(f"❌ 파일 읽기 오류: {read_error}")
            return JSONResponse({
                "success": False,
                "error": f"파일 읽기 실패: {str(read_error)}",
                "processing_time": round(time.time() - start_time, 2)
            })
        
        # 3. 주얼리 특화 STT 분석 실행
        if JEWELRY_ENHANCEMENT_AVAILABLE:
            try:
                from core.analyzer import get_analyzer
                
                analyzer = get_analyzer(enable_jewelry_enhancement=enable_jewelry)
                
                # 임시 파일 생성 및 분석
                file_ext = Path(filename).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                    temp_file.write(content)
                    temp_path = temp_file.name
                
                try:
                    # 주얼리 특화 분석 실행
                    result = await analyzer.analyze_audio_file(
                        temp_path, 
                        language=language,
                        enable_jewelry_features=enable_jewelry
                    )
                    
                    if result["success"]:
                        # 파일 정보 추가
                        result["filename"] = filename
                        result["file_size"] = file_size_str
                        
                        print(f"✅ 주얼리 특화 분석 완료: {result.get('total_processing_time', 0)}초")
                        
                        if result.get("jewelry_corrections"):
                            print(f"🔧 {len(result['jewelry_corrections'])}개 주얼리 용어 수정")
                        
                        if result.get("detected_jewelry_terms"):
                            print(f"💎 {len(result['detected_jewelry_terms'])}개 주얼리 용어 발견")
                    
                    return JSONResponse(result)
                    
                finally:
                    # 임시 파일 정리
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                    
            except Exception as jewelry_error:
                print(f"❌ 주얼리 특화 분석 오류: {jewelry_error}")
                # 기본 STT로 fallback
                enable_jewelry = False
        
        # 4. 기본 STT 분석 (주얼리 특화 실패 시)
        if not WHISPER_AVAILABLE:
            return JSONResponse({
                "success": False,
                "error": "Whisper 라이브러리가 설치되지 않았습니다. 'pip install openai-whisper' 명령어로 설치하세요.",
                "processing_time": round(time.time() - start_time, 2)
            })
        
        # 기본 Whisper 분석
        file_ext = Path(filename).suffix.lower()
        supported_formats = ['.mp3', '.wav', '.m4a', '.aac', '.flac']
        
        if file_ext not in supported_formats:
            return JSONResponse({
                "success": False,
                "error": f"지원하지 않는 파일 형식: {file_ext}. 지원 형식: {', '.join(supported_formats)}",
                "processing_time": round(time.time() - start_time, 2)
            })
        
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            print("🔄 기본 Whisper 모델 로딩...")
            model = whisper.load_model("base")
            
            print("🎤 기본 음성 인식 시작...")
            whisper_options = {"language": language if language != "auto" else None}
            result = model.transcribe(temp_path, **whisper_options)
            
            transcribed_text = result["text"].strip()
            detected_language = result.get("language", language)
            
            processing_time = round(time.time() - start_time, 2)
            
            return JSONResponse({
                "success": True,
                "filename": filename,
                "file_size": file_size_str,
                "transcribed_text": transcribed_text,
                "detected_language": detected_language,
                "processing_time": processing_time,
                "jewelry_enhancement": False,
                "note": "주얼리 특화 기능이 비활성화되어 기본 STT만 사용되었습니다."
            })
            
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        error_msg = str(e)
        
        print(f"❌ 예상치 못한 오류: {error_msg}")
        print(f"🔍 오류 상세:\\n{traceback.format_exc()}")
        
        return JSONResponse({
            "success": False,
            "error": f"서버 내부 오류: {error_msg}",
            "processing_time": processing_time
        })

@app.get("/status")
async def get_status():
    """시스템 상태 확인"""
    try:
        status_info = {
            "whisper_available": WHISPER_AVAILABLE,
            "jewelry_enhancement_available": JEWELRY_ENHANCEMENT_AVAILABLE,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if JEWELRY_ENHANCEMENT_AVAILABLE:
            try:
                jewelry_info = get_jewelry_features_info()
                status_info["jewelry_features"] = jewelry_info
            except:
                status_info["jewelry_features"] = {"error": "기능 정보 로드 실패"}
        
        return JSONResponse(status_info)
        
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

if __name__ == "__main__":
    print("=" * 80)
    print("💎 솔로몬드 주얼리 특화 STT 시스템")
    print("=" * 80)
    print(f"🐍 Python: {sys.version}")
    print(f"🎤 Whisper: {'✅ 사용 가능' if WHISPER_AVAILABLE else '❌ 설치 필요'}")
    print(f"💎 주얼리 특화: {'✅ 활성화' if JEWELRY_ENHANCEMENT_AVAILABLE else '❌ 비활성화'}")
    print(f"📁 최대 파일 크기: {MAX_FILE_SIZE // (1024*1024)}MB")
    print(f"🌐 접속 주소: http://localhost:8080")
    print(f"🔧 상태 확인: http://localhost:8080/status")
    print("=" * 80)
    
    if not WHISPER_AVAILABLE:
        print("⚠️  Whisper 설치 필요:")
        print("   pip install openai-whisper")
        print("=" * 80)
    
    if not JEWELRY_ENHANCEMENT_AVAILABLE:
        print("⚠️  주얼리 특화 모듈 확인 필요:")
        print("   core/jewelry_enhancer.py 및 관련 모듈")
        print("=" * 80)
    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8080,
            reload=False,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\\n👋 솔로몬드 시스템이 안전하게 종료되었습니다.")
    except Exception as e:
        print(f"\\n❌ 서버 시작 오류: {e}")
