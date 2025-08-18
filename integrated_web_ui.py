#!/usr/bin/env python3
"""
🌐 SOLOMOND AI 통합 웹 UI
사용자 친화적인 파일 업로드 + 실시간 분석 결과 표시
"""

from fastapi import FastAPI, UploadFile, File, Request, WebSocket, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
import asyncio
import json
import uuid
from datetime import datetime
import httpx
import os

app = FastAPI(title="SOLOMOND AI 통합 웹 UI", version="2.0.0")

# 분석 세션 저장소
analysis_sessions = {}

@app.get("/", response_class=HTMLResponse)
async def main_ui():
    """메인 통합 UI 페이지"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🤖 SOLOMOND AI 통합 분석 시스템</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 40px;
                color: white;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            
            .main-content {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-bottom: 40px;
            }
            
            .upload-section, .status-section {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            
            .upload-area {
                border: 3px dashed #ddd;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                transition: all 0.3s ease;
                cursor: pointer;
                margin-bottom: 20px;
            }
            
            .upload-area:hover {
                border-color: #667eea;
                background: #f8f9ff;
            }
            
            .upload-area.dragover {
                border-color: #667eea;
                background: #e8f0fe;
            }
            
            .file-input {
                display: none;
            }
            
            .upload-btn {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 25px;
                font-size: 16px;
                cursor: pointer;
                transition: transform 0.2s;
                margin: 10px;
            }
            
            .upload-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            
            .file-list {
                margin-top: 20px;
            }
            
            .file-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px;
                background: #f5f5f5;
                border-radius: 5px;
                margin-bottom: 5px;
            }
            
            .progress-bar {
                width: 100%;
                height: 10px;
                background: #e0e0e0;
                border-radius: 5px;
                overflow: hidden;
                margin: 10px 0;
            }
            
            .progress-fill {
                height: 100%;
                background: linear-gradient(45deg, #667eea, #764ba2);
                width: 0%;
                transition: width 0.3s ease;
            }
            
            .status-item {
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 8px;
                border-left: 4px solid;
            }
            
            .status-waiting { background: #fff3cd; border-color: #ffc107; }
            .status-processing { background: #d1ecf1; border-color: #17a2b8; }
            .status-completed { background: #d4edda; border-color: #28a745; }
            .status-error { background: #f8d7da; border-color: #dc3545; }
            
            .results-section {
                grid-column: 1 / -1;
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                margin-top: 20px;
            }
            
            .analysis-result {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 15px;
            }
            
            .result-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }
            
            .result-content {
                line-height: 1.6;
            }
            
            .ai-badge {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 12px;
            }
            
            @media (max-width: 768px) {
                .main-content {
                    grid-template-columns: 1fr;
                }
                .header h1 {
                    font-size: 2em;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 SOLOMOND AI</h1>
                <p>통합 컨퍼런스 분석 시스템 - 파일을 업로드하면 AI가 종합 분석해드립니다</p>
            </div>
            
            <div class="main-content">
                <!-- 파일 업로드 섹션 -->
                <div class="upload-section">
                    <h2>📁 파일 업로드</h2>
                    <div class="upload-area" id="uploadArea">
                        <div>
                            <p style="font-size: 24px; margin-bottom: 10px;">📎</p>
                            <p>파일을 드래그하거나 클릭하여 업로드</p>
                            <p style="font-size: 14px; color: #666; margin-top: 10px;">
                                지원 형식: 텍스트, 이미지, 오디오, 비디오
                            </p>
                        </div>
                    </div>
                    <input type="file" id="fileInput" class="file-input" multiple accept="*/*">
                    <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                        파일 선택
                    </button>
                    <button class="upload-btn" onclick="startAnalysis()">
                        🚀 분석 시작
                    </button>
                    
                    <div class="file-list" id="fileList"></div>
                </div>
                
                <!-- 상태 모니터링 섹션 -->
                <div class="status-section">
                    <h2>📊 분석 상태</h2>
                    <div id="statusList">
                        <div class="status-item status-waiting">
                            <strong>대기 중</strong><br>
                            파일을 업로드하고 분석을 시작하세요
                        </div>
                    </div>
                    
                    <div class="progress-bar" style="margin-top: 20px;">
                        <div class="progress-fill" id="overallProgress"></div>
                    </div>
                    <p style="text-align: center; margin-top: 10px;">
                        전체 진행률: <span id="progressText">0%</span>
                    </p>
                </div>
            </div>
            
            <!-- 결과 표시 섹션 -->
            <div class="results-section" id="resultsSection" style="display: none;">
                <h2>🎯 분석 결과</h2>
                <div id="analysisResults"></div>
            </div>
        </div>

        <script>
            let selectedFiles = [];
            let currentSessionId = null;

            // 파일 업로드 영역 이벤트
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');

            uploadArea.addEventListener('click', () => fileInput.click());
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                handleFiles(e.dataTransfer.files);
            });

            fileInput.addEventListener('change', (e) => {
                handleFiles(e.target.files);
            });

            function handleFiles(files) {
                selectedFiles = Array.from(files);
                displayFileList();
            }

            function displayFileList() {
                const fileList = document.getElementById('fileList');
                fileList.innerHTML = '<h3>선택된 파일:</h3>';
                
                selectedFiles.forEach((file, index) => {
                    const fileItem = document.createElement('div');
                    fileItem.className = 'file-item';
                    fileItem.innerHTML = `
                        <span>${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                        <button onclick="removeFile(${index})" style="background: #dc3545; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;">삭제</button>
                    `;
                    fileList.appendChild(fileItem);
                });
            }

            function removeFile(index) {
                selectedFiles.splice(index, 1);
                displayFileList();
            }

            async function startAnalysis() {
                if (selectedFiles.length === 0) {
                    alert('분석할 파일을 선택해주세요.');
                    return;
                }

                // 세션 ID 생성
                currentSessionId = 'session_' + Date.now();
                
                // 상태 업데이트
                updateStatus('processing', '분석 시작 중...', 0);
                
                // FormData 생성
                const formData = new FormData();
                selectedFiles.forEach(file => {
                    formData.append('files', file);
                });

                try {
                    // API 호출
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });

                    if (response.ok) {
                        const result = await response.json();
                        displayResults(result);
                        updateStatus('completed', '분석 완료!', 100);
                    } else {
                        throw new Error('분석 실패');
                    }
                } catch (error) {
                    updateStatus('error', '분석 중 오류가 발생했습니다: ' + error.message, 0);
                }
            }

            function updateStatus(type, message, progress) {
                const statusList = document.getElementById('statusList');
                const timestamp = new Date().toLocaleTimeString();
                
                statusList.innerHTML = `
                    <div class="status-item status-${type}">
                        <strong>${timestamp}</strong><br>
                        ${message}
                    </div>
                `;

                // 진행률 업데이트
                const progressFill = document.getElementById('overallProgress');
                const progressText = document.getElementById('progressText');
                progressFill.style.width = progress + '%';
                progressText.textContent = progress + '%';
            }

            function displayResults(result) {
                const resultsSection = document.getElementById('resultsSection');
                const analysisResults = document.getElementById('analysisResults');
                
                resultsSection.style.display = 'block';
                
                let html = `
                    <div class="analysis-result">
                        <div class="result-header">
                            <h3>📋 분석 요약</h3>
                            <span class="ai-badge">AI 분석</span>
                        </div>
                        <div class="result-content">
                            <p><strong>분석 ID:</strong> ${result.analysis_id}</p>
                            <p><strong>분석 시간:</strong> ${new Date(result.timestamp).toLocaleString()}</p>
                            <p><strong>분석된 파일 수:</strong> ${result.files_count}개</p>
                        </div>
                    </div>
                `;

                // 파일별 분석 결과
                if (result.files_analyzed) {
                    result.files_analyzed.forEach(file => {
                        html += `
                            <div class="analysis-result">
                                <div class="result-header">
                                    <h4>📄 ${file.filename}</h4>
                                    <span style="font-size: 12px; color: #666;">${file.analysis_type}</span>
                                </div>
                                <div class="result-content">
                                    <p>${file.content}</p>
                                </div>
                            </div>
                        `;
                    });
                }

                // AI 분석 결과
                if (result.ai_analysis && result.ai_analysis.ai_summary) {
                    html += `
                        <div class="analysis-result">
                            <div class="result-header">
                                <h3>🤖 AI 종합 분석</h3>
                                <span class="ai-badge">${result.ai_analysis.model_used}</span>
                            </div>
                            <div class="result-content">
                                <p>${result.ai_analysis.ai_summary}</p>
                            </div>
                        </div>
                    `;
                }

                analysisResults.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/analyze")
async def analyze_files_ui(files: List[UploadFile] = File(...)):
    """파일 분석 - Module1 API 연동"""
    try:
        # Module1 API로 전달
        files_data = []
        for file in files:
            content = await file.read()
            files_data.append(("files", (file.filename, content, file.content_type)))
            await file.seek(0)

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                "http://localhost:8001/analyze",
                files=files_data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail="Module1 API 호출 실패")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 실패: {str(e)}")

@app.get("/health")
async def health_check():
    """서비스 상태 확인"""
    return {
        "status": "healthy",
        "service": "Integrated Web UI",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("통합 웹 UI 시작...")
    print("접속 주소: http://localhost:8080")
    print("파일을 드래그하여 업로드하고 AI 분석을 받아보세요!")
    
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)